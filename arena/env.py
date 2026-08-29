"""The simulated MCP session as a PettingZoo AEC game, plus a Gymnasium wrapper.

One episode is one MCP session. Each "MCP step" is two AEC turns:

1. ``red`` picks the next tool call (an index into the episode's registry).
2. ``blue`` sees the whole call sequence including that pending proposal and
   returns a verdict: ``allow`` / ``flag`` / ``quarantine``.
   * ``allow``     — the call executes.
   * ``flag``      — the call executes but is recorded (partial credit if it is
                     genuinely on the attack chain, a cost if it is not).
   * ``quarantine``— the session ends immediately; the pending call never runs.

The episode ends on quarantine, on the attack objective completing, or at the
step cap.

**Anti-leakage invariant (architecture.md S4).** Blue's observation is built in
``_encode_call`` / ``_blue_obs`` from *observable* call metadata only — tool
side-effect class, declared sensitivity, category, position, and Blue's own past
verdicts. It never reads :class:`arena.taint.TaintTracker`. Attack success is
adjudicated from the taint graph, which Blue does not see. ``tests/test_env.py``
locks this in: perturbing the internal taint state leaves Blue's observation
byte-identical, and two episodes with the same tool sequence but different
objectives produce identical Blue observations.

ponytail: the feature encoders live here for now. M4 lifts them into
``arena/features.py`` and adds the baseline detectors that consume them; the
observation *layout* is meant to be stable across that move.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from gymnasium import Env as GymEnv
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

from arena.config import ArenaConfig, RewardConfig
from arena.features import (
    ALLOW,
    CALL_FEATS,
    DOMAINS,
    FLAG,
    QUARANTINE,
    VERDICTS,
    blue_observation_space,
    domain_onehot,
    encode_call,
    history_matrix,
    red_observation_space,
)
from arena.rewards import EpisodeOutcome, compute_rewards, heuristic_plausibility
from arena.scenarios import Scenario, ScenarioGenerator
from arena.taint import TaintTracker
from arena.tools import ToolSpec

RED = "red_0"
BLUE = "blue_0"
AGENTS = (RED, BLUE)

# Verdict constants (ALLOW/FLAG/QUARANTINE) are re-exported from features so
# callers can `from arena.env import QUARANTINE` without knowing the layout lives
# elsewhere.
__all__ = [
    "ARENAEnv", "SingleAgentARENA", "raw_env",
    "RED", "BLUE", "AGENTS", "ALLOW", "FLAG", "QUARANTINE", "VERDICTS",
    "ResolvedCall",
]

_KIND_IX = {"exfiltration": 0, "unauthorized_action": 1, "injected_control": 2}


@dataclass
class ResolvedCall:
    step_index: int
    tool: ToolSpec
    verdict: int  # ALLOW or FLAG (QUARANTINE never produces a resolved call)


class ARENAEnv(AECEnv):
    """Two-player AEC env: ``red`` proposes tool calls, ``blue`` adjudicates."""

    metadata = {"render_modes": ["human"], "name": "arena_v0", "is_parallelizable": False}

    def __init__(
        self,
        config: ArenaConfig | None = None,
        *,
        scenario: Scenario | None = None,
        render_mode: str | None = None,
    ) -> None:
        """``scenario`` pins every episode to one fixed scenario (tests, targeted
        eval). Otherwise scenarios are drawn from a seeded generator built from
        ``config`` (default :class:`ArenaConfig`)."""
        super().__init__()
        self._config = config or ArenaConfig()
        self._reward_cfg: RewardConfig = self._config.reward
        self._fixed_scenario = scenario
        self._gen: ScenarioGenerator | None = (
            None if scenario is not None else ScenarioGenerator.from_config(self._config.scenario)
        )
        self.render_mode = render_mode

        self.possible_agents = list(AGENTS)
        self._max_steps = scenario.max_steps if scenario is not None else self._config.scenario.max_steps
        self._n_tools_max = (
            len(scenario.registry) if scenario is not None else self._config.scenario.n_tools_max
        )

        self._obs_spaces = {
            RED: red_observation_space(self._max_steps, self._n_tools_max),
            BLUE: blue_observation_space(self._max_steps),
        }
        self._act_spaces = {
            RED: spaces.Discrete(self._n_tools_max),
            BLUE: spaces.Discrete(len(VERDICTS)),
        }

        self._np_random = np.random.default_rng()
        self._scenario: Scenario | None = None
        self._tracker = TaintTracker()
        self._history: list[ResolvedCall] = []
        self._step_index = 0
        self._pending_tool_idx: int | None = None
        self._flagged_steps: list[int] = []
        self._quarantined = False
        self._quarantine_step: int | None = None
        self._completion_step: int | None = None
        self._ended = False

    # --- PettingZoo API ---------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            if self._gen is not None:
                self._gen = ScenarioGenerator.from_config(self._config.scenario, seed=seed)

        if self._fixed_scenario is not None:
            self._scenario = self._fixed_scenario
        else:
            assert self._gen is not None
            self._scenario = self._gen.sample()

        self._tracker.reset()
        self._history = []
        self._step_index = 0
        self._pending_tool_idx = None
        self._flagged_steps = []
        self._quarantined = False
        self._quarantine_step = None
        self._completion_step = None
        self._ended = False

        self.agents = list(AGENTS)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._selector = AgentSelector(self.agents)
        self.agent_selection = self._selector.reset()

    def observe(self, agent: str) -> dict:
        return self._red_obs() if agent == RED else self._blue_obs()

    def step(self, action) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0

        if agent == RED:
            # Action space is a fixed Discrete(n_tools_max); a given episode's
            # registry may be smaller. `registry_mask` in Red's observation marks
            # the valid indices (0..len-1); anything past that wraps, so every
            # action still maps to a real tool.
            # ponytail: modulo, not a no-op turn. Revisit if a trained Red is
            # seen exploiting the wrap instead of learning within the mask.
            self._pending_tool_idx = int(action) % len(self._scenario.registry)
        else:
            self._resolve(int(action))

        self.agent_selection = self._selector.next()
        self._accumulate_rewards()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        sc = self._scenario
        print(f"[{sc.scenario_id}] {sc.episode_type.value} step {self._step_index}/{sc.max_steps}")
        for rc in self._history:
            print(f"  {rc.step_index:2d} {rc.tool.name:22s} {['allow', 'flag'][rc.verdict]}")
        if self._quarantined:
            print(f"  -- quarantined at step {self._quarantine_step}")

    def close(self) -> None:  # nothing to release
        pass

    # --- resolution -----------------------------------------------------

    def _resolve(self, verdict: int) -> None:
        if verdict not in VERDICTS:
            raise ValueError(f"blue verdict {verdict} not in {VERDICTS}")
        sc = self._scenario
        tool = sc.registry[self._pending_tool_idx]

        if verdict == QUARANTINE:
            self._quarantined = True
            self._quarantine_step = self._step_index
            self._end_episode()
            return

        # allow or flag: the call executes
        self._tracker.step(tool, self._step_index, input_refs=None)
        self._history.append(ResolvedCall(self._step_index, tool, verdict))
        if verdict == FLAG:
            self._flagged_steps.append(self._step_index)

        if sc.objective is not None and self._completion_step is None:
            if self._tracker.objective_completed(sc.objective):
                self._completion_step = self._step_index

        self._step_index += 1
        self._pending_tool_idx = None

        if self._completion_step is not None or self._step_index >= sc.max_steps:
            self._end_episode()

    def _end_episode(self) -> None:
        if self._ended:
            return
        self._ended = True
        sc = self._scenario
        tools = [rc.tool for rc in self._history]
        malicious = self._tracker.attack_path(sc.objective) if sc.objective is not None else ()

        outcome = EpisodeOutcome(
            is_adversarial=sc.is_adversarial,
            objective_completed=self._completion_step is not None,
            n_steps=len(self._history),
            plausibility=heuristic_plausibility(tools),
            quarantined=self._quarantined,
            quarantine_step=self._quarantine_step,
            completion_step=self._completion_step,
            flagged_steps=tuple(self._flagged_steps),
            malicious_steps=malicious,
        )
        rb = compute_rewards(outcome, self._reward_cfg)
        self.rewards[RED] = float(rb.r_red)
        self.rewards[BLUE] = float(rb.r_blue)
        self._last_outcome = outcome
        self._last_breakdown = rb

        step_cap = not self._quarantined and self._completion_step is None
        for a in self.agents:
            self.terminations[a] = True
            self.truncations[a] = step_cap
            self.infos[a] = {
                "scenario_id": sc.scenario_id,
                "episode_type": sc.episode_type.value,
                "family": sc.family.value if sc.family else None,
                "objective_completed": outcome.objective_completed,
                "quarantined": self._quarantined,
                "n_steps": outcome.n_steps,
            }

    # --- observation builders (OBSERVABLE METADATA ONLY) ---------------

    def _history_rows(self) -> list[tuple[ToolSpec, int, int]]:
        return [(rc.tool, rc.step_index, rc.verdict) for rc in self._history]

    def _pending(self) -> tuple[ToolSpec, int] | None:
        if self._pending_tool_idx is None:
            return None
        return self._scenario.registry[self._pending_tool_idx], self._step_index

    def _blue_obs(self) -> dict:
        mat, length = history_matrix(
            self._history_rows(), self._pending(), self._max_steps, for_red=False
        )
        return {
            "domain": domain_onehot(self._scenario.domain.value),
            "calls": mat,
            "length": np.array([length], dtype=np.int32),
        }

    def _red_obs(self) -> dict:
        sc = self._scenario
        reg = sc.registry

        task = np.zeros(len(DOMAINS) + 1, dtype=np.float32)
        task[: len(DOMAINS)] = domain_onehot(sc.domain.value)
        task[-1] = self._step_index / max(1, self._max_steps)

        registry = np.zeros((self._n_tools_max, CALL_FEATS), dtype=np.float32)
        mask = np.zeros(self._n_tools_max, dtype=np.int8)
        for i, tool in enumerate(reg):
            registry[i] = encode_call(
                tool, 0, self._max_steps, resolved=False, verdict=None, for_red=False
            )
            mask[i] = 1

        objective = np.zeros(4 + 2 * self._n_tools_max, dtype=np.float32)
        if sc.objective is not None:
            obj = sc.objective
            objective[_KIND_IX[obj.kind.value]] = 1.0
            objective[3] = obj.min_chain_len / max(1, self._max_steps)
            for name in obj.source_tools:
                if name in reg.names:
                    objective[4 + reg.index(name)] = 1.0
            for name in obj.sink_tools:
                if name in reg.names:
                    objective[4 + self._n_tools_max + reg.index(name)] = 1.0

        mat, length = history_matrix(
            self._history_rows(), self._pending(), self._max_steps, for_red=True
        )
        return {
            "task": task,
            "registry": registry,
            "registry_mask": mask,
            "objective": objective,
            "history": mat,
            "length": np.array([length], dtype=np.int32),
        }

    # --- convenience for tests / eval --------------------------------

    @property
    def scenario(self) -> Scenario | None:
        return self._scenario

    @property
    def last_outcome(self) -> EpisodeOutcome | None:
        return getattr(self, "_last_outcome", None)

    @property
    def last_breakdown(self):
        """The :class:`~arena.rewards.RewardBreakdown` from the finished episode.

        ``self.rewards`` is pruned once PettingZoo's dead-step handling drains the
        agents, so read final rewards here instead.
        """
        return getattr(self, "_last_breakdown", None)

    def legal_red_actions(self) -> np.ndarray:
        """Indices Red may pick this turn (currently the whole registry)."""
        return np.arange(len(self._scenario.registry))


def raw_env(config: ArenaConfig | None = None, **kw) -> ARENAEnv:
    return ARENAEnv(config, **kw)


# --- Gymnasium single-agent view -------------------------------------

Policy = Callable[[dict], int]


class SingleAgentARENA(GymEnv):
    """Expose one side of :class:`ARENAEnv` as a standard Gymnasium env, with the
    other side driven by a fixed ``opponent`` policy.

    This is what PPO best-response training (M5) and the exploitability sweep
    (M8) run against: "train Red vs frozen Blue" is ``learner=RED`` here.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        learner: str,
        opponent: Policy,
        *,
        config: ArenaConfig | None = None,
        scenario: Scenario | None = None,
    ) -> None:
        if learner not in AGENTS:
            raise ValueError(f"learner must be one of {AGENTS}, got {learner!r}")
        self._aec = ARENAEnv(config, scenario=scenario)
        self._learner = learner
        self._opponent_agent = BLUE if learner == RED else RED
        self._opponent = opponent
        self.observation_space = self._aec.observation_space(learner)
        self.action_space = self._aec.action_space(learner)

    def _play_opponent_until_learner(self) -> None:
        aec = self._aec
        while not (aec.terminations[self._learner] or aec.truncations[self._learner]):
            if aec.agent_selection == self._learner:
                return
            opp = aec.agent_selection
            action = self._opponent(aec.observe(opp))
            aec.step(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._aec.reset(seed=seed)
        self._play_opponent_until_learner()
        return self._aec.observe(self._learner), dict(self._aec.infos[self._learner])

    def step(self, action):
        aec = self._aec
        if aec.terminations[self._learner] or aec.truncations[self._learner]:
            raise RuntimeError("step() called on a finished episode; call reset()")

        aec.step(action)
        self._play_opponent_until_learner()
        # _cumulative_rewards[learner] accumulates every transition since the
        # learner last acted — including the terminal step, wherever it landed —
        # and is zeroed when the learner acts again. That is exactly this gym
        # transition's return, counted once.
        reward = float(aec._cumulative_rewards[self._learner])

        term = bool(aec.terminations[self._learner])
        trunc = bool(aec.truncations[self._learner])
        obs = self._aec.observe(self._learner)
        return obs, reward, term, trunc, dict(aec.infos[self._learner])

    @property
    def aec(self) -> ARENAEnv:
        return self._aec
