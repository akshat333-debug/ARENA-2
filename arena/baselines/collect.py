"""Roll episodes and record one row per Blue decision, for baseline training.

The ground-truth ``on_attack_path`` label comes from the taint graph and is used
**only** to train / evaluate baselines offline. At inference a baseline sees
``blue_obs`` and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from arena.config import ArenaConfig
from arena.env import BLUE, RED, ARENAEnv
from arena.features import ALLOW
from arena.scenarios import ScenarioGenerator
from arena.scripted import BenignRoller, ScriptedAttacker
from arena.tools import ToolSpec

RedPolicy = Callable[[dict], int]
BluePolicy = Callable[[dict], int]


@dataclass(frozen=True)
class Decision:
    """One point where Blue had to act."""

    blue_obs: dict
    pending_tool: ToolSpec
    tools_incl_pending: tuple[ToolSpec, ...]
    step_index: int
    on_attack_path: bool  # ground truth — training/eval only
    adversarial: bool
    episode_completed: bool  # did the episode ultimately complete its objective


def _rollout(env: ARENAEnv, red: RedPolicy, blue: BluePolicy, seed: int) -> list[dict]:
    """Run one episode; return the raw per-decision captures (label filled later)."""
    env.reset(seed=seed)
    captures: list[dict] = []
    for agent in env.agent_iter():
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if agent == RED:
            env.step(red(env.observe(RED)))
            continue
        # blue's turn: capture before acting
        obs = env.observe(BLUE)
        pend = env._pending()  # (tool, step_index)
        executed = tuple(rc.tool for rc in env._history)
        captures.append(
            {
                "blue_obs": obs,
                "pending_tool": pend[0],
                "tools_incl_pending": executed + (pend[0],),
                "step_index": pend[1],
            }
        )
        env.step(blue(obs))
    return captures


def collect_decisions(
    n_adversarial: int,
    n_benign: int,
    *,
    config: ArenaConfig | None = None,
    seed: int = 0,
    blue: BluePolicy | None = None,
) -> list[Decision]:
    """Roll ``n_adversarial`` scripted-attack episodes and ``n_benign`` benign
    episodes, all against ``blue`` (default: allow-everything, so the full chain
    plays out and every step is observed)."""
    cfg = config or ArenaConfig()
    blue = blue or (lambda o: ALLOW)
    gen = ScenarioGenerator.from_config(cfg.scenario, seed=seed)

    rows: list[Decision] = []
    made = 0
    want_adv, want_ben = n_adversarial, n_benign
    while want_adv > 0 or want_ben > 0:
        sc = gen.sample()
        if sc.is_adversarial and want_adv > 0:
            want_adv -= 1
        elif (not sc.is_adversarial) and want_ben > 0:
            want_ben -= 1
        else:
            continue

        env = ARENAEnv(cfg, scenario=sc)
        red = ScriptedAttacker(sc) if sc.is_adversarial else BenignRoller(sc, seed=seed + made)
        caps = _rollout(env, red, blue, seed=seed + made)
        made += 1

        malicious = set(env._tracker.attack_path(sc.objective)) if sc.objective else set()
        completed = env.last_outcome.objective_completed if env.last_outcome else False
        for c in caps:
            rows.append(
                Decision(
                    blue_obs=c["blue_obs"],
                    pending_tool=c["pending_tool"],
                    tools_incl_pending=c["tools_incl_pending"],
                    step_index=c["step_index"],
                    on_attack_path=c["step_index"] in malicious,
                    adversarial=sc.is_adversarial,
                    episode_completed=completed,
                )
            )
    return rows


def split(rows: list[Decision], frac: float = 0.75, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    cut = int(len(rows) * frac)
    train = [rows[i] for i in idx[:cut]]
    val = [rows[i] for i in idx[cut:]]
    return train, val
