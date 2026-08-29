"""Scripted (non-learned) policies for both sides, plus the opponent wrapper that
mixes an attacker with benign traffic over a scenario distribution.

Used to generate labelled traffic for the M4 baselines, as Blue's opponent during
self-play (M6), as opponents in the M8 exploitability sweep, and as sanity
references throughout.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arena.features import ALLOW, QUARANTINE
from arena.scenarios import ObjectiveKind, Scenario

RedPolicy = Callable[[dict], int]


class ScriptedAttacker:
    """Plays a minimal winning chain for ``scenario``: a source, then the
    propagator repeated to reach ``min_chain_len``, then a sink. Emits index 0
    once the chain is spent (a harmless filler read in practice)."""

    def __init__(self, scenario: Scenario) -> None:
        if scenario.objective is None:
            raise ValueError("ScriptedAttacker needs an adversarial scenario")
        reg = scenario.registry
        obj = scenario.objective
        src = obj.source_tools[0]
        if obj.kind is ObjectiveKind.INJECTED_CONTROL:
            sink = reg.sinks[0].name
        else:
            sink = obj.sink_tools[0]
        prop = reg.propagators[0].name
        n_prop = max(1, obj.min_chain_len - 2)
        names = [src] + [prop] * n_prop + [sink]
        self._plan = [reg.index(n) for n in names]
        self._i = 0

    def __call__(self, red_obs: dict) -> int:
        a = self._plan[self._i] if self._i < len(self._plan) else 0
        self._i += 1
        return a

    def reset(self) -> None:
        self._i = 0


class BenignRoller:
    """Plausible benign traffic. It *does* read sensitive files — legitimate dev
    tooling routinely reads ``.env``, a clinician reads a patient record — and it
    *does* hit external sinks. What it never does is let sensitive data reach a
    sink: once it has touched a source this episode, it stops calling sinks. So no
    episode it drives is accidentally an attack, yet a lone sensitive read is not
    a give-away tell (which would unfairly flatter the single-shot detector)."""

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self._reg = scenario.registry
        self._nonsink = [i for i, t in enumerate(self._reg) if not t.is_sink]
        self._nonsink = self._nonsink or [0]
        self._all = list(range(len(self._reg)))
        self._rng = np.random.default_rng(seed)
        self._touched_source = False

    def reset(self) -> None:
        self._touched_source = False

    def __call__(self, red_obs: dict) -> int:
        pool = self._nonsink if self._touched_source else self._all
        i = int(self._rng.choice(pool))
        if self._reg[i].is_source:
            self._touched_source = True
        return i


class AdaptiveRed:
    """A Red opponent for a *distribution* of scenarios.

    An env driven by a config draws a fresh scenario every reset, but
    :class:`ScriptedAttacker` / :class:`BenignRoller` are per-scenario. This
    wrapper watches the live env and switches inner policy whenever the episode
    changes: an attacker on adversarial episodes, benign traffic on benign ones.
    That mixture is what Blue has to be trained against — a Blue that only ever
    sees attacks learns to quarantine everything (M5).

    ``attacker`` overrides the adversarial branch with a fixed callable — this is
    how self-play (M6) feeds Blue a *frozen learned* Red on attack episodes while
    still keeping benign traffic realistic. When ``None`` (default) a fresh
    :class:`ScriptedAttacker` is built for each adversarial scenario.
    """

    def __init__(self, env, *, seed: int = 0, attacker: RedPolicy | None = None) -> None:
        self._env = getattr(env, "aec", env)
        self._seed = seed
        self._attacker = attacker
        self._episode: int | None = None
        self._inner = None

    def __call__(self, red_obs: dict) -> int:
        sc = self._env.scenario
        if sc is None:
            raise RuntimeError("env has no active scenario; call reset() first")
        # Key off the env's episode counter, NOT scenario_id: reset(seed=) rebuilds
        # the generator and restarts its counter, so ids repeat across resets. Keying
        # off the id silently reuses a spent attacker, which emits filler actions —
        # the episode then looks adversarial while no attack is actually played.
        ep = self._env.episode_index
        if ep != self._episode:
            self._episode = ep
            if sc.is_adversarial:
                self._inner = self._attacker if self._attacker is not None else ScriptedAttacker(sc)
            else:
                self._inner = BenignRoller(sc, seed=self._seed + ep)
        return self._inner(red_obs)


#: Back-compat alias — the class predates the ``attacker`` parameter.
AdaptiveScriptedRed = AdaptiveRed


def passive_blue(blue_obs: dict) -> int:
    """Allows everything — the fully-exploitable reference defender."""
    return ALLOW


def paranoid_blue(blue_obs: dict) -> int:
    """Quarantines everything — zero misses, maximal false-positive cost."""
    return QUARANTINE
