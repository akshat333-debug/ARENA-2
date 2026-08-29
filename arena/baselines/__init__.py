"""Baseline defenders — the yardsticks ARENA's trained Blue must beat.

Three, matching the proposal (project.md S8):

* :class:`~arena.baselines.allowlist.StaticAllowList` — a per-call allow-list with
  no memory. Today's default. Misses every split chain.
* :class:`~arena.baselines.single_shot.SingleShotDetector` — a trained per-call
  classifier in the spirit of TAMAS's single-shot detectors. Judges each call in
  isolation, so it still cannot separate the split chain from benign traffic.
* :class:`~arena.baselines.causal.CausalMonitor` — a trained trajectory-level
  classifier in the spirit of CASPIAN's cross-channel causal monitor. Has the
  sequence context; this is the strong baseline.

Every baseline is a Blue policy: ``baseline(blue_obs) -> verdict`` plugs straight
into ``SingleAgentARENA(BLUE, ...)`` and the M8 eval harness. Trainable ones take
``fit(decisions)`` where ``decisions`` come from
:func:`arena.baselines.collect.collect_decisions`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from arena.features import ALLOW


class BlueBaseline(ABC):
    """A non-adaptive defender. ``fit`` is a no-op for rule-based baselines."""

    name: str = "baseline"

    def fit(self, decisions) -> "BlueBaseline":  # noqa: D401 - trivial default
        return self

    @abstractmethod
    def score(self, blue_obs: dict) -> float:
        """P(the pending call is part of an attack), in [0, 1]."""

    def act(self, blue_obs: dict) -> int:
        return self.verdict_from_score(self.score(blue_obs))

    def verdict_from_score(self, s: float) -> int:
        return ALLOW if s < self.threshold else self.quarantine_verdict

    #: Decision threshold on :meth:`score`. Tuned by ``fit`` for trained baselines.
    threshold: float = 0.5
    #: What to return when the score clears the threshold. QUARANTINE by default;
    #: FLAG for a softer baseline.
    quarantine_verdict: int = 2  # QUARANTINE

    def __call__(self, blue_obs: dict) -> int:
        return self.act(blue_obs)


from arena.baselines.allowlist import StaticAllowList  # noqa: E402
from arena.baselines.causal import CausalMonitor  # noqa: E402
from arena.baselines.single_shot import SingleShotDetector  # noqa: E402

__all__ = ["BlueBaseline", "StaticAllowList", "SingleShotDetector", "CausalMonitor"]
