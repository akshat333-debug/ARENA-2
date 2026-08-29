"""Opponent-checkpoint league (architecture.md S7, AlphaStar-style).

M6 showed the failure this fixes: when Blue trains against only the *current*
frozen Red — which is weak, because Blue already beat it — Blue unlearns
quarantining, and the pair drifts to ``(passive Blue, mediocre Red)``. A league
keeps a pool of past checkpoints per side and trains each side against a *sample*
of the opposing side's history, so Blue never stops seeing strong attackers and
Red never gets to coast against a passive Blue.

Two pieces:

* :class:`League` — the pools and the sampling policy. Pure bookkeeping; holds
  CPU ``state_dict``s, round-trips through its own ``state_dict``.
* :class:`FrozenPolicySampler` — a plain ``obs -> int`` opponent that draws a
  frozen policy from a pool and resamples it when the episode changes. Frozen:
  ``TorchPolicyAdapter.act`` is under ``no_grad`` and the sampled parameters are
  never handed to an optimiser.

Prioritised sampling by win-rate (PFSP) is a future refinement; M7 ships uniform
sampling with a tunable bias toward the latest checkpoint, which is enough to
stop the drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from arena.config import PolicyConfig
from arena.policies import ActorCritic, TorchPolicyAdapter, make_policy

SIDES = ("red", "blue")


def _cpu_state_dict(sd: dict) -> dict:
    return {k: v.detach().to("cpu").clone() for k, v in sd.items()}


@dataclass
class Checkpoint:
    side: str
    generation: int
    state_dict: dict = field(repr=False)
    #: Optional: fraction of episodes the *other* side won against this one when
    #: it was added. Unused by uniform sampling; kept for a future PFSP mode.
    opponent_win_rate: float | None = None

    def key(self) -> tuple[str, int]:
        return (self.side, self.generation)


class League:
    def __init__(
        self,
        *,
        pool_max: int | None = None,
        p_latest: float = 0.3,
        seed: int = 0,
    ) -> None:
        if pool_max is not None and pool_max < 1:
            raise ValueError(f"pool_max must be >= 1 or None, got {pool_max}")
        if not 0.0 <= p_latest <= 1.0:
            raise ValueError(f"p_latest must be in [0, 1], got {p_latest}")
        self.pool_max = pool_max
        self.p_latest = p_latest
        self._pools: dict[str, list[Checkpoint]] = {"red": [], "blue": []}
        self._rng = np.random.default_rng(seed)

    # --- population -------------------------------------------------

    def add(self, side: str, generation: int, state_dict: dict,
            *, opponent_win_rate: float | None = None) -> Checkpoint:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        ckpt = Checkpoint(side, generation, _cpu_state_dict(state_dict), opponent_win_rate)
        pool = self._pools[side]
        pool.append(ckpt)
        # Evict oldest, but never the most recent — the latest checkpoint is the
        # one the drift-guard most needs to keep.
        if self.pool_max is not None and len(pool) > self.pool_max:
            del pool[0]
        return ckpt

    def pool_size(self, side: str) -> int:
        return len(self._pools[side])

    def __len__(self) -> int:
        return sum(len(p) for p in self._pools.values())

    def is_empty(self, side: str) -> bool:
        return not self._pools[side]

    def latest(self, side: str) -> Checkpoint | None:
        pool = self._pools[side]
        return pool[-1] if pool else None

    def checkpoints(self, side: str) -> tuple[Checkpoint, ...]:
        return tuple(self._pools[side])

    # --- sampling --------------------------------------------------

    def sample(self, side: str) -> Checkpoint:
        """Draw an opponent checkpoint. With probability ``p_latest`` return the
        most recent; otherwise draw uniformly from the whole pool (which also
        contains the latest, so it is never excluded)."""
        pool = self._pools[side]
        if not pool:
            raise IndexError(f"{side} pool is empty")
        if len(pool) == 1 or self._rng.random() < self.p_latest:
            return pool[-1]
        return pool[int(self._rng.integers(len(pool)))]

    def sample_probs(self, side: str) -> np.ndarray:
        """The exact per-checkpoint probability :meth:`sample` induces — for
        tests and for logging."""
        n = len(self._pools[side])
        if n == 0:
            return np.zeros(0)
        if n == 1:
            return np.ones(1)
        p = np.full(n, (1.0 - self.p_latest) / n)
        p[-1] += self.p_latest
        return p

    # --- persistence --------------------------------------------

    def state_dict(self) -> dict:
        return {
            "pool_max": self.pool_max,
            "p_latest": self.p_latest,
            "pools": {
                side: [
                    {"side": c.side, "generation": c.generation,
                     "state_dict": c.state_dict, "opponent_win_rate": c.opponent_win_rate}
                    for c in pool
                ]
                for side, pool in self._pools.items()
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        self.pool_max = sd["pool_max"]
        self.p_latest = sd["p_latest"]
        self._pools = {
            side: [
                Checkpoint(c["side"], c["generation"], _cpu_state_dict(c["state_dict"]),
                           c.get("opponent_win_rate"))
                for c in pool
            ]
            for side, pool in sd["pools"].items()
        }


PolicyFactory = Callable[[], ActorCritic]


class FrozenPolicySampler:
    """A frozen ``obs -> int`` opponent drawn from a :class:`League` pool.

    ``side`` is the side of the *opponent* (training Blue -> ``side="red"``). The
    sampler resamples a checkpoint whenever the env's episode index changes, so a
    single training phase faces a distribution of past opponents rather than one
    fixed policy. Materialised policies are cached by ``(side, generation)`` so a
    long run does not rebuild a network every episode.

    ``fallback`` is used only while the pool is still empty (generation 0 before
    the first checkpoint lands) — typically the current live opponent policy.
    """

    def __init__(
        self,
        env,
        league: League,
        side: str,
        *,
        make_net: PolicyFactory,
        fallback: ActorCritic | None = None,
        deterministic: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self._env = getattr(env, "aec", env)
        self._league = league
        self._side = side
        self._make_net = make_net
        self._deterministic = deterministic
        self._device = device
        self._cache: dict[tuple[str, int], ActorCritic] = {}
        self._fallback_adapter = (
            TorchPolicyAdapter(fallback, deterministic=deterministic, device=device)
            if fallback is not None else None
        )
        self._episode: int | None = None
        self._adapter: TorchPolicyAdapter | None = None

    def _materialise(self, ckpt: Checkpoint) -> ActorCritic:
        cached = self._cache.get(ckpt.key())
        if cached is not None:
            return cached
        net = self._make_net().to(self._device)
        net.load_state_dict(ckpt.state_dict)
        net.eval()
        self._cache[ckpt.key()] = net
        return net

    def _refresh(self) -> None:
        if self._league.is_empty(self._side):
            if self._fallback_adapter is None:
                raise RuntimeError(
                    f"{self._side} pool is empty and no fallback policy was given"
                )
            self._adapter = self._fallback_adapter
            return
        ckpt = self._league.sample(self._side)
        self._adapter = TorchPolicyAdapter(
            self._materialise(ckpt), deterministic=self._deterministic, device=self._device
        )

    def __call__(self, obs: dict) -> int:
        ep = self._env.episode_index
        if ep != self._episode or self._adapter is None:
            self._episode = ep
            self._refresh()
        return self._adapter(obs)


def net_factory(side: str, *, max_steps: int, n_tools_max: int,
                cfg: PolicyConfig | None = None) -> PolicyFactory:
    """A zero-arg builder for a fresh policy of the given side and shape — what
    :class:`FrozenPolicySampler` needs to rehydrate a checkpoint."""
    def _build() -> ActorCritic:
        return make_policy(side, max_steps=max_steps, n_tools_max=n_tools_max, cfg=cfg)
    return _build


__all__ = ["League", "Checkpoint", "FrozenPolicySampler", "net_factory", "SIDES"]
