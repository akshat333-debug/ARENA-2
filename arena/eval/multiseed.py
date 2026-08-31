"""Multi-seed evaluation — turning one run into a result (M11 future-work #6).

The M8/M11 leaderboard is a **single seed**. Every headline it produced —
including the withdrawn "3x less exploitable" and its replacement, "arena_blue
0.780 vs causal_monitor 0.690" — rests on one draw of the RNG. Neither is a
finding until the spread across seeds is smaller than the gap being claimed.

This module runs the whole leaderboard over N seeds and reports mean +/- spread,
plus the thing that actually settles the question: the **paired per-seed
difference** between two defenders. Paired, because inside one seed every
defender sees the same scenarios, the same decision set and the same
best-response budget — so the difference has far less variance than the two
means do, and comparing the means alone throws that away.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from arena.config import ArenaConfig
from arena.eval.harness import LeaderboardRow, evaluate_defenders

#: seed -> {name: defender}. A factory, not a dict, because a defender that is
#: itself trained (arena_blue) must be retrained per seed or the "multi-seed"
#: result silently holds one of its arms fixed.
DefenderFactory = Callable[[int], dict[str, object]]

METRICS = ("auroc", "tpr_at_5pct_fpr", "exploitability", "operating_fpr", "fresh_red_start")


@dataclass(frozen=True)
class AggregateRow:
    name: str
    n_seeds: int
    mean: dict[str, float]
    std: dict[str, float]
    #: Raw per-seed values, kept so a caller can do its own statistics.
    values: dict[str, tuple[float, ...]]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n_seeds": self.n_seeds,
            **{f"{m}_mean": round(self.mean[m], 4) for m in self.mean},
            **{f"{m}_std": round(self.std[m], 4) for m in self.std},
        }


@dataclass(frozen=True)
class PairedDelta:
    """Per-seed difference ``a - b`` on one metric."""

    metric: str
    a: str
    b: str
    deltas: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.deltas))

    @property
    def std(self) -> float:
        return float(np.std(self.deltas, ddof=1)) if len(self.deltas) > 1 else 0.0

    @property
    def n_favouring_a(self) -> int:
        """Seeds where ``a`` scored strictly lower than ``b`` (better, for
        exploitability). Direction is metric-dependent — the caller reads it."""
        return int(sum(1 for d in self.deltas if d < 0))

    def separated(self) -> bool:
        """True when every seed agrees on the sign — the weakest honest claim
        that the difference is not noise. Deliberately not a p-value: with 3-5
        seeds a t-test would be theatre, and a unanimous sign test is something
        a reader can check by eye."""
        return len(self.deltas) > 1 and (
            all(d < 0 for d in self.deltas) or all(d > 0 for d in self.deltas)
        )

    def summary(self) -> str:
        arrow = "<" if self.mean < 0 else ">"
        verdict = "consistent across all seeds" if self.separated() else "SIGN FLIPS ACROSS SEEDS"
        return (
            f"{self.metric}: {self.a} {arrow} {self.b} by {abs(self.mean):.3f} "
            f"(+/-{self.std:.3f}, n={len(self.deltas)}) — {verdict}"
        )


def multiseed_leaderboard(
    defenders: DefenderFactory | dict[str, object],
    config: ArenaConfig | None = None,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    n_decision_adv: int | None = None,
    n_decision_benign: int | None = None,
    br_steps: int | None = None,
    n_eval: int | None = None,
    progress: Callable[[int, list[LeaderboardRow]], None] | None = None,
) -> tuple[list[AggregateRow], dict[int, list[LeaderboardRow]]]:
    """Run the leaderboard once per seed; return aggregates and the raw rows.

    ``defenders`` may be a plain dict (reused every seed — fine for the
    stateless baselines) or a factory called with the seed.
    """
    if not seeds:
        raise ValueError("need at least one seed")
    cfg = config or ArenaConfig()
    per_seed: dict[int, list[LeaderboardRow]] = {}

    for s in seeds:
        d = defenders(s) if callable(defenders) else defenders
        rows = evaluate_defenders(
            d, cfg,
            n_decision_adv=n_decision_adv, n_decision_benign=n_decision_benign,
            br_steps=br_steps, n_eval=n_eval, seed=s,
        )
        per_seed[s] = rows
        if progress is not None:
            progress(s, rows)

    names = [r.name for r in per_seed[seeds[0]]]
    aggregates = []
    for name in names:
        vals = {m: [] for m in METRICS}
        for s in seeds:
            row = next(r for r in per_seed[s] if r.name == name)
            for m in METRICS:
                vals[m].append(float(getattr(row, m)))
        aggregates.append(AggregateRow(
            name=name,
            n_seeds=len(seeds),
            mean={m: float(np.mean(v)) for m, v in vals.items()},
            std={m: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for m, v in vals.items()},
            values={m: tuple(v) for m, v in vals.items()},
        ))
    aggregates.sort(key=lambda r: (r.mean["exploitability"], -r.mean["auroc"]))
    return aggregates, per_seed


def paired_delta(
    per_seed: dict[int, list[LeaderboardRow]],
    a: str,
    b: str,
    metric: str = "exploitability",
) -> PairedDelta:
    """Per-seed ``a - b`` on ``metric``. Both must appear in every seed."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    deltas = []
    for s in sorted(per_seed):
        rows = per_seed[s]
        try:
            ra = next(r for r in rows if r.name == a)
            rb = next(r for r in rows if r.name == b)
        except StopIteration:
            raise KeyError(f"seed {s} is missing {a!r} or {b!r}") from None
        deltas.append(float(getattr(ra, metric)) - float(getattr(rb, metric)))
    return PairedDelta(metric=metric, a=a, b=b, deltas=tuple(deltas))


def format_aggregate(rows: list[AggregateRow]) -> str:
    head = (f"{'defender':<20}{'AUROC':>16}{'TPR@5%FPR':>16}"
            f"{'exploitability':>18}{'op.FPR':>14}")
    lines = [head, "-" * len(head)]
    for r in rows:
        def cell(m, w):
            return f"{r.mean[m]:.3f}+/-{r.std[m]:.3f}".rjust(w)
        lines.append(
            f"{r.name:<20}{cell('auroc', 16)}{cell('tpr_at_5pct_fpr', 16)}"
            f"{cell('exploitability', 18)}{cell('operating_fpr', 14)}"
        )
    lines.append(f"\nn_seeds = {rows[0].n_seeds}" if rows else "")
    return "\n".join(lines)
