"""Fast tests for arena/eval/multiseed.py (M11 future-work #6).

Aggregation and the paired-delta statistics are pure functions over
LeaderboardRows, so they are tested on hand-built rows — no training.
"""

from __future__ import annotations

import pytest

from arena.eval.harness import LeaderboardRow
from arena.eval.multiseed import (
    METRICS,
    AggregateRow,
    format_aggregate,
    multiseed_leaderboard,
    paired_delta,
)


def row(name, auroc=0.9, tpr=0.5, expl=0.4, fpr=0.05, fresh=0.1, n=100):
    return LeaderboardRow(
        name=name, auroc=auroc, tpr_at_5pct_fpr=tpr, exploitability=expl,
        fresh_red_start=fresh, operating_fpr=fpr, n_decisions=n,
    )


# --- paired_delta ---------------------------------------------------------


def test_paired_delta_computes_per_seed_differences():
    per_seed = {
        0: [row("a", expl=0.30), row("b", expl=0.50)],
        1: [row("a", expl=0.40), row("b", expl=0.55)],
        2: [row("a", expl=0.35), row("b", expl=0.45)],
    }
    d = paired_delta(per_seed, "a", "b")
    assert d.deltas == pytest.approx((-0.20, -0.15, -0.10))
    assert d.mean == pytest.approx(-0.15)
    assert d.n_favouring_a == 3
    assert d.separated() is True
    assert "consistent across all seeds" in d.summary()


def test_paired_delta_flags_a_sign_flip_as_not_separated():
    """The whole point: a mean difference that changes sign between seeds is
    noise, and must not read as a result."""
    per_seed = {
        0: [row("a", expl=0.30), row("b", expl=0.50)],
        1: [row("a", expl=0.60), row("b", expl=0.45)],
    }
    d = paired_delta(per_seed, "a", "b")
    assert d.separated() is False
    assert "SIGN FLIPS" in d.summary()
    assert d.n_favouring_a == 1


def test_paired_delta_single_seed_is_never_separated():
    """One seed can never establish separation, whatever the gap looks like."""
    d = paired_delta({0: [row("a", expl=0.1), row("b", expl=0.9)]}, "a", "b")
    assert d.separated() is False
    assert d.std == 0.0


def test_paired_delta_honours_metric_and_direction():
    per_seed = {
        0: [row("a", auroc=0.95), row("b", auroc=0.80)],
        1: [row("a", auroc=0.93), row("b", auroc=0.85)],
    }
    d = paired_delta(per_seed, "a", "b", metric="auroc")
    assert d.mean > 0 and d.separated()
    assert ">" in d.summary()


def test_paired_delta_rejects_unknown_metric_and_missing_name():
    per_seed = {0: [row("a"), row("b")]}
    with pytest.raises(ValueError):
        paired_delta(per_seed, "a", "b", metric="nope")
    with pytest.raises(KeyError):
        paired_delta(per_seed, "a", "missing")


# --- aggregation ----------------------------------------------------------


def test_multiseed_aggregates_mean_and_std(monkeypatch):
    calls = []

    def fake_eval(defenders, cfg, **kw):
        s = kw["seed"]
        calls.append(s)
        return [row("a", expl=0.3 + 0.1 * s), row("b", expl=0.5)]

    monkeypatch.setattr("arena.eval.multiseed.evaluate_defenders", fake_eval)
    agg, per_seed = multiseed_leaderboard({"a": None, "b": None}, seeds=(0, 1, 2))

    assert calls == [0, 1, 2]
    assert set(per_seed) == {0, 1, 2}
    a = next(r for r in agg if r.name == "a")
    assert a.n_seeds == 3
    assert a.mean["exploitability"] == pytest.approx(0.4)
    assert a.std["exploitability"] == pytest.approx(0.1)
    assert a.values["exploitability"] == pytest.approx((0.3, 0.4, 0.5))
    b = next(r for r in agg if r.name == "b")
    assert b.std["exploitability"] == 0.0  # identical every seed


def test_multiseed_calls_factory_per_seed(monkeypatch):
    """A defender that is itself trained must be rebuilt per seed, or the
    'multi-seed' result holds one arm fixed and the spread is a lie."""
    seen = []

    def factory(seed):
        seen.append(seed)
        return {"a": None}

    monkeypatch.setattr(
        "arena.eval.multiseed.evaluate_defenders",
        lambda d, c, **kw: [row("a", expl=0.1 * kw["seed"])],
    )
    multiseed_leaderboard(factory, seeds=(3, 4))
    assert seen == [3, 4]


def test_multiseed_sorted_by_mean_exploitability(monkeypatch):
    monkeypatch.setattr(
        "arena.eval.multiseed.evaluate_defenders",
        lambda d, c, **kw: [row("worse", expl=0.9), row("better", expl=0.2)],
    )
    agg, _ = multiseed_leaderboard({"x": None}, seeds=(0, 1))
    assert [r.name for r in agg] == ["better", "worse"]


def test_multiseed_rejects_empty_seeds():
    with pytest.raises(ValueError):
        multiseed_leaderboard({"a": None}, seeds=())


def test_metrics_cover_every_numeric_leaderboard_column():
    numeric = {
        f for f, v in vars(row("a")).items()
        if isinstance(v, float) and f != "n_decisions"
    }
    assert set(METRICS) == numeric


# --- formatting -----------------------------------------------------------


def test_format_aggregate_shows_mean_and_spread():
    agg = [AggregateRow(
        name="causal_monitor", n_seeds=3,
        mean={m: 0.5 for m in METRICS}, std={m: 0.02 for m in METRICS},
        values={m: (0.48, 0.5, 0.52) for m in METRICS},
    )]
    out = format_aggregate(agg)
    assert "causal_monitor" in out
    assert "0.500+/-0.020" in out
    assert "n_seeds = 3" in out


def test_format_aggregate_empty_does_not_crash():
    out = format_aggregate([])
    assert "defender" in out          # header still renders
    assert "n_seeds" not in out       # but no bogus count
