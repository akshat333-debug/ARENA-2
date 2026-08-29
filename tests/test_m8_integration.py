"""M8 integration: the leaderboard harness end to end.

The M8 gate beyond "metrics match a fixture" (that is test_metrics.py): the
harness runs every defender on the same scale and the ranking it produces is the
one the project claims.
"""

from __future__ import annotations

import pytest

from arena.baselines import CausalMonitor, SingleShotDetector, StaticAllowList
from arena.baselines.collect import collect_decisions, split
from arena.config import load_config
from arena.eval import evaluate_defenders, exploitability_curve, format_leaderboard
from arena.scripted import passive_blue, paranoid_blue

pytestmark = pytest.mark.slow

CFG = load_config("small.yaml")


@pytest.fixture(scope="module")
def rows():
    tr, _ = split(collect_decisions(120, 120, config=CFG, seed=1), seed=0)
    defenders = {
        "static_allowlist": StaticAllowList(),
        "single_shot": SingleShotDetector().fit(tr),
        "causal_monitor": CausalMonitor().fit(tr),
    }
    return {
        r.name: r
        for r in evaluate_defenders(
            defenders, CFG,
            n_decision_adv=120, n_decision_benign=120,
            br_steps=12_000, n_eval=200, seed=0,
        )
    }


def test_every_defender_gets_a_full_row(rows):
    assert set(rows) == {"static_allowlist", "single_shot", "causal_monitor"}
    for r in rows.values():
        assert 0.0 <= r.auroc <= 1.0
        assert 0.0 <= r.tpr_at_5pct_fpr <= 1.0
        assert 0.0 <= r.exploitability <= 1.0
        assert r.fresh_red_start < 0.5  # fresh best-response Red starts near chance


def test_calibration_puts_trainable_defenders_near_the_target_fpr(rows):
    for name in ("single_shot", "causal_monitor"):
        assert rows[name].operating_fpr == pytest.approx(CFG.eval.calibration_fpr, abs=0.04)


def test_auroc_ordering_reflects_information_available(rows):
    assert rows["static_allowlist"].auroc < rows["single_shot"].auroc < rows["causal_monitor"].auroc
    assert rows["causal_monitor"].auroc >= 0.85


def test_causal_monitor_is_the_least_exploitable_baseline_at_a_matched_fpr(rows):
    cm = rows["causal_monitor"].exploitability
    ss = rows["single_shot"].exploitability
    al = rows["static_allowlist"].exploitability
    assert cm < al, f"causal {cm:.3f} should be less exploitable than allowlist {al:.3f}"
    assert cm <= ss + 0.05, f"causal {cm:.3f} should not be worse than single-shot {ss:.3f}"


def test_single_shot_cannot_reach_the_causal_monitor_tpr_at_5pct_fpr(rows):
    assert rows["causal_monitor"].tpr_at_5pct_fpr - rows["single_shot"].tpr_at_5pct_fpr > 0.3


def test_format_leaderboard_is_a_table(rows):
    text = format_leaderboard(list(rows.values()))
    assert "AUROC" in text and "exploitability" in text
    for name in rows:
        assert name in text


def test_reference_policies_bracket_the_baselines():
    """A passive Blue is more exploitable than any real detector; a paranoid one
    is unexploitable (but that is not a good defender — see the FPR column)."""
    ref = {r.name: r for r in evaluate_defenders(
        {"passive": passive_blue, "paranoid": paranoid_blue}, CFG,
        n_decision_adv=80, n_decision_benign=80, br_steps=10_000, n_eval=150, seed=0,
    )}
    assert ref["passive"].exploitability > 0.3
    assert ref["paranoid"].exploitability == 0.0


def test_exploitability_curve_over_reference_blues_is_ordered():
    pts = exploitability_curve(
        [("passive", passive_blue), ("paranoid", paranoid_blue)],
        CFG, br_steps=8_000, n_eval=150, seed=0,
    )
    assert pts[0].result.exploitability > pts[1].result.exploitability
