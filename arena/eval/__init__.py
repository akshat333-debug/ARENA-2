"""Evaluation harness (M8): exploitability (primary) + AUROC / TPR@5%FPR."""

from arena.eval.exploitability import (
    ExploitabilityResult,
    curve_from_league,
    exploitability,
    exploitability_curve,
)
from arena.eval.harness import (
    LeaderboardRow,
    evaluate_defenders,
    format_leaderboard,
    policy_score_fn,
)
from arena.eval.metrics import roc_auc, roc_curve, tpr_at_fpr

__all__ = [
    "roc_auc", "roc_curve", "tpr_at_fpr",
    "exploitability", "exploitability_curve", "curve_from_league", "ExploitabilityResult",
    "evaluate_defenders", "format_leaderboard", "policy_score_fn", "LeaderboardRow",
]
