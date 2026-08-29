"""Detection metrics — the canonical implementations.

AUROC and TPR at a fixed FPR, deliberately matching CASPIAN (2026) so ARENA's
secondary numbers sit on the same scale as the most recent published detector
(project.md S8). Rank-based, no sklearn dependency in the hot path; checked
against ``sklearn.metrics`` values in the tests.

``arena.baselines.evaluate`` (M4) keeps its own light copies for the baseline
demo; these are what the M8 harness and the leaderboard use.
"""

from __future__ import annotations

import numpy as np

ArrayLike = "np.ndarray | list[float]"


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must match")
    if s.size == 0:
        raise ValueError("empty inputs")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return s, y


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve via the Mann–Whitney U statistic, with proper
    mid-rank handling of ties. Returns 0.5 when either class is absent."""
    s, y = _as_arrays(scores, labels)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # mid-ranks
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    _, inv, counts = np.unique(s, return_counts=True, return_inverse=True)
    csum = np.cumsum(counts)
    midrank = (csum - counts + csum + 1) / 2.0
    ranks = midrank[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fpr, tpr, thresholds)`` sorted by decreasing threshold, endpoints
    included. Thin, exact — used for plotting the leaderboard's ROC panel."""
    s, y = _as_arrays(scores, labels)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve needs both classes present")

    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, s.size - 1]
    tps = np.cumsum(y)[idx]
    fps = np.cumsum(1 - y)[idx]
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thr = np.r_[np.inf, s[idx]]
    return fpr, tpr, thr


def tpr_at_fpr(scores, labels, fpr: float = 0.05) -> float:
    """True-positive rate at the operating point whose false-positive rate is
    ``<= fpr``. The threshold is the ``1 - fpr`` quantile of the negative scores
    (nudged above ties); TPR is measured on the positives.

    This is the M8 headline secondary metric — a defender that only catches
    attacks by also drowning benign traffic scores near zero here."""
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")
    s, y = _as_arrays(scores, labels)
    neg = s[y == 0]
    pos = s[y == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    thr = np.nextafter(float(np.quantile(neg, 1.0 - fpr)), np.inf)
    return float(np.mean(pos >= thr))


def confusion_at_threshold(scores, labels, threshold: float) -> dict:
    s, y = _as_arrays(scores, labels)
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }
