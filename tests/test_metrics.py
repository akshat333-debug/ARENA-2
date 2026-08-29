"""M8 unit tests: detection metrics against hand-computed and sklearn values."""

from __future__ import annotations

import numpy as np
import pytest

from arena.eval.metrics import (
    confusion_at_threshold,
    roc_auc,
    roc_curve,
    tpr_at_fpr,
)

sklearn_metrics = pytest.importorskip("sklearn.metrics")


# --- roc_auc: hand-computed --------------------------------------


def test_auc_textbook_example_is_exactly_three_quarters():
    # 2 positives, 2 negatives; one positive (0.35) ranks below one negative (0.4)
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_auc_perfect_separation_is_one():
    assert roc_auc([0.1, 0.2, 0.9, 0.95], [0, 0, 1, 1]) == 1.0


def test_auc_perfectly_wrong_is_zero():
    assert roc_auc([0.9, 0.95, 0.1, 0.2], [0, 0, 1, 1]) == 0.0


def test_auc_all_scores_equal_is_one_half():
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_auc_handles_ties_with_midranks():
    # positives {0.5, 0.5}, negatives {0.5, 0.1}: the tied pair contributes 0.5
    assert roc_auc([0.5, 0.1, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_auc_single_class_returns_half():
    assert roc_auc([0.2, 0.7, 0.9], [1, 1, 1]) == 0.5
    assert roc_auc([0.2, 0.7, 0.9], [0, 0, 0]) == 0.5


@pytest.mark.parametrize("seed", range(6))
def test_auc_matches_sklearn_on_random_data(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(20, 120)
    scores = rng.random(n)
    labels = rng.integers(0, 2, n)
    if labels.min() == labels.max():
        labels[0] = 1 - labels[0]
    assert roc_auc(scores, labels) == pytest.approx(
        sklearn_metrics.roc_auc_score(labels, scores), abs=1e-9
    )


# --- roc_curve --------------------------------------------------


def test_roc_curve_matches_sklearn_without_intermediate_dropping():
    scores = [0.1, 0.4, 0.35, 0.8, 0.6, 0.2]
    labels = [0, 0, 1, 1, 1, 0]
    f, t, _ = roc_curve(scores, labels)
    sf, st, _ = sklearn_metrics.roc_curve(labels, scores, drop_intermediate=False)
    assert np.allclose(f, sf) and np.allclose(t, st)


def test_roc_curve_area_equals_roc_auc():
    rng = np.random.default_rng(3)
    scores = rng.random(80)
    labels = rng.integers(0, 2, 80)
    f, t, _ = roc_curve(scores, labels)
    assert np.trapezoid(t, f) == pytest.approx(roc_auc(scores, labels), abs=1e-9)


def test_roc_curve_starts_at_origin_ends_at_one_one():
    f, t, _ = roc_curve([0.1, 0.9, 0.5, 0.5], [0, 1, 0, 1])
    assert (f[0], t[0]) == (0.0, 0.0)
    assert (f[-1], t[-1]) == (1.0, 1.0)


def test_roc_curve_needs_both_classes():
    with pytest.raises(ValueError):
        roc_curve([0.1, 0.2], [1, 1])


# --- tpr_at_fpr -----------------------------------------------


def test_tpr_at_fpr_hand_computed():
    neg = [0.1, 0.2, 0.3, 0.4]
    pos = [0.35, 0.5, 0.6]
    scores = neg + pos
    labels = [0, 0, 0, 0, 1, 1, 1]
    # 25% FPR -> threshold at the 0.75 quantile of negs = 0.325 (nudged up).
    # positives >= 0.325: 0.35, 0.5, 0.6 -> all three.
    assert tpr_at_fpr(scores, labels, 0.25) == pytest.approx(1.0)
    # 0% budget is impossible to express as an open interval; 1/4 -> stricter:
    # threshold just above max neg (0.4); only 0.5 and 0.6 pass -> 2/3.
    assert tpr_at_fpr(scores, labels, 1e-6) == pytest.approx(2 / 3)


def test_tpr_at_fpr_perfect_detector_is_one_even_at_tiny_fpr():
    scores = [0.1, 0.15, 0.2, 0.9, 0.95, 0.99]
    labels = [0, 0, 0, 1, 1, 1]
    assert tpr_at_fpr(scores, labels, 0.01) == 1.0


def test_tpr_at_fpr_rejects_out_of_range():
    with pytest.raises(ValueError):
        tpr_at_fpr([0.1, 0.9], [0, 1], 0.0)
    with pytest.raises(ValueError):
        tpr_at_fpr([0.1, 0.9], [0, 1], 1.0)


def test_tpr_at_fpr_nan_when_a_class_is_missing():
    assert np.isnan(tpr_at_fpr([0.1, 0.2], [0, 0], 0.05))


# --- confusion_at_threshold --------------------------------


def test_confusion_counts_and_rates():
    scores = [0.9, 0.8, 0.3, 0.2, 0.6]
    labels = [1, 0, 1, 0, 1]
    c = confusion_at_threshold(scores, labels, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert c["tpr"] == pytest.approx(2 / 3)
    assert c["fpr"] == pytest.approx(1 / 2)
    assert c["precision"] == pytest.approx(2 / 3)


# --- input validation --------------------------------------


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        roc_auc([0.1, 0.2, 0.3], [0, 1])


def test_non_binary_labels_raise():
    with pytest.raises(ValueError):
        roc_auc([0.1, 0.2], [0, 2])


def test_empty_raises():
    with pytest.raises(ValueError):
        roc_auc([], [])
