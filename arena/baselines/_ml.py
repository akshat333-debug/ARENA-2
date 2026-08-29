"""Shared fitting helper for the trained baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedClassifier:
    scaler: StandardScaler
    clf: LogisticRegression
    threshold: float

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


def fit_classifier(
    X: np.ndarray, y: np.ndarray, val_X: np.ndarray, val_y: np.ndarray
) -> FittedClassifier:
    """Standardise, fit logistic regression, pick the threshold that maximises
    Youden's J (TPR − FPR) on the validation split. Falls back gracefully when a
    split has only one class."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    if len(np.unique(y)) < 2:
        # Degenerate training set: predict the single class, never fire.
        clf.fit(np.vstack([X, X[:1]]), np.concatenate([y, [1 - y[0]]]))
        return FittedClassifier(scaler, clf, threshold=1.01)
    clf.fit(scaler.transform(X), y)

    if len(val_X) == 0 or len(np.unique(val_y)) < 2:
        return FittedClassifier(scaler, clf, threshold=0.5)

    p = clf.predict_proba(scaler.transform(np.asarray(val_X, dtype=np.float64)))[:, 1]
    val_y = np.asarray(val_y, dtype=int)
    best_t, best_j = 0.5, -1.0
    for t in np.unique(np.concatenate([[0.0], p, [1.0]])):
        pred = p >= t
        tp = int(np.sum(pred & (val_y == 1)))
        fn = int(np.sum(~pred & (val_y == 1)))
        fp = int(np.sum(pred & (val_y == 0)))
        tn = int(np.sum(~pred & (val_y == 0)))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_t, best_j = float(t), j
    return FittedClassifier(scaler, clf, threshold=best_t)
