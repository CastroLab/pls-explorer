"""Biology-flavored baselines ported from PLS_play3.

These answer the *interpretive* question (how does PLS compare against
simple, interpretable mapping rules?) — distinct from the *statistical*
nulls in nulls.py, which test whether the W matrix has structure beyond
trivial gain or random retuning.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class ClusterLinearRegression:
    """Per-cluster scalar regression: Y[:, j] = a_c * X[:, j] + b_c for the
    cluster c that glomerulus j belongs to. All glomeruli in the same cluster
    share a single (slope, intercept) pair fit by pooling that cluster's data.

    Used as a baseline to compare against full PLSR — answers the question
    "is high simply a per-cluster affine map of low?"
    """

    def __init__(self, cluster_assignments: np.ndarray):
        self.clusters = np.asarray(cluster_assignments)
        self.models: dict = {}

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        for c in np.unique(self.clusters):
            mask = self.clusters == c
            x_pool = X[:, mask].reshape(-1, 1)
            y_pool = Y[:, mask].reshape(-1)
            lr = LinearRegression().fit(x_pool, y_pool)
            self.models[c] = {"model": lr, "mask": mask}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y_pred = np.zeros_like(X)
        for c, info in self.models.items():
            mask = info["mask"]
            n_c = int(mask.sum())
            x_c = X[:, mask].reshape(-1, 1)
            y_pred = info["model"].predict(x_c).reshape(-1, n_c)
            Y_pred[:, mask] = y_pred
        return Y_pred

    def score(self, X, Y) -> float:
        return r2_score(np.asarray(Y).reshape(-1), self.predict(X).reshape(-1))


class PerGlomScaling:
    """Y[:, j] = alpha_j * X[:, j]. Identical to nulls.homogeneous_gain_null
    in spirit but exposed as a fit/predict baseline."""

    def __init__(self):
        self.alpha_ = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        num = (X * Y).sum(axis=0)
        den = (X * X).sum(axis=0)
        self.alpha_ = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) * self.alpha_[np.newaxis, :]

    def score(self, X, Y):
        return r2_score(np.asarray(Y).reshape(-1), self.predict(X).reshape(-1))


class SimpleScaling:
    """Y = alpha * X with a single scalar alpha across all entries."""

    def __init__(self):
        self.alpha_ = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        num = float((X * Y).sum())
        den = float((X * X).sum())
        self.alpha_ = num / den if den > 0 else 0.0
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) * self.alpha_

    def score(self, X, Y):
        return r2_score(np.asarray(Y).reshape(-1), self.predict(X).reshape(-1))


def mean_predictor() -> DummyRegressor:
    """Per-feature mean predictor — the most pessimistic baseline."""
    return DummyRegressor(strategy="mean")
