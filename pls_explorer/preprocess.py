"""Preprocessing for PLS: log, scaling, row normalization.

Convention: every function takes an ndarray of shape (n_samples, n_features)
and returns an ndarray of the same shape. Stateful transformations (mean
removal, scaling) are absorbed into the sklearn Pipeline; the functions here
operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np


def log1p_positive(X: np.ndarray, offset: float = 1.0) -> np.ndarray:
    """log(X + offset). Wachowiak 2022 reports lognormal response amplitudes.

    Negative entries (rare; occur post-baseline subtraction) are clipped at 0
    before the log so the transform stays defined.
    """
    return np.log(np.clip(X, 0, None) + offset)


def row_l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row to unit norm. Strips magnitude; tests whether
    the spatial *pattern* of activation across glomeruli is preserved.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def build_scaler(scaling: str):
    """Return an sklearn transformer for the requested scaling mode.

    Modes
    -----
    'mean_center'  : subtract per-column mean only (no variance scaling)
    'autoscale'    : subtract mean + divide by std (StandardScaler)
    'pareto'       : subtract mean + divide by sqrt(std)
    'none'         : no-op
    """
    from sklearn.preprocessing import StandardScaler, FunctionTransformer

    if scaling == "mean_center":
        return StandardScaler(with_mean=True, with_std=False)
    if scaling == "autoscale":
        return StandardScaler(with_mean=True, with_std=True)
    if scaling == "pareto":
        return ParetoScaler()
    if scaling == "none":
        return FunctionTransformer(validate=False)
    raise ValueError(f"Unknown scaling mode: {scaling!r}")


class ParetoScaler:
    """Mean-center then divide by sqrt(column std). Sklearn-compatible."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.sqrt(X.std(axis=0, ddof=0))
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kwargs):
        return self
