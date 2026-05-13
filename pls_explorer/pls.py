"""PLS regression pipelines: standard PLSR and sparse PLS via Lasso pre-selection."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import MultiTaskLasso
from sklearn.pipeline import Pipeline

from .preprocess import build_scaler


def make_pipeline(
    n_components: int,
    scaling: str = "mean_center",
    sparse: bool = False,
    sparse_lasso_alpha: float = 0.5,
) -> Pipeline:
    """Build a fit/predict-ready sklearn Pipeline for PLSR.

    Steps:
        scaler_x  : per the requested scaling mode
        [optional] feature_selection : MultiTaskLasso-based feature selection
        pls       : sklearn PLSRegression with scale=False (we handle scaling
                    upstream so all variants are commensurable).
    """
    steps = [("scaler_x", build_scaler(scaling))]
    if sparse:
        steps.append((
            "feature_selection",
            SelectFromModel(MultiTaskLasso(alpha=sparse_lasso_alpha, max_iter=5000)),
        ))
    steps.append(("pls", PLSRegression(n_components=n_components, scale=False)))
    return Pipeline(steps)


def fit_pls(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    scaling: str = "mean_center",
    sparse: bool = False,
    sparse_lasso_alpha: float = 0.5,
) -> Pipeline:
    """Fit a PLS pipeline on (X, Y); return the fitted Pipeline."""
    pipe = make_pipeline(
        n_components=n_components,
        scaling=scaling,
        sparse=sparse,
        sparse_lasso_alpha=sparse_lasso_alpha,
    )
    pipe.fit(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))
    return pipe


def extract_B(pipe: Pipeline) -> np.ndarray:
    """Full regression coefficient matrix B mapping scaled X to Y.

    For sklearn PLSRegression, B = pls.coef_.T (so X_scaled @ B is the
    centered Y prediction). For sparse pipelines, B is back-projected to the
    full feature space with zeros for non-selected features so all variants
    share a comparable (n_x_features, n_y_features) shape.
    """
    pls = pipe.named_steps["pls"]
    B_dense = pls.coef_.T  # (n_selected, n_y_features)

    if "feature_selection" in pipe.named_steps:
        selector = pipe.named_steps["feature_selection"]
        mask = selector.get_support()
        n_x_full = mask.size
        B = np.zeros((n_x_full, B_dense.shape[1]))
        B[mask, :] = B_dense
        return B
    return B_dense


def pls_predict(pipe: Pipeline, X: np.ndarray, relu_clip: bool = True) -> np.ndarray:
    """Predict Y from X using the fitted pipeline; optionally clip at 0."""
    Y_hat = pipe.predict(np.asarray(X, dtype=float))
    if relu_clip:
        Y_hat = np.clip(Y_hat, 0, None)
    return Y_hat


def plsc_svd(X: np.ndarray, Y: np.ndarray) -> dict:
    """PLS Correlation via SVD of the cross-product matrix R = X^T Y.

    Returns
    -------
    {'U': ndarray, 'singular_values': ndarray, 'V': ndarray,
     'L_X': scores X @ V, 'L_Y': scores Y @ U}
    Per Krishnan et al. 2011 / Van Roon 2014.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    R = Xc.T @ Yc
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    V = Vt.T
    return {
        "U": U,
        "singular_values": s,
        "V": V,
        "L_X": Xc @ V,
        "L_Y": Yc @ U,
    }
