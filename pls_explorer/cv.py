"""Cross-validation: family-stratified k-fold + LOO; fold-level evaluation."""

from __future__ import annotations

from typing import Callable, Iterator, Optional

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .pls import fit_pls, pls_predict


def make_folds(
    n_samples: int,
    scheme: str = "family_stratified",
    n_splits: int = 4,
    chemical_group: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(train_idx, test_idx), ...] for the requested scheme."""
    if scheme == "loo":
        loo = LeaveOneOut()
        idx = np.arange(n_samples)
        return [(tr, te) for tr, te in loo.split(idx)]

    if scheme == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        idx = np.arange(n_samples)
        return [(tr, te) for tr, te in kf.split(idx)]

    if scheme == "family_stratified":
        if chemical_group is None:
            raise ValueError("family_stratified requires chemical_group")
        labels = LabelEncoder().fit_transform(chemical_group)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        idx = np.arange(n_samples)
        return [(tr, te) for tr, te in skf.split(idx, labels)]

    raise ValueError(f"Unknown cv scheme: {scheme!r}")


def cross_val_predict_pls(
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_components: int,
    scaling: str = "mean_center",
    sparse: bool = False,
    sparse_lasso_alpha: float = 0.5,
    relu_clip: bool = True,
) -> np.ndarray:
    """Out-of-fold predictions for every sample, in row order of Y.

    Each test fold is predicted by a fresh pipeline trained on the corresponding
    train rows. Output shape = Y.shape.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Y_hat = np.full_like(Y, np.nan, dtype=float)
    for tr, te in folds:
        pipe = fit_pls(
            X[tr], Y[tr],
            n_components=n_components,
            scaling=scaling,
            sparse=sparse,
            sparse_lasso_alpha=sparse_lasso_alpha,
        )
        Y_hat[te] = pls_predict(pipe, X[te], relu_clip=relu_clip)
    return Y_hat


def cross_val_score_curve(
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_components_grid: list[int],
    scaling: str = "mean_center",
    sparse: bool = False,
    sparse_lasso_alpha: float = 0.5,
    relu_clip: bool = True,
) -> dict:
    """Sweep n_components, returning per-grid-point Q² and RMSECV.

    Returns
    -------
    {
        'n_components_grid': list[int],
        'q2': ndarray (len = grid),
        'rmsecv': ndarray (len = grid),
        'rmsec': ndarray (len = grid),
        'y_hat_oof': list of ndarrays (per grid point), each shape Y.shape,
    }
    """
    from .metrics import q2_global, rmse, r2_score_global

    q2_curve = np.empty(len(n_components_grid))
    rmsecv_curve = np.empty(len(n_components_grid))
    rmsec_curve = np.empty(len(n_components_grid))
    yhats = []
    for i, nc in enumerate(n_components_grid):
        Y_hat = cross_val_predict_pls(
            X, Y, folds, n_components=nc,
            scaling=scaling, sparse=sparse,
            sparse_lasso_alpha=sparse_lasso_alpha, relu_clip=relu_clip,
        )
        yhats.append(Y_hat)
        q2_curve[i] = q2_global(Y, Y_hat)
        rmsecv_curve[i] = rmse(Y, Y_hat)
        # in-sample RMSE for the same nc, used to compute RMSECV/RMSEC overfit ratio
        from .pls import fit_pls as _fit
        pipe = _fit(X, Y, n_components=nc, scaling=scaling,
                    sparse=sparse, sparse_lasso_alpha=sparse_lasso_alpha)
        from .pls import pls_predict as _predict
        Y_cal = _predict(pipe, X, relu_clip=relu_clip)
        rmsec_curve[i] = rmse(Y, Y_cal)

    return {
        "n_components_grid": list(n_components_grid),
        "q2": q2_curve,
        "rmsecv": rmsecv_curve,
        "rmsec": rmsec_curve,
        "y_hat_oof": yhats,
    }
