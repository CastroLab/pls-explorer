"""Model-level permutation test: row-shuffle Y inside the CV loop.

Yields a null distribution of Q² values that the observed Q² is tested against.
"""

from __future__ import annotations

import numpy as np

from .cv import cross_val_predict_pls
from .metrics import q2_global


def permutation_q2_null(
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_components: int,
    n_permutations: int = 1000,
    scaling: str = "mean_center",
    sparse: bool = False,
    sparse_lasso_alpha: float = 0.5,
    relu_clip: bool = True,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Refit the same pipeline on row-permuted Y; collect Q² under the null.

    Returns
    -------
    {
        'q2_observed': float,
        'q2_null': ndarray of shape (n_permutations,),
        'p_value': float,
    }
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]

    Y_hat = cross_val_predict_pls(
        X, Y, folds, n_components=n_components, scaling=scaling,
        sparse=sparse, sparse_lasso_alpha=sparse_lasso_alpha, relu_clip=relu_clip,
    )
    q2_obs = q2_global(Y, Y_hat)

    q2_null = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        Y_perm = Y[perm]
        Y_hat_perm = cross_val_predict_pls(
            X, Y_perm, folds, n_components=n_components, scaling=scaling,
            sparse=sparse, sparse_lasso_alpha=sparse_lasso_alpha, relu_clip=relu_clip,
        )
        q2_null[i] = q2_global(Y_perm, Y_hat_perm)

    p_value = float((q2_null >= q2_obs).mean())
    return {"q2_observed": float(q2_obs), "q2_null": q2_null, "p_value": p_value}
