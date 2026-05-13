"""Null-model generators for the low → high PLS analysis.

Two domain-specific nulls per email exchange with Burton & Wachowiak
(see vault Resources/resource-071.md):

Null 1 — homogeneous gain / multiplicative broadening
    Y_null1[:, j] = alpha_j * X[:, j], where alpha_j is the OLS scalar that
    minimizes ||Y[:, j] - alpha * X[:, j]||^2.
    Expected PLS signature on (X, Y_null1): B approximately diagonal.

Null 2 — random retuning / fictive high matrices
    Y_null2[:, j] = X[:, random_index]. Preserves the marginal distribution
    of low-conc responses, breaks the specific low->high correspondence.
    Expected PLS signature on (X, Y_null2): B noisy/unstructured.
"""

from __future__ import annotations

import numpy as np


def homogeneous_gain_null(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build Null-1 (multiplicative gain).

    Returns
    -------
    Y_null : ndarray, shape (n_samples, n_features)
        alpha_j * X[:, j] for each feature j.
    alpha : ndarray, shape (n_features,)
        Per-feature scaling factors recovered by OLS.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    num = (X * Y).sum(axis=0)
    den = (X * X).sum(axis=0)
    alpha = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    Y_null = X * alpha[np.newaxis, :]
    return Y_null, alpha


def random_retuning_null(
    X: np.ndarray,
    mode: str = "column_resample",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Build Null-2 (random retuning).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Low-concentration response matrix.
    mode : str
        'column_resample' : Y_null[:, j] = X[:, k] where k is sampled with
                            replacement from {0, ..., n_features-1}. Each
                            high-conc column independently re-tunes to some
                            random low-conc glom. (Default, matches "randomly
                            sampling low-concentration glomeruli".)
        'column_permute'  : Y_null = X[:, perm] for a random permutation
                            without replacement.
        'row_permute'     : Y_null = X[perm, :] for a row permutation
                            (classic statistical permutation null).
    rng : Generator or int seed
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    n_rows, n_cols = X.shape

    if mode == "column_resample":
        idx = rng.integers(0, n_cols, size=n_cols)
        return X[:, idx]
    if mode == "column_permute":
        return X[:, rng.permutation(n_cols)]
    if mode == "row_permute":
        return X[rng.permutation(n_rows), :]
    raise ValueError(f"Unknown null2 mode: {mode!r}")


def build_null_targets(
    X: np.ndarray,
    Y: np.ndarray,
    null2_mode: str = "column_resample",
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Return {'real': Y, 'null1': Y_null1, 'null2': Y_null2} for a paired run."""
    rng = np.random.default_rng(rng)
    Y_null1, _ = homogeneous_gain_null(X, Y)
    Y_null2 = random_retuning_null(X, mode=null2_mode, rng=rng)
    return {"real": np.asarray(Y, dtype=float), "null1": Y_null1, "null2": Y_null2}
