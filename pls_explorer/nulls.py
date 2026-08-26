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


def subset_mix(
    X: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator | int | None = None,
    K: int = 10,
) -> np.ndarray:
    """Build a "subset-mix" null.

    For each column j of Y, replace it with the mean of X[:, k] taken across
    K randomly chosen other glomeruli (k != j, sampled without replacement).

    K=1 reduces to a single-swap variant of Null-2 (without-replacement, j
    excluded); larger K averages over more donors and produces a high-conc
    column that approaches the population mean of X as K grows.

    Y itself is unused in the construction but kept in the signature for API
    parity with the other null builders, and the returned array has the same
    shape as Y.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Low-concentration response matrix.
    Y : ndarray, shape (n_samples, n_features)
        Used only for shape; entries are ignored.
    rng : Generator or int seed
    K : int
        Number of donor glomeruli to average per target glomerulus.
    """
    rng_ = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_gloms = X.shape[1]
    if K < 1 or K > n_gloms - 1:
        raise ValueError(f"K must be in [1, n_gloms-1={n_gloms - 1}], got {K}")
    Y_new = np.zeros_like(Y)
    all_idx = np.arange(n_gloms)
    for j in range(n_gloms):
        candidates = np.setdiff1d(all_idx, [j], assume_unique=True)
        chosen = rng_.choice(candidates, size=K, replace=False)
        Y_new[:, j] = X[:, chosen].mean(axis=1)
    return Y_new


def diagonal_plus_random_null(
    X: np.ndarray,
    Y: np.ndarray,
    K: int = 10,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Build Null-3 (diagonal + random): a stronger competitor than pure random.

        Y_null[:, j] = alpha_j * X[:, j]   (diagonal: per-glom homogeneous gain == Null-1)
                       + off[:, j]          (off-diagonal: randomized residual)

    The residual R = Y - diag(gain) carries the cross-glomerular (cluster)
    structure. Each off[:, j] is the MEAN of the residuals of K randomly chosen
    OTHER glomeruli (a diffuse mix of many random donors, NOT a single random
    swap), rescaled so the off-diagonal's total energy matches ||R||_F. This
    grants the trivial diagonal and *randomizes* the off-diagonal, so the PLS
    fit on (X, Y_null) has a diagonal but no cluster organization — the right
    null for asking whether the real off-diagonal is genuinely cluster-structured
    rather than random mixing layered on a diagonal.

    Parameters
    ----------
    X, Y : ndarray, shape (n_samples, n_features)
    K : int
        Number of random donor glomeruli averaged into each off-diagonal column.
    rng : Generator or int seed
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_rows, n_cols = X.shape
    num = (X * Y).sum(axis=0)
    den = (X * X).sum(axis=0)
    alpha = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    D = X * alpha[np.newaxis, :]
    R = Y - D
    K = int(min(max(K, 1), n_cols - 1))
    off = np.empty_like(R)
    idx = np.arange(n_cols)
    for j in range(n_cols):
        donors = rng.choice(idx[idx != j], size=K, replace=False)
        off[:, j] = R[:, donors].mean(axis=1)
    rn = float(np.linalg.norm(R))
    on = float(np.linalg.norm(off))
    if on > 0:
        off *= rn / on                       # energy-match to the real residual
    return D + off


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
