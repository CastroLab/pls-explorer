"""Component selection: one-sigma rule + Van der Voet randomization."""

from __future__ import annotations

import numpy as np


def one_sigma_ncomp(rmsecv: np.ndarray, grid: list[int]) -> int:
    """Smallest n_components whose RMSECV is within 1 SE of the global minimum.

    SE here is approximated as the std of the RMSECV curve over the grid —
    a coarse heuristic suitable for a single CV run. For a per-fold SE
    estimate, pass an aggregated CV result through this function with the
    fold-wise std computed upstream.
    """
    rmsecv = np.asarray(rmsecv, dtype=float)
    grid = list(grid)
    if rmsecv.size == 0:
        raise ValueError("Empty RMSECV curve")
    se = np.std(rmsecv, ddof=1) if rmsecv.size > 1 else 0.0
    min_val = rmsecv.min()
    threshold = min_val + se
    for i, v in enumerate(rmsecv):
        if v <= threshold:
            return grid[i]
    return grid[int(np.argmin(rmsecv))]


def van_der_voet_ncomp(
    Y: np.ndarray,
    y_hat_per_grid: list[np.ndarray],
    grid: list[int],
    n_permutations: int = 1000,
    alpha: float = 0.01,
    rng: np.random.Generator | int | None = None,
) -> int:
    """Van der Voet (1994) randomization test for picking n_components.

    Compares squared residuals of the global-minimum model against each
    candidate model via a sign-permutation test on per-sample squared error
    differences. Returns the smallest n_components NOT significantly worse
    than the global minimum at level alpha.

    Parameters
    ----------
    Y : ndarray (n_samples, n_features)
        True targets.
    y_hat_per_grid : list of ndarrays
        Out-of-fold predictions per candidate n_components.
    grid : list of int
    n_permutations : int
    alpha : float
        Significance threshold; default 0.01 per Mevik's pls package.

    Notes
    -----
    The per-sample squared-error vector is e_i = ||y_i - y_hat_i||²; the test
    statistic is t = mean(e_alt - e_min) and the null is sign-randomized
    differences (Wilcoxon-style).
    """
    rng = np.random.default_rng(rng)
    Y = np.asarray(Y, dtype=float)
    err = np.stack([np.sum((Y - yh) ** 2, axis=1) for yh in y_hat_per_grid], axis=0)
    # err shape: (n_grid, n_samples)
    rmsecv_per_model = np.sqrt(err.mean(axis=1))
    best_idx = int(np.argmin(rmsecv_per_model))
    n_samples = err.shape[1]

    # For each candidate i, test whether err[i] > err[best]
    p_values = np.ones(len(grid))
    for i in range(len(grid)):
        if i == best_idx:
            p_values[i] = 1.0
            continue
        diff = err[i] - err[best_idx]
        observed = diff.mean()
        # null: random sign flips on each per-sample difference
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_samples))
        null_stats = (signs * diff).mean(axis=1)
        p_values[i] = float((null_stats >= observed).mean())

    not_worse = np.where(p_values > alpha)[0]
    if len(not_worse) == 0:
        return grid[best_idx]
    return grid[int(not_worse.min())]


def select_ncomp(
    Y: np.ndarray,
    y_hat_per_grid: list[np.ndarray],
    rmsecv: np.ndarray,
    grid: list[int],
    method: str = "one_sigma",
    **kwargs,
) -> dict:
    """Run a selection method and return a dict with both selectors' picks."""
    one_sigma = one_sigma_ncomp(rmsecv, grid)
    vdv = van_der_voet_ncomp(Y, y_hat_per_grid, grid, **kwargs)
    chosen = one_sigma if method == "one_sigma" else vdv
    return {
        "chosen": chosen,
        "method": method,
        "one_sigma": one_sigma,
        "van_der_voet": vdv,
        "agree": one_sigma == vdv,
    }
