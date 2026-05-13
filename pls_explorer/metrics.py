"""Performance metrics and W-structure summaries."""

from __future__ import annotations

import numpy as np


# ---- regression metrics ---------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error over all entries."""
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt((e * e).mean()))


def q2_global(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cross-validated R² treating the entire matrix as one prediction problem.

    Q² = 1 - SS_res / SS_tot, where SS_tot is computed against the column
    means of y_true (the per-column mean is the natural null for multi-output).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def q2_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column Q² (length n_features). For features (gloms) whose variance
    in y_true is zero, returns NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    return out


def r2_score_global(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """In-sample R² with the same global formula as q2_global (no CV)."""
    return q2_global(y_true, y_pred)


def per_odorant_residual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-row L2 norm of residuals (length n_samples)."""
    return np.linalg.norm(np.asarray(y_true) - np.asarray(y_pred), axis=1)


# ---- variance explained per component -------------------------------------

def variance_explained(pipe) -> dict:
    """Per-component cumulative R²X and R²Y from a fitted PLS pipeline.

    Sklearn doesn't expose this directly; we recompute from the scores and
    loadings: R²X_cum[h] = 1 - ||X_centered - T[:, :h+1] @ P[:, :h+1].T||² / ||X_centered||².
    Similar for Y using x_scores and y_loadings.
    """
    pls = pipe.named_steps["pls"]
    # sklearn keeps centered X/Y deflated through fitting; reconstruct via scores
    T = pls.x_scores_      # (n, H)
    P = pls.x_loadings_    # (p, H)
    U = pls.y_scores_      # (n, H)
    Q = pls.y_loadings_    # (q, H)

    # Reconstruct centered X and Y (PLS uses centered values internally)
    X_c = T @ P.T
    Y_c = U @ Q.T

    total_X = np.sum(X_c ** 2)
    total_Y = np.sum(Y_c ** 2)
    H = T.shape[1]

    r2x_cum = np.empty(H)
    r2y_cum = np.empty(H)
    for h in range(H):
        Xh = T[:, : h + 1] @ P[:, : h + 1].T
        Yh = U[:, : h + 1] @ Q[:, : h + 1].T
        r2x_cum[h] = 1.0 - np.sum((X_c - Xh) ** 2) / total_X if total_X > 0 else np.nan
        r2y_cum[h] = 1.0 - np.sum((Y_c - Yh) ** 2) / total_Y if total_Y > 0 else np.nan

    return {"r2x_cumulative": r2x_cum, "r2y_cumulative": r2y_cum}


# ---- VIP ------------------------------------------------------------------

def compute_vip(pipe) -> np.ndarray:
    """Variable Importance in Projection score per X-feature.

    VIP_j = sqrt(P * sum_h(SS_h * w_jh²) / sum_h SS_h),
    where SS_h = ||t_h||² * ||q_h||² (variance in Y captured by component h).
    """
    pls = pipe.named_steps["pls"]
    W = pls.x_weights_     # (p, H)
    T = pls.x_scores_      # (n, H)
    Q = pls.y_loadings_    # (q, H)
    P_feat = W.shape[0]

    SS = (T ** 2).sum(axis=0) * (Q ** 2).sum(axis=0)
    total_SS = SS.sum()
    if total_SS <= 0:
        return np.zeros(P_feat)

    # normalize weights to unit length so VIP comparisons across H are clean
    W_norm = W / np.linalg.norm(W, axis=0, keepdims=True)
    vip = np.sqrt(P_feat * (W_norm ** 2 @ SS) / total_SS)
    return vip


# ---- B-matrix structure ---------------------------------------------------

def diagonality_index(B: np.ndarray) -> float:
    """Ratio of diagonal energy to total energy in |B| for a square block.

    For non-square B (n_x_features != n_y_features), uses the leading
    min(n_x, n_y) × min(n_x, n_y) block. Returns NaN if B is empty.

    1.0 = pure diagonal; 0.0 = no diagonal contribution.
    """
    B = np.abs(np.asarray(B, dtype=float))
    if B.size == 0:
        return float("nan")
    k = min(B.shape)
    if k == 0:
        return float("nan")
    diag_energy = np.sum(np.diag(B[:k, :k]) ** 2)
    total_energy = np.sum(B[:k, :k] ** 2)
    return float(diag_energy / total_energy) if total_energy > 0 else float("nan")


def off_diagonal_entropy(B: np.ndarray) -> float:
    """Shannon entropy of |B|² over all entries (normalized by log(size)).

    Higher = more uniform / "noisier" structure (Null-2-like).
    Lower  = mass concentrated in a few entries (Null-1-like, if diagonal).
    """
    B = np.abs(np.asarray(B, dtype=float)) ** 2
    s = B.sum()
    if s <= 0:
        return float("nan")
    p = (B / s).ravel()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(H / np.log(B.size))


def w_structure_summary(B: np.ndarray) -> dict:
    """Compact dict of B-matrix structure metrics."""
    return {
        "diagonality_index": diagonality_index(B),
        "off_diagonal_entropy": off_diagonal_entropy(B),
        "frobenius_norm": float(np.linalg.norm(B)),
        "shape": list(B.shape),
        "max_abs": float(np.max(np.abs(B))) if B.size else float("nan"),
    }
