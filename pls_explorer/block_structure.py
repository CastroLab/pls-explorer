"""Cluster-block analysis of B: is the linear map organized by NMF cluster?

The headline test: reorder B by glomerular cluster id, decompose the total
Frobenius energy into within-cluster (diagonal blocks) and between-cluster
(off-diagonal blocks), and compare against a cluster-label permutation null.

Why this matters: a high `E_within / E_total` under the real cluster labels
combined with a low value under random cluster permutations is direct
evidence that the linear map operates at the cluster level — the headline
"group-level coding" claim.

For paired low/high glomerular data, both axes of B index the same physical
glomeruli, so `row_clusters == col_clusters` (the same vector). The null
permutation must therefore reassign rows and columns *together*, with one
shared permutation per draw.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def decompose_block_energy(
    B: np.ndarray,
    row_clusters: np.ndarray,
    col_clusters: Optional[np.ndarray] = None,
) -> dict:
    """Per-cluster-block Frobenius norm of B, plus within/between summaries.

    Parameters
    ----------
    B : ndarray (n_row, n_col)
    row_clusters : ndarray of length n_row
    col_clusters : ndarray of length n_col, defaults to row_clusters
        For paired data both axes share the same glomeruli.

    Returns
    -------
    {
      'block_norm'        : (K_r, K_c) ndarray; per-block sqrt(sum |B[i,j]|²)
      'unique_row'        : sorted unique row clusters
      'unique_col'        : sorted unique col clusters
      'E_total'           : ||B||²_F
      'E_within'          : sum of |B|² over diagonal blocks (when row==col)
      'E_within_frac'     : E_within / E_total
      'E_between'         : E_total - E_within (when row==col)
      'E_between_frac'    : 1 - E_within_frac
    }
    """
    B = np.asarray(B, dtype=float)
    row_clusters = np.asarray(row_clusters)
    col_clusters = row_clusters if col_clusters is None else np.asarray(col_clusters)
    unique_r = np.sort(np.unique(row_clusters))
    unique_c = np.sort(np.unique(col_clusters))

    B2 = B ** 2
    Kr, Kc = len(unique_r), len(unique_c)
    block_sq = np.zeros((Kr, Kc))
    for i, r in enumerate(unique_r):
        rows = row_clusters == r
        for j, c in enumerate(unique_c):
            cols = col_clusters == c
            block_sq[i, j] = B2[np.ix_(rows, cols)].sum()
    block_norm = np.sqrt(block_sq)

    E_total = float(B2.sum())
    out = {
        "block_norm": block_norm,
        "block_sq": block_sq,
        "unique_row": unique_r,
        "unique_col": unique_c,
        "E_total": E_total,
    }
    same_clustering = (
        len(unique_r) == len(unique_c)
        and np.array_equal(unique_r, unique_c)
        and len(row_clusters) == len(col_clusters)
        and np.array_equal(row_clusters, col_clusters)
    )
    if same_clustering:
        E_within = float(np.diag(block_sq).sum())
        out["E_within"] = E_within
        out["E_within_frac"] = E_within / E_total if E_total > 0 else float("nan")
        out["E_between"] = E_total - E_within
        out["E_between_frac"] = 1.0 - out["E_within_frac"]
    return out


def permutation_null_block(
    B: np.ndarray,
    clusters: np.ndarray,
    n_permutations: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Null distribution of E_within/E_total under random cluster reassignment.

    Both row and col cluster labels are permuted with the *same* permutation
    of glomerulus indices (since they're physically the same gloms in paired
    data). This preserves cluster sizes and tests whether the OBSERVED cluster
    identities — not just any 7-way partition — concentrate B's mass on the
    diagonal blocks.
    """
    rng = np.random.default_rng(rng)
    clusters = np.asarray(clusters)
    obs = decompose_block_energy(B, clusters, clusters)
    observed = obs["E_within_frac"]

    null = np.empty(n_permutations)
    for k in range(n_permutations):
        perm = rng.permutation(clusters.size)
        c_shuf = clusters[perm]
        # Without recomputing the full block_norm matrix: the within-block
        # energy under shuffled labels = sum over clusters of |B[c_shuf==r,
        # c_shuf==r]|². Compute directly.
        Eb_within = 0.0
        for r in obs["unique_row"]:
            mask = c_shuf == r
            Eb_within += (B[np.ix_(mask, mask)] ** 2).sum()
        null[k] = Eb_within / obs["E_total"] if obs["E_total"] > 0 else np.nan

    p = float(np.mean(null >= observed)) if not np.isnan(observed) else float("nan")
    return {
        "observed_within_frac": float(observed),
        "null_within_frac": null,
        "p_value": p,
        "n_permutations": n_permutations,
    }


def top_off_diagonal_blocks(
    block_norm: np.ndarray,
    unique_clusters: np.ndarray,
    k: int = 3,
) -> list[tuple[int, int, float]]:
    """Return the top-k off-diagonal cluster pairs by block norm.

    Excludes diagonal (i == j) entries. Treats the matrix as directed
    (preserves which cluster is "from" and which is "to") — for square
    paired data, you can symmetrize after if you prefer.
    """
    K = len(unique_clusters)
    triples = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            triples.append((unique_clusters[i], unique_clusters[j], block_norm[i, j]))
    triples.sort(key=lambda t: t[2], reverse=True)
    return triples[:k]


def reorder_by_clusters(
    B: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort B's rows and cols by cluster id; return (B_reord, order, boundaries).

    boundaries[i] gives the column index where cluster i begins. Useful for
    drawing cluster-boundary lines on the reordered heatmap.
    """
    clusters = np.asarray(clusters)
    order = np.argsort(clusters, kind="stable")
    B_reord = B[np.ix_(order, order)]
    sorted_c = clusters[order]
    boundaries = np.where(np.diff(sorted_c) != 0)[0] + 1
    return B_reord, order, boundaries
