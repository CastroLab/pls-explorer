"""Refit RidgeCV (the model_zoo winner, Q²=0.343) on the full data and
re-run the §3 cluster-block analysis on its coefficient matrix. Compare
side-by-side with the canonical PLS-R k=8 baseline.

The load-bearing question: is the cluster-aligned structure of B a property
of the broadening rule, or a PLS-R artifact?

Run with: /opt/anaconda3/envs/castrolab-dev/bin/python scratch/ridge_cluster_blocks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from pls_explorer.block_structure import (
    decompose_block_energy,
    permutation_null_block,
)
from pls_explorer.config import RESULTS_DIR
from pls_explorer.data import load_paired
from pls_explorer.metrics import diagonality_index
from pls_explorer.preprocess import log1p_positive


def fit_ridgecv_and_extract_B(X, Y, alphas):
    """Fit RidgeCV with multi-output Y, return the (p, q) coefficient matrix
    B such that Y_pred = (X - X_mean) @ B + Y_mean.

    sklearn's RidgeCV with fit_intercept=True handles centering internally;
    `coef_` has shape (q, p) and represents the mapping from centered X to
    centered Y. We transpose to (p, q) to match PLS-R's extract_B convention.
    """
    rcv = RidgeCV(alphas=alphas)
    rcv.fit(X, Y)
    return rcv.coef_.T, rcv.alpha_


def summarize(B, clusters, name, n_perm=500, rng=42):
    diag = diagonality_index(B)
    dec = decompose_block_energy(B, clusters)
    perm = permutation_null_block(B, clusters, n_permutations=n_perm, rng=rng)
    sigma = (
        (perm["observed_within_frac"] - perm["null_within_frac"].mean())
        / perm["null_within_frac"].std()
    )
    return {
        "name": name,
        "shape": B.shape,
        "||B||_F²": float((B ** 2).sum()),
        "diagonality": float(diag),
        "E_within/E_total": float(dec["E_within_frac"]),
        "E_between/E_total": float(dec["E_between_frac"]),
        "perm null mean": float(perm["null_within_frac"].mean()),
        "perm null std": float(perm["null_within_frac"].std()),
        "σ above null": float(sigma),
        "perm p": float(perm["p_value"]),
    }


def main():
    print("=" * 72)
    print("Ridge-vs-PLS cluster-block comparison")
    print("=" * 72)

    data = load_paired("nonresponders_dropped")
    X = log1p_positive(np.asarray(data.X, dtype=float))
    Y = log1p_positive(np.asarray(data.Y, dtype=float))
    clusters = data.glom_clusters
    n_clusters = len(set(clusters.tolist()))
    print(
        f"\nX shape: {X.shape}   Y shape: {Y.shape}   "
        f"clusters: {n_clusters}"
    )

    # --- fit RidgeCV ------------------------------------------------------
    alphas = np.logspace(-3, 4, 15)
    B_ridge, alpha_star = fit_ridgecv_and_extract_B(X, Y, alphas)
    print(f"\nRidgeCV: chosen α = {alpha_star:.4g} (grid: 10^-3 … 10^4, 15 pts)")
    print(f"B_ridge shape: {B_ridge.shape}   ||B||_F²: {(B_ridge ** 2).sum():.3f}")

    # --- load canonical PLS-R k=8 B for side-by-side ----------------------
    B_pls = np.load(RESULTS_DIR / "baseline" / "real" / "B.npy")
    print(f"B_PLS shape:   {B_pls.shape}   ||B||_F²: {(B_pls ** 2).sum():.3f}")

    # --- summarize both ---------------------------------------------------
    res_pls = summarize(B_pls, clusters, name="PLS-R k=8 (canonical)")
    res_ridge = summarize(B_ridge, clusters, name="RidgeCV")

    df = pd.DataFrame([res_pls, res_ridge]).set_index("name")
    print("\n--- side-by-side ---")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df.T.to_string())

    # --- direct similarity between the two Bs -----------------------------
    cos = float(
        (B_pls.flatten() @ B_ridge.flatten())
        / (np.linalg.norm(B_pls) * np.linalg.norm(B_ridge))
    )
    print(f"\ncos(B_PLS, B_Ridge) = {cos:.4f}")

    # cluster-block-level: do the 7x7 block heatmaps agree?
    dec_pls = decompose_block_energy(B_pls, clusters)
    dec_ridge = decompose_block_energy(B_ridge, clusters)
    block_pls = dec_pls["block_norm"]
    block_ridge = dec_ridge["block_norm"]
    block_cos = float(
        (block_pls.flatten() @ block_ridge.flatten())
        / (np.linalg.norm(block_pls) * np.linalg.norm(block_ridge))
    )
    print(
        f"cos(block_norm_PLS, block_norm_Ridge) = {block_cos:.4f}"
        f"  (7x7 cluster-to-cluster remixing tables)"
    )

    # write a small JSON for downstream / notebook consumption
    out = Path(__file__).resolve().parent / "ridge_cluster_blocks.json"
    out.write_text(json.dumps(
        {
            "alpha_star": alpha_star,
            "pls": res_pls,
            "ridge": res_ridge,
            "cos_B": cos,
            "cos_block_norm": block_cos,
        },
        indent=2,
    ))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
