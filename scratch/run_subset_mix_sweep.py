"""Sweep K in {5, 10, 25, 50, 100} for the subset_mix null and report
Q², diagonality, and within-cluster energy fraction of the fitted W.

Mirrors the canonical baseline pipeline used in runner._save_fit:
  - data_variant=nonresponders_dropped
  - log1p_positive(X, Y)
  - mean-center inside the sklearn pipeline (scaling='mean_center')
  - family-stratified 4-fold CV (cv_random_state=42)
  - relu_clip=True on predictions
  - n_components=8 (baseline's one_sigma-selected value)

Writes:
  scratch/subset_mix_null_report.md   — table + interpretation
  scratch/subset_mix_W.npz             — W matrices keyed K_<value>
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pls_explorer.config import VariantConfig, CONFIGS_DIR
from pls_explorer.cv import cross_val_predict_pls, make_folds
from pls_explorer.data import load_paired, pool_small_groups
from pls_explorer.metrics import diagonality_index, q2_global
from pls_explorer.nulls import subset_mix
from pls_explorer.pls import extract_B, fit_pls
from pls_explorer.preprocess import log1p_positive

K_GRID = [5, 10, 25, 50, 100]
N_COMPONENTS = 8                # baseline's one_sigma pick (see results/baseline/component_selection.json)
RNG_SEED_BASE = 42              # match baseline.yaml rng_seed
OUT_DIR = Path("/Users/jcastro/code/pls-explorer/scratch")


def within_cluster_energy(B: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Σ_c Σ_{i,j in c} B[i,j]² / Σ_{i,j} B[i,j].

    Uses the same cluster labels for X (rows) and Y (columns) — both are the
    full 376-glom NMF cluster vector.
    """
    B2 = np.asarray(B, dtype=float) ** 2
    total = B2.sum()
    if total <= 0:
        return float("nan")
    labels = np.asarray(cluster_labels)
    in_block = 0.0
    for c in np.unique(labels):
        mask = labels == c
        in_block += B2[np.ix_(mask, mask)].sum()
    return float(in_block / total)


def qualitative_description(diag: float, within: float, q2: float) -> str:
    """Heuristic one-line W description."""
    parts = []
    if diag > 0.04:
        parts.append("strong diagonal")
    elif diag > 0.015:
        parts.append("weak diagonal")
    else:
        parts.append("no diagonal")
    if within > 0.35:
        parts.append("block-structured (cluster-aligned)")
    elif within > 0.20:
        parts.append("modest cluster blocks")
    else:
        parts.append("diffuse off-diagonal mass")
    if q2 < 0.05:
        parts.append("near-zero predictive power")
    elif q2 < 0.15:
        parts.append("weak predictivity")
    else:
        parts.append("substantial predictivity")
    return ", ".join(parts)


def main():
    cfg = VariantConfig.from_yaml(str(CONFIGS_DIR / "baseline.yaml"))

    # --- load + preprocess exactly like runner.run_variant ---
    pd_data = load_paired(cfg.data_variant)
    X_raw = pd_data.X.to_numpy(dtype=float)
    Y_raw = pd_data.Y.to_numpy(dtype=float)
    glom_clusters = pd_data.glom_clusters
    print(f"X={X_raw.shape}, Y={Y_raw.shape}, n_clusters={len(np.unique(glom_clusters))}")

    X = log1p_positive(X_raw) if cfg.log_transform else X_raw.copy()
    Y_real = log1p_positive(Y_raw) if cfg.log_transform else Y_raw.copy()

    # --- CV folds (family-stratified, same seed) ---
    cgroup = pool_small_groups(pd_data.chemical_group, min_size=3)
    folds = make_folds(
        n_samples=X.shape[0],
        scheme=cfg.cv_scheme,
        n_splits=cfg.cv_n_splits,
        chemical_group=cgroup,
        random_state=cfg.cv_random_state,
    )

    rows = []
    W_dict = {}
    for K in K_GRID:
        # Distinct rng per K but deterministic
        rng = np.random.default_rng(RNG_SEED_BASE + K)
        Y_null = subset_mix(X, Y_real, rng=rng, K=K)

        # OOF predictions for Q²
        Y_hat = cross_val_predict_pls(
            X, Y_null, folds, n_components=N_COMPONENTS,
            scaling=cfg.scaling, sparse=cfg.sparse_pls,
            sparse_lasso_alpha=cfg.sparse_lasso_alpha,
            relu_clip=cfg.relu_clip,
        )
        q2 = q2_global(Y_null, Y_hat)

        # Full-data fit for B / W
        pipe = fit_pls(X, Y_null, n_components=N_COMPONENTS, scaling=cfg.scaling,
                       sparse=cfg.sparse_pls, sparse_lasso_alpha=cfg.sparse_lasso_alpha)
        B = extract_B(pipe)
        diag = diagonality_index(B)
        within = within_cluster_energy(B, glom_clusters)
        desc = qualitative_description(diag, within, q2)

        print(f"K={K:3d}  Q²={q2:+.4f}  diag={diag:.4f}  within={within:.4f}  -> {desc}")
        rows.append({
            "K": K, "q2": q2, "diagonality": diag,
            "within_cluster_energy": within, "description": desc,
        })
        W_dict[f"K_{K}"] = B

    # --- save W matrices ---
    np.savez(OUT_DIR / "subset_mix_W.npz", **W_dict)
    print(f"Saved W matrices to {OUT_DIR / 'subset_mix_W.npz'}")

    # --- write report markdown ---
    lines = [
        "# subset_mix null sweep",
        "",
        "Sweep K in {5, 10, 25, 50, 100} for the new `subset_mix` null: each",
        "glomerulus j's high-conc column is replaced by the mean of K randomly",
        "chosen *other* low-conc columns (sampled without replacement, j excluded).",
        "",
        "Pipeline: `baseline.yaml` (log1p → mean-center → PLS, n_components=8,",
        "family-stratified 4-fold CV, relu_clip=True). Clusters are the 7 NMF",
        "parent clusters carried on the column MultiIndex of the loaded data.",
        "",
        "## Reference values (canonical baseline fit)",
        "",
        "- **Real W**: Q² = 0.272, diagonality = 0.018, within-cluster energy ≈ 0.40",
        "- **Null-1 (homogeneous gain)**: diagonality = 0.049",
        "- **Null-2 (single random retuning)**: diagonality = 0.003",
        "",
        "## Sweep results",
        "",
        "| K | Q² | diagonality | within-cluster energy | qualitative W |",
        "|---|-----|------------|----------------------|---------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['K']} | {r['q2']:+.3f} | {r['diagonality']:.4f} | "
            f"{r['within_cluster_energy']:.3f} | {r['description']} |"
        )

    # interpretation will be appended after we see numbers
    (OUT_DIR / "subset_mix_null_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote partial table to {OUT_DIR / 'subset_mix_null_report.md'}")

    # also dump raw rows as JSON for downstream tooling
    with open(OUT_DIR / "subset_mix_null_report.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
