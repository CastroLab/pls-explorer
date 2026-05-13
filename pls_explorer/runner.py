"""Orchestrate a full variant run: real + null1 + null2 fits, save artifacts.

Writes the canonical results layout:

    results/<variant>/
        config.yaml
        real/
            metrics.json
            B.npy, B_bootstrap_ci.npz
            VIP.npy, VIP_ci.npz
            loadings.npz
            y_hat_oof.npy
            per_glom_q2.npy
            per_odorant_residuals.npy
            permutation.npz
            figures/
                rmsecv_components.html
                permutation_null.html
                B_heatmap.html
                outer_product_comp0.html
                vip_bars.html
                predicted_vs_actual.html
                per_glom_q2.html
        null1/                              same layout
        null2/                              same layout
        W_3way.html                         3-panel B comparison (deliverable)
        report.md                           auto-generated 1-page summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import bootstrap as boot
from . import permutation as perm
from . import viz
from .baselines import ClusterLinearRegression, PerGlomScaling, SimpleScaling
from .config import VariantConfig, CONFIGS_DIR
from .cv import cross_val_predict_pls, cross_val_score_curve, make_folds
from .data import load_paired, pool_small_groups
from .metrics import (compute_vip, per_odorant_residual, q2_global,
                      q2_per_column, rmse, variance_explained,
                      w_structure_summary)
from .nulls import build_null_targets
from .pls import extract_B, fit_pls, pls_predict
from .preprocess import log1p_positive, row_l2_normalize
from .selection import select_ncomp


def _preprocess_X_Y(X_raw, Y_raw, cfg):
    """Apply log + row normalization per config. Scaling is done inside the
    sklearn pipeline (build_scaler) so X and Y are still in their natural
    units here."""
    X = np.asarray(X_raw, dtype=float)
    Y = np.asarray(Y_raw, dtype=float)
    if cfg.log_transform:
        X = log1p_positive(X)
        Y = log1p_positive(Y)
    if cfg.row_normalize:
        X = row_l2_normalize(X)
        Y = row_l2_normalize(Y)
    return X, Y


def _save_fit(out: Path, label: str, X, Y, cfg, folds, chemical_group,
              n_components: int, run_perm: bool, run_bootstrap: bool):
    """Fit, evaluate, save artifacts for one target (real | null1 | null2)."""
    out_label = out / label
    figs_dir = out_label / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # === out-of-fold predictions at selected n_components ===
    Y_hat = cross_val_predict_pls(
        X, Y, folds, n_components=n_components,
        scaling=cfg.scaling, sparse=cfg.sparse_pls,
        sparse_lasso_alpha=cfg.sparse_lasso_alpha,
        relu_clip=cfg.relu_clip,
    )
    q2_g = q2_global(Y, Y_hat)
    q2_col = q2_per_column(Y, Y_hat)
    resid = per_odorant_residual(Y, Y_hat)
    rmsecv = rmse(Y, Y_hat)

    # === fit on full data for B, loadings, VIP ===
    pipe = fit_pls(X, Y, n_components=n_components, scaling=cfg.scaling,
                   sparse=cfg.sparse_pls,
                   sparse_lasso_alpha=cfg.sparse_lasso_alpha)
    Y_cal = pls_predict(pipe, X, relu_clip=cfg.relu_clip)
    rmsec = rmse(Y, Y_cal)
    B = extract_B(pipe)
    vip = compute_vip(pipe)
    var_exp = variance_explained(pipe)

    # === bootstrap ===
    boot_B = None
    boot_V = None
    if run_bootstrap:
        boot_B = boot.bootstrap_B(
            X, Y, n_components=n_components, n_bootstrap=cfg.n_bootstrap,
            scaling=cfg.scaling, sparse=cfg.sparse_pls,
            sparse_lasso_alpha=cfg.sparse_lasso_alpha, rng=cfg.rng_seed + 1,
        )
        boot_V = boot.bootstrap_VIP(
            X, Y, n_components=n_components, n_bootstrap=cfg.n_bootstrap,
            scaling=cfg.scaling, sparse=cfg.sparse_pls,
            sparse_lasso_alpha=cfg.sparse_lasso_alpha, rng=cfg.rng_seed + 2,
        )

    # === permutation null Q² ===
    perm_res = None
    if run_perm:
        perm_res = perm.permutation_q2_null(
            X, Y, folds, n_components=n_components,
            n_permutations=cfg.n_permutations, scaling=cfg.scaling,
            sparse=cfg.sparse_pls,
            sparse_lasso_alpha=cfg.sparse_lasso_alpha,
            relu_clip=cfg.relu_clip, rng=cfg.rng_seed + 3,
        )

    # === metrics ===
    metrics = {
        "label": label,
        "variant": cfg.name,
        "n_components": int(n_components),
        "n_odorants": int(Y.shape[0]),
        "n_glom": int(Y.shape[1]),
        "q2_global": float(q2_g),
        "rmsecv": float(rmsecv),
        "rmsec": float(rmsec),
        "rmsecv_over_rmsec": float(rmsecv / rmsec) if rmsec > 0 else float("nan"),
        "r2x_cumulative": var_exp["r2x_cumulative"].tolist(),
        "r2y_cumulative": var_exp["r2y_cumulative"].tolist(),
        "w_structure": w_structure_summary(B),
        "config": cfg.__dict__,
    }
    if perm_res is not None:
        metrics["permutation_p_value"] = perm_res["p_value"]
        metrics["permutation_n"] = int(cfg.n_permutations)

    # === save numerical artifacts ===
    np.save(out_label / "B.npy", B)
    np.save(out_label / "VIP.npy", vip)
    np.save(out_label / "y_hat_oof.npy", Y_hat)
    np.save(out_label / "per_glom_q2.npy", q2_col)
    np.save(out_label / "per_odorant_residuals.npy", resid)
    np.savez(out_label / "loadings.npz",
             x_loadings=pipe.named_steps["pls"].x_loadings_,
             y_loadings=pipe.named_steps["pls"].y_loadings_,
             x_weights=pipe.named_steps["pls"].x_weights_,
             x_scores=pipe.named_steps["pls"].x_scores_,
             y_scores=pipe.named_steps["pls"].y_scores_)
    if boot_B is not None:
        np.savez(out_label / "B_bootstrap_ci.npz", **{k: v for k, v in boot_B.items()
                                                       if isinstance(v, np.ndarray)})
        np.savez(out_label / "VIP_ci.npz", **{k: v for k, v in boot_V.items()
                                                if isinstance(v, np.ndarray)})
    if perm_res is not None:
        np.savez(out_label / "permutation.npz",
                 q2_null=perm_res["q2_null"],
                 q2_observed=perm_res["q2_observed"],
                 p_value=perm_res["p_value"])

    with open(out_label / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # === figures (plotly HTML) ===
    BR = boot_B["bootstrap_ratio"] if boot_B is not None else None
    fig = viz.plot_B_heatmap(B, bootstrap_ratio=BR,
                             title=f"{cfg.name} / {label}: B matrix (n_comp={n_components})")
    fig.write_html(str(figs_dir / "B_heatmap.html"), include_plotlyjs="cdn")

    fig = viz.plot_outer_product_marginals(pipe.named_steps["pls"], component=0,
                                           title=f"{cfg.name} / {label}: outer-product LV 1")
    fig.write_html(str(figs_dir / "outer_product_comp0.html"), include_plotlyjs="cdn")

    if perm_res is not None:
        fig = viz.plot_permutation_null(perm_res["q2_null"], perm_res["q2_observed"],
                                        perm_res["p_value"],
                                        title=f"{cfg.name} / {label}: permutation null")
        fig.write_html(str(figs_dir / "permutation_null.html"), include_plotlyjs="cdn")

    B_sign = np.sign(B.sum(axis=1))
    vip_ci = (boot_V["vip_ci_low"], boot_V["vip_ci_high"]) if boot_V is not None else None
    fig = viz.plot_vip_bars(vip, B_sign=B_sign, vip_ci=vip_ci,
                            title=f"{cfg.name} / {label}: VIP")
    fig.write_html(str(figs_dir / "vip_bars.html"), include_plotlyjs="cdn")

    fig = viz.plot_per_glom_q2_heatmap(q2_col, title=f"{cfg.name} / {label}: per-glom Q²")
    fig.write_html(str(figs_dir / "per_glom_q2.html"), include_plotlyjs="cdn")

    return {"metrics": metrics, "B": B, "Y_hat": Y_hat, "pipe": pipe, "perm": perm_res}


def run_variant(cfg: VariantConfig, n_components: int | None = None,
                run_perm: bool = True, run_bootstrap: bool = True,
                verbose: bool = True) -> dict:
    """End-to-end run for a single variant config."""
    out = cfg.results_dir()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    # === load data ===
    log(f"[{cfg.name}] loading data variant={cfg.data_variant}")
    pd_data = load_paired(cfg.data_variant)
    X_raw = pd_data.X.to_numpy(dtype=float)
    Y_raw = pd_data.Y.to_numpy(dtype=float)
    log(f"  X: {X_raw.shape}, Y: {Y_raw.shape}, "
        f"clusters: {len(np.unique(pd_data.glom_clusters))}")

    # === preprocess (outside the sklearn pipeline: log + row-L2) ===
    X, Y = _preprocess_X_Y(X_raw, Y_raw, cfg)

    # === folds (chemical-class stratified by default) ===
    cgroup = pool_small_groups(pd_data.chemical_group, min_size=3)
    folds = make_folds(
        n_samples=X.shape[0],
        scheme=cfg.cv_scheme,
        n_splits=cfg.cv_n_splits,
        chemical_group=cgroup,
        random_state=cfg.cv_random_state,
    )
    log(f"  CV: {cfg.cv_scheme} n_splits={cfg.cv_n_splits} → {len(folds)} folds")

    # === build null targets ===
    targets = build_null_targets(X, Y, null2_mode=cfg.null2_mode, rng=cfg.rng_seed)

    # === component selection (sweep on real targets only) ===
    if n_components is None:
        log("  sweeping n_components on real Y")
        curve = cross_val_score_curve(
            X, targets["real"], folds, n_components_grid=cfg.n_components_grid,
            scaling=cfg.scaling, sparse=cfg.sparse_pls,
            sparse_lasso_alpha=cfg.sparse_lasso_alpha, relu_clip=cfg.relu_clip,
        )
        sel = select_ncomp(
            targets["real"], curve["y_hat_oof"], curve["rmsecv"],
            curve["n_components_grid"], method="one_sigma",
            n_permutations=500, rng=cfg.rng_seed + 4,
        )
        n_components = sel["chosen"]
        log(f"  one_sigma → {sel['one_sigma']}, van_der_voet → {sel['van_der_voet']}, "
            f"chosen={n_components}, agree={sel['agree']}")
        # save the sweep + selection plot
        fig = viz.plot_rmsecv_components(
            curve["n_components_grid"], curve["q2"], curve["rmsecv"], curve["rmsec"],
            chosen_ncomp=n_components,
            title=f"{cfg.name}: component selection",
        )
        fig.write_html(str(out / "component_selection.html"), include_plotlyjs="cdn")
        with open(out / "component_selection.json", "w") as f:
            json.dump({"grid": curve["n_components_grid"],
                       "q2": curve["q2"].tolist(),
                       "rmsecv": curve["rmsecv"].tolist(),
                       "rmsec": curve["rmsec"].tolist(),
                       "selection": sel}, f, indent=2, default=str)

    # === fit each target ===
    results = {}
    for label in ["real", "null1", "null2"]:
        log(f"  fitting {label}")
        results[label] = _save_fit(
            out, label, X, targets[label], cfg, folds, cgroup,
            n_components=n_components, run_perm=run_perm,
            run_bootstrap=run_bootstrap,
        )

    # === deliverable: 3-way W comparison ===
    diagonality = {
        "real": results["real"]["metrics"]["w_structure"]["diagonality_index"],
        "null1": results["null1"]["metrics"]["w_structure"]["diagonality_index"],
        "null2": results["null2"]["metrics"]["w_structure"]["diagonality_index"],
    }
    fig = viz.plot_W_3way(
        results["real"]["B"], results["null1"]["B"], results["null2"]["B"],
        diagonality=diagonality,
        title=f"{cfg.name}: B matrix — real vs Null-1 (mult.) vs Null-2 (random)",
    )
    fig.write_html(str(out / "W_3way.html"), include_plotlyjs="cdn")

    # === outer-product comparison across 3 fits, comp 0 ===
    fig = make_outer_product_3way(
        results, glom_clusters=pd_data.glom_clusters, component=0,
        title=f"{cfg.name}: outer-product loadings LV 1, real vs nulls",
    )
    fig.write_html(str(out / "outer_product_3way_comp0.html"), include_plotlyjs="cdn")

    # === auto-generated report ===
    write_report(out, cfg, results, n_components)

    log(f"[{cfg.name}] done → {out}")
    return {"results": results, "n_components": n_components, "out": out}


def make_outer_product_3way(results, glom_clusters, component=0, title=""):
    """Three side-by-side outer-product heatmaps for real | null1 | null2."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = ["real", "null1", "null2"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=labels,
                        shared_yaxes=True, horizontal_spacing=0.05)
    outers = []
    for label in labels:
        pls = results[label]["pipe"].named_steps["pls"]
        outer = np.outer(pls.x_loadings_[:, component], pls.y_loadings_[:, component])
        outers.append(outer)
    vmax = max(float(np.percentile(np.abs(o), 99)) for o in outers)
    for i, o in enumerate(outers, start=1):
        fig.add_trace(
            go.Heatmap(z=o, colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
                       showscale=(i == 3)),
            row=1, col=i,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    fig.update_layout(title=title, template="plotly_white",
                      width=1500, height=540)
    return fig


def write_report(out: Path, cfg, results, n_components: int):
    """Auto-emit results/<variant>/report.md — 1 page summary."""
    rows = []
    for label in ["real", "null1", "null2"]:
        m = results[label]["metrics"]
        rows.append((label, m["q2_global"], m["rmsecv"], m["rmsecv_over_rmsec"],
                     m["w_structure"]["diagonality_index"],
                     m["w_structure"]["off_diagonal_entropy"],
                     m.get("permutation_p_value", float("nan"))))
    lines = [
        f"# {cfg.name}",
        "",
        f"- data_variant: `{cfg.data_variant}`",
        f"- preprocessing: log={cfg.log_transform}, scaling={cfg.scaling}, row_normalize={cfg.row_normalize}",
        f"- CV: {cfg.cv_scheme} n_splits={cfg.cv_n_splits}",
        f"- n_components: **{n_components}**",
        f"- permutations: {cfg.n_permutations}, bootstrap: {cfg.n_bootstrap}",
        "",
        "## Three-way comparison",
        "",
        "| target | Q² | RMSECV | RMSECV/RMSEC | diagonality | off-diag entropy | perm p |",
        "|---|---|---|---|---|---|---|",
    ]
    for (label, q2, rcv, ratio, diag, ent, p) in rows:
        lines.append(f"| {label} | {q2:.3f} | {rcv:.3f} | {ratio:.3f} | {diag:.3f} | {ent:.3f} | {p:.3f} |")
    lines += [
        "",
        f"![W 3-way](./W_3way.html)",
        f"![Outer product comp 0](./outer_product_3way_comp0.html)",
        "",
        "## Per-target artifacts",
        "",
        "- `<label>/B.npy`, `<label>/VIP.npy`, `<label>/loadings.npz`",
        "- `<label>/figures/{B_heatmap,outer_product_comp0,permutation_null,vip_bars,per_glom_q2}.html`",
    ]
    (out / "report.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True, help="Path to yaml config")
    p.add_argument("--n-components", type=int, default=None,
                   help="Force n_components; skips selection sweep")
    p.add_argument("--skip-perm", action="store_true", help="Skip permutation test")
    p.add_argument("--skip-boot", action="store_true", help="Skip bootstrap")
    p.add_argument("--fast", action="store_true",
                   help="n_permutations=50, n_bootstrap=50 — smoke test mode")
    args = p.parse_args()

    cfg = VariantConfig.from_yaml(args.config)
    if args.fast:
        cfg.n_permutations = 50
        cfg.n_bootstrap = 50
    run_variant(
        cfg, n_components=args.n_components,
        run_perm=not args.skip_perm, run_bootstrap=not args.skip_boot,
    )


if __name__ == "__main__":
    main()
