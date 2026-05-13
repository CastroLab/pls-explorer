"""Plotly visualizations for the PLS analysis pipeline.

Organized per the vault Resources doc 'gold-standard visualization suite':

  Phase 1 — Selection & validation
      plot_rmsecv_components       R²/Q² vs ncomp + overfit ratio
      plot_permutation_null        Q² null histogram + observed

  Phase 2 — PLSC exploratory
      plot_score_score_scatter     LV score-score scatter
      plot_singular_value_scree    % covariance + perm p-values
      plot_bootstrap_ratio_bars    BR bar plot, ±2 thresholded

  Phase 3 — PLSR predictive
      plot_predicted_vs_actual     per-glom facet grid
      plot_per_glom_q2_heatmap     Q² along glomerulus axis
      plot_residuals_vs_fitted     diagnostic
      plot_residuals_vs_leverage   outlier flagging

  Phase 4 — Coefficients
      plot_B_heatmap               B [n_x × n_y], diverging, BR overlay
      plot_W_3way                  real | null1 | null2 side-by-side  ← deliverable
      plot_vip_bars                VIP sorted, colored by B sign

  Phase 5 — Olfactory-specific
      plot_paired_activation       X and Y heatmaps, same odor dendrogram
      plot_outer_product_marginals PLS_play3-style component viz, plotly port
      plot_worst_predicted         low + high profiles, top-decile residuals
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


QUALITATIVE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# =============================================================================
# Phase 1 — Selection & validation
# =============================================================================

def plot_rmsecv_components(
    grid: Sequence[int],
    q2: Sequence[float],
    rmsecv: Sequence[float],
    rmsec: Sequence[float],
    chosen_ncomp: Optional[int] = None,
    title: str = "Component selection",
) -> go.Figure:
    """Twin-axis: Q² (left) and RMSECV/RMSEC overfit ratio (right) vs ncomp.

    Vertical dashed line at the chosen ncomp; horizontal at Q²=0 and at
    overfit ratio 2 (rule of thumb upper bound).
    """
    grid = list(grid)
    rmsecv = np.asarray(rmsecv, float)
    rmsec = np.asarray(rmsec, float)
    overfit = rmsecv / np.where(rmsec > 0, rmsec, np.nan)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=grid, y=q2, mode="lines+markers", name="Q² (CV)",
                   line=dict(color=QUALITATIVE_COLORS[0])),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=grid, y=overfit, mode="lines+markers",
                   name="RMSECV / RMSEC",
                   line=dict(color=QUALITATIVE_COLORS[3], dash="dot")),
        secondary_y=True,
    )
    fig.add_hline(y=0, line=dict(color="gray", width=0.5),
                  secondary_y=False)
    fig.add_hline(y=2, line=dict(color="gray", width=0.5, dash="dot"),
                  secondary_y=True)
    if chosen_ncomp is not None:
        fig.add_vline(x=chosen_ncomp, line=dict(color="black", dash="dash"))
        fig.add_annotation(x=chosen_ncomp, y=max(q2), text=f"n={chosen_ncomp}",
                           showarrow=False, yshift=10)

    fig.update_xaxes(title="n_components", tickmode="linear")
    fig.update_yaxes(title="Q²", secondary_y=False)
    fig.update_yaxes(title="RMSECV / RMSEC", secondary_y=True)
    fig.update_layout(title=title, template="plotly_white", width=720, height=420)
    return fig


def plot_permutation_null(
    q2_null: np.ndarray,
    q2_observed: float,
    p_value: float,
    title: str = "Permutation null distribution",
) -> go.Figure:
    """Histogram of permuted Q² + vertical line at observed Q²."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=q2_null, nbinsx=40, name="Null (Y permuted)",
        marker=dict(color=QUALITATIVE_COLORS[7], opacity=0.7),
    ))
    fig.add_vline(x=q2_observed, line=dict(color=QUALITATIVE_COLORS[3], width=2.5))
    fig.add_annotation(
        x=q2_observed, y=1, xref="x", yref="paper",
        text=f"observed Q² = {q2_observed:.3f}<br>p = {p_value:.3f}",
        showarrow=False, yanchor="top", xanchor="left",
        bgcolor="rgba(255,255,255,0.85)", bordercolor="black",
    )
    fig.update_layout(
        title=title, xaxis_title="Q²", yaxis_title="count",
        template="plotly_white", width=640, height=400,
    )
    return fig


# =============================================================================
# Phase 4 — Coefficients (B matrix is the headline figure)
# =============================================================================

def plot_B_heatmap(
    B: np.ndarray,
    x_glom_clusters: Optional[np.ndarray] = None,
    y_glom_clusters: Optional[np.ndarray] = None,
    bootstrap_ratio: Optional[np.ndarray] = None,
    br_threshold: float = 2.0,
    title: str = "B matrix (low → high)",
) -> go.Figure:
    """Diverging heatmap of B [low-glom × high-glom].

    If bootstrap_ratio is provided, cells with |BR| > br_threshold are marked
    with a small dot overlay (sparse) or a transparent stipple (dense).
    """
    B = np.asarray(B, float)
    vmax = float(np.percentile(np.abs(B), 99))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=B, colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="B", thickness=12),
        name="B",
    ))

    if bootstrap_ratio is not None:
        BR = np.asarray(bootstrap_ratio, float)
        sig_y, sig_x = np.where(np.abs(BR) > br_threshold)
        if sig_y.size and sig_y.size < 5000:  # avoid drowning the figure
            fig.add_trace(go.Scatter(
                x=sig_x, y=sig_y, mode="markers",
                marker=dict(size=2, color="black", symbol="circle"),
                name=f"|BR| > {br_threshold}",
                showlegend=True,
            ))

    fig.update_layout(
        title=title,
        xaxis_title="high-conc glomerulus index",
        yaxis_title="low-conc glomerulus index",
        template="plotly_white",
        width=720, height=620,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_W_3way(
    B_real: np.ndarray,
    B_null1: np.ndarray,
    B_null2: np.ndarray,
    diagonality: Optional[dict] = None,
    title: str = "W comparison: real | Null-1 | Null-2",
) -> go.Figure:
    """Side-by-side B heatmaps for the three fits. Shared colorscale."""
    Bs = [np.asarray(B_real, float), np.asarray(B_null1, float), np.asarray(B_null2, float)]
    vmax = max(float(np.percentile(np.abs(B), 99)) for B in Bs)

    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=[
            f"real ({diagonality['real']:.2f})" if diagonality else "real",
            f"Null-1 mult ({diagonality['null1']:.2f})" if diagonality else "Null-1 multiplicative",
            f"Null-2 random ({diagonality['null2']:.2f})" if diagonality else "Null-2 random retuning",
        ],
        horizontal_spacing=0.05,
    )
    for i, B in enumerate(Bs, start=1):
        fig.add_trace(
            go.Heatmap(
                z=B, colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
                showscale=(i == 3),
                colorbar=dict(title="B", thickness=12) if i == 3 else None,
            ),
            row=1, col=i,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=i)
        fig.update_xaxes(title="high-conc glom", row=1, col=i)
    fig.update_yaxes(title="low-conc glom", row=1, col=1)
    fig.update_layout(
        title=title + (" — subtitle: diagonality index" if diagonality else ""),
        template="plotly_white",
        width=1500, height=540,
    )
    return fig


def plot_vip_bars(
    vip: np.ndarray,
    B_sign: Optional[np.ndarray] = None,
    vip_ci: Optional[tuple[np.ndarray, np.ndarray]] = None,
    threshold: float = 1.0,
    top_n: Optional[int] = 40,
    title: str = "VIP scores (low-conc glomeruli)",
) -> go.Figure:
    """Horizontal bar plot of VIP, sorted desc. Optionally colored by sign of
    the mean B coefficient for each X-feature and overlaid with bootstrap CIs.
    """
    vip = np.asarray(vip, float)
    order = np.argsort(-vip)
    if top_n is not None:
        order = order[:top_n]
    y = np.arange(len(order))
    x = vip[order]

    if B_sign is not None:
        signs = np.sign(np.asarray(B_sign)[order])
        colors = [QUALITATIVE_COLORS[0] if s >= 0 else QUALITATIVE_COLORS[3] for s in signs]
    else:
        colors = QUALITATIVE_COLORS[0]

    fig = go.Figure()
    error_x = None
    if vip_ci is not None:
        lo, hi = vip_ci
        lo = np.asarray(lo, float)[order]
        hi = np.asarray(hi, float)[order]
        error_x = dict(type="data", symmetric=False, array=hi - x, arrayminus=x - lo)

    fig.add_trace(go.Bar(
        x=x, y=y, orientation="h",
        marker=dict(color=colors),
        error_x=error_x,
        hovertemplate="glom %{customdata}<br>VIP %{x:.3f}<extra></extra>",
        customdata=order,
    ))
    fig.add_vline(x=threshold, line=dict(color="black", dash="dash"))
    fig.update_layout(
        title=title,
        xaxis_title="VIP",
        yaxis=dict(autorange="reversed", title="glom rank"),
        template="plotly_white",
        width=620, height=max(360, 16 * len(order)),
    )
    return fig


# =============================================================================
# Phase 3 — Predictive diagnostics
# =============================================================================

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    glom_ids: Optional[Sequence[int]] = None,
    n_panels: int = 12,
    title: str = "Predicted vs actual (CV)",
) -> go.Figure:
    """Faceted panels of CV-predicted vs observed per glomerulus."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n_glom = y_true.shape[1]
    if glom_ids is None:
        # Pick glomeruli with highest variance in y_true to make the panel useful
        order = np.argsort(-y_true.var(axis=0))[:n_panels]
    else:
        order = np.asarray(glom_ids)[:n_panels]
    cols = 4
    rows = int(np.ceil(len(order) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"glom {j}" for j in order])
    for k, j in enumerate(order):
        r, c = k // cols + 1, k % cols + 1
        fig.add_trace(
            go.Scatter(x=y_true[:, j], y=y_pred[:, j], mode="markers",
                       marker=dict(size=5, opacity=0.6, color=QUALITATIVE_COLORS[0]),
                       showlegend=False),
            row=r, col=c,
        )
        lo = float(min(y_true[:, j].min(), y_pred[:, j].min()))
        hi = float(max(y_true[:, j].max(), y_pred[:, j].max()))
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                       line=dict(color="black", dash="dash"), showlegend=False),
            row=r, col=c,
        )
    fig.update_layout(title=title, template="plotly_white",
                      width=900, height=220 * rows)
    return fig


def plot_per_glom_q2_heatmap(
    q2: np.ndarray,
    glom_clusters: Optional[np.ndarray] = None,
    title: str = "Per-glomerulus Q²",
) -> go.Figure:
    """Strip heatmap of Q² across glomeruli, ordered by cluster id if given.

    A second strip below shows the cluster id as categorical bands so you can
    see at a glance which clusters contain the well- and poorly-predicted
    glomeruli.
    """
    q2 = np.asarray(q2, float)
    if glom_clusters is not None:
        order = np.argsort(glom_clusters)
        q2 = q2[order]
        clusters_ordered = np.asarray(glom_clusters)[order].astype(float)
    else:
        clusters_ordered = np.zeros_like(q2)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.04,
    )
    # Main strip: Q² per glomerulus. Explicit y coords + range so the
    # 1-row heatmap has nonzero height in VS Code's notebook renderer.
    fig.add_trace(
        go.Heatmap(
            z=q2.reshape(1, -1),
            x=np.arange(q2.size), y=[0], dy=1,
            colorscale="Viridis", zmin=-0.2, zmax=1.0,
            colorbar=dict(title="Q²", thickness=12, len=0.78, y=0.62),
            hovertemplate="glom %{x}<br>Q² %{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Cluster strip
    fig.add_trace(
        go.Heatmap(
            z=clusters_ordered.reshape(1, -1),
            x=np.arange(q2.size), y=[0], dy=1,
            colorscale=[[i / 9, c] for i, c in enumerate(QUALITATIVE_COLORS[:10])],
            zmin=0, zmax=max(9, float(clusters_ordered.max())),
            showscale=False,
            hovertemplate="glom %{x}<br>cluster %{z:.0f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_yaxes(visible=False, range=[-0.5, 0.5], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.5, 0.5], row=2, col=1,
                     title="cluster")
    fig.update_xaxes(title="glomerulus (sorted by cluster)", row=2, col=1)
    fig.update_layout(
        title=title, template="plotly_white",
        width=960, height=320, margin=dict(t=60, b=50, l=40, r=40),
    )
    return fig


def plot_residuals_vs_fitted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals vs fitted",
) -> go.Figure:
    """Diagnostic: standardized residuals vs fitted values."""
    res = (np.asarray(y_true, float) - np.asarray(y_pred, float)).ravel()
    fitted = np.asarray(y_pred, float).ravel()
    std = res.std() if res.std() > 0 else 1.0
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=fitted, y=res / std, mode="markers",
        marker=dict(size=3, opacity=0.35, color=QUALITATIVE_COLORS[0]),
        showlegend=False,
    ))
    fig.add_hline(y=0, line=dict(color="black", width=0.5))
    fig.update_layout(
        title=title, xaxis_title="fitted Ŷ",
        yaxis_title="standardized residual",
        template="plotly_white", width=680, height=420,
    )
    return fig


# =============================================================================
# Phase 5 — Olfactory-specific
# =============================================================================

def plot_paired_activation(
    X: np.ndarray,
    Y: np.ndarray,
    odorant_labels: Optional[Sequence[str]] = None,
    glom_clusters: Optional[np.ndarray] = None,
    title: str = "Low (X) vs high (Y) glomerular activation",
) -> go.Figure:
    """Side-by-side heatmaps of X and Y with shared row order."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["low", "high"],
                        shared_yaxes=True, horizontal_spacing=0.05)
    vmax = float(np.percentile(np.abs(np.concatenate([X.ravel(), Y.ravel()])), 99))
    fig.add_trace(
        go.Heatmap(z=X, colorscale="Viridis", zmin=0, zmax=vmax, showscale=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=Y, colorscale="Viridis", zmin=0, zmax=vmax,
                   colorbar=dict(title="ΔF/F", thickness=12)),
        row=1, col=2,
    )
    fig.update_xaxes(title="glom", row=1, col=1)
    fig.update_xaxes(title="glom", row=1, col=2)
    fig.update_yaxes(title="odorant", row=1, col=1)
    if odorant_labels is not None:
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(odorant_labels))),
                         ticktext=list(odorant_labels), row=1, col=1)
    fig.update_layout(title=title, template="plotly_white", width=1100, height=620)
    return fig


def plot_outer_product_marginals(
    pls_model,
    component: int = 0,
    x_group_labels: Optional[Sequence] = None,
    y_group_labels: Optional[Sequence] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Plotly port of PLS_play3's `plot_pls_outer_product_with_marginals`.

    Layout (4-cell grid):
        top-left   : empty
        top-right  : Y-loadings skyline (colored by y_group_labels)
        bot-left   : X-loadings skyline, rotated (colored by x_group_labels)
        bot-right  : outer product np.outer(x_loadings, y_loadings)
    """
    x_loadings = pls_model.x_loadings_[:, component]
    y_loadings = pls_model.y_loadings_[:, component]
    outer = np.outer(x_loadings, y_loadings)
    vmax = float(np.percentile(np.abs(outer), 99))

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.22, 0.78],
        column_widths=[0.22, 0.78],
        vertical_spacing=0.04, horizontal_spacing=0.04,
        shared_xaxes=False, shared_yaxes=False,
    )

    # Top: Y-loadings skyline
    yidx = np.arange(len(y_loadings))
    if y_group_labels is not None:
        groups = np.asarray(y_group_labels)
        for k, g in enumerate(np.unique(groups)):
            mask = groups == g
            fig.add_trace(
                go.Bar(x=yidx[mask], y=y_loadings[mask],
                       marker=dict(color=QUALITATIVE_COLORS[k % 10]),
                       name=str(g), showlegend=False),
                row=1, col=2,
            )
    else:
        fig.add_trace(
            go.Bar(x=yidx, y=y_loadings,
                   marker=dict(color=QUALITATIVE_COLORS[0]),
                   showlegend=False),
            row=1, col=2,
        )

    # Left: X-loadings skyline (horizontal bars)
    xidx = np.arange(len(x_loadings))
    if x_group_labels is not None:
        groups = np.asarray(x_group_labels)
        for k, g in enumerate(np.unique(groups)):
            mask = groups == g
            fig.add_trace(
                go.Bar(y=xidx[mask], x=x_loadings[mask], orientation="h",
                       marker=dict(color=QUALITATIVE_COLORS[k % 10]),
                       name=str(g), showlegend=False),
                row=2, col=1,
            )
    else:
        fig.add_trace(
            go.Bar(y=xidx, x=x_loadings, orientation="h",
                   marker=dict(color=QUALITATIVE_COLORS[0]),
                   showlegend=False),
            row=2, col=1,
        )

    # Center: outer product heatmap
    fig.add_trace(
        go.Heatmap(z=outer, colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
                   showscale=True, colorbar=dict(title="x⊗y", thickness=10)),
        row=2, col=2,
    )
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=1)

    fig.update_layout(
        title=title or f"Outer-product loadings, component {component + 1}",
        template="plotly_white",
        width=900, height=820,
        showlegend=False,
        barmode="overlay",
    )
    return fig


def plot_worst_predicted(
    X: np.ndarray,
    Y: np.ndarray,
    Y_pred: np.ndarray,
    odorant_names: Sequence[str],
    n_worst: int = 6,
    title: str = "Worst-predicted odorants (CV residual)",
) -> go.Figure:
    """For the top-residual odorants, plot low (X) and high (Y, Y_pred) profiles."""
    res = np.linalg.norm(Y - Y_pred, axis=1)
    order = np.argsort(-res)[:n_worst]
    fig = make_subplots(rows=n_worst, cols=1,
                        subplot_titles=[f"{odorant_names[i]} (resid {res[i]:.2f})" for i in order])
    for k, i in enumerate(order):
        row = k + 1
        fig.add_trace(go.Scatter(y=X[i], mode="lines",
                                 line=dict(color=QUALITATIVE_COLORS[0]),
                                 name="low" if k == 0 else None,
                                 showlegend=(k == 0)),
                      row=row, col=1)
        fig.add_trace(go.Scatter(y=Y[i], mode="lines",
                                 line=dict(color=QUALITATIVE_COLORS[2]),
                                 name="high" if k == 0 else None,
                                 showlegend=(k == 0)),
                      row=row, col=1)
        fig.add_trace(go.Scatter(y=Y_pred[i], mode="lines",
                                 line=dict(color=QUALITATIVE_COLORS[3], dash="dot"),
                                 name="Ŷ" if k == 0 else None,
                                 showlegend=(k == 0)),
                      row=row, col=1)
    fig.update_layout(title=title, template="plotly_white",
                      width=900, height=180 * n_worst)
    return fig


# =============================================================================
# Phase 2 — PLSC exploratory (used when run_plsc=True in config)
# =============================================================================

def plot_score_score_scatter(
    plsc: dict,
    component: int = 0,
    color_by: Optional[Sequence] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """L_X[:, c] vs L_Y[:, c] scatter. Color by odorant class if provided."""
    Lx = plsc["L_X"][:, component]
    Ly = plsc["L_Y"][:, component]
    if color_by is not None:
        groups = np.asarray(color_by)
        fig = go.Figure()
        for k, g in enumerate(np.unique(groups)):
            mask = groups == g
            fig.add_trace(go.Scatter(
                x=Lx[mask], y=Ly[mask], mode="markers",
                marker=dict(size=8, color=QUALITATIVE_COLORS[k % 10]),
                name=str(g),
            ))
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Lx, y=Ly, mode="markers",
                                 marker=dict(size=8, color=QUALITATIVE_COLORS[0]),
                                 showlegend=False))
    fig.update_layout(
        title=title or f"PLSC scores, LV {component + 1}",
        xaxis_title=f"L_X[:, {component + 1}]",
        yaxis_title=f"L_Y[:, {component + 1}]",
        template="plotly_white",
        width=560, height=480,
    )
    return fig


def plot_singular_value_scree(
    singular_values: np.ndarray,
    p_values: Optional[np.ndarray] = None,
    title: str = "Singular value scree",
) -> go.Figure:
    """% covariance explained per LV. Annotates p-values above bars."""
    sv = np.asarray(singular_values, float)
    pct = sv ** 2 / np.sum(sv ** 2) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(sv) + 1)), y=pct,
                         marker=dict(color=QUALITATIVE_COLORS[0])))
    if p_values is not None:
        for i, p in enumerate(p_values):
            fig.add_annotation(x=i + 1, y=pct[i],
                               text=f"p={p:.3f}", showarrow=False, yshift=8)
    fig.update_layout(
        title=title, xaxis_title="LV", yaxis_title="% covariance explained",
        template="plotly_white", width=640, height=400,
    )
    return fig


def plot_bootstrap_ratio_bars(
    saliences: np.ndarray,
    bootstrap_se: np.ndarray,
    threshold: float = 2.0,
    title: str = "Bootstrap ratios",
    top_n: Optional[int] = 60,
) -> go.Figure:
    """Bar plot of salience/bootstrap_SE colored by red/blue/gray over ±threshold."""
    sal = np.asarray(saliences, float)
    se = np.asarray(bootstrap_se, float)
    BR = np.where(se > 0, sal / se, 0.0)
    order = np.argsort(-np.abs(BR))
    if top_n is not None:
        order = order[:top_n]
    BR = BR[order]
    colors = ["#1f77b4" if v > threshold else "#d62728" if v < -threshold else "#bbbbbb"
              for v in BR]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(order))), y=BR,
                         marker=dict(color=colors), showlegend=False))
    fig.add_hline(y=threshold, line=dict(color="black", dash="dash"))
    fig.add_hline(y=-threshold, line=dict(color="black", dash="dash"))
    fig.update_layout(
        title=title, xaxis_title="rank", yaxis_title="bootstrap ratio",
        template="plotly_white", width=900, height=360,
    )
    return fig
