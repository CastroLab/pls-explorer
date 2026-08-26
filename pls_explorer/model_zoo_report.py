"""Build 03_model_zoo.ipynb: horse race of regression methods against the
PLS-R Q²=0.27 baseline.

Run with:
    python -m pls_explorer.model_zoo_report

Writes notebooks/03_model_zoo.ipynb (executed in place). All models share
the same family-stratified 4-fold CV and the same log-transform preprocessing
as the canonical baseline variant. Each model gets out-of-fold predictions;
Q² is computed identically to the baseline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import nbformat as nbf

from .config import REPO_ROOT


def _md(src: str):
    return nbf.v4.new_markdown_cell(src)


def _code(src: str):
    return nbf.v4.new_code_cell(src)


SETUP = """
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook_connected'

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV,
    MultiTaskLasso, MultiTaskElasticNet,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pls_explorer.data import load_paired, pool_small_groups
from pls_explorer.cv import make_folds
from pls_explorer.preprocess import log1p_positive
from pls_explorer.metrics import q2_global, q2_per_column, rmse
"""


LOAD_DATA = """
data = load_paired('nonresponders_dropped')
X_raw = np.asarray(data.X, dtype=float)
Y_raw = np.asarray(data.Y, dtype=float)

# Same preprocessing as canonical baseline: log(clip(x, 0) + 1)
X = log1p_positive(X_raw)
Y = log1p_positive(Y_raw)

# Family-stratified 4-fold CV (matches baseline; chemical groups with <3
# members pooled into 'other' so stratification is feasible).
group_pooled = pool_small_groups(data.chemical_group, min_size=3)
folds = make_folds(
    n_samples=X.shape[0],
    scheme='family_stratified',
    n_splits=4,
    chemical_group=group_pooled,
    random_state=42,
)

print(f'X shape: {X.shape}   Y shape: {Y.shape}')
print(f'CV folds (family-stratified, k=4):')
for i, (tr, te) in enumerate(folds):
    print(f'  fold {i}: {len(tr):>3d} train, {len(te):>3d} test')
"""


# Each model is a callable `fit_predict(X_tr, Y_tr, X_te) -> Y_hat_te`.
# We define them as closures so the zoo dict stays compact and readable.
ZOO_DEFINITIONS = """
def _wrap_sklearn(estimator, scale=False, scale_std=False):
    '''Return a fit_predict callable wrapping a sklearn estimator.
    scale=True adds StandardScaler(with_mean=True, with_std=scale_std).'''
    def fit_predict(X_tr, Y_tr, X_te):
        if scale:
            sc = StandardScaler(with_mean=True, with_std=scale_std)
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
        else:
            X_tr_s, X_te_s = X_tr, X_te
        # Mean-center Y (matching baseline pipeline)
        y_mean = Y_tr.mean(axis=0, keepdims=True)
        est = estimator
        # clone to avoid state leak across folds
        from sklearn.base import clone
        est = clone(estimator)
        est.fit(X_tr_s, Y_tr - y_mean)
        return est.predict(X_te_s) + y_mean
    return fit_predict


# --- Trivial baselines ---------------------------------------------------
def fp_mean(X_tr, Y_tr, X_te):
    return np.broadcast_to(Y_tr.mean(axis=0, keepdims=True), (X_te.shape[0], Y_tr.shape[1])).copy()

def fp_identity(X_tr, Y_tr, X_te):
    # Y_hat = X (in log space): the "no broadening" null.
    return X_te.copy()


# --- Latent / rank-restricted -------------------------------------------
def fp_plsr(n_components):
    def fp(X_tr, Y_tr, X_te):
        x_mean = X_tr.mean(axis=0, keepdims=True)
        y_mean = Y_tr.mean(axis=0, keepdims=True)
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_tr - x_mean, Y_tr - y_mean)
        return pls.predict(X_te - x_mean) + y_mean
    return fp

def fp_pcr(n_components):
    '''PCA-then-OLS on the PCA scores.'''
    def fp(X_tr, Y_tr, X_te):
        x_mean = X_tr.mean(axis=0, keepdims=True)
        y_mean = Y_tr.mean(axis=0, keepdims=True)
        pca = PCA(n_components=n_components, svd_solver='full')
        Z_tr = pca.fit_transform(X_tr - x_mean)
        Z_te = pca.transform(X_te - x_mean)
        beta = np.linalg.lstsq(Z_tr, Y_tr - y_mean, rcond=None)[0]
        return Z_te @ beta + y_mean
    return fp

def fp_rrr(rank):
    '''Reduced-rank regression via SVD of the OLS coefficient matrix.'''
    def fp(X_tr, Y_tr, X_te):
        x_mean = X_tr.mean(axis=0, keepdims=True)
        y_mean = Y_tr.mean(axis=0, keepdims=True)
        Xc = X_tr - x_mean
        Yc = Y_tr - y_mean
        # OLS coefficient (min-norm if Xc is rank-deficient)
        B_full = np.linalg.lstsq(Xc, Yc, rcond=None)[0]   # shape (p, q)
        # Truncate to rank-r via SVD of the implied fitted Y_hat = Xc B_full
        # Equivalent: SVD on (Xc B_full) and project Y onto top-r left singular vecs
        # Simpler closed form: take SVD of B_full and keep top-r singular dirs.
        U, s, Vt = np.linalg.svd(B_full, full_matrices=False)
        B_rrr = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
        return (X_te - x_mean) @ B_rrr + y_mean
    return fp


ZOO = {
    # name: (fit_predict, short description)
    'MEAN':              (fp_mean,                                'predict column-mean Y'),
    'IDENTITY':          (fp_identity,                            'Y_hat = X (no broadening)'),

    'OLS (min-norm)':    (_wrap_sklearn(LinearRegression()),                                                'least squares, rank-deficient'),
    'Ridge α=1':         (_wrap_sklearn(Ridge(alpha=1.0)),                                                  'L2, α=1'),
    'RidgeCV':           (_wrap_sklearn(RidgeCV(alphas=np.logspace(-3, 4, 15))),                            'L2, α tuned LOO'),
    'Lasso α=0.01':      (_wrap_sklearn(MultiTaskLasso(alpha=0.01, max_iter=10000)),                       'joint L1 sparsity'),
    'ElasticNet':        (_wrap_sklearn(MultiTaskElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),    'L1+L2 joint'),

    'PLS-R k=8':         (fp_plsr(8),                             'PLS-R, 8 latent comps (baseline)'),
    'PLS-R k=15':        (fp_plsr(15),                            'PLS-R, 15 latent comps (over-budget)'),
    'PCR k=8':           (fp_pcr(8),                              'PCA→OLS, 8 PCs'),
    'RRR k=8':           (fp_rrr(8),                              'reduced-rank OLS, rank 8'),

    'KernelRidge RBF':   (_wrap_sklearn(KernelRidge(kernel='rbf', alpha=1.0), scale=True, scale_std=True), 'RBF kernel, α=1, γ=default'),
    'KernelRidge Poly2': (_wrap_sklearn(KernelRidge(kernel='polynomial', degree=2, alpha=1.0), scale=True, scale_std=True), 'degree-2 poly kernel'),

    'RandomForest':      (_wrap_sklearn(RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)),       'RF, 500 trees, multi-output'),
    'ExtraTrees':        (_wrap_sklearn(ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=42)),         'ExtraTrees, 500'),
    'HistGB per-output': (_wrap_sklearn(MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42), n_jobs=-1)), 'GBM per output, 100 iter'),

    'MLP (64)':          (_wrap_sklearn(MLPRegressor(hidden_layer_sizes=(64,), alpha=1.0, max_iter=3000,
                                                     early_stopping=True, validation_fraction=0.25,
                                                     random_state=42), scale=True, scale_std=True),
                          '1 hidden layer, 64 units, strong L2'),
    'MLP (128, 64)':     (_wrap_sklearn(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1.0, max_iter=3000,
                                                     early_stopping=True, validation_fraction=0.25,
                                                     random_state=42), scale=True, scale_std=True),
                          '2 hidden layers, strong L2'),

    'kNN k=5':           (_wrap_sklearn(KNeighborsRegressor(n_neighbors=5), scale=True, scale_std=True),   'k=5 neighbors in X'),
    'kNN k=3':           (_wrap_sklearn(KNeighborsRegressor(n_neighbors=3), scale=True, scale_std=True),   'k=3 neighbors in X'),
}

print(f'Zoo size: {len(ZOO)} models')
for name, (_, descr) in ZOO.items():
    print(f'  {name:24s}  {descr}')
"""


RUN_HORSE_RACE = """
def run_one(fit_predict, X, Y, folds):
    Y_hat = np.full_like(Y, np.nan, dtype=float)
    for tr, te in folds:
        Y_hat[te] = fit_predict(X[tr], Y[tr], X[te])
    return Y_hat

results = []
for name, (fp, descr) in ZOO.items():
    t0 = time.time()
    try:
        Y_hat = run_one(fp, X, Y, folds)
        q2 = q2_global(Y, Y_hat)
        rms = rmse(Y, Y_hat)
        per_glom = q2_per_column(Y, Y_hat)
        # frac of glomeruli with positive per-glom Q²
        frac_pos = float(np.mean(per_glom > 0))
        elapsed = time.time() - t0
        results.append({
            'model': name, 'description': descr,
            'Q²': q2, 'RMSE': rms,
            'frac gloms Q²>0': frac_pos,
            'wall (s)': elapsed,
            'status': 'ok',
        })
        print(f'  {name:24s}  Q²={q2:+.4f}  RMSE={rms:.3f}  pos-gloms={frac_pos:.2f}  ({elapsed:.1f}s)')
    except Exception as e:
        elapsed = time.time() - t0
        results.append({
            'model': name, 'description': descr,
            'Q²': np.nan, 'RMSE': np.nan,
            'frac gloms Q²>0': np.nan,
            'wall (s)': elapsed,
            'status': f'FAIL: {type(e).__name__}: {e}',
        })
        print(f'  {name:24s}  FAILED  ({type(e).__name__}: {e})')

results_df = pd.DataFrame(results)
"""


RESULTS_TABLE = """
# Sort by Q² descending, NaN/failures at bottom
ranked = results_df.sort_values('Q²', ascending=False, na_position='last').reset_index(drop=True)
ranked.style.format({'Q²': '{:+.4f}', 'RMSE': '{:.3f}', 'frac gloms Q²>0': '{:.2f}', 'wall (s)': '{:.1f}'})
"""


PLOT_BAR = """
plot_df = ranked.dropna(subset=['Q²']).copy()
# Colors: PLS-R baseline highlighted, beat-baseline in green, below in gray
baseline_q2 = float(plot_df.loc[plot_df['model'] == 'PLS-R k=8', 'Q²'].iloc[0])
colors = []
for q in plot_df['Q²']:
    if q > baseline_q2 + 1e-6:   colors.append('#2ca02c')   # green: beats baseline
    elif q > baseline_q2 - 1e-6: colors.append('#d62728')   # red: equals baseline (PLS-R itself)
    else:                         colors.append('#888')     # gray: below
fig = go.Figure()
fig.add_trace(go.Bar(
    y=plot_df['model'][::-1],
    x=plot_df['Q²'][::-1],
    orientation='h',
    marker_color=colors[::-1],
    text=[f'{q:+.3f}' for q in plot_df['Q²'][::-1]],
    textposition='outside',
))
fig.add_vline(x=baseline_q2, line_dash='dash', line_color='#d62728',
              annotation_text=f'PLS-R baseline ({baseline_q2:+.3f})',
              annotation_position='top right')
fig.add_vline(x=0, line_color='#aaa', line_width=1)
fig.update_layout(
    title='Held-out Q² across the model zoo (family-stratified 4-fold CV)',
    xaxis_title='Q² (global, on log-transformed Y)',
    yaxis_title='',
    height=max(400, 32 * len(plot_df) + 100),
    margin=dict(l=180, r=80, t=80, b=60),
    showlegend=False,
)
fig.show()
"""


INTERPRETATION = """
print('=== headline ===')
top = ranked.iloc[0]
print(f'Best model: {top["model"]}   Q²={top["Q²"]:+.4f}')
baseline = ranked.loc[ranked['model'] == 'PLS-R k=8'].iloc[0]
print(f'Baseline PLS-R k=8: Q²={baseline["Q²"]:+.4f}')
gap = top['Q²'] - baseline['Q²']
print(f'Gap over baseline: {gap:+.4f}  ({100*gap/abs(baseline["Q²"]):+.1f}% relative)')

print()
print('=== beat-baseline summary ===')
beat = ranked[ranked['Q²'] > baseline['Q²'] + 1e-4]
print(f'{len(beat)} of {len(ranked)} models beat PLS-R k=8 by ≥ 0.0001 Q²:')
for _, row in beat.iterrows():
    print(f'  {row["model"]:24s}  Q²={row["Q²"]:+.4f}  Δ={row["Q²"]-baseline["Q²"]:+.4f}')

print()
print('=== sanity floors ===')
for name in ['MEAN', 'IDENTITY', 'OLS (min-norm)']:
    if name in ranked['model'].values:
        row = ranked[ranked['model'] == name].iloc[0]
        print(f'  {name:24s}  Q²={row["Q²"]:+.4f}')
"""


PREAMBLE = """\
# Model zoo: who can beat PLS-R Q²=0.27?

A horse race across ~20 regression methods on the same paired low/high-conc
glomerular data, the same family-stratified 4-fold CV, and the same
log-transform preprocessing as the canonical baseline. The headline number
to beat is **PLS-R k=8, Q² = 0.272**.

**Why a horse race here.** PLS-R is the chemometrics workhorse for exactly
this regime (small n, large p, multi-output), but it's not the only option
and there's a fair question about whether the 0.27 ceiling is "PLS being
bad" or "the data being noisy / nonlinear-where-it-counts." A sweep of
linear-regularized methods (Ridge, Lasso, ElasticNet, PCR, RRR), kernel
methods (KernelRidge with RBF and polynomial kernels), tree ensembles
(RF, ExtraTrees, HistGB), neural nets (MLP), and local methods (k-NN)
plus trivial floors (mean, identity) lets us see how much room there is
above PLS-R, and whether any single competitor finds the room.

**Setup**

- X: low-conc glomerular activity, 59 odorants × 376 glomeruli.
- Y: high-conc glomerular activity, same shape and same row alignment.
- Preprocessing: `log(max(0, raw) + 1)` on X and Y; per-fold mean-centering inside each model's pipeline.
- CV: family-stratified k=4 over chemical-class labels, classes with <3 members pooled into 'other'.
- Metric: Q² (global, multi-output) on out-of-fold predictions, computed on the log-transformed Y space (identical to the baseline analysis).

Out-of-fold predictions are accumulated, then a single Q² is computed
across the full matrix — same convention as the baseline.
"""


SECTION_DATA = """\
## Data + folds
"""

SECTION_ZOO = """\
## The model zoo

Each entry is a `(fit_predict_callable, description)` pair. The callable
takes `(X_train, Y_train, X_test)` and returns `Y_hat_test`. Mean-centering
is done inside the wrapper for consistency with the PLS baseline.
"""

SECTION_RACE = """\
## Run the race

Out-of-fold predictions for every model; Q² accumulated globally.
"""

SECTION_RESULTS = """\
## Ranked results
"""

SECTION_PLOT = """\
## Bar chart — Q² across the zoo
"""

SECTION_INTERP = """\
## Interpretation
"""


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(PREAMBLE),
        _code(SETUP.strip()),
        _md(SECTION_DATA),
        _code(LOAD_DATA.strip()),
        _md(SECTION_ZOO),
        _code(ZOO_DEFINITIONS.strip()),
        _md(SECTION_RACE),
        _code(RUN_HORSE_RACE.strip()),
        _md(SECTION_RESULTS),
        _code(RESULTS_TABLE.strip()),
        _md(SECTION_PLOT),
        _code(PLOT_BAR.strip()),
        _md(SECTION_INTERP),
        _code(INTERPRETATION.strip()),
    ]
    nb.metadata.update({
        "kernelspec": {
            "display_name": "Python 3 (castrolab-dev)",
            "language": "python",
            "name": "python3",
        },
    })
    return nb


def main() -> None:
    nb = build_notebook()
    out = REPO_ROOT / "notebooks" / "03_model_zoo.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {out}; executing…")
    subprocess.run(
        [
            "/opt/anaconda3/envs/castrolab-dev/bin/jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=1800",
            str(out),
        ],
        check=True,
    )
    print(f"done: {out}")


if __name__ == "__main__":
    main()
