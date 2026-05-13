# pls-explorer

PLS analysis of paired low/high-concentration glomerular responses from the
Wachowiak omp8x dataset. Tests whether high-concentration glomerular activity
is predictable from low-concentration activity in a structured way, and
compares the real PLS weight matrix against two domain-specific nulls.

## Question

For X = low-conc responses (odorants × glomeruli) and Y = high-conc responses
(same shape), fit `Y = X B` via PLSR. Compare the structure of B against:

- **Null 1 — homogeneous gain**: `Y_null1[:, j] = alpha_j * X[:, j]`. Expected
  PLS signature: B approximately diagonal.
- **Null 2 — random retuning**: `Y_null2[:, j] = X[:, random_index]`. Expected
  PLS signature: B noisy and unstructured.

If real B differs from both, concentration-dependent broadening is neither
trivial gain scaling nor arbitrary noise — it has interpretable structure.

## Layout

```
pls_explorer/
  config.py        paths, defaults
  data.py          load X, Y, clusters from glom-explorer
  preprocess.py    mean-center, autoscale, pareto, log, row-L2
  nulls.py         homogeneous_gain, random_retuning
  pls.py           NIPALS PLSR pipeline
  cv.py            family-stratified folds, evaluate
  selection.py     one-sigma + Van der Voet ncomp selection
  permutation.py   row-permutation Q² null
  bootstrap.py     bootstrap_B, bootstrap_VIP, Procrustes
  metrics.py       Q², R²X/R²Y, RMSECV, diagonality index, VIP
  baselines.py     ClusterLinearRegression, scaling baselines
  viz.py           plotly figures (B heatmap, outer-product, etc.)
  runner.py        run_variant(config) → results/<variant>/

configs/           yaml per variant
notebooks/         00_data_audit, 01_canonical (grand notebook), 02_compare
results/           gitignored; shared across worktrees
```

## Worktrees

Variants are explored in parallel via `git worktree`:

```
~/code/pls-explorer                 main / consolidation
~/code/pls-explorer-baseline        variant/baseline
~/code/pls-explorer-row-norm        variant/row-normalized
~/code/pls-explorer-autoscale       variant/autoscaled
~/code/pls-explorer-pareto          variant/pareto
~/code/pls-explorer-sparse          variant/sparse-pls
```

Each variant writes `results/<variant>/{real,null1,null2}/` with identical
file layout, so the consolidation notebook on main can read across variants
without rerunning.

## Quick start

```bash
cd ~/code/pls-explorer
/opt/anaconda3/envs/castrolab-dev/bin/pip install -e .
/opt/anaconda3/envs/castrolab-dev/bin/python -m pls_explorer.runner --config configs/baseline.yaml
```
