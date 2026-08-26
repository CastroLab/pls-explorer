# subset_mix null sweep

Sweep K in {5, 10, 25, 50, 100} for the new `subset_mix` null: each
glomerulus j's high-conc column is replaced by the mean of K randomly
chosen *other* low-conc columns (sampled without replacement, j excluded).

Pipeline: `baseline.yaml` (log1p → mean-center → PLS, n_components=8,
family-stratified 4-fold CV, relu_clip=True). Clusters are the 7 NMF
parent clusters carried on the column MultiIndex of the loaded data.

## Reference values (canonical baseline fit)

- **Real W**: Q² = 0.272, diagonality = 0.018, within-cluster energy ≈ 0.40
- **Null-1 (homogeneous gain)**: diagonality = 0.049
- **Null-2 (single random retuning)**: diagonality = 0.003

## Sweep results

| K | Q² | diagonality | within-cluster energy | qualitative W |
|---|-----|------------|----------------------|---------------|
| 5 | +0.296 | 0.0032 | 0.162 | no diagonal, diffuse off-diagonal mass, substantial predictivity |
| 10 | +0.346 | 0.0027 | 0.164 | no diagonal, diffuse off-diagonal mass, substantial predictivity |
| 25 | +0.448 | 0.0022 | 0.133 | no diagonal, diffuse off-diagonal mass, substantial predictivity |
| 50 | +0.534 | 0.0022 | 0.139 | no diagonal, diffuse off-diagonal mass, substantial predictivity |
| 100 | +0.631 | 0.0022 | 0.145 | no diagonal, diffuse off-diagonal mass, substantial predictivity |

## Interpretation

As K grows the donor average becomes an increasingly stable, near-population-mean
target, so the regression problem becomes *easier*, not harder: Q² rises
monotonically from 0.30 at K=5 to 0.63 at K=100, well above the real fit's
Q² ≈ 0.27. Crucially, *none* of this predictive gain comes through diagonal
structure — diagonality stays pinned at 0.002–0.003 across the whole sweep,
essentially identical to Null-2's 0.003 and an order of magnitude below the
real W's 0.018 (and ~20× below Null-1's 0.049). Within-cluster energy hovers
around 0.13–0.16 — roughly what a random off-diagonal mass would give for 7
clusters covering ~14% of all (i,j) pairs — vs. the real W's strongly
cluster-enriched ~0.40. Structurally, subset-mix at every K is the same
animal as Null-2 (random, non-cluster-aligned, no diagonal); it just makes
the response smoother and therefore more linearly compressible as K grows.
The K most distinct from the real W is the largest one (K=100, high Q² but
zero structural overlap); none of the K values *resemble* the real W —
subset-mix lacks both the diagonal and the cluster-block signatures that
distinguish the real fit.
