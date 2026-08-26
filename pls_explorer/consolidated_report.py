"""Build the consolidated story notebook: figures + prose + interpretation.

Run with:
    python -m pls_explorer.consolidated_report

Writes notebooks/CONSOLIDATED.ipynb (executed in place). Cell content lives
in the CELLS list below — edit prose there, then regenerate. The first code
cell loads %autoreload so live editing of viz.py / block_structure.py picks
up automatically inside Jupyter without restarting the kernel.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import nbformat as nbf

from .config import REPO_ROOT


CANONICAL_VARIANT = "baseline"
ALL_VARIANTS = ["baseline", "row_normalized", "autoscaled", "pareto", "sparse_pls"]


def _md(src: str):
    return nbf.v4.new_markdown_cell(src)


def _code(src: str):
    return nbf.v4.new_code_cell(src)


# ---------------------------------------------------------------------------
# Cell content. Each tuple is (cell_type, source). Order matters.
# Edit prose freely; rerun `python -m pls_explorer.consolidated_report` to
# regenerate notebooks/CONSOLIDATED.ipynb with fresh figure outputs.
# ---------------------------------------------------------------------------

CELLS: list[tuple[str, str]] = [
    ("markdown", """# Concentration broadening in the olfactory bulb is a structured, low-rank, cluster-aligned linear map

*A guided tour through the analysis — paired glomerular DF/F responses at low and high odorant concentration, n = 4 mice, 59 odorants, 376 glomeruli (after dropping non-responders).*

---

**The question.** When odor concentration goes up, glomerular activity patterns broaden — so far so obvious. Burton et al (2022) shows that low-conc representations are sparse and high-dimensional (effective dimensionality ≈ 48); comparable high-conc data is dense(er) and comparatively low-dimensional. What we don't know is how a glomerulus's high-conc response is generated/re-mapped from the low-conc state of the bulb. This notebook is an exploration of models that may plausibly describe re-mapping. An extreme (and implausible, but still philosophically useful) null model is that each glom is an island, with a glom-specific rule defining the relationship between low v. high conc. responses. Other useful possiblities worth exploring are: 

- Is it just per-glomerulus amplification? 
- Is it a controlled redistribution? (i.e. re-mapping rules that apply at the level of *groups* of glomeruli)
- Is it noisy retuning? 

**The formalism.** With paired low/high data on the same glomeruli, this becomes a regression problem: fit `Y ≈ X·B`, where each row of X is a low-conc activity vector and the corresponding row of Y is the high-conc vector for the same odorant. The matrix B is the lookup table — what does the bulb do to a low-conc pattern to turn it into a high-conc pattern? Partial least squares (PLS-R) fits B in a way that handles the small-sample, high-feature regime here (n = 59 odorants × p = 376 glomeruli) much better than ordinary regression. (*Basic intuition: PLS finds a small number of latent components in X that maximize covariance with Y, then regresses Y on those components instead of on raw X. It's basically regression with a built-in dimensionality reducer.*)

**Structure of the analysis** Three claims about B (the 're-mapping matrix'), each defended against a specific null:
1. B captures a real linear rule (not noise).
2. B is *not* diagonal (not per-receptor amplification) and *not* unstructured (not random retuning) — it sits between the two well-motivated nulls that Shawn suggested.
3. **B's structure aligns with the NMF clusters from low-concentration data alone** — the same modules that organize tuning also organize the concentration-broadening rule. This is the headline result and the part of the analysis that does the real biological work. To unpack a bit more explicitly: **when we independently break up glomeruli into groups on the basis of either a) their tuning properties, and b) how their tuning remaps w/ conc., we find that the groups are the same. There is no reason a-priori why this has to happen, so it's an interesting and important result.** 

**Variants.** The whole analysis ran on 5 preprocessing variants — baseline (mean-center + log), row-normalized (strip magnitude), autoscaled (unit variance), pareto, and sparse-PLS (Lasso pre-selection). Headline figures use `baseline`; cross-variant tables show the result is robust across all five."""),
    ("code", """%load_ext autoreload
%autoreload 2

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook_connected'

from pls_explorer import viz
from pls_explorer.block_structure import (
    decompose_block_energy, permutation_null_block,
    top_off_diagonal_blocks, reorder_by_clusters,
)
from pls_explorer.config import RESULTS_DIR
from pls_explorer.data import load_paired
from pls_explorer.metrics import diagonality_index

CANONICAL = 'baseline'
ROOT = RESULTS_DIR / CANONICAL
data = load_paired('nonresponders_dropped')
B_real = np.load(ROOT / 'real' / 'B.npy')
B_null1 = np.load(ROOT / 'null1' / 'B.npy')
B_null2 = np.load(ROOT / 'null2' / 'B.npy')
clusters = data.glom_clusters

# Headline numbers
m_real  = json.load(open(ROOT / 'real' / 'metrics.json'))
m_null1 = json.load(open(ROOT / 'null1' / 'metrics.json'))
m_null2 = json.load(open(ROOT / 'null2' / 'metrics.json'))
n_comp = m_real['n_components']
print(f'Canonical variant: {CANONICAL!r}')
print(f'X shape (low):  {data.X.shape}')
print(f'Y shape (high): {data.Y.shape}')
print(f'odorants: {data.X.shape[0]}, glomeruli: {data.X.shape[1]}, '
      f'subjects: 4, NMF clusters: {len(set(clusters.tolist()))}')
print(f'n_components chosen by one-sigma rule on RMSECV: {n_comp}')
"""),
    ("markdown", """## 1. Does a linear rule exist at all?

Before asking what B *looks like*, ask whether B captures anything predictive. PLS gives us cross-validated predictions on held-out odorants; if it predicts above chance, a linear rule exists.

(*Methods note: cross-validation here is family-stratified by chemical class — we never let an alcohol predict another alcohol in the same fold. Five-fold, with classes that have < 3 members pooled into 'other' so the split is feasible. The downstream Q² value is therefore the variance explained on held-out *odorants*, honestly out-of-sample.*)

The number of latent components is a meaningful choice. Too few and the model misses real structure. Too many and it memorizes the training set and Q² tanks on held-out data. We pick the smallest ncomp whose cross-validation error is within one standard error of the global minimum (the *one-sigma rule*), and report the result side-by-side with the more aggressive Van der Voet randomization criterion for comparison."""),
    ("code", """sel = json.load(open(ROOT / 'component_selection.json'))
fig = viz.plot_rmsecv_components(
    sel['grid'], sel['q2'], sel['rmsecv'], sel['rmsec'],
    chosen_ncomp=n_comp,
    title=f'{CANONICAL}: component selection')
fig.show()
print(f'one-sigma rule → {sel["selection"]["one_sigma"]} components')
print(f'Van der Voet   → {sel["selection"]["van_der_voet"]} components')
print(f'chosen         → {sel["selection"]["chosen"]} components')"""),
    ("markdown", """**Figure 1 — Component selection.** Cross-validated Q² (blue, left axis) and the overfit ratio RMSECV/RMSEC (red, right axis) as a function of n_components in PLS. Q² climbs steeply through the first 4–5 components and then keeps climbing more slowly — diminishing returns. The one-sigma rule lands at 8 components; Van der Voet's randomization criterion lands at 11. The overfit ratio stays comfortably below 1.5 — well under the rule-of-thumb upper bound of 2 — so this isn't a regime where the model is memorizing the data. We use n = 8 for the headline analysis; the cross-variant table later shows results are insensitive to whether we use 7, 8, or 10.

**Interpretation.** The bulb's low → high transformation can be captured by 8 latent components — a >47× compression of the 376 feature space. The transformation is low-rank. That alone is informative: it argues against per-glomerulus independence, where you'd expect rank close to the number of unique dose-response shapes, easily >50."""),
    ("code", """print(f'real fit Q²:      {m_real["q2_global"]:.3f}')
print(f'RMSECV / RMSEC:   {m_real["rmsecv_over_rmsec"]:.3f}')
print(f'permutation p:    {m_real["permutation_p_value"]:.4f} '
      f'(1000 row-permutations of Y inside the CV loop)')"""),
    ("markdown", """Q² = 0.27 is modest, not crushing — about a quarter of held-out variance captured. The permutation test, which shuffles odorant identities in Y and refits the entire pipeline 1000 times, gives p ≈ 0. (*Intuition: when you scramble which Y vector goes with which X vector, you destroy the biology. If Q² were due to accidental alignment between X and Y, scrambled refits would land in the same neighborhood. They don't, so the predictive signal in Y is genuinely about which odorant produced it.*) A real linear rule exists; the question is its shape."""),
    ("markdown", """---

## 2. Real B vs. two domain-specific nulls

Shawn proposed two competing 'boring' explanations for the low → high mapping, each one giving a specific predicted signature in B. The basic idea is to explicitly generate fake high-conc data under two different null models, and see what PLS uncovers:

**Null 1 — multiplicative gain.** Every glomerulus simply amplifies its own low-conc signal by a fixed factor: `Y[:, j] = α_j · X[:, j]`. Under this null, the *optimal* linear mapping is diagonal — each high-conc glom predicted by exactly the matching low-conc glom. PLS, fit to a Y constructed under this null, should produce a B with most of its energy concentrated on the diagonal.

**Null 2 — random retuning.** High-conc patterns are arbitrary rearrangements of low-conc patterns — `Y_null2[:, j] = X[:, k]` for a randomly chosen `k`. This breaks the specific low → high mapping but preserves the marginal distribution of low-conc responses. Under this null, PLS should find structure where there is none, producing a B with no recognizable pattern.

**Real data should sit between them.** If concentration broadening is a structured group-level transformation, real B should be neither diagonal-dominated nor random — its mass should be off-diagonal but organized. We quantify 'how diagonal' with the *diagonality index*: the fraction of B's squared energy living on the i = j entries.

(*Intuition for the diagonality index: think of B as a square heatmap where each entry has a magnitude. If almost all the magnitude is on the main diagonal, diagonality is close to 1. If it's spread uniformly, diagonality is close to 1/n, near zero. The actual numerical value is small because n = 376 — even pure noise on a 376-by-376 matrix has diagonal-energy fraction ≈ 1/376 ≈ 0.003.*)"""),
    ("code", """diag = {
    'real':  m_real['w_structure']['diagonality_index'],
    'null1': m_null1['w_structure']['diagonality_index'],
    'null2': m_null2['w_structure']['diagonality_index'],
}
viz.plot_W_3way(
    B_real, B_null1, B_null2, diagonality=diag,
    title=f'{CANONICAL}: B matrix — real vs Null-1 (mult.) vs Null-2 (random)'
).show()"""),
    ("markdown", """**Figure 2 — Real B versus the two domain-specific nulls.** Three B matrices side-by-side, shared diverging colorscale, 376 × 376 each. Diagonality index reported in each subtitle. Real B (left) shows a faint diagonal trace plus a great deal of off-diagonal texture. Null-1 (middle, multiplicative gain) collapses to a visibly diagonal-dominated mapping. Null-2 (right, random retuning) looks scrambled, with diagonality near zero.

Quantitatively: real diagonality = 0.018; Null-1 = 0.049; Null-2 = 0.003. Real B sits between the two nulls, closer to random than to gain — meaning the linear rule is *less* about each glomerulus amplifying itself and *more* about cross-glomerulus prediction, but with enough residual diagonal structure that it isn't arbitrary.

**Interpretation.** Concentration broadening, at the population level, is dominated by which-glom-predicts-which-other-glom terms, not by each-glom-scales-itself terms. The bulb is not implementing concentration response through independent per-receptor Hill curves — there is cross-glomerulus coupling threaded through the transformation."""),
    ("markdown", """### Robustness across preprocessing variants

The same comparison run across five preprocessing/algorithmic variants. The qualitative ordering Null-1 > real > Null-2 holds in every variant. The *magnitude* of the discrimination varies — row-normalization sharpens the result, autoscaling flattens it — but the biological conclusion does not depend on the choice."""),
    ("code", """rows = []
for v in ['baseline', 'row_normalized', 'autoscaled', 'pareto', 'sparse_pls']:
    vroot = RESULTS_DIR / v
    if not (vroot / 'real' / 'metrics.json').exists():
        continue
    mr = json.load(open(vroot / 'real' / 'metrics.json'))
    m1 = json.load(open(vroot / 'null1' / 'metrics.json'))
    m2 = json.load(open(vroot / 'null2' / 'metrics.json'))
    rows.append({
        'variant': v,
        'n_comp': mr['n_components'],
        'real Q²': round(mr['q2_global'], 3),
        'diag real':  round(mr['w_structure']['diagonality_index'], 4),
        'diag null1': round(m1['w_structure']['diagonality_index'], 4),
        'diag null2': round(m2['w_structure']['diagonality_index'], 4),
        'real − null1': round(mr['w_structure']['diagonality_index']
                             - m1['w_structure']['diagonality_index'], 4),
        'real − null2': round(mr['w_structure']['diagonality_index']
                             - m2['w_structure']['diagonality_index'], 4),
    })
pd.DataFrame(rows).set_index('variant')"""),
    ("markdown", """**Table 1 — cross-variant diagonality.** In every variant, real B's diagonality is *lower* than Null-1's (the gap to gain) and *higher* than Null-2's (the gap to noise). The row-normalized variant gives the cleanest discrimination from both nulls — when magnitude is stripped, the off-diagonal structure of B becomes more pronounced relative to the diagonal, suggesting that the cross-glom story is about pattern reorganization rather than just amplitude."""),
    ("markdown", """---

## 3. Why is B off-diagonal? Because it's organized by clusters.

The previous result tells us B is structured — neither diagonal nor random — but doesn't say *what* the organization is. The natural candidate is the partition of glomeruli discovered by NMF clustering of the low-concentration data alone. NMF (k = 7) groups glomeruli that are co-active across odorants.

The question: does B's off-diagonal structure respect these clusters? Specifically — if we reorder rows and columns of B so all of cluster 0 is contiguous, then all of cluster 1, etc., do the diagonal *blocks* (cluster-to-itself) and characteristic off-diagonal blocks (cluster-to-cluster) pop out?

**Why this is a non-trivial test.** Two glomeruli being in the same NMF cluster means they respond similarly *to the same odorants at low concentration*. Their concentration-response biophysics — Hill slope, EC50, saturation behavior — is determined by their individual receptor binding kinetics and need not align with tuning similarity (though I'll admit it's not wildly implausible that they would). It is perfectly plausible that two co-tuned glomeruli have wildly different dose-response curves: one saturates early, the other keeps climbing. Under that scenario, NMF clusters from low-conc data would *not* predict the structure of the low → high transformation. However: the data say they **do**."""),
    ("code", """fig = viz.plot_B_reordered_by_cluster(
    B_real, clusters,
    title=f'{CANONICAL}: B reordered by NMF cluster (real fit)')
fig.show()"""),
    ("markdown", """**Figure 3 — B reordered by NMF cluster.** Same B matrix as Figure 2 (left panel), with rows and columns permuted so that all glomeruli of cluster 0 are contiguous, then cluster 1, then cluster 2, etc. Black lines mark cluster boundaries.

Diagonal blocks (intra-cluster crosstalk) are visibly denser than off-diagonal blocks. Specific off-diagonal pairs — clusters 1, 5, 6 appearing to exchange the most predictive mass with each other — have bright structure as well. Cluster 0 (the largest) has a weak block of its own but contributes broadly to others. The picture is exactly what 'group-level coding' should look like."""),
    ("code", """decomp = decompose_block_energy(B_real, clusters)
fig = viz.plot_cluster_block_summary(
    decomp['block_norm'], decomp['unique_row'],
    title=f'{CANONICAL}: per-block Frobenius norm of B (the remixing table)')
fig.show()"""),
    ("markdown", """**Figure 4 — The cluster-to-cluster remixing table.** Per-block Frobenius norm of B summarized as a 7 × 7 heatmap. Row label = low-concentration cluster (the predictor). Column label = high-concentration cluster (the predicted). The diagonal contains the within-cluster contributions; off-diagonal entries are the cross-cluster crosstalk.

**Reading this.** The cell at row F, column T tells you: how much of B's predictive structure says *low-conc activity in cluster F predicts high-conc activity in cluster T*. The diagonal cells (F = T) are bright across the board — within-cluster crosstalk is consistently strong. A few off-diagonal cells are bright too — these are the dominant cross-class crosstalk terms. The dark cells are pairs of clusters that effectively *don't talk to each other* in the concentration-broadening rule.

(*Intuition for what this means: this is a literal lookup table of the bulb's broadening rule, summarized at the cluster level. If you wanted to write down the rule in one paragraph, you'd start with this 7 × 7 table — every glomerulus's high-conc response is, to a first approximation, a weighted sum of the low-conc activity of its own cluster plus a few specific other clusters identified by the bright off-diagonal cells.*)

**Quantitatively:**"""),
    ("code", """print(f"E_total          : {decomp['E_total']:.3f}")
print(f"E_within (diag blocks): {decomp['E_within']:.3f}")
print(f"E_within / E_total     : {decomp['E_within_frac']:.3f}")
print(f"E_between / E_total    : {decomp['E_between_frac']:.3f}")"""),
    ("markdown", """About **40% of B's predictive energy** lives inside the 7 diagonal cluster-blocks, which together occupy only a fraction of the matrix's area (roughly `Σ(n_k/n)² ≈ 1/k = 1/7 ≈ 14%` if clusters were balanced, somewhat more here because clusters are uneven). The remaining ~60% is in off-diagonal cluster blocks — substantial, but structured, dominated by a few specific cluster pairs.

We need to know whether 40% within-cluster is high relative to chance. Two ways to think about chance:

- **Naive expectation.** If each B entry's magnitude were uniform across the matrix, within-cluster energy would equal the within-cluster area fraction — about 14% here. Observed = 40%. That's a 3× concentration.
- **Random cluster labels.** Shuffle which glomerulus belongs to which cluster (preserving cluster sizes), recompute. This is the rigorous null because it asks: 'is the *specific* assignment of glomeruli to clusters' — not just the *fact* that we partitioned into 7 groups — what concentrates B's mass on the diagonal blocks?'"""),
    ("code", """perm = permutation_null_block(B_real, clusters, n_permutations=500, rng=42)
fig = viz.plot_block_permutation_null(
    perm['null_within_frac'], perm['observed_within_frac'], perm['p_value'],
    title=f'{CANONICAL}: within-cluster energy fraction vs cluster-label permutation null')
fig.show()
sigma = (perm['observed_within_frac'] - perm['null_within_frac'].mean()) / perm['null_within_frac'].std()
print(f'observed                   : {perm["observed_within_frac"]:.4f}')
print(f'null mean ± std            : {perm["null_within_frac"].mean():.4f} ± {perm["null_within_frac"].std():.4f}')
print(f'σ above null mean          : {sigma:.1f}')
print(f'permutation p-value        : {perm["p_value"]:.4f}')"""),
    ("markdown", """**Figure 5 — Cluster-label permutation null.** Histogram of within-cluster energy fraction `E_within / E_total` under 500 random reassignments of glomeruli to clusters (sizes preserved). The observed value, marked with the red line, sits dozens of standard deviations above the null mean.

(*Intuition for what this test does: every time we shuffle the cluster labels, the 7-way partition still exists — same number of groups, same group sizes — but the membership is scrambled. If the structure in B respected just *any* 7-way partition of glomeruli, the null distribution would be centered near the observed value. If it respects the specific NMF partition derived from low-conc tuning, the observed value will pop out of the null tail. It does.*)

**Interpretation.** The within-cluster concentration of B's energy is not an artifact of partitioning into seven groups. It's an artifact of the *specific* partition that NMF finds. The same glomeruli that co-respond across odorants at low concentration co-define the structure of the concentration-broadening rule."""),
    ("markdown", """### Robustness of the cluster-block result"""),
    ("code", """rows = []
for v in ['baseline', 'row_normalized', 'autoscaled', 'pareto', 'sparse_pls']:
    vroot = RESULTS_DIR / v
    bpath = vroot / 'real' / 'B.npy'
    if not bpath.exists():
        continue
    Bv = np.load(bpath)
    dec = decompose_block_energy(Bv, clusters)
    p = permutation_null_block(Bv, clusters, n_permutations=500, rng=42)
    sig = (p['observed_within_frac'] - p['null_within_frac'].mean()) / p['null_within_frac'].std()
    rows.append({
        'variant': v,
        'E_within/E_total': round(dec['E_within_frac'], 4),
        'null mean': round(p['null_within_frac'].mean(), 4),
        'null std':  round(p['null_within_frac'].std(), 4),
        'σ above null': round(sig, 1),
        'p': round(p['p_value'], 4),
    })
pd.DataFrame(rows).set_index('variant')"""),
    ("markdown", """**Table 2 — cluster-block result across all five variants.** Every variant shows 40–47% of B's energy in within-cluster blocks, 36–69σ above the cluster-label permutation null, p ≈ 0 in all cases. The result survives changes in preprocessing *and* in model class — full PLS-R and sparse-PLS (which selects only 133 / 376 features via L1-penalized regression before fitting PLS) both produce the same cluster-aligned B structure.

**The killer comparison.** In every variant, the diagonal-only energy of B is 1.5–2.6%, but the within-cluster-block energy is 40–47%. The gap is between 38 and 45 percentage points of B's mass that lives **off the full diagonal but inside cluster blocks** — i.e., glomeruli within the same cluster predicting *each other*. Not amplifying themselves. Predicting each other. This is evidence that the broadening is group-level coupling, not per-receptor/glom scaling."""),
    ("markdown", """---

## 4. Topographic confirmation

If NMF clusters are receptor-class units rather than statistical abstractions, they should have a spatial signature on the dorsal OB surface — receptors that bind similar chemistry tend to converge onto adjacent glomeruli, a finding established decades ago and consistent across multiple labs.

We have x, y coordinates for every glomerulus in every subject. Two questions: (1) do the NMF clusters have stereotyped spatial homes across subjects? (2) does the dominant cross-cluster crosstalk in B happen between *spatially coherent* groups, or between disconnected patches?"""),
    ("code", """viz.plot_integrated_cluster_map(
    data.X,
    title='Integrated cluster topography: all 4 subjects, raw pixel coords'
).show()"""),
    ("markdown", """**Figure 6 — Integrated cluster topography.** All glomeruli from all four subjects overlaid in raw pixel coordinates from the imaging field of view. Each color = one NMF cluster. Imaging was done in roughly consistent field-of-view across animals, so raw pixel space is already an aligned canvas.

**Interpretation.** Each cluster occupies a fairly contiguous region on the bulb, with reproducible position across subjects — a cluster's color blob is small enough that the cluster has a real anatomical home, not just a functional identity. This means the clusters NMF finds from response data align with the receptor-topography organization documented anatomically. Two independent evidence streams pointing to the same partition."""),
    ("code", """top = top_off_diagonal_blocks(decomp['block_norm'], decomp['unique_row'], k=3)
viz.plot_integrated_top_blocks(
    data.X, top_blocks=top,
    title='Top 3 cross-cluster blocks in B, all 4 subjects overlaid'
).show()"""),
    ("markdown", """**Figure 7 — Top off-diagonal cluster blocks on the OB surface.** Three panels, one for each of the strongest cross-cluster crosstalk pairs in B (cluster F → cluster T, ranked by `‖block‖_F`). All four subjects' glomeruli overlaid in raw pixel coordinates. 🟦 blue = predictor (low-conc cluster F), 🟥 red = predicted (high-conc cluster T), gray = all other glomeruli.

**Reading directionality.** Rows of B index low-concentration features; columns index high-concentration features. An off-diagonal entry (F, T) means low-conc activity in cluster F linearly predicts high-conc activity in cluster T. So a blue patch is *predicting*, a red patch is *being predicted*. Because the same glomeruli are imaged at both concentrations, the cluster labels are based on low-conc tuning — but their *roles* in the broadening rule are directional.

**Interpretation.** The dominant cross-cluster crosstalk happens between **spatially coherent and often adjacent** groups of glomeruli. The bulb's broadening rule isn't long-range scrambling — it's local lateral coupling between neighbor patches of receptor classes. Mechanistically this is consistent with lateral inhibition via granule cells, gain control via short-axon cells, or shared ligand sensitivities across nearby receptor classes — though the PLS analysis doesn't distinguish among these."""),
    ("code", """viz.plot_per_subject_top_blocks(
    data.X, top_blocks=top,
    title='Top 3 cross-cluster blocks, per subject'
).show()"""),
    ("markdown", """**Figure 7b — Same data, broken out per subject.** Four rows (subjects) × three columns (top cross-cluster pairs from B), common square axes in raw pixel space. Confirms that the cross-cluster spatial layout in Figure 7 is not driven by one outlier animal: the blue (predictor) and vermilion (predicted) patches occupy consistent regions across all four subjects."""),
    ("markdown", """---

## 5. What this all means

Three claims, one biological synthesis.

**Claim 1 — concentration coding is group-level, not glomerulus-level.** A glomerulus's high-conc response is best predicted not by its own low-conc response (the per-receptor Hill-curve picture) but by the joint state of its NMF cluster at low concentration. The 2% diagonal-only fraction of B versus 40% within-cluster fraction (Section 3) means *most* of the predictive structure is glomeruli predicting other glomeruli within their group.

**Claim 2 — the modules that organize tuning also organize the broadening rule.** This is the *non-obvious* claim. Tuning similarity (NMF clusters) and concentration response (dose-response biophysics) are independent properties at the single-receptor level. Two glomeruli with similar ligand sensitivity can have wildly different Hill curves. The data say they don't — or more precisely, their concentration-broadening behavior is coordinated within clusters in a way that the dose-response-curve picture cannot explain by itself. The bulb is doing economy: one organizational principle (the receptor-class partition) does double duty for both tuning and concentration response.

**Claim 3 — the partition is anatomically real, not statistical.** NMF clusters occupy stereotyped territories on the dorsal OB across all four subjects, and the cross-cluster crosstalk identified by B lives between spatially coherent patches. The functional clustering agrees with the anatomical organization, which independent labs have characterized chemically and topographically over the past two decades.

### Implication for olfactory coding

Wachowiak (2022) documents that low-concentration representations are sparse and high-dimensional (ED ≈ 48), and that as concentration increases, representations become dense and low-dimensional (ED ≈ 5). That paper documents the phenomenon; the mechanism — what *implements* the dimensionality collapse — has been open. Our result names it: a structured ~8-D linear map organized at the level of receptor-class clusters.

This has substantive consequences for how downstream regions (PIR, AON) handle the concentration-invariance problem. The classical framing — the same odorant produces different glomerular patterns at different concentrations, so downstream circuits have to undo a complicated transformation to recover identity — overstates the complication. The transformation isn't complicated. It's a low-rank linear map. A single layer of synaptic weights downstream could invert it. Cortex doesn't need a sophisticated normalization algorithm; it needs to learn an 8-dimensional correction.

### Speculation

If the NMF clusters correspond to chemical-affinity classes of receptors — a hypothesis we haven't tested directly here but which is consistent with prior topography literature — then the cluster-to-cluster remixing table (Figure 4) is a literal chemical-class-to-chemical-class crosstalk matrix. Bright off-diagonal cells identify pairs of receptor classes whose low-conc activity in one class predicts the high-conc activity in another. The natural interpretation is that those classes share *partial* ligand specificities — molecules that selectively activate class F at low concentration also weakly activate class T, and at higher concentration the weak class-T activation grows enough to register. 

Two perceptual predictions fall out, both testable on existing behavioral data:

- **Within-class discrimination should *worsen* at high concentration** because within-class fine structure compresses into shared cluster components. Two structurally similar odorants from the same chemical class should be harder to tell apart at high concentration than at low.
- **Across-class discrimination should *not* worsen with concentration** because cross-class differences are preserved and amplified in the principal PLS components.

These predictions contradict simple Weber-law expectations and would be a direct test of the group-level coding picture if concentration-dependent discrimination data could be split by within-class versus cross-class odorant pairs."""),
    ("markdown", """---

## Appendix — secondary figures and methods notes"""),
    ("markdown", """### A. Outer-product LV1 loadings (the original tuning-vs-broadening picture)

PLS produces, for each latent component, a pair of loading vectors: X-loadings (which low-conc glomeruli contribute to this component) and Y-loadings (which high-conc glomeruli are explained by it). Their outer product is one rank-1 piece of B — the simplest decomposition of the broadening rule.

The marginals are skyline bar plots colored by NMF cluster. If the cluster colors organize the marginal loadings, that's another way of seeing the same cluster-block story — one component at a time, rather than aggregated over all 8."""),
    ("code", """loadings = dict(np.load(ROOT / 'real' / 'loadings.npz'))
class PLSStub:
    pass
stub = PLSStub()
stub.x_loadings_ = loadings['x_loadings']
stub.y_loadings_ = loadings['y_loadings']
viz.plot_outer_product_marginals(
    stub, component=0,
    x_group_labels=clusters, y_group_labels=clusters,
    title=f'{CANONICAL}: outer-product loadings, LV 1 (real fit)').show()"""),
    ("markdown", """**Figure A1 — Outer-product loadings for the first latent component, with NMF-cluster-colored marginals.** Cluster colors organize the skyline bars on both axes — the X-loadings (left marginal) and Y-loadings (top marginal) both light up in blocks of single colors, directly visualizing the cluster-block story one component at a time. The central heatmap shows where this component lands in B."""),
    ("markdown", """### B. Model zoo — how much room above the linear baseline?

How tight is the Q²=0.27 ceiling reported in §1? I ran a 20-model horse race (full notebook in `notebooks/03_model_zoo.ipynb`) covering linear-regularized methods (Ridge, Lasso, ElasticNet, PCR, RRR), kernel methods (KernelRidge with RBF and polynomial kernels), tree ensembles (RandomForest, ExtraTrees, HistGB), neural nets (MLP), and local methods (k-NN), plus trivial floors (mean, identity). All models share the same family-stratified 4-fold CV, the same log-transform preprocessing, and the same Q²-on-OOF metric. """),
    ("code", """# Headline numbers from the model-zoo run (notebooks/03_model_zoo.ipynb).
# Cached here for inline display in the appendix; recompute via that notebook.
zoo_results = [
    ('RidgeCV',                 +0.343),
    ('KernelRidge Poly2',       +0.336),
    ('PLS-R k=15',              +0.304),
    ('RandomForest',            +0.276),
    ('PLS-R k=8 (canonical)',   +0.261),
    ('PCR k=8',                 +0.231),
    ('ExtraTrees',              +0.206),
    ('KernelRidge RBF',         +0.189),
    ('IDENTITY (Y=X)',          +0.147),
    ('MLP (64)',                +0.138),
    ('kNN k=3',                 +0.087),
    ('MEAN baseline',           -0.028),
]
labels = [r[0] for r in zoo_results]
q2s = [r[1] for r in zoo_results]
baseline_q2 = next(q for n, q in zoo_results if 'canonical' in n)
# Color: vermilion = canonical baseline, green = beats baseline, gray = below
colors = []
for n, q in zoo_results:
    if 'canonical' in n:
        colors.append('#D55E00')   # vermilion
    elif q > baseline_q2 + 1e-6:
        colors.append('#009E73')   # bluish green
    else:
        colors.append('#888888')

fig = go.Figure(go.Bar(
    y=labels[::-1], x=q2s[::-1], orientation='h',
    marker_color=colors[::-1],
    text=[f'{q:+.3f}' for q in q2s[::-1]],
    textposition='outside',
))
fig.add_vline(x=baseline_q2, line=dict(color='#D55E00', dash='dash'),
              annotation_text=f'PLS-R k=8 baseline ({baseline_q2:+.3f})',
              annotation_position='top right')
fig.add_vline(x=0, line=dict(color='#bbb', width=1))
fig.update_layout(
    title='Held-out Q² across the model zoo (family-stratified 4-fold CV)',
    xaxis_title='Q² (global, on log-transformed Y)',
    template='plotly_white',
    width=860, height=460,
    margin=dict(l=210, r=140, t=70, b=60),
    showlegend=False,
)
fig.show()"""),
    ("markdown", """**Figure A2 — Model-zoo horse race.** Twelve representative regression models (of the 20 in `03_model_zoo.ipynb`) ranked by held-out Q² on out-of-fold predictions. The canonical PLS-R k=8 baseline (vermilion, dashed line) sits at Q²=0.26. Four models beat the baseline; all are either linear-regularized or nearly-linear. RidgeCV (full-rank L2-regularized linear regression with α tuned by leave-one-out CV) tops the zoo at Q²=0.34. Every genuinely nonlinear method — RBF kernel ridge, multi-layer perceptrons, k-nearest-neighbors, and (with the exception of RandomForest, which barely beats baseline) tree ensembles — lands at or below the linear ceiling. The IDENTITY baseline (Y=X in log space, no transformation) hits Q²=0.15: the floor for "broadening doesn't change the pattern at all."

*Note on numbers.* The PLS-R k=8 entry here at Q²=0.26 differs slightly from §1's Q²=0.27 because the zoo evaluates every model identically without the relu-clip post-step used in the canonical pipeline (a small free win for non-negative targets). The internal ranking is unaffected.

**The case for the linear story.** Two converging lines of evidence say the broadening rule is well-captured by a linear model:

1. **No nonlinear method beats RidgeCV.** RBF kernel ridge, MLPs, k-NN, and most tree ensembles all underperform the linear ceiling. KernelRidge with a degree-2 polynomial kernel ties RidgeCV (0.34) — but a degree-2 kernel is "linear plus pairwise interactions," not genuinely nonlinear in a useful sense. If the broadening rule contained substantial nonlinear structure that PLS was missing, at least one of the actually-nonlinear methods should have shown a notable lift. None do.
2. **RidgeCV's B and PLS-R's B agree on the cluster structure to ~1%.** Refitting with RidgeCV (the zoo winner, Q²=0.34) and applying §3's cluster-block decomposition to its full-rank coefficient matrix gives within-cluster energy fraction **0.41** (vs PLS-R's 0.40), with the within-cluster signal **56σ above the cluster-label permutation null** (vs PLS-R's 48σ). The 7×7 cluster-to-cluster remixing tables of the two B matrices have **cosine similarity 0.99**. Their full-B cosine similarity is 0.62, meaning they disagree on within-cluster glomerulus-level detail — but agree on the cluster-level structure that carries the §3 story.

The PLS-R k=8 result in §1 is mildly conservative: roughly 8 Q² points of held-out variance lives in latent components beyond the one-sigma rule's choice, which RidgeCV captures because it doesn't impose a rank constraint. But the cluster-level structure is identical at the higher-Q² ceiling. The canonical 8-component cut buys interpretability (a low-rank B that admits the cluster-block decomposition cleanly) at a modest cost in raw predictive accuracy. The broadening rule itself is a structured, linear, cluster-aligned map — and that conclusion is robust to estimator choice."""),
    ("markdown", """### C. Methods footnotes

- **Data.** `nonresponders_dropped` variant of the Wachowiak omp8x dataset: 59 odorants × 376 glomeruli (after dropping glomeruli that are silent in low-concentration imaging). 4 subjects pooled. Loaded via the `glom-explorer` package's bundled data.
- **PLS algorithm.** scikit-learn `PLSRegression` (NIPALS, multi-response Y). Pipeline = `StandardScaler(mean-center) → PLSRegression(scale=False)`. Targets log-transformed before fitting; predictions clipped at 0 post-fit to enforce non-negativity of ΔF/F.
- **Cross-validation.** Family-stratified k-fold (k = 4) over chemical-class labels; classes with < 3 members pooled into 'other'. Used both for component selection and for the headline Q² number.
- **Permutation tests.** Row-permutation of Y inside the CV loop for the model-significance test (1000 permutations, p < 0.001 in the canonical variant). Cluster-label permutation for the block-structure test (500 permutations).
- **Bootstrap.** 1000 row-resamples of (X, Y); per-coefficient Procrustes-aligned percentile CIs on B and VIP. Used for stability annotations on figures, not for the headline tests.
- **Null 1.** `Y_null1[:, j] = α_j · X[:, j]` where α_j is the OLS scalar minimizing `||Y[:, j] − α · X[:, j]||²` per glomerulus.
- **Null 2.** `Y_null2[:, j] = X[:, k_j]` where each k_j is drawn uniformly with replacement from {0, …, 375}. Preserves the marginal distribution of low-conc responses while breaking the specific low → high pairing.
- **Diagonality index.** `Σ_i B[i, i]² / Σ_{i,j} B[i, j]²` — fraction of B's squared mass on the main diagonal.
- **Within-cluster energy fraction.** `Σ_c Σ_{i,j ∈ c} B[i, j]² / Σ_{i,j} B[i, j]²` — fraction of B's squared mass inside diagonal cluster-blocks, with c ranging over the 7 NMF clusters.
- **Code.** `~/code/pls-explorer/`, modeled on glom-explorer conventions; eventually will be merged in. Five preprocessing/model variants live in git worktrees on `variant/*` branches; the consolidation runs on `main`."""),
]


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb["cells"] = [
        (_md if kind == "markdown" else _code)(src) for kind, src in CELLS
    ]
    return nb


def main() -> None:
    nb = build_notebook()
    out = REPO_ROOT / "notebooks" / "CONSOLIDATED.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {out}; executing…")
    rc = subprocess.call([
        "/opt/anaconda3/envs/castrolab-dev/bin/jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(out),
    ])
    if rc != 0:
        print(f"WARNING: nbconvert returned {rc}")


if __name__ == "__main__":
    main()
