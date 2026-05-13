"""pls-explorer — Paired-matrix PLS for the Wachowiak omp8x dataset."""

__version__ = "0.1.0"

from .config import VariantConfig, ANALYSIS_DEFAULTS, RESULTS_DIR, CONFIGS_DIR
from .data import PairedData, load_paired, pool_small_groups
from .preprocess import log1p_positive, row_l2_normalize, build_scaler, ParetoScaler
from .nulls import homogeneous_gain_null, random_retuning_null, build_null_targets
from .pls import make_pipeline, fit_pls, extract_B, pls_predict, plsc_svd
from .cv import make_folds, cross_val_predict_pls, cross_val_score_curve
from .metrics import (
    rmse, q2_global, q2_per_column, r2_score_global,
    per_odorant_residual, variance_explained, compute_vip,
    diagonality_index, off_diagonal_entropy, w_structure_summary,
)
from .selection import one_sigma_ncomp, van_der_voet_ncomp, select_ncomp
from .permutation import permutation_q2_null
from .bootstrap import bootstrap_B, bootstrap_VIP
from .baselines import ClusterLinearRegression, PerGlomScaling, SimpleScaling, mean_predictor
from . import viz
from .runner import run_variant
