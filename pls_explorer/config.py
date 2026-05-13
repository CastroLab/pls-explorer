"""Paths, defaults, and config loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"

ANALYSIS_DEFAULTS = {
    "data_variant": "nonresponders_dropped",
    "log_transform": True,
    "scaling": "mean_center",
    "row_normalize": False,
    "n_components_grid": list(range(1, 16)),
    "cv_scheme": "family_stratified",
    "cv_n_splits": 4,
    "cv_random_state": 42,
    "n_permutations": 1000,
    "n_bootstrap": 1000,
    "relu_clip": True,
    "sparse_pls": False,
    "sparse_lasso_alpha": 0.5,
    "null2_mode": "column_resample",
    "rng_seed": 42,
}


@dataclass
class VariantConfig:
    """One run's analytical choices."""
    name: str
    data_variant: str = "nonresponders_dropped"
    log_transform: bool = True
    scaling: str = "mean_center"
    row_normalize: bool = False
    n_components_grid: list = field(default_factory=lambda: list(range(1, 16)))
    cv_scheme: str = "family_stratified"
    cv_n_splits: int = 4
    cv_random_state: int = 42
    n_permutations: int = 1000
    n_bootstrap: int = 1000
    relu_clip: bool = True
    sparse_pls: bool = False
    sparse_lasso_alpha: float = 0.5
    null2_mode: str = "column_resample"
    rng_seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path | str) -> "VariantConfig":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        merged = {**ANALYSIS_DEFAULTS, **raw}
        merged["name"] = raw.get("name", path.stem)
        valid = {k: v for k, v in merged.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def results_dir(self) -> Path:
        d = RESULTS_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
