"""Load X (low) and Y (high) glomerular response matrices via glom-explorer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import glom_explorer as gx


@dataclass
class PairedData:
    """Paired low/high glomerular response matrices, aligned by odorant.

    Attributes
    ----------
    X, Y : DataFrame
        Shape (n_odorants, n_glomeruli). Columns share a 6-level MultiIndex
        (gloms, subject, parent_cluster, subcluster, x, y).
    odorants : DataFrame
        Odorant metadata (Name, SMILES, Group, Group_num) indexed by row.
    glom_clusters : ndarray
        Parent cluster id per glomerulus (length n_glomeruli).
    odorant_clusters : ndarray
        NMF odorant cluster id per odorant (length n_odorants).
    chemical_group : ndarray
        Chemical-class label per odorant (length n_odorants), e.g. 'A', 'B', ...
    """
    X: pd.DataFrame
    Y: pd.DataFrame
    odorants: pd.DataFrame
    glom_clusters: np.ndarray
    odorant_clusters: np.ndarray
    chemical_group: np.ndarray


def load_paired(variant: str = "nonresponders_dropped") -> PairedData:
    """Load paired (low, high) clustered glomerular data + metadata.

    Parameters
    ----------
    variant : str
        Key in glom_explorer's NMF_VARIANTS registry. Default 'nonresponders_dropped'.
    """
    low, high = gx.load_clustered_variant(variant)
    odorants = gx.load_odorants()

    glom_clusters = low.columns.get_level_values("parent_cluster").astype(int).to_numpy()
    odorant_clusters = low.index.get_level_values("odorant_cluster").astype(int).to_numpy()
    odorant_idx = low.index.get_level_values("odorant").astype(int).to_numpy()
    chemical_group = odorants["Group"].iloc[odorant_idx].to_numpy()

    return PairedData(
        X=low, Y=high,
        odorants=odorants,
        glom_clusters=glom_clusters,
        odorant_clusters=odorant_clusters,
        chemical_group=chemical_group,
    )


def pool_small_groups(chemical_group: np.ndarray, min_size: int = 3) -> np.ndarray:
    """Pool chemical groups with < min_size members into 'other' so stratified
    folds remain feasible. Returns a pooled label array (object dtype)."""
    s = pd.Series(chemical_group)
    counts = s.value_counts()
    keep = counts[counts >= min_size].index
    return s.where(s.isin(keep), other="other").to_numpy()
