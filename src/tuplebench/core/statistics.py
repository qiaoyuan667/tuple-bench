"""Paired cluster bootstrap, conditional on trained models and saved responses."""
from __future__ import annotations
import numpy as np
from .io import require

def cluster_mean_ci(values: list[float], clusters: list[str], draws: int=2000, seed: int=0, level: float=0.95) -> dict:
    require(len(values) == len(clusters), 'Mismatched cluster data')
    if not values:
        return {'mean': None, 'ci': None, 'n_worlds': 0, 'n_clusters': 0, 'status': 'empty'}
    y = np.array(values, float)
    names, inverse = np.unique(clusters, return_inverse=True)
    n = len(names)
    sums = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    mean = float(y.mean())
    if n < 2:
        return {'mean': mean, 'ci': None, 'n_worlds': len(y), 'n_clusters': n, 'status': 'insufficient_clusters'}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(draws, n))
    bs = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    a = (1 - level) / 2
    ci = [float(x) for x in np.quantile(bs, [a, 1 - a])]
    return {'mean': mean, 'ci': ci, 'n_worlds': len(y), 'n_clusters': n, 'status': 'descriptive_fixed_fit_cluster_bootstrap', 'small_cluster_warning': n < 20}

def paired_difference(left: list[float], right: list[float], clusters: list[str], **kwargs):
    require(len(left) == len(right), 'Paired comparison requires identical populations')
    return cluster_mean_ci((np.asarray(left) - np.asarray(right)).tolist(), clusters, **kwargs)
