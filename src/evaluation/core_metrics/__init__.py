"""
Core Metrics (Tier 1) - Standard time-series synthesis benchmarks.

These metrics come from TimeGAN and Diffusion-TS literature.
"""

from .discriminative import discriminative_score
from .predictive import predictive_utility
from .context_fid import context_fid
from .correlation import correlation_score
from .dtw import dtw_distance
from .stylized_facts import kurtosis_score, volatility_clustering_score

__all__ = [
    'discriminative_score',
    'predictive_utility',
    'context_fid',
    'correlation_score',
    'dtw_distance',
    'kurtosis_score',
    'volatility_clustering_score',
]
