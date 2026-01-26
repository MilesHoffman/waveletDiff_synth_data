"""
Advanced Metrics (Tier 2) - Domain-specific evaluation "players".

Visual Scout: JS Divergence, ACF Similarity
Statistician: Alpha-Precision, Beta-Recall (Full Manifold)
Integrity Officer: DCR, Memorization Ratio
"""

from .visual_scout import js_divergence, acf_similarity
from .statistician import alpha_precision, beta_recall
from .integrity_officer import dcr_score, memorization_ratio

__all__ = [
    'js_divergence',
    'acf_similarity',
    'alpha_precision',
    'beta_recall',
    'dcr_score',
    'memorization_ratio',
]
