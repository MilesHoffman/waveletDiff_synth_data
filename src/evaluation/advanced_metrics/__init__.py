"""
Advanced Metrics (Tier 2) - Domain-specific evaluation "players".

This package implements the Tier 2 metrics:
- Visual Scout: JS Divergence, ACF Similarity
- Statistician: Alpha-Precision, Beta-Recall (Full Manifold)
- Integrity Officer: DCR, Memorization Ratio

It also provides a bridge for legacy function names to maintain backward compatibility.
"""

# New API Imports
from .visual_scout import js_divergence, acf_similarity
from .statistician import alpha_precision, beta_recall
from .integrity_officer import dcr_score, memorization_ratio

# Global exports for the new API
__all__ = [
    'js_divergence',
    'acf_similarity',
    'alpha_precision',
    'beta_recall',
    'dcr_score',
    'memorization_ratio',
    # Legacy names for backward compatibility
    'calculate_distribution_fidelity',
    'calculate_structural_alignment',
    'calculate_financial_reality',
    'calculate_memorization_ratio',
    'calculate_diversity_metrics',
    'calculate_fld',
    'calculate_dcr',
    'calculate_manifold_precision_recall',
    'calculate_mmd'
]

# --- Legacy Bridge Functions ---
# These functions allow old code (like stale wrappers.py) to still function 
# while transitioning to the new architecture.

def calculate_distribution_fidelity(real, synthetic):
    """Bridge for legacy distributions checks."""
    return {
        "JS_Divergence": js_divergence(real, synthetic),
        "Note": "This is a bridge function. Use js_divergence directly."
    }

def calculate_structural_alignment(real, synthetic):
    """Bridge for legacy structural checks (Placeholder)."""
    return {"Note": "PCA/tSNE should be run via visualizations or new metrics."}

def calculate_financial_reality(real, synthetic):
    """Bridge for legacy financial checks."""
    return {
        "ACF_Similarity": acf_similarity(real, synthetic),
        "Note": "This is a bridge function. Use acf_similarity directly."
    }

def calculate_memorization_ratio_legacy(real, synthetic): # Avoid name collision
    return memorization_ratio(real, synthetic)

calculate_memorization_ratio = memorization_ratio

def calculate_diversity_metrics(real, synthetic):
    """Bridge for legacy diversity checks."""
    return {"Coverage": beta_recall(real, synthetic)}

def calculate_fld(real, synthetic):
    """Bridge for legacy FLD (Placeholder)."""
    return 0.0

def calculate_dcr(real, synthetic):
    """Bridge for legacy DCR."""
    return dcr_score(real, synthetic)['mean']

def calculate_manifold_precision_recall(real, synthetic):
    """Bridge for legacy precision/recall."""
    return alpha_precision(real, synthetic), beta_recall(real, synthetic)

def calculate_mmd(real, synthetic):
    """Bridge for legacy MMD (Disabled by user request)."""
    return 0.0
