"""
Evaluation utilities for WaveletDiff models.

This module provides a comprehensive evaluation suite with:
- Core Metrics (Tier 1): Standard time-series synthesis benchmarks
- Advanced Metrics (Tier 2): Domain-specific "players" for financial data
- Visualizations: t-SNE, PCA, PDF, candlestick plots
- Reporting: Styled scorecard display

Usage:
    from evaluation import EvaluationRunner, EvaluationConfig
    from evaluation import visualizations as viz
    from evaluation import reporting
    
    config = EvaluationConfig(exclude_volume=True)
    runner = EvaluationRunner(config)
    results = runner.run(real_dollar, synth_dollar, real_reparam, synth_reparam)
    
    reporting.display_scorecard(results['dollar'])
"""

# Main orchestrator
from .runner import EvaluationRunner, EvaluationConfig, EvaluationResult

# Preprocessing utilities
from .preprocessing import prepare_evaluation_data

# Reporting
from .reporting import display_scorecard, generate_summary_scorecard

# Core Metrics (Tier 1)
from .core_metrics import (
    discriminative_score,
    predictive_utility,
    context_fid,
    correlation_score,
    dtw_distance,
)

# Advanced Metrics (Tier 2)
from .advanced_metrics import (
    js_divergence,
    acf_similarity,
    alpha_precision,
    beta_recall,
    dcr_score,
    memorization_ratio,
)

# Legacy exports are available via direct import when needed:
# from evaluation.discriminative_metrics import discriminative_score_metrics
# from evaluation.predictive_metrics import predictive_score_metrics
# from evaluation.context_fid import Context_FID
# from evaluation.cross_correlation import CrossCorrelLoss
# from evaluation.dtw import dtw_js_divergence_distance

__all__ = [
    # New API
    'EvaluationRunner',
    'EvaluationConfig',
    'EvaluationResult',
    'prepare_evaluation_data',
    'display_scorecard',
    
    # Core Metrics
    'discriminative_score',
    'predictive_utility',
    'context_fid',
    'correlation_score',
    'dtw_distance',
    
    # Advanced Metrics
    'js_divergence',
    'acf_similarity',
    'alpha_precision',
    'beta_recall',
    'dcr_score',
    'memorization_ratio',
]