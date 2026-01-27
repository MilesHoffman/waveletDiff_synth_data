"""
Legacy Proxy for Predictive Score.
Routes to src.evaluation.core_metrics.predictive
"""
from evaluation.core_metrics.predictive import predictive_utility as predictive_utility_func

def predictive_score_metrics(real, generated, **kwargs):
    result = predictive_utility_func(real, generated, **kwargs)
    if isinstance(result, tuple):
        return result[0] # Return TSTR score only
    return result