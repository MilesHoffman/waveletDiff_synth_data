"""
Legacy Proxy for Cross-Correlation Loss.
Routes to src.evaluation.core_metrics.correlation
"""
import torch
from evaluation.core_metrics.correlation import correlation_score

class CrossCorrelLoss:
    """Wrapper class for correlation_score to maintain legacy API."""
    def __init__(self, x, name='CrossCorrelLoss'):
        self.x = x
        self.name = name
    
    def compute(self, y):
        # correlation_score expects numpy arrays
        x_np = self.x.cpu().numpy()
        y_np = y.cpu().numpy()
        score = correlation_score(x_np, y_np)
        return torch.tensor(score)