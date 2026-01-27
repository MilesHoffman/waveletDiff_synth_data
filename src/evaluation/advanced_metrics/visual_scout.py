"""
Visual Scout - JS Divergence and ACF Similarity.

These metrics check if synthetic data "moves" like real data:
- JS Divergence on log-returns distribution
- ACF similarity to verify volatility clustering
"""

import numpy as np
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf
from typing import Optional


def js_divergence(
    real_log_returns: np.ndarray,
    synth_log_returns: np.ndarray,
    n_bins: int = 50
) -> float:
    """
    Compute Jensen-Shannon divergence between log-return distributions.
    
    Args:
        real_log_returns: Real log-returns of shape (N, T-1)
        synth_log_returns: Synthetic log-returns of shape (N, T-1)
        n_bins: Number of bins for histogram
        
    Returns:
        JS divergence (lower is better, 0 = identical distributions)
    """
    # Flatten to 1D
    real_flat = real_log_returns.flatten()
    synth_flat = synth_log_returns.flatten()
    
    # Create shared bins
    all_data = np.concatenate([real_flat, synth_flat])
    bins = np.linspace(
        np.percentile(all_data, 1),
        np.percentile(all_data, 99),
        n_bins + 1
    )
    
    # Compute histograms
    real_hist, _ = np.histogram(real_flat, bins=bins, density=True)
    synth_hist, _ = np.histogram(synth_flat, bins=bins, density=True)
    
    # Normalize and add epsilon
    eps = 1e-10
    real_hist = (real_hist + eps) / (real_hist.sum() + eps * len(real_hist))
    synth_hist = (synth_hist + eps) / (synth_hist.sum() + eps * len(synth_hist))
    
    # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    m = 0.5 * (real_hist + synth_hist)
    js = 0.5 * entropy(real_hist, m) + 0.5 * entropy(synth_hist, m)
    
    return float(js)


def acf_similarity(
    real_log_returns: np.ndarray,
    synth_log_returns: np.ndarray,
    nlags: int = 20,
    squared: bool = True
) -> float:
    """
    Compute ACF similarity between real and synthetic data.
    
    For financial data, we typically compute ACF on squared returns
    to capture volatility clustering (a key stylized fact).
    
    Args:
        real_log_returns: Real log-returns of shape (N, T-1)
        synth_log_returns: Synthetic log-returns of shape (N, T-1)
        nlags: Number of lags to compute
        squared: If True, compute ACF on squared returns (volatility)
        
    Returns:
        ACF similarity score (higher is better, 1.0 = perfect match)
        Computed as 1 - MSE between ACF curves
    """
    def compute_avg_acf(data: np.ndarray, nlags: int, squared: bool) -> np.ndarray:
        """Compute average ACF across all samples."""
        if squared:
            data = data ** 2
        
        acf_values = []
        for i in range(len(data)):
            sample = data[i]
            # Remove any NaN/Inf
            sample = sample[np.isfinite(sample)]
            if len(sample) > nlags + 1:
                try:
                    acf_i = acf(sample, nlags=nlags, fft=True)
                    acf_values.append(acf_i)
                except:
                    continue
        
        if len(acf_values) == 0:
            return np.zeros(nlags + 1)
        
        return np.mean(acf_values, axis=0)
    
    real_acf = compute_avg_acf(real_log_returns, nlags, squared)
    synth_acf = compute_avg_acf(synth_log_returns, nlags, squared)
    
    # Compute MSE
    mse = np.mean((real_acf - synth_acf) ** 2)
    
    # Convert to similarity (1 - normalized MSE)
    # Clip MSE to [0, 1] range for interpretability
    similarity = 1.0 - min(mse, 1.0)
    
    return float(similarity)
