"""
Stylized Facts Metrics - Financial Realism Checks.

Implements checks for:
1. Fat Tails (Kurtosis)
2. Volatility Clustering (ARCH barrier)
3. Leverage Effect (Return-Volatility Correlation)
"""

import numpy as np
from scipy.stats import kurtosis
import torch

def _compute_kurtosis(data: np.ndarray) -> np.ndarray:
    """Compute kurtosis for each feature."""
    # Data shape: (N, T, D)
    # Flatten across N and T to get distribution for each feature D
    # Or should we compute per-sequence? 
    # Usually "Fat Tails" is a property of the return distribution.
    
    # Check if data is returns or prices. Assuming input is normalized returns or similar.
    # If standard scaled, mean ~ 0, std ~ 1.
    
    N, T, D = data.shape
    kurt_vals = []
    
    for d in range(D):
        feat_data = data[:, :, d].flatten()
        k = kurtosis(feat_data, fisher=True) # Excess kurtosis (normal=0)
        kurt_vals.append(k)
        
    return np.array(kurt_vals)

def kurtosis_score(real_data: np.ndarray, synth_data: np.ndarray) -> float:
    """
    Measure the difference in Kurtosis (Fat Tails) between Real and Synthetic.
    
    Returns:
        Mean absolute difference in excess kurtosis across features.
    """
    real_kurt = _compute_kurtosis(real_data)
    synth_kurt = _compute_kurtosis(synth_data)
    
    return float(np.mean(np.abs(real_kurt - synth_kurt)))


def _compute_vol_cluster(data: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Compute Autocorrelation of Absolute/Squared Returns (Volatility Clustering).
    """
    # Data shape: (N, T, D)
    # We want ACF of |r_t|
    
    abs_data = np.abs(data)
    
    # Subtract mean of absolute returns
    abs_data = abs_data - abs_data.mean(axis=1, keepdims=True)
    
    N, T, D = data.shape
    if T <= max_lag:
        max_lag = max(1, T - 1)
        
    acfs = []
    
    # Vectorized ACF calculation?
    # Let's simple loop for clarity per feature
    for d in range(D):
        x = abs_data[:, :, d] # (N, T)
        
        # Compute ACF for each lag
        lag_corrs = []
        for lag in range(1, max_lag + 1):
            # Corr(x_t, x_{t-lag})
            series_head = x[:, :-lag]
            series_tail = x[:, lag:]
            
            # Simple cov / var
            prod = series_head * series_tail
            mean_prod = prod.mean() # effectively cov since we zero-centered
            var = x.var() + 1e-8
            
            corr = mean_prod / var
            lag_corrs.append(corr)
            
        acfs.append(lag_corrs)
        
    return np.array(acfs) # (D, max_lag)


def volatility_clustering_score(real_data: np.ndarray, synth_data: np.ndarray) -> float:
    """
    Measure the preservation of Volatility Clustering (ARCH effect).
    
    Returns:
        Mean Euclidean distance between the Volatility ACF curves.
    """
    # Volatility needs returns, assuming input is already stationary-ish or returns
    # Users pass (N,T,D).
    
    real_vol_acf = _compute_vol_cluster(real_data)
    synth_vol_acf = _compute_vol_cluster(synth_data)
    
    # Distance between curves
    # (D, L)
    diff = real_vol_acf - synth_vol_acf
    dist = np.sqrt(np.sum(diff**2, axis=1)) # Euclidean dist per feature
    
    return float(np.mean(dist))
