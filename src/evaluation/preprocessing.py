"""
Data preprocessing utilities for evaluation metrics.

Handles data transformations required by different metrics:
- Min-max scaling to [0,1] for RNN-based metrics
- Log-returns for financial distribution metrics
- Flattening for distance-based metrics
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_log_returns(data: np.ndarray, price_col: int = 3, is_reparam: bool = False) -> np.ndarray:
    """
    Compute log returns from data.
    
    Args:
        data: Shape (N, T, D)
        price_col: Index of the close price column (default=3 for OHLCV)
        is_reparam: If True, data is already reparameterized features.
                   We usage 'body_norm' (index 1) as proxy for returns.
        
    Returns:
        Log returns of shape (N, T) or (N, T-1)
    """
    if is_reparam:
        # In reparam space, index 1 is 'body_norm' which is (Close-Open)/Anchor
        # This is already a return-like stationary feature.
        # We perform no diff(), just return the feature as-is.
        # Shape: (N, T)
        return data[:, :, 1]
    
    # Clip to avoid log(0) or log(negative)
    close = np.maximum(data[:, :, price_col], 1e-8)
    log_returns = np.diff(np.log(close), axis=1)
    return log_returns


def scale_to_01(
    data: np.ndarray, 
    reference_stats: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Min-max scale data to [0,1] range.
    """
    if reference_stats is None:
        d_min = np.min(data, axis=(0, 1), keepdims=True)
        d_max = np.max(data, axis=(0, 1), keepdims=True)
    else:
        d_min = reference_stats['min']
        d_max = reference_stats['max']
    
    scaled = (data - d_min) / (d_max - d_min + 1e-8)
    return scaled, {'min': d_min, 'max': d_max}


def standardize(
    data: np.ndarray, 
    reference_stats: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Z-score standardize data to (mean=0, std=1).
    """
    if reference_stats is None:
        mean = np.mean(data, axis=(0, 1), keepdims=True)
        std = np.std(data, axis=(0, 1), keepdims=True)
        # Handle constant features (std=0)
        std = np.where(std < 1e-8, 1.0, std)
    else:
        mean = reference_stats['mean']
        std = reference_stats['std']
    
    standardized = (data - mean) / std
    return standardized, {'mean': mean, 'std': std}


def flatten_samples(data: np.ndarray) -> np.ndarray:
    """
    Flatten time series samples for distance computations.
    """
    return data.reshape(data.shape[0], -1)


def sanitize_data(data: np.ndarray) -> np.ndarray:
    """
    Replace Infs and NaNs with finite values.
    """
    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    return data


def prepare_evaluation_data(
    real: np.ndarray, 
    synth: np.ndarray,
    exclude_volume: bool = True,
    close_col: int = 3,
    is_reparam: bool = False
) -> Dict:
    """
    Prepare all data formats needed by evaluation metrics.
    
    Uses REAL data statistics only for scaling to avoid data leakage.
    
    Args:
        real: Real data of shape (N, T, D)
        synth: Synthetic data of shape (N, T, D)
        exclude_volume: If True, exclude the last feature (volume) for OHLC-only metrics
        close_col: Index of the close price column for log-returns
        is_reparam: Whether input data is already reparameterized
        
    Returns:
        Dictionary with 'real' and 'synth' sub-dicts containing:
        - 'raw': Original data
        - 'scaled_01': Min-max scaled to [0,1]
        - 'standardized': Z-score standardized (mean 0, std 1)
        - 'log_returns': Log returns (or body_norm if reparam)
        - 'flattened': Flattened raw data
        - 'flattened_standardized': Flattened standardized data (preferred for distances)
    """
    # Optionally exclude volume
    # NOTE: In reparam space, volume is index 4. If exclude_volume=True, we slice :-1.
    if exclude_volume and real.shape[2] > 1:
        real_processed = real[..., :-1]
        synth_processed = synth[..., :-1]
    else:
        real_processed = real
        synth_processed = synth
    
    # 1. Min-Max Scaling (legacy/visualization)
    real_scaled, stats_minmax = scale_to_01(real_processed)
    synth_scaled, _ = scale_to_01(synth_processed, stats_minmax)
    
    # 2. Standardization (Z-score) - Preferred for Neural Nets & Distances
    real_std, stats_std = standardize(real_processed)
    synth_std, _ = standardize(synth_processed, stats_std)
    
    # Sanitize inputs
    real_scaled = sanitize_data(real_scaled)
    synth_scaled = sanitize_data(synth_scaled)
    real_std = sanitize_data(real_std)
    synth_std = sanitize_data(synth_std)
    
    # 3. Log Returns / Stationary Proxy
    # Note: reparam data typically has Volume at -1. if exclude_volume=True, 
    # we effectively remove it. compute_log_returns logic for reparam works on feature indices.
    # If we sliced the data, indices might shift. 
    # BUT: compute_log_returns takes 'data', which usually refers to the RAW input or processed?
    # Let's pass the 'processed' (potentially sliced) data.
    # If is_reparam=True, we want index 1 (body_norm). Slicing doesn't affect index 1 (as long as D > 1).
    
    real_log_ret = compute_log_returns(real_processed, min(close_col, real_processed.shape[2]-1), is_reparam)
    synth_log_ret = compute_log_returns(synth_processed, min(close_col, synth_processed.shape[2]-1), is_reparam)
    
    return {
        'real': {
            'raw': real_processed,
            'scaled_01': real_scaled,
            'standardized': real_std,
            'log_returns': real_log_ret,
            'flattened': flatten_samples(real_processed),
            'flattened_standardized': flatten_samples(real_std),
        },
        'synth': {
            'raw': synth_processed,
            'scaled_01': synth_scaled,
            'standardized': synth_std,
            'log_returns': synth_log_ret,
            'flattened': flatten_samples(synth_processed),
            'flattened_standardized': flatten_samples(synth_std),
        },
        'stats': {
            'minmax': stats_minmax,
            'standard': stats_std
        }
    }
