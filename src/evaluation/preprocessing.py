"""
Data preprocessing utilities for evaluation metrics.

Handles data transformations required by different metrics:
- Min-max scaling to [0,1] for RNN-based metrics
- Log-returns for financial distribution metrics
- Flattening for distance-based metrics
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_log_returns(data: np.ndarray, price_col: int = 3) -> np.ndarray:
    """
    Compute log returns from close prices.
    
    Args:
        data: Shape (N, T, D) where D includes OHLCV features
        price_col: Index of the close price column (default=3 for OHLCV)
        
    Returns:
        Log returns of shape (N, T-1)
    """
    close = data[:, :, price_col]
    log_returns = np.diff(np.log(close + 1e-8), axis=1)
    return log_returns


def scale_to_01(
    data: np.ndarray, 
    reference_stats: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Min-max scale data to [0,1] range.
    
    Args:
        data: Shape (N, T, D)
        reference_stats: If provided, use these min/max values instead of computing from data
        
    Returns:
        Tuple of (scaled_data, stats_dict)
    """
    if reference_stats is None:
        d_min = np.min(data, axis=(0, 1), keepdims=True)
        d_max = np.max(data, axis=(0, 1), keepdims=True)
    else:
        d_min = reference_stats['min']
        d_max = reference_stats['max']
    
    scaled = (data - d_min) / (d_max - d_min + 1e-8)
    return scaled, {'min': d_min, 'max': d_max}


def flatten_samples(data: np.ndarray) -> np.ndarray:
    """
    Flatten time series samples for distance computations.
    
    Args:
        data: Shape (N, T, D)
        
    Returns:
        Flattened array of shape (N, T*D)
    """
    return data.reshape(data.shape[0], -1)


def sanitize_data(data: np.ndarray) -> np.ndarray:
    """
    Replace Infs and NaNs with finite values to prevent sklearn errors.
    
    Args:
        data: Input array
        
    Returns:
        Sanitized array with Infs/NaNs replaced
    """
    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    return data


def prepare_evaluation_data(
    real: np.ndarray, 
    synth: np.ndarray,
    exclude_volume: bool = True,
    close_col: int = 3
) -> Dict:
    """
    Prepare all data formats needed by evaluation metrics.
    
    Uses REAL data statistics only for scaling to avoid data leakage.
    
    Args:
        real: Real data of shape (N, T, D)
        synth: Synthetic data of shape (N, T, D)
        exclude_volume: If True, exclude the last feature (volume) for OHLC-only metrics
        close_col: Index of the close price column for log-returns
        
    Returns:
        Dictionary with 'real' and 'synth' sub-dicts containing:
        - 'raw': Original data
        - 'scaled_01': Min-max scaled to [0,1]
        - 'log_returns': Log returns from close prices
        - 'flattened': Flattened for distance metrics
    """
    # Optionally exclude volume
    if exclude_volume and real.shape[2] > 1:
        real_ohlc = real[..., :-1]
        synth_ohlc = synth[..., :-1]
    else:
        real_ohlc = real
        synth_ohlc = synth
    
    # Scale using REAL data statistics only
    real_scaled, stats = scale_to_01(real_ohlc)
    synth_scaled, _ = scale_to_01(synth_ohlc, stats)
    
    # Sanitize for numerical stability
    real_scaled = sanitize_data(real_scaled)
    synth_scaled = sanitize_data(synth_scaled)
    
    return {
        'real': {
            'raw': real_ohlc,
            'scaled_01': real_scaled,
            'log_returns': compute_log_returns(real_ohlc, min(close_col, real_ohlc.shape[2]-1)),
            'flattened': flatten_samples(real_ohlc),
        },
        'synth': {
            'raw': synth_ohlc,
            'scaled_01': synth_scaled,
            'log_returns': compute_log_returns(synth_ohlc, min(close_col, synth_ohlc.shape[2]-1)),
            'flattened': flatten_samples(synth_ohlc),
        },
        'stats': stats
    }
