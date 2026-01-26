"""
DTW Distance - JS divergence of DTW distance distributions.

Measures temporal alignment similarity between real and synthetic data
by comparing distributions of DTW distances.
"""

import numpy as np
from typing import Optional, Dict
import warnings
from tqdm import tqdm


def _dtw_distance_1d(ts1: np.ndarray, ts2: np.ndarray, window: Optional[int] = None) -> float:
    """Compute DTW distance between two 1D time series."""
    n, m = len(ts1), len(ts2)
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        if window is None:
            j_start, j_end = 1, m + 1
        else:
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)
        
        for j in range(j_start, j_end):
            cost = abs(ts1[i-1] - ts2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    return dtw_matrix[n, m]


def _dtw_distance_multi(ts1: np.ndarray, ts2: np.ndarray, window: Optional[int] = None) -> float:
    """Compute mean DTW distance across features for multi-feature time series."""
    if ts1.ndim == 1 and ts2.ndim == 1:
        return _dtw_distance_1d(ts1, ts2, window)
    
    if ts1.ndim == 2 and ts2.ndim == 2:
        distances = []
        for f in range(ts1.shape[1]):
            d = _dtw_distance_1d(ts1[:, f], ts2[:, f], window)
            distances.append(d)
        return np.mean(distances)
    
    raise ValueError("Time series must have matching dimensions")


def dtw_distance(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    n_samples: int = 100,
    n_reference: int = 50,
    n_bins: int = 50
) -> float:
    """
    Compute DTW-based JS divergence between real and synthetic distributions.
    
    Args:
        real_data: Real data of shape (N, T, D) - raw scale
        synth_data: Synthetic data of shape (N, T, D) - raw scale
        n_samples: Number of samples to use from each distribution
        n_reference: Size of reference set for distance computation
        n_bins: Number of bins for histogram
        
    Returns:
        JS divergence of DTW distance distributions (lower is better)
    """
    # Sample from data
    n_samples = min(n_samples, len(real_data), len(synth_data))
    real_idx = np.random.choice(len(real_data), n_samples, replace=False)
    synth_idx = np.random.choice(len(synth_data), n_samples, replace=False)
    
    real_samples = real_data[real_idx]
    synth_samples = synth_data[synth_idx]
    
    # Create reference set from combined data
    n_ref = min(n_reference, n_samples)
    all_samples = np.concatenate([real_samples, synth_samples], axis=0)
    ref_idx = np.random.choice(len(all_samples), n_ref, replace=False)
    reference_set = all_samples[ref_idx]
    
    def compute_distances(samples, ref_set):
        """Compute mean DTW distance from each sample to reference set."""
        distances = []
        for sample in tqdm(samples, desc="DTW Distance", leave=False):
            sample_dists = [_dtw_distance_multi(sample, ref) for ref in ref_set]
            distances.append(np.mean(sample_dists))
        return np.array(distances)
    
    real_distances = compute_distances(real_samples, reference_set)
    synth_distances = compute_distances(synth_samples, reference_set)
    
    # Compute JS divergence from histograms
    all_dist = np.concatenate([real_distances, synth_distances])
    bins = np.linspace(all_dist.min() * 0.95, all_dist.max() * 1.05, n_bins + 1)
    
    real_hist, _ = np.histogram(real_distances, bins=bins, density=True)
    synth_hist, _ = np.histogram(synth_distances, bins=bins, density=True)
    
    # Normalize and add epsilon
    eps = 1e-10
    real_hist = (real_hist + eps) / (real_hist.sum() + eps * len(real_hist))
    synth_hist = (synth_hist + eps) / (synth_hist.sum() + eps * len(synth_hist))
    
    # JS divergence
    m = 0.5 * (real_hist + synth_hist)
    kl_real = np.sum(real_hist * np.log(real_hist / m))
    kl_synth = np.sum(synth_hist * np.log(synth_hist / m))
    js_div = 0.5 * (kl_real + kl_synth)
    
    return float(js_div)
