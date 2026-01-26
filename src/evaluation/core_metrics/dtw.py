"""
DTW Distance - JS divergence of DTW distance distributions.

Measures temporal alignment similarity between real and synthetic data
by comparing distributions of DTW distances.

RESTORED FROM SOURCE: This implementation uses parallelization and the 
original logic from the source repository.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import warnings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def _dtw_distance(
    ts1: np.ndarray,
    ts2: np.ndarray,
    window: Optional[int] = None,
    normalize: bool = False
) -> float:
    """
    Compute the Dynamic Time Warping distance between two time series.
    """
    ts1 = np.asarray(ts1)
    ts2 = np.asarray(ts2)
    
    # Handle multi-feature time series
    if ts1.ndim == 2 and ts2.ndim == 2:
        if ts1.shape[1] != ts2.shape[1]:
            raise ValueError(f"Number of features must match: {ts1.shape[1]} vs {ts2.shape[1]}")
        
        feature_distances = []
        for feature_idx in range(ts1.shape[1]):
            feature_dist = _dtw_distance(ts1[:, feature_idx], ts2[:, feature_idx], 
                                       window, normalize)
            feature_distances.append(feature_dist)
        
        return np.mean(feature_distances)
    
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
    
    distance = dtw_matrix[n, m]
    
    if normalize:
        # Backtrack to find path length
        path_length = 0
        i, j = n, m
        while i > 0 and j > 0:
            path_length += 1
            if i == 1: j -= 1
            elif j == 1: i -= 1
            else:
                m_val = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
                if dtw_matrix[i-1, j-1] == m_val: i -= 1; j -= 1
                elif dtw_matrix[i-1, j] == m_val: i -= 1
                else: j -= 1
        distance = distance / max(path_length, 1)
    
    return float(distance)


def _compute_sample_distances(args):
    """Helper function for multiprocessing."""
    sample, reference_set, window, normalize = args
    sample = np.asarray(sample)
    sample_distances = []
    for ref_sample in reference_set:
        ref_sample = np.asarray(ref_sample)
        dist = _dtw_distance(sample, ref_sample, window, normalize)
        sample_distances.append(dist)
    return np.mean(sample_distances)


def dtw_distance(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    n_samples: int = 100,
    n_reference: int = 50,
    n_bins: int = 50,
    n_jobs: Optional[int] = None,
    window: Optional[int] = None,
    normalize: bool = True
) -> float:
    """
    Measure DTW-based Jensen-Shannon divergence between two distributions.
    
    RESTORED FROM SOURCE: This implements the precise JS divergence of 
    distributions approach used in the original repo.
    """
    # Sample from data for performance
    n_samples = min(n_samples, len(real_data), len(synth_data))
    real_indices = np.random.choice(len(real_data), n_samples, replace=False)
    synth_indices = np.random.choice(len(synth_data), n_samples, replace=False)
    
    generated_samples = [synth_data[i] for i in synth_indices]
    real_samples = [real_data[i] for i in real_indices]

    # Create reference set
    n_ref = min(n_reference, n_samples * 2)
    all_samples = generated_samples + real_samples
    ref_indices = np.random.choice(len(all_samples), size=n_ref, replace=False)
    reference_samples = [all_samples[i] for i in ref_indices]

    # Set number of jobs
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    def compute_distribution(samples, ref_set):
        if n_jobs > 1 and len(samples) > 1:
            args_list = [(s, ref_set, window, normalize) for s in samples]
            with mp.Pool(processes=n_jobs) as pool:
                distances = list(tqdm(
                    pool.imap(_compute_sample_distances, args_list),
                    total=len(samples),
                    desc="DTW Parallel",
                    leave=False
                ))
            return np.array(distances)
        else:
            distances = []
            for s in tqdm(samples, desc="DTW Sequential", leave=False):
                distances.append(_compute_sample_distances((s, ref_set, window, normalize)))
            return np.array(distances)

    gen_dist = compute_distribution(generated_samples, reference_samples)
    real_dist = compute_distribution(real_samples, reference_samples)

    # Compute histograms
    all_d = np.concatenate([gen_dist, real_dist])
    bins = np.linspace(np.min(all_d), np.max(all_d), n_bins + 1)
    
    gen_hist, _ = np.histogram(gen_dist, bins=bins, density=True)
    real_hist, _ = np.histogram(real_dist, bins=bins, density=True)
    
    # Probabilities
    gen_p = (gen_hist + 1e-10) / (np.sum(gen_hist) + 1e-10 * n_bins)
    real_p = (real_hist + 1e-10) / (np.sum(real_hist) + 1e-10 * n_bins)
    
    # JS Divergence
    m = 0.5 * (gen_p + real_p)
    kl_gen = np.sum(gen_p * np.log(gen_p / m))
    kl_real = np.sum(real_p * np.log(real_p / m))
    js_div = 0.5 * kl_gen + 0.5 * kl_real
    
    return float(js_div)
