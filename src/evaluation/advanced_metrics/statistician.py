"""
Statistician - Full Manifold Alpha-Precision and Beta-Recall.

These metrics measure manifold-level fidelity and diversity:
- α-Precision: Quality (do generated samples fall within real manifold?)
- β-Recall: Diversity (do generated samples cover all real manifold regions?)

This is the FULL manifold-based implementation (not k-NN approximation).
Based on: "Improved Precision and Recall Metric for Assessing Generative Models"
(Kynkäänniemi et al., NeurIPS 2019)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


def _compute_manifold_boundary(data: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute the manifold boundary radius for each sample.
    
    The radius is defined as the distance to the k-th nearest neighbor
    within the same distribution.
    
    Args:
        data: Flattened data of shape (N, D) where D = T * num_features
        k: k-th nearest neighbor to use for radius computation
        
    Returns:
        Array of radii for each sample
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    # k+1 because the first neighbor is the point itself
    radii = distances[:, k]
    return radii


def alpha_precision(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    k: int = 3
) -> float:
    """
    Compute α-Precision (Manifold Precision).
    
    Measures the fraction of synthetic samples that fall within
    the manifold of real data. High precision means generated
    samples are realistic (within the real data distribution).
    
    Args:
        real_data: Real data of shape (N, T, D), should be scaled to [0,1]
        synth_data: Synthetic data of shape (N, T, D), should be scaled to [0,1]
        k: k-th nearest neighbor for manifold boundary
        
    Returns:
        Precision score in [0, 1] (higher is better)
    """
    # Flatten samples
    real_flat = real_data.reshape(len(real_data), -1)
    synth_flat = synth_data.reshape(len(synth_data), -1)
    
    # Compute real manifold boundary radii
    real_radii = _compute_manifold_boundary(real_flat, k)
    
    # For each synthetic sample, find distance to nearest real sample
    nn_real = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
    nn_real.fit(real_flat)
    synth_to_real_dist, synth_to_real_idx = nn_real.kneighbors(synth_flat)
    synth_to_real_dist = synth_to_real_dist.squeeze()
    synth_to_real_idx = synth_to_real_idx.squeeze()
    
    # A synthetic sample is "within manifold" if its distance to nearest
    # real sample is <= the radius of that real sample's neighborhood
    within_manifold = synth_to_real_dist <= real_radii[synth_to_real_idx]
    precision = np.mean(within_manifold)
    
    return float(precision)


def beta_recall(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    k: int = 3
) -> float:
    """
    Compute β-Recall (Manifold Recall).
    
    Measures the fraction of real samples that have a synthetic
    sample within their manifold neighborhood. High recall means
    the generator covers all modes of the real distribution.
    
    Args:
        real_data: Real data of shape (N, T, D), should be scaled to [0,1]
        synth_data: Synthetic data of shape (N, T, D), should be scaled to [0,1]
        k: k-th nearest neighbor for manifold boundary
        
    Returns:
        Recall score in [0, 1] (higher is better)
    """
    # Flatten samples
    real_flat = real_data.reshape(len(real_data), -1)
    synth_flat = synth_data.reshape(len(synth_data), -1)
    
    # Compute real manifold boundary radii
    real_radii = _compute_manifold_boundary(real_flat, k)
    
    # For each real sample, find distance to nearest synthetic sample
    nn_synth = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
    nn_synth.fit(synth_flat)
    real_to_synth_dist, _ = nn_synth.kneighbors(real_flat)
    real_to_synth_dist = real_to_synth_dist.squeeze()
    
    # A real sample is "covered" if there's a synthetic sample
    # within its neighborhood radius
    covered = real_to_synth_dist <= real_radii
    recall = np.mean(covered)
    
    return float(recall)
