"""
Integrity Officer - DCR and Memorization Ratio.

These metrics detect if the model is "cheating" by copying training data:
- DCR (Distance to Closest Record): How close are generated samples to training data?
- Memorization Ratio: Fraction of samples that appear to be memorized
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict


def dcr_score(
    real_data: np.ndarray,
    synth_data: np.ndarray
) -> Dict[str, float]:
    """
    Compute Distance to Closest Record (DCR) statistics.
    
    DCR measures the Euclidean distance from each synthetic sample
    to its nearest neighbor in the real (training) data.
    
    Low DCR values indicate potential memorization.
    
    Args:
        real_data: Flattened real data of shape (N, T*D)
        synth_data: Flattened synthetic data of shape (N, T*D)
        
    Returns:
        Dict with 'mean', 'std', 'min', 'median', 'max' DCR values
    """
    # Ensure data is 2D (flattened)
    if real_data.ndim > 2:
        real_data = real_data.reshape(len(real_data), -1)
    if synth_data.ndim > 2:
        synth_data = synth_data.reshape(len(synth_data), -1)
    
    # Find nearest neighbor in real data for each synthetic sample
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
    nn.fit(real_data)
    distances, _ = nn.kneighbors(synth_data)
    distances = distances.squeeze()
    
    return {
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'median': float(np.median(distances)),
        'max': float(np.max(distances)),
    }


def memorization_ratio(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    k: int = 2
) -> float:
    """
    Compute the Memorization Ratio using the 1/3 Rule.
    
    A synthetic sample is considered "memorized" if:
    d(synth, NN1) < (1/3) * d(synth, NN2)
    
    Where NN1 and NN2 are the 1st and 2nd nearest neighbors
    in the real (training) data.
    
    This detects if the model is reproducing specific training
    examples rather than generating novel samples.
    
    Args:
        real_data: Flattened real data of shape (N, T*D)
        synth_data: Flattened synthetic data of shape (N, T*D)
        k: Number of neighbors to consider (default 2 for NN1, NN2)
        
    Returns:
        Memorization ratio in [0, 1] (lower is better, <0.05 is good)
    """
    # Ensure data is 2D (flattened)
    if real_data.ndim > 2:
        real_data = real_data.reshape(len(real_data), -1)
    if synth_data.ndim > 2:
        synth_data = synth_data.reshape(len(synth_data), -1)
    
    # Find k nearest neighbors for each synthetic sample
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    nn.fit(real_data)
    distances, _ = nn.kneighbors(synth_data)
    
    # 1/3 Rule: memorized if d(NN1) < (1/3) * d(NN2)
    d_nn1 = distances[:, 0]  # Distance to nearest neighbor
    d_nn2 = distances[:, 1]  # Distance to 2nd nearest neighbor
    
    # Avoid division by zero
    threshold = d_nn2 / 3.0
    memorized = d_nn1 < threshold
    
    ratio = np.mean(memorized)
    
    return float(ratio)
