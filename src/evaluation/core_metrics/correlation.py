"""
Correlation Score - Cross-correlation matrix divergence.

Measures how well the synthetic data preserves feature
cross-correlations from the real data.
"""

import torch
import numpy as np
from typing import Optional


def _cacf_torch(x: torch.Tensor, max_lag: int, dim=(0, 1)) -> torch.Tensor:
    """Compute cross-autocorrelation function."""
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / (x.std(dim, keepdims=True) + 1e-8)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    
    cacf_list = []
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, 1)
        cacf_list.append(cacf_i)
    
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def correlation_score(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    sample_size: Optional[int] = 1000
) -> float:
    """
    Compute cross-correlation divergence between real and synthetic data.
    
    Args:
        real_data: Real data of shape (N, T, D) - raw scale
        synth_data: Synthetic data of shape (N, T, D) - raw scale
        sample_size: Number of samples to use for computation
        
    Returns:
        Correlation score (lower is better)
    """
    # Sample if needed
    if sample_size is not None:
        sample_size = min(sample_size, len(real_data), len(synth_data))
        real_idx = np.random.choice(len(real_data), sample_size, replace=False)
        synth_idx = np.random.choice(len(synth_data), sample_size, replace=False)
        real_data = real_data[real_idx]
        synth_data = synth_data[synth_idx]
    
    x_real = torch.from_numpy(real_data).float()
    x_synth = torch.from_numpy(synth_data).float()
    
    # Compute cross-correlation matrices
    cross_correl_real = _cacf_torch(x_real, 1).mean(0)[0]
    cross_correl_synth = _cacf_torch(x_synth, 1).mean(0)[0]
    
    # Compute L1 divergence
    loss = torch.abs(cross_correl_synth - cross_correl_real).sum()
    
    return float(loss / 10.0)
