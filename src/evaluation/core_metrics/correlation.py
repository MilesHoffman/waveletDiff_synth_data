"""
Correlation Score - Cross-correlation matrix divergence.

Measures how well the synthetic data preserves feature
cross-correlations from the real data.

RESTORED FROM SOURCE: This implementation uses the exact cacf_torch logic 
from cross_correlation.py in the source repository.
"""

import torch
import numpy as np
from typing import Optional


def _cacf_torch(x: torch.Tensor, max_lag: int, dim=(0, 1)) -> torch.Tensor:
    """
    Compute cross-autocorrelation function.
    Restored from source cross_correlation.py.
    """
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    # Normalize
    x = (x - x.mean(dim, keepdims=True)) / (x.std(dim, keepdims=True) + 1e-8)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    
    cacf_list = list()
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
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
    
    Restored from source CrossCorrelLoss logic.
    """
    # Sample if needed for performance
    sample_size = min(sample_size, len(real_data), len(synth_data))
    real_idx = np.random.choice(len(real_data), sample_size, replace=False)
    synth_idx = np.random.choice(len(synth_data), sample_size, replace=False)
    
    x_real = torch.from_numpy(real_data[real_idx]).float()
    x_synth = torch.from_numpy(synth_data[synth_idx]).float()
    
    # Compute cross-correlation matrices (max_lag=1)
    cross_correl_real = _cacf_torch(x_real, 1).mean(0)[0]
    cross_correl_synth = _cacf_torch(x_synth, 1).mean(0)[0]
    
    # Compute divergence (L1 norm of difference)
    # Matching CrossCorrelLoss.compute logic: loss = norm_foo(fake - real) / 10.
    loss = torch.abs(cross_correl_synth - cross_correl_real).sum()
    
    return float(loss / 10.0)
