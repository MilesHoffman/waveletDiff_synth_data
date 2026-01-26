"""
Context-FID - Frechet Inception Distance on TS2Vec embeddings.

Uses TS2Vec to encode time series into embeddings, then computes
FID between real and synthetic embedding distributions.
"""

import numpy as np
import scipy.linalg
from typing import Optional
import warnings


def _calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def context_fid(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    output_dims: int = 320,
    batch_size: int = 8
) -> float:
    """
    Compute Context-FID using TS2Vec embeddings.
    
    Args:
        real_data: Real data of shape (N, T, D) - raw scale
        synth_data: Synthetic data of shape (N, T, D) - raw scale
        output_dims: Dimension of TS2Vec output embeddings
        batch_size: Batch size for TS2Vec
        
    Returns:
        Context-FID score (lower is better)
    """
    try:
        from ..ts2vec_model.ts2vec import TS2Vec
    except ImportError:
        # Fallback for different import contexts
        try:
            from ts2vec_model.ts2vec import TS2Vec
        except ImportError:
            warnings.warn("TS2Vec not available, returning NaN for Context-FID")
            return float('nan')
    
    model = TS2Vec(
        input_dims=real_data.shape[-1], 
        device=0 if __import__('torch').cuda.is_available() else 'cpu', 
        batch_size=batch_size, 
        lr=0.001, 
        output_dims=output_dims,
        max_train_length=3000
    )
    
    model.fit(real_data, verbose=False)
    
    real_repr = model.encode(real_data, encoding_window='full_series')
    synth_repr = model.encode(synth_data, encoding_window='full_series')
    
    # Context-FID uses same indices logic
    idx = np.random.permutation(real_data.shape[0])
    real_repr = real_repr[idx]
    synth_repr = synth_repr[idx]
    
    return float(_calculate_fid(real_repr, synth_repr))
