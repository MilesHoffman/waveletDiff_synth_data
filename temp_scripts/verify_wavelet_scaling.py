import sys
import os
import numpy as np
import torch
import pywt

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from data.loaders import load_stocks_data
from data.module import WaveletTimeSeriesDataModule

def verify_wavelet_scaling():
    print("=== Wavelet Robust Scaling Verification ===")
    
    config = {
        'dataset': {'name': 'stocks', 'seq_len': 24},
        'training': {'batch_size': 32},
        'data': {'data_dir': 'data', 'normalize_data': True},
        'wavelet': {'type': 'db4', 'levels': 3},
        'conditioning': {'past_days': 200}
    }
    
    print("Initializing DataModule...")
    # This automatically calls _convert_to_wavelet_coefficients which now has the scaling logic
    dm = WaveletTimeSeriesDataModule(config=config)
    
    # 1. Check forward scaling statistics
    print("\n1. Forward Scaling Check")
    coeffs = dm.data_tensor.numpy()
    info = dm.get_wavelet_info()
    
    print(f"Shape of scaled coefficients: {coeffs.shape}")
    
    starts = info['level_start_indices']
    dims = info['level_dims']
    stats = info['robust_stats']
    
    for level_idx, (start, dim) in enumerate(zip(starts, dims)):
        end = start + dim
        level_data = coeffs[:, start:end, 0] # checking first feature
        
        # Calculate stats of the SCALED data. Should be median ~ 0, IQR ~ 1.349
        actual_median = np.median(level_data)
        q75, q25 = np.percentile(level_data, [75, 25])
        actual_iqr = q75 - q25
        
        original_median = stats['medians'][level_idx, 0]
        original_scale = stats['scale_factors'][level_idx, 0]
        
        print(f"Level {level_idx}:")
        print(f"  Original Median: {original_median:>10.4f} | Scaled Median: {actual_median:>10.4f}")
        print(f"  Original Scale:  {original_scale:>10.4f} | Scaled IQR:    {actual_iqr:>10.4f} (Expected ~1.349)")

    # 2. Check inverse scaling identity
    print("\n2. Inverse Scaling Identity Check")
    
    # Convert scaled wavelets back to time series
    reconstructed_ts = dm.convert_wavelet_to_timeseries(dm.data_tensor).numpy()
    
    # Compare with original raw_data_tensor
    original_ts = dm.raw_data_tensor.numpy()
    
    diff = np.abs(original_ts - reconstructed_ts)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max Absolute Difference: {max_diff:.8f}")
    print(f"Mean Absolute Difference: {mean_diff:.8f}")
    
    if max_diff < 1e-4:
        print("SUCCESS: Wavelet Forward -> Scaling -> Inverse Scaling -> Inverse Wavelet is lossless.")
    else:
        print("FAILURE: Identity mapping failed. High reconstruction error.")

if __name__ == "__main__":
    verify_wavelet_scaling()
