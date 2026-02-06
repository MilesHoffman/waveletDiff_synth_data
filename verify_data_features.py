
import sys
import os
import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.module import WaveletTimeSeriesDataModule
from utils import ConfigManager

def check_features():
    print("--- Verifying Data Features ---")
    
    # Load default config
    cm = ConfigManager()
    config = cm.load(dataset_name="stocks")
    
    # Initialize Data Module
    print("Initializing Data Module...")
    # Override data_dir to absolute path just in case
    config['data']['data_dir'] = os.path.abspath("data")
    print(f"Using Data Dir: {config['data']['data_dir']}")
    
    dm = WaveletTimeSeriesDataModule(config=config)
    
    # 1. Check Dimensions
    raw_data = dm.raw_data_tensor
    print(f"\nDimensions Check:")
    print(f"  Shape: {raw_data.shape} (Expected: [N, 24, 8])")
    
    feature_names = dm.norm_stats['feature_names']
    print(f"  Features: {feature_names}")
    
    if raw_data.shape[-1] != 8:
        print("FAIL: Expected 8 features, got", raw_data.shape[-1])
        return
        
    # 2. Check Day Features (Indices 5 and 6)
    print(f"\nDay Feature Check (First 10 steps of sample 0):")
    day_sin = raw_data[0, :10, 5].numpy()
    day_cos = raw_data[0, :10, 6].numpy()
    
    # Reconstruct angle
    angles = np.arctan2(day_sin, day_cos)
    days_reconstructed = (angles / (2 * np.pi) * 7.0) % 7
    days_rounded = np.round(days_reconstructed).astype(int)
    
    for i in range(10):
        print(f"  Step {i}: Sin={day_sin[i]:.3f}, Cos={day_cos[i]:.3f} -> Day={days_rounded[i]}")
        
    # Check continuity (weekend gaps should show as jumps in day index)
    
    # 3. Check Gap Feature (Index 7)
    print(f"\nGap Feature Check (First 5 nonzero gaps):")
    gap_norm = raw_data[0, :, 7].numpy()
    
    # Index 0 is always 0 (boundary condition)
    print(f"  Gap[0]: {gap_norm[0]:.4f} (Should be 0.0)")
    
    valid_gaps = gap_norm[gap_norm != 0]
    print(f"  Sample Gaps: {valid_gaps[:5]}")
    
    # 4. Check Inverse Normalization Slicing
    print(f"\nInverse Normalization Check:")
    print("  Input shape:", raw_data.shape)
    
    # Dummy inverse call
    output_dollar = dm.inverse_normalize(raw_data.numpy()[:10], fixed_anchor=100.0)
    print("  Output shape:", output_dollar.shape)
    
    if output_dollar.shape[-1] == 5:
        print("SUCCESS: Inverse normalized data has 5 channels (OHLCV). auxiliary features stripped.")
    else:
        print(f"FAIL: Output has {output_dollar.shape[-1]} channels, expected 5.")

if __name__ == "__main__":
    check_features()
