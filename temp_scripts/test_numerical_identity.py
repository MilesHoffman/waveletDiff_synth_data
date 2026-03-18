
import numpy as np
import torch
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.loaders import load_stocks_data, _select_ohlcv_columns
from data.module import WaveletTimeSeriesDataModule

def test_numerical_identity():
    data_dir = "data"
    spy_path = os.path.join(data_dir, "stocks", "SPY_stock_data.csv")
    
    if not os.path.exists(spy_path):
        print(f"Error: Could not find {spy_path}")
        return

    # 1. Load the original CSV manually to get ground truth prices
    df = pd.read_csv(spy_path)
    # Handle metadata/header row if present (matching loader logic)
    try:
        pd.to_numeric(df['Open'].iloc[0])
    except:
        df = df.iloc[1:].reset_index(drop=True)
    
    df = _select_ohlcv_columns(df)
    raw_ohlcv = df.values.astype(np.float64) # [TotalTimesteps, 5]

    # 2. Load data using the SOTA loader (which applies reparameterization)
    seq_len = 24
    past_days = 20
    data_tensor, norm_stats = load_stocks_data(data_dir, seq_len=seq_len, past_days=past_days)
    windows = data_tensor.numpy() # [N, seq_len, 22]

    # 3. Initialize DataModule for inverse_normalize
    config = {
        'dataset': {'name': 'stocks', 'seq_len': seq_len},
        'training': {'batch_size': 32},
        'data': {'data_dir': data_dir, 'normalize_data': True},
        'wavelet': {'type': 'sym2', 'levels': 3},
        'conditioning': {'past_days': past_days},
        'performance': {'matmul_precision': 'medium', 'precision': '32'},
        'paths': {'output_dir': 'outputs'}
    }
    dm = WaveletTimeSeriesDataModule(config=config, data_tensor=data_tensor)
    dm.norm_stats = norm_stats

    # 4. Reconstruct OHLCV
    # We'll test the first 100 windows to be thorough but efficient
    n_test = min(100, len(windows))
    reconstructed = dm.inverse_normalize(windows[:n_test], sample_indices=np.arange(n_test))

    # 5. Compare with Original
    # The loader has a 'valid_start' offset.
    valid_start = max(200, past_days + 20)
    
    print(f"Verification: Comparing {n_test} windows starting from index {valid_start}")
    
    all_passed = True
    max_diff = 0.0
    
    for i in range(n_test):
        # The i-th window in 'windows' starts at index 'i' in the SLICED data,
        # which corresponds to 'valid_start + i' in the ORIGINAL data.
        original_window = raw_ohlcv[valid_start + i : valid_start + i + seq_len]
        recon_window = reconstructed[i]
        
        # Compare columns 0-3 (OHLC). Volume (index 4) has some eps/log noise, but should be close.
        diff = np.abs(original_window[:, :4] - recon_window[:, :4])
        current_max = np.max(diff)
        max_diff = max(max_diff, current_max)
        
        if current_max > 1e-5:
            print(f"FAILED: Window {i} failed identity check. Max diff: {current_max:.8f}")
            print(f"   Original (first 2 steps):\n{original_window[:2, :4]}")
            print(f"   Reconstructed (first 2 steps):\n{recon_window[:2, :4]}")
            all_passed = False
            break

    if all_passed:
        print(f"SUCCESS: All {n_test} windows are numerically identical to original values.")
        print(f"   Maximum observed difference: {max_diff:.2e}")
        
    # Check Volume specifically (since it uses expm1 type logic)
    vol_diff = np.abs(raw_ohlcv[valid_start : valid_start + n_test, 4] - reconstructed[:, 0, 4])
    print(f"   Volume Max Diff (first step of windows): {np.max(vol_diff):.2e}")

if __name__ == "__main__":
    test_numerical_identity()
