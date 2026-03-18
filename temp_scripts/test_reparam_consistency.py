
import numpy as np
import torch
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.loaders import load_stocks_data
from data.module import WaveletTimeSeriesDataModule

def test_consistency():
    # Mock config
    config = {
        'dataset': {
            'name': 'stocks',
            'seq_len': 24
        },
        'training': {
            'batch_size': 32
        },
        'data': {
            'data_dir': 'data',
            'normalize_data': True
        },
        'wavelet': {
            'type': 'sym2',
            'levels': 3
        },
        'conditioning': {
            'past_days': 20
        },
        'performance': {
            'matmul_precision': 'medium',
            'precision': '32'
        },
        'paths': {
            'output_dir': 'outputs'
        }
    }

    # Load real data using the SOTA loader
    data_dir = "data"
    try:
        # Load a small subset by overriding the loader's behavior if possible, 
        # or just slicing the output.
        # To make it fast, we can't easily change load_stocks_data without editing it.
        # But we can at least reduce the sequence length or past_days.
        data_tensor, norm_stats = load_stocks_data(data_dir, seq_len=24, past_days=20)
        print(f"Loaded data with shape: {data_tensor.shape}")
        
        # Take a slice of raw data (the windowed features)
        n_test = 5
        windows = data_tensor[:n_test].numpy()
        
        # Initialize DataModule to use its inverse_normalize
        dm = WaveletTimeSeriesDataModule(config=config, data_tensor=data_tensor[:n_test])
        dm.norm_stats = norm_stats
        
        # Reconstruct OHLCV
        reconstructed_ohlcv = dm.inverse_normalize(windows, sample_indices=np.arange(n_test))
        print(f"Reconstructed OHLCV shape: {reconstructed_ohlcv.shape}")
        
        # Check consistency for Open, High, Low, Close (indices 0, 1, 2, 3)
        # We need the original OHLC for these same windows to compare.
        # This is tricky because load_stocks_data doesn't return the raw windows.
        # However, we can check the OHLC invariants at least.
        
        for i in range(n_test):
            ohlc = reconstructed_ohlcv[i]
            # Invariants: High >= Open, Close; Low <= Open, Close; High >= Low
            high_ok = np.all(ohlc[:, 1] >= ohlc[:, 0] - 1e-5) and np.all(ohlc[:, 1] >= ohlc[:, 3] - 1e-5)
            low_ok = np.all(ohlc[:, 2] <= ohlc[:, 0] + 1e-5) and np.all(ohlc[:, 2] <= ohlc[:, 3] + 1e-5)
            range_ok = np.all(ohlc[:, 1] >= ohlc[:, 2] - 1e-5)
            
            print(f"Window {i}: High Invariant: {high_ok}, Low Invariant: {low_ok}, Range Invariant: {range_ok}")
            
            if not (high_ok and low_ok and range_ok):
                print(f"  Failing OHLC values (first few steps):\n{ohlc[:3]}")

    except Exception as e:
        print(f"Error during consistency test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_consistency()
