
import sys
import os
import torch
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.loaders import load_stocks_data
from data.module import WaveletTimeSeriesDataModule

def verify_ohlcv_identity():
    from utils.config import INTERNAL_DEFAULTS
    
    # 1. Generate Synthetic Data
    rng = np.random.default_rng(42)
    n_bars = 500 
    close = np.cumprod(1.0 + rng.normal(0.0, 0.01, n_bars)) * 100.0
    open_p = close * 0.99
    high_p = close * 1.01
    low_p = close * 0.98
    volume = rng.lognormal(15.0, 0.5, n_bars)
    
    data = np.stack([open_p, high_p, low_p, close, volume], axis=1)
    df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "Date", pd.date_range("2020-01-01", periods=n_bars))
    
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        csv_path = f.name
        
    try:
        config = {**INTERNAL_DEFAULTS}
        config['dataset'] = {'name': 'stocks', 'seq_len': 24}
        config['data'] = {'normalize_data': True, 'data_dir': csv_path}
        config['conditioning'] = {'past_days': 200}
        
        dm = WaveletTimeSeriesDataModule(config=config)
        indices = np.array([10, 50, 100]) # Test multiple windows
        
        # This gives us the pre-wavelet windows (normalized)
        # raw_data_tensor shape is (samples, seq_len, 22)
        norm_windows = dm.raw_data_tensor[indices].numpy()
        
        # Identity Test using DataModule's own reconstruction
        recon_ohlcv = dm._inverse_reparameterize_ohlc(norm_windows, sample_indices=indices)
        
        # Get Ground Truth (aligned with valid_start)
        # valid_start = 260
        vs = 260
        
        results = {}
        cols = ["Open", "High", "Low", "Close", "Volume"]
        for i, idx in enumerate(indices):
            window_results = []
            for t in range(24):
                gt_row = df.iloc[vs + idx + t][["open", "high", "low", "close", "volume"]].values.astype(float)
                recon_row = recon_ohlcv[i, t]
                
                diff = np.abs(gt_row - recon_row)
                rel_diff = diff / (gt_row + 1e-10)
                window_results.append(rel_diff)
            
            results[idx] = np.mean(window_results, axis=0)

        print("-" * 60)
        print(f"{'Metric':<10} | {'Open':<10} | {'High':<10} | {'Low':<10} | {'Close':<10} | {'Volume':<10}")
        print("-" * 60)
        for idx in indices:
            row = results[idx]
            print(f"Window {idx:<3} | {row[0]:.2e} | {row[1]:.2e} | {row[2]:.2e} | {row[3]:.2e} | {row[4]:.2e}")
        
    finally:
        os.unlink(csv_path)

if __name__ == "__main__":
    verify_ohlcv_identity()
