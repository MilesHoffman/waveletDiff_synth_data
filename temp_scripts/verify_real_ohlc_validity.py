
import sys
import os
import torch
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.loaders import load_stocks_data
from data.module import WaveletTimeSeriesDataModule

def verify_real_ohlc_validity():
    from utils.config import INTERNAL_DEFAULTS
    
    # 1. Generate Synthetic Data
    rng = np.random.default_rng(42)
    n_bars = 500 
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    close = np.cumprod(1.0 + rng.normal(0.0, 0.01, n_bars)) * 100.0
    open_p = close * (1.0 + rng.normal(0.0, 0.005, n_bars))
    high_p = np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n_bars)))
    low_p = np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n_bars)))
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
        
        dm = WaveletTimeSeriesDataModule(config=config)
        
        # Test 100 windows
        indices = np.arange(0, 100)
        norm_windows = dm.raw_data_tensor[indices].numpy()
        
        # Identity Test
        ohlcv = dm.inverse_normalize(norm_windows, sample_indices=indices)
        open_p, high_p, low_p, close_p = ohlcv[..., 0], ohlcv[..., 1], ohlcv[..., 2], ohlcv[..., 3]
        
        eps = 1e-7
        valid = (
            (high_p >= open_p  - eps) & (high_p >= close_p - eps) &
            (low_p  <= open_p  + eps) & (low_p  <= close_p + eps) &
            (high_p >= low_p   - eps)
        )
        
        valid_pct = np.mean(valid) * 100.0
        print(f"Real Data OHLC Valid Pct: {valid_pct:.2f}%")
        
        if valid_pct == 100.0:
            print("SUCCESS: Real data has 100% validity. The evaluation script is correct.")
        else:
            print("FAILURE: Real data is invalid! There is a bug in reconstruction or evaluation.")
        
    finally:
        os.unlink(csv_path)

if __name__ == "__main__":
    verify_real_ohlc_validity()
