
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.loaders import load_stocks_data
from data.module import WaveletTimeSeriesDataModule

def debug_volume():
    from utils.config import INTERNAL_DEFAULTS
    import tempfile
    import pandas as pd

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
        indices = np.array([0]) 
        norm_features = dm.raw_data_tensor[indices].numpy()
        
        # Verify all 24 steps
        med = dm.norm_stats['vol_medians'][0]
        iqr = dm.norm_stats['vol_iqrs'][0]
        valid_start = 260
        
        max_diff = 0
        for s in range(24):
            feat_vol_log_dev = norm_features[0, s, 21]
            log_vol_recon = (feat_vol_log_dev * iqr) + med
            vol_recon = np.exp(log_vol_recon) - 1e-10
            
            gt_vol = df.iloc[valid_start + s]['volume']
            diff = abs(gt_vol - vol_recon)
            rel_diff = diff / gt_vol
            max_diff = max(max_diff, rel_diff)
            
            if s % 5 == 0:
                print(f"Step {s}: Rel Diff = {rel_diff:.2e}")
                
        print(f"\nFinal Window Verification:")
        print(f"Max Relative Difference: {max_diff:.2e}")
        
        if max_diff < 1e-6:
            print("SUCCESS: Volume reconstruction is numerically identical across the sequence.")
        else:
            print("FAILURE: Drift detected in sequence reconstruction.")

    finally:
        os.unlink(csv_path)

if __name__ == "__main__":
    debug_volume()
