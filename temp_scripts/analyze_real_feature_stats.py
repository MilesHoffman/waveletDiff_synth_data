
import sys
import os
import torch
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.loaders import load_stocks_data
from data.module import WaveletTimeSeriesDataModule

def analyze_real_feature_stats():
    from utils.config import INTERNAL_DEFAULTS
    
    # generate a slightly longer series to have more statistics
    rng = np.random.default_rng(42)
    n_bars = 2000
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
        
        # indices = np.arange(0, len(dm.raw_data_tensor))
        norm_windows = dm.raw_data_tensor.numpy()
        
        # Features at indices:
        # 0: overnight_gap, 1: intraday_return, 2: total_log_return, 3: normalized_range,
        # 4: log_upper_shadow, 5: log_lower_shadow, 21: vol_log_dev
        feature_indices = [0, 1, 2, 3, 4, 5, 21]
        feature_names = ["overnight_gap", "intraday_return", "total_log_return", "normalized_range", "log_upper_shadow", "log_lower_shadow", "vol_log_dev"]
        
        print("-" * 110)
        print(f"{'Feature':<18} | {'Min':<10} | {'Max':<10} | {'Median':<10} | {'1th Pct':<10} | {'99th Pct':<10}")
        print("-" * 110)
        
        for idx, name in zip(feature_indices, feature_names):
            vals = norm_windows[:, :, idx].flatten()
            min_v = np.min(vals)
            max_v = np.max(vals)
            med_v = np.median(vals)
            p1 = np.percentile(vals, 1)
            p99 = np.percentile(vals, 99)
            print(f"{name:<18} | {min_v:>10.4f} | {max_v:>10.4f} | {med_v:>10.4f} | {p1:>10.4f} | {p99:>10.4f}")
            
    finally:
        os.unlink(csv_path)

if __name__ == "__main__":
    analyze_real_feature_stats()
