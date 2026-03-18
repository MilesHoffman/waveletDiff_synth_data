
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
    n_bars = 400
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
        
        dm = WaveletTimeSeriesDataModule(config=config)
        indices = np.array([0])
        norm_features = dm.raw_data_tensor[indices].numpy()
        
        # Check last step of window 0 (where anchor comes from)
        s = 23
        feat_vol_log_dev = norm_features[0, s, 21]
        
        tf = dm.norm_stats['transformers']['vol_log_dev']
        vol_log_dev_raw = tf.inverse_transform(feat_vol_log_dev.reshape(-1, 1)).item()
        
        med = dm.norm_stats['vol_medians'][0]
        iqr = dm.norm_stats['vol_iqrs'][0]
        
        log_vol_recon = (vol_log_dev_raw * iqr) + med
        vol_recon = np.exp(log_vol_recon) - 1e-10
        
        # GT Volume
        gt_vol = df.iloc[220 + s]['volume']
        
        print(f"Step {s} (Anchor):")
        print(f"GT Volume:    {gt_vol:.12f}")
        print(f"Recon Volume: {vol_recon:.12f}")
        print(f"Diff:         {abs(gt_vol - vol_recon):.12f}")
        print(f"Rel Diff:     {abs(gt_vol - vol_recon)/gt_vol:.12e}")
        
    finally:
        os.unlink(csv_path)

if __name__ == "__main__":
    debug_volume()
