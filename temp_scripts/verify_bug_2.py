
import sys
import os
import pandas as pd
import numpy as np
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.loaders import load_stocks_data

def verify_header_bug():
    # 1. Create a CSV with lowercase "open"
    n_days = 300
    data = {
        'open': np.random.randn(n_days) + 100,
        'high': np.random.randn(n_days) + 101,
        'low': np.random.randn(n_days) + 99,
        'close': np.random.randn(n_days) + 100,
        'volume': np.random.lognormal(15, 0.5, n_days)
    }
    df = pd.DataFrame(data)
    df.insert(0, 'Date', pd.date_range('2020-01-01', periods=n_days))
    
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        csv_path = f.name
        
    try:
        # Load data using the updated loader
        print(f"Testing loader with {csv_path}...")
        # valid_start = max(200, 200 + 60) = 260. We need at least 260 + 24 = 284 days.
        # Original df has 300 rows.
        windows, stats = load_stocks_data(csv_path, seq_len=24, normalize_data=True, past_days=200)
        
        # If the bug was present (and not fixed), it would have printed:
        # "Detected non-numeric first row (metadata/headers), dropping..."
        # And the resulting windows would be shifted or shorter.
        
        # Check alignment: anchor for window 0 should be close[259]
        # (valid_start = 260. anchor is close[valid_start - 1] = close[259])
        anchor_val = stats['anchors'][0]
        gt_anchor = df.iloc[259]['close']
        
        print(f"Anchor Value: {anchor_val:.6f}")
        print(f"GT Anchor:    {gt_anchor:.6f}")
        
        if abs(anchor_val - gt_anchor) < 1e-10:
            print("SUCCESS: Header correctly detected, no index shift.")
        else:
            print(f"FAILURE: Index shift detected! Diff: {abs(anchor_val - gt_anchor):.6f}")

    finally:
        os.unlink(csv_path)

if __name__ == "__main__":
    verify_header_bug()
