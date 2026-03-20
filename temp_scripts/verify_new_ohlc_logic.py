import sys
import os
import numpy as np
import pandas as pd
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from data.loaders import load_stocks_data
from data.module import WaveletTimeSeriesDataModule

def verify_new_ohlc_logic():
    print("=== Step 1: Real Data Identity Check (Reparam -> Inverse) ===")
    
    # Load a small snippet of real data
    data_dir = "data/stocks/SPY_stock_data.csv"
    if not os.path.exists(data_dir):
        # Fallback for different environments
        data_dir = "data/stocks/SPY_stock_data.csv"
        
    config = {
        'dataset': {'name': 'stocks', 'seq_len': 24},
        'training': {'batch_size': 32},
        'data': {'data_dir': 'data', 'normalize_data': True},
        'wavelet': {'type': 'db4', 'levels': 3},
        'conditioning': {'past_days': 200}
    }
    
    # 1. Load and Reparameterize
    print(f"Loading and reparameterizing {data_dir}...")
    windows, norm_stats = load_stocks_data("data", seq_len=24, past_days=200)
    
    # Initialize DataModule to access inverse logic
    dm = WaveletTimeSeriesDataModule(config=config)
    dm.norm_stats = norm_stats # Inject the stats from the loader
    
    # 2. Inverse Transform
    print("Running inverse reparameterization...")
    # Convert windows back to OHLCV
    # windows shape is (N, T, 22)
    reconstructed_ohlcv = dm.inverse_normalize(windows.numpy(), sample_indices=np.arange(len(windows)))
    
    # 3. Compare with original
    # We need the original prices for the windows. 
    # load_stocks_data slices the data, so we need to align the original CSV.
    df = pd.read_csv(data_dir)
    # The valid_start in loader is max(200, past_days + vol_rolling_period) = 260 usually
    valid_start = norm_stats.get('valid_start', 260) 
    # Actually, loaders.py doesn't return valid_start in norm_stats, but we can infer it
    # or just compare the first window's Close with the reconstructed Close.
    
    first_window_recon = reconstructed_ohlcv[0]
    
    # Identity Check
    print("\nIdentity Check (First Window):")
    print(f"{'Step':<5} | {'Type':<6} | {'Original':<10} | {'Recon':<10} | {'Diff':<10}")
    
    # We need the original OHLCV from the CSV to compare.
    # The valid_start is at index 260. Window 0 is [260:260+24].
    # But window 0 uses anchor = Close[259].
    # Let's just compare reconstructed Close[t] with original Close[valid_start+t].
    
    # Try to find valid_start automatically
    # original Close[valid_start-1] should match anchor
    anchor_0 = norm_stats['anchors'][0]
    original_closes = df['Close'].values
    valid_start = np.where(np.abs(original_closes - anchor_0) < 1e-4)[0]
    if len(valid_start) > 0:
        valid_start = valid_start[0] + 1
        print(f"Detected valid_start: {valid_start}")
        original_window = df.iloc[valid_start : valid_start+24][['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for t in range(5): # first 5 timesteps
            for c_idx, col_name in enumerate(cols):
                orig = original_window[t, c_idx]
                recon = first_window_recon[t, c_idx]
                diff = abs(orig - recon)
                print(f"{t:<5} | {col_name:<6} | {orig:<10.4f} | {recon:<10.4f} | {diff:<10.4e}")
            print("-" * 55)
    else:
        print("Could not find anchor in original data, skipping identity table.")
    
    # Check structural validity of reconstructed data
    print("\n=== Step 2: Structural Validity Check (High >= O,C >= Low) ===")
    
    high = reconstructed_ohlcv[..., 1]
    low = reconstructed_ohlcv[..., 2]
    open_ = reconstructed_ohlcv[..., 0]
    close = reconstructed_ohlcv[..., 3]
    
    h_v_o = (high >= open_ - 1e-6).all()
    h_v_c = (high >= close - 1e-6).all()
    l_v_o = (low <= open_ + 1e-6).all()
    l_v_c = (low <= close + 1e-6).all()
    
    print(f"High >= Open:  {h_v_o}")
    print(f"High >= Close: {h_v_c}")
    print(f"Low <= Open:   {l_v_o}")
    print(f"Low <= Close:  {l_v_c}")
    
    if all([h_v_o, h_v_c, l_v_o, l_v_c]):
        print("SUCCESS: All reconstructed candles are structurally valid.")
    else:
        print("FAILURE: Structural violations detected in reconstruction!")

    print("\n=== Step 3: Mock Noise Robustness Check ===")
    print("Generating extreme random noise (-10 to +10) for structural features...")
    
    # Create random noise for a single window (1, 24, 22)
    # Simulate a model failing and outputting wild values
    mock_noise = np.random.uniform(-10, 10, (10, 24, 22))
    
    # Run through inverse
    mock_ohlcv = dm.inverse_normalize(mock_noise, fixed_anchor=100.0)
    
    m_high = mock_ohlcv[..., 1]
    m_low = mock_ohlcv[..., 2]
    m_open = mock_ohlcv[..., 0]
    m_close = mock_ohlcv[..., 3]
    
    mh_v_o = (m_high >= m_open - 1e-6).all()
    mh_v_c = (m_high >= m_close - 1e-6).all()
    ml_v_o = (m_low <= m_open + 1e-6).all()
    ml_v_c = (m_low <= m_close + 1e-6).all()
    
    print(f"Mock High >= Open:  {mh_v_o}")
    print(f"Mock High >= Close: {mh_v_c}")
    print(f"Mock Low <= Open:   {ml_v_o}")
    print(f"Mock Low <= Close:  {ml_v_c}")
    
    if all([mh_v_o, mh_v_c, ml_v_o, ml_v_c]):
        print("SUCCESS: Even with extreme noise, Logit/Sigmoid guarantees structural validity.")
    else:
        print("FAILURE: Structural violations in mock data reconstruction!")

if __name__ == "__main__":
    try:
        verify_new_ohlc_logic()
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
