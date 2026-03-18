
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.loaders import load_stocks_data

def analyze_scaling(data_path):
    print(f"Analyzing scaling for {data_path}...")
    
    # Load data using the SOTA loader
    try:
        data_tensor, norm_stats = load_stocks_data(data_path, seq_len=24, past_days=200)
        # Handle if data_tensor is a torch.Tensor
        if hasattr(data_tensor, 'numpy'):
            features = data_tensor.numpy()
        else:
            features = np.array(data_tensor)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading data: {e}")
        return
    feature_names = norm_stats['feature_names']
    
    print("\nFeature Statistics:")
    print(f"{'Feature':<20} | {'Mean':>10} | {'Std':>10} | {'Med':>10} | {'IQR':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 95)
    
    for i, name in enumerate(feature_names):
        vals = features[:, :, i].flatten()
        mean = np.mean(vals)
        std = np.std(vals)
        med = np.median(vals)
        q75, q25 = np.percentile(vals, [75, 25])
        iqr = q75 - q25
        fmin = np.min(vals)
        fmax = np.max(vals)
        
        print(f"{name[:20]:<20} | {mean:10.4f} | {std:10.4f} | {med:10.4f} | {iqr:10.4f} | {fmin:10.4f} | {fmax:10.4f}")

    if 'path_signatures' in norm_stats:
        sigs = norm_stats['path_signatures']
        print("\nPath Signature Statistics (Overall):")
        print(f"Shape: {sigs.shape}")
        print(f"Mean: {np.mean(sigs):.4f}")
        print(f"Std:  {np.std(sigs):.4f}")
        print(f"Min:  {np.min(sigs):.4f}")
        print(f"Max:  {np.max(sigs):.4f}")
        
        # Check per-dimension variance
        sig_std = np.std(sigs, axis=0)
        print(f"Per-dim Std Range: [{np.min(sig_std):.4f}, {np.max(sig_std):.4f}]")
        
        # Count high-variance dimensions
        high_var = np.sum(sig_std > 10.0)
        print(f"Dimensions with Std > 10: {high_var}")
        
        if high_var > 0:
            idx = np.argmax(sig_std)
            print(f"Max Std dim {idx}: {sig_std[idx]:.4f}")

if __name__ == "__main__":
    # Look for stock data
    data_dir = "data"
    spy_path = os.path.join(data_dir, "stocks", "SPY_stock_data_train.csv")
    if os.path.exists(spy_path):
        analyze_scaling(spy_path)
    else:
        print(f"Could not find {spy_path}")
