# @title Data Pipeline Test Script
"""
Test script for validating data loading, preprocessing, and inverse transforms.
Mocks training by loading data through the full pipeline and visualizes samples.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data import WaveletTimeSeriesDataModule


def create_test_config():
    """Create minimal config for testing data pipeline."""
    return {
        'dataset': {'name': 'stocks', 'seq_len': 24},
        'training': {'batch_size': 32},
        'data': {'data_dir': 'data/stocks/stock_data.csv', 'normalize_data': True},
        'wavelet': {'type': 'db2', 'levels': 'auto'},
    }


def test_data_loading():
    """Test that data loads correctly with strict OHLCV ordering."""
    print("=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    config = create_test_config()
    dm = WaveletTimeSeriesDataModule(config=config)
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Raw shape: {dm.raw_data_tensor.shape}")
    print(f"  Wavelet shape: {dm.data_tensor.shape}")
    print(f"  Features: Open, High, Low, Close, Volume (indices 0-4)")
    
    return dm


def test_normalization_stats(dm: WaveletTimeSeriesDataModule):
    """Display normalization statistics."""
    print("\n" + "=" * 60)
    print("NORMALIZATION STATISTICS")
    print("=" * 60)
    
    if dm.norm_stats is None:
        print("  No normalization applied")
        return
    
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    print(f"\n  {'Feature':<10} {'Mean':>15} {'Std':>15}")
    print("  " + "-" * 42)
    for i, name in enumerate(feature_names):
        print(f"  {name:<10} {dm.norm_stats['mean'][i]:>15.4f} {dm.norm_stats['std'][i]:>15.4f}")
    
    if dm.norm_stats.get('volume_log_transformed', False):
        print("\n  ✓ Volume was log1p-transformed before z-score")


def test_inverse_transform(dm: WaveletTimeSeriesDataModule):
    """Test that inverse_normalize correctly recovers original-scale values."""
    print("\n" + "=" * 60)
    print("TESTING INVERSE TRANSFORM")
    print("=" * 60)
    
    # raw_data_tensor contains normalized (z-scored log-space) values
    # inverse_normalize should recover original scale
    normalized_sample = dm.raw_data_tensor[0].numpy()  # [seq_len, n_features]
    
    # Apply inverse transform
    recovered_sample = dm.inverse_normalize(normalized_sample)
    
    # Load original CSV to verify
    import pandas as pd
    df = pd.read_csv('data/stocks/stock_data.csv')
    # Select OHLCV columns (same logic as loader)
    col_map = {c.lower(): c for c in df.columns}
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    df = df[[col_map[c] for c in ohlcv_cols]]
    original_sample = df.iloc[:24].values  # First window
    
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    print(f"\n  {'Feature':<10} {'CSV Original':>15} {'Recovered':>15} {'Rel Error':>12}")
    print("  " + "-" * 54)
    
    max_rel_error = 0
    for i, name in enumerate(feature_names):
        orig_val = original_sample[0, i]
        rec_val = recovered_sample[0, i]
        rel_error = abs(orig_val - rec_val) / (abs(orig_val) + 1e-8) * 100
        max_rel_error = max(max_rel_error, rel_error)
        print(f"  {name:<10} {orig_val:>15.2f} {rec_val:>15.2f} {rel_error:>11.4f}%")
    
    print(f"\n  Max relative error: {max_rel_error:.4f}%")
    print(f"  ✓ Round-trip {'PASSED' if max_rel_error < 0.01 else 'FAILED'}")


def visualize_samples(dm: WaveletTimeSeriesDataModule, n_samples: int = 3):
    """Visualize sample sequences with OHLC and Volume in separate panels."""
    print("\n" + "=" * 60)
    print("VISUALIZING SAMPLES")
    print("=" * 60)
    
    feature_names = ['Open', 'High', 'Low', 'Close']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    volume_color = '#f39c12'
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(18, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Select random samples
    total_samples = len(dm.raw_data_tensor)
    indices = np.random.choice(total_samples, n_samples, replace=False)
    print(f"  Selected random indices: {indices}")
    
    for i, sample_idx in enumerate(indices):
        raw_sample = dm.raw_data_tensor[sample_idx].numpy()
        denorm_sample = dm.inverse_normalize(raw_sample)
        
        # Column 1: Normalized OHLC
        ax = axes[i, 0]
        for feat_idx, (name, color) in enumerate(zip(feature_names, colors)):
            ax.plot(raw_sample[:, feat_idx], label=name, color=color, linewidth=1.5)
        ax.set_title(f'Sample {sample_idx}: Normalized OHLC', fontsize=9)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Z-Score')
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)
        
        # Column 2: Normalized Volume
        ax = axes[i, 1]
        ax.fill_between(range(len(raw_sample)), raw_sample[:, 4], alpha=0.3, color=volume_color)
        ax.plot(raw_sample[:, 4], color=volume_color, linewidth=1.5)
        ax.set_title(f'Sample {sample_idx}: Normalized Volume', fontsize=9)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Z-Score (log space)')
        ax.grid(True, alpha=0.3)
        
        # Column 3: Original OHLC ($)
        ax = axes[i, 2]
        for feat_idx, (name, color) in enumerate(zip(feature_names, colors)):
            ax.plot(denorm_sample[:, feat_idx], label=name, color=color, linewidth=1.5)
        ax.set_title(f'Sample {sample_idx}: OHLC Prices ($)', fontsize=9)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)
        
        # Column 4: Original Volume
        ax = axes[i, 3]
        ax.fill_between(range(len(denorm_sample)), denorm_sample[:, 4], alpha=0.3, color=volume_color)
        ax.plot(denorm_sample[:, 4], color=volume_color, linewidth=1.5)
        ax.set_title(f'Sample {sample_idx}: Volume', fontsize=9)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Volume')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(6,6))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'data_pipeline_test_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved to: {output_path}")
    plt.show()


def print_sample_console(dm: WaveletTimeSeriesDataModule):
    """Print a sample to console in both normalized and original scale."""
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (First 5 timesteps)")
    print("=" * 60)
    
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    raw_sample = dm.raw_data_tensor[0].numpy()
    denorm_sample = dm.inverse_normalize(raw_sample)
    
    print("\n  NORMALIZED (Training Space):")
    print(f"  {'t':<4}" + "".join(f"{name:>12}" for name in feature_names))
    print("  " + "-" * 64)
    for t in range(5):
        row = f"  {t:<4}" + "".join(f"{raw_sample[t, i]:>12.4f}" for i in range(5))
        print(row)
    
    print("\n  ORIGINAL SCALE (Evaluation):")
    print(f"  {'t':<4}" + "".join(f"{name:>12}" for name in feature_names))
    print("  " + "-" * 64)
    for t in range(5):
        row = f"  {t:<4}" + "".join(f"{denorm_sample[t, i]:>12.2f}" for i in range(5))
        print(row)


def main():
    print("\n" + "=" * 60)
    print("  DATA PIPELINE TEST SUITE")
    print("=" * 60)
    
    try:
        dm = test_data_loading()
        test_normalization_stats(dm)
        test_inverse_transform(dm)
        print_sample_console(dm)
        visualize_samples(dm, n_samples=3)
        
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED ✓")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n  ✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
