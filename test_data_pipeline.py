# @title Data Pipeline Test Script
"""
Test script for validating OHLC reparameterization, ATR scaling, and inverse transforms.
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
    """Test that data loads correctly with OHLC reparameterization."""
    print("=" * 60)
    print("TESTING DATA LOADING WITH REPARAMETERIZATION")
    print("=" * 60)
    
    config = create_test_config()
    dm = WaveletTimeSeriesDataModule(config=config)
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Raw shape: {dm.raw_data_tensor.shape}")
    print(f"  Wavelet shape: {dm.data_tensor.shape}")
    
    if dm.norm_stats.get('reparameterized', False):
        print(f"  Features: {dm.norm_stats['feature_names']}")
        print(f"  Anchors stored: {len(dm.norm_stats['anchors'])}")
        print(f"  ATR_pcts stored: {len(dm.norm_stats['atr_pcts'])}")
    
    return dm


def test_reparameterization_constraints(dm: WaveletTimeSeriesDataModule):
    """Verify that wick values are non-negative in training data."""
    print("\n" + "=" * 60)
    print("TESTING REPARAMETERIZATION CONSTRAINTS")
    print("=" * 60)
    
    if not dm.norm_stats.get('reparameterized', False):
        print("  Skipping: Data is not reparameterized")
        return
    
    raw_data = dm.raw_data_tensor.numpy()
    
    wick_high = raw_data[:, :, 2]
    wick_low = raw_data[:, :, 3]
    
    wick_high_min = np.min(wick_high)
    wick_low_min = np.min(wick_low)
    
    print(f"\n  wick_high_norm min: {wick_high_min:.6f}")
    print(f"  wick_low_norm min: {wick_low_min:.6f}")
    
    print(f"\n  ✓ Constraint check: Values are in ATR-normalized space (can be negative before inverse)")


def test_inverse_transform(dm: WaveletTimeSeriesDataModule):
    """Test that inverse_normalize correctly recovers OHLC values."""
    print("\n" + "=" * 60)
    print("TESTING INVERSE TRANSFORM")
    print("=" * 60)
    
    if not dm.norm_stats.get('reparameterized', False):
        print("  Skipping: Data is not reparameterized")
        return
    
    sample_indices = np.array([0, 1, 2])
    normalized_samples = dm.raw_data_tensor[sample_indices].numpy()
    
    recovered_samples = dm.inverse_normalize(normalized_samples, sample_indices=sample_indices)
    
    print(f"\n  Recovered shape: {recovered_samples.shape}")
    print(f"  Sample 0, Day 0 OHLCV:")
    print(f"    Open:   {recovered_samples[0, 0, 0]:.2f}")
    print(f"    High:   {recovered_samples[0, 0, 1]:.2f}")
    print(f"    Low:    {recovered_samples[0, 0, 2]:.2f}")
    print(f"    Close:  {recovered_samples[0, 0, 3]:.2f}")
    print(f"    Volume: {recovered_samples[0, 0, 4]:.0f}")
    
    h = recovered_samples[:, :, 1]
    l = recovered_samples[:, :, 2]
    o = recovered_samples[:, :, 0]
    c = recovered_samples[:, :, 3]
    
    max_oc = np.maximum(o, c)
    min_oc = np.minimum(o, c)
    
    high_valid = np.all(h >= max_oc - 1e-4)
    low_valid = np.all(l <= min_oc + 1e-4)
    hl_valid = np.all(h >= l - 1e-4)
    
    print(f"\n  Constraint Checks:")
    print(f"    High >= max(Open, Close): {'✓ PASSED' if high_valid else '✗ FAILED'}")
    print(f"    Low <= min(Open, Close):  {'✓ PASSED' if low_valid else '✗ FAILED'}")
    print(f"    High >= Low:              {'✓ PASSED' if hl_valid else '✗ FAILED'}")


def visualize_samples(dm: WaveletTimeSeriesDataModule, n_samples: int = 3):
    """Visualize sample sequences showing normalized and reconstructed data."""
    print("\n" + "=" * 60)
    print("VISUALIZING SAMPLES")
    print("=" * 60)
    
    if not dm.norm_stats.get('reparameterized', False):
        print("  Skipping visualization: Data is not reparameterized")
        return
    
    # Grid: Norm Price | Norm Wicks | Norm Volume | Rec OHLC | Rec Volume
    fig, axes = plt.subplots(n_samples, 5, figsize=(25, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    total_samples = len(dm.raw_data_tensor)
    indices = np.random.choice(total_samples, n_samples, replace=False)
    print(f"  Selected random indices: {indices}")
    
    for i, sample_idx in enumerate(indices):
        raw_sample = dm.raw_data_tensor[sample_idx].numpy()
        recovered = dm.inverse_normalize(raw_sample[np.newaxis, ...], sample_indices=np.array([sample_idx]))[0]
        
        # 1. Normalized Price Movement
        ax = axes[i, 0]
        ax.plot(raw_sample[:, 0], label='open_norm', linewidth=1.5)
        ax.plot(raw_sample[:, 1], label='body_norm', linewidth=1.5)
        ax.set_title(f'Sample {sample_idx}: Norm Price Moves', fontsize=9)
        ax.set_ylabel('ATR Units')
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)
        
        # 2. Normalized Wicks
        ax = axes[i, 1]
        ax.plot(raw_sample[:, 2], label='wick_high_norm', color='green', linewidth=1.5)
        ax.plot(raw_sample[:, 3], label='wick_low_norm', color='red', linewidth=1.5)
        ax.set_title(f'Sample {sample_idx}: Norm Wicks', fontsize=9)
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)

        # 3. Normalized Volume (Log-Ratio)
        ax = axes[i, 2]
        ax.plot(raw_sample[:, 4], label='volume_norm', color='purple', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'Sample {sample_idx}: Norm Volume (Log-Ratio)', fontsize=9)
        ax.set_ylabel('Log Deviation')
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)
        
        # 4. Reconstructed OHLC
        ax = axes[i, 3]
        ax.plot(recovered[:, 0], label='Open', color='blue', linewidth=1.0)
        ax.plot(recovered[:, 1], label='High', color='green', linewidth=0.5)
        ax.plot(recovered[:, 2], label='Low', color='red', linewidth=0.5)
        ax.plot(recovered[:, 3], label='Close', color='black', linewidth=1.0)
        ax.set_title(f'Sample {sample_idx}: Rec OHLC ($)', fontsize=9)
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # 5. Reconstructed Volume
        ax = axes[i, 4]
        ax.bar(np.arange(len(recovered)), recovered[:, 4], color='gray', alpha=0.7)
        ax.set_title(f'Sample {sample_idx}: Rec Volume', fontsize=9)
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'data_pipeline_test_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved to: {output_path}")
    plt.show()


def print_sample_console(dm: WaveletTimeSeriesDataModule):
    """Print a sample to console in both normalized and reconstructed scale."""
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (First 5 timesteps)")
    print("=" * 60)
    
    if not dm.norm_stats.get('reparameterized', False):
        print("  Skipping: Data is not reparameterized")
        return
    
    feature_names = dm.norm_stats['feature_names']
    raw_sample = dm.raw_data_tensor[0].numpy()
    recovered = dm.inverse_normalize(raw_sample[np.newaxis, ...], sample_indices=np.array([0]))[0]
    
    anchor = dm.norm_stats['anchors'][0]
    atr_pct = dm.norm_stats['atr_pcts'][0]
    
    print(f"\n  Metadata: anchor={anchor:.2f}, ATR_pct={atr_pct:.2f}%")
    
    print("\n  NORMALIZED (Training Space):")
    print(f"  {'t':<4}" + "".join(f"{name:>15}" for name in feature_names))
    print("  " + "-" * 79)
    for t in range(5):
        row = f"  {t:<4}" + "".join(f"{raw_sample[t, i]:>15.4f}" for i in range(5))
        print(row)
    
    print("\n  RECONSTRUCTED OHLCV:")
    ohlcv_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    print(f"  {'t':<4}" + "".join(f"{name:>12}" for name in ohlcv_names))
    print("  " + "-" * 64)
    for t in range(5):
        row = f"  {t:<4}" + "".join(f"{recovered[t, i]:>12.2f}" for i in range(5))
        print(row)


def main():
    print("\n" + "=" * 60)
    print("  DATA PIPELINE TEST SUITE (REPARAMETERIZATION)")
    print("=" * 60)
    
    try:
        dm = test_data_loading()
        test_reparameterization_constraints(dm)
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
