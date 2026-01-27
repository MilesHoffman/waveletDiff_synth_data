
import numpy as np
from scipy.stats import skew, kurtosis

def calculate_statistical_metrics(real, generated):
    """
    Calculate and display comprehensive statistical metrics for Real vs Generated data.
    
    Args:
        real: Real data array (N, T, D)
        generated: Generated data array (N, T, D)
        
    Returns:
        tuple: (aggregate_metrics, real_stats_dict, gen_stats_dict)
    """
    # Flatten over samples and time to get distribution of values per feature
    r_flat = real.reshape(-1, real.shape[2])
    g_flat = generated.reshape(-1, generated.shape[2])
    n_features = real.shape[2]
    
    # Calculate per-feature statistics
    real_stats = {
        "Mean": np.mean(r_flat, axis=0),
        "Std": np.std(r_flat, axis=0),
        "Skewness": skew(r_flat, axis=0),
        "Kurtosis": kurtosis(r_flat, axis=0),
        "Min": np.min(r_flat, axis=0),
        "Max": np.max(r_flat, axis=0),
    }
    
    gen_stats = {
        "Mean": np.mean(g_flat, axis=0),
        "Std": np.std(g_flat, axis=0),
        "Skewness": skew(g_flat, axis=0),
        "Kurtosis": kurtosis(g_flat, axis=0),
        "Min": np.min(g_flat, axis=0),
        "Max": np.max(g_flat, axis=0),
    }
    
    # Print comparison table
    print("=" * 80)
    print("STATISTICAL COMPARISON: Real vs Generated Data")
    print("=" * 80)
    
    # Per-feature comparison
    for f in range(n_features):
        print(f"\n--- Feature {f} ---")
        print(f"{'Metric':<15} {'Real':>12} {'Generated':>12} {'Diff (Abs)':>12}")
        print("-" * 55)
        for stat_name in real_stats:
            r_val = real_stats[stat_name][f]
            g_val = gen_stats[stat_name][f]
            diff = abs(r_val - g_val)
            print(f"{stat_name:<15} {r_val:>12.4f} {g_val:>12.4f} {diff:>12.4f}")
    
    # Aggregate metrics (averaged across features)
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS (Averaged Across All Features)")
    print("=" * 80)
    
    aggregate = {
        "Mean MAE": np.mean(np.abs(real_stats["Mean"] - gen_stats["Mean"])),
        "Std MAE": np.mean(np.abs(real_stats["Std"] - gen_stats["Std"])),
        "Skewness MAE": np.mean(np.abs(real_stats["Skewness"] - gen_stats["Skewness"])),
        "Kurtosis MAE": np.mean(np.abs(real_stats["Kurtosis"] - gen_stats["Kurtosis"])),
    }
    
    print(f"{'Metric':<20} {'Value':>12} {'Interpretation'}")
    print("-" * 60)
    for k, v in aggregate.items():
        # Interpretation thresholds
        quality = "✅ Good" if v < 0.05 else ("⚠️ Moderate" if v < 0.15 else "❌ High")
        print(f"{k:<20} {v:>12.6f} {quality}")
    
    return aggregate, real_stats, gen_stats
