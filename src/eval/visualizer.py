import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Local imports
try:
    from src.eval import visualizations
    from src.eval.metrics import BASELINE_STOCKS, INTERPRETATION
except ImportError:
    # relative import fallback if running as package
    from . import visualizations
    from .metrics import BASELINE_STOCKS, INTERPRETATION

def plot_samples(real_data, generated_data, output_dir, n_samples=5):
    """
    Plot random samples of real and generated data.
    """
    # Create the plot directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure n_samples isn't larger than dataset
    n_samples = min(n_samples, len(real_data), len(generated_data))
    
    # Select random indices
    real_indices = np.random.choice(len(real_data), n_samples, replace=False)
    gen_indices = np.random.choice(len(generated_data), n_samples, replace=False)
    
    real_samples = real_data[real_indices]
    gen_samples = generated_data[gen_indices]
    
    # Setup subplots
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3*n_samples))
    
    features = ["Open", "High", "Low", "Close", "Volume"]
    if real_samples.shape[-1] != len(features):
        features = [f"Feature {i}" for i in range(real_samples.shape[-1])]
    
    for i in range(n_samples):
        # Real
        for feat_idx in range(real_samples.shape[-1]):
            axes[i, 0].plot(real_samples[i, :, feat_idx], label=features[feat_idx])
        axes[i, 0].set_title("Real Sample")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Generated
        for feat_idx in range(gen_samples.shape[-1]):
            axes[i, 1].plot(gen_samples[i, :, feat_idx], label=features[feat_idx])
        axes[i, 1].set_title("Generated Sample")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_comparison.png"))
    plt.close()

def plot_distributions(real_data, generated_data, output_dir):
    """
    Plot feature distributions (histograms).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_features = real_data.shape[-1]
    features = ["Open", "High", "Low", "Close", "Volume"]
    if n_features != len(features):
        features = [f"Feature {i}" for i in range(n_features)]
        
    fig, axes = plt.subplots(1, n_features, figsize=(4*n_features, 4))
    if n_features == 1:
        axes = [axes]
        
    for i in range(n_features):
        sns.kdeplot(real_data[:, :, i].flatten(), ax=axes[i], label='Real', fill=True, alpha=0.3)
        sns.kdeplot(generated_data[:, :, i].flatten(), ax=axes[i], label='Generated', fill=True, alpha=0.3)
        axes[i].set_title(features[i])
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_comparison.png"))
    plt.close()

def plot_pca(real_data, generated_data, output_dir, n_samples=1000):
    """
    Plot PCA visualization of the data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten data: (N, T, D) -> (N, T*D)
    n_real = min(n_samples, len(real_data))
    n_gen = min(n_samples, len(generated_data))
    
    real_f = real_data[:n_real].reshape(n_real, -1)
    gen_f = generated_data[:n_gen].reshape(n_gen, -1)
    
    # Combine
    combined = np.concatenate([real_f, gen_f], axis=0)
    labels = np.array([0] * n_real + [1] * n_gen)
    
    # PCA
    pca = PCA(n_components=2)
    embedded = pca.fit_transform(combined)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embedded[labels==0, 0], embedded[labels==0, 1], alpha=0.5, label='Real', s=10)
    plt.scatter(embedded[labels==1, 0], embedded[labels==1, 1], alpha=0.5, label='Generated', s=10)
    plt.title("PCA Projection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "pca_projection.png"))
    plt.close()

def display_metrics_comparison(json_path):
    """
    Display a comparison table of computed metrics vs Reference (Stocks).
    """
    if not os.path.exists(json_path):
        print(f"Metrics file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        results = json.load(f)

    print("\n" + "="*80)
    print(f"{'METRIC':<20} | {'YOUR SCORE':<15} | {'BASELINE (STOCKS)':<20} | {'STATUS':<10}")
    print("-" * 80)
    
    for key in BASELINE_STOCKS:
        # Match keys from json (might be 'discriminative' or 'discriminative_score'?)
        # metrics.py uses 'discriminative', 'predictive'...
        if key in results:
            score = results[key]
            base = BASELINE_STOCKS[key]
            diff = score - base
            # Interpretation: Good if <= base + tolerance. 
            # Tolerance 0.005 for solid "GOOD", 0.02 for "OK"
            status = "GOOD" if diff <= 0.005 else ("OK" if diff <= 0.02 else "BAD")
            
            print(f"{key:<20} | {score:<15.4f} | {base:<20.4f} | {status:<10}")
            
    print("="*80)
    
    # Stylized Facts
    if 'real_kurtosis' in results:
         print(f"\nStylized Facts:\n  Kurtosis: Real={results['real_kurtosis']:.2f}, Syn={results['syn_kurtosis']:.2f}")
         print(f"  Leverage: Real={results['real_leverage']:.2f}, Syn={results['syn_leverage']:.2f}")

def visualize_evaluation(real_data, generated_data, output_dir, eval_results_path=None):
    """
    Main visualization function.
    Args:
        real_data: (N, T, D)
        generated_data: (N, T, D)
        output_dir: Save path
        eval_results_path: Optional path to .json results for display
    """
    print("Generating visualizations...")
    
    # 1. Standard Samples
    plot_samples(real_data, generated_data, output_dir)
    
    # 2. Distributions (Exclude Volume - assumed last index if dim > 4)
    # Heuristic: Find feature with largest range -> likely volume
    max_vals = np.max(real_data, axis=(0,1))
    vol_idx = np.argmax(max_vals)
    # Only exclude if it looks like volume (orders of magnitude larger)
    # If all similar scale (normalized?), then don't exclude.
    # Users data passed here is likely RAW (unnormalized) because plot_samples uses it.
    # So max_val check is valid.
    exclude_indices = [vol_idx]
    
    # Use my new visualizations module for distributions to support exclusion
    # But for backward compat/consistency with existing files, I'll update existing or call new.
    # I'll call NEW module for cleaner code.
    features = ["Open", "High", "Low", "Close", "Volume"]
    if real_data.shape[-1] != len(features):
        features = [f"Feature {i}" for i in range(real_data.shape[-1])]
        
    visualizations.plot_distributions(real_data, generated_data, 
                                     feature_names=features, 
                                     exclude_indices=exclude_indices,
                                     save_path=os.path.join(output_dir, "distribution_comparison.png"))
    
    # 3. t-SNE (New)
    visualizations.plot_tsne(real_data, generated_data, save_path=os.path.join(output_dir, "tsne.png"))
    
    # 4. Stylized Facts Visuals
    # Scalogram (Feature 0 - Close?)
    visualizations.plot_scalogram(real_data[0, :, 3], generated_data[0, :, 3], # Index 3 is Close usually? 
                                  # "Open, High, Low, Close, Volume" -> Index 3 is Close.
                                  # If not 5 features, use 0.
                                  save_path=os.path.join(output_dir, "scalogram.png"))
                                  
    visualizations.plot_qq(real_data, generated_data, feature_idx=3, save_path=os.path.join(output_dir, "qq_plot.png"))
    visualizations.plot_acf(real_data, generated_data, feature_idx=3, save_path=os.path.join(output_dir, "acf_plot.png"))
    visualizations.plot_psd(real_data, generated_data, feature_idx=3, save_path=os.path.join(output_dir, "psd_plot.png"))

    print(f"Visualizations saved to {output_dir}")
    
    if eval_results_path is None:
        # Try to find the latest eval_results JSON in output_dir
        import glob
        json_files = glob.glob(os.path.join(output_dir, "eval_results_*.json"))
        if json_files:
            eval_results_path = max(json_files, key=os.path.getctime)
            
    if eval_results_path and os.path.exists(eval_results_path):
        display_metrics_comparison(eval_results_path)
