
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
COLORS = {"Real": "#d62728", "Generated": "#1f77b4"}

def plot_distribution_reduction(real, generated, n_samples=1000):
    """
    Plot t-SNE and PCA visualizations of real vs generated data.
    
    Args:
        real: Real data array (N, T, D)
        generated: Generated data array (N, T, D)
        n_samples: Number of samples to use for visualization
    """
    # Flatten time series for t-SNE (Standard approach in TimeGAN/Diffusion-TS)
    # Shape: (N, T*D)
    n_samples = min(n_samples, len(real), len(generated))
    
    real_flat = real[:n_samples].reshape(n_samples, -1)
    gen_flat = generated[:n_samples].reshape(n_samples, -1)
    
    # Concatenate
    data = np.concatenate([real_flat, gen_flat], axis=0)
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    
    # Plot
    plt.figure(figsize=(16, 6))
    
    # t-SNE Plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=tsne_results[:n_samples, 0], y=tsne_results[:n_samples, 1], 
                    color=COLORS["Real"], alpha=0.3, label="Real", s=20)
    sns.scatterplot(x=tsne_results[n_samples:, 0], y=tsne_results[n_samples:, 1],
                    color=COLORS["Generated"], alpha=0.3, label="Generated", s=20)
    plt.title("t-SNE Visualization")
    plt.legend()
    
    # PCA Plot (for variance check)
    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data)
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=pca_results[:n_samples, 0], y=pca_results[:n_samples, 1], 
                    color=COLORS["Real"], alpha=0.3, label="Real", s=20)
    sns.scatterplot(x=pca_results[n_samples:, 0], y=pca_results[n_samples:, 1],
                    color=COLORS["Generated"], alpha=0.3, label="Generated", s=20)
    plt.title("PCA Visualization")
    plt.legend()
    
    plt.show()

def plot_pdf(real, generated):
    """
    Plot Probability Density Function (KDE) of all data values.
    """
    plt.figure(figsize=(10, 6))
    
    # Flatten all data to compare value distributions
    sns.kdeplot(real.flatten(), fill=True, color=COLORS["Real"], label="Real", alpha=0.3)
    sns.kdeplot(generated.flatten(), fill=True, color=COLORS["Generated"], label="Generated", alpha=0.3)
    
    plt.title("Probability Density Function (All Values)")
    plt.xlabel("Data Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_samples(real, generated, n_samples=5):
    """
    Plot side-by-side comparison of real and generated time series samples.
    """
    # Assuming shape (N, T, D)
    # We exclude the last feature if it is volume (assuming > 1 feature)
    n_features = real.shape[2]
    plot_features = n_features - 1 if n_features > 1 else n_features
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 4, 6), sharey=True)
    
    for i in range(n_samples):
        # Real Samples
        for f in range(plot_features):
            axes[0, i].plot(real[i, :, f], alpha=0.8)
        axes[0, i].set_title(f"Real Sample {i}")
        if i == 0: axes[0, i].set_ylabel("Value (MinMax Scaled)")
        
        # Generated Samples
        for f in range(plot_features):
            axes[1, i].plot(generated[i, :, f], alpha=0.8)
        axes[1, i].set_title(f"Gen Sample {i}")
        if i == 0: axes[1, i].set_ylabel("Value (MinMax Scaled)")
    
    # Create a dummy legend
    from matplotlib.lines import Line2D
    lines = [Line2D([0], [0], color=f"C{i}", lw=2) for i in range(plot_features)]
    fig.legend(lines, [f"Feature {i}" for i in range(plot_features)], loc='lower center', ncol=plot_features)
    
    plt.tight_layout()
    print(f"Note: Showing {plot_features} features (excluding last/volume feature if D>1)")
    plt.show()
