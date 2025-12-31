
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def visualize_evaluation(real_data, generated_data, output_dir):
    """
    Main visualization function.
    """
    print("Generating visualizations...")
    plot_samples(real_data, generated_data, output_dir)
    plot_distributions(real_data, generated_data, output_dir)
    plot_pca(real_data, generated_data, output_dir)
    print(f"Visualizations saved to {output_dir}")
