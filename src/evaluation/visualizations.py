
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

def plot_samples(real, generated, n_samples=5, feature_names=None):
    """
    Plot OHLC and Volume comparison of real and generated samples.
    
    Args:
        real: Real data array (N, T, D) where D is 5 for OHLCV
        generated: Generated data array (N, T, D)
        n_samples: Number of samples to visualize
        feature_names: Optional list of feature names. Defaults to OHLCV.
    """
    n_features = real.shape[2]
    has_volume = n_features >= 5
    n_ohlc = 4 if n_features >= 4 else n_features
    
    if feature_names is None:
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume'][:n_features]
    
    ohlc_colors = {'Open': 'blue', 'High': 'green', 'Low': 'red', 'Close': 'black'}
    
    # Grid: 2 rows (Real/Gen) x (n_samples OHLC columns + n_samples Volume columns if applicable)
    n_cols = n_samples * 2 if has_volume else n_samples
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3.5, 6))
    
    for i in range(n_samples):
        ohlc_col = i
        vol_col = n_samples + i if has_volume else None
        
        for row, (data, label) in enumerate([(real, 'Real'), (generated, 'Gen')]):
            sample = data[i]
            
            # --- OHLC Subplot ---
            ax_ohlc = axes[row, ohlc_col]
            for f_idx in range(n_ohlc):
                name = feature_names[f_idx]
                color = ohlc_colors.get(name, f'C{f_idx}')
                lw = 1.5 if name in ['Open', 'Close'] else 0.8
                ax_ohlc.plot(sample[:, f_idx], label=name, color=color, linewidth=lw)
            
            ax_ohlc.set_title(f'{label} {i}: OHLC ($)', fontsize=9)
            ax_ohlc.set_ylabel('Price')
            ax_ohlc.grid(True, alpha=0.3)
            if row == 0 and i == 0:
                ax_ohlc.legend(loc='upper right', fontsize=7)
            
            # --- Volume Subplot (if applicable) ---
            if has_volume and vol_col is not None:
                ax_vol = axes[row, vol_col]
                ax_vol.plot(sample[:, 4], color='purple', linewidth=1.5)
                ax_vol.set_title(f'{label} {i}: Volume', fontsize=9)
                ax_vol.set_ylabel('Volume')
                ax_vol.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_candlesticks(real, generated, n_samples=5):
    """
    Plot OHLC candlestick comparison of real and generated samples.
    
    Args:
        real: Real data array (N, T, 5) for OHLCV
        generated: Generated data array (N, T, 5)
        n_samples: Number of samples to visualize
    """
    n_samples = min(n_samples, len(real), len(generated))
    
    # 2 rows (Real vs Gen) x n_samples
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 4, 8), sharey='row')
    
    for row, (data, label) in enumerate([(real, 'Real'), (generated, 'Generated')]):
        for i in range(n_samples):
            ax = axes[row, i]
            sample = data[i] # (T, 5)
            
            # Use the time index as X
            t = np.arange(len(sample))
            
            # Plot wires (High-Low)
            ax.vlines(t, sample[:, 2], sample[:, 1], color='black', linestyle='-', linewidth=1, alpha=0.6)
            
            # Determine color for bodies (Open vs Close)
            # Bullish: Close >= Open (usually Green)
            # Bearish: Close < Open (usually Red)
            bullish = sample[:, 3] >= sample[:, 0]
            
            # Plot Bullish bodies
            ax.vlines(t[bullish], sample[bullish, 0], sample[bullish, 3], color='green', linewidth=4, alpha=0.8)
            # Plot Bearish bodies
            ax.vlines(t[~bullish], sample[~bullish, 3], sample[~bullish, 0], color='red', linewidth=4, alpha=0.8)
            
            if i == 0:
                ax.set_ylabel(f"{label} Price")
            
            ax.set_title(f"{label} Sample {i}")
            ax.grid(True, alpha=0.3)

    plt.suptitle("OHLC Candlestick Comparison (Real vs Generated)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
