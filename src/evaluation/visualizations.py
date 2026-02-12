
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
    # Removed sharey='row' to prevent wide-range samples from shrinking others
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 4, 8))
    
    for row, (data, label) in enumerate([(real, 'Real'), (generated, 'Generated')]):
        for i in range(n_samples):
            ax = axes[row, i]
            sample = data[i] # (T, 5)
            
            # --- Tight scaling logic ---
            # Determine min/max of the specific sample range
            s_min, s_max = np.min(sample[:, :4]), np.max(sample[:, :4])
            padding = (s_max - s_min) * 0.1  # 10% padding
            ax.set_ylim(s_min - padding, s_max + padding)
            
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

def plot_financial_stylized_facts(real, generated, feature_names=None):
    """
    Plot standardized financial stylized facts:
    1. Log-Return Distribution (Linear & Log scale)
    2. Q-Q Plot (Normality check)
    3. Volatility Clustering (ACF of squared returns)
    4. Correlation Heatmaps
    """
    import scipy.stats as stats
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.tsa.stattools import acf
    
    n_features = real.shape[2]
    if feature_names is None:
        feature_names = [f'Feat {i}' for i in range(n_features)]
        
    # Calculate returns
    real_ret = np.diff(real, axis=1) / (real[:, :-1, :] + 1e-8)
    synth_ret = np.diff(generated, axis=1) / (generated[:, :-1, :] + 1e-8)
    
    # Flatten for distribution plots
    real_flat = real_ret.flatten()
    synth_flat = synth_ret.flatten()
    
    # 1. Fat Tail Analysis: Clipped KDE + Empirical CCDF
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # --- Left Panel: KDE with x-axis clipped to 1st-99th percentile ---
    combined = np.concatenate([real_flat, synth_flat])
    lo, hi = np.percentile(combined, [1, 99])
    clip_real = real_flat[(real_flat >= lo) & (real_flat <= hi)]
    clip_synth = synth_flat[(synth_flat >= lo) & (synth_flat <= hi)]

    sns.kdeplot(clip_real, ax=axes[0], fill=True, color=COLORS["Real"], label="Real", alpha=0.3, linewidth=2)
    sns.kdeplot(clip_synth, ax=axes[0], fill=True, color=COLORS["Generated"], label="Generated", alpha=0.3, linewidth=2)
    axes[0].set_xlim(lo, hi)
    axes[0].set_title("Return Distribution (1st–99th Percentile)")
    axes[0].set_xlabel("Return")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=11)
    
    # --- Right Panel: Empirical CCDF on log-log scale P(|r| > x) ---
    def empirical_ccdf(data):
        """Compute complementary CDF of absolute values."""
        abs_data = np.sort(np.abs(data))
        n = len(abs_data)
        ccdf = 1.0 - np.arange(1, n + 1) / n
        return abs_data, ccdf

    r_x, r_ccdf = empirical_ccdf(real_flat)
    s_x, s_ccdf = empirical_ccdf(synth_flat)
    
    # Subsample for plotting performance
    step_r = max(1, len(r_x) // 2000)
    step_s = max(1, len(s_x) // 2000)
    
    axes[1].plot(r_x[::step_r], r_ccdf[::step_r], color=COLORS["Real"], label="Real", alpha=0.8, linewidth=1.5)
    axes[1].plot(s_x[::step_s], s_ccdf[::step_s], color=COLORS["Generated"], label="Generated", alpha=0.8, linewidth=1.5)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title("Tail Weight: P(|r| > x)  [Log-Log]")
    axes[1].set_xlabel("|Return|")
    axes[1].set_ylabel("P(|r| > x)")
    axes[1].axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='5% VaR')
    axes[1].axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, label='1% VaR')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, which='both', alpha=0.2)
    
    plt.suptitle("Fat Tail Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 2. Q-Q Plots (vs Normal)
    # We plot Real vs Normal and Synth vs Normal side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    stats.probplot(real_flat, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot: Real vs Normal")
    axes[0].get_lines()[0].set_color(COLORS["Real"])
    axes[0].get_lines()[0].set_alpha(0.5)
    
    stats.probplot(synth_flat, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot: Synthetic vs Normal")
    axes[1].get_lines()[0].set_color(COLORS["Generated"])
    axes[1].get_lines()[0].set_alpha(0.5)
    
    plt.suptitle("Normality Check (Q-Q Plots)")
    plt.tight_layout()
    plt.show()
    
    # 3. Volatility Clustering (ACF of Squared Returns)
    # Average ACF across features
    def get_avg_acf(data, lags=20):
        acfs = []
        for i in range(data.shape[2]):
            feat_data = data[..., i]
            feat_sq = feat_data ** 2
            for j in range(len(feat_sq)):
                if np.var(feat_sq[j]) < 1e-9:
                    acfs.append(np.zeros(lags + 1))
                    continue
                try:
                    a = acf(feat_sq[j], nlags=lags, fft=False)
                    if len(a) < lags + 1:
                        a = np.pad(a, (0, lags + 1 - len(a)))
                    acfs.append(a[:lags + 1])
                except Exception:
                    acfs.append(np.zeros(lags + 1))
        return np.nan_to_num(np.mean(acfs, axis=0))

    lags = 30
    real_vol_acf = get_avg_acf(real_ret, lags)
    synth_vol_acf = get_avg_acf(synth_ret, lags)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(lags+1), real_vol_acf, marker='o', label="Real", color=COLORS["Real"])
    plt.plot(range(lags+1), synth_vol_acf, marker='x', linestyle='--', label="Generated", color=COLORS["Generated"])
    plt.title("Volatility Clustering (ACF of Squared Returns)")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 4. Correlation Matrices
    # Flatten (N*T, D)
    real_corr = np.corrcoef(real_ret.reshape(-1, n_features), rowvar=False)
    synth_corr = np.corrcoef(synth_ret.reshape(-1, n_features), rowvar=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", center=0, annot=True, fmt=".2f",
                xticklabels=feature_names, yticklabels=feature_names)
    axes[0].set_title("Real Correlation Matrix")
    
    sns.heatmap(synth_corr, ax=axes[1], cmap="coolwarm", center=0, annot=True, fmt=".2f",
                xticklabels=feature_names, yticklabels=feature_names)
    axes[1].set_title("Synthetic Correlation Matrix")
    
    plt.suptitle("Cross-Asset Correlation Structure")
    plt.tight_layout()
    plt.show()
