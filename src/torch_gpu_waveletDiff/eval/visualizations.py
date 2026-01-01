
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import scipy.stats as stats
import pywt

def plot_tsne(real_data, syn_data, sample_size=1000, save_path=None):
    """
    Plots t-SNE visualization of Real vs Generated data.
    
    Args:
        real_data: (N, T, D) numpy array
        syn_data: (N, T, D) numpy array
        sample_size: int, number of samples to use for plot
        save_path: str, optional path to save image
    """
    # Flatten time and feature dimensions for t-SNE
    # Or, we can average over time, or just take the last time step. 
    # Usually for time series t-SNE, people flatten or use specific embeddings. 
    # Following standard practice for this repo type (TimeGAN etc): Flatten [N, T*D]
    
    N_real, T, D = real_data.shape
    N_syn = syn_data.shape[0]
    
    n = min(sample_size, N_real, N_syn)
    
    idx_real = np.random.choice(N_real, n, replace=False)
    idx_syn = np.random.choice(N_syn, n, replace=False)
    
    real_sample = real_data[idx_real].reshape(n, -1)
    syn_sample = syn_data[idx_syn].reshape(n, -1)
    
    data_combined = np.concatenate([real_sample, syn_sample], axis=0)
    labels = np.concatenate([np.zeros(n), np.ones(n)]) # 0: Real, 1: Generated
    
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(data_combined)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[labels==0, 0], embedding[labels==0, 1], 
                c='red', label='Real', alpha=0.3, s=10)
    plt.scatter(embedding[labels==1, 0], embedding[labels==1, 1], 
                c='blue', label='Generated', alpha=0.3, s=10)
    
    plt.legend()
    plt.title("t-SNE Visualization: Real vs Generated")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_distributions(real_data, syn_data, feature_names=None, exclude_indices=None, save_path=None):
    """
    Plots probability density functions (KDE) for each feature.
    
    Args:
        real_data: (N, T, D) array
        syn_data: (N, T, D) array
        feature_names: List of strings
        exclude_indices: List of ints, feature indices to exclude (e.g. Volume)
    """
    N, T, D = real_data.shape
    
    # Flatten over time: treat every time step as a sample point for distribution
    real_flat = real_data.reshape(-1, D) # (N*T, D)
    syn_flat = syn_data.reshape(-1, D)
    
    if exclude_indices is None:
        exclude_indices = []
        
    features_to_plot = [i for i in range(D) if i not in exclude_indices]
    n_plots = len(features_to_plot)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
        
    for ax, feat_idx in zip(axes, features_to_plot):
        fname = feature_names[feat_idx] if feature_names else f"Feature {feat_idx}"
        
        sns.kdeplot(real_flat[:, feat_idx], ax=ax, color='red', fill=True, label='Real', alpha=0.3)
        sns.kdeplot(syn_flat[:, feat_idx], ax=ax, color='blue', fill=True, label='Generated', alpha=0.3)
        
        ax.set_title(f"Distribution of {fname}")
        ax.legend()
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_scalogram(real_sample, syn_sample, wavelet='cmor1.5-1.0', save_path=None):
    """
    Plots Continuous Wavelet Transform (CWT) scalogram for a single real vs synthetic sample.
    
    Args:
        real_sample: (T,) 1D array - single feature time series
        syn_sample: (T,) 1D array
    """
    # Scales for CWT
    scales = np.arange(1, 128)
    
    coef_real, freqs_real = pywt.cwt(real_sample, scales, wavelet)
    coef_syn, freqs_syn = pywt.cwt(syn_sample, scales, wavelet)
    
    power_real = np.abs(coef_real)**2
    power_syn = np.abs(coef_syn)**2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Real
    im1 = ax1.imshow(power_real, extent=[0, len(real_sample), 1, 128], cmap='jet', aspect='auto',
                     vmax=power_real.max(), vmin=0)
    ax1.set_title("Real Data CWT Scalogram")
    ax1.set_ylabel("Scale")
    fig.colorbar(im1, ax=ax1)
    
    # Synthetic
    im2 = ax2.imshow(power_syn, extent=[0, len(syn_sample), 1, 128], cmap='jet', aspect='auto',
                     vmax=power_real.max(), vmin=0) # Use same scale as real for comparison
    ax2.set_title("Generated Data CWT Scalogram")
    ax2.set_ylabel("Scale")
    ax2.set_xlabel("Time")
    fig.colorbar(im2, ax=ax2)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_qq(real_data, syn_data, feature_idx=0, save_path=None):
    """
    Q-Q plot of Real vs Synthetic quantiles for a specific feature.
    """
    real_flat = real_data[:, :, feature_idx].flatten()
    syn_flat = syn_data[:, :, feature_idx].flatten()
    
    # Sort
    real_sorted = np.sort(real_flat)
    syn_sorted = np.sort(syn_flat)
    
    # Interpolate to match size if different (though usually we pass equal size)
    n = min(len(real_sorted), len(syn_sorted))
    # Downsample to n
    real_final = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(real_sorted)), real_sorted)
    syn_final = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(syn_sorted)), syn_sorted)
    
    plt.figure(figsize=(6,6))
    plt.scatter(real_final, syn_final, alpha=0.1, s=1)
    
    # Identity line
    min_val = min(real_final.min(), syn_final.min())
    max_val = max(real_final.max(), syn_final.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel("Real Quantiles")
    plt.ylabel("Generated Quantiles")
    plt.title(f"Q-Q Plot (Feature {feature_idx})")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_psd(real_data, syn_data, feature_idx=0, save_path=None):
    """
    Power Spectral Density comparison.
    """
    from scipy.signal import welch
    
    # Average PSD over all samples
    f_real, Pxx_real = welch(real_data[:, :, feature_idx], axis=1)
    f_syn, Pxx_syn = welch(syn_data[:, :, feature_idx], axis=1)
    
    mean_pxx_real = np.mean(Pxx_real, axis=0)
    mean_pxx_syn = np.mean(Pxx_syn, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_real, mean_pxx_real, label='Real', color='red')
    plt.semilogy(f_syn, mean_pxx_syn, label='Generated', color='blue')
    
    plt.title(f"Power Spectral Density (Feature {feature_idx})")
    plt.xlabel("Frequency")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_acf(real_data, syn_data, feature_idx=0, max_lag=50, save_path=None):
    """
    Plot Autocorrelation of Returns, Squared Returns, and Abs Returns.
    Checks for Volatility Clustering.
    """
    from statsmodels.tsa.stattools import acf
    
    def get_returns(data):
        # returns = (t - (t-1)) / (t-1) ... standard return
        # Log returns often used: ln(t) - ln(t-1)
        # Avoid division by zero, let's use simple diff for now or log diff if data > 0
        # Given normalized data (0-1), simple diff is safer but physical meaning is lost if not de-normalized.
        # Assuming we just want to check temporal structure:
        return np.diff(data, axis=1)

    real_ret = get_returns(real_data[:, :, feature_idx])
    syn_ret = get_returns(syn_data[:, :, feature_idx])
    
    # 1. ACF of Returns (Linear dependence)
    # 2. ACF of Squared Returns (Volatility clustering)
    
    def compute_avg_acf(data_returns, transform=lambda x: x):
        acfs = []
        for i in range(data_returns.shape[0]):
            ts = transform(data_returns[i])
            # drop nans/inf if any
            ts = ts[np.isfinite(ts)]
            if len(ts) < max_lag: 
                continue
            try:
                lag_acf = acf(ts, nlags=max_lag, fft=True)
                acfs.append(lag_acf)
            except:
                continue
                
        if len(acfs) == 0:
            return None
        return np.nanmean(np.array(acfs), axis=0)

    acf_real_ret = compute_avg_acf(real_ret)
    acf_syn_ret = compute_avg_acf(syn_ret)
    
    acf_real_sqw = compute_avg_acf(real_ret, lambda x: x**2)
    acf_syn_sqw = compute_avg_acf(syn_ret, lambda x: x**2)

    # Determine valid lags from whichever result is available
    valid_result = next((r for r in [acf_real_ret, acf_syn_ret, acf_real_sqw, acf_syn_sqw] if r is not None), None)
    
    if valid_result is None:
        print("Warning: Insufficient data length for ACF plot. Skipping.")
        return

    lags = np.arange(len(valid_result))
    
    # Handle None cases for plotting (fill with zeros or skip plot lines)
    acf_real_ret = acf_real_ret if acf_real_ret is not None else np.zeros_like(lags)
    acf_syn_ret = acf_syn_ret if acf_syn_ret is not None else np.zeros_like(lags)
    acf_real_sqw = acf_real_sqw if acf_real_sqw is not None else np.zeros_like(lags)
    acf_syn_sqw = acf_syn_sqw if acf_syn_sqw is not None else np.zeros_like(lags)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Linear Returns
    ax1.plot(lags, acf_real_ret, label='Real', color='red')
    ax1.plot(lags, acf_syn_ret, label='Generated', color='blue')
    ax1.set_title("ACF of Raw Returns (Market Efficiency Check)")
    ax1.axhline(0, linestyle='--', color='black', alpha=0.5)
    ax1.legend()
    
    # Squared Returns
    ax2.plot(lags, acf_real_sqw, label='Real', color='red')
    ax2.plot(lags, acf_syn_sqw, label='Generated', color='blue')
    ax2.set_title("ACF of Squared Returns (Volatility Clustering)")
    ax2.axhline(0, linestyle='--', color='black', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

