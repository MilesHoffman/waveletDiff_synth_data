
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.stattools import acf

def calculate_distribution_fidelity(real, synthetic):
    """
    Calculates Wasserstein Distance and KS Test statistics.
    
    Args:
        real (np.ndarray): Real data of shape (N, T, D) or (N, D).
        synthetic (np.ndarray): Synthetic data of shape (N, T, D) or (N, D).
        
    Returns:
        dict: Wasserstein distances and KS test results per feature.
    """
    # Flatten time dimension if present: (N, T, D) -> (N*T, D)
    if len(real.shape) == 3:
        real_flat = real.reshape(-1, real.shape[2])
        synth_flat = synthetic.reshape(-1, synthetic.shape[2])
    else:
        real_flat = real
        synth_flat = synthetic
        
    n_features = real_flat.shape[1]
    results = {}
    
    # 1. Wasserstein Distance (Earth Mover's Distance)
    w_dists = []
    for i in range(n_features):
        w_dist = wasserstein_distance(real_flat[:, i], synth_flat[:, i])
        w_dists.append(w_dist)
        results[f"Wasserstein_Feat_{i}"] = w_dist
    results["Wasserstein_Mean"] = np.mean(w_dists)
    
    # 2. Kolmogorov-Smirnov (KS) Test on Returns (Percent Change)
    # We need time series structure for returns, so assume input is (N, T, D)
    if len(real.shape) == 3:
        # Calculate returns: (X_t - X_{t-1}) / X_{t-1}
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        real_returns = np.diff(real, axis=1) / (real[:, :-1, :] + eps)
        synth_returns = np.diff(synthetic, axis=1) / (synthetic[:, :-1, :] + eps)
        
        real_returns_flat = real_returns.reshape(-1, n_features)
        synth_returns_flat = synth_returns.reshape(-1, n_features)
        
        ks_stats = []
        p_values = []
        for i in range(n_features):
            ks_stat, p_val = ks_2samp(real_returns_flat[:, i], synth_returns_flat[:, i])
            ks_stats.append(ks_stat)
            p_values.append(p_val)
            results[f"KS_Stat_Feat_{i}"] = ks_stat
            results[f"KS_PVal_Feat_{i}"] = p_val
            
        results["KS_Stat_Mean"] = np.mean(ks_stats)
        results["KS_PVal_Mean"] = np.mean(p_values)
        
    return results

def calculate_structural_alignment(real, synthetic, n_samples=2000):
    """
    Calculates PCA EVR Correlation and t-SNE 1-NN Accuracy.
    """
    # Flatten: (N, T*D)
    n_samples = min(n_samples, len(real), len(synthetic))
    real_flat = real[:n_samples].reshape(n_samples, -1)
    synth_flat = synthetic[:n_samples].reshape(n_samples, -1)
    
    results = {}
    
    # 1. PCA Explained Variance Ratio (EVR) Correlation
    pca_r = PCA(n_components='mle')
    pca_r.fit(real_flat)
    evr_r = pca_r.explained_variance_ratio_
    
    pca_s = PCA(n_components=len(evr_r)) # Force same number of components for comparison
    pca_s.fit(synth_flat)
    evr_s = pca_s.explained_variance_ratio_
    
    # Handle length mismatch if 'mle' picked different n_components
    min_len = min(len(evr_r), len(evr_s))
    evr_corr = np.corrcoef(evr_r[:min_len], evr_s[:min_len])[0, 1]
    
    results["PCA_EVR_Corr"] = evr_corr
    
    # 2. t-SNE 1-NN Accuracy
    # Combine data
    X = np.concatenate([real_flat, synth_flat], axis=0)
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)]) # 0: Real, 1: Synthetic
    
    # Compute t-SNE embeddings
    # Using fixed random_state for reproducibility
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    # Train 1-NN classifier on the embeddings
    # We use LOO (Leave-One-Out) effectively by using a KNN classifier
    # But usually 1-NN accuracy for TSTR is done on raw data.
    # User specifically asked for "in the t-SNE space".
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_embedded, y)
    
    # Predict (sanity check on training set? No, that would be 100% since k=1 includes self)
    # The standard way for "1-NN Accuracy" as a metric (like in TimeGAN papers) is Leave-One-Out.
    # But standard KNN.score uses the training data if provided.
    # To do this correctly without LOO overhead, we can just split train/test or similar.
    # However, "1-NN classifier to distinguish... in t-SNE space" usually implies training on a subset and testing on another,
    # OR simple cross-validation.
    # Let's do 5-fold CV to be robust.
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(knn, X_embedded, y, cv=5)
    results["tSNE_1NN_Acc"] = np.mean(scores)
    
    return results

def calculate_financial_reality(real, synthetic, lags=[1, 5, 20]):
    """
    Calculates ACF MSE, Cross-Corr Matrix Norm, and Volatility Clustering.
    """
    n_features = real.shape[2]
    results = {}
    
    # 1. Autocorrelation (ACF) Score
    # Average ACF over all features
    acf_mse = 0
    for i in range(n_features):
        # Flatten time series: (N, T)
        # We average the ACF curve over samples
        real_feat = real[..., i]
        synth_feat = synthetic[..., i]
        
        # Calculate returns
        real_ret = np.diff(real_feat, axis=1) / (real_feat[:, :-1] + 1e-8)
        synth_ret = np.diff(synth_feat, axis=1) / (synth_feat[:, :-1] + 1e-8)
        
        # Calculate ACF per sample and average
        def avg_acf(data, nlags=20):
            acfs = []
            for sample in data:
                try:
                    # nlags must be < sample length
                    acfs.append(acf(sample, nlags=nlags, fft=False))
                except:
                    pass
            return np.mean(acfs, axis=0) if acfs else np.zeros(nlags+1)

        real_acf_curve = avg_acf(real_ret, nlags=max(lags))
        synth_acf_curve = avg_acf(synth_ret, nlags=max(lags))
        
        # Calculate MSE at specific lags
        mse_i = 0
        for lag in lags:
            if lag < len(real_acf_curve):
                mse_i += (real_acf_curve[lag] - synth_acf_curve[lag])**2
        acf_mse += mse_i / len(lags)
        
    results["ACF_MSE"] = acf_mse / n_features
    
    # 2. Cross-Correlation Matrix Distance
    # Flatten: (N*T, D)
    real_flat = real.reshape(-1, n_features)
    synth_flat = synthetic.reshape(-1, n_features)
    
    real_corr = np.corrcoef(real_flat, rowvar=False)
    synth_corr = np.corrcoef(synth_flat, rowvar=False)
    
    # Frobenius norm of difference
    # Handle NaN if constant features
    real_corr = np.nan_to_num(real_corr)
    synth_corr = np.nan_to_num(synth_corr)
    
    corr_diff_norm = np.linalg.norm(real_corr - synth_corr)
    results["CrossCorr_Norm_Diff"] = corr_diff_norm
    
    # 3. Volatility Clustering (ACF of Squared Returns)
    vol_mse = 0
    for i in range(n_features):
        real_feat = real[..., i]
        synth_feat = synthetic[..., i]
        
        real_ret = np.diff(real_feat, axis=1) / (real_feat[:, :-1] + 1e-8)
        synth_ret = np.diff(synth_feat, axis=1) / (synth_feat[:, :-1] + 1e-8)
        
        # Squared returns
        real_sq_ret = real_ret ** 2
        synth_sq_ret = synth_ret ** 2
        
        def avg_acf_vol(data, nlags=20):
            acfs = []
            for sample in data:
                 try:
                    acfs.append(acf(sample, nlags=nlags, fft=False))
                 except: pass
            return np.mean(acfs, axis=0) if acfs else np.zeros(nlags+1)
            
        real_vol_curve = avg_acf_vol(real_sq_ret, nlags=max(lags))
        synth_vol_curve = avg_acf_vol(synth_sq_ret, nlags=max(lags))
        
        # MSE across whole curve (or lags? User said "match the real curve")
        # Let's take MSE of the entire curve up to max lag
        mse_i = np.mean((real_vol_curve - synth_vol_curve)**2)
        vol_mse += mse_i
        
    results["Volatility_MSE"] = vol_mse / n_features
    
    return results

def calculate_memorization_ratio(real, synthetic):
    """
    Calculates the 1/3 Rule Memorization Ratio.
    
    Metric: Fraction of generated samples where d(gen, NN1) < 1/3 * d(gen, NN2).
    NN1 and NN2 are nearest neighbors in the training (real) set.
    """
    if len(real.shape) == 3:
        # Flatten: (N, T*D)
        real_flat = real.reshape(real.shape[0], -1)
        synth_flat = synthetic.reshape(synthetic.shape[0], -1)
    else:
        real_flat = real
        synth_flat = synthetic
        
    # We need 2 nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(real_flat)
    distances, _ = nbrs.kneighbors(synth_flat)
    
    # distances[:, 0] is d(gen, NN1)
    # distances[:, 1] is d(gen, NN2)
    d1 = distances[:, 0]
    d2 = distances[:, 1]
    
    # Avoid division by zero
    ratio = d1 / (d2 + 1e-10)
    
    # Flag if d1 < 1/3 * d2  => ratio < 1/3
    memorized_mask = ratio < (1/3)
    memorization_ratio = np.mean(memorized_mask)
    
    return memorization_ratio

def calculate_diversity_metrics(real, synthetic, k=5):
    """
    Calculates Coverage metric.
    
    Coverage: Fraction of real samples that have at least one generated sample 
    within their topological neighborhood (distance to k-th NN).
    """
    if len(real.shape) == 3:
        real_flat = real.reshape(real.shape[0], -1)
        synth_flat = synthetic.reshape(synthetic.shape[0], -1)
    else:
        real_flat = real
        synth_flat = synthetic
        
    n_real = len(real_flat)
    
    # 1. Estimate manifold radii for real samples
    # For each real sample, find distance to k-th NN in real set
    nbrs_real = NearestNeighbors(n_neighbors=k+1).fit(real_flat) # +1 because self is 0
    distances_real, _ = nbrs_real.kneighbors(real_flat)
    
    # Radius is distance to k-th neighbor (index k, since 0 is self)
    real_radii = distances_real[:, k]
    
    # 2. Check coverage
    # For each real sample, find nearest neighbor in VALID (synthetic) set
    nbrs_synth = NearestNeighbors(n_neighbors=1).fit(synth_flat)
    distances_to_synth, _ = nbrs_synth.kneighbors(real_flat)
    min_dist_to_synth = distances_to_synth[:, 0]
    
    # Covered if nearest synthetic is within real radius
    covered_mask = min_dist_to_synth <= real_radii
    coverage_score = np.mean(covered_mask)
    
    return {"Coverage": coverage_score}

def calculate_fld(real, synthetic, n_components=10):
    """
    Calculates Feature Likelihood Divergence (FLD) proxy.
    
    Since we don't have a broad test set, we fit a GMM on Real data 
    and compare the likelihoods of Real samples vs Synthetic samples.
    
    Ideally: | LogLikelihood(Real) - LogLikelihood(Synth) | should be small (close to 0).
    Large difference means Synth is not matching the density of Real.
    """
    if len(real.shape) == 3:
        real_flat = real.reshape(real.shape[0], -1)
        synth_flat = synthetic.reshape(synthetic.shape[0], -1)
    else:
        real_flat = real
        synth_flat = synthetic

    # Fit GMM on Real Data
    # Min of 10 or n_samples/10 components to avoid overfitting small data
    n_comp = min(n_components, real.shape[0] // 20)
    n_comp = max(1, n_comp)
    
    gmm = GaussianMixture(n_components=n_comp, covariance_type='diag', random_state=42)
    gmm.fit(real_flat)
    
    # Score samples (Log Likelihood)
    ll_real = gmm.score_samples(real_flat)
    ll_synth = gmm.score_samples(synth_flat)
    
    # FLD Score: Divergence in mean log-likelihood
    # We want Synth to have similar 'probability' of belonging to the distribution as Real data.
    fld_score = abs(np.mean(ll_real) - np.mean(ll_synth))
    
    return fld_score
