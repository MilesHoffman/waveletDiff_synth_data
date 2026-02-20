"""
Stylized Facts Metrics - Financial Realism Checks.

Implements checks for:
1. Fat Tails (Kurtosis)
2. Volatility Clustering (ARCH barrier)
3. Leverage Effect (Return-Volatility Correlation)
"""

import numpy as np
def tail_index_error(real_data: np.ndarray, synth_data: np.ndarray, tail_fraction: float = 0.05) -> float:
    """
    Hill Estimator Error (Tail Index).
    Computes absolute difference in power-law tail decay (alpha) between real and synthetic data.
    """
    def _compute_hill(data: np.ndarray) -> np.ndarray:
        if data.ndim == 2: data = data[:, :, np.newaxis]
        N, T, D = data.shape
        alphas = []
        for d in range(D):
            # Focus on extreme returns (absolute values)
            feat_data = np.abs(data[:, :, d].flatten())
            # Sort descending
            sorted_data = np.sort(feat_data)[::-1]
            k = max(int(len(sorted_data) * tail_fraction), 10)
            tail_data = sorted_data[:k]
            threshold = sorted_data[k]
            
            if threshold <= 0 or len(tail_data) == 0:
                alphas.append(0.0)
                continue
                
            # Hill estimator formula
            log_ratio = np.log(tail_data / threshold)
            gamma = np.mean(log_ratio)
            alpha = 1.0 / gamma if gamma > 0 else 0.0
            alphas.append(alpha)
        return np.array(alphas)

    real_alpha = _compute_hill(real_data)
    synth_alpha = _compute_hill(synth_data)
    return float(np.mean(np.abs(real_alpha - synth_alpha)))

def empirical_var_es_error(real_data: np.ndarray, synth_data: np.ndarray, alpha: float = 0.05) -> float:
    """
    Computes the MAE between the empirical Value at Risk (VaR) and Expected Shortfall (ES).
    """
    def _compute_risk(data: np.ndarray) -> tuple:
        if data.ndim == 2: data = data[:, :, np.newaxis]
        N, T, D = data.shape
        var_list, es_list = [], []
        for d in range(D):
            feat_data = data[:, :, d].flatten()
            var = np.percentile(feat_data, alpha * 100)
            es = np.mean(feat_data[feat_data <= var]) if len(feat_data[feat_data <= var]) > 0 else var
            var_list.append(var)
            es_list.append(es)
        return np.array(var_list), np.array(es_list)

    real_var, real_es = _compute_risk(real_data)
    synth_var, synth_es = _compute_risk(synth_data)
    
    var_error = np.mean(np.abs(real_var - synth_var))
    es_error = np.mean(np.abs(real_es - synth_es))
    return float((var_error + es_error) / 2.0)

def price_volume_asymmetry_error(real_data: np.ndarray, synth_data: np.ndarray, vol_idx: int = 4, close_idx: int = 3) -> float:
    """
    Computes absolute error in Price-Volume Correlation Asymmetry.
    Real markets have higher volume during sell-offs than rallies.
    """
    def _compute_asymmetry(data: np.ndarray) -> float:
        if data.ndim == 2: return 0.0  # Needs feature dim
        # Calculate returns from close
        close = data[:, :, close_idx]
        returns = np.diff(close, axis=1)
        vol = data[:, 1:, vol_idx]
        
        # Flatten
        r_flat = returns.flatten()
        v_flat = vol.flatten()
        
        # Separate positive and negative returns
        pos_mask = r_flat > 0
        neg_mask = r_flat < 0
        
        if np.sum(pos_mask) < 2 or np.sum(neg_mask) < 2:
            return 0.0
            
        corr_pos = np.corrcoef(r_flat[pos_mask], v_flat[pos_mask])[0, 1]
        corr_neg = np.corrcoef(np.abs(r_flat[neg_mask]), v_flat[neg_mask])[0, 1]
        
        return abs(corr_neg - corr_pos) # Asymmetry gap

    real_asym = _compute_asymmetry(real_data)
    synth_asym = _compute_asymmetry(synth_data)
    return float(abs(real_asym - synth_asym))

def volume_acf_error(real_data: np.ndarray, synth_data: np.ndarray, vol_idx: int = 4, max_lag: int = 30) -> float:
    """
    Computes the MAE between real and synthetic Volume ACF (temporal trading persistence).
    """
    if real_data.ndim < 3 or real_data.shape[2] <= vol_idx:
        return 0.0
        
    real_vol_slice = real_data[:, :, [vol_idx]]
    synth_vol_slice = synth_data[:, :, [vol_idx]]
    
    # Re-use _compute_vol_cluster but for raw volume instead of absolute returns
    # (Since volume is already positive, it functions purely as an ACF)
    # Note: _compute_vol_cluster does zero-centering which is correct for ACF.
    real_vol_acf = _compute_vol_cluster(real_vol_slice, max_lag)
    synth_vol_acf = _compute_vol_cluster(synth_vol_slice, max_lag)
    
    # Mean Absolute Error across all lags
    mae = np.mean(np.abs(real_vol_acf - synth_vol_acf))
    return float(mae)


def _compute_vol_cluster(data: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Compute Autocorrelation of Absolute/Squared Returns (Volatility Clustering).
    """
    # Handle 2D input (N, T) -> (N, T, 1)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    # Data shape: (N, T, D)
    # We want ACF of |r_t|
    
    abs_data = np.abs(data)
    
    # Subtract mean of absolute returns
    abs_data = abs_data - abs_data.mean(axis=1, keepdims=True)
    
    N, T, D = data.shape
    # Ensure max_lag isn't larger than sequence length
    if T <= max_lag:
        max_lag = max(1, T - 1)
        
    acfs = []
    
    # Vectorized ACF calculation?
    # Let's simple loop for clarity per feature
    for d in range(D):
        x = abs_data[:, :, d] # (N, T)
        
        # Compute ACF for each lag
        lag_corrs = []
        for lag in range(1, max_lag + 1):
            # Corr(x_t, x_{t-lag})
            series_head = x[:, :-lag]
            series_tail = x[:, lag:]
            
            # Simple cov / var
            prod = series_head * series_tail
            mean_prod = prod.mean() # effectively cov since we zero-centered
            var = x.var() + 1e-8
            
            corr = mean_prod / var
            lag_corrs.append(corr)
            
        acfs.append(lag_corrs)
        
    return np.array(acfs) # (D, max_lag)


def volatility_clustering_score(real_data: np.ndarray, synth_data: np.ndarray) -> float:
    """
    Measure the preservation of Volatility Clustering (ARCH effect).
    
    Returns:
        Mean Euclidean distance between the Volatility ACF curves.
    """
    # Volatility needs returns, assuming input is already stationary-ish or returns
    # Users pass (N,T,D).
    
    real_vol_acf = _compute_vol_cluster(real_data)
    synth_vol_acf = _compute_vol_cluster(synth_data)
    
    # Distance between curves
    # (D, L)
    diff = real_vol_acf - synth_vol_acf
    dist = np.sqrt(np.sum(diff**2, axis=1)) # Euclidean dist per feature
    
    return float(np.mean(dist))
