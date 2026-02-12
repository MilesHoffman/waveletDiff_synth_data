"""
Specialized Financial Metrics for Time Series Evaluation.

This module implements advanced quantitative finance metrics to assess the
stylized facts of synthetic financial data, including:
- Tail Dependence (Joint extreme events)
- Hurst Exponent (Long-term memory)
- Leverage Effect (Volatility asymmetry)
- Drawdown Dynamics
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple, Optional

def calculate_log_returns(data: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Computes log returns for a given price series."""
    # data: (N, T) or (T,)
    if data.ndim == 1:
        return np.diff(np.log(data + epsilon))
    return np.diff(np.log(data + epsilon), axis=-1)

def calculate_tail_dependence(
    real: np.ndarray, 
    synthetic: np.ndarray, 
    q: float = 0.05
) -> Dict[str, float]:
    """
    Calculates Lower and Upper Tail Dependence Coefficients.
    
    Measures the probability of simultaneous extreme events in asset pairs.
    Since we are comparing two datasets (Real vs Synth) distributionally,
    we compute the coefficient on the correlation structure WITHIN each dataset
    and compare the scalar summary (e.g., mean tail dep across all pairs).
    
    For a single asset simulation, this metric makes sense if we have multiple assets (D > 1).
    If D=1, this metric is not applicable for "cross-asset" dependence, 
    but we can look at "temporal" tail dependence (Serial Extremogram).
    
    Assuming D > 1 for cross-sectional tail dependence.
    """
    if real.ndim == 2:
        # (N, T) -> Single asset -> No cross-sectional tail dependence
        return {
            "Tail_Dep_Lower_Diff": 0.0,
            "Tail_Dep_Upper_Diff": 0.0,
            "Real_Tail_L": 0.0,
            "Synth_Tail_L": 0.0
        }
        
    n_features = real.shape[2]
    if n_features < 2:
        return {
            "Tail_Dep_Lower_Diff": 0.0,
            "Tail_Dep_Upper_Diff": 0.0,
            "Real_Tail_L": 0.0,
            "Synth_Tail_L": 0.0
        }

    def get_avg_tail_dep(data: np.ndarray, quantile: float) -> Tuple[float, float]:
        # data: (N, T, D) -> Flatten to (N*T, D) to treat as i.i.d observations of the joint vector
        flat_data = data.reshape(-1, data.shape[2])
        
        # Convert to uniform margins (probability integral transform) empirically
        # rankdata / len gives [1/N, ..., 1] approx U[0,1]
        n_obs = flat_data.shape[0]
        u_data = np.zeros_like(flat_data)
        for i in range(n_features):
            u_data[:, i] = stats.rankdata(flat_data[:, i]) / (n_obs + 1)
            
        # Compute pairwise tail dependence
        lower_coeffs = []
        upper_coeffs = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Lower: P(U_i < q | U_j < q) = P(U_i < q, U_j < q) / P(U_j < q)
                # Count pairs where both are < q
                mask_lower = (u_data[:, i] < quantile) & (u_data[:, j] < quantile)
                count_lower = np.sum(mask_lower)
                # Denominator is roughly q * N
                # Theoretical limit uses limit q->0, we use empirical q
                est_lower = count_lower / (n_obs * quantile)
                lower_coeffs.append(est_lower)
                
                # Upper: P(U_i > 1-q | U_j > 1-q)
                threshold = 1.0 - quantile
                mask_upper = (u_data[:, i] > threshold) & (u_data[:, j] > threshold)
                count_upper = np.sum(mask_upper)
                est_upper = count_upper / (n_obs * quantile)
                upper_coeffs.append(est_upper)
                
        return np.nanmean(lower_coeffs), np.nanmean(upper_coeffs)

    real_l, real_u = get_avg_tail_dep(real, q)
    synth_l, synth_u = get_avg_tail_dep(synthetic, q)
    
    return {
        "Tail_Dep_Lower_Diff": abs(real_l - synth_l),
        "Tail_Dep_Upper_Diff": abs(real_u - synth_u),
        "Real_Tail_L": real_l,
        "Synth_Tail_L": synth_l
    }

def calculate_hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimates Hurst Exponent using Rescaled Range (R/S) Analysis.
    H = 0.5: Random Walk
    H < 0.5: Mean Reverting
    H > 0.5: Trending / Long Memory
    """
    # R/S statistic on full series for several window sizes
    N = len(series)
    min_window = 10
    
    if N < min_window * 2:
        return 0.5
        
    window_sizes = np.logspace(np.log10(min_window), np.log10(N/2), num=10).astype(int)
    window_sizes = np.unique(window_sizes) # Remove duplicates
    
    avg_rs = []
    
    for w in window_sizes:
        # Split into N/w chunks
        num_chunks = N // w
        rs_chunk_sum = 0
        
        for i in range(num_chunks):
            chunk = series[i*w : (i+1)*w]
            mean = np.mean(chunk)
            # Deviations
            y = chunk - mean
            # Cumulative deviations
            z = np.cumsum(y)
            # Range
            R = np.max(z) - np.min(z)
            # Standard deviation
            S = np.std(chunk)
            if S == 0: S = 1e-8
            
            rs_chunk_sum += R/S
            
        avg_rs.append(rs_chunk_sum / num_chunks)
    
    # Regression log(RS) ~ log(w)
    if len(avg_rs) < 2:
        return 0.5
        
    slope, _, _, _, _ = stats.linregress(np.log(window_sizes), np.log(avg_rs))
    return slope

def calculate_hurst_metrics(real: np.ndarray, synthetic: np.ndarray) -> Dict[str, float]:
    """Computes Hurst Exponent for Real and Synthetic data."""
    # Handle (N, T) inputs by expanding to (N, T, 1)
    if real.ndim == 2:
        real = real[:, :, None]
    if synthetic.ndim == 2:
        synthetic = synthetic[:, :, None]
        
    # real: (N, T, D)
    n_features = real.shape[2]
    
    hurst_real = []
    hurst_synth = []
    
    # Average Hurst across samples and features
    # Computational cost: O(N * D * T * log T). Can be heavy.
    # Subsample if N is large
    n_eval = min(len(real), 100) 
    
    for i in range(n_eval):
        for f in range(n_features):
            # Use returns for stationarity check? 
            # Traditionally Hurst is calculated on Returns for market efficiency (H~0.5)
            # or on Log-Prices for trending. 
            # Let's use Returns.
            r_ret = np.diff(real[i, :, f])
            s_ret = np.diff(synthetic[i, :, f])
            
            # Skip if constant
            if np.std(r_ret) > 1e-6:
                hurst_real.append(calculate_hurst_exponent(r_ret))
            if np.std(s_ret) > 1e-6:
                hurst_synth.append(calculate_hurst_exponent(s_ret))
                
    h_r = np.mean(hurst_real) if hurst_real else 0.5
    h_s = np.mean(hurst_synth) if hurst_synth else 0.5
    
    return {
        "Hurst_Real": h_r,
        "Hurst_Synth": h_s,
        "Hurst_Diff": abs(h_r - h_s)
    }

def calculate_leverage_effect(real: np.ndarray, synthetic: np.ndarray) -> Dict[str, float]:
    """
    Measures the asymmetry between past returns and future volatility.
    Corr(r_t, r_{t+1}^2) or Corr(r_t, |r_{t+1}|).
    Typically negative in equities (price drop -> high vol).
    """
    if real.ndim == 2:
        real = real[:, :, None]
    if synthetic.ndim == 2:
        synthetic = synthetic[:, :, None]
        
    n_features = real.shape[2]
    
    def get_leverage_corr(data):
        # data: (N, T, D)
        # Compute per sample, then average
        corrs = []
        for i in range(len(data)):
            for f in range(n_features):
                series = data[i, :, f]
                ret = np.diff(series)
                if len(ret) < 5: continue
                
                # Lag 1 correlation: r_t vs |r_{t+1}|^2
                r_t = ret[:-1]
                vol_t1 = np.abs(ret[1:]) ** 2
                
                if np.std(r_t) < 1e-8 or np.std(vol_t1) < 1e-8:
                    continue
                    
                corr = np.corrcoef(r_t, vol_t1)[0, 1]
                if not np.isnan(corr):
                    corrs.append(corr)
        return np.mean(corrs) if corrs else 0.0

    lev_real = get_leverage_corr(real)
    lev_synth = get_leverage_corr(synthetic)
    
    return {
        "Leverage_Real": lev_real,
        "Leverage_Synth": lev_synth,
        "Leverage_Diff": abs(lev_real - lev_synth)
    }

def calculate_drawdown_stats(real: np.ndarray, synthetic: np.ndarray) -> Dict[str, float]:
    """Checks Maximum Drawdown distributions."""
    def get_max_drawdowns(data):
        mdds = []
        for i in range(len(data)):
            for f in range(data.shape[2]):
                # Assuming data is price path
                prices = data[i, :, f]
                # Cumulative max
                peaks = np.maximum.accumulate(prices)
                # Drawdown
                dd = (peaks - prices) / (peaks + 1e-8)
                mdds.append(np.max(dd))
        return mdds

    mdd_real = get_max_drawdowns(real)
    mdd_synth = get_max_drawdowns(synthetic)
    
    ks_stat, p_val = stats.ks_2samp(mdd_real, mdd_synth)
    
    return {
        "MaxDD_KS_Stat": ks_stat,
        "MaxDD_Real_Mean": np.mean(mdd_real),
        "MaxDD_Synth_Mean": np.mean(mdd_synth)
    }
