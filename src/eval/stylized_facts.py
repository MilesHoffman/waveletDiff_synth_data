
import numpy as np
import scipy.stats as stats

def calculate_kurtosis(real_data, syn_data, feature_idx=0):
    """
    Calculates the kurtosis of returns for Real and Synthetic data.
    High kurtosis ("fat tails") is a common stylized fact of financial returns.
    
    Args:
        real_data: (N, T, D)
        syn_data: (N, T, D)
        feature_idx: int, index of feature to compute (usually Price/Close)
        
    Returns:
        dict: {'real_kurtosis': float, 'syn_kurtosis': float}
    """
    def get_returns(data):
        # Flatten simple returns
        diff = np.diff(data, axis=1).flatten()
        return diff[np.isfinite(diff)]

    real_ret = get_returns(real_data[:, :, feature_idx])
    syn_ret = get_returns(syn_data[:, :, feature_idx])
    
    k_real = stats.kurtosis(real_ret)
    k_syn = stats.kurtosis(syn_ret)
    
    return {'real_kurtosis': k_real, 'syn_kurtosis': k_syn}

def leverage_effect(real_data, syn_data, feature_idx=0, lag=1):
    """
    Analyzes the Leverage Effect: Correlation between returns at time t 
    and volatility (squared returns) at time t+lag.
    
    In equity markets, this is typically negative (price drop -> higher volatility).
    
    Args:
        real_data: (N, T, D)
        syn_data: (N, T, D)
        
    Returns:
        dict: {'real_leverage': float, 'syn_leverage': float}
    """
    def compute_leverage(data):
        # data: (N, T)
        # returns: (N, T-1)
        ret = np.diff(data, axis=1)
        
        corrs = []
        for i in range(ret.shape[0]):
            r_t = ret[i]
            # Volatility proxy: squared returns
            vol = r_t**2
            
            # Corr(r_t, vol_{t+k})
            # r_t: 0 to end-lag
            # vol: lag to end
            
            if len(r_t) <= lag: continue
            
            curr_ret = r_t[:-lag]
            future_vol = vol[lag:]
            
            if len(curr_ret) < 2: continue
            
            # Avoid constant input warnings
            if np.std(curr_ret) == 0 or np.std(future_vol) == 0:
                continue
                
            c, _ = stats.pearsonr(curr_ret, future_vol)
            if np.isfinite(c):
                corrs.append(c)
                
        if not corrs:
            return 0.0
        return np.mean(corrs)

    real_feat = real_data[:, :, feature_idx]
    syn_feat = syn_data[:, :, feature_idx]
    
    lev_real = compute_leverage(real_feat)
    lev_syn = compute_leverage(syn_feat)
    
    return {'real_leverage': lev_real, 'syn_leverage': lev_syn}
