import os
import re

def refactor_loaders():
    path = 'src/data/loaders.py'
    with open(path, 'r') as f:
        content = f.read()

    # 1. Update the docstring
    content = content.replace(
        "        [0]  overnight_gap:     log(Open_t / Close_{t-1})\n        [1]  intraday_return:   log(Close_t / Open_t)\n        [2]  total_log_return:  log(Close_t / Close_{t-1})\n        [3]  normalized_range:  (High - Low) / Close_{t-1}\n        [4]  wick_high_ratio:   (High - max(O,C)) / (H-L)        [0,1]\n        [5]  wick_low_ratio:    (min(O,C) - Low) / (H-L)         [0,1]\n        [6]  prev_intraday_ret: log(Close_{t-1} / Open_{t-1})",
        "        [0]  logit_open_pos:    logit((log(O)-log(L))/(log(H/L)+eps))\n        [1]  logit_close_pos:   logit((log(C)-log(L))/(log(H/L)+eps))\n        [2]  total_log_return:  log(Close_t / Close_{t-1})\n        [3]  log_log_hl_ratio:  log(log(H_t/L_t) + eps)\n        [4]  overnight_gap:     log(Open_t / Close_{t-1}) (redundant)\n        [5]  intraday_return:   log(Close_t / Open_t) (redundant)\n        [6]  prev_intraday_ret: log(Close_{t-1} / Open_{t-1})"
    )

    # 2. Update raw features computation
    old_raw = """    overnight_gap_raw = np.log(open_prices / (prev_close + eps))
    intraday_return_raw = np.log(close_prices / (open_prices + eps))
    total_log_return_raw = np.log(close_prices / (prev_close + eps))

    bar_range = high_prices - low_prices
    normalized_range_raw = bar_range / (prev_close + eps)

    safe_range = np.where(bar_range < eps, eps, bar_range)
    max_oc = np.maximum(open_prices, close_prices)
    min_oc = np.minimum(open_prices, close_prices)
    # New Log-Shadow Formulas
    log_upper_shadow_raw = np.log(high_prices / (max_oc + eps) + eps)
    log_lower_shadow_raw = np.log((min_oc + eps) / (low_prices + eps) + eps)

    prev_intraday_raw = np.log(prev_close / (prev_open + eps))"""

    new_raw = """    overnight_gap_raw = np.log(open_prices / (prev_close + eps))
    intraday_return_raw = np.log(close_prices / (open_prices + eps))
    total_log_return_raw = np.log(close_prices / (prev_close + eps))
    prev_intraday_raw = np.log(prev_close / (prev_open + eps))

    # New logit/sigmoid structural features
    log_hl_ratio = np.log((high_prices + eps) / (low_prices + eps))
    log_hl_ratio_clipped = np.clip(log_hl_ratio, 1e-8, None)
    log_log_hl_ratio_raw = np.log(log_hl_ratio_clipped)
    
    c_pos = (np.log(close_prices + eps) - np.log(low_prices + eps)) / log_hl_ratio_clipped
    o_pos = (np.log(open_prices + eps) - np.log(low_prices + eps)) / log_hl_ratio_clipped
    
    c_pos_clipped = np.clip(c_pos, 1e-5, 1.0 - 1e-5)
    o_pos_clipped = np.clip(o_pos, 1e-5, 1.0 - 1e-5)
    
    from scipy.special import logit
    logit_close_pos_raw = logit(c_pos_clipped)
    logit_open_pos_raw = logit(o_pos_clipped)"""

    content = content.replace(old_raw, new_raw)

    # 3. Slicing
    old_slice = """    overnight_gap_raw = overnight_gap_raw[valid_start:]
    intraday_return_raw = intraday_return_raw[valid_start:]
    total_log_return_raw = total_log_return_raw[valid_start:]
    normalized_range_raw = normalized_range_raw[valid_start:]
    log_upper_shadow_raw = log_upper_shadow_raw[valid_start:]
    log_lower_shadow_raw = log_lower_shadow_raw[valid_start:]
    prev_intraday_raw = prev_intraday_raw[valid_start:]"""

    new_slice = """    logit_open_pos_raw = logit_open_pos_raw[valid_start:]
    logit_close_pos_raw = logit_close_pos_raw[valid_start:]
    total_log_return_raw = total_log_return_raw[valid_start:]
    log_log_hl_ratio_raw = log_log_hl_ratio_raw[valid_start:]
    overnight_gap_raw = overnight_gap_raw[valid_start:]
    intraday_return_raw = intraday_return_raw[valid_start:]
    prev_intraday_raw = prev_intraday_raw[valid_start:]"""

    content = content.replace(old_slice, new_slice)

    # 4. Scaling
    old_scale = """    overnight_gap_scaled, gap_med, gap_iqr = robust_scale(overnight_gap_raw)
    intraday_return_scaled, idr_med, idr_iqr = robust_scale(intraday_return_raw)
    total_log_return_scaled, tlr_med, tlr_iqr = robust_scale(total_log_return_raw)
    normalized_range_scaled, nr_med, nr_iqr = robust_scale(normalized_range_raw)

    # Log Shadows Robust Scaling
    log_upper_shadow_scaled, lus_med, lus_iqr = robust_scale(log_upper_shadow_raw)
    log_lower_shadow_scaled, lls_med, lls_iqr = robust_scale(log_lower_shadow_raw)

    prev_intraday_scaled, pidr_med, pidr_iqr = robust_scale(prev_intraday_raw)"""

    new_scale = """    logit_open_pos_scaled, lop_med, lop_iqr = robust_scale(logit_open_pos_raw)
    logit_close_pos_scaled, lcp_med, lcp_iqr = robust_scale(logit_close_pos_raw)
    total_log_return_scaled, tlr_med, tlr_iqr = robust_scale(total_log_return_raw)
    log_log_hl_ratio_scaled, llhl_med, llhl_iqr = robust_scale(log_log_hl_ratio_raw)

    overnight_gap_scaled, gap_med, gap_iqr = robust_scale(overnight_gap_raw)
    intraday_return_scaled, idr_med, idr_iqr = robust_scale(intraday_return_raw)
    prev_intraday_scaled, pidr_med, pidr_iqr = robust_scale(prev_intraday_raw)"""

    content = content.replace(old_scale, new_scale)

    # 5. Stacking
    old_stack = """        # ── Concatenate 22 features ──
        window_features = np.stack([
            safe(overnight_gap_scaled, s, e),       # [0]
            safe(intraday_return_scaled, s, e),     # [1]
            safe(total_log_return_scaled, s, e),    # [2]
            safe(normalized_range_scaled, s, e),    # [3]
            safe(log_upper_shadow_scaled, s, e),    # [4]
            safe(log_lower_shadow_scaled, s, e),    # [5]
            safe(prev_intraday_scaled, s, e),       # [6]"""

    new_stack = """        # ── Concatenate 22 features ──
        window_features = np.stack([
            safe(logit_open_pos_scaled, s, e),      # [0]
            safe(logit_close_pos_scaled, s, e),     # [1]
            safe(total_log_return_scaled, s, e),    # [2]
            safe(log_log_hl_ratio_scaled, s, e),    # [3]
            safe(overnight_gap_scaled, s, e),       # [4]
            safe(intraday_return_scaled, s, e),     # [5]
            safe(prev_intraday_scaled, s, e),       # [6]"""

    content = content.replace(old_stack, new_stack)

    # 6. Norm stats
    old_stats = """        'robust_scales': {
            'overnight_gap': {'median': gap_med, 'iqr': gap_iqr},
            'intraday_return': {'median': idr_med, 'iqr': idr_iqr},
            'total_log_return': {'median': tlr_med, 'iqr': tlr_iqr},
            'normalized_range': {'median': nr_med, 'iqr': nr_iqr},
            'log_upper_shadow': {'median': lus_med, 'iqr': lus_iqr},
            'log_lower_shadow': {'median': lls_med, 'iqr': lls_iqr},
            'prev_intraday': {'median': pidr_med, 'iqr': pidr_iqr},"""

    new_stats = """        'robust_scales': {
            'logit_open_pos': {'median': lop_med, 'iqr': lop_iqr},
            'logit_close_pos': {'median': lcp_med, 'iqr': lcp_iqr},
            'total_log_return': {'median': tlr_med, 'iqr': tlr_iqr},
            'log_log_hl_ratio': {'median': llhl_med, 'iqr': llhl_iqr},
            'overnight_gap': {'median': gap_med, 'iqr': gap_iqr},
            'intraday_return': {'median': idr_med, 'iqr': idr_iqr},
            'prev_intraday': {'median': pidr_med, 'iqr': pidr_iqr},"""

    content = content.replace(old_stats, new_stats)

    # 7. Feature Names
    old_names = """        'feature_names': [
            'overnight_gap', 'intraday_return', 'total_log_return',
            'normalized_range', 'log_upper_shadow', 'log_lower_shadow',
            'prev_intraday_ret', 'cum_return',"""

    new_names = """        'feature_names': [
            'logit_open_pos', 'logit_close_pos', 'total_log_return',
            'log_log_hl_ratio', 'overnight_gap', 'intraday_return',
            'prev_intraday_ret', 'cum_return',"""

    content = content.replace(old_names, new_names)

    with open(path, 'w') as f:
        f.write(content)

def refactor_module():
    path = 'src/data/module.py'
    with open(path, 'r') as f:
        content = f.read()

    old_inverse = """    def _inverse_reparameterize_ohlc(self, data: np.ndarray,
                                      sample_indices: np.ndarray = None,
                                      fixed_anchor: float = None) -> np.ndarray:
        \"\"\"
        Inverse reparameterization for SOTA 22-feature OHLC data.

        Reconstructs OHLCV from Log-Return based features using
        Robust Scaling inverse and sequential price chaining.

        Feature Index Map:
            [0] overnight_gap (Robust Scaled log-return)
            [1] intraday_return (Robust Scaled log-return)
            [3] normalized_range (Robust Scaled ratio)
            [4] wick_high_ratio [0,1]
            [5] wick_low_ratio [0,1]
            [21] vol_log_dev (Log-Deviation from Rolling Median)
        \"\"\"
        n_samples = data.shape[0]
        seq_len = data.shape[1]

        # ── Resolve Price Anchors ──
        if fixed_anchor is not None:
            anchors = np.full((n_samples,), fixed_anchor)
        elif sample_indices is not None:
            anchors = self.norm_stats['anchors'][sample_indices]
        else:
            all_anchors = self.norm_stats['anchors']
            indices = np.random.choice(len(all_anchors), size=n_samples, replace=True)
            anchors = all_anchors[indices]

        # ── Resolve Volume Anchors ──
        if sample_indices is not None:
            vol_medians = self.norm_stats['vol_medians'][sample_indices]
            vol_iqrs = self.norm_stats['vol_iqrs'][sample_indices]
        else:
            _indices = indices if sample_indices is None and fixed_anchor is not None else np.random.choice(
                len(self.norm_stats['vol_medians']), size=n_samples, replace=True)
            vol_medians = self.norm_stats['vol_medians'][_indices]
            vol_iqrs = self.norm_stats['vol_iqrs'][_indices]

        # ── Retrieve Robust Scaling Stats for Inverse ──
        rs = self.norm_stats['robust_scales']
        gap_med, gap_iqr = rs['overnight_gap']['median'], rs['overnight_gap']['iqr']
        idr_med, idr_iqr = rs['intraday_return']['median'], rs['intraday_return']['iqr']
        nr_med, nr_iqr = rs['normalized_range']['median'], rs['normalized_range']['iqr']
        lus_med, lus_iqr = rs['log_upper_shadow']['median'], rs['log_upper_shadow']['iqr']
        lls_med, lls_iqr = rs['log_lower_shadow']['median'], rs['log_lower_shadow']['iqr']

        # ── Extract & Unscale Structural Features ──
        gap_scaled = data[..., 0]
        idr_scaled = data[..., 1]
        range_scaled = data[..., 3]
        lus_scaled = data[..., 4]
        lls_scaled = data[..., 5]
        vol_log_dev = np.clip(data[..., 21], -10.0, 10.0)

        # Invert Robust Scaling to raw Log-Returns and Log-Shadows
        gap_log_return = (gap_scaled * gap_iqr) + gap_med
        intraday_log_return = (idr_scaled * idr_iqr) + idr_med
        normalized_range = (range_scaled * nr_iqr) + nr_med
        normalized_range = np.maximum(normalized_range, 0.0)

        log_upper_shadow = (lus_scaled * lus_iqr) + lus_med
        log_lower_shadow = (lls_scaled * lls_iqr) + lls_med

        # ── Reconstruct Prices via Sequential Chaining ──
        open_prices = np.zeros((n_samples, seq_len))
        close_prices = np.zeros((n_samples, seq_len))
        high_prices = np.zeros((n_samples, seq_len))
        low_prices = np.zeros((n_samples, seq_len))

        for t in range(seq_len):
            if t == 0:
                prev_close = anchors
            else:
                prev_close = close_prices[:, t - 1]

            open_prices[:, t] = prev_close * np.exp(gap_log_return[:, t])
            close_prices[:, t] = open_prices[:, t] * np.exp(intraday_log_return[:, t])

            total_range = prev_close * normalized_range[:, t]
            total_range = np.maximum(total_range, 0.0)

            max_oc = np.maximum(open_prices[:, t], close_prices[:, t])
            min_oc = np.minimum(open_prices[:, t], close_prices[:, t])

            # Reconstruct High/Low using Log-Shadows (exact inverse of log(ratio + eps))
            eps = 1e-10
            high_prices[:, t] = (max_oc + eps) * (np.exp(log_upper_shadow[:, t]) - eps)
            low_prices[:, t] = (min_oc + eps) / (np.exp(log_lower_shadow[:, t]) - eps) - eps"""

    new_inverse = """    def _inverse_reparameterize_ohlc(self, data: np.ndarray,
                                      sample_indices: np.ndarray = None,
                                      fixed_anchor: float = None) -> np.ndarray:
        \"\"\"
        Inverse reparameterization for SOTA 22-feature OHLC data.

        Reconstructs OHLCV from Logit/Sigmoid Log-Return based features using
        Robust Scaling inverse and structurally guaranteed sequential chaining.

        Feature Index Map:
            [0] logit_open_pos (Robust Scaled)
            [1] logit_close_pos (Robust Scaled)
            [2] total_log_return (Robust Scaled)
            [3] log_log_hl_ratio (Robust Scaled)
            [21] vol_log_dev (Log-Deviation from Rolling Median)
        \"\"\"
        n_samples = data.shape[0]
        seq_len = data.shape[1]

        # ── Resolve Price Anchors ──
        if fixed_anchor is not None:
            anchors = np.full((n_samples,), fixed_anchor)
        elif sample_indices is not None:
            anchors = self.norm_stats['anchors'][sample_indices]
        else:
            all_anchors = self.norm_stats['anchors']
            indices = np.random.choice(len(all_anchors), size=n_samples, replace=True)
            anchors = all_anchors[indices]

        # ── Resolve Volume Anchors ──
        if sample_indices is not None:
            vol_medians = self.norm_stats['vol_medians'][sample_indices]
            vol_iqrs = self.norm_stats['vol_iqrs'][sample_indices]
        else:
            _indices = indices if sample_indices is None and fixed_anchor is not None else np.random.choice(
                len(self.norm_stats['vol_medians']), size=n_samples, replace=True)
            vol_medians = self.norm_stats['vol_medians'][_indices]
            vol_iqrs = self.norm_stats['vol_iqrs'][_indices]

        # ── Retrieve Robust Scaling Stats for Inverse ──
        rs = self.norm_stats['robust_scales']
        lop_med, lop_iqr = rs['logit_open_pos']['median'], rs['logit_open_pos']['iqr']
        lcp_med, lcp_iqr = rs['logit_close_pos']['median'], rs['logit_close_pos']['iqr']
        tlr_med, tlr_iqr = rs['total_log_return']['median'], rs['total_log_return']['iqr']
        llhl_med, llhl_iqr = rs['log_log_hl_ratio']['median'], rs['log_log_hl_ratio']['iqr']

        # ── Extract & Unscale Structural Features ──
        lop_scaled = data[..., 0]
        lcp_scaled = data[..., 1]
        tlr_scaled = data[..., 2]
        llhl_scaled = data[..., 3]
        vol_log_dev = np.clip(data[..., 21], -10.0, 10.0)

        # Invert Robust Scaling
        logit_open_pos = (lop_scaled * lop_iqr) + lop_med
        logit_close_pos = (lcp_scaled * lcp_iqr) + lcp_med
        total_log_return = (tlr_scaled * tlr_iqr) + tlr_med
        log_log_hl_ratio = (llhl_scaled * llhl_iqr) + llhl_med

        # ── Convert Logit/Sigmoid values to Structural Ratios ──
        def expit(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
            
        o_pos = expit(logit_open_pos)
        c_pos = expit(logit_close_pos)
        
        # Clip log_log_hl_ratio to prevent overflow in exp
        log_log_hl_ratio = np.clip(log_log_hl_ratio, -20.0, 10.0)
        log_hl_ratio = np.exp(log_log_hl_ratio)

        # ── Reconstruct Prices via Sequential Chaining ──
        open_prices = np.zeros((n_samples, seq_len))
        close_prices = np.zeros((n_samples, seq_len))
        high_prices = np.zeros((n_samples, seq_len))
        low_prices = np.zeros((n_samples, seq_len))

        for t in range(seq_len):
            if t == 0:
                prev_close = anchors
            else:
                prev_close = close_prices[:, t - 1]

            # 1. Reconstruct Close from Anchor via total_log_return
            close_prices[:, t] = prev_close * np.exp(total_log_return[:, t])
            
            # 2. Use Close position and H/L ratio to find Low
            log_close = np.log(close_prices[:, t] + 1e-10)
            log_low = log_close - c_pos[:, t] * log_hl_ratio[:, t]
            
            # 3. Find High from Low and H/L ratio
            log_high = log_low + log_hl_ratio[:, t]
            
            # 4. Find Open from Low and Open position
            log_open = log_low + o_pos[:, t] * log_hl_ratio[:, t]
            
            # Convert back to price
            low_prices[:, t] = np.exp(log_low)
            high_prices[:, t] = np.exp(log_high)
            open_prices[:, t] = np.exp(log_open)"""

    content = content.replace(old_inverse, new_inverse)
    with open(path, 'w') as f:
        f.write(content)

def refactor_preprocessing():
    path = 'src/evaluation/preprocessing.py'
    with open(path, 'r') as f:
        content = f.read()

    old_return = """    if is_reparam:
        # In reparam space, index 1 is 'body_norm' which is (Close-Open)/Anchor
        # This is already a return-like stationary feature.
        # We perform no diff(), just return the feature as-is.
        # Shape: (N, T)
        return data[:, :, 1]"""

    new_return = """    if is_reparam:
        # In the new logit/sigmoid reparam space, index 2 is 'total_log_return'.
        # This is already a stationary return feature.
        # We perform no diff(), just return the feature as-is.
        # Shape: (N, T)
        return data[:, :, 2]"""

    content = content.replace(old_return, new_return)
    with open(path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    refactor_loaders()
    refactor_module()
    refactor_preprocessing()
    print("Refactoring complete.")
