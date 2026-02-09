"""
Dataset loading utilities for various time series datasets.

Extracted from the main data module to keep it clean and focused.
"""

import pandas as pd
import numpy as np
import torch
import os
from typing import Dict, Any, Tuple

ATR_PERIOD = 14


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    """
    Compute Average True Range (ATR) using Wilder's smoothing.
    
    Returns array of same length as input, with first (period-1) values as NaN.
    """
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )
    
    atr = np.full_like(tr, np.nan)
    atr[period - 1] = np.mean(tr[:period])
    
    multiplier = 1.0 / period
    for i in range(period, len(tr)):
        atr[i] = atr[i - 1] * (1 - multiplier) + tr[i] * multiplier
    
    return atr


def compute_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Compute Simple Moving Average."""
    ret = np.cumsum(data, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    sma = ret[period - 1:] / period
    
    # Pad beginning with NaN to match original length
    pad = np.full(period - 1, np.nan)
    return np.concatenate([pad, sma])


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Compute Relative Strength Index (RSI) using Wilder's smoothing.
    
    Returns array of same length as input, with first `period` values as NaN.
    """
    delta = np.diff(close, prepend=close[0])
    
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    avg_gain[period] = np.mean(gains[1:period + 1])
    avg_loss[period] = np.mean(losses[1:period + 1])
    
    multiplier = 1.0 / period
    for i in range(period + 1, len(close)):
        avg_gain[i] = avg_gain[i - 1] * (1 - multiplier) + gains[i] * multiplier
        avg_loss[i] = avg_loss[i - 1] * (1 - multiplier) + losses[i] * multiplier
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def compute_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                volume: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Compute Money Flow Index (MFI).
    
    Returns array of same length as input, with first `period` values as NaN.
    """
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    
    tp_diff = np.diff(typical_price, prepend=typical_price[0])
    
    positive_flow = np.where(tp_diff > 0, raw_money_flow, 0.0)
    negative_flow = np.where(tp_diff < 0, raw_money_flow, 0.0)
    
    mfi = np.full_like(close, np.nan)
    
    for i in range(period, len(close)):
        pos_sum = np.sum(positive_flow[i - period + 1:i + 1])
        neg_sum = np.sum(negative_flow[i - period + 1:i + 1])
        
        money_ratio = pos_sum / (neg_sum + 1e-10)
        mfi[i] = 100.0 - (100.0 / (1.0 + money_ratio))
    
    return mfi



def reparameterize_ohlc_window(ohlc: np.ndarray, atr_values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Reparameterize a single OHLC window into percentage-space, ATR-normalized features.
    
    Args:
        ohlc: Window of shape (seq_len, 4) with columns [Open, High, Low, Close]
        atr_values: ATR values for the window, shape (seq_len,)
    
    Returns:
        reparam: Reparameterized features (seq_len, 5): 
                [body_norm, wick_high_ratio, wick_low_ratio, bar_range_norm, cum_ret_norm]
        anchor: First Open price (scalar)
        atr_pct: Mean ATR as percentage of anchor (scalar)
    """
    anchor = ohlc[0, 0]
    mean_atr = np.mean(atr_values)
    atr_pct = (mean_atr / anchor) * 100.0
    
    atr_pct = max(atr_pct, 1e-6)
    
    open_prices = ohlc[:, 0]
    high_prices = ohlc[:, 1]
    low_prices = ohlc[:, 2]
    close_prices = ohlc[:, 3]
    
    # Base percentage moves relative to anchor
    body_pct = ((close_prices - open_prices) / anchor) * 100.0
    bar_range_pct = ((high_prices - low_prices) / anchor) * 100.0
    cum_ret_pct = ((close_prices - anchor) / anchor) * 100.0
    
    # Standard normalization for magnitude channels
    body_norm = body_pct / atr_pct
    bar_range_norm = bar_range_pct / atr_pct
    cum_ret_norm = cum_ret_pct / atr_pct
    
    # Ratio-based wicks (bounded [0, 1])
    # wick = distance / total_range
    total_range = high_prices - low_prices
    total_range = np.where(total_range < 1e-9, 1e-9, total_range) # Prevent div by zero
    
    max_oc = np.maximum(open_prices, close_prices)
    min_oc = np.minimum(open_prices, close_prices)
    
    wick_high_ratio = (high_prices - max_oc) / total_range
    wick_low_ratio = (min_oc - low_prices) / total_range
    
    reparam = np.stack([
        body_norm, 
        wick_high_ratio, 
        wick_low_ratio, 
        bar_range_norm, 
        cum_ret_norm
    ], axis=1)
    
    return reparam.astype(np.float32), float(anchor), float(atr_pct)


def create_sliding_windows(data: np.ndarray, 
                          seq_len: int, 
                          stride: int = 1,
                          normalize: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Create sliding window samples from long time series data (legacy, non-OHLC).
    """
    total_timesteps, n_features = data.shape
    
    if seq_len > total_timesteps:
        raise ValueError(f"seq_len ({seq_len}) cannot be larger than total timesteps ({total_timesteps})")
    
    norm_stats = None
    if normalize:
        data = data.astype(np.float32)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std = np.where(data_std == 0, 1.0, data_std)
        
        norm_stats = {
            'mean': data_mean,
            'std': data_std
        }
        data = (data - data_mean) / data_std
    
    n_samples = (total_timesteps - seq_len) // stride + 1
    
    windows = []
    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        
        if end_idx <= total_timesteps:
            windows.append(data[start_idx:end_idx])
    
    return np.array(windows), norm_stats


def load_ett_data(dataset_name: str, data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load ETT dataset."""
    ett_path = os.path.join(data_dir, "ETT-small", f"ETT{dataset_name[-2:]}.csv")
    if not os.path.exists(ett_path):
        raise FileNotFoundError(f"ETT data not found at: {ett_path}")
    
    df = pd.read_csv(ett_path)
    data, norm_stats = create_sliding_windows(df.values[:, 1:], seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    return torch.FloatTensor(data), norm_stats


def load_fmri_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load fMRI dataset."""
    fmri_path = os.path.join(data_dir, "fMRI", "sim4.mat")
    if not os.path.exists(fmri_path):
        raise FileNotFoundError(f"fMRI data not found at: {fmri_path}")

    from scipy.io import loadmat
    data = loadmat(fmri_path)
    data, norm_stats = create_sliding_windows(data['ts'], seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    
    return torch.FloatTensor(data), norm_stats


def load_exchange_rate_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load Exchange Rate dataset."""
    exchange_rate_path = os.path.join(data_dir, "exchange_rate", "exchange_rate.txt")
    if not os.path.exists(exchange_rate_path):
        raise FileNotFoundError(f"Exchange rate data not found at: {exchange_rate_path}")

    df = pd.read_csv(exchange_rate_path, header=None)
    data, norm_stats = create_sliding_windows(df.values, seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    
    return torch.FloatTensor(data), norm_stats


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _select_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and reorder DataFrame to strict OHLCV columns (case-insensitive)."""
    col_map = {c.lower(): c for c in df.columns}
    
    missing = [c for c in OHLCV_COLUMNS if c not in col_map]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}. Available: {list(df.columns)}")
    
    return df[[col_map[c] for c in OHLCV_COLUMNS]]


def load_stocks_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """
    Load Stocks dataset with OHLC reparameterization, ATR-based scaling, and technical indicators.
    
    Features output (16 channels):
        [0] gap_norm: Overnight gap (ATR-normalized)
        [1] body_norm: Close - Open (ATR-normalized)
        [2] wick_high_ratio: Upper wick / bar range
        [3] wick_low_ratio: Lower wick / bar range
        [4] volume_norm: log(Volume / SMA_20(Volume))
        [5] day_sin: Sin(day of week)
        [6] day_cos: Cos(day of week)
        [7] cum_ret_norm: Cumulative return (ATR-normalized)
        [8] bar_range_norm: High - Low (ATR-normalized)
        [9] sma_200_dev: Close - SMA_200 (ATR-normalized)
        [10] sma_100_dev: Close - SMA_100 (ATR-normalized)
        [11] sma_50_dev: Close - SMA_50 (ATR-normalized)
        [12] sma_20_dev: Close - SMA_20 (ATR-normalized)
        [13] atr_ratio: log(ATR / SMA_20(ATR))
        [14] rsi_norm: (RSI - 50) / 50
        [15] mfi_norm: (MFI - 50) / 50
    """
    # 1. Resolve Path
    stocks_path = data_dir
    
    # Check if we need to look in parent dir (common when running from src/)
    if not os.path.exists(stocks_path) and not os.path.isabs(stocks_path):
        if os.path.exists(os.path.join("..", stocks_path)):
            stocks_path = os.path.join("..", stocks_path)
            # Update data_dir base for directory scans below
            data_dir = stocks_path

    if not os.path.isfile(stocks_path):
        # Try finding SPY in subfolder first
        spy_cand = os.path.join(data_dir, "stocks", "SPY_stock_data.csv")
        generic_cand = os.path.join(data_dir, "stocks", "stock_data.csv")
        
        if os.path.exists(spy_cand):
            stocks_path = spy_cand
        elif os.path.exists(generic_cand):
            stocks_path = generic_cand
        else:
            raise FileNotFoundError(f"Could not find stock data in {data_dir}")

    print(f"Loading stock data from {stocks_path}...")
    df = pd.read_csv(stocks_path)
    
    # 2. Handle Multi-ticker / Metadata headers
    try:
        # Check if 'Open' is numeric, if not, it might be a multi-index CSV or have meta-rows
        pd.to_numeric(df['Open'].iloc[0])
    except (ValueError, KeyError, TypeError, IndexError):
        print("Detected non-numeric first row (metadata/headers), dropping...")
        df = df.iloc[1:].reset_index(drop=True)

    # 3. Date / Day of Week
    # We prioritize the 'Date' column for real calendar context
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        day_of_week = df['Date'].dt.dayofweek.values # 0=Monday, 6=Sunday
    else:
        print("No 'Date' column found. Falling back to synthetic 5-day cycle.")
        day_of_week = np.arange(len(df)) % 5

    # Ensure OHLCV columns exist
    df = _select_ohlcv_columns(df)
    
    data = df.values.astype(np.float32)
    
    open_prices = data[:, 0]
    high_prices = data[:, 1]
    low_prices = data[:, 2]
    close_prices = data[:, 3]
    volume = data[:, 4]
    
    # Compute indicators on full dataset
    atr = compute_atr(high_prices, low_prices, close_prices, period=ATR_PERIOD)
    
    # Compute Volume SMA
    vol_sma_period = 20
    vol_sma = compute_sma(volume, period=vol_sma_period)
    
    # --- NEW: Technical Indicators ---
    # SMAs for price context
    sma_200 = compute_sma(close_prices, period=200)
    sma_100 = compute_sma(close_prices, period=100)
    sma_50 = compute_sma(close_prices, period=50)
    sma_20 = compute_sma(close_prices, period=20)
    
    # ATR SMA for relative volatility
    atr_sma_20 = compute_sma(atr, period=20)
    
    # RSI and MFI
    rsi = compute_rsi(close_prices, period=14)
    mfi = compute_mfi(high_prices, low_prices, close_prices, volume, period=14)
    
    # Start after ALL indicators are valid (200-day SMA is the limiter)
    valid_start = 200
    
    open_prices = open_prices[valid_start:]
    high_prices = high_prices[valid_start:]
    low_prices = low_prices[valid_start:]
    close_prices = close_prices[valid_start:]
    volume = volume[valid_start:]
    atr = atr[valid_start:]
    vol_sma = vol_sma[valid_start:]
    day_of_week = day_of_week[valid_start:]
    
    # Slice new indicators
    sma_200 = sma_200[valid_start:]
    sma_100 = sma_100[valid_start:]
    sma_50 = sma_50[valid_start:]
    sma_20 = sma_20[valid_start:]
    atr_sma_20 = atr_sma_20[valid_start:]
    rsi = rsi[valid_start:]
    mfi = mfi[valid_start:]

    
    total_timesteps = len(open_prices)
    
    if seq_len > total_timesteps:
        raise ValueError(f"seq_len ({seq_len}) cannot be larger than available timesteps ({total_timesteps})")
    
    n_samples = total_timesteps - seq_len + 1
    
    windows = []
    anchors = []
    atr_pcts = []
    vol_smas_window = [] # Store SMA reference for reconstruction
    
    eps = 1e-6
    
    for i in range(n_samples):
        start_idx = i
        end_idx = start_idx + seq_len
        
        ohlc_window = np.stack([
            open_prices[start_idx:end_idx],
            high_prices[start_idx:end_idx],
            low_prices[start_idx:end_idx],
            close_prices[start_idx:end_idx]
        ], axis=1)
        
        atr_window = atr[start_idx:end_idx]
        
        reparam, anchor, atr_pct = reparameterize_ohlc_window(ohlc_window, atr_window)
        
        # Volume Normalization: Log-Ratio relative to SMA
        # V_norm = log( (Volume + eps) / (SMA_20 + eps) )
        curr_vol = volume[start_idx:end_idx]
        curr_sma = vol_sma[start_idx:end_idx]
        
        vol_norm = np.log((curr_vol + eps) / (curr_sma + eps))
        
        # --- NEW FEATURES: Day and Gap ---
        # 1. Day of Week Encoding (Cyclic 7-day)
        curr_days = day_of_week[start_idx:end_idx]
        day_sin = np.sin(2 * np.pi * curr_days / 7.0)
        day_cos = np.cos(2 * np.pi * curr_days / 7.0)
        
        # 2. Overnight Gap 
        prev_close_window = np.zeros_like(ohlc_window[:, 0])
        prev_close_window[1:] = ohlc_window[:-1, 3]
        if start_idx > 0:
            prev_close_window[0] = close_prices[start_idx - 1]
        else:
            prev_close_window[0] = ohlc_window[0, 0]
            
        gap_raw = ohlc_window[:, 0] - prev_close_window
        gap_norm = (gap_raw / anchor) * 100.0 / atr_pct
        
        # --- NEW: Technical Indicator Features ---
        # SMA Deviations: (Close - SMA) / (anchor * atr_pct / 100)
        # This normalizes distance from SMA by local volatility
        scale_factor = anchor * atr_pct / 100.0
        
        curr_close = close_prices[start_idx:end_idx]
        sma_200_dev = (curr_close - sma_200[start_idx:end_idx]) / scale_factor
        sma_100_dev = (curr_close - sma_100[start_idx:end_idx]) / scale_factor
        sma_50_dev = (curr_close - sma_50[start_idx:end_idx]) / scale_factor
        sma_20_dev = (curr_close - sma_20[start_idx:end_idx]) / scale_factor
        
        # ATR Ratio: log(ATR / SMA_20(ATR))
        # Centered around 0 when ATR equals its recent average
        curr_atr = atr[start_idx:end_idx]
        curr_atr_sma = atr_sma_20[start_idx:end_idx]
        atr_ratio = np.log((curr_atr + eps) / (curr_atr_sma + eps))
        
        # RSI/MFI: (X - 50) / 50 → Maps [0, 100] to [-1, 1]
        rsi_norm = (rsi[start_idx:end_idx] - 50.0) / 50.0
        mfi_norm = (mfi[start_idx:end_idx] - 50.0) / 50.0
        
        # Concatenate all 17 features
        window_features = np.concatenate([
            gap_norm.reshape(-1, 1),
            reparam[:, 0:1],  # body_norm
            reparam[:, 1:2],  # wick_high_ratio
            reparam[:, 2:3],  # wick_low_ratio
            vol_norm.reshape(-1, 1),
            day_sin.reshape(-1, 1),
            day_cos.reshape(-1, 1),
            reparam[:, 4:5],  # cum_ret_norm
            reparam[:, 3:4],  # bar_range_norm
            # --- NEW ---
            sma_200_dev.reshape(-1, 1),
            sma_100_dev.reshape(-1, 1),
            sma_50_dev.reshape(-1, 1),
            sma_20_dev.reshape(-1, 1),
            atr_ratio.reshape(-1, 1),
            rsi_norm.reshape(-1, 1),
            mfi_norm.reshape(-1, 1)
        ], axis=1)
        
        windows.append(window_features)
        anchors.append(anchor)
        atr_pcts.append(atr_pct)
        vol_smas_window.append(curr_sma)
    
    windows = np.array(windows, dtype=np.float32)
    anchors = np.array(anchors, dtype=np.float32)
    atr_pcts = np.array(atr_pcts, dtype=np.float32)
    vol_smas_window = np.array(vol_smas_window, dtype=np.float32)
    
    norm_stats = {
        'reparameterized': True,
        'anchors': anchors,
        'atr_pcts': atr_pcts,
        'vol_smas': vol_smas_window,
        'volume_type': 'log_ratio_sma',
        'feature_names': [
            'gap_norm', 'body_norm', 'wick_high_ratio', 'wick_low_ratio', 'volume_norm',
            'day_sin', 'day_cos', 'cum_ret_norm', 'bar_range_norm',
            'sma_200_dev', 'sma_100_dev', 'sma_50_dev', 'sma_20_dev',
            'atr_ratio', 'rsi_norm', 'mfi_norm'
        ]
    }
    
    print(f"Loaded {n_samples} windows with 16-channel feature pipeline")
    print(f"  ATR_pct range: [{atr_pcts.min():.2f}%, {atr_pcts.max():.2f}%]")
    print(f"  Volume Norm mean: {windows[:, :, 4].mean():.4f} (should be ~0)")
    print(f"  RSI Norm mean: {windows[:, :, 14].mean():.4f} (should be ~0)")
    
    return torch.FloatTensor(windows), norm_stats



def load_eeg_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load EEG Eye State dataset."""
    from scipy.io import arff
    
    eeg_path = os.path.join(data_dir, "EEG", "EEG_Eye_State.arff")
    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"EEG data not found at: {eeg_path}")
    
    eeg_data, eeg_meta = arff.loadarff(eeg_path)
    eeg_df = pd.DataFrame(eeg_data)

    data, norm_stats = create_sliding_windows(eeg_df.values[:, :-1], seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    
    return torch.FloatTensor(data), norm_stats
