"""
Dataset loading utilities for various time series datasets.

Implements the SOTA 22-feature pipeline for financial time series,
with Robust Scaling, Log-Ratio normalizations, and path signature
global conditioning via lead-lag transformed log returns and volume.
"""

import pandas as pd
import numpy as np
import torch
import os
from scipy.stats import skew as scipy_skew
from typing import Dict, Any, Tuple

try:
    import esig
    HAS_ESIG = True
except ImportError:
    HAS_ESIG = False


# ─── Rolling Indicator Functions ────────────────────────────────────────────────

def compute_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Compute Simple Moving Average."""
    ret = np.cumsum(data, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    sma = ret[period - 1:] / period

    pad = np.full(period - 1, np.nan)
    return np.concatenate([pad, sma])


def compute_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Money Flow Index (MFI). Output range [0, 1]."""
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
        mfi[i] = (100.0 - (100.0 / (1.0 + money_ratio))) / 100.0

    return mfi


def compute_rolling_yang_zhang(open_prices: np.ndarray, high: np.ndarray,
                                low: np.ndarray, close: np.ndarray,
                                period: int = 20) -> np.ndarray:
    """
    Rolling Yang-Zhang volatility estimator.

    Decomposes into overnight, close-to-open, and Rogers-Satchell components.
    Returns log-transformed values for embedding stability.
    """
    n = len(close)
    yz = np.full(n, np.nan)

    for i in range(period, n):
        s = i - period
        o_w = open_prices[s:i]
        h_w = high[s:i]
        l_w = low[s:i]
        c_w = close[s:i]

        log_oc = np.log(o_w[1:] / c_w[:-1])
        log_co = np.log(c_w / o_w)
        log_ho = np.log(h_w / o_w)
        log_lo = np.log(l_w / o_w)
        log_hc = np.log(h_w / c_w)
        log_lc = np.log(l_w / c_w)

        sigma_o = np.var(log_oc, ddof=1) if len(log_oc) > 1 else 0.0
        sigma_c = np.var(log_co, ddof=1) if period > 1 else 0.0
        sigma_rs = np.mean(log_ho * log_hc + log_lo * log_lc)

        k = 0.34 / (1.34 + (period + 1) / (period - 1)) if period > 1 else 0.34
        sigma_yz_sq = sigma_o + k * sigma_c + (1 - k) * sigma_rs
        yz[i] = np.sqrt(max(sigma_yz_sq, 0.0))

    return yz


def compute_rolling_hurst(close: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Rolling Hurst Exponent via Rescaled Range (R/S) analysis.

    H < 0.5: Mean-reverting, H = 0.5: Random walk, H > 0.5: Trending.
    """
    n = len(close)
    hurst = np.full(n, np.nan)
    log_returns = np.diff(np.log(close), prepend=0.0)
    log_returns[0] = 0.0

    for i in range(period, n):
        ts = log_returns[i - period + 1:i + 1]
        mean_ts = np.mean(ts)
        deviate = np.cumsum(ts - mean_ts)
        r = np.max(deviate) - np.min(deviate)
        s = np.std(ts, ddof=1)

        if s < 1e-12 or r < 1e-12:
            hurst[i] = 0.5
        else:
            hurst[i] = np.log(r / s) / np.log(period)
            hurst[i] = np.clip(hurst[i], 0.0, 1.0)

    return hurst


def compute_rolling_skewness(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Rolling Fisher skewness of log-returns."""
    n = len(close)
    skewness = np.full(n, np.nan)
    log_returns = np.diff(np.log(close), prepend=0.0)
    log_returns[0] = 0.0

    for i in range(period, n):
        window = log_returns[i - period + 1:i + 1]
        if np.std(window) < 1e-10:
            skewness[i] = 0.0
        else:
            skewness[i] = float(scipy_skew(window, bias=False))

    return skewness


def compute_rolling_semivariance(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Rolling downside semivariance of log-returns."""
    n = len(close)
    semivar = np.full(n, np.nan)
    log_returns = np.diff(np.log(close), prepend=0.0)
    log_returns[0] = 0.0

    for i in range(period, n):
        window = log_returns[i - period + 1:i + 1]
        mean_r = np.mean(window)
        downside = window[window < mean_r] - mean_r
        if len(downside) > 1:
            semivar[i] = np.mean(downside ** 2)
        else:
            semivar[i] = 0.0

    return semivar


def compute_rolling_amihud(close: np.ndarray, volume: np.ndarray,
                           period: int = 20) -> np.ndarray:
    """Rolling Amihud Illiquidity Ratio: mean(|return| / volume)."""
    n = len(close)
    amihud = np.full(n, np.nan)
    log_returns = np.abs(np.diff(np.log(close), prepend=0.0))
    log_returns[0] = 0.0

    for i in range(period, n):
        ret_w = log_returns[i - period + 1:i + 1]
        vol_w = volume[i - period + 1:i + 1]
        ratios = ret_w / (vol_w + 1e-10)
        amihud[i] = np.mean(ratios)

    return amihud


# ─── Path Signature Conditioning ───────────────────────────────────────────────

SIG_DEPTH = 4
SIG_CHANNELS = 5  # (time, price_lead, price_lag, vol_lead, vol_lag)


def _lead_lag_transform(price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Construct a 5D lead-lag path from price and volume sequences.

    Time is kept monotonic (no lead-lag). Price and volume are
    duplicated into lead/lag channels. The output has shape
    (2*N - 1, 5) where N = len(price).
    """
    N = len(price)
    time_norm = np.linspace(0.0, 1.0, N)

    out_len = 2 * N - 1
    path = np.zeros((out_len, SIG_CHANNELS), dtype=np.float64)

    for i in range(N - 1):
        # Odd step: lag updates
        path[2 * i, 0] = time_norm[i]
        path[2 * i, 1] = price[i + 1]   # price lead
        path[2 * i, 2] = price[i]        # price lag
        path[2 * i, 3] = volume[i + 1]   # volume lead
        path[2 * i, 4] = volume[i]       # volume lag

        # Even step: lead catches up
        path[2 * i + 1, 0] = time_norm[i]
        path[2 * i + 1, 1] = price[i + 1]
        path[2 * i + 1, 2] = price[i + 1]
        path[2 * i + 1, 3] = volume[i + 1]
        path[2 * i + 1, 4] = volume[i + 1]

    # Final point
    path[-1, 0] = time_norm[-1]
    path[-1, 1] = price[-1]
    path[-1, 2] = price[-1]
    path[-1, 3] = volume[-1]
    path[-1, 4] = volume[-1]

    return path


def _compute_logsig_esig(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute log signature using the esig/roughpy backend."""
    import roughpy
    ctx = roughpy.get_context(path.shape[1], depth, roughpy.DPReal)
    stream = roughpy.LieIncrementStream.from_increments(
        np.diff(path, axis=0), ctx=ctx
    )
    sig = stream.log_signature(roughpy.RealInterval(0, len(path) - 1))
    return np.asarray(sig).astype(np.float32)


def _compute_logsig_fallback(path: np.ndarray, depth: int) -> np.ndarray:
    """
    Pure-numpy fallback for depth-1 log signature (just the increments).
    Only used when no signature library is available.
    """
    increments = path[-1] - path[0]
    return increments.astype(np.float32)


def compute_path_signature(close_seq: np.ndarray, vol_log_dev_seq: np.ndarray,
                           depth: int = SIG_DEPTH) -> np.ndarray:
    """
    Compute log path signature from a look-back window.

    Args:
        close_seq: Close prices for the look-back window (raw, not normalized).
        vol_log_dev_seq: Volume log-deviation values for the same window.
        depth: Signature truncation depth.

    Returns:
        1D numpy array of log-signature features.
    """
    eps = 1e-10
    cum_return = np.log(close_seq / (close_seq[0] + eps))
    path = _lead_lag_transform(cum_return, vol_log_dev_seq)

    if HAS_ESIG:
        return _compute_logsig_esig(path, depth)
    return _compute_logsig_fallback(path, depth)


# ─── Robust Scaling Utilities ──────────────────────────────────────────────────

def robust_scale(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Global Robust Scaling: (x - median) / IQR. Returns scaled array, median, iqr."""
    valid = arr[~np.isnan(arr)]
    median = float(np.median(valid))
    q75, q25 = np.percentile(valid, [75, 25])
    iqr = float(q75 - q25)
    if iqr < 1e-10:
        iqr = 1.0
    return (arr - median) / iqr, median, iqr


def log_zscore(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Log-transform then Z-Score for strictly positive, right-skewed data."""
    valid = arr[~np.isnan(arr)]
    log_arr = np.where(np.isnan(arr), np.nan, np.log(arr + 1e-10))
    valid_log = log_arr[~np.isnan(log_arr)]
    mean = float(np.mean(valid_log))
    std = float(np.std(valid_log))
    if std < 1e-10:
        std = 1.0
    return (log_arr - mean) / std, mean, std


def zscore(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Standard Z-Score normalization."""
    valid = arr[~np.isnan(arr)]
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    if std < 1e-10:
        std = 1.0
    return (arr - mean) / std, mean, std


# ─── Legacy Dataset Loaders ────────────────────────────────────────────────────

def create_sliding_windows(data: np.ndarray,
                          seq_len: int,
                          stride: int = 1,
                          normalize: bool = True) -> Tuple[np.ndarray, dict]:
    """Create sliding window samples from long time series data (legacy, non-OHLC)."""
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


# ─── SOTA 22-Feature Stocks Loader ─────────────────────────────────────────────

def load_stocks_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True,
                     past_days: int = 200) -> Tuple[torch.Tensor, dict]:
    """
    Load Stocks dataset with the SOTA 22-feature pipeline.

    Features output (22 channels):
        [0]  overnight_gap:     log(Open_t / Close_{t-1})
        [1]  intraday_return:   log(Close_t / Open_t)
        [2]  total_log_return:  log(Close_t / Close_{t-1})
        [3]  normalized_range:  (High - Low) / Close_{t-1}
        [4]  wick_high_ratio:   (High - max(O,C)) / (H-L)        [0,1]
        [5]  wick_low_ratio:    (min(O,C) - Low) / (H-L)         [0,1]
        [6]  prev_intraday_ret: log(Close_{t-1} / Open_{t-1})
        [7]  cum_return:        log(Close_t / Close_window_start)
        [8]  day_sin:           sin(2*pi*dow/5)                   [-1,1]
        [9]  day_cos:           cos(2*pi*dow/5)                   [-1,1]
        [10] sma_20_dist:       (Close / SMA_20) - 1
        [11] sma_50_dist:       (Close / SMA_50) - 1
        [12] sma_100_dist:      (Close / SMA_100) - 1
        [13] sma_200_dist:      (Close / SMA_200) - 1
        [14] hurst:             Rolling 20-day Hurst              [0,1]
        [15] yz_vol:            Rolling 20-day Yang-Zhang
        [16] skewness:          Rolling 20-day Fisher Skew
        [17] semivariance:      Rolling 20-day Downside Var
        [18] amihud:            Rolling 20-day |Ret|/Vol
        [19] vol_shock:         log(Volume_t / Volume_{t-1})
        [20] mfi:               MFI_14 / 100                     [0,1]
        [21] vol_log_dev:       (log(V_t) - Median_W) / (IQR_W / 1.349)

    Global Conditioning:
        Depth-4 log path signature of lead-lag transformed
        (cum_return, vol_log_dev) over the preceding `past_days`.
    """
    # ── 1. Resolve Path ──
    stocks_path = data_dir

    if not os.path.exists(stocks_path) and not os.path.isabs(stocks_path):
        if os.path.exists(os.path.join("..", stocks_path)):
            stocks_path = os.path.join("..", stocks_path)
            data_dir = stocks_path

    if not os.path.isfile(stocks_path):
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

    # ── 2. Handle Multi-ticker / Metadata headers ──
    try:
        pd.to_numeric(df['Open'].iloc[0])
    except (ValueError, KeyError, TypeError, IndexError):
        print("Detected non-numeric first row (metadata/headers), dropping...")
        df = df.iloc[1:].reset_index(drop=True)

    # ── 3. Date / Day of Week ──
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        day_of_week = df['Date'].dt.dayofweek.values
    else:
        print("No 'Date' column found. Falling back to synthetic 5-day cycle.")
        day_of_week = np.arange(len(df)) % 5

    df = _select_ohlcv_columns(df)
    data = df.values.astype(np.float64)

    open_prices = data[:, 0]
    high_prices = data[:, 1]
    low_prices = data[:, 2]
    close_prices = data[:, 3]
    volume = data[:, 4]

    eps = 1e-10

    # ── 4. Compute Full-Length Rolling Indicators ──
    sma_200 = compute_sma(close_prices, period=200)
    sma_100 = compute_sma(close_prices, period=100)
    sma_50 = compute_sma(close_prices, period=50)
    sma_20 = compute_sma(close_prices, period=20)

    mfi = compute_mfi(high_prices, low_prices, close_prices, volume, period=14)

    rolling_yz = compute_rolling_yang_zhang(open_prices, high_prices, low_prices, close_prices, period=20)
    rolling_hurst = compute_rolling_hurst(close_prices, period=20)
    rolling_skew = compute_rolling_skewness(close_prices, period=20)
    rolling_semivar = compute_rolling_semivariance(close_prices, period=20)
    rolling_amihud = compute_rolling_amihud(close_prices, volume, period=20)

    # ── 5. Compute per-bar raw features on full dataset ──
    prev_close = np.roll(close_prices, 1)
    prev_close[0] = close_prices[0]
    prev_open = np.roll(open_prices, 1)
    prev_open[0] = open_prices[0]

    overnight_gap_raw = np.log(open_prices / (prev_close + eps))
    intraday_return_raw = np.log(close_prices / (open_prices + eps))
    total_log_return_raw = np.log(close_prices / (prev_close + eps))

    bar_range = high_prices - low_prices
    normalized_range_raw = bar_range / (prev_close + eps)

    safe_range = np.where(bar_range < eps, eps, bar_range)
    max_oc = np.maximum(open_prices, close_prices)
    min_oc = np.minimum(open_prices, close_prices)
    wick_high_ratio = np.clip((high_prices - max_oc) / safe_range, 0.0, 1.0)
    wick_low_ratio = np.clip((min_oc - low_prices) / safe_range, 0.0, 1.0)

    prev_intraday_raw = np.log(prev_close / (prev_open + eps))

    vol_shock_raw = np.log(volume / (np.roll(volume, 1) + eps))
    vol_shock_raw[0] = 0.0

    # Volume Log-Deviation: rolling median/IQR of log(volume)
    log_volume = np.log(volume + eps)
    vol_log_dev = np.full_like(log_volume, np.nan)
    vol_median_arr = np.full_like(log_volume, np.nan)
    vol_iqr_arr = np.full_like(log_volume, np.nan)
    vol_rolling_period = 20

    for i in range(vol_rolling_period, len(log_volume)):
        window = log_volume[i - vol_rolling_period:i]
        med = np.median(window)
        q75, q25 = np.percentile(window, [75, 25])
        iqr = (q75 - q25) / 1.349
        if iqr < eps:
            iqr = 1.0
        vol_log_dev[i] = (log_volume[i] - med) / iqr
        vol_median_arr[i] = med
        vol_iqr_arr[i] = iqr

    # SMA distances (log-ratio-based, stationary and symmetric)
    sma_20_dist = np.log((close_prices + eps) / (sma_20 + eps))
    sma_50_dist = np.log((close_prices + eps) / (sma_50 + eps))
    sma_100_dist = np.log((close_prices + eps) / (sma_100 + eps))
    sma_200_dist = np.log((close_prices + eps) / (sma_200 + eps))

    # ── 6. Determine Valid Start ──
    # Ensure enough history for rolling indicators (200) + path signature lookback
    valid_start = max(200, past_days + 20)

    # ── 7. Slice to valid region ──
    open_prices = open_prices[valid_start:]
    high_prices = high_prices[valid_start:]
    low_prices = low_prices[valid_start:]
    close_prices = close_prices[valid_start:]
    volume = volume[valid_start:]
    day_of_week = day_of_week[valid_start:]

    overnight_gap_raw = overnight_gap_raw[valid_start:]
    intraday_return_raw = intraday_return_raw[valid_start:]
    total_log_return_raw = total_log_return_raw[valid_start:]
    normalized_range_raw = normalized_range_raw[valid_start:]
    wick_high_ratio = wick_high_ratio[valid_start:]
    wick_low_ratio = wick_low_ratio[valid_start:]
    prev_intraday_raw = prev_intraday_raw[valid_start:]
    vol_shock_raw = vol_shock_raw[valid_start:]
    vol_log_dev = vol_log_dev[valid_start:]
    vol_median_arr = vol_median_arr[valid_start:]
    vol_iqr_arr = vol_iqr_arr[valid_start:]

    sma_20_dist = sma_20_dist[valid_start:]
    sma_50_dist = sma_50_dist[valid_start:]
    sma_100_dist = sma_100_dist[valid_start:]
    sma_200_dist = sma_200_dist[valid_start:]

    mfi = mfi[valid_start:]
    rolling_yz = rolling_yz[valid_start:]
    rolling_hurst = rolling_hurst[valid_start:]
    rolling_skew = rolling_skew[valid_start:]
    rolling_semivar = rolling_semivar[valid_start:]
    rolling_amihud = rolling_amihud[valid_start:]

    # ── 8. Apply SOTA Normalization (Global Robust Scaling / Log-ZScore) ──
    overnight_gap_scaled, gap_med, gap_iqr = robust_scale(overnight_gap_raw)
    intraday_return_scaled, idr_med, idr_iqr = robust_scale(intraday_return_raw)
    total_log_return_scaled, tlr_med, tlr_iqr = robust_scale(total_log_return_raw)
    normalized_range_scaled, nr_med, nr_iqr = robust_scale(normalized_range_raw)
    prev_intraday_scaled, pidr_med, pidr_iqr = robust_scale(prev_intraday_raw)
    vol_shock_scaled, vs_med, vs_iqr = robust_scale(vol_shock_raw)

    sma_20_dist_scaled, sma20_med, sma20_iqr = robust_scale(sma_20_dist)
    sma_50_dist_scaled, sma50_med, sma50_iqr = robust_scale(sma_50_dist)
    sma_100_dist_scaled, sma100_med, sma100_iqr = robust_scale(sma_100_dist)
    sma_200_dist_scaled, sma200_med, sma200_iqr = robust_scale(sma_200_dist)

    yz_scaled, yz_log_mean, yz_log_std = log_zscore(rolling_yz)
    semivar_scaled, sv_log_mean, sv_log_std = log_zscore(rolling_semivar)
    amihud_scaled, am_log_mean, am_log_std = log_zscore(rolling_amihud)
    skew_scaled, skew_mean, skew_std = zscore(rolling_skew)

    # ── 9. Build Windows ──
    total_timesteps = len(open_prices)

    if seq_len > total_timesteps:
        raise ValueError(f"seq_len ({seq_len}) cannot be larger than available timesteps ({total_timesteps})")

    n_samples = total_timesteps - seq_len + 1

    windows = []
    anchors = []
    vol_medians = []
    vol_iqrs = []
    path_signatures = []

    # Full-length arrays (pre-slice) for path signature lookback
    full_data = df.values.astype(np.float64)
    g_close = full_data[:, 3]

    # Full-length vol_log_dev (pre-slice) for path signature
    full_log_volume = np.log(full_data[:, 4] + eps)
    full_vol_log_dev = np.full_like(full_log_volume, 0.0)
    for vi in range(20, len(full_log_volume)):
        w = full_log_volume[vi - 20:vi]
        med = np.median(w)
        q75, q25 = np.percentile(w, [75, 25])
        iqr = (q75 - q25) / 1.349
        if iqr < eps:
            iqr = 1.0
        full_vol_log_dev[vi] = (full_log_volume[vi] - med) / iqr

    # Compute a test signature size
    _test_close = g_close[:past_days]
    _test_vld = full_vol_log_dev[:past_days]
    _test_sig = compute_path_signature(_test_close, _test_vld, depth=SIG_DEPTH)
    sig_dim = len(_test_sig)
    print(f"  Path signature dimension: {sig_dim} (depth={SIG_DEPTH}, channels={SIG_CHANNELS})")

    for i in range(n_samples):
        s = i
        e = s + seq_len

        # Day encoding
        curr_days = day_of_week[s:e]
        day_sin = np.sin(2 * np.pi * curr_days / 5.0)
        day_cos = np.cos(2 * np.pi * curr_days / 5.0)

        # Cumulative return relative to window start
        window_close = close_prices[s:e]
        cum_return = np.log(window_close / (window_close[0] + eps))

        # Volume anchors for this window (use the values at window start)
        w_vol_med = vol_median_arr[s] if not np.isnan(vol_median_arr[s]) else 0.0
        w_vol_iqr = vol_iqr_arr[s] if not np.isnan(vol_iqr_arr[s]) else 1.0

        # Price anchor for reconstruction
        anchor = float(close_prices[s])

        # NaN-safe feature fill
        def safe(arr, start, end):
            slc = arr[start:end].copy()
            slc[np.isnan(slc)] = 0.0
            return slc

        # ── Concatenate 22 features ──
        window_features = np.stack([
            safe(overnight_gap_scaled, s, e),       # [0]
            safe(intraday_return_scaled, s, e),     # [1]
            safe(total_log_return_scaled, s, e),    # [2]
            safe(normalized_range_scaled, s, e),    # [3]
            wick_high_ratio[s:e],                   # [4]
            wick_low_ratio[s:e],                    # [5]
            safe(prev_intraday_scaled, s, e),       # [6]
            cum_return,                             # [7]
            day_sin,                                # [8]
            day_cos,                                # [9]
            safe(sma_20_dist_scaled, s, e),         # [10]
            safe(sma_50_dist_scaled, s, e),         # [11]
            safe(sma_100_dist_scaled, s, e),        # [12]
            safe(sma_200_dist_scaled, s, e),        # [13]
            safe(rolling_hurst[s:e], s, e) if False else np.nan_to_num(rolling_hurst[s:e], nan=0.5),  # [14]
            safe(yz_scaled, s, e),                  # [15]
            safe(skew_scaled, s, e),                # [16]
            safe(semivar_scaled, s, e),             # [17]
            safe(amihud_scaled, s, e),              # [18]
            safe(vol_shock_scaled, s, e),           # [19]
            np.nan_to_num(mfi[s:e], nan=0.5),       # [20]
            safe(vol_log_dev, s, e),                # [21]
        ], axis=1)

        windows.append(window_features)
        anchors.append(anchor)
        vol_medians.append(w_vol_med)
        vol_iqrs.append(w_vol_iqr)

        # Path signature from the preceding `past_days` (no lookahead)
        global_start = valid_start + s
        sig_start = global_start - past_days
        sig_close = g_close[sig_start:global_start]
        sig_vld = full_vol_log_dev[sig_start:global_start]
        sig = compute_path_signature(sig_close, sig_vld, depth=SIG_DEPTH)
        path_signatures.append(sig)

    windows = np.array(windows, dtype=np.float32)
    anchors = np.array(anchors, dtype=np.float32)
    vol_medians = np.array(vol_medians, dtype=np.float32)
    vol_iqrs = np.array(vol_iqrs, dtype=np.float32)
    path_signatures = np.array(path_signatures, dtype=np.float32)

    norm_stats = {
        'reparameterized': True,
        'anchors': anchors,
        'vol_medians': vol_medians,
        'vol_iqrs': vol_iqrs,
        'robust_scales': {
            'overnight_gap': {'median': gap_med, 'iqr': gap_iqr},
            'intraday_return': {'median': idr_med, 'iqr': idr_iqr},
            'total_log_return': {'median': tlr_med, 'iqr': tlr_iqr},
            'normalized_range': {'median': nr_med, 'iqr': nr_iqr},
            'prev_intraday': {'median': pidr_med, 'iqr': pidr_iqr},
            'vol_shock': {'median': vs_med, 'iqr': vs_iqr},
            'sma_20_dist': {'median': sma20_med, 'iqr': sma20_iqr},
            'sma_50_dist': {'median': sma50_med, 'iqr': sma50_iqr},
            'sma_100_dist': {'median': sma100_med, 'iqr': sma100_iqr},
            'sma_200_dist': {'median': sma200_med, 'iqr': sma200_iqr},
        },
        'log_zscore_stats': {
            'yz_vol': {'mean': yz_log_mean, 'std': yz_log_std},
            'semivariance': {'mean': sv_log_mean, 'std': sv_log_std},
            'amihud': {'mean': am_log_mean, 'std': am_log_std},
        },
        'zscore_stats': {
            'skewness': {'mean': skew_mean, 'std': skew_std},
        },
        'volume_type': 'log_deviation_median',
        'feature_names': [
            'overnight_gap', 'intraday_return', 'total_log_return',
            'normalized_range', 'wick_high_ratio', 'wick_low_ratio',
            'prev_intraday_ret', 'cum_return',
            'day_sin', 'day_cos',
            'sma_20_dist', 'sma_50_dist', 'sma_100_dist', 'sma_200_dist',
            'hurst', 'yz_vol', 'skewness', 'semivariance',
            'amihud', 'vol_shock', 'mfi', 'vol_log_dev'
        ],
        'path_signatures': path_signatures,
        'path_sig_dim': sig_dim,
        'past_days': past_days,
    }

    print(f"Loaded {n_samples} windows with SOTA 22-channel feature pipeline")
    print(f"  Anchor price range: [{anchors.min():.2f}, {anchors.max():.2f}]")
    print(f"  Vol LogDev mean: {windows[:, :, 21].mean():.4f} (should be ~0)")
    print(f"  Path signature: dim={sig_dim}, past_days={past_days}")
    print(f"  Sig mean={path_signatures.mean():.4f}, std={path_signatures.std():.4f}")

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
