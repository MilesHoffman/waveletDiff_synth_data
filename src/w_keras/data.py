"""
Data loading pipeline for WaveletDiff (Keras 3 / JAX).
Uses tf.data for high-performance TPU prefetching.
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import pywt

def create_sliding_windows(data, seq_len, stride=1, normalize=True):
    """
    Create sliding window samples from long time series data.
    """
    total_timesteps, n_features = data.shape
    
    if normalize:
        # Ensure data is float type
        data = data.astype(np.float32)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std = np.where(data_std == 0, 1.0, data_std)
        data = (data - data_mean) / data_std

    n_samples = (total_timesteps - seq_len) // stride + 1
    
    # Efficient windowing using numpy strides
    # shape: (n_samples, seq_len, n_features)
    windows = np.lib.stride_tricks.sliding_window_view(data, (seq_len, n_features))
    windows = windows.squeeze(axis=1) # Remove extra dim from sliding_window_view if n_features matches
    # The above squeeze might be risky if n_features != window view logic, let's stick to simple list comp for safety 
    # or manual loop since this is "offline" preparation.
    # Actually, easiest is just the loop if dataset isn't huge.
    
    windows = []
    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        if end_idx <= total_timesteps:
            windows.append(data[start_idx:end_idx])
            
    return np.array(windows, dtype=np.float32)

def prepare_wavelet_data(csv_path, seq_len, wavelet_type='db2', levels='auto'):
    """
    Loads CSV, creates windows, and transforms to wavelet coefficients.
    Returns:
        wavelet_coeffs: (N, total_coeffs, features)
        wavelet_info: dict with shapes/metadata
    """
    df = pd.read_csv(csv_path)
    # Filter core cols if needed, but assuming user provides cleaned CSV or we take all
    # For stocks specifically:
    if 'Open' in df.columns:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # 1. Sliding Windows
    time_series_windows = create_sliding_windows(df.values, seq_len).astype(np.float32)
    # shape: (N, seq_len, features)

    n_samples, _, n_features = time_series_windows.shape

    # 2. Wavelet Transform
    # Determine levels
    if levels == 'auto':
        levels = int(np.clip(pywt.dwt_max_level(seq_len, wavelet_type), 3, 7))
    
    # Get shape info from first sample
    sample = time_series_windows[0, :, 0]
    coeffs = pywt.wavedec(sample, wavelet_type, level=levels, mode='symmetric')
    coeffs_shapes = [c.shape for c in coeffs]
    level_dims = [np.prod(s) for s in coeffs_shapes]
    total_coeffs = sum(level_dims)
    
    # Pre-allocate output container
    # Shape: (N, total_coeffs, features)
    wavelet_data = np.zeros((n_samples, total_coeffs, n_features), dtype=np.float32)
    
    print(f"Transforming {n_samples} samples to Wavelets ({wavelet_type}, L={levels})...")
    
    # Vectorization is hard with pywt, doing loop (it's one-time cost)
    for i in range(n_samples):
        for f in range(n_features):
            signal = time_series_windows[i, :, f]
            c = pywt.wavedec(signal, wavelet_type, level=levels, mode='symmetric')
            # Flatten and concatenate all levels
            c_flat = np.concatenate([arr.flatten() for arr in c])
            wavelet_data[i, :, f] = c_flat
            
    
    wavelet_info = {
        'levels': levels,
        'level_dims': level_dims,
        'level_start_indices': [0] + list(np.cumsum(level_dims[:-1])),
        'n_features': n_features,
        'original_seq_len': seq_len,
        'n_samples': n_samples
    }
    
    return wavelet_data, wavelet_info

def load_dataset(csv_path, batch_size, seq_len, wavelet_type='db2', levels='auto', diffusion_config=None):
    """
    Creates a tf.data.Dataset for training.
    If diffusion_config is present, applies noise addition pipeline.
    """
    print("Preparing data...")
    data, info = prepare_wavelet_data(csv_path, seq_len, wavelet_type, levels)
    
    # Convert to tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices(data)
    
    # Optimization pipeline
    ds = ds.cache() 
    
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    
    ds = ds.shuffle(buffer_size=min(len(data), 10000))
    ds = ds.batch(batch_size, drop_remainder=True) 
    
    if diffusion_config:
        # Diffusion Training Mode
        # Apply noise logic. 
        # Note: Input is float32 (from prepare func).
        # We assume mapper produces tuple ((x_t, t), target)
        # We can let the mapper handle precisions.
        
        ts_mapper = get_diffusion_mapper(
            T=1000, 
            prediction_target=diffusion_config.get('PREDICTION_TARGET', 'noise')
        )
        ds = ds.map(ts_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Output of mapper is float32 usually.
        # We can cast to bfloat16 here for TPU transfer if desired, 
        # but Keras Mixed Precision handles inputs largely.
        # Explicit casting to bfloat16 helps bandwidth.
        def cast_tuple(inputs, target):
            (x_t, t), y = inputs, target
            x_t = tf.cast(x_t, tf.bfloat16)
            # t is float32 normalized, keep it or cast? t used in embeddings usually float32 is safer for sin/cos
            y = tf.cast(y, tf.bfloat16)
            return (x_t, t), y
            
        ds = ds.map(cast_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        
    else:
        # Standard Inference/Raw Mode
        def optimize_transfer(batch):
            return tf.cast(batch, tf.bfloat16)
        ds = ds.map(optimize_transfer, num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds, info
