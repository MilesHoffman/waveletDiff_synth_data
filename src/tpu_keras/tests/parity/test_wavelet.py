import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pywt
import keras_core as keras
import jax
import jax.numpy as jnp
from src.tpu_keras.layers.wavelet import DWT1D, IDWT1D

def test_wavelet_parity():
    print("Running DWT Parity Test...")
    
    # Setup
    batch_size = 2
    seq_len = 128
    channels = 2
    wavelet = 'db4'
    levels = 2
    
    # Generate random input
    x_np = np.random.randn(batch_size, seq_len, channels).astype('float32')
    x_jax = jnp.array(x_np)
    
    # PyWT Reference
    print(f"Computing PyWT reference (wavelet={wavelet}, levels={levels})...")
    pywt_coeffs_batch = []
    for b in range(batch_size):
        sample_coeffs = []
        for c in range(channels):
            coeffs = pywt.wavedec(x_np[b, :, c], wavelet, level=levels, mode='reflect')
            sample_coeffs.append(coeffs)
        pywt_coeffs_batch.append(sample_coeffs)
    
    # Keras/JAX Layer
    print("Computing Keras/JAX DWT...")
    dwt_layer = DWT1D(wavelet=wavelet, levels=levels)
    keras_coeffs = dwt_layer(x_jax)
    
    # Verify DWT Parity
    print("Verifying DWT parity...")
    for i, kc in enumerate(keras_coeffs):
        kc_np = np.array(kc)
        for b in range(batch_size):
            for c in range(channels):
                ref = pywt_coeffs_batch[b][c][i]
                # Check shapes
                if kc_np[b, :len(ref), c].shape != ref.shape:
                    print(f"  SHAPE MISMATCH: Level {i}, Batch {b}, Channel {c}. Exp {ref.shape}, Got {kc_np[b, :len(ref), c].shape}")
                
                diff = np.abs(kc_np[b, :len(ref), c] - ref)
                max_diff = np.max(diff)
                if max_diff > 1e-4:
                    print(f"  FAILED: Level {i}, Batch {b}, Channel {c}, Max Diff: {max_diff}")

    # Verify IDWT (Reconstruction)
    print("Verifying IDWT reconstruction...")
    idwt_layer = IDWT1D(wavelet=wavelet)
    x_rec = idwt_layer(keras_coeffs)
    
    x_rec_np = np.array(x_rec)
    recon_error = np.mean((x_np - x_rec_np[:, :seq_len, :])**2)
    print(f"  Reconstruction MSE: {recon_error}")
    
    if recon_error < 1e-4:
        print("SUCCESS: Wavelet layers are numerically stable.")
    else:
        print("WARNING: High reconstruction error.")

if __name__ == "__main__":
    test_wavelet_parity()
