import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras_core as keras
import jax
import jax.numpy as jnp
from src.tpu_keras.layers.wavelet import DWT1D, IDWT1D

def test_reconstruction():
    print("Testing JAX Wavelet Reconstruction (PR)...")
    
    batch_size = 1
    seq_len = 64
    channels = 1
    wavelet = 'db4'
    levels = 2
    
    x = np.random.randn(batch_size, seq_len, channels).astype('float32')
    x_jax = jnp.array(x)
    
    dwt = DWT1D(wavelet=wavelet, levels=levels)
    idwt = IDWT1D(wavelet=wavelet)
    
    coeffs = dwt(x_jax)
    x_rec = idwt(coeffs)
    
    x_rec_np = np.array(x_rec)
    # Check max error
    error = np.max(np.abs(x - x_rec_np[:, :seq_len, :]))
    print(f"Max Reconstruction Error: {error}")
    
    if error < 1e-4:
        print("SUCCESS: Perfect Reconstruction achieved.")
    else:
        print("FAILED: Reconstruction error is too high.")

if __name__ == "__main__":
    test_reconstruction()
