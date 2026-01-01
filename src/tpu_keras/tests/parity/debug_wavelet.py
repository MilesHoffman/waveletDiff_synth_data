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
    batch_size = 1
    seq_len = 16
    channels = 1
    wavelet = 'db2'
    levels = 1
    
    # Generate random input
    x_np = np.random.randn(batch_size, seq_len, channels).astype('float32')
    x_jax = jnp.array(x_np)
    
    # PyWT Reference
    print(f"Input: {x_np.flatten()[:5]}...")
    coeffs = pywt.wavedec(x_np[0, :, 0], wavelet, level=levels, mode='reflect')
    print(f"PyWT A{levels}: {coeffs[0][:5]}")
    print(f"PyWT D{levels}: {coeffs[1][:5]}")
    
    # Keras/JAX Layer
    dwt_layer = DWT1D(wavelet=wavelet, levels=levels)
    keras_coeffs = dwt_layer(x_jax)
    
    print(f"JAX A{levels}: {np.array(keras_coeffs[0]).flatten()[:5]}")
    print(f"JAX D{levels}: {np.array(keras_coeffs[1]).flatten()[:5]}")
    
    # Check shapes
    print(f"PyWT Shapes: {[c.shape for c in coeffs]}")
    print(f"JAX Shapes: {[c.shape for c in keras_coeffs]}")

if __name__ == "__main__":
    test_wavelet_parity()
