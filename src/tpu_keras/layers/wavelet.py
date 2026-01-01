import keras
from keras import ops
import jax
import jax.numpy as jnp
import numpy as np
from src.tpu_keras.layers.wavelet_utils import get_wavelet_filters

class DWT1D(keras.layers.Layer):
    """1D Discrete Wavelet Transform using matrix multiplication (for verification)."""
    def __init__(self, wavelet='db4', levels=1, **kwargs):
        super().__init__(**kwargs)
        self.wavelet = wavelet
        self.levels = levels
        
        filters = get_wavelet_filters(wavelet)
        self.h = filters['dec_lo']
        self.g = filters['dec_hi']

    def _get_matrix(self, n, h, g):
        # Create DWT matrix for length n
        m = len(h)
        # Pad filters with zeros to length n if needed or use circular shift
        # This is a simplified version for seq_len multiple of 2
        W = np.zeros((n, n))
        for i in range(n // 2):
            for j in range(m):
                W[i, (2*i + j) % n] = h[j]
                W[i + n//2, (2*i + j) % n] = g[j]
        return jnp.array(W)

    def call(self, x):
        # x: (batch, seq_len, channels)
        batch, n, c = x.shape
        res = x
        coeffs = []
        for _ in range(self.levels):
            curr_n = res.shape[1]
            W = self._get_matrix(curr_n, self.h, self.g)
            # Apply to each channel
            # res_t: (channels, batch, n)
            res_t = res.transpose(2, 0, 1)
            # W: (n, n)
            # We want out[c, b, i] = sum_j W[i, j] * res_t[c, b, j]
            # This is out_t = jnp.matmul(res_t, W.T)
            out_t = jnp.matmul(res_t, W.T) # (c, b, n)
            out = out_t.transpose(1, 2, 0) # (b, n, c)
            
            # Split into A (first n/2) and D (last n/2)
            A = out[:, :curr_n//2, :]
            D = out[:, curr_n//2:, :]
            coeffs.append(D)
            res = A
        coeffs.append(res)
        return coeffs[::-1]

class IDWT1D(keras.layers.Layer):
    """1D Inverse Discrete Wavelet Transform using matrix multiplication."""
    def __init__(self, wavelet='db4', **kwargs):
        super().__init__(**kwargs)
        self.wavelet = wavelet
        filters = get_wavelet_filters(wavelet)
        # For orthogonal wavelets, reconstruction matrix is W.T
        self.h = filters['dec_lo']
        self.g = filters['dec_hi']

    def _get_matrix(self, n, h, g):
        m = len(h)
        W = np.zeros((n, n))
        for i in range(n // 2):
            for j in range(m):
                W[i, (2*i + j) % n] = h[j]
                W[i + n//2, (2*i + j) % n] = g[j]
        return jnp.array(W)

    def call(self, coeffs):
        res = coeffs[0]
        for i in range(1, len(coeffs)):
            D = coeffs[i]
            # Combine A (res) and D
            combined = jnp.concatenate([res, D], axis=1) # (b, n, c)
            n = combined.shape[1]
            W = self._get_matrix(n, self.h, self.g)
            
            # Reconstruction is x = W.T @ combined
            # res_t: (channels, batch, n)
            # W: (n, n), so W.T: (n, n)
            # x[c, b, i] = sum_j W.T[i, j] * res_t[c, b, j]
            res_t = combined.transpose(2, 0, 1)
            out_t = jnp.matmul(res_t, W) # (c, b, n)
            res = out_t.transpose(1, 2, 0)
        return res
