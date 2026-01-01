import numpy as np
import pywt

def get_wavelet_filters(wavelet_name='db4'):
    """Returns decomposition and reconstruction filters for JAX convolution."""
    wavelet = pywt.Wavelet(wavelet_name)
    
    # pywt.dec_lo: low-pass decomposition
    # pywt.dec_hi: high-pass decomposition
    # JAX lax.conv is cross-correlation. To match pywt, we use filters as-is?
    # Actually, pywt decomposition is dot product of signal and filter.
    # Signal [s0, s1, s2, ...] Filter [f0, f1, f2, ...]
    # Result = s0*f0 + s1*f1 + ...
    # Standard convolution flipper: (s * f)[n] = sum s[k] f[n-k]
    # lax.conv (cross-correlation): (s * f)[n] = sum s[n+k] f[k]
    # To match dot product s0*f0 + s1*f1..., lax.conv is perfect with UNFLIPPED filters.
    
    dec_lo = np.array(wavelet.dec_lo)
    dec_hi = np.array(wavelet.dec_hi)
    
    # Reconstruction
    rec_lo = np.array(wavelet.rec_lo)
    rec_hi = np.array(wavelet.rec_hi)
    
    return {
        'dec_lo': dec_lo,
        'dec_hi': dec_hi,
        'rec_lo': rec_lo,
        'rec_hi': rec_hi
    }
