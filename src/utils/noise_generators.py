"""
Noise generation utilities for diffusion models.
Supports standard Gaussian noise and heavy-tailed Student-T noise.
"""

import torch

def generate_noise(shape, device, dtype=torch.float32, prior="gaussian", nu=3.0):
    """
    Generate noise for the diffusion process.
    
    Args:
        shape: Shape of the desired noise tensor
        device: Device to place the tensor on
        dtype: Data type of the tensor
        prior: Type of noise ("gaussian" or "student-t")
        nu: Degrees of freedom for Student-T noise
        
    Returns:
        Noise tensor of the specified shape
    """
    if prior == "gaussian":
        return torch.randn(shape, device=device, dtype=dtype)
    elif prior == "student-t":
        # Generate standard normal Z ~ N(0, 1)
        z = torch.randn(shape, device=device, dtype=dtype)
        
        # We need a Chi-Square(nu) random variable.
        # Chi-Square(nu) is Gamma(nu/2, 1/2).
        # PyTorch's torch._standard_gamma generates from Gamma(alpha, 1).
        # So we use standard_gamma with alpha = nu/2, and then multiply by 2.
        
        # torch.full avoids CPU->GPU scalar transfer during CUDAGraph replay
        alpha_expanded = torch.full(shape, nu / 2.0, device=device, dtype=dtype)
        v = (2.0 * torch._standard_gamma(alpha_expanded)).clamp(min=1e-6)
        
        # Student-t is T = Z * sqrt(nu / V)
        t_noise = z * torch.sqrt(nu / v)
        
        # Variance normalization
        # Var(T) = nu / (nu - 2) for nu > 2
        # We multiply by sqrt((nu - 2) / nu)
        if nu > 2.0:
            scale_val = ((nu - 2.0) / nu) ** 0.5
            return t_noise * scale_val
        else:
            return t_noise
    else:
        raise ValueError(f"Unknown noise prior: {prior}")
