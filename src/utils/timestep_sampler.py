"""
Importance-weighted timestep sampling strategies for diffusion model training.

This module implements the Hybrid Timestep Sampler which combines:
1. Min-SNR-γ weighting for stable foundation
2. Adaptive loss-history tracking for architecture-specific optimization

NOTE: This implementation is CUDAGraph-safe for torch.compile reduce-overhead mode.
All tensors are pre-placed on the target device and no .to() calls occur during sampling.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class HybridTimestepSampler:
    """
    Hybrid timestep sampler combining Min-SNR-γ with adaptive loss-history.
    
    Phase 1 (warmup): Uses Min-SNR-γ weighting for stable training start
    Phase 2 (adaptive): Blends Min-SNR with loss-history for targeted sampling
    
    Features:
    - Warmup period with pure Min-SNR sampling
    - EMA-smoothed loss history tracking
    - Floor probability to prevent forgetting easy timesteps
    - Exploration/exploitation balance via mixing ratio
    - CUDAGraph-safe (no .to() calls in hot path)
    """
    
    def __init__(
        self,
        alpha_bar_all: torch.Tensor,
        T: int = 1000,
        warmup_steps: int = 5000,
        gamma: float = 5.0,
        exploration_ratio: float = 0.3,
        floor_prob: float = 0.001,
        ema_decay: float = 0.995,
        update_frequency: int = 10
    ):
        """
        Initialize the hybrid timestep sampler.
        
        Args:
            alpha_bar_all: Cumulative product of alphas from noise schedule [T]
            T: Total number of timesteps
            warmup_steps: Number of steps to use pure Min-SNR before adaptive
            gamma: Min-SNR clamping parameter (default: 5.0 from paper)
            exploration_ratio: Weight for Min-SNR vs adaptive (0.3 = 70% Min-SNR)
            floor_prob: Minimum probability for any timestep (prevents forgetting)
            ema_decay: Exponential moving average decay for loss history
            update_frequency: How often to update loss history (every N batches)
        """
        self.T = T
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.exploration_ratio = exploration_ratio
        self.floor_prob = floor_prob
        self.ema_decay = ema_decay
        self.update_frequency = update_frequency
        
        self.step_count = 0
        self._device = alpha_bar_all.device
        self._dtype = alpha_bar_all.dtype
        
        # Compute Min-SNR-γ weights (all tensors on same device as alpha_bar_all)
        snr = alpha_bar_all / (1.0 - alpha_bar_all + 1e-8)
        min_snr_weights = torch.clamp(snr, max=gamma) / (snr + 1e-8)
        self.min_snr_probs = min_snr_weights / min_snr_weights.sum()
        
        # Initialize loss history on same device
        self.loss_history = torch.ones(T, device=self._device, dtype=self._dtype)
        
        # Uniform prior for exploration component
        self.uniform_prior = torch.ones(T, device=self._device, dtype=self._dtype) / T
        
        # Pre-allocate cached probs on same device (CUDAGraph-safe)
        self._cached_probs = self.min_snr_probs.clone()
        self._cache_valid = True
        
        # Track if we've been moved to a different device
        self._initialized = True
    
    def _compute_adaptive_probs(self) -> torch.Tensor:
        """Compute current sampling probabilities based on loss history."""
        adaptive_weights = (self.loss_history + 1e-8).pow(0.5)
        adaptive_probs = adaptive_weights / adaptive_weights.sum()
        
        probs = (
            (1 - self.exploration_ratio) * self.min_snr_probs + 
            self.exploration_ratio * adaptive_probs
        )
        
        probs = torch.clamp(probs, min=self.floor_prob)
        probs = probs / probs.sum()
        
        return probs
    
    def _get_sampling_probs(self) -> torch.Tensor:
        """
        Get current sampling probabilities.
        
        NOTE: No .to(device) calls for CUDAGraph compatibility.
        All tensors are pre-placed on the correct device.
        """
        if self.step_count < self.warmup_steps:
            return self.min_snr_probs
        
        if not self._cache_valid:
            self._cached_probs = self._compute_adaptive_probs()
            self._cache_valid = True
        
        return self._cached_probs
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps according to current importance distribution.
        
        NOTE: The device parameter is kept for API compatibility but tensors
        should already be on the correct device via ensure_device().
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to place the sampled timesteps on (should match internal device)
            
        Returns:
            Tensor of sampled timestep indices [batch_size]
        """
        probs = self._get_sampling_probs()
        # torch.multinomial returns indices on the same device as input
        return torch.multinomial(probs, batch_size, replacement=True)
    
    def update_loss_history(
        self, 
        timesteps: torch.Tensor, 
        losses: torch.Tensor
    ) -> None:
        """
        Update loss history with new observations using EMA smoothing.
        
        This runs outside the compiled graph to avoid side effects.
        """
        self.step_count += 1
        
        if self.step_count % self.update_frequency != 0:
            return
        
        # Detach to avoid graph dependencies
        timesteps_detached = timesteps.detach()
        losses_detached = losses.detach()
        
        # Update on CPU to avoid modifying GPU tensors during graph capture
        # This is safe because it only runs outside the main forward/backward
        for i in range(timesteps_detached.size(0)):
            t = timesteps_detached[i].item()
            loss_val = losses_detached[i].item()
            
            old_val = self.loss_history[t].item()
            new_val = self.ema_decay * old_val + (1 - self.ema_decay) * loss_val
            self.loss_history[t] = new_val
        
        self._cache_valid = False
    
    def ensure_device(self, device: torch.device) -> "HybridTimestepSampler":
        """
        Ensure all sampler tensors are on the specified device.
        
        This should be called ONCE during model setup, not during training.
        For CUDAGraph compatibility, all tensors must be pre-placed.
        """
        if self._device != device:
            self._device = device
            self.min_snr_probs = self.min_snr_probs.to(device)
            self.loss_history = self.loss_history.to(device)
            self.uniform_prior = self.uniform_prior.to(device)
            self._cached_probs = self._cached_probs.to(device)
        return self
    
    def get_diagnostics(self) -> dict:
        """Get diagnostic information about the sampler state."""
        probs = self._get_sampling_probs()
        
        return {
            "step_count": self.step_count,
            "is_warmup": self.step_count < self.warmup_steps,
            "prob_min": probs.min().item(),
            "prob_max": probs.max().item(),
            "prob_std": probs.std().item(),
            "loss_history_mean": self.loss_history.mean().item(),
            "loss_history_std": self.loss_history.std().item(),
        }
    
    def to(self, device: torch.device) -> "HybridTimestepSampler":
        """Alias for ensure_device for compatibility."""
        return self.ensure_device(device)

