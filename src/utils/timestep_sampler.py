"""
Importance-weighted timestep sampling strategies for diffusion model training.

This module implements the Hybrid Timestep Sampler which combines:
1. Min-SNR-γ weighting for stable foundation
2. Adaptive loss-history tracking for architecture-specific optimization
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
        
        # Compute Min-SNR-γ weights
        # SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))
        snr = alpha_bar_all / (1.0 - alpha_bar_all + 1e-8)
        
        # Min-SNR-γ weight: min(SNR, γ) / SNR
        # This down-weights high-SNR (easy/clean) timesteps
        min_snr_weights = torch.clamp(snr, max=gamma) / (snr + 1e-8)
        
        # Normalize to probability distribution
        self.min_snr_probs = min_snr_weights / min_snr_weights.sum()
        
        # Initialize loss history with uniform prior (no bias initially)
        self.loss_history = torch.ones(T, device=self._device)
        
        # Uniform prior for exploration component
        self.uniform_prior = torch.ones(T, device=self._device) / T
        
        # Cache for current sampling probabilities (avoid recomputing every call)
        self._cached_probs: Optional[torch.Tensor] = None
        self._cache_valid = False
    
    def _compute_adaptive_probs(self) -> torch.Tensor:
        """Compute current sampling probabilities based on loss history."""
        # Apply power transform to prevent outlier dominance
        # sqrt (power=0.5) balances between uniform and pure loss-proportional
        adaptive_weights = (self.loss_history + 1e-8).pow(0.5)
        adaptive_probs = adaptive_weights / adaptive_weights.sum()
        
        # Blend Min-SNR foundation with adaptive adjustment
        # exploration_ratio controls how much we trust Min-SNR vs loss history
        probs = (
            (1 - self.exploration_ratio) * self.min_snr_probs + 
            self.exploration_ratio * adaptive_probs
        )
        
        # Apply floor probability to prevent any timestep from being ignored
        probs = torch.clamp(probs, min=self.floor_prob)
        
        # Re-normalize after clamping
        probs = probs / probs.sum()
        
        return probs
    
    def _get_sampling_probs(self, device: torch.device) -> torch.Tensor:
        """Get current sampling probabilities, using cache when valid."""
        # During warmup, use pure Min-SNR
        if self.step_count < self.warmup_steps:
            return self.min_snr_probs.to(device)
        
        # After warmup, use cached adaptive probs or recompute
        if not self._cache_valid:
            self._cached_probs = self._compute_adaptive_probs()
            self._cache_valid = True
        
        return self._cached_probs.to(device)
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps according to current importance distribution.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to place the sampled timesteps on
            
        Returns:
            Tensor of sampled timestep indices [batch_size]
        """
        probs = self._get_sampling_probs(device)
        return torch.multinomial(probs, batch_size, replacement=True)
    
    def update_loss_history(
        self, 
        timesteps: torch.Tensor, 
        losses: torch.Tensor
    ) -> None:
        """
        Update loss history with new observations using EMA smoothing.
        
        This should be called periodically (not every batch) to reduce overhead.
        The sampler tracks step_count internally to manage update frequency.
        
        Args:
            timesteps: Tensor of timestep indices [batch_size]
            losses: Tensor of per-sample losses [batch_size]
        """
        self.step_count += 1
        
        # Only update on specified frequency to reduce overhead
        if self.step_count % self.update_frequency != 0:
            return
        
        # Move to CPU for accumulation (loss_history lives on init device)
        timesteps_cpu = timesteps.detach().cpu()
        losses_cpu = losses.detach().cpu()
        
        # Update loss history with EMA for each observed timestep
        for t_idx, loss_val in zip(timesteps_cpu, losses_cpu):
            t = t_idx.item()
            old_val = self.loss_history[t]
            new_val = loss_val.item()
            
            # EMA update: new = decay * old + (1 - decay) * new
            self.loss_history[t] = self.ema_decay * old_val + (1 - self.ema_decay) * new_val
        
        # Invalidate cache so next sample() recomputes probabilities
        self._cache_valid = False
    
    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about the sampler state.
        
        Returns:
            Dictionary with sampler statistics for logging
        """
        probs = self._get_sampling_probs(self._device)
        
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
        """Move sampler tensors to specified device."""
        self._device = device
        self.min_snr_probs = self.min_snr_probs.to(device)
        self.loss_history = self.loss_history.to(device)
        self.uniform_prior = self.uniform_prior.to(device)
        if self._cached_probs is not None:
            self._cached_probs = self._cached_probs.to(device)
        return self
