"""Multi-objective optimization metrics tracking."""

import numpy as np
from typing import List, Dict


class MultiObjectiveTracker:
    """
    Tracks multiple objectives during training trials.
    
    Objectives:
    1. Training loss (minimize)
    2. Step time (minimize)
    3. Gradient norm variance (minimize - stability metric)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.losses = []
        self.step_times = []
        self.grad_norms = []
    
    def add_step(self, loss: float, step_time: float, grad_norm: float):
        """Record metrics for a single training step."""
        self.losses.append(loss)
        self.step_times.append(step_time)
        self.grad_norms.append(grad_norm)
    
    def get_intermediate_loss(self, window: int = 100) -> float:
        """Get average loss over recent window."""
        if len(self.losses) < window:
            return np.mean(self.losses) if self.losses else float('inf')
        return np.mean(self.losses[-window:])
    
    def get_objectives(self, window: int = 500) -> tuple:
        """
        Calculate final objective values.
        
        Returns:
            (avg_loss, avg_step_time, grad_norm_variance)
        """
        if not self.losses:
            return (float('inf'), float('inf'), float('inf'))
        
        # Use last N steps for final metrics
        n = min(window, len(self.losses))
        
        avg_loss = np.mean(self.losses[-n:])
        avg_step_time = np.mean(self.step_times[-n:])
        grad_norm_variance = np.var(self.grad_norms[-n:])
        
        return (avg_loss, avg_step_time, grad_norm_variance)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get detailed statistics for logging."""
        if not self.losses:
            return {}
        
        return {
            'final_loss': np.mean(self.losses[-500:]) if len(self.losses) >= 500 else np.mean(self.losses),
            'min_loss': np.min(self.losses),
            'avg_step_time_ms': np.mean(self.step_times) * 1000,
            'max_step_time_ms': np.max(self.step_times) * 1000,
            'avg_grad_norm': np.mean(self.grad_norms),
            'max_grad_norm': np.max(self.grad_norms),
            'grad_norm_variance': np.var(self.grad_norms),
            'grad_norm_std': np.std(self.grad_norms),
            'total_steps': len(self.losses),
        }
    
    def has_exploding_gradients(self, threshold: float = 100.0) -> bool:
        """Check if gradients have exploded."""
        if not self.grad_norms:
            return False
        return np.max(self.grad_norms) > threshold
    
    def has_diverged(self, loss_threshold: float = 10.0) -> bool:
        """Check if loss has diverged."""
        if len(self.losses) < 100:
            return False
        recent_avg = np.mean(self.losses[-100:])
        return recent_avg > loss_threshold or np.isnan(recent_avg)
