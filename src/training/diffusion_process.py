"""
Diffusion process implementations for sampling (DDPM/DDIM).

Optimized for torch.compile with reduce-overhead (CUDAGraphs):
- Pre-computed schedule tensors eliminate per-step buffer indexing
- Branchless step functions use tensor masks instead of if/else
- Compiled denoising kernels enable CUDAGraph replay
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union
from utils.noise_generators import generate_noise


class DiffusionSampler(ABC):
    """Abstract base class for diffusion sampling methods."""
    
    def __init__(self, model, T=1000):
        self.model = model
        self.T = T
        self.device = next(model.parameters()).device
        self.onnx_session = getattr(model, 'onnx_session', None)
    
    def _get_prediction(self, x_t, t_norm, scale, conditions, guidance_scale):
        """Get model prediction with optional CFG guidance, routing to ONNX if enabled."""
        if self.onnx_session is not None:
            # 1. Prepare ONNX Input dictionary mapping
            inputs = {
                'x': x_t.cpu().numpy(),
                't': t_norm.cpu().numpy()
            }
            if scale is not None:
                # Ensure batch dimension (num_samples, 1) instead of (num_samples,)
                inputs['scale'] = scale.unsqueeze(1).cpu().numpy() if scale.dim() == 1 else scale.cpu().numpy()
            if conditions is not None:
                # Use profile names if available from the data_module, otherwise default to cond{i}
                profile_names = getattr(self.model.data_module, 'quarter_profile_names', [])
                for i, cond in enumerate(conditions):
                    name = profile_names[i] if i < len(profile_names) else f'cond{i}'
                    # Ensure batch dimension (num_samples, 1) instead of (num_samples,)
                    cond_np = cond.unsqueeze(1).cpu().numpy() if cond.dim() == 1 else cond.cpu().numpy()
                    inputs[name] = cond_np
                    
            # 2. Execute Graph
            ort_outs = self.onnx_session.run(None, inputs)
            
            # 3. Pull output back to PyTorch scope
            pred = torch.from_numpy(ort_outs[0]).to(self.device)
            if guidance_scale is not None and guidance_scale != 1.0 and conditions is not None:
                # To do CFG via ONNX, we must run the uncond pass
                uncond_inputs = inputs.copy()
                # Create a blank list of conditions equivalent to None for the model wrapper
                for i in range(len(conditions)):
                    name = profile_names[i] if 'profile_names' in locals() and i < len(profile_names) else f'cond{i}'
                    uncond_inputs[name] = torch.zeros_like(inputs[name])
                uncond_outs = self.onnx_session.run(None, uncond_inputs)
                uncond_pred = torch.from_numpy(uncond_outs[0]).to(self.device)
                return uncond_pred + guidance_scale * (pred - uncond_pred)
            return pred

        # Native PyTorch routing
        if guidance_scale is not None and guidance_scale != 1.0 and conditions is not None:
            uncond_pred = self.model(x_t, t_norm, scale=scale, conditions=None)
            cond_pred = self.model(x_t, t_norm, scale=scale, conditions=conditions)
            return uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        return self.model(x_t, t_norm, scale=scale, conditions=conditions)
    
    @abstractmethod
    def sample(self, x_t_initial: torch.Tensor, 
               scale: Optional[torch.Tensor] = None,
               conditions: Optional[List[torch.Tensor]] = None,
               guidance_scale: Optional[float] = None,
               store_intermediates: bool = False,
               store_specific_timesteps: Optional[List[int]] = None,
               show_progress: bool = True) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Generate samples using the diffusion process."""
        pass
    
    def generate(self, n_samples: int, input_dim: int, num_features: int, 
                 scale: Optional[torch.Tensor] = None,
                 conditions: Optional[List[torch.Tensor]] = None,
                 guidance_scale: Optional[float] = None,
                 show_progress: bool = True, **kwargs) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Generate new samples from random noise."""
        shape = (n_samples, input_dim, num_features)
        x_t_initial = generate_noise(
            shape, 
            device=self.device, 
            prior=self.model.noise_prior, 
            nu=self.model.nu
        )
        return self.sample(x_t_initial, scale=scale, conditions=conditions,
                           guidance_scale=guidance_scale, show_progress=show_progress, **kwargs)
    
    def reconstruct(self, x_0_original: torch.Tensor, show_progress: bool = True, **kwargs) -> torch.Tensor:
        """Reconstruct samples by adding noise then denoising."""
        batch_size = x_0_original.shape[0]
        t_start = torch.full((batch_size,), self.T, device=self.device)
        x_t_initial, _ = self.model.compute_forward_process(x_0_original, t_start)
        result = self.sample(x_t_initial, store_intermediates=False, show_progress=show_progress, **kwargs)
        if isinstance(result, dict):
            return result[0]
        return result


# ── Branchless denoising kernels (CUDAGraph-safe) ──────────────────────────

def _ddpm_noise_step(x_t, prediction, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t, noise_mask, noise):
    """Branchless DDPM step for noise-prediction models."""
    coeff1 = 1.0 / torch.sqrt(alpha_t)
    coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
    x_mean = coeff1 * (x_t - coeff2 * prediction)
    return x_mean + noise_mask * torch.sqrt(beta_t) * noise


def _ddpm_coeff_step(x_t, prediction, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t, noise_mask, noise):
    """Branchless DDPM step for coefficient-prediction models."""
    coeff1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
    coeff2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
    x_mean = coeff1 * prediction + coeff2 * x_t
    return x_mean + noise_mask * torch.sqrt(beta_t) * noise


def _ddim_noise_step(x_t, prediction, alpha_bar_t, alpha_bar_prev, beta_t, sigma, noise):
    """Branchless DDIM step for noise-prediction models."""
    x_0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * prediction) / torch.sqrt(alpha_bar_t)
    x_next = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1.0 - alpha_bar_prev) * prediction
    return x_next + sigma * noise


def _ddim_coeff_step(x_t, prediction, alpha_bar_t, alpha_bar_prev, beta_t, sigma, noise):
    """Branchless DDIM step for coefficient-prediction models."""
    x_0_pred = prediction
    noise_pred = (x_t - torch.sqrt(alpha_bar_t) * x_0_pred) / torch.sqrt(1.0 - alpha_bar_t)
    x_next = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
    return x_next + sigma * noise


class DDPMSampler(DiffusionSampler):
    """Standard DDPM sampling (stochastic), optimized for torch.compile."""
    
    def _precompute_schedule(self, timesteps):
        """Pre-index all schedule parameters into contiguous tensors."""
        t_prev_list = [timesteps[i + 1] if i < len(timesteps) - 1 else 0 for i in range(len(timesteps))]
        
        alpha_t = self.model.alpha_all[timesteps]
        alpha_bar_t = self.model.alpha_bar_all[timesteps]
        beta_t = self.model.beta_all[timesteps]
        
        # For t_prev=0, alpha_bar_prev should be 1.0 (no noise at t=0)
        alpha_bar_prev = torch.ones(len(timesteps), device=self.device)
        valid_prev_mask = torch.tensor([tp > 0 for tp in t_prev_list], device=self.device)
        valid_prev_indices = torch.tensor([tp if tp > 0 else 0 for tp in t_prev_list], device=self.device, dtype=torch.long)
        alpha_bar_prev[valid_prev_mask] = self.model.alpha_bar_all[valid_prev_indices[valid_prev_mask]]
        
        # noise_mask: 1.0 for all steps except the final one (t_prev=0)
        noise_mask = valid_prev_mask.float()
        
        # Pre-compute normalized timesteps
        t_norms = torch.tensor(timesteps, device=self.device, dtype=torch.float32) / self.T
        
        return {
            'alpha_t': alpha_t,
            'alpha_bar_t': alpha_bar_t,
            'alpha_bar_prev': alpha_bar_prev,
            'beta_t': beta_t,
            'noise_mask': noise_mask,
            't_norms': t_norms,
        }
    
    def sample(self, x_t_initial: torch.Tensor, 
               scale: Optional[torch.Tensor] = None,
               conditions: Optional[List[torch.Tensor]] = None,
               guidance_scale: Optional[float] = None,
               store_intermediates: bool = False,
               store_specific_timesteps: Optional[List[int]] = None,
               show_progress: bool = True) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Sample using DDPM with pre-computed schedule."""
        x_t = x_t_initial.clone()
        timesteps = list(range(self.T, 0, -1))
        total_steps = len(timesteps)
        
        # Pre-compute all schedule values
        sched = self._precompute_schedule(timesteps)
        
        # Select branchless step kernel based on prediction target (constant per model)
        step_fn = _ddpm_noise_step if self.model.prediction_target == "noise" else _ddpm_coeff_step
        
        # Storage
        stored_samples = {}
        if store_intermediates or store_specific_timesteps is not None:
            stored_samples[self.T] = x_t.clone()
        
        if show_progress:
            print(f"Starting DDPM sampling with {total_steps} steps...")
        
        progress_milestones = [0.2, 0.4, 0.6, 0.8, 1.0]
        milestone_idx = 0
        batch_size = x_t.shape[0]
        
        with torch.no_grad():
            for i in range(total_steps):
                torch.compiler.cudagraph_mark_step_begin()
                t_norm = sched['t_norms'][i].expand(batch_size)
                prediction = self._get_prediction(x_t, t_norm, scale, conditions, guidance_scale)
                
                noise = generate_noise(
                    shape=x_t.shape, 
                    device=self.device, 
                    prior=self.model.noise_prior, 
                    nu=self.model.nu
                )
                
                x_t = step_fn(
                    x_t, prediction,
                    sched['alpha_t'][i],
                    sched['alpha_bar_t'][i],
                    sched['alpha_bar_prev'][i],
                    sched['beta_t'][i],
                    sched['noise_mask'][i],
                    noise
                )
                
                # Progress reporting
                if show_progress and milestone_idx < len(progress_milestones):
                    current_progress = (i + 1) / total_steps
                    if current_progress >= progress_milestones[milestone_idx]:
                        percentage = int(progress_milestones[milestone_idx] * 100)
                        print(f"DDPM Sampling: {percentage}% complete ({i + 1}/{total_steps} steps)")
                        milestone_idx += 1
                
                # Store intermediates
                should_store = False
                if store_specific_timesteps is not None and timesteps[i] in store_specific_timesteps:
                    should_store = True
                elif store_intermediates and (timesteps[i] % 100 == 0 or i == total_steps - 1):
                    should_store = True
                if should_store:
                    stored_samples[timesteps[i]] = x_t.clone()
        
        if store_intermediates or store_specific_timesteps is not None:
            stored_samples[0] = x_t.clone()
            return stored_samples
        return x_t


class DDIMSampler(DiffusionSampler):
    """DDIM sampling (deterministic when eta=0), optimized for torch.compile."""
    
    def __init__(self, model, T=1000, eta=0.0, ddim_steps=None):
        super().__init__(model, T)
        self.eta = eta
        self.ddim_steps = ddim_steps
    
    def _get_sampling_timesteps(self) -> List[int]:
        """Get the timesteps to use for DDIM sampling."""
        if self.ddim_steps is not None and self.ddim_steps < self.T:
            step_size = self.T // self.ddim_steps
            timesteps = list(range(self.T, 0, -step_size))
            if timesteps[-1] != 1:
                timesteps.append(1)
            timesteps = sorted(timesteps, reverse=True)
        else:
            timesteps = list(range(self.T, 0, -1))
        return timesteps
    
    def _precompute_schedule(self, timesteps):
        """Pre-index all schedule parameters into contiguous tensors."""
        t_prev_list = [timesteps[i + 1] if i < len(timesteps) - 1 else 0 for i in range(len(timesteps))]
        
        alpha_bar_t = self.model.alpha_bar_all[timesteps]
        beta_t = self.model.beta_all[timesteps]
        
        alpha_bar_prev = torch.ones(len(timesteps), device=self.device)
        valid_prev_mask = torch.tensor([tp > 0 for tp in t_prev_list], device=self.device)
        valid_prev_indices = torch.tensor([tp if tp > 0 else 0 for tp in t_prev_list], device=self.device, dtype=torch.long)
        alpha_bar_prev[valid_prev_mask] = self.model.alpha_bar_all[valid_prev_indices[valid_prev_mask]]
        
        # Pre-compute sigma for each step (branchless: sigma=0 when t_prev=0 or eta=0)
        if self.eta > 0.0:
            sigma = self.eta * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            ) * torch.sqrt(beta_t)
            sigma = sigma * valid_prev_mask.float()
        else:
            sigma = torch.zeros(len(timesteps), device=self.device)
        
        t_norms = torch.tensor(timesteps, device=self.device, dtype=torch.float32) / self.T
        
        return {
            'alpha_bar_t': alpha_bar_t,
            'alpha_bar_prev': alpha_bar_prev,
            'beta_t': beta_t,
            'sigma': sigma,
            't_norms': t_norms,
        }
    
    def sample(self, x_t_initial: torch.Tensor, 
               scale: Optional[torch.Tensor] = None,
               conditions: Optional[List[torch.Tensor]] = None,
               guidance_scale: Optional[float] = None,
               store_intermediates: bool = False,
               store_specific_timesteps: Optional[List[int]] = None,
               show_progress: bool = True) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Sample using DDIM with pre-computed schedule."""
        x_t = x_t_initial.clone()
        timesteps = self._get_sampling_timesteps()
        total_steps = len(timesteps)
        
        # Pre-compute all schedule values
        sched = self._precompute_schedule(timesteps)
        
        # Select branchless step kernel
        step_fn = _ddim_noise_step if self.model.prediction_target == "noise" else _ddim_coeff_step
        
        # Storage
        stored_samples = {}
        if store_intermediates or store_specific_timesteps is not None:
            stored_samples[self.T] = x_t.clone()
            
        if self.eta == 0.0:
            zero_noise = torch.zeros_like(x_t)
        
        if show_progress:
            ddim_type = "accelerated" if self.ddim_steps and self.ddim_steps < self.T else "full"
            print(f"Starting DDIM sampling ({ddim_type}) with {total_steps} steps...")
        
        progress_milestones = [0.2, 0.4, 0.6, 0.8, 1.0]
        milestone_idx = 0
        batch_size = x_t.shape[0]
        
        with torch.no_grad():
            for i in range(total_steps):
                torch.compiler.cudagraph_mark_step_begin()
                t_norm = sched['t_norms'][i].expand(batch_size)
                prediction = self._get_prediction(x_t, t_norm, scale, conditions, guidance_scale).clone()
                
                noise = generate_noise(
                    shape=x_t.shape, 
                    device=self.device, 
                    prior=self.model.noise_prior, 
                    nu=self.model.nu
                )
                
                x_t = step_fn(
                    x_t, prediction,
                    sched['alpha_bar_t'][i],
                    sched['alpha_bar_prev'][i],
                    sched['beta_t'][i],
                    sched['sigma'][i],
                    noise
                )
                
                if show_progress and milestone_idx < len(progress_milestones):
                    current_progress = (i + 1) / total_steps
                    if current_progress >= progress_milestones[milestone_idx]:
                        percentage = int(progress_milestones[milestone_idx] * 100)
                        print(f"DDIM Sampling: {percentage}% complete ({i + 1}/{total_steps} steps)")
                        milestone_idx += 1
                
                should_store = False
                if store_specific_timesteps is not None and timesteps[i] in store_specific_timesteps:
                    should_store = True
                elif store_intermediates and (timesteps[i] % 100 == 0 or i == total_steps - 1):
                    should_store = True
                if should_store:
                    stored_samples[timesteps[i]] = x_t.clone()
        
        if store_intermediates or store_specific_timesteps is not None:
            stored_samples[0] = x_t.clone()
            return stored_samples
        return x_t


class DiffusionTrainer:
    """Utility class for training and evaluation with different sampling methods."""
    
    def __init__(self, model):
        self.model = model
        self.ddpm_sampler = DDPMSampler(model, T=model.T)
        self.ddim_sampler = DDIMSampler(model, T=model.T, eta=model.ddim_eta, ddim_steps=model.ddim_steps)
    
    def generate_samples(self, n_samples: int, use_ddim: bool = False, show_progress: bool = True, **kwargs) -> torch.Tensor:
        """Generate samples using either DDPM or DDIM."""
        sampler = self.ddim_sampler if use_ddim else self.ddpm_sampler
        input_dim = self.model.input_dim
        num_features = self.model.num_features

        result = sampler.generate(n_samples, input_dim, num_features, show_progress=show_progress, **kwargs)

        if isinstance(result, dict):
            return result[0]
        return result
    
    def reconstruct_samples(self, x_0_original: torch.Tensor, use_ddim: bool = True, show_progress: bool = True, **kwargs) -> torch.Tensor:
        """Reconstruct samples using either DDPM or DDIM."""
        sampler = self.ddim_sampler if use_ddim else self.ddpm_sampler
        return sampler.reconstruct(x_0_original, show_progress=show_progress, **kwargs)
