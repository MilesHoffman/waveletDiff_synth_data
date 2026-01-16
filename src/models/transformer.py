"""
Core WaveletDiffusionTransformer model.

This module contains the main transformer model for wavelet-based diffusion
without plotting functionality (kept separate for clean architecture).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os

from .layers import TimeEmbedding, WaveletLevelTransformer
from .attention import CrossLevelAttention
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR, OneCycleLR
from utils.noise_schedules import get_noise_schedule
from utils.timestep_sampler import HybridTimestepSampler


# Hyperparameters
T = 1000


class WaveletDiffusionTransformer(pl.LightningModule):
    """Main wavelet diffusion transformer model.
    
    This class handles the core model architecture and training logic,
    while visualization and plotting functionality is kept in separate modules.
    """
    
    def __init__(self, data_module=None, config=None, **kwargs):
        # Enforce config usage
        if config is None:
            raise ValueError("WaveletDiffusionTransformer now requires 'config' to be provided.")
        
        embed_dim = config['model']['embed_dim']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        time_embed_dim = config['model']['time_embed_dim']
        dropout = config['model']['dropout']
        prediction_target = config['model']['prediction_target']
        use_cross_level_attention = config['attention']['use_cross_level_attention']
        ddim_eta = config['sampling']['ddim_eta']
        ddim_steps = config['sampling']['ddim_steps']
        energy_weight = config['energy']['weight']
        max_epochs = config['training']['epochs']
        noise_schedule = config['noise']['schedule']
        scheduler_type = config['optimizer']['scheduler_type']
        warmup_epochs = config['optimizer']['warmup_epochs']
        lr = config['optimizer']['lr']
        cosine_eta_min = kwargs.get('cosine_eta_min', 1e-6)
        plateau_patience = kwargs.get('plateau_patience', 50)
        plateau_factor = kwargs.get('plateau_factor', 0.7)
        plateau_factor = kwargs.get('plateau_factor', 0.7)
        super().__init__()
        self.data_module = data_module
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        self.prediction_target = prediction_target  # "noise" or "coefficient"
        self.use_cross_level_attention = use_cross_level_attention
        self.max_epochs = max_epochs
        self.log_every_n_epochs = config['training'].get('log_every_n_epochs', 1)
        self.compile_config = config.get('compile', {})
        
        # Store the number of diffusion timesteps as an instance attribute
        self.T = T
        
        # Noise schedule parameters
        self.noise_schedule = noise_schedule
        
        # Scheduler parameters
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.cosine_eta_min = cosine_eta_min
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        
        # optimizer parameters
        self.weight_decay = 1e-5
        self.onecycle_max_lr = 1e-3
        self.onecycle_pct_start = 0.3
        
        # Step-based scheduling helpers
        self.steps_per_epoch = None
        self.total_training_steps = None
        
        self.ddim_eta = ddim_eta
        self.ddim_steps = ddim_steps
        
        # Get wavelet coefficient structure from data module
        if data_module is not None:
            wavelet_info = data_module.get_wavelet_info()
            self.dataset_name = data_module.dataset_name
            self.input_dim = data_module.get_input_dim()
            self.coeffs_shapes = wavelet_info['coeffs_shapes']
            self.levels = wavelet_info['levels']
            # Ensure strictly int types for compile compatibility (no numpy scalars)
            self.level_dims = [int(x) for x in wavelet_info['level_dims']]
            self.level_start_indices = [int(x) for x in wavelet_info['level_start_indices']]
            self.num_features = wavelet_info['n_features']
        else:
            raise ValueError("Data module must be provided to initialize wavelet structure")
        
        # Initialize wavelet loss function
        self.energy_weight = energy_weight
        self.use_energy_term = (energy_weight > 0)
        from .wavelet_losses import get_wavelet_loss_fn
        self.wavelet_loss_fn = get_wavelet_loss_fn(
            level_dims=self.level_dims,
            level_start_indices=self.level_start_indices,
            strategy="coefficient_weighted",
            approximation_weight=2.0,
            use_energy_term=(energy_weight > 0),
            energy_weight=energy_weight
        )

        if energy_weight > 0:
            print(f"Using coefficient_weighted loss strategy with energy term:")
            print(f"  Energy weight: {energy_weight}")
        else:
            print(f"Using coefficient_weighted loss strategy (no energy term)")

        # Create separate transformers for each wavelet level
        self.level_transformers = nn.ModuleList()
        for i, dim in enumerate(self.level_dims):
            # Different configurations for different levels
            # Approximation coefficients (level 0) might need more capacity
            level_embed_dim = embed_dim * 2 if i == 0 else embed_dim
            level_num_layers = num_layers + 2 if i == 0 else num_layers
            
            # Create wavelet level transformer
            level_transformer = WaveletLevelTransformer(
                level_dim=dim,
                num_features=self.num_features,
                embed_dim=level_embed_dim,
                num_heads=num_heads,
                num_layers=level_num_layers,
                time_embed_dim=time_embed_dim,
                dropout=dropout
            )
            self.level_transformers.append(level_transformer)
        
        # Time embedding network
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Cross-level attention mechanism
        if self.use_cross_level_attention:
            # Get the actual embedding dimensions used by each level transformer
            level_embed_dims = []
            for i, dim in enumerate(self.level_dims):
                base_embed_dim = embed_dim * 2 if i == 0 else embed_dim
                level_embed_dims.append(base_embed_dim)
            
            self.cross_level_attention = CrossLevelAttention(
                level_embed_dims=level_embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                time_embed_dim=time_embed_dim,
                attention_mode="cross_only"
            )
        else:
            self.cross_level_attention = None
        
        print(f"Created {len(self.level_transformers)} level-specific transformers (Channel-based):")
        for i, dim in enumerate(self.level_dims):
            embed_dim_used = embed_dim * 2 if i == 0 else embed_dim
            layers_used = num_layers + 2 if i == 0 else num_layers
            print(f"  Level {i}: {dim} coefficients, {embed_dim_used} embed_dim, {layers_used} layers")
        
        if self.use_cross_level_attention:
            print(f"Cross-level attention enabled with common dimension: {self.cross_level_attention.common_dim}")
        else:
            print("Cross-level attention disabled")
        
        self.apply(self._init_weights)
        
        # Print detailed model information
        self.print_model_info()
        
        # Initialize noise schedule
        self._initialize_noise_schedule()
        
        # Register diffusion parameters as buffers (automatically handles device placement)
        self.register_buffer('beta_all', self.schedule_params["beta_all"].clone())
        self.register_buffer('alpha_all', self.schedule_params["alpha_all"].clone())
        self.register_buffer('alpha_bar_all', self.schedule_params["alpha_bar_all"].clone())
        
        # Initialize hybrid timestep sampler for importance-weighted sampling
        # Combines Min-SNR-γ stability with adaptive loss-history optimization
        self.timestep_sampler = HybridTimestepSampler(
            alpha_bar_all=self.alpha_bar_all,
            T=self.T,
            warmup_steps=5000,       # Use Min-SNR for first 5000 steps
            gamma=5.0,               # Min-SNR clamping (paper default)
            exploration_ratio=0.3,   # 70% Min-SNR, 30% adaptive
            floor_prob=0.001,        # Minimum sampling probability
            ema_decay=0.995,         # Smooth loss history updates
            update_frequency=10      # Update history every 10 batches
        )
        
        # Loss tracking
        self.training_losses = []
        self.epoch_losses = []

    def _initialize_noise_schedule(self):
        """Initialize the noise schedule based on the specified type."""
        self.schedule_params = get_noise_schedule(
            schedule_type=self.noise_schedule,
            timesteps=T
        )
        print(f"Initialized {self.noise_schedule} noise schedule")

    def _init_weights(self, m):
        """Initialize model weights using transformer-optimized initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def print_model_info(self):
        """Print detailed model information."""
        
        print(f"\n{'='*60}")
        print(f"WAVELET DIFFUSION TRANSFORMER MODEL INFO")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Input dimension: {self.input_dim}")
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Time embedding dimension: {self.time_embed_dim}")
        print(f"Prediction target: {self.prediction_target}")
        print(f"Noise schedule: {self.noise_schedule}")
        print(f"Cross-level attention: {'Enabled' if self.use_cross_level_attention else 'Disabled'}")
        print(f"Number of wavelet levels: {len(self.level_dims)}")
        
        print(f"\nWavelet level details:")
        if self.coeffs_shapes is not None:
            for i, (dim, shape) in enumerate(zip(self.level_dims, self.coeffs_shapes)):
                print(f"  Level {i}: {dim} coefficients, shape {shape}")
        else:
            for i, dim in enumerate(self.level_dims):
                print(f"  Level {i}: {dim} coefficients")

        print(f"\nEnergy Loss Configuration:")
        print(f"Energy term enabled: {self.energy_weight > 0}")
        if self.energy_weight > 0:
            print(f"  Energy weight: {self.energy_weight}")
        
        print(f"{'='*60}\n")

    def forward(self, x, t):
        """Forward pass through level-specific transformers with optional cross-level attention."""
        # x shape: (batch_size, total_coeffs_per_feature, num_features)
        # Last "coefficient" contains time embedding in first feature position
        batch_size, total_coeffs_plus_time, num_features = x.shape
        
        # Create time embedding
        time_embed = self.time_embedding(t)
        
        if self.use_cross_level_attention:
            # Process each level and collect intermediate embeddings for cross-level attention
            level_embeddings = []
            level_coeffs_list = []
            
            for i, (start_idx, dim) in enumerate(zip(self.level_start_indices, self.level_dims)):
                # Extract coefficients for this level from all features
                end_idx = start_idx + dim
                level_coeffs = x[:, start_idx:end_idx, :].contiguous()  # (batch_size, dim, num_features)
                level_coeffs_list.append(level_coeffs)

                # Get intermediate embeddings
                level_embedding = self.level_transformers[i].get_embeddings(level_coeffs, time_embed)
                level_embeddings.append(level_embedding)
            
            # Apply cross-level attention
            cross_attended_embeddings = self.cross_level_attention(level_embeddings, time_embed)
            
            # Apply final projections to get outputs
            level_outputs = []
            for i, (cross_attended, transformer) in enumerate(zip(cross_attended_embeddings, self.level_transformers)):
                # Apply final projection from the level transformer
                level_output = transformer.final_projection(cross_attended)
                level_outputs.append(level_output)
        else:
            # Process each level separately
            level_outputs = []
            for i, (start_idx, dim) in enumerate(zip(self.level_start_indices, self.level_dims)):
                # Extract coefficients for this level
                end_idx = start_idx + dim
                level_coeffs = x[:, start_idx:end_idx, :].contiguous()
                
                # Process through level-specific transformer
                level_output = self.level_transformers[i](level_coeffs, time_embed)
                level_outputs.append(level_output)
        
        # Concatenate all level outputs
        return torch.cat(level_outputs, dim=1)  # (batch_size, total_coeffs_per_feature, num_features)

    def compute_forward_process(self, x_0, t):
        """Compute forward diffusion process (add noise)."""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar_all[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def compute_loss(self, x_0, t):
        """Compute training loss."""
        x_t, noise = self.compute_forward_process(x_0, t)
        t_norm = (t.float() / self.T).clone() # Clone to prevent CUDAGraphs overwrite
        prediction = self(x_t, t_norm)
        
        if self.prediction_target == "noise":
            target = noise
        elif self.prediction_target == "coefficient":
            target = x_0
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
        
        return self.wavelet_loss_fn(target, prediction)
    
    def compute_loss_with_per_sample(self, x_0, t):
        """
        Compute training loss and per-sample losses for timestep sampler updates.
        
        Returns:
            total_loss: Scalar loss for backpropagation
            per_sample_losses: [batch_size] tensor of per-sample losses
        """
        x_t, noise = self.compute_forward_process(x_0, t)
        t_norm = (t.float() / self.T).clone()
        prediction = self(x_t, t_norm)
        
        if self.prediction_target == "noise":
            target = noise
        elif self.prediction_target == "coefficient":
            target = x_0
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
        
        # Total loss for backpropagation
        total_loss = self.wavelet_loss_fn(target, prediction)
        
        # Per-sample MSE for timestep sampler (simple approximation)
        # Compute mean over coefficient and feature dimensions
        per_sample_losses = F.mse_loss(target, prediction, reduction='none').mean(dim=(1, 2))
        
        return total_loss, per_sample_losses

    def on_train_start(self):
        """Called at the very beginning of training - initialize device-dependent components."""
        # Ensure timestep sampler tensors are on the correct device BEFORE training
        # This is critical for CUDAGraph compatibility - no .to() calls during training
        self.timestep_sampler.ensure_device(self.device)
        
        # Pre-initialize wavelet loss weights tensor on the correct device
        # This avoids lazy tensor creation during the first training step
        if hasattr(self.wavelet_loss_fn, 'loss_computer'):
            loss_computer = self.wavelet_loss_fn.loss_computer
            if loss_computer._level_weights_tensor is None:
                loss_computer._level_weights_tensor = torch.tensor(
                    loss_computer.level_weights, 
                    device=self.device, 
                    dtype=torch.float32
                )

    def on_train_batch_start(self, batch, batch_idx):
        """Hook called before training step - safe for graph markers."""
        # Mark step start for reduce-overhead mode to prevent CUDAGraphs memory errors
        if self.compile_config.get('enabled', False) and self.compile_config.get('mode') == 'reduce-overhead':
            torch.compiler.cudagraph_mark_step_begin()
            
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Hook called after training step - safe for logging (runs in eager mode)."""
        # Defer logging to here to avoid graph breaks in training_step logic
        if isinstance(outputs, dict):
            if 'nan_rate' in outputs:
                self.log('nan_rate', outputs['nan_rate'], prog_bar=True, on_step=False, on_epoch=True)
            if 'train_loss' in outputs:
                self.log('train_loss', outputs['train_loss'], prog_bar=True, on_step=False, on_epoch=True)
            
            # Update timestep sampler history in eager mode (safe from CUDAGraphs)
            if 'sampling_t' in outputs and 'per_sample_losses' in outputs:
                self.timestep_sampler.update_loss_history(
                    outputs['sampling_t'], 
                    outputs['per_sample_losses']
                )

    def training_step(self, batch, batch_idx):
        """Training step with importance-weighted timestep sampling."""
        x_0 = batch[0]
        
        # Unconditional NaN handling (compile-safe, no control flow)
        x_0 = torch.nan_to_num(x_0, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Use importance-weighted timestep sampling instead of uniform
        t = self.timestep_sampler.sample(x_0.size(0), self.device)
        
        # Compute loss with per-sample breakdown for sampler updates
        loss, per_sample_losses = self.compute_loss_with_per_sample(x_0, t)
        
        # NOTE: We do NOT call update_loss_history here because it has side effects
        # and Python loops that break CUDAGraphs. Instead, we pass the data
        # to on_train_batch_end which runs in eager mode.
        
        # Compile-safe loss stability using torch.where instead of Python if
        loss = torch.where(
            torch.isnan(loss) | torch.isinf(loss),
            torch.tensor(0.01, device=loss.device, dtype=loss.dtype),
            loss
        )
        
        # Track NaN occurrences explicitly
        nan_detected = (torch.isnan(loss) | torch.isinf(loss)).float()
        
        # Return dict for logging in on_train_batch_end (avoids side effects in compiled graph)
        return {
            'loss': loss,         # Required by Lightning for backprop
            'train_loss': loss,   # For logging
            'nan_rate': nan_detected,
            'sampling_t': t,      # For sampler update
            'per_sample_losses': per_sample_losses # For sampler update
        }

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Get epoch loss from callback metrics (set by self.log in training_step)
        epoch_avg = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()
        self.epoch_losses.append(epoch_avg)
        
        # Log LR for progress bar
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, on_epoch=True)

        if self.trainer.is_global_zero:
            # Log every 100 epochs (at the end of the epoch)
            if (self.current_epoch + 1) % 100 == 0:
                self._log_level_losses_epoch_end()

    def _log_level_losses_epoch_end(self):
        """Log level losses using a sample from the training data."""
        try:
            # Get a sample batch from the training dataloader
            sample_batch = next(iter(self.trainer.train_dataloader))
            x_0 = sample_batch[0][:8].to(self.device)  # Use first 8 samples
            t = torch.randint(0, self.T, (x_0.size(0),), device=self.device)
            
            with torch.no_grad():
                x_t, noise = self.compute_forward_process(x_0, t)
                t_norm = t.float() / self.T
                prediction = self(x_t, t_norm)
                
                target = noise if self.prediction_target == "noise" else x_0
                
                # Get individual level losses
                level_losses = self.wavelet_loss_fn.get_level_losses(target, prediction)
                weights = self.wavelet_loss_fn.get_weights()
                
                # Print level loss summary
                print(f"Epoch {self.current_epoch + 1} Level Losses:")
                for i, (loss, weight) in enumerate(zip(level_losses, weights)):
                    print(f"  Level {i}: {loss.item():.6f} (weight: {weight:.4f})")
                
                # Print energy statistics if available
                if self.use_energy_term and hasattr(self.wavelet_loss_fn, 'get_energy_loss'):
                    # Get the energy variables again since they were defined in the energy term block above
                    energy_loss = self.wavelet_loss_fn.get_energy_loss(target, prediction)
                    energy_stats = self.wavelet_loss_fn.get_energy_stats(target, prediction)
                    reconstruction_loss = sum(w * l for w, l in zip(weights, level_losses))
                    total_loss_with_energy = reconstruction_loss + self.energy_weight * energy_loss
                    energy_contribution_pct = (self.energy_weight * energy_loss / total_loss_with_energy * 100).item()
                    
                    print(f"  Energy Loss: {energy_loss.item():.6f}")
                    print(f"  Energy Target Mean: {energy_stats['energy_target_mean']:.6f}")
                    print(f"  Energy Pred Mean: {energy_stats['energy_pred_mean']:.6f}")
                    print(f"  Energy Relative Error: {energy_stats['energy_relative_error']:.6f}")
                    print(f"  Energy Absolute Error: {energy_stats['energy_absolute_error']:.6f}")
                    print(f"  Energy Contribution: {energy_contribution_pct:.1f}% of total loss")
        
        except Exception as e:
            print(f"Could not log level losses: {e}")

    def setup(self, stage=None):
        """Compute steps per epoch for step-based schedulers."""
        # Prefer Lightning's estimate when available
        if self.trainer is not None and getattr(self.trainer, "estimated_stepping_batches", None):
            self.total_training_steps = int(self.trainer.estimated_stepping_batches)
            self.steps_per_epoch = max(1, self.total_training_steps // max(1, self.max_epochs))
            return

        # Fallback: infer from dataloader length
        try:
            if self.trainer is not None and getattr(self.trainer, "datamodule", None) is not None:
                train_dl = self.trainer.datamodule.train_dataloader()
                if train_dl is not None:
                    self.steps_per_epoch = len(train_dl)
        except Exception:
            self.steps_per_epoch = None

        if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
            self.total_training_steps = int(self.max_epochs * self.steps_per_epoch)
        else:
            self.total_training_steps = None

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """Optimization: set_to_none=True is faster than zeroing."""
        optimizer.zero_grad(set_to_none=True)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """
        Override default clipping to handle 'Atomic'/'Fused' optimizer issues.
        PyTorch Lightning's default clipper sometimes errors with FusedAdamW.
        """
        if self.trainer.precision == "16-mixed":
            # Manually unscale if needed (though fused optimizer usually handles it)
            # This is a safe-guard against the RuntimeError
            try:
                self.trainer.precision_plugin.scaler.unscale_(optimizer)
            except Exception:
                pass
        
        # Apply the clipping manually
        if gradient_clip_algorithm == "norm":
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        elif gradient_clip_algorithm == "value":
            torch.nn.utils.clip_grad_value_(self.parameters(), gradient_clip_val)

    def configure_optimizers(self):
        """Configure optimizer and scheduler with multiple options."""
        # Use fused optimizer when CUDA is available (faster on A100)
        use_fused = torch.cuda.is_available() and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
            fused=use_fused
        )
        
        # Ensure step-based quantities are available
        if self.total_training_steps is None:
            # Fallback to avoid None if setup has not run yet
            self.steps_per_epoch = 1
            self.total_training_steps = int(self.max_epochs)
        warmup_steps = int(self.warmup_epochs * max(self.steps_per_epoch, 1))

        if self.scheduler_type == "cosine_warmup":
            # Single LambdaLR that handles both warmup and cosine annealing
            def _cosine_warmup_lambda(current_step: int):
                # Handle degenerate cases safely
                total_steps = max(int(self.total_training_steps or 0), 1)
                warmup = max(int(warmup_steps), 0)
                warmup = min(warmup, total_steps - 1)  # ensure there is at least 1 decay step

                if warmup > 0 and current_step < warmup:
                    # Linear warmup from 0 -> 1
                    return (current_step + 1) / float(warmup)

                # Cosine annealing after warmup
                decay_steps = max(total_steps - warmup, 1)
                progress = (current_step - warmup) / float(decay_steps)
                # Clamp to [0, 1]
                if progress < 0.0:
                    progress = 0.0
                elif progress > 1.0:
                    progress = 1.0

                # Scale between eta_min/lr and 1 using cosine
                min_scale = float(self.cosine_eta_min) / float(self.lr if self.lr != 0 else 1e-12)
                scale = min_scale + (1.0 - min_scale) * 0.5 * (1.0 + np.cos(np.pi * progress))
                return float(scale)

            scheduler = LambdaLR(optimizer, lr_lambda=_cosine_warmup_lambda)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        elif self.scheduler_type == "plateau_warmup":
            # Step-based warmup via LambdaLR + epoch-based ReduceLROnPlateau
            def _warmup_lambda(current_step: int):
                if warmup_steps <= 0:
                    return 1.0
                return min((current_step + 1) / float(warmup_steps), 1.0)

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=_warmup_lambda)
            plateau_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.plateau_factor,
                patience=self.plateau_patience,
                threshold=1e-4
            )
            return [optimizer], [
                {"scheduler": warmup_scheduler, "interval": "step"},
                {"scheduler": plateau_scheduler, "monitor": "train_loss", "interval": "epoch"}
            ]
        
        elif self.scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.onecycle_max_lr,
                total_steps=self.total_training_steps,
                pct_start=self.onecycle_pct_start,
                anneal_strategy='cos'
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        elif self.scheduler_type == "cosine":
            # Original cosine annealing without warmup
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.total_training_steps, 
                eta_min=self.cosine_eta_min
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        elif self.scheduler_type == "plateau":
            # Original plateau scheduler without warmup
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=self.plateau_factor,
                patience=self.plateau_patience,
                threshold=1e-4
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "frequency": 1,
                    "interval": "epoch"
                }
            }
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}. "
                           f"Choose from: 'cosine_warmup', 'plateau_warmup', 'onecycle', 'cosine', 'plateau'")

    def get_cross_level_attention_weights(self, x, t):
        # x shape: (batch_size, total_coeffs_per_feature, num_features)
        # t shape: (batch_size,) - time steps
        if not self.use_cross_level_attention:
            return None
        
        with torch.no_grad():
            batch_size, total_coeffs, num_features = x.shape

            # Create time embedding
            t_norm = t.float() / self.T
            time_embed = self.time_embedding(t)
            
            # Get embeddings from each level
            level_embeddings = []
            for i, (start_idx, dim) in enumerate(zip(self.level_start_indices, self.level_dims)):
                end_idx = start_idx + dim
                level_coeffs = x[:, start_idx:end_idx, :]
                level_embedding = self.level_transformers[i].get_embeddings(level_coeffs, time_embed)
                level_embeddings.append(level_embedding)
            
            # Get cross-level attention weights
            attention_weights = self.cross_level_attention.get_cross_level_attention_weights(
                level_embeddings, time_embed
            )
            
            return attention_weights

