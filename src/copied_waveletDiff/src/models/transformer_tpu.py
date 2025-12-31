"""
Core WaveletDiffusionTransformer model (TPU EXPERIMENTAL).

This module contains the main transformer model for wavelet-based diffusion,
stripped of all XLA-unfriendly operations (NaN checks, .item(), prints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
import os

from .layers import TimeEmbedding, WaveletLevelTransformer
from .attention import CrossLevelAttention
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR, OneCycleLR
from utils.noise_schedules import get_noise_schedule


# Hyperparameters
T = 1000


class WaveletDiffusionTransformerTPU(pl.LightningModule):
    """Main wavelet diffusion transformer model (TPU Optimized).
    
    This version removes all NaN checks, prints, and .item() calls from the training loop.
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
        super().__init__()
        self.data_module = data_module
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        self.prediction_target = prediction_target  # "noise" or "coefficient"
        self.use_cross_level_attention = use_cross_level_attention
        self.max_epochs = max_epochs
        
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
            self.level_dims = wavelet_info['level_dims']
            self.level_start_indices = wavelet_info['level_start_indices']
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
        print(f"WAVELET DIFFUSION TRANSFORMER MODEL INFO (TPU EXPERIMENTAL)")
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
                level_coeffs = x[:, start_idx:end_idx, :]  # (batch_size, dim, num_features)
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
                level_coeffs = x[:, start_idx:end_idx, :]
                
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
        t_norm = t.float() / self.T
        prediction = self(x_t, t_norm)
        
        if self.prediction_target == "noise":
            target = noise
        elif self.prediction_target == "coefficient":
            target = x_0
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
        
        return self.wavelet_loss_fn(target, prediction)

    def training_step(self, batch, batch_idx):
        """Training step (TPU OPTIMIZED). 
        REMOVED: NaN checks, prints, .item() calls.
        """
        x_0 = batch[0]
        
        # REMOVED: Check input for NaN using python control flow
        
        t = torch.randint(0, self.T, (x_0.size(0),), device=self.device)
        loss = self.compute_loss(x_0, t)
        
        # REMOVED: Enhanced loss monitoring and stability checks (NaN/Inf)
        # REMOVED: self.training_losses.append(loss.item())
        
        return loss

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Simple logging only on main process
        if self.trainer.is_global_zero and len(self.training_losses) > 0:
            epoch_avg = np.mean(self.training_losses[-len(self.trainer.train_dataloader):])
            self.epoch_losses.append(epoch_avg)
            print(f"Epoch {self.current_epoch} - Avg Loss: {epoch_avg:.6f}")

    def _log_level_losses_epoch_end(self):
        """Log level losses using a sample from the training data."""
        # Disabled for TPU experiment to avoid any overhead
        pass

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

    def configure_optimizers(self):
        """Configure optimizer and scheduler with multiple options."""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Ensure step-based quantities are available
        if self.total_training_steps is None:
            # Fallback to avoid None if setup has not run yet
            self.steps_per_epoch = 1
            self.total_training_steps = int(self.max_epochs)
        warmup_steps = int(self.warmup_epochs * max(self.steps_per_epoch, 1))

        # TPU Experiment always uses OneCycle for now as per config
        if self.scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.onecycle_max_lr,
                total_steps=self.total_training_steps,
                pct_start=self.onecycle_pct_start,
                anneal_strategy='cos'
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
            
        else:
             # Default fallback
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.total_training_steps, 
                eta_min=self.cosine_eta_min
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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
