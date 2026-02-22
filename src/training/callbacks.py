"""
Custom PyTorch Lightning callbacks for WaveletDiff training.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from copy import deepcopy

class EMACallback(Callback):
    """
    Maintains an Exponential Moving Average (EMA) of model weights.
    
    The EMA weights are updated continuously during training. When checkpoints
    are saved, the 'ema_state_dict' is injected into the checkpoint for later
    use during inference or evaluation.
    """
    def __init__(self, decay: float = 0.9999, use_ema_for_validation: bool = True):
        super().__init__()
        self.decay = decay
        self.use_ema_for_validation = use_ema_for_validation
        self.ema_model = None
        self.original_state_dict = None
        # We use SWA utils which provides a robust EMA implementation
        self.ema_avg_fn = get_ema_multi_avg_fn(self.decay)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize the EMA model with the starting weights."""
        if self.ema_model is None:
            # Create the averaged model
            self.ema_model = AveragedModel(pl_module, multi_avg_fn=self.ema_avg_fn, use_buffers=True)
            print(f"Initialized EMACallback with decay rate {self.decay}")
            
        # ALWAYS move to device on fit start (e.g. recovering from a loaded checkpoint)
        self.ema_model.to(pl_module.device)

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx
    ) -> None:
        """Update EMA weights after each training step."""
        if self.ema_model is not None:
            self.ema_model.update_parameters(pl_module)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Swap to EMA weights before validation."""
        if self.use_ema_for_validation and self.ema_model is not None:
            # Save original state dict to device memory (faster but requires more VRAM)
            self.original_state_dict = {k: v.clone() for k, v in pl_module.state_dict().items()}
            
            # Extract underlying module state
            ema_state = self.ema_model.module.state_dict()
            pl_module.load_state_dict(ema_state)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore original weights after validation."""
        if self.use_ema_for_validation and self.original_state_dict is not None:
            # Restore state instantly from VRAM
            pl_module.load_state_dict(self.original_state_dict)
            self.original_state_dict = None

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> None:
        """Inject the EMA state dict into the saved checkpoint."""
        if self.ema_model is not None:
            # Save the underlying module's state dict, not the AveragedModel wrapper
            checkpoint["ema_state_dict"] = self.ema_model.module.state_dict()

    def on_load_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> None:
        """Restore EMA model state if it exists in the checkpoint."""
        if "ema_state_dict" in checkpoint:
            # Need to initialize ema_model if it doesn't exist yet
            if self.ema_model is None:
                self.ema_model = AveragedModel(pl_module, multi_avg_fn=self.ema_avg_fn, use_buffers=True)
            
            # Load the state into the wrapper's module
            self.ema_model.module.load_state_dict(checkpoint["ema_state_dict"])
            print("Successfully loaded ema_state_dict from checkpoint.")
