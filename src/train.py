import os
import sys
import argparse
import time
from datetime import timedelta
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer
import numpy as np

from models import WaveletDiffusionTransformer
from training import DiffusionTrainer
from data import WaveletTimeSeriesDataModule
from utils import ConfigManager

# Set PyTorch precision optimization for modern GPUs
try:
    torch.set_float32_matmul_precision('medium')
    print("Enabled optimized matmul precision")
except Exception as e:
    print(f"Could not set matmul precision: {e}")
    print("Continuing with default precision...")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train WaveletDiff model')
    parser.add_argument('--experiment_name', type=str, default='default_experiment',
                       help='Experiment name/ID for organizing outputs (default: default_experiment)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    parser.add_argument('--seq_len', type=int, default=None,
                       help='Sequence length (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--wavelet_type', type=str, default=None,
                       help='Wavelet type (overrides config)')
    parser.add_argument('--wavelet_levels', type=str, default=None,
                       help='Wavelet levels (overrides config, can be "auto" or integer)')
    
    # Model Architecture Overrides
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--time_embed_dim', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--prediction_target', type=str, default=None)
    
    # Training & Optimizer Overrides
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--scheduler_type', type=str, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--use_cross_level_attention', type=str, default=None, help='true or false')
    parser.add_argument('--energy_weight', type=float, default=None)
    parser.add_argument('--noise_schedule', type=str, default=None)
    parser.add_argument('--log_every_n_epochs', type=int, default=None)
    parser.add_argument('--enable_progress_bar', type=str, default='true', help='true or false')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Determine dataset name for dataset-specific config
    dataset_name = args.dataset
    if dataset_name is None:
        # Try to load from base config first
        # This will now fallback to INTERNAL_DEFAULTS in ConfigManager
        base_config = config_manager.load(dataset_name=None)
        dataset_name = base_config.get('dataset', {}).get('name', 'stocks')
    
    # Load configuration with dataset-specific overrides
    config = config_manager.load(dataset_name=dataset_name)
    
    # Apply command line overrides
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.seq_len:
        config['dataset']['seq_len'] = args.seq_len
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.wavelet_type:
        config['wavelet']['type'] = args.wavelet_type
    if args.wavelet_levels:
        try:
            config['wavelet']['levels'] = int(args.wavelet_levels)
        except ValueError:
            config['wavelet']['levels'] = args.wavelet_levels
            
    # Apply generalized overrides
    if args.embed_dim: config['model']['embed_dim'] = args.embed_dim
    if args.num_heads: config['model']['num_heads'] = args.num_heads
    if args.num_layers: config['model']['num_layers'] = args.num_layers
    if args.time_embed_dim: config['model']['time_embed_dim'] = args.time_embed_dim
    if args.dropout is not None: config['model']['dropout'] = args.dropout
    if args.prediction_target: config['model']['prediction_target'] = args.prediction_target
    
    if args.lr: config['optimizer']['lr'] = args.lr
    if args.scheduler_type: config['optimizer']['scheduler_type'] = args.scheduler_type
    if args.warmup_epochs: config['optimizer']['warmup_epochs'] = args.warmup_epochs
    if args.data_dir: config['data']['data_dir'] = args.data_dir
    
    if args.use_cross_level_attention:
        config['attention']['use_cross_level_attention'] = args.use_cross_level_attention.lower() == 'true'
    
    if args.energy_weight is not None: config['energy']['weight'] = args.energy_weight
    if args.noise_schedule: config['noise']['schedule'] = args.noise_schedule
    if args.log_every_n_epochs: config['training']['log_every_n_epochs'] = args.log_every_n_epochs
    
    enable_progress_bar = args.enable_progress_bar.lower() == 'true'
    
    print(f"Starting WaveletDiff Training")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Sequence Length: {config['dataset']['seq_len']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Prediction Target: {config['model']['prediction_target']}")
    print(f"Cross-level Attention: {'Enabled' if config['attention']['use_cross_level_attention'] else 'Disabled'} (cross_only)")
    print(f"Loss Strategy: coefficient_weighted (approximation_weight=2)")
    print(f"Energy Term: {'Enabled' if config['energy']['weight'] > 0 else 'Disabled'} (level_feature, absolute)")
    print(f"Noise Schedule: {config['noise']['schedule']}")
    print(f"Logging Frequency: every {config['training']['log_every_n_epochs']} epoch(s)")
    
    # Set up data module
    print("\n" + "="*60)
    print("SETTING UP DATA MODULE")
    print("="*60)
    
    data_module = WaveletTimeSeriesDataModule(config=config)
    
    print(f"Data module setup complete")
    print(f"Input dimension: {data_module.get_input_dim()}")
    print(f"Dataset size: {len(data_module.dataset)}")
    print(f"Wavelet: {data_module.wavelet_type} with {data_module.wavelet_info['levels']} levels")

    # Get wavelet info
    wavelet_info = data_module.get_wavelet_info()
    print(f"   Wavelet levels: {wavelet_info['levels']}")
    for i, shape in enumerate(wavelet_info['coeffs_shapes']):
        print(f"     Level {i}: {shape} -> {np.prod(shape)} coefficients")
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = WaveletDiffusionTransformer(data_module=data_module, config=config)

    # Create experiment directories
    dataset_name = config['dataset']['name']
    
    # Create experiment folder structure
    experiment_name = getattr(args, 'experiment_name', 'default_experiment')
    experiment_dir = Path(config['paths']['output_dir']) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = f"checkpoint.ckpt"
    model_path = experiment_dir / model_filename
    
    print(f"Experiment: {experiment_name}")
    print(f"Model checkpoint will be saved to: {model_path}")

    # Training
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Custom epoch-level progress bar callback
    from pytorch_lightning.callbacks import TQDMProgressBar

    class EpochProgressBar(TQDMProgressBar):
        def __init__(self):
            super().__init__()
            self.main_progress_bar = None

        def init_train_tqdm(self):
            # This is called for the batch-level bar, we want to skip it
            return super().init_train_tqdm()

        def on_train_epoch_start(self, trainer, pl_module):
            # No-op per-batch bar
            pass

        def on_train_start(self, trainer, pl_module):
            from tqdm import tqdm
            self.main_progress_bar = tqdm(
                total=trainer.max_epochs,
                desc="Training Progress",
                dynamic_ncols=True,
                unit="epoch"
            )

        def on_train_epoch_end(self, trainer, pl_module):
            # Update the main progress bar
            metrics = trainer.callback_metrics
            postfix = {
                "loss": f"{metrics.get('train_loss', 0.0):.6f}",
                "lr": f"{metrics.get('lr', 0.0):.8f}"
            }
            self.main_progress_bar.set_postfix(postfix)
            self.main_progress_bar.update(1)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # Override parent to do nothing ensures we don't crash
            # trying to access the uninitialized _train_progress_bar
            pass

        def on_train_end(self, trainer, pl_module):
            if self.main_progress_bar:
                self.main_progress_bar.close()

    callbacks = [Timer()]
    if enable_progress_bar:
        callbacks.append(EpochProgressBar())

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu',
        devices='auto',
        strategy="ddp",
        precision="32",
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=1.0,
        detect_anomaly=False,
        gradient_clip_algorithm="norm",
        logger=False
    )
    
    # Train model
    start_time = time.time()
    trainer.fit(model, data_module)
    training_time = time.time() - start_time
    
    print(f"Training completed in {timedelta(seconds=training_time)}")
    
    # Save model if requested
    if config['training']['save_model']:
        trainer.save_checkpoint(str(model_path))
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
