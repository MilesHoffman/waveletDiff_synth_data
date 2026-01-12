import os
import sys
import argparse
import time
from datetime import timedelta
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer, TQDMProgressBar
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
    
    
class EpochProgressBar(TQDMProgressBar):
    """
    Custom progress bar that only shows the main epoch progress,
    hiding the batch-level inner loops.
    """
    def init_train_tqdm(self):
        """Override to disable the training batch bar."""
        bar = super().init_train_tqdm()
        bar.disable = True
        return bar

    def init_validation_tqdm(self):
        """Override to disable the validation batch bar."""
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    def on_train_epoch_end(self, trainer, pl_module):
        """Update the main progress bar with metrics at the end of each epoch."""
        super().on_train_epoch_end(trainer, pl_module)
        
        # Access metrics stored in the model
        metrics = {}
        if hasattr(pl_module, "latest_loss"):
            metrics["loss"] = f"{pl_module.latest_loss:.4f}"
        if hasattr(pl_module, "latest_lr"):
            metrics["lr"] = f"{pl_module.latest_lr:.2e}"
            
        if metrics:
            self.main_progress_bar.set_postfix(metrics)



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
    parser.add_argument('--compile_mode', type=str, default=None,
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile mode (default: None/disabled)')
    parser.add_argument('--log_every_n_epochs', type=int, default=1,
                       help='Print epoch loss every N epochs (default: 1)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (overrides config default of ../data)')
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=50,
                       help='Progress bar refresh rate in steps (default: 50)')
    
    args = parser.parse_args()
    
    # Load configuration with path relative to this script's location
    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir.parent / "configs"
    config_manager = ConfigManager(config_dir=str(config_dir))
    
    # Determine dataset name for dataset-specific config
    dataset_name = args.dataset
    if dataset_name is None:
        # Try to load from base config first
        base_config = config_manager.load(dataset_name=None)
        dataset_name = base_config.get('dataset', {}).get('name', 'etth1')
    
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
    
    # Set logging frequency (always apply, has default)
    # Set logging frequency (always apply, has default)
    config['training']['log_every_n_epochs'] = args.log_every_n_epochs
    
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir

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
    
    # Apply torch.compile if requested
    if args.compile_mode:
        print(f"Compiling model with mode: {args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)
        print("Model compiled successfully")

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
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu',
        devices='auto',
        strategy="auto",
        precision="32",
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[
            Timer(),
            EpochProgressBar(refresh_rate=args.progress_bar_refresh_rate)
        ],
        log_every_n_steps=50,
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
