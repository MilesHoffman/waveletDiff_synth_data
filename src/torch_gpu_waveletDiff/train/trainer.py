"""
Wavelet Diffusion Training Backend using PyTorch Lightning Trainer.

This module provides a notebook-friendly interface that mirrors the source repo's
training approach using pl.Trainer (instead of Fabric).
"""

import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS"] = "0"

import sys
import torch
import time
import pandas as pd
import multiprocessing
import logging

# import logging
# torch._inductor.config.disable_progress = True
# torch._logging.set_logs(inductor=logging.ERROR, dynamo=logging.ERROR)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Timer, ModelCheckpoint, Callback
from torch.utils.data import TensorDataset, DataLoader

# Fix import paths for "strict copy" modules
current_file_path = os.path.abspath(__file__)
src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
copied_src_path = os.path.join(src_root, 'src', 'copied_waveletDiff', 'src')

if copied_src_path not in sys.path:
    sys.path.insert(0, copied_src_path)
    print(f"Added {copied_src_path} to sys.path for strict-copy compatibility.")

from src.copied_waveletDiff.src.data.loaders import create_sliding_windows
from src.copied_waveletDiff.src.data.module import WaveletTimeSeriesDataModule


class EpochProgressCallback(Callback):
    """Custom callback for epoch-level logging."""
    
    def __init__(self, log_interval=1):
        super().__init__()
        self.log_interval = log_interval
    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_interval == 0:
            avg_loss = trainer.callback_metrics.get('train_loss', None)
            if avg_loss is not None:
                print(f"Epoch {trainer.current_epoch} | Loss: {avg_loss:.6f}")


def setup_environment(matmul_precision="medium", seed=None):
    """
    Sets up the training environment (precision, seeds).
    Matches source repo's train.py setup.
    """
    # Set matmul precision (source uses 'medium')
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision(matmul_precision)
            print(f"Matmul precision set to: {matmul_precision}")
        except Exception as e:
            print(f"Could not set matmul precision: {e}")
    
    # Optional: Set seed for reproducibility
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")


def get_datamodule(repo_dir, dataset_name="stocks", seq_len=24, batch_size=512,
                   wavelet_type="db2", wavelet_levels="auto", data_path=None):
    """
    Creates the WaveletTimeSeriesDataModule.
    Returns the datamodule and config dict.
    """
    # Handle paths
    if data_path:
        stocks_path = os.path.join(repo_dir, data_path)
    else:
        stocks_path = os.path.join(repo_dir, 'src', 'copied_waveletDiff', 'data', 'stocks', 'stock_data.csv')
    
    data_dir = os.path.dirname(stocks_path)
    
    print(f"Loading data from {stocks_path}...")
    
    df = pd.read_csv(stocks_path)
    # Source repo uses ALL columns via df.values (no filtering)
    
    # Create windows
    custom_data_windows, _ = create_sliding_windows(
        df.values,
        seq_len=seq_len,
        normalize=True
    )

    full_data_tensor = torch.FloatTensor(custom_data_windows)
    
    # Build config
    config = {
        'dataset': {'name': dataset_name, 'seq_len': seq_len},
        'training': {'batch_size': batch_size, 'epochs': 1},  # epochs placeholder
        'data': {'data_dir': data_dir, 'normalize_data': False},
        'wavelet': {'type': wavelet_type, 'levels': wavelet_levels}
    }

    # Create DataModule
    datamodule = WaveletTimeSeriesDataModule(config=config, data_tensor=full_data_tensor)
    
    return datamodule, config


def init_model(datamodule, config,
               embed_dim=256, num_heads=8, num_layers=8, time_embed_dim=128,
               dropout=0.1, prediction_target="noise", use_cross_level_attention=True,
               learning_rate=2e-4, compile_mode=None, compile_fullgraph=False,
               compile_cache_dir=None):
    """
    Initializes the WaveletDiffusionTransformer model.
    Returns the model and updated config.
    """
    import os
    import tarfile
    
    # Use a fast local directory for compilation to avoid Google Drive FUSE overhead
    local_cache_dir = "/tmp/torch_compile_cache"
    bundle_name = "torch_compile_cache.tar.gz"
    
    if compile_cache_dir:
        bundle_path = os.path.join(compile_cache_dir, bundle_name)
        if os.path.exists(bundle_path):
            print(f"Extracting compilation cache from Drive: {bundle_path}...")
            try:
                os.makedirs("/tmp", exist_ok=True)
                with tarfile.open(bundle_path, "r:gz") as tar:
                    tar.extractall(path="/tmp")
                print(f"✅ Cache extracted to {local_cache_dir}")
            except Exception as e:
                print(f"Failed to extract cache bundle: {e}. Will recompile locally.")
        
        # Point torch.compile to the local directory
        os.makedirs(local_cache_dir, exist_ok=True)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = local_cache_dir
        os.environ['TRITON_CACHE_DIR'] = local_cache_dir

    import importlib
    import src.copied_waveletDiff.src.models.transformer as models_transformer
    import src.copied_waveletDiff.src.models.wavelet_losses as models_wavelet_losses
    import src.copied_waveletDiff.src.models.layers as models_layers
    import src.copied_waveletDiff.src.models.attention as models_attention
    
    # Force reload
    print("Reloading model modules...")
    importlib.reload(models_layers)
    importlib.reload(models_attention)
    importlib.reload(models_wavelet_losses)
    importlib.reload(models_transformer)
    
    from src.copied_waveletDiff.src.models.transformer import WaveletDiffusionTransformer

    # Update config with model hyperparams (matching source repo structure)
    config.update({
        'model': {
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'time_embed_dim': time_embed_dim,
            'dropout': dropout,
            'prediction_target': prediction_target
        },
        'attention': {'use_cross_level_attention': use_cross_level_attention},
        'noise': {'schedule': "exponential"},
        'sampling': {'ddim_eta': 0.0, 'ddim_steps': None},
        'energy': {'weight': 0.0},
        'optimizer': {
            'scheduler_type': 'onecycle',
            'lr': learning_rate,
            'warmup_epochs': 50,
            'cosine_eta_min': 1e-6
        }
    })

    # Instantiate model
    model = WaveletDiffusionTransformer(data_module=datamodule, config=config)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("WAVELET DIFFUSION TRANSFORMER MODEL INFO")
    print("="*60)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Input dimension: {datamodule.get_input_dim()}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("="*60 + "\n")

    # Optional: Compile model's forward pass ONLY (not the entire module)
    # This is critical for CUDAGraphs compatibility with PL's training_step
    # - training_step has dynamic control flow (buffer allocation, hasattr checks) that breaks CUDAGraphs
    # - forward() is pure tensor math that CUDAGraphs can optimize
    if compile_mode and torch.cuda.is_available():
        print(f"Applying torch.compile to model.forward (mode='{compile_mode}', fullgraph={compile_fullgraph})...")
        try:
            model.forward = torch.compile(model.forward, mode=compile_mode, fullgraph=compile_fullgraph)
            print("model.forward compiled successfully.")
        except Exception as e:
            print(f"Compilation failed: {e}. Using eager mode.")
    
    return model, config


def train(model, datamodule, config,
          num_epochs=100,
          precision="32",
          gradient_clip_val=1.0,
          log_every_n_steps=50,
          checkpoint_dir=None,
          save_every_n_epochs=None,
          enable_progress_bar=True):
    """
    Trains the model using pl.Trainer (matches source repo exactly).
    
    Args:
        model: WaveletDiffusionTransformer instance
        datamodule: WaveletTimeSeriesDataModule instance
        config: Configuration dict
        num_epochs: Number of training epochs
        precision: "32" for FP32, "bf16-mixed" for mixed precision
        gradient_clip_val: Max gradient norm for clipping
        log_every_n_steps: Steps between logging
        checkpoint_dir: Directory to save checkpoints (None = no saving)
        save_every_n_epochs: Save checkpoint every N epochs
        enable_progress_bar: Show progress bar
    """
    # Update config with epochs
    config['training']['epochs'] = num_epochs
    
    # Callbacks
    callbacks = [Timer(), EpochProgressCallback(log_interval=1)]
    
    if checkpoint_dir and save_every_n_epochs:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='waveletdiff-{epoch:04d}',
            save_top_k=-1,  # Save all
            every_n_epochs=save_every_n_epochs
        )
        callbacks.append(checkpoint_callback)
    
    # Create trainer (matching source repo's train.py)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        precision=precision,
        callbacks=callbacks,
        enable_checkpointing=checkpoint_dir is not None,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        detect_anomaly=False,
        logger=False
    )
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Precision: {precision}")
    print(f"Gradient clip: {gradient_clip_val}")
    
    # Train!
    start_time = time.time()
    trainer.fit(model, datamodule)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f}s ({training_time/60:.1f} min)")
    
    return trainer, model


def persist_compilation_cache(compile_cache_dir):
    """
    Bundles the local compilation cache and saves it to Google Drive as a single archive.
    This avoids the slow process of writing 500+ small files to GDrive FUSE.
    """
    import os
    import tarfile
    
    if not compile_cache_dir:
        print("No compile_cache_dir provided. Skipping persistence.")
        return
        
    local_cache_dir = "/tmp/torch_compile_cache"
    bundle_name = "torch_compile_cache.tar.gz"
    bundle_path = os.path.join(compile_cache_dir, bundle_name)
    
    if not os.path.exists(local_cache_dir) or not os.listdir(local_cache_dir):
        print(f"Local cache directory {local_cache_dir} is empty or not found. Nothing to persist.")
        return
        
    print(f"Bundling compilation cache to: {bundle_path}...")
    try:
        os.makedirs(compile_cache_dir, exist_ok=True)
        # Archive the local directory
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(local_cache_dir, arcname="torch_compile_cache")
        print(f"✅ Compilation cache successfully persisted to Drive ({os.path.getsize(bundle_path)/1024/1024:.2f} MB).")
    except Exception as e:
        print(f"Failed to persist compilation cache: {e}")
