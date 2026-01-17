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
    
    # Compile Options
    parser.add_argument('--compile_enabled', type=str, default='false', help='true or false')
    parser.add_argument('--compile_mode', type=str, default='default', 
                       help='Compile mode: default, reduce-overhead, max-autotune')
    parser.add_argument('--compile_fullgraph', type=str, default='false', help='true or false')
    
    # Checkpoint Options
    parser.add_argument('--save_weights_only', type=str, default='true', help='true or false - if true, optimizer states are excluded (smaller file)')
    
    # Performance Options
    parser.add_argument('--precision', type=str, default='32',
                       help='Training precision: 32, bf16-mixed, 16-mixed')
    parser.add_argument('--matmul_precision', type=str, default='medium',
                       help='Matmul precision: highest, high, medium')

    # Profiling Options
    parser.add_argument('--profile_enabled', type=str, default='false', help='Enable PyTorch Profiler')
    parser.add_argument('--profile_wait_steps', type=int, default=5, help='Steps before profiling starts')
    parser.add_argument('--profile_warmup_steps', type=int, default=3, help='Warmup steps for profiler')
    parser.add_argument('--profile_active_steps', type=int, default=5, help='Steps to actively profile')
    parser.add_argument('--profile_wait_epochs', type=int, default=0, help='Epochs to wait before profiling (adds to wait steps)')

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
    
    # Apply compile overrides
    if args.compile_enabled:
        config['compile']['enabled'] = args.compile_enabled.lower() == 'true'
    if args.compile_mode:
        config['compile']['mode'] = args.compile_mode
    if args.compile_fullgraph:
         config['compile']['fullgraph'] = args.compile_fullgraph.lower() == 'true'
    
    # Apply performance overrides
    if args.precision:
        config['performance']['precision'] = args.precision
    if args.matmul_precision:
        config['performance']['matmul_precision'] = args.matmul_precision
    
    # Set matmul precision for optimized GPU performance
    try:
        torch.set_float32_matmul_precision(config['performance']['matmul_precision'])
        print(f"Set matmul precision to: {config['performance']['matmul_precision']}")
    except Exception as e:
        print(f"Could not set matmul precision: {e}")

    # Enable cuDNN benchmark mode for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("Enabled cuDNN benchmark mode")

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
    
    # Apply torch.compile if enabled
    if config['compile']['enabled']:
        compile_mode = config['compile']['mode']
        print(f"\n" + "="*60)
        print(f"COMPILING MODEL (mode: {compile_mode})")
        print("="*60)
        model = torch.compile(
            model,
            mode=compile_mode,
            dynamic=config['compile']['dynamic'],
            fullgraph=config['compile']['fullgraph'],
            backend=config['compile']['backend']
        )
        print("Model compiled successfully")

    # Training
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Custom epoch-level progress bar callback
    from pytorch_lightning.callbacks import TQDMProgressBar

    # Custom epoch-level progress bar callback
    from pytorch_lightning.callbacks import TQDMProgressBar

    class EpochProgressBar(TQDMProgressBar):
        def __init__(self, log_every_n_epochs=1):
            super().__init__()
            self.main_progress_bar = None
            self.log_every_n_epochs = log_every_n_epochs

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
            train_loss = metrics.get('train_loss', 0.0)
            lr = metrics.get('lr', 0.0)
            
            postfix = {
                "loss": f"{train_loss:.6f}",
                "lr": f"{lr:.2e}"
            }
            self.main_progress_bar.set_postfix(postfix)
            self.main_progress_bar.update(1)
            
            # Safe logging that doesn't break the progress bar
            if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
                self.main_progress_bar.write(f"Epoch {trainer.current_epoch + 1} - Avg Loss: {train_loss:.6f} - LR: {lr:.2e}")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # Override parent to do nothing ensures we don't crash
            # trying to access the uninitialized _train_progress_bar
            pass

        def on_train_end(self, trainer, pl_module):
            if self.main_progress_bar:
                self.main_progress_bar.close()

    callbacks = [Timer()]
    if enable_progress_bar:
        callbacks.append(EpochProgressBar(log_every_n_epochs=config['training']['log_every_n_epochs']))

    # Setup Profiler - Using custom callback to bypass Lightning's broken PyTorchProfiler
    import warnings
    import logging
    
    # Suppress Kineto warnings at the C++ level by filtering the logger
    logging.getLogger("torch.profiler").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Profiler is not initialized.*")
    warnings.filterwarnings("ignore", message=".*kineto.*")
    
    profile_enabled = args.profile_enabled.lower() == 'true'
    profiler_callback = None
    
    if profile_enabled:
        from pytorch_lightning.callbacks import Callback
        
        profiler_dir = experiment_dir / "profiler"
        profiler_dir.mkdir(parents=True, exist_ok=True)
        
        class TorchProfilerCallback(Callback):
            """Custom profiler callback that uses torch.profiler directly."""
            
            def __init__(self, output_dir, wait=0, warmup=1, active=5):
                super().__init__()
                self.output_dir = Path(output_dir)
                self.wait = wait
                self.warmup = warmup
                self.active = active
                self.profiler = None
                self.step_count = 0
                
            def on_train_start(self, trainer, pl_module):
                try:
                    self.profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(
                            wait=self.wait,
                            warmup=self.warmup,
                            active=self.active,
                            repeat=1
                        ),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=False
                    )
                    self.profiler.__enter__()
                    print(f"Profiler started: wait={self.wait}, warmup={self.warmup}, active={self.active}")
                except Exception as e:
                    print(f"Warning: Could not start profiler: {e}")
                    self.profiler = None
                    
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if self.profiler is not None:
                    try:
                        self.profiler.step()
                        self.step_count += 1
                    except Exception:
                        pass
                        
            def on_train_end(self, trainer, pl_module):
                if self.profiler is not None:
                    try:
                        self.profiler.__exit__(None, None, None)
                        print(f"Profiler finished. {self.step_count} steps recorded.")
                        
                        # Find the latest trace file
                        import glob
                        trace_files = glob.glob(str(self.output_dir / "*.json"))
                        if trace_files:
                            latest_trace = max(trace_files, key=os.path.getctime)
                            print(f"Analyzing trace file: {latest_trace}")
                            self.analyze_trace_file(latest_trace)
                        else:
                            print(f"Warning: No trace files found in {self.output_dir}")
                            
                    except Exception as e:
                        print(f"Warning: Profiler cleanup error (non-fatal): {e}")
                    finally:
                        self.profiler = None

            def analyze_trace_file(self, trace_path):
                import json
                try:
                    with open(trace_path, 'r') as f:
                        data = json.load(f)
                    
                    if not isinstance(data, dict) or 'traceEvents' not in data:
                        # Some formats might be a list directly, or invalid
                        print(f"Warning: Could not parse trace format")
                        return

                    events = data['traceEvents']
                    
                    # Calculate GPU Utilization
                    gpu_kernels = [e for e in events if e.get('cat') == 'kernel']
                    total_gpu_time = sum(e.get('dur', 0) for e in gpu_kernels)
                    
                    # Estimate active time from range of events
                    timestamps = [e.get('ts', 0) for e in events if 'ts' in e]
                    if not timestamps:
                        print("No events with timestamps found.")
                        return

                    start_time = min(timestamps)
                    end_time = max(timestamps)
                    total_time = end_time - start_time
                    
                    gpu_utilization = (total_gpu_time / total_time * 100) if total_time > 0 else 0
                    
                    # Filter and aggregate kernels
                    kernel_times = {}
                    for k in gpu_kernels:
                        name = k.get('name', 'unknown')
                        dur = k.get('dur', 0)
                        kernel_times[name] = kernel_times.get(name, 0) + dur
                        
                    top_kernels = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    print("\n" + "="*60)
                    print("PROFILER REPORT")
                    print("="*60)
                    print(f"Total Recorded Time: {total_time/1e6:.2f} s")
                    print(f"Estimated GPU Utilization: {gpu_utilization:.2f}%")
                    print("-" * 30)
                    print("Top 5 Bottleneck Operations (Kernels):")
                    for name, duration in top_kernels:
                         # Clean up name if too long
                        clean_name = name.split("<")[0] if "<" in name else name
                        print(f"  {duration/1000:.2f} ms : {clean_name}")
                    print("="*60 + "\n")
                    
                except Exception as e:
                    print(f"Warning: Could not analyze trace file: {e}")

        
        # Calculate wait steps
        wait_steps = args.profile_wait_steps
        if args.profile_wait_epochs > 0:
            dataset_len = len(data_module.dataset)
            batch_size = config['training']['batch_size']
            steps_per_epoch = max(1, dataset_len // batch_size)
            wait_steps += args.profile_wait_epochs * steps_per_epoch
            print(f"Profiler configured: waiting {wait_steps} steps before profiling")
        
        profiler_callback = TorchProfilerCallback(
            output_dir=profiler_dir,
            wait=wait_steps,
            warmup=args.profile_warmup_steps,
            active=args.profile_active_steps
        )
        callbacks.append(profiler_callback)
        print(f"Custom profiler callback enabled. Output: {profiler_dir}")

    # Setup trainer (no Lightning profiler - using custom callback instead)
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu',
        devices='auto',
        strategy="auto",
        precision=config['performance']['precision'],
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=1.0,
        detect_anomaly=False,
        gradient_clip_algorithm="norm",
        logger=False,
        profiler=None  # Disabled - using custom callback
    )

    
    # Train model
    start_time = time.time()
    trainer.fit(model, data_module)
    training_time = time.time() - start_time
    
    print(f"Training completed in {timedelta(seconds=training_time)}")
    
    # Save model if requested
    if config['training']['save_model']:
        weights_only = args.save_weights_only.lower() == 'true'
        if weights_only:
            print(f"Saving weights-only checkpoint to {model_path} (Optimizer states excluded)...")
        else:
            print(f"Saving full checkpoint to {model_path}...")
            
        trainer.save_checkpoint(str(model_path), weights_only=weights_only)
        print(f"Model saved!")

if __name__ == "__main__":
    main()
