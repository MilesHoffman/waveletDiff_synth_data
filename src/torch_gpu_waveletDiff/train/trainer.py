import os
# CRITICAL: Must be set BEFORE import torch to suppress "Autotune Choices Stats"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS"] = "0"

import sys
import torch
import time
import pandas as pd
import multiprocessing
import json

# Fix import paths for "strict copy" modules which utilize absolute imports (e.g. "from utils import...")
# We need to add 'src/copied_waveletDiff/src' to sys.path
current_file_path = os.path.abspath(__file__) # .../src/torch_gpu_waveletDiff/train/trainer.py
src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))) # .../
copied_src_path = os.path.join(src_root, 'src', 'copied_waveletDiff', 'src')

if copied_src_path not in sys.path:
    # Prepend to ensure it takes precedence or at least is found
    sys.path.insert(0, copied_src_path)
    print(f"Added {copied_src_path} to sys.path for strict-copy compatibility.")
import lightning as L
from lightning.fabric import Fabric
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import logging

# Suppress verbose compilation logs
torch._inductor.config.disable_progress = True
torch._logging.set_logs(inductor=logging.ERROR, dynamo=logging.ERROR)

# Delayed imports to allow sys.path setup in notebook
# These imports will be done inside functions or we assume sys.path is already set when this module is imported.
try:
    from src.copied_waveletDiff.src.data.loaders import create_sliding_windows
    from src.copied_waveletDiff.src.data.module import WaveletTimeSeriesDataModule
    from src.copied_waveletDiff.src.models.transformer import WaveletDiffusionTransformer
except ImportError:
    # This might happen if imported before sys.path is set. 
    # We will handle it by re-importing inside functions if needed, 
    # or just trusting the notebook flow.
    pass

def setup_fabric(accelerator="auto", devices="auto", precision=None, matmul_precision="high"):
    """
    Initializes Lightning Fabric.
    """
    # Set Matmul Precision (Global)
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision(matmul_precision)
            print(f"[Rank 0] Matmul precision set to: {matmul_precision}")
        except Exception as e:
             print(f"[WARNING] Could not set matmul precision: {e}")

    # Dynamic Precision Detection if not provided
    if precision is None:
        is_tpu = 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ
        if is_tpu:
            precision = "bf16-true"
        elif torch.cuda.is_available():
            precision = "bf16-mixed"
        else:
            precision = "bf16-true" # Fallback/CPU

    fabric = Fabric(accelerator=accelerator, devices=devices, precision=precision)
    fabric.launch()
    
    if fabric.is_global_zero:
        print(f"[Rank 0] Fabric initialized on device: {fabric.device} with precision: {precision}")
        
    return fabric

def get_dataloaders(fabric, repo_dir, dataset_name, seq_len, batch_size, wavelet_type="db2", wavelet_levels="auto", data_path=None):
    """
    Loads data and creates Fabric-optimized DataLoaders.
    """
    from src.copied_waveletDiff.src.data.loaders import create_sliding_windows
    from src.copied_waveletDiff.src.data.module import WaveletTimeSeriesDataModule

    if os.path.isabs(data_path):
        stocks_path = data_path
    else:
        stocks_path = os.path.join(repo_dir, data_path)

    # Data Dir is usually the parent of the specific dataset file or a broader folder
    data_dir = os.path.dirname(stocks_path)

    if fabric.is_global_zero:
        print(f"Loading data from {stocks_path}...")

    df = pd.read_csv(stocks_path)
    # Source repo uses ALL columns (Open, High, Low, Close, Adj_Close, Volume)
    # We must match this exactly for shape [..., 6]
    CORE_COLS = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    df_filtered = df[CORE_COLS]

    # Create windows (CPU)
    custom_data_windows, _ = create_sliding_windows(
        df_filtered.values,
        seq_len=seq_len,
        normalize=True
    )

    full_data_tensor = torch.FloatTensor(custom_data_windows)

    # Config
    config = {
        'dataset': {'name': dataset_name, 'seq_len': seq_len},
        'training': {'batch_size': batch_size, 'epochs': 1}, # epochs dummy
        'data': {'data_dir': data_dir, 'normalize_data': False},
        'wavelet': {'type': wavelet_type, 'levels': wavelet_levels}
    }

    # Create DataModule
    datamodule = WaveletTimeSeriesDataModule(config=config, data_tensor=full_data_tensor)

    # XLA/GPU Optimization: Cast to target dtype BEFORE creating dataset
    if fabric.strategy.precision.precision == 'bf16-true' or 'bf16' in str(fabric.strategy.precision):
         # basic check, exact property depends on fabric version
         # safely assuming if setup_fabric decided on bf16, we cast
         if fabric.device.type != "cpu":
             if fabric.is_global_zero: print(f"Optimizing: Casting data to bfloat16 for {fabric.device.type}...")
             full_data_tensor = full_data_tensor.to(torch.bfloat16)

    dataset = TensorDataset(full_data_tensor)

    # Workers
    # Optimization: If dataset is in-memory (TensorDataset), using workers adds IPC overhead. Use 0.
    if isinstance(dataset, TensorDataset):
        num_workers = 0
        if fabric.is_global_zero: print("Using 0 num_workers (Main Process) for in-memory TensorDataset optimization.")
    else:
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(4, max(0, cpu_count - 2))
        if fabric.is_global_zero: print(f"Using {num_workers} num_workers...")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if fabric.device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    )

    loader = fabric.setup_dataloaders(loader)
    
    return loader, datamodule, config

def init_model(fabric, datamodule, config, 
               embed_dim=256, num_heads=8, num_layers=8, time_embed_dim=128, 
               dropout=0.1, prediction_target="noise", use_cross_level_attention=True,
               learning_rate=2e-4, weight_decay=1e-5, max_lr=None, pct_start=0.3, 
               grad_clip_norm=1.0,
               compile_mode="default", compile_fullgraph=False):
    """
    Initializes the WaveletDiffusionTransformer model and optimizer.
    """
    import importlib
    import src.copied_waveletDiff.src.models.transformer as models_transformer
    import src.copied_waveletDiff.src.models.wavelet_losses as models_wavelet_losses
    import src.copied_waveletDiff.src.models.layers as models_layers
    import src.copied_waveletDiff.src.models.attention as models_attention
    
    # Force reload to pick up any code changes (critical for interactive debugging)
    if fabric.is_global_zero:
        print("[Rank 0] Reloading model modules to ensure latest code is used...")
    importlib.reload(models_layers)
    importlib.reload(models_attention)
    importlib.reload(models_wavelet_losses)
    importlib.reload(models_transformer)
    
    from src.copied_waveletDiff.src.models.transformer import WaveletDiffusionTransformer

    # Update Config with Model Hyperparams
    # Ensure nested dicts exist
    if 'training' not in config: config['training'] = {}
    
    config['training']['grad_clip_norm'] = grad_clip_norm
    
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
            'max_lr': max_lr,
            'pct_start': pct_start,
            'warmup_epochs': 5, # Legacy, overriden by pct_start in train_loop implementation below
            'cosine_eta_min': 1e-6
        }
    })

    # Instantiate
    model = WaveletDiffusionTransformer(data_module=datamodule, config=config)
    
    # Model summary info (rank 0 only)
    if fabric.is_global_zero:
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

    # Move to device
    if fabric.is_global_zero: print(f"[Rank 0] Moving model to {fabric.device}...")
    model.to(fabric.device)

    # Compile
    if fabric.device.type == "cuda" and compile_mode not in [None, "none", "False", False]:
        # COMPILE_MODE passed from argument
        if fabric.is_global_zero: 
            print(f"[Rank 0] Applying torch.compile(mode='{compile_mode}', fullgraph={compile_fullgraph})...")
            if compile_mode == "reduce-overhead":
                 print("[Rank 0] Standard Warning: reduce-overhead may cause OOM on cold start. Use 'default' or 'max-autotune' if crashing.")
            print("[Rank 0] Note: You will see a delay at Step 0 while kernels are built.")
        
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=compile_fullgraph)
        except Exception as e:
            print(f"[WARNING] Compilation failed: {e}. Falling back to eager execution.")
    elif fabric.is_global_zero:
        print("[Rank 0] Skipping torch.compile (compile_mode is None/False)...")

    # Optimizer
    use_fused = fabric.device.type == "cuda"
    if fabric.is_global_zero and use_fused: print("[Rank 0] Using Fused AdamW...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=use_fused
    )

    # Fabric Setup
    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method('compute_loss')

    return model, optimizer, config

def train_loop(fabric, model, optimizer, train_loader, config,
               num_epochs=100, epoch_log_interval=1, save_epoch_interval=10,
               checkpoint_dir="checkpoints", enable_profiler=False,
               enable_grad_clipping=True, enable_diagnostics=False):
    """
    Executes the epoch-based training loop (mirrors source repo).
    
    Args:
        num_epochs: Number of full passes through the dataset.
        epoch_log_interval: Epochs between logging (default: 1 = every epoch).
        save_epoch_interval: Epochs between checkpoint saves.
        enable_diagnostics: If True, logs per-level loss breakdown and gradient norms.
    """
    
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    # Profiler settings
    PROFILER_WARMUP = 25
    PROFILER_ACTIVE = 5

    if enable_profiler:
        effective_epochs = 1
        if fabric.is_global_zero:
            print(f"PROFILER ENABLED: Overriding to {effective_epochs} epoch(s)")
    else:
        effective_epochs = num_epochs

    # Scheduler (OneCycleLR needs total_steps)
    max_lr = config['optimizer'].get('max_lr')
    if max_lr is None:
        max_lr = config['optimizer']['lr'] * 4
    
    pct_start = config['optimizer'].get('pct_start', 0.3)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start
    )

    if fabric.is_global_zero:
        print(f"Training for {num_epochs} epochs ({steps_per_epoch} steps/epoch, {total_steps} total steps)")

    model.train()
    global_step = 0

    # Profiler Context
    if enable_profiler:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available(): activities.append(ProfilerActivity.CUDA)
        prof_schedule = schedule(wait=1, warmup=PROFILER_WARMUP, active=PROFILER_ACTIVE, repeat=1)
        handler = tensorboard_trace_handler('./log/profiler')
        profiler_ctx = profile(
            activities=activities, schedule=prof_schedule, on_trace_ready=handler,
            record_shapes=False, profile_memory=False, with_stack=True
        )
    else:
        profiler_ctx = nullcontext()

    # Create Checkpoint Dir
    if fabric.is_global_zero:
        os.makedirs(checkpoint_dir, exist_ok=True)

    with profiler_ctx as prof:
        if fabric.is_global_zero:
            desc = "PROFILING" if enable_profiler else f"Training {num_epochs} Epochs"
            epoch_iterator = tqdm(range(effective_epochs), desc=desc, unit="epoch")
        else:
            epoch_iterator = range(effective_epochs)

        for epoch in epoch_iterator:
            epoch_loss = torch.zeros((), device=fabric.device)
            
            # Inner loop: simple enumeration, no progress bar
            for batch_idx, batch in enumerate(train_loader):
                x_0 = batch[0]

                # Forward & Loss
                optimizer.zero_grad(set_to_none=True)
                with record_function("forward_pass"):
                    t = torch.randint(0, model.T, (x_0.size(0),), device=fabric.device)
                    loss = model.compute_loss(x_0, t)

                # Backward
                with record_function("backward_pass"):
                    fabric.backward(loss)

                # Compute gradient norm for diagnostics
                grad_norm = None
                if enable_diagnostics:
                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm_sq += p.grad.data.norm(2).item() ** 2
                    grad_norm = total_norm_sq ** 0.5

                # Optimization
                with record_function("optimizer_step"):
                    if enable_grad_clipping:
                        clip_norm = config['training'].get('grad_clip_norm', 1.0)
                        fabric.clip_gradients(model, optimizer, max_norm=clip_norm, error_if_nonfinite=False)
                    
                    optimizer.step()
                    scheduler.step()

                epoch_loss += loss.detach()
                global_step += 1

                if enable_profiler:
                    prof.step()
                    if global_step >= PROFILER_WARMUP + PROFILER_ACTIVE + 2:
                        break

            # End of Epoch
            epoch_avg_loss = epoch_loss.item() / steps_per_epoch
            current_lr = scheduler.get_last_lr()[0]
            
            # Update progress bar with latest metrics
            if fabric.is_global_zero:
                epoch_iterator.set_postfix({"loss": f"{epoch_avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
            
            # Log epoch summary at specified interval
            if fabric.is_global_zero and ((epoch + 1) % epoch_log_interval == 0):
                epoch_iterator.write(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_avg_loss:.4f} | LR: {current_lr:.2e}")
                
                if enable_diagnostics:
                    # Compute gradient norm from the last batch
                    if grad_norm is not None:
                        epoch_iterator.write(f"  └─ grad_norm (pre-clip): {grad_norm:.4f}")
                    
                    with torch.inference_mode():
                        t_diag = torch.randint(0, model.T, (x_0.size(0),), device=fabric.device, dtype=torch.long)
                        x_t_diag, noise_diag = model.compute_forward_process(x_0.detach(), t_diag)
                        t_norm_diag = t_diag.float() / model.T
                        prediction_diag = model(x_t_diag, t_norm_diag)
                        target_diag = noise_diag if model.prediction_target == "noise" else x_0.detach()
                        
                        level_losses = model.wavelet_loss_fn.get_level_losses(target_diag, prediction_diag)
                        level_str = " | ".join([f"L{i}:{ll.item():.4f}" for i, ll in enumerate(level_losses)])
                        epoch_iterator.write(f"  └─ level_losses: {level_str}")

            # Epoch Checkpointing
            if (epoch + 1) % save_epoch_interval == 0 and not enable_profiler:
                save_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.ckpt")
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}
                fabric.save(save_path, state)
                
                if fabric.is_global_zero:
                    epoch_iterator.write(f"Saved checkpoint to {save_path}")
                    config_path = os.path.join(checkpoint_dir, "config.json")
                    if not os.path.exists(config_path):
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=4)
                        epoch_iterator.write(f"Saved configuration to {config_path}")

    # Profiling Report
    if enable_profiler and fabric.is_global_zero:
        print("\n" + "="*80)
        print("PROFILING REPORT")
        print("="*80)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        if fabric.device.type == "cuda":
            print("\nCUDA Time:")
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        print("="*80 + "\n")

    # Final Save
    if not enable_profiler:
        last_save_path = os.path.join(checkpoint_dir, "last.ckpt")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": num_epochs}
        fabric.save(last_save_path, state)
        if fabric.is_global_zero:
             print(f"\nSaved final checkpoint to {last_save_path}", flush=True)
             config_path = os.path.join(checkpoint_dir, "config.json")
             with open(config_path, 'w') as f:
                  json.dump(config, f, indent=4)

    print("Training/Profiling Finished.")

