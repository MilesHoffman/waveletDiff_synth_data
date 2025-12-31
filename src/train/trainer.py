import os
import sys
import torch
import time
import pandas as pd
import multiprocessing
import json
import lightning as L
from lightning.fabric import Fabric
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

# Delayed imports to allow sys.path setup in notebook
# These imports will be done inside functions or we assume sys.path is already set when this module is imported.
# To be safe and cleaner, we'll put them at top level but assume user runs setup cell first.
try:
    from data.loaders import create_sliding_windows
    from data.module import WaveletTimeSeriesDataModule
    from models.transformer import WaveletDiffusionTransformer
except ImportError:
    # This might happen if imported before sys.path is set. 
    # We will handle it by re-importing inside functions if needed, 
    # or just trusting the notebook flow.
    pass

def setup_fabric(accelerator="auto", devices="auto", precision=None):
    """
    Initializes Lightning Fabric.
    """
    # Dynamic Precision Detection if not provided
    if precision is None:
        is_tpu = 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ
        if is_tpu:
            precision = "bf16-true"
        elif torch.cuda.is_available():
            precision = "bf16-mixed"
            # Set Matmul Precision for GPUs
            torch.set_float32_matmul_precision('high')
        else:
            precision = "bf16-true" # Fallback/CPU

    fabric = Fabric(accelerator=accelerator, devices=devices, precision=precision)
    fabric.launch()
    
    if fabric.is_global_zero:
        print(f"[Rank 0] Fabric initialized on device: {fabric.device} with precision: {precision}")
        
    return fabric

def get_dataloaders(fabric, repo_dir, dataset_name, seq_len, batch_size, wavelet_type="db2", wavelet_levels="auto"):
    """
    Loads data and creates Fabric-optimized DataLoaders.
    """
    # Re-import inside to be safe
    from data.loaders import create_sliding_windows
    from data.module import WaveletTimeSeriesDataModule

    stocks_path = os.path.join(repo_dir, "WaveletDiff_source", "data", "stocks", "stock_data.csv")
    data_dir = os.path.join(repo_dir, "WaveletDiff_source", "data")

    if fabric.is_global_zero:
        print(f"Loading data from {stocks_path}...")

    df = pd.read_csv(stocks_path)
    CORE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
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
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(4, max(0, cpu_count - 2))
    if fabric.is_global_zero:
        print(f"Using {num_workers} num_workers...")

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
               learning_rate=2e-4, weight_decay=1e-5):
    """
    Initializes the WaveletDiffusionTransformer model and optimizer.
    """
    from models.transformer import WaveletDiffusionTransformer

    # Update Config with Model Hyperparams
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
            'warmup_epochs': 5,
            'cosine_eta_min': 1e-6
        }
    })

    # Instantiate
    model = WaveletDiffusionTransformer(data_module=datamodule, config=config)
    
    # Model summary info (rank 0 only)
    if fabric.is_global_zero:
        print("\n" + "="*60)
        print("WAVELET DIFFUSION TRANSFORMER MODEL INFO")
        print("="*60)
        print(f"Dataset: {config['dataset']['name']}")
        print(f"Input dimension: {datamodule.get_input_dim()}")
        # ... could add more logging ...
        print("="*60 + "\n")

    # Move to device
    if fabric.is_global_zero: print(f"[Rank 0] Moving model to {fabric.device}...")
    model.to(fabric.device)

    # Compile
    if fabric.device.type == "cuda":
        COMPILE_MODE = "reduce-overhead"
        if fabric.is_global_zero: 
            print(f"[Rank 0] Applying torch.compile(mode='{COMPILE_MODE}')...")
            print("[Rank 0] Note: You will see a ~60s delay at Step 0 while kernels are built.")
        
        try:
            model = torch.compile(model, mode=COMPILE_MODE)
        except Exception as e:
            print(f"[WARNING] Compilation failed: {e}. Falling back to eager execution.")

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
               total_steps=15000, log_interval=None, save_interval=5000,
               checkpoint_dir="checkpoints", enable_profiler=False,
               enable_grad_clipping=True):
    """
    Executes the training loop.
    """
    
    if log_interval is None:
        log_interval = max(1, int(total_steps * 0.01))

    # Profiler settings
    PROFILER_WARMUP = 25
    PROFILER_ACTIVE = 5

    effective_steps = total_steps
    if enable_profiler:
        effective_steps = PROFILER_WARMUP + PROFILER_ACTIVE + 2
        if fabric.is_global_zero:
            print(f"PROFILER ENABLED: Overriding total steps to {effective_steps}")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['optimizer']['lr'] * 4,
        total_steps=total_steps,
        pct_start=0.3
    )

    train_iter = iter(train_loader)

    if fabric.is_global_zero:
        desc = "PROFILING" if enable_profiler else f"{fabric.device.type.upper()} Training"
        pbar = tqdm(range(effective_steps), desc=desc)
    else:
        pbar = range(effective_steps)

    model.train()
    running_loss = 0.0

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
        for step in pbar:
            # Data Loading
            with record_function("data_loading"):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                x_0 = batch[0]

            # Forward & Loss
            optimizer.zero_grad()
            with record_function("forward_pass"):
                t = torch.randint(0, model.T, (x_0.size(0),), device=fabric.device)
                loss = model.compute_loss(x_0, t)

            # Backward
            with record_function("backward_pass"):
                fabric.backward(loss)

            # Optimization
            with record_function("optimizer_step"):
                if enable_grad_clipping:
                    fabric.clip_gradients(model, optimizer, max_norm=1.0, error_if_nonfinite=False)
                
                optimizer.step()
                scheduler.step()

            # Logging
            current_loss = loss.item()
            running_loss += current_loss

            if (step + 1) % log_interval == 0 and not enable_profiler:
                avg_loss = running_loss / log_interval
                if fabric.world_size > 1:
                    avg_loss = fabric.all_reduce(avg_loss, reduce_op="mean")

                if fabric.is_global_zero:
                    current_lr = scheduler.get_last_lr()[0]
                    pct = ((step + 1) / total_steps) * 100
                    # TQDM update
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
                    # Use standard print for logs to persist in notebook output cells
                    print(f"[Step {step+1:5d} | {pct:3.0f}%] loss: {avg_loss:.4f} | lr: {current_lr:.2e}")
                    
                running_loss = 0.0

            # Checkpointing
            if (step + 1) % save_interval == 0 and not enable_profiler:
                save_path = os.path.join(checkpoint_dir, f"step_{step+1}.ckpt")
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
                fabric.save(save_path, state)
                
                if fabric.is_global_zero:
                    print(f"\nSaved checkpoint to {save_path}", flush=True)
                    # Save Config
                    config_path = os.path.join(checkpoint_dir, "config.json")
                    if not os.path.exists(config_path):
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=4)
                        print(f"Saved configuration to {config_path}")

            if enable_profiler:
                prof.step()

    # Final Save
    if not enable_profiler:
        last_save_path = os.path.join(checkpoint_dir, "last.ckpt")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        fabric.save(last_save_path, state)
        if fabric.is_global_zero:
             print(f"\nSaved final checkpoint to {last_save_path}", flush=True)
             config_path = os.path.join(checkpoint_dir, "config.json")
             with open(config_path, 'w') as f:
                  json.dump(config, f, indent=4)

    print("Training/Profiling Finished.")
