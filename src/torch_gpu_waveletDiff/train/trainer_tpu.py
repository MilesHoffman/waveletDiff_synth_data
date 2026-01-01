"""
TPU Optimized Trainer experimental module.
Implements aggressive optimizations: No .item(), num_workers=8, pin_memory=False, no syncing.
"""
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

# Delayed imports
try:
    from data.loaders import create_sliding_windows
    from data.module import WaveletTimeSeriesDataModule
    from models.transformer_tpu import WaveletDiffusionTransformerTPU
except ImportError:
    pass

def setup_fabric(accelerator="auto", devices="auto", precision=None):
    """
    Initializes Lightning Fabric (TPU mode).
    """
    if precision is None:
        # Force bf16-true for TPU speed
        precision = "bf16-true"

    fabric = Fabric(accelerator=accelerator, devices=devices, precision=precision)
    fabric.launch()
    
    if fabric.is_global_zero:
        print(f"[Rank 0] Fabric initialized on device: {fabric.device} with precision: {precision} (TPU Optimized)")
        
    return fabric

def get_dataloaders(fabric, repo_dir, dataset_name, seq_len, batch_size, wavelet_type="db2", wavelet_levels="auto", data_path=None):
    """
    Loads data and creates TPU-optimized DataLoaders with specific user requests.
    """
    from data.loaders import create_sliding_windows
    from data.module import WaveletTimeSeriesDataModule

    if os.path.isabs(data_path):
        stocks_path = data_path
    else:
        stocks_path = os.path.join(repo_dir, data_path)

    data_dir = os.path.dirname(stocks_path)

    if fabric.is_global_zero:
        print(f"Loading data from {stocks_path}...")

    df = pd.read_csv(stocks_path)
    CORE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_filtered = df[CORE_COLS]

    custom_data_windows, _ = create_sliding_windows(
        df_filtered.values,
        seq_len=seq_len,
        normalize=True
    )

    full_data_tensor = torch.FloatTensor(custom_data_windows)

    config = {
        'dataset': {'name': dataset_name, 'seq_len': seq_len},
        'training': {'batch_size': batch_size, 'epochs': 1}, 
        'data': {'data_dir': data_dir, 'normalize_data': False},
        'wavelet': {'type': wavelet_type, 'levels': wavelet_levels}
    }

    datamodule = WaveletTimeSeriesDataModule(config=config, data_tensor=full_data_tensor)

    # Cast to bf16
    if fabric.device.type != "cpu":
        if fabric.is_global_zero: print(f"Optimizing: Casting data to bfloat16 for {fabric.device.type}...")
        full_data_tensor = full_data_tensor.to(torch.bfloat16)

    dataset = TensorDataset(full_data_tensor)

    # USER REQUEST: Set num_workers to 8 (explicit, no auto-detection)
    num_workers = 8
    
    if fabric.is_global_zero:
        print(f"Using {num_workers} num_workers (User Forced)...")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False, # USER REQUEST: Set pin_memory=False for TPU
        persistent_workers=True
    )

    loader = fabric.setup_dataloaders(loader)
    
    return loader, datamodule, config

def init_model(fabric, datamodule, config, 
               embed_dim=256, num_heads=8, num_layers=8, time_embed_dim=128, 
               dropout=0.1, prediction_target="noise", use_cross_level_attention=True,
               learning_rate=2e-4, weight_decay=1e-5):
    """
    Initializes the WaveletDiffusionTransformerTPU model.
    """
    from models.transformer_tpu import WaveletDiffusionTransformerTPU

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

    model = WaveletDiffusionTransformerTPU(data_module=datamodule, config=config)
    
    if fabric.is_global_zero:
         print(f"[Rank 0] Moving model to {fabric.device}...")
    model.to(fabric.device)

    # TPU specific compile? Usually handled by XLA seamlessly, but we can explicit compile if needed with compile=True in XLA,
    # but for pure PyTorch XLA we usually just run.

    use_fused = False # No fused Adam on TPU usually, standard AdamW
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    model, optimizer = fabric.setup(model, optimizer)
    return model, optimizer, config

def train_loop(fabric, model, optimizer, train_loader, config,
               total_steps=15000, log_interval=None, save_interval=5000,
               checkpoint_dir="checkpoints_tpu", enable_profiler=False,
               enable_grad_clipping=False): # CLIP DISABLED BY DEFAULT
    
    if log_interval is None:
        log_interval = max(1, int(total_steps * 0.01))

    effective_steps = total_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['optimizer']['lr'] * 4,
        total_steps=total_steps,
        pct_start=0.3
    )

    train_iter = iter(train_loader)

    if fabric.is_global_zero:
        pbar = tqdm(range(effective_steps), desc="TPU Optimized Training")
    else:
        pbar = range(effective_steps)

    model.train()
    
    # Unified efficient logging for all devices
    running_loss = torch.zeros((), device=fabric.device)

    if fabric.is_global_zero:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for step in pbar:
        # Data Loading
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        x_0 = batch[0]

        # Forward & Loss
        optimizer.zero_grad(set_to_none=True)
        t = torch.randint(0, model.T, (x_0.size(0),), device=fabric.device)
        loss = model.compute_loss(x_0, t)

        # Backward
        fabric.backward(loss)

        # Optimization (Grad Clipping DISABLED)
        optimizer.step()
        scheduler.step()

        # Logging Accumulation (Async)
        running_loss += loss.detach()

        if (step + 1) % log_interval == 0:
            # Sync only once per interval
            # on XLA, .item() causes a sync. We do this ONLY here.
            avg_loss = running_loss.item() / log_interval
            running_loss.zero_()

            if fabric.world_size > 1:
                avg_loss = fabric.all_reduce(avg_loss, reduce_op="mean")

            if fabric.is_global_zero:
                current_lr = scheduler.get_last_lr()[0]
                pct = ((step + 1) / total_steps) * 100
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                print(f"[Step {step+1:5d} | {pct:3.0f}%] loss: {avg_loss:.4f} | lr: {current_lr:.2e}")

        # Checkpointing
        if (step + 1) % save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"step_{step+1}.ckpt")
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            fabric.save(save_path, state)
            if fabric.is_global_zero:
                print(f"\nSaved checkpoint to {save_path}", flush=True)

    print("TPU Training Finished.")
