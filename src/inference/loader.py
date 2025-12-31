import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
from lightning.fabric import Fabric

def load_model(checkpoint_path, config, device='cuda'):
    """
    Loads the WaveletDiff model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        config (dict): Configuration dictionary used for training.
        device (str, optional): Device to load the model on. Defaults to 'cuda'.
        
    Returns:
        model: Loaded WaveletDiffusionTransformer model.
        fabric: Lightning Fabric instance.
    """
    # Ensure WaveletDiff source is in path
    # This assumes the repo structure setup in the notebook
    # We might need to be flexible here, but for now assuming standard setup
    
    # Initialize Fabric
    fabric = Fabric(accelerator="auto", devices=1, precision="bf16-mixed" if device == 'cuda' else "32-true")
    fabric.launch()
    
    # Import here to avoid early failures if paths aren't set yet
    try:
        from models.transformer import WaveletDiffusionTransformer
        from data.module import WaveletTimeSeriesDataModule
    except ImportError:
        # Try to find the source automatically
        current_dir = Path(__file__).parent.absolute()
        repo_root = current_dir.parent.parent # src/inference -> src -> root
        wd_source = repo_root / "WaveletDiff_source" / "src"
        if wd_source.exists():
             sys.path.append(str(wd_source))
             from models.transformer import WaveletDiffusionTransformer
             from data.module import WaveletTimeSeriesDataModule
        else:
             raise ImportError("Could not find WaveletDiff_source/src. Please ensure the repository is cloned and paths are correct.")

    # Create dummy datamodule for model init (needed for shapes)
    # Ideally config has the necessary info
    # We might need to construct a dummy tensor if the model checks shapes eagerly
    # But WaveletTimeSeriesDataModule usually needs data to infer shapes if not in config
    # Let's hope config has it or we can pass None if handled gracefully
    
    # Re-construct DataModule to get dimensions if needed
    # If the checkpoint has the config, we are good. 
    # But here we pass config as arg.
    
    # We need to construct the model with the same arguments
    # The training script does:
    # model = WaveletDiffusionTransformer(data_module=datamodule, config=model_base_config)
    
    # We might need to load the data to get the datamodule right
    # Or mock it.
    
    # Let's see if we can load the datamodule from the config
    # The config should have 'dataset' info.
    # If we are just doing inference, maybe we don't need the full dataset loaded,
    # but the model needs `input_dim`, `wavelet_levels` etc.
    
    # Ensure minimal config values are present to prevent KeyErrors
    # This acts as a robust fallback if the user provides an empty or partial config
    config.setdefault('dataset', {})
    config['dataset'].setdefault('name', 'stocks')
    config['dataset'].setdefault('seq_len', 24) # Default used in training nb
    
    config.setdefault('training', {})
    config['training'].setdefault('batch_size', 512)
    
    config.setdefault('data', {})
    config['data'].setdefault('data_dir', 'data') # Dummy path if not real loading
    
    # DYNAMIC PATH FIX: Override data_dir to match current environment
    current_repo_data_dir = repo_root / "WaveletDiff_source" / "data"
    if current_repo_data_dir.exists():
        full_data_path = str(current_repo_data_dir.absolute())
        config.setdefault('data', {})
        config['data']['data_dir'] = full_data_path
        
        # FORCE 5-COLUMN FILTERING for 'stocks' dataset
        # The training notebook filters: ['Open', 'High', 'Low', 'Close', 'Volume']
        if config['dataset'].get('name') == 'stocks':
            import pandas as pd
            from data.loaders import create_sliding_windows
            
            stocks_csv = current_repo_data_dir / "stocks" / "stock_data.csv"
            if stocks_csv.exists():
                df = pd.read_csv(stocks_csv)
                CORE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in CORE_COLS):
                    df_filtered = df[CORE_COLS]
                    
                    # Create windows
                    raw_data_np, _ = create_sliding_windows(
                        df_filtered.values,
                        seq_len=config['dataset']['seq_len'],
                        normalize=True
                    )
                    
                    # Pass explicit tensor to DataModule
                    datamodule = WaveletTimeSeriesDataModule(
                        config=config, 
                        data_tensor=torch.FloatTensor(raw_data_np)
                    )
                else:
                    datamodule = WaveletTimeSeriesDataModule(config=config)
            else:
                 datamodule = WaveletTimeSeriesDataModule(config=config)
        else:
             datamodule = WaveletTimeSeriesDataModule(config=config)
    else:
        datamodule = WaveletTimeSeriesDataModule(config=config)
    
    model = WaveletDiffusionTransformer(data_module=datamodule, config=config)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device) # Fabric saves simple dict
    
    # Fabric checkpoint structure: {'model': ..., 'optimizer': ...}
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint # Fallback if standard save
        
    # Remove _orig_mod prefix if compiled
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, fabric, datamodule
