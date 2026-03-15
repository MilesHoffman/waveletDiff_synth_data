"""
Utility script to export a loaded WaveletDiff Checkpoint into a dynamic ONNX file.
"""
import os
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from models import WaveletDiffusionTransformer
from data import WaveletTimeSeriesDataModule
from utils import ConfigManager, load_config, merge_configs

def main():
    parser = argparse.ArgumentParser(description='Export WaveletDiff checkpoint to ONNX')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name (matches train.py experiment_name)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (optional override)')
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    dataset_name = args.dataset or 'stocks'
    config = config_manager.load(dataset_name=dataset_name)
    
    experiment_dir = Path(config['paths']['output_dir']) / args.experiment_name
    saved_config_path = experiment_dir / "config.yaml"
    
    if saved_config_path.exists():
        saved_config = load_config(str(saved_config_path))
        config = merge_configs(config, saved_config)
    
    checkpoint_path = experiment_dir / 'checkpoint.ckpt'
    ckpts = list(experiment_dir.glob("*.ckpt"))
    if not checkpoint_path.exists() and ckpts:
        checkpoint_path = ckpts[0]
        
    onnx_path = experiment_dir / 'model.onnx'
    
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading PyTorch Model from Checkout: {checkpoint_path}")
    
    data_module = WaveletTimeSeriesDataModule(config=config)
    
    # Explicitly check for EMA state dict
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "ema_state_dict" in ckpt:
        print("Found 'ema_state_dict'. Using EMA weights for ONNX export.")
        # Temporarily swap the main state_dict so Lightning loads the EMA weights
        ckpt["state_dict"] = ckpt["ema_state_dict"]
        temp_ckpt_path = checkpoint_path.parent / "temp_ema_checkpoint.ckpt"
        torch.save(ckpt, temp_ckpt_path)
        
        model = WaveletDiffusionTransformer.load_from_checkpoint(
            temp_ckpt_path,
            data_module=data_module,
            config=config,
        )
        os.remove(temp_ckpt_path)
    else:
        print("No 'ema_state_dict' found. Falling back to raw training weights.")
        model = WaveletDiffusionTransformer.load_from_checkpoint(
            checkpoint_path,
            data_module=data_module,
            config=config,
        )
    
    print(f"Exporting model to ONNX format at {onnx_path}...")
    model.eval()
    model.to("cpu")
    
    batch_size = 2
    device = torch.device('cpu')
    
    dummy_x = torch.randn(batch_size, model.input_dim, model.num_features, device=device)
    dummy_t = torch.full((batch_size,), 0.5, device=device)
    
    # Setup PyTorch 2.0 Dynamo dynamic shapes for variable batch sizes
    from torch.export import Dim
    batch_dim = Dim("batch_size")
    
    # For Dynamo, dynamic shapes map positional arguments
    dynamic_shapes = [
        {0: batch_dim},  # x
        {0: batch_dim}   # t
    ]
    
    input_names = ['x', 't']
    dummy_inputs = [dummy_x, dummy_t]
    
    has_scale = getattr(data_module, 'has_conditioning', False)
    has_path_sig = getattr(data_module, 'has_path_sig_conditioning', False)
    
    if has_scale:
        dummy_scale = torch.tensor([0.05] * batch_size, device=device).unsqueeze(1) if getattr(data_module, 'scale_is_2d', False) else torch.tensor([0.05] * batch_size, device=device)
        dummy_inputs.append(dummy_scale)
        input_names.append('scale')
        dynamic_shapes.append({0: batch_dim})
        
    if has_path_sig and getattr(model, 'path_sig_embedding', None) is not None:
        sig_dim = getattr(data_module, 'path_sig_dim', 205)
        dummy_sig = torch.randn(batch_size, sig_dim, device=device)
        dummy_inputs.append(dummy_sig)
        input_names.append('path_sig')
        dynamic_shapes.append({0: batch_dim})
    
    try:
        has_cond = has_path_sig and getattr(model, 'path_sig_embedding', None) is not None
        scale_str = "scale" if has_scale else "None"
        
        class ExportWrapper(nn.Module):
            def __init__(self, m, use_scale, use_cond):
                super().__init__()
                self.m = m
                self.use_scale = use_scale
                self.use_cond = use_cond
                
            def forward(self, x, t, *args):
                idx = 0
                scale = None
                cond = None
                if self.use_scale:
                    scale = args[idx]
                    idx += 1
                if self.use_cond:
                    cond = args[idx]
                    idx += 1
                return self.m(x, t, conditions=cond)
        
        wrapper = ExportWrapper(model, has_scale, has_cond)
        wrapper.eval()
        
        torch.onnx.export(
            wrapper,
            tuple(dummy_inputs),
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=['output'],
            dynamic_shapes=tuple(dynamic_shapes),
            dynamo=True
        )
            
        print(f"ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"ONNX Export Failed: {e}")

if __name__ == "__main__":
    main()
