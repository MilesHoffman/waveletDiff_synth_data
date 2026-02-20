"""
Sample generation script for trained WaveletDiff models.
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

from models import WaveletDiffusionTransformer
from training import DiffusionTrainer
from data import WaveletTimeSeriesDataModule
from utils import ConfigManager


def main():
    """Generate samples from a trained model."""
    parser = argparse.ArgumentParser(description='Generate samples from WaveletDiff model')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name (matches train.py experiment_name)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to generate (default: match real dataset size)')
    parser.add_argument('--sampling_method', type=str, choices=['ddpm', 'ddim'], default=None,
                       help='Sampling method (overrides config)')
    parser.add_argument('--compile_mode', type=str, choices=['none', 'default', 'reduce-overhead', 'max-autotune'], default='none',
                       help='torch.compile mode')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory/path')
    
    # Model Architecture Overrides (for legacy checkpoints without saved config)
    parser.add_argument('--seq_len', type=int, default=None, help='Sequence length (override)')
    parser.add_argument('--embed_dim', type=int, default=None, help='Embedding dimension (override)')
    parser.add_argument('--num_layers', type=int, default=None, help='Number of layers (override)')
    parser.add_argument('--num_heads', type=int, default=None, help='Number of heads (override)')
    parser.add_argument('--wavelet_levels', type=int, default=None, help='Wavelet levels (override)')
    parser.add_argument('--guidance_scale', type=float, default=None,
                       help='CFG guidance scale (1.0=no guidance, >1.0=stronger conditioning)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    
    # 1. Determine dataset name
    dataset_name = args.dataset or 'etth1'
    
    # 2. Load base defaults for this dataset
    config = config_manager.load(dataset_name=dataset_name)
    
    # 3. Determine experiment directory
    experiment_name = args.experiment_name
    experiment_dir = Path(config['paths']['output_dir']) / experiment_name
    
    # 4. Try to load saved config from experiment directory (Priority!)
    from utils import load_config, merge_configs
    saved_config_path = experiment_dir / "config.yaml"
    if saved_config_path.exists():
        print(f"Loading saved configuration from: {saved_config_path}")
        saved_config = load_config(str(saved_config_path))
        # Merge saved config ON TOP of defaults (to ensure we have all keys but use saved values)
        config = merge_configs(config, saved_config)
    else:
        print(f"Warning: No saved config found at {saved_config_path}")
        print("Using default config for dataset. If model architecture differs, specify overrides via CLI.")

    # 5. Apply CLI overrides (Highest Priority)
    if args.dataset: config['dataset']['name'] = args.dataset
    if args.sampling_method: config['sampling']['method'] = args.sampling_method
    if args.data_dir: config['data']['data_dir'] = args.data_dir
    
    # Architecture overrides
    if args.seq_len: config['dataset']['seq_len'] = args.seq_len
    if args.embed_dim: config['model']['embed_dim'] = args.embed_dim
    if args.num_layers: config['model']['num_layers'] = args.num_layers
    if args.num_heads: config['model']['num_heads'] = args.num_heads
    if args.wavelet_levels: config['wavelet']['levels'] = args.wavelet_levels

    # Construct model path
    model_path = experiment_dir / 'checkpoint.ckpt'
    
    if not model_path.exists():
        # Fallback: look for any .ckpt file
        ckpts = list(experiment_dir.glob("*.ckpt"))
        if ckpts:
            model_path = ckpts[0]
            print(f"Using found checkpoint: {model_path.name}")
        else:
            print(f"Error: No checkpoint found in {experiment_dir}")
            print(f"Make sure you've trained a model or fully unpacked the archive.")
            sys.exit(1)
    
    print(f"Generating Samples from WaveletDiff Model")
    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Model: {model_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Sampling Method: {config['sampling']['method'].upper()}")
    
    # Set up data module
    data_module = WaveletTimeSeriesDataModule(config=config)
    
    # Load model
    print("Loading model...")
    try:
        model = WaveletDiffusionTransformer.load_from_checkpoint(
            model_path,
            data_module=data_module,
            config=config,
        )
    except RuntimeError as e:
        print(f"\nFATAL ERROR: Failed to load model checkpoint.")
        print(f"Error details: {e}")
        print("-" * 60)
        print("This is likely due to a CONFIGURATION MISMATCH between the default config and your checkpoint.")
        print("Common causes:")
        print("  1. The checkpoint was trained with a different Sequence Length (default: 24).")
        print("  2. The checkpoint was trained with different Model Dimensions (embed_dim, layers).")
        print("-" * 60)
        print("SOLUTION: Specify the correct architecture parameters via CLI.")
        print("Example: python sample.py ... --seq_len 64 --embed_dim 256")
        print("-" * 60)
        sys.exit(1)
        
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")

    # Generate using specified method
    sampling_method = config['sampling']['method']
    use_ddim = (sampling_method == "ddim")
    
    # Prepare conditioning: use sequential indices for 1-to-1 mapping with real data
    scale_conditioning = None
    sample_indices = None
    if getattr(data_module, 'has_conditioning', False):
        num_avail = len(data_module.norm_stats['atr_pcts'])
        if num_samples <= num_avail:
            sample_indices = np.arange(num_samples)
        else:
            sample_indices = np.random.choice(num_avail, size=num_samples, replace=True)
        scale_pcts = data_module.norm_stats['atr_pcts'][sample_indices]
        scale_conditioning = torch.FloatTensor(scale_pcts).to(model.device)
        print(f"Scale conditioning: {num_samples} ATR values (1-to-1={num_samples <= num_avail})")
    
    # Prepare quarter-profile conditions
    conditions = None
    guidance_scale = args.guidance_scale or config.get('conditioning', {}).get('guidance_scale', 1.0)
    if getattr(data_module, 'has_quarter_conditioning', False) and sample_indices is not None:
        qp = data_module.norm_stats['quarter_profiles']
        profile_names = data_module.quarter_profile_names
        conditions = []
        for name in profile_names:
            profile_vals = qp[name][sample_indices]
            conditions.append(torch.FloatTensor(profile_vals).to(model.device))
        print(f"Quarter conditioning: {profile_names}, guidance_scale={guidance_scale}")

    # Apply torch.compile if requested
    if args.compile_mode != 'none':
        print(f"Compiling model with mode='{args.compile_mode}'...")
        model = torch.compile(model, mode=args.compile_mode)
        
        # Warmup: trigger compilation EXACTLY mapping the batch structures
        print("Warming up compiled model...")
        dummy_x = torch.randn(num_samples, model.input_dim, model.num_features, device=model.device)
        dummy_t = torch.full((num_samples,), 0.5, device=model.device)
        with torch.no_grad():
            _ = model(dummy_x, dummy_t, scale=scale_conditioning, conditions=conditions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete.")
    
    # Create trainer for evaluation
    trainer_util = DiffusionTrainer(model)
    
    print(f"Generating {sampling_method.upper()} samples...")
    samples = trainer_util.generate_samples(
        num_samples, use_ddim=use_ddim, scale=scale_conditioning,
        conditions=conditions, guidance_scale=guidance_scale
    )
    
    # Convert to time series
    print("Converting to time series...")
    samples_ts = data_module.convert_wavelet_to_timeseries(samples)
    
    # Save results in the same experiment directory
    real_samples_path = experiment_dir / "real_samples.npy"
    output_file_path = experiment_dir / f"{sampling_method}_samples.npy"
    real_samples_norm_path = experiment_dir / "real_samples_norm.npy"
    output_norm_path = experiment_dir / f"{sampling_method}_samples_norm.npy"
    
    # Get reparameterized time series (what the model actually generates)
    samples_norm = samples_ts.cpu().numpy()
    real_data_norm = data_module.raw_data_tensor.numpy()
    
    # Use matching subset of real data when generating fewer than full dataset
    if num_samples < num_real:
        real_data_norm_subset = real_data_norm[sample_indices] if sample_indices is not None else real_data_norm[:num_samples]
    else:
        real_data_norm_subset = real_data_norm
    
    # Inverse normalize to get Dollar values (Original Scale)
    FIXED_ANCHOR = 100.0
    print(f"Inverse normalizing to Dollar space (Index-{FIXED_ANCHOR})...")
    samples_dollar = data_module.inverse_normalize(samples_norm.copy(), sample_indices=sample_indices, fixed_anchor=FIXED_ANCHOR)
    
    real_indices = sample_indices if sample_indices is not None else np.arange(len(real_data_norm_subset))
    real_samples_dollar = data_module.inverse_normalize(real_data_norm_subset.copy(), sample_indices=real_indices, fixed_anchor=FIXED_ANCHOR)
    
    # Save Dollar Space
    np.save(real_samples_path, real_samples_dollar)
    np.save(output_file_path, samples_dollar)
    
    # Save Reparameterized Space
    np.save(real_samples_norm_path, real_data_norm_subset)
    np.save(output_norm_path, samples_norm)
    
    print(f"Saved {len(real_samples_dollar)} real + {len(samples_dollar)} generated (matched 1-to-1)")
    print(f"Real: {real_samples_path}")
    print(f"Generated: {output_file_path}")


if __name__ == "__main__":
    main()

