import os
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

# Import local modules
# Assuming running from repo root or src is in path
try:
    from ..inference.loader import load_model
    from ..inference.generator import WaveletDiffGenerator
    from .metrics import MetricsEvaluator
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from inference.loader import load_model
    from inference.generator import WaveletDiffGenerator
    from eval.metrics import MetricsEvaluator

def run_evaluation(checkpoint_path, data_path, output_dir, num_samples=None, device='cuda', batch_size=None, **kwargs):
    """
    Main evaluation pipeline.
    kwargs: Additional arguments passed to metrics evaluator or generator
        - metric_batch_size: Batch size for discriminative/predictive metrics
        - metric_epochs: Epochs for discriminative/predictive metrics
        - use_ddim: bool, use DDIM sampling
    """
    print(f"Starting Evaluation for {checkpoint_path}")
    
    # Extract kwargs
    use_ddim = kwargs.pop('use_ddim', False)
    
    # 1. Load Real Data
    print(f"Loading real data from {data_path}...")
    
    if str(data_path).endswith('.npy'):
        real_data = np.load(data_path)
    elif str(data_path).endswith('.csv'):
        # Will try to get from DataModule if None
        real_data = None 
    else:
        raise ValueError("Unsupported data file format. Use .npy")
        
    # 2. Load Model & Config
    config_path = Path(checkpoint_path).parent / "config.json"
    if not config_path.exists():
         print("Warning: Config.json not found near checkpoint. Using default/inferred config.")
         config = {'dataset': {'name': 'stocks'}} 
    else:
         with open(config_path, 'r') as f:
             config = json.load(f)

    model, fabric, datamodule = load_model(checkpoint_path, config, device)
    
    # Graph Optimization (A100/L4)
    if kwargs.get('compile_model', False):
        print("Compiling model for graph optimization (torch.compile)...")
        try:
             # 'reduce-overhead' minimizes python overhead in the generation loop
             model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
             print(f"Compilation (reduce-overhead) failed, retrying with default mode: {e}")
             model = torch.compile(model)
    
    # If real_data is None/CSV, get it from datamodule
    if real_data is None:
         if hasattr(datamodule, 'raw_data_tensor'):
             real_data = datamodule.raw_data_tensor.numpy()
         else:
             datamodule.setup()
             real_data = datamodule.train_dataset.tensors[0].numpy()
             
    # 3. Generate Samples
    if num_samples is None:
        num_samples = len(real_data)
        
    generator = WaveletDiffGenerator(model, datamodule, fabric, config)
    generated_data = generator.generate(num_samples, batch_size=batch_size, use_ddim=use_ddim)
    
    # Save Generated Data
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_path = os.path.join(output_dir, f"generated_samples_{timestamp}.npy")
    np.save(gen_path, generated_data)
    print(f"Saved generated samples to {gen_path}")
    
    # Save Real Data for visualization consistency
    real_path = os.path.join(output_dir, "real_samples_used.npy")
    np.save(real_path, real_data)
    print(f"Saved real samples to {real_path}")
    
    # 4. Compute Metrics
    # Pass remaining kwargs to MetricsEvaluator (e.g. metric_batch_size, metric_epochs)
    print("Initializing Metrics Evaluator with params:", kwargs)
    evaluator = MetricsEvaluator(real_data, generated_data, device=device, **kwargs)
    results = evaluator.run_all()
    
    # 5. Save Results
    results_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    results_serializable = {k: float(v) for k, v in results.items()}
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
        
    print("Evaluation Results:")
    print(json.dumps(results_serializable, indent=4))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='eval_outputs')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--ddim', action='store_true')
    args = parser.parse_args()
    
    run_evaluation(args.checkpoint, args.data, args.output, args.samples, use_ddim=args.ddim)
