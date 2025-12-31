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
    from src.inference.loader import load_model
    from src.inference.generator import WaveletDiffGenerator
    from src.eval.metrics import MetricsEvaluator
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from inference.loader import load_model
    from inference.generator import WaveletDiffGenerator
    from eval.metrics import MetricsEvaluator

def run_evaluation(checkpoint_path, data_path, output_dir, num_samples=None, device='cuda', batch_size=None):
    """
    Main evaluation pipeline.
    """
    print(f"Starting Evaluation for {checkpoint_path}")
    
    # 1. Load Real Data
    print(f"Loading real data from {data_path}...")
    # Assuming data is in .npy or .csv?
    # Original notebook uses .npy for real_samples.npy saved during training or sampling.
    # But we might want to load original dataset from csv if .npy not available.
    # For now, let's assume the user points to a .npy file or we process the CSV like dataloader does.
    # The user request said "The notebook will contain things like paths...".
    
    if str(data_path).endswith('.npy'):
        real_data = np.load(data_path)
    elif str(data_path).endswith('.csv'):
        # TODO: Implement CSV loading if needed, but robust eval usually uses processed NPY
        # for apple-to-apple with training.
        # Check if we can use the DataModule to load it.
        # We'll defer this to the loader/generator if we initialize DataModule.
        real_data = None # Will try to get from DataModule
    else:
        raise ValueError("Unsupported data file format. Use .npy")
        
    # 2. Load Model & Config
    # Checkpoint path usually in: experiment_dir/checkpoint.ckpt
    # Config usually in: experiment_dir/config.json ??
    # If not, we rely on args or defaults in loader.
    
    # Construct config path assuming standard structure
    config_path = Path(checkpoint_path).parent / "config.json"
    if not config_path.exists():
         # Try to find config in the repo or defaults
         # For now, create a basic config or fail
         # We can try to load config from the "configs" dir in WaveletDiff_source if we know the dataset name.
         # Let's assume the notebook passes the config or we find it.
         print("Warning: Config.json not found near checkpoint. Using default/inferred config.")
         config = {'dataset': {'name': 'stocks'}} # Value to trigger default load
    else:
         with open(config_path, 'r') as f:
             config = json.load(f)

    model, fabric, datamodule = load_model(checkpoint_path, config, device)
    
    # If real_data is None/CSV, get it from datamodule
    if real_data is None:
         # Datamodule .raw_data_tensor if available
         if hasattr(datamodule, 'raw_data_tensor'):
             real_data = datamodule.raw_data_tensor.numpy()
         else:
             # Try to setup
             datamodule.setup()
             real_data = datamodule.train_dataset.tensors[0].numpy() # Approximation
             
    # 3. Generate Samples
    if num_samples is None:
        num_samples = len(real_data)
        
    generator = WaveletDiffGenerator(model, datamodule, fabric, config)
    generated_data = generator.generate(num_samples, batch_size=batch_size)
    
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
    evaluator = MetricsEvaluator(real_data, generated_data, device=device)
    results = evaluator.run_all()
    
    # 5. Save Results
    results_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    # Convert numpy types to native for JSON serialization
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
    args = parser.parse_args()
    
    run_evaluation(args.checkpoint, args.data, args.output, args.samples)
