
"""
Run full evaluation pipeline using the src.evaluation modules.
This script is designed to be called from the evaluation notebook.
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.evaluation import (
    discriminative_score_metrics,
    predictive_score_metrics,
    Context_FID,
    CrossCorrelLoss,
    dtw_wasserstein_distance,
    display_scores
)
from src.evaluation.metric_utils import visualization

def run_metrics(real_path, fake_path, output_dir, device='cuda'):
    print(f"Loading real data from {real_path}")
    real_data = np.load(real_path)
    
    print(f"Loading generated data from {fake_path}")
    fake_data = np.load(fake_path)
    
    # Ensure shapes match [N, T, D]
    if real_data.ndim == 2:
        real_data = np.expand_dims(real_data, axis=2)
    if fake_data.ndim == 2:
        fake_data = np.expand_dims(fake_data, axis=2)
        
    print(f"Real Data Shape: {real_data.shape}")
    print(f"Fake Data Shape: {fake_data.shape}")
    
    # Trim to matching size if needed
    min_len = min(len(real_data), len(fake_data))
    real_data = real_data[:min_len]
    fake_data = fake_data[:min_len]
    print(f"Evaluated Samples: {min_len}")

    results = {}
    
    # 1. Visualization
    print("\n--- Generating Visualizations ---")
    try:
        visualization(real_data, fake_data, 'pca')
        print("PCA plot generated.")
        visualization(real_data, fake_data, 'tsne')
        print("t-SNE plot generated.")
    except Exception as e:
        print(f"Visualization failed: {e}")

    # 2. Discriminative Score
    print("\n--- Calculating Discriminative Score ---")
    try:
        # Signature: (ori_data, generated_data) -> (score, fake_acc, real_acc)
        disc_score, fake_acc, real_acc = discriminative_score_metrics(real_data, fake_data)
        results['discriminative_score'] = disc_score
        results['discriminative_fake_acc'] = fake_acc
        results['discriminative_real_acc'] = real_acc
        print(f"Discriminative Score: {disc_score:.4f}")
        print(f"  Fake Accuracy: {fake_acc:.4f}")
        print(f"  Real Accuracy: {real_acc:.4f}")
    except Exception as e:
        print(f"Discriminative Score failed: {e}")

    # 3. Predictive Score
    print("\n--- Calculating Predictive Score ---")
    try:
        # Signature: (ori_data, generated_data, window_size=20) -> score
        pred_score = predictive_score_metrics(real_data, fake_data)
        results['predictive_score'] = pred_score
        print(f"Predictive Score (MAE): {pred_score:.4f}")
    except Exception as e:
        print(f"Predictive Score failed: {e}")
        
    # 4. Context-FID
    print("\n--- Calculating Context-FID ---")
    try:
        # Signature: (ori_data, generated_data) -> score
        # Note: it handles mounting drive/etc internally if needed, but here we pass numpy
        context_fid = Context_FID(real_data, fake_data)
        print(f"Context-FID Score: {context_fid:.4f}")
        results['context_fid'] = context_fid
    except Exception as e:
        print(f"Context-FID failed: {e}")

    # 5. DTW
    print("\n--- Calculating DTW Distance ---")
    try:
        # Sample subset for DTW as it is slow
        n_dtw = min(100, len(real_data))
        dtw_score = dtw_wasserstein_distance(real_data[:n_dtw], fake_data[:n_dtw])
        print(f"DTW Wasserstein Distance: {dtw_score:.4f}")
        results['dtw_distance'] = dtw_score
    except Exception as e:
        print(f"DTW failed: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data", required=True)
    parser.add_argument("--fake_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_metrics(args.real_data, args.fake_data, args.output_dir, args.device)
