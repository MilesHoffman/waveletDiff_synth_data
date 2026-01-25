
import numpy as np
import torch
import torch.nn as nn
from typing import List, Callable

from discriminative_metrics import discriminative_score_metrics
from predictive_metrics import predictive_score_metrics
from context_fid import Context_FID
from cross_correlation import CrossCorrelLoss
from dtw import dtw_js_divergence_distance
from advanced_metrics import (
    calculate_distribution_fidelity, 
    calculate_structural_alignment, 
    calculate_financial_reality, 
    calculate_memorization_ratio, 
    calculate_diversity_metrics, 
    calculate_fld,
    calculate_dcr,
    calculate_manifold_precision_recall,
    calculate_mmd\n)

def run_discriminative_benchmark(real, generated, iterations=5):
    """Run discriminative score benchmark over multiple iterations."""
    scores = []
    print(f"Running Discriminative Score Benchmark ({iterations} iterations)...")
    for i in range(iterations):
        score, _, _ = discriminative_score_metrics(real, generated)
        scores.append(score)
        print(f"Iter {i}: {score:.4f}")
    return scores

def run_predictive_benchmark(real, generated, iterations=5):
    """Run predictive score benchmark over multiple iterations."""
    scores = []
    print(f"Running Predictive Score Benchmark ({iterations} iterations)...")
    for i in range(iterations):
        score = predictive_score_metrics(real, generated)
        scores.append(score)
        print(f"Iter {i}: {score:.4f}")
    return scores

def run_context_fid_benchmark(real, generated, iterations=5):
    """Run Context-FID benchmark over multiple iterations."""
    scores = []
    print(f"Running Context-FID Benchmark ({iterations} iterations)...")
    for i in range(iterations):
        score = Context_FID(real, generated)
        scores.append(score)
        print(f"Iter {i}: {score:.4f}")
    return scores

def run_cross_correlation_benchmark(real, generated, iterations=5, sample_size=1000):
    """Run Cross-Correlation Loss benchmark."""
    def random_choice(size, num_select=100):
        select_idx = np.random.randint(low=0, high=size, size=(num_select,))
        return select_idx

    x_real = torch.from_numpy(real)
    x_fake = torch.from_numpy(generated)
    scores = []
    
    # Ensure sample size isn't larger than dataset
    sample_size = min(sample_size, len(real), len(generated))

    print(f"Running Cross-Correlation Benchmark ({iterations} iterations)...")
    for i in range(iterations):
        real_idx = random_choice(x_real.shape[0], sample_size)
        fake_idx = random_choice(x_fake.shape[0], sample_size)
        
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
        loss = corr.compute(x_fake[fake_idx, :, :])
        
        scores.append(loss.item())
        print(f"Iter {i}: {loss.item():.4f}")
    return scores

def run_dtw_benchmark(real, generated, iterations=5):
    """Run DTW JS Divergence benchmark."""
    scores = []
    print(f"Running DTW Benchmark ({iterations} iterations)...")
    for i in range(iterations):
        js_dist = dtw_js_divergence_distance(real, generated, n_samples=100)['js_divergence']
        scores.append(js_dist)
        print(f"Iter {i}: {js_dist:.4f}")
    return scores

def run_advanced_metrics(real_raw, generated_raw):
    """Run all advanced financial metrics."""
    print("Running Distribution Fidelity Checks (Wasserstein, KS Test)...")
    dist_results = calculate_distribution_fidelity(real_raw, generated_raw)
    
    print("\nRunning Structural Alignment Checks (PCA, t-SNE)...")
    struct_results = calculate_structural_alignment(real_raw, generated_raw)
    
    print("\nRunning Financial Reality Checks (ACF, Cross-Corr, Volatility)...")
    fin_results = calculate_financial_reality(real_raw, generated_raw)
    
    return dist_results, struct_results, fin_results

def run_new_metrics(real_raw, generated_raw):
    """Run memorization, diversity, and FLD metrics."""
    print("\nRunning Memorization Check (1/3 Rule)...")
    mem_ratio = calculate_memorization_ratio(real_raw, generated_raw)
    
    print("\nRunning Diversity Check (Coverage)...")
    div_results = calculate_diversity_metrics(real_raw, generated_raw)
    
    print("\nRunning Feature Likelihood Divergence (FLD)...")
    fld_score = calculate_fld(real_raw, generated_raw)
    
    # New Metrics (DCR, MMD, Precision/Recall)
    print("\nRunning Distance to Closest Record (DCR)...")
    dcr_score = calculate_dcr(real_raw, generated_raw)
    
    print("\nRunning Manifold Precision & Recall...")
    precision, recall = calculate_manifold_precision_recall(real_raw, generated_raw)
    
    print("\nRunning Maximum Mean Discrepancy (MMD)...")
    mmd_score = calculate_mmd(real_raw, generated_raw)
    
    return mem_ratio, div_results, fld_score, dcr_score, precision, recall, mmd_score
