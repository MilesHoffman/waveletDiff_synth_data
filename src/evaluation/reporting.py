"""
Reporting utilities for evaluation results.

Provides formatted scorecard display for EvaluationResult objects.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def display_scorecard(result) -> pd.DataFrame:
    """
    Display a formatted scorecard for an EvaluationResult.
    
    Args:
        result: EvaluationResult object from EvaluationRunner
        
    Returns:
        Styled Pandas DataFrame for display
    """
    summary_data = []
    
    # === Core Metrics (Tier 1) ===
    core = result.core_metrics
    
    if 'discriminative' in core:
        summary_data.append({
            "Tier": "Core (Tier 1)",
            "Category": "Opposing Coach",
            "Metric": "Discriminative Score",
            "Value": core['discriminative'],
            "Goal": "lower",
            "Ideal": "→ 0",
            "Description": "Classifier accuracy deviation from 0.5"
        })
    
    if 'predictive_tstr' in core:
        summary_data.append({
            "Tier": "Core (Tier 1)",
            "Category": "Game Analyst",
            "Metric": "Predictive (TSTR)",
            "Value": core['predictive_tstr'],
            "Goal": "lower",
            "Ideal": "→ TRTR",
            "Description": "Train on Synth, Test on Real (MAE)"
        })
        summary_data.append({
            "Tier": "Core (Tier 1)",
            "Category": "Game Analyst",
            "Metric": "Predictive (TRTR)",
            "Value": core['predictive_trtr'],
            "Goal": "baseline",
            "Ideal": "(Baseline)",
            "Description": "Train on Real, Test on Real (MAE)"
        })
        summary_data.append({
            "Tier": "Core (Tier 1)",
            "Category": "Game Analyst",
            "Metric": "Utility Gap",
            "Value": core['utility_gap'],
            "Goal": "lower",
            "Ideal": "→ 0",
            "Description": "|TSTR - TRTR|"
        })
    
    if 'context_fid' in core:
        summary_data.append({
            "Tier": "Core (Tier 1)",
            "Category": "Embedding",
            "Metric": "Context-FID",
            "Value": core['context_fid'],
            "Goal": "lower",
            "Ideal": "→ 0",
            "Description": "FID on TS2Vec embeddings"
        })
    
    if 'correlation' in core:
        summary_data.append({
            "Tier": "Core (Tier 1)",
            "Category": "Structure",
            "Metric": "Correlation Score",
            "Value": core['correlation'],
            "Goal": "lower",
            "Ideal": "→ 0",
            "Description": "Cross-correlation matrix divergence"
        })
    
    if 'dtw' in core:
        summary_data.append({
            "Tier": "Core (Tier 1)",
            "Category": "Temporal",
            "Metric": "DTW Distance",
            "Value": core['dtw'],
            "Goal": "lower",
            "Ideal": "→ 0",
            "Description": "JS divergence of DTW distributions"
        })
    
    # === Advanced Metrics (Tier 2) ===
    adv = result.advanced_metrics
    
    if 'visual_scout' in adv:
        vs = adv['visual_scout']
        summary_data.append({
            "Tier": "Advanced (Tier 2)",
            "Category": "Visual Scout",
            "Metric": "JS Divergence",
            "Value": vs['js_divergence'],
            "Goal": "lower",
            "Ideal": "→ 0",
            "Description": "Log-returns distribution similarity"
        })
        summary_data.append({
            "Tier": "Advanced (Tier 2)",
            "Category": "Visual Scout",
            "Metric": "ACF Similarity",
            "Value": vs['acf_similarity'],
            "Goal": "higher",
            "Ideal": "→ 1",
            "Description": "Volatility clustering preservation"
        })
    
    if 'statistician' in adv:
        stat = adv['statistician']
        summary_data.append({
            "Tier": "Advanced (Tier 2)",
            "Category": "Statistician",
            "Metric": "α-Precision",
            "Value": stat['alpha_precision'],
            "Goal": "higher",
            "Ideal": "→ 1",
            "Description": "Manifold fidelity (quality)"
        })
        summary_data.append({
            "Tier": "Advanced (Tier 2)",
            "Category": "Statistician",
            "Metric": "β-Recall",
            "Value": stat['beta_recall'],
            "Goal": "higher",
            "Ideal": "→ 1",
            "Description": "Manifold diversity (coverage)"
        })
    
    if 'integrity_officer' in adv:
        io = adv['integrity_officer']
        dcr = io['dcr']
        summary_data.append({
            "Tier": "Advanced (Tier 2)",
            "Category": "Integrity Officer",
            "Metric": "DCR (Mean)",
            "Value": dcr['mean'],
            "Goal": "higher",
            "Ideal": "High",
            "Description": "Distance to closest training sample"
        })
        summary_data.append({
            "Tier": "Advanced (Tier 2)",
            "Category": "Integrity Officer",
            "Metric": "Memorization Ratio",
            "Value": io['memorization_ratio'],
            "Goal": "lower",
            "Ideal": "< 0.05",
            "Description": "Fraction of memorized samples"
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    if df.empty:
        return df
    
    # Apply styling
    def style_value(row):
        val = row['Value']
        goal = row['Goal']
        
        if goal == 'baseline':
            return 'background-color: #bdc3c7; color: black'
        elif goal == 'higher':
            if val >= 0.9: return 'background-color: #2ecc71; color: black'
            elif val >= 0.7: return 'background-color: #82e0aa; color: black'
            elif val >= 0.5: return 'background-color: #f9e79f; color: black'
            else: return 'background-color: #e74c3c; color: black'
        else:  # lower
            if val <= 0.05: return 'background-color: #2ecc71; color: black'
            elif val <= 0.15: return 'background-color: #82e0aa; color: black'
            elif val <= 0.3: return 'background-color: #f9e79f; color: black'
            else: return 'background-color: #e74c3c; color: black'
    
    styled = df.style.apply(
        lambda row: [style_value(row) if col == 'Value' else '' for col in df.columns], 
        axis=1
    )
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    
    return styled


def generate_summary_scorecard(metrics_dict):
    """
    Legacy function for backward compatibility.
    
    Generates a styled Pandas DataFrame scorecard from a dictionary of metrics.
    
    Args:
        metrics_dict: Dictionary containing metric results from various evaluation runs.
        
    Returns:
        pd.io.formats.style.Styler: Styled DataFrame object for display
    """
    summary_data = []

    # 1. Statistical (Aggregated MAEs)
    if 'stat_results' in metrics_dict:
        for k, v in metrics_dict['stat_results'].items():
            summary_data.append({
                "Category": "Statistical (MAE)", 
                "Metric": k, 
                "Value": v, 
                "Goal": "lower", 
                "Description": "Diff in statistical moments (Mean/Std/Skew/Kurt)."
            })

    # 2. THE 5 ORIGINAL CORE METRICS
    # Discriminative Score
    if 'discriminative_score' in metrics_dict:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "Discriminative Score", 
            "Value": np.mean(metrics_dict['discriminative_score']), 
            "Goal": "lower", 
            "Description": "Classifier accuracy deviation from 0.5 (Real vs Fake)."
        })
    
    # Predictive Score
    if 'predictive_score' in metrics_dict:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "Predictive Score", 
            "Value": np.mean(metrics_dict['predictive_score']), 
            "Goal": "lower", 
            "Description": "MAE of TSTR (Train on Synthetic, Test on Real)."
        })
    
    # Context-FID (check multiple key names)
    cfid_val = metrics_dict.get('context_fid_score') or metrics_dict.get('context_fid') or metrics_dict.get('cfid')
    if cfid_val is not None:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "Context-FID", 
            "Value": np.mean(cfid_val) if hasattr(cfid_val, '__iter__') and not isinstance(cfid_val, (int, float)) else cfid_val, 
            "Goal": "lower", 
            "Description": "FID score on TS2Vec embeddings."
        })
    
    # Correlation Score / Cross-Correlation Loss (check multiple key names)
    corr_val = metrics_dict.get('correlational_score') or metrics_dict.get('correlation_score') or metrics_dict.get('cross_correl')
    if corr_val is not None:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "Cross-Correl Loss", 
            "Value": np.mean(corr_val) if hasattr(corr_val, '__iter__') and not isinstance(corr_val, (int, float)) else corr_val, 
            "Goal": "lower", 
            "Description": "Difference in cross-correlation matrices."
        })
    
    # DTW Distance / JS Divergence (check multiple key names)
    dtw_val = metrics_dict.get('js_results') or metrics_dict.get('dtw_score') or metrics_dict.get('dtw_distance')
    if dtw_val is not None:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "DTW (JS Divergence)", 
            "Value": np.mean(dtw_val) if hasattr(dtw_val, '__iter__') and not isinstance(dtw_val, (int, float)) else dtw_val, 
            "Goal": "lower", 
            "Description": "DTW-based distribution distance."
        })


    # 3. Distribution Fidelity
    if 'dist_results' in metrics_dict:
        dist_res = metrics_dict['dist_results']
        if "Wasserstein_Mean" in dist_res:
            summary_data.append({
                "Category": "Distribution Fidelity", "Metric": "Wasserstein (Mean)", 
                "Value": dist_res.get("Wasserstein_Mean", 0), "Goal": "lower", 
                "Description": "Earth Mover's Distance between features."
            })
        if "KS_Stat_Mean" in dist_res:
            summary_data.append({
                "Category": "Distribution Fidelity", "Metric": "KS Test Stat (Mean)", 
                "Value": dist_res.get("KS_Stat_Mean", 0), "Goal": "lower", 
                "Description": "Kolmogorov-Smirnov statistic."
            })

    # 4. New Metrics
    if 'mem_ratio' in metrics_dict:
        summary_data.append({
            "Category": "Integrity", "Metric": "Memorization Ratio", 
            "Value": metrics_dict['mem_ratio'], "Goal": "lower", 
            "Description": "Fraction of memorized samples."
        })
    if 'precision' in metrics_dict:
        summary_data.append({
            "Category": "Manifold", "Metric": "α-Precision", 
            "Value": metrics_dict['precision'], "Goal": "higher", 
            "Description": "Fidelity (synth in real manifold)."
        })
    if 'recall' in metrics_dict:
        summary_data.append({
            "Category": "Manifold", "Metric": "β-Recall", 
            "Value": metrics_dict['recall'], "Goal": "higher", 
            "Description": "Diversity (real covered by synth)."
        })
    if 'dcr_score' in metrics_dict:
        summary_data.append({
            "Category": "Integrity", "Metric": "DCR (Mean)", 
            "Value": metrics_dict['dcr_score'], "Goal": "higher", 
            "Description": "Distance to closest record."
        })

    df = pd.DataFrame(summary_data)

    if df.empty:
        return df

    def style_value(row):
        val = row['Value']
        goal = row['Goal']
        
        if goal == 'higher':
            if val >= 0.9: return 'background-color: #2ecc71; color: black'
            elif val >= 0.7: return 'background-color: #82e0aa; color: black'
            elif val >= 0.5: return 'background-color: #f9e79f; color: black'
            else: return 'background-color: #e74c3c; color: black'
        else:
            if val <= 0.05: return 'background-color: #2ecc71; color: black'
            elif val <= 0.15: return 'background-color: #82e0aa; color: black'
            elif val <= 0.3: return 'background-color: #f9e79f; color: black'
            else: return 'background-color: #e74c3c; color: black'

    styled = df.style.apply(
        lambda row: [style_value(row) if col == 'Value' else '' for col in df.columns], 
        axis=1
    )
    
    return styled


def display_feature_stats(real_stats_detail, gen_stats_detail):
    """Display the detailed per-feature statistics dataframe."""
    n_features = len(real_stats_detail['Mean'])
    feature_data = []

    for f in range(n_features):
        for stat_name in real_stats_detail:
            r_val = real_stats_detail[stat_name][f]
            g_val = gen_stats_detail[stat_name][f]
            diff = abs(r_val - g_val)
            feature_data.append({
                "Feature": f,
                "Stat": stat_name,
                "Real": r_val,
                "Synthetic": g_val,
                "Abs Diff": diff
            })

    df_features = pd.DataFrame(feature_data)
    return df_features.style.background_gradient(cmap='RdYlGn_r', subset=['Abs Diff'])
