
import pandas as pd
import numpy as np

def generate_summary_scorecard(metrics_dict):
    """
    Generate a styled Pandas DataFrame scorecard from a dictionary of metrics.
    
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

    # 2. Discriminative & Predictive
    if 'discriminative_score' in metrics_dict:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "Discriminative Score", 
            "Value": np.mean(metrics_dict['discriminative_score']), 
            "Goal": "lower", 
            "Description": "Classifier accuracy deviation from 0.5 (Real vs Fake)."
        })
    if 'predictive_score' in metrics_dict:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "Predictive Score", 
            "Value": np.mean(metrics_dict['predictive_score']), 
            "Goal": "lower", 
            "Description": "MAE of TSTR (Train on Synthetic, Test on Real)."
        })
    if 'context_fid_score' in metrics_dict:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "Context-FID", 
            "Value": np.mean(metrics_dict['context_fid_score']), 
            "Goal": "lower", 
            "Description": "FID score on embeddings (e.g. Inception/Transformer)."
        })
    if 'correlational_score' in metrics_dict:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "Cross-Correl Loss", 
            "Value": np.mean(metrics_dict['correlational_score']), 
            "Goal": "lower", 
            "Description": "Difference in cross-correlation matrices."
        })
    if 'js_results' in metrics_dict:
        summary_data.append({
            "Category": "Model Quality", 
            "Metric": "DTW (JS Divergence)", 
            "Value": np.mean(metrics_dict['js_results']), 
            "Goal": "lower", 
            "Description": "DTW-based distribution distance."
        })
    if 'dist_results' in metrics_dict and 'JS_Div_Mean' in metrics_dict['dist_results']:
         summary_data.append({
            "Category": "Fidelity (Vibe)", 
            "Metric": "JS Divergence (Mean)", 
            "Value": metrics_dict['dist_results']['JS_Div_Mean'], 
            "Goal": "lower", 
            "Description": "Jensen-Shannon Divergence on Log-Returns."
        })


    # 3. Distribution Fidelity (from dist_results)
    if 'dist_results' in metrics_dict:
        dist_res = metrics_dict['dist_results']
        summary_data.append({
            "Category": "Distribution Fidelity", "Metric": "Wasserstein (Mean)", 
            "Value": dist_res.get("Wasserstein_Mean", 0), "Goal": "lower", "Description": "Earth Mover's Distance between features."
        })
        summary_data.append({
            "Category": "Distribution Fidelity", "Metric": "KS Test Stat (Mean)", 
            "Value": dist_res.get("KS_Stat_Mean", 0), "Goal": "lower", "Description": "Kolmogorov-Smirnov statistic (max diff in CDF)."
        })
        summary_data.append({
            "Category": "Distribution Fidelity", "Metric": "KS P-Value (Mean)", 
            "Value": dist_res.get("KS_PVal_Mean", 0), "Goal": "higher", "Description": "Statistical significance of KS test."
        })

    # 4. Structural Alignment
    if 'struct_results' in metrics_dict:
        struct_res = metrics_dict['struct_results']
        summary_data.append({
            "Category": "Structural Alignment", "Metric": "PCA EVR Correlation", 
            "Value": struct_res.get("PCA_EVR_Corr", 0), "Goal": "higher", "Description": "Correlation of PCA Explained Variance Ratios."
        })
        # t-SNE 1-NN: ideal is 0.5, so we show distance from 0.5
        tsne_val = struct_res.get("tSNE_1NN_Acc", 0.5)
        tsne_error = abs(tsne_val - 0.5)
        summary_data.append({
            "Category": "Structural Alignment", "Metric": "t-SNE 1-NN (|x-0.5|)", 
            "Value": tsne_error, "Goal": "lower", "Description": "Classifier accuracy in t-SNE space (Ideal=0.5)."
        })

    # 5. Financial Reality
    if 'fin_results' in metrics_dict:
        fin_res = metrics_dict['fin_results']
        summary_data.append({
            "Category": "Financial Reality", "Metric": "ACF MSE (Lags 1,5,20)", 
            "Value": fin_res.get("ACF_MSE", 0), "Goal": "lower", "Description": "MSE of Autocorrelation Functions."
        })
        summary_data.append({
            "Category": "Financial Reality", "Metric": "Cross-Corr Matrix Diff", 
            "Value": fin_res.get("CrossCorr_Norm_Diff", 0), "Goal": "lower", "Description": "Norm difference of correlation matrices."
        })
        summary_data.append({
            "Category": "Financial Reality", "Metric": "Volatility Clustering MSE", 
            "Value": fin_res.get("Volatility_MSE", 0), "Goal": "lower", "Description": "MSE of squared returns ACF (Volatility)."
        })

    # 6. New Metrics
    if 'mem_ratio' in metrics_dict:
        summary_data.append({
            "Category": "New Metrics", "Metric": "Memorization Ratio (1/3 Rule)", 
            "Value": metrics_dict['mem_ratio'], "Goal": "lower", "Description": "Fraction of samples that are near-duplicates of training data."
        })
    if 'div_results' in metrics_dict:
        summary_data.append({
            "Category": "New Metrics", "Metric": "Diversity (Coverage)", 
            "Value": metrics_dict['div_results'].get("Coverage", 0), "Goal": "higher", "Description": "Fraction of real data covered by synthetic samples."
        })
    if 'fld_score' in metrics_dict:
        summary_data.append({
            "Category": "New Metrics", "Metric": "FLD (Likelihood Divergence)", 
            "Value": metrics_dict['fld_score'], "Goal": "lower", "Description": "Divergence in likelihood under real data density (GMM)."
        })

    # New Diversity & Overfitting Metrics (from dcr_score, precision, recall, mmd_score args if present)
    if 'dcr_score' in metrics_dict:
         summary_data.append({
            "Category": "Diversity & Overfitting", "Metric": "DCR (Distance to Closest Record)", 
            "Value": metrics_dict['dcr_score'], "Goal": "higher", "Description": "Mean distance to nearest real neighbor (Low = Memorization)."
        })
    if 'precision' in metrics_dict:
         summary_data.append({
            "Category": "Diversity & Overfitting", "Metric": "Manifold Precision", 
            "Value": metrics_dict['precision'], "Goal": "higher", "Description": "Fidelity: % of synth samples in real manifold."
        })
    if 'recall' in metrics_dict:
         summary_data.append({
            "Category": "Diversity & Overfitting", "Metric": "Manifold Recall", 
            "Value": metrics_dict['recall'], "Goal": "higher", "Description": "Diversity: % of real samples covered by synth manifold."
        })
    if 'mmd_score' in metrics_dict:
         summary_data.append({
            "Category": "Diversity & Overfitting", "Metric": "MMD (RBF Kernel)", 
            "Value": metrics_dict['mmd_score'], "Goal": "lower", "Description": "Maximum Mean Discrepancy between distributions."
        })

    # Create DataFrame
    df_results = pd.DataFrame(summary_data)

    if df_results.empty:
        return df_results

    # Custom styling based on Goal (with black text for readability)
    def style_value(row):
        val = row['Value']
        goal = row['Goal']
        
        if goal == 'higher':
            # Higher is better: Green for high, Red for low
            if val >= 0.99: return 'background-color: #2ecc71; color: black'
            elif val >= 0.9: return 'background-color: #82e0aa; color: black'
            elif val >= 0.7: return 'background-color: #f9e79f; color: black'
            else: return 'background-color: #e74c3c; color: black'
        else:
            # Lower is better: Green for low, Red for high  
            if val <= 0.01: return 'background-color: #2ecc71; color: black'
            elif val <= 0.05: return 'background-color: #82e0aa; color: black'
            elif val <= 0.15: return 'background-color: #f9e79f; color: black'
            elif val <= 0.3: return 'background-color: #f5b041; color: black'
            else: return 'background-color: #e74c3c; color: black'

    # Apply style
    styled = df_results.style.apply(lambda row: [style_value(row) if col == 'Value' else '' for col in df_results.columns], axis=1)
    
    # Set display options globally for this cell context (side effect, but useful in notebook)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    
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
