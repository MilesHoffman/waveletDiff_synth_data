"""
EvaluationRunner - Orchestrates evaluation across both data spaces.

Provides a clean API for running all metrics with proper data preprocessing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import warnings

from .preprocessing import prepare_evaluation_data


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    n_iterations: int = 1
    exclude_volume: bool = True
    compute_advanced: bool = True
    compute_legacy: bool = True
    generate_plots: bool = False
    
    # Core metric parameters
    discriminative_iterations: int = 2000
    predictive_iterations: int = 5000
    dtw_n_samples: int = 100
    correlation_sample_size: int = 1000
    
    # Advanced metric parameters
    acf_nlags: int = 20
    manifold_k: int = 3
    memorization_k: int = 2
    
    # Data parameters
    close_col: int = 3  # Index of close price for log-returns


@dataclass 
class EvaluationResult:
    """Container for all evaluation results."""
    space: str  # 'dollar' or 'reparam'
    core_metrics: Dict[str, Any] = field(default_factory=dict)
    advanced_metrics: Dict[str, Any] = field(default_factory=dict)


class EvaluationRunner:
    """
    Orchestrates evaluation across both data spaces.
    
    Usage:
        config = EvaluationConfig(exclude_volume=True)
        runner = EvaluationRunner(config)
        results = runner.run(
            real_dollar=real_ohlcv,
            synth_dollar=synth_ohlcv,
            real_reparam=real_norm,
            synth_reparam=synth_norm
        )
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def run(
        self, 
        real_dollar: np.ndarray, 
        synth_dollar: np.ndarray,
        real_reparam: Optional[np.ndarray] = None, 
        synth_reparam: Optional[np.ndarray] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Run full evaluation suite on both data spaces.
        
        Args:
            real_dollar: Real data in dollar space (N, T, D)
            synth_dollar: Synthetic data in dollar space (N, T, D)
            real_reparam: Real data in reparameterized space (optional)
            synth_reparam: Synthetic data in reparameterized space (optional)
            
        Returns:
            Dict with 'dollar' and optionally 'reparam' EvaluationResult objects
        """
        results = {}
        
        # Dollar Space Evaluation
        print("=" * 60)
        print("EVALUATION: DOLLAR SPACE")
        print("=" * 60)
        results['dollar'] = self._evaluate_space(
            real_dollar, synth_dollar, space='dollar'
        )
        
        # Reparam Space Evaluation (if available)
        if real_reparam is not None and synth_reparam is not None:
            print("\n" + "=" * 60)
            print("EVALUATION: REPARAMETERIZED SPACE")
            print("=" * 60)
            results['reparam'] = self._evaluate_space(
                real_reparam, synth_reparam, space='reparam'
            )
        
        return results
    
    def _evaluate_space(
        self, 
        real: np.ndarray, 
        synth: np.ndarray, 
        space: str
    ) -> EvaluationResult:
        """Evaluate a single data space."""
        # Detect if this is the reparameterized space
        is_reparam = (space == 'reparam')
        
        # Prepare data formats
        data = prepare_evaluation_data(
            real, synth, 
            exclude_volume=self.config.exclude_volume,
            close_col=self.config.close_col,
            is_reparam=is_reparam
        )
        
        result = EvaluationResult(space=space)
        
        # === Core Metrics (Tier 1) ===
        print("\n--- Core Metrics (Tier 1) ---")
        result.core_metrics = self._run_core_metrics(data)
        
        # === Advanced Metrics (Tier 2) ===
        if self.config.compute_advanced:
            print("\n--- Advanced Metrics (Tier 2) ---")
            result.advanced_metrics = self._run_advanced_metrics(data, is_reparam)

        # === Legacy Metrics (Source) ===
        if self.config.compute_legacy:
            print("\n--- Legacy Metrics (Source Implementation) ---")
            result.core_metrics.update(self._run_legacy_metrics(data))
            
        # === Visualizations ===
        if self.config.generate_plots:
            print("\n--- Generating Visualizations ---")
            self._run_visualizations(data)
        
        return result
    
    def _run_visualizations(self, data: dict):
        """Generate plots."""
        from .visualizations import plot_financial_stylized_facts
        
        # We use the 'raw' (price-like) data for this plot as it computes its own returns
        try:
            plot_financial_stylized_facts(
                data['real']['raw'],
                data['synth']['raw']
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    def _run_core_metrics(self, data: dict) -> dict:
        """Run Tier 1 metrics."""
        from .core_metrics import (
            discriminative_score, 
            predictive_utility, 
            context_fid, 
            correlation_score, 
            dtw_distance
        )
        
        metrics = {}
        
        # 1. Discriminative Score
        print("Computing Discriminative Score...")
        # OPTIMIZATION: Use standardized data for LSTMs
        disc_score, fake_acc, real_acc = discriminative_score(
            data['real']['standardized'], 
            data['synth']['standardized'],
            iterations=self.config.discriminative_iterations
        )
        metrics['discriminative'] = disc_score
        metrics['discriminative_fake_acc'] = fake_acc
        metrics['discriminative_real_acc'] = real_acc
        print(f"  → Discriminative: {metrics['discriminative']:.4f} (Real: {real_acc:.2f}, Fake: {fake_acc:.2f})")
        
        # 2. Predictive Utility
        print("Computing Predictive Utility (TSTR/TRTR)...")
        # OPTIMIZATION: Use standardized data for LSTMs
        tstr, trtr, gap = predictive_utility(
            data['real']['standardized'], 
            data['synth']['standardized'],
            iterations=self.config.predictive_iterations
        )
        metrics['predictive_tstr'] = tstr
        metrics['predictive_trtr'] = trtr
        metrics['utility_gap'] = gap
        print(f"  → TSTR: {tstr:.4f}, TRTR: {trtr:.4f}, Gap: {gap:.4f}")
        
        # 3. Context-FID
        print("Computing Context-FID...")
        try:
            # OPTIMIZATION: Use standardized data for TS2Vec (FIXED BUG)
            metrics['context_fid'] = context_fid(
                data['real']['standardized'], 
                data['synth']['standardized']
            )
            print(f"  → Context-FID: {metrics['context_fid']:.4f}")
        except Exception as e:
            warnings.warn(f"Context-FID failed: {e}")
            metrics['context_fid'] = float('nan')
        
        # 4. Correlation Score
        print("Computing Correlation Score...")
        # Uses raw/standardized difference? Code uses raw usually.
        # But for 'correlation_score' function it typically wants shape (N, T, D).
        # Let's keep 'raw' here as correlation is scale-invariant (Pearson),
        # but we pass the 'raw' (which is actually 'processed' in prepare_data)
        metrics['correlation'] = correlation_score(
            data['real']['raw'], 
            data['synth']['raw'],
            sample_size=self.config.correlation_sample_size
        )
        print(f"  → Correlation: {metrics['correlation']:.4f}")
        
        # 5. DTW Distance
        print("Computing DTW Distance...")
        # OPTIMIZATION: Use standardized data so features with larger magnitudes don't dominate
        dtw_result = dtw_distance(
            data['real']['standardized'], 
            data['synth']['standardized'],
            n_samples=self.config.dtw_n_samples
        )
        # Handle dict return
        if isinstance(dtw_result, dict):
            metrics['dtw'] = dtw_result['js_divergence']
            metrics['dtw_details'] = dtw_result
        else:
            metrics['dtw'] = dtw_result
            
        print(f"  → DTW (JS Div): {metrics['dtw']:.4f}")
        
        return metrics
    
    def _run_advanced_metrics(self, data: dict, is_reparam: bool = False) -> dict:
        """Run Tier 2 metrics."""
        from .advanced_metrics import (
            js_divergence, 
            acf_similarity,
            alpha_precision, 
            beta_recall,
            dcr_score, 
            memorization_ratio
        )
        from .core_metrics import (
            tail_index_error,
            empirical_var_es_error,
            price_volume_asymmetry_error,
            volume_acf_error,
            volatility_clustering_score
        )
        from .advanced_metrics.financial_metrics import (
            calculate_tail_dependence,
            calculate_hurst_metrics,
            calculate_leverage_effect,
            calculate_drawdown_stats
        )
        
        metrics = {}
        
        # Visual Scout
        print("Computing Visual Scout metrics...")
        metrics['visual_scout'] = {
            'js_divergence': js_divergence(
                data['real']['log_returns'],
                data['synth']['log_returns']
            ),
            'acf_similarity': acf_similarity(
                data['real']['log_returns'],
                data['synth']['log_returns'],
                nlags=self.config.acf_nlags
            ),
        }
        print(f"  → JS Divergence: {metrics['visual_scout']['js_divergence']:.4f}")
        print(f"  → ACF Similarity: {metrics['visual_scout']['acf_similarity']:.4f}")
        
        # Financial Realism (Stylized Facts)
        print("Computing Financial Realism & Tail Risk metrics...")
        metrics['stylized_facts'] = {
            'tail_index_error': tail_index_error(
                data['real']['log_returns'],
                data['synth']['log_returns']
            ),
            'var_es_error': empirical_var_es_error(
                data['real']['log_returns'],
                data['synth']['log_returns']
            ),
            'volatility_clustering': volatility_clustering_score(
                data['real']['log_returns'],
                data['synth']['log_returns']
            )
        }
        
        # Volume features need raw (un-differenced) OHLCV data
        if 'raw' in data['real'] and not self.config.exclude_volume:
            print("Computing Volume Micro-structure metrics...")
            try:
                metrics['stylized_facts']['price_vol_asym'] = price_volume_asymmetry_error(
                    data['real']['raw'],
                    data['synth']['raw']
                )
                metrics['stylized_facts']['volume_acf'] = volume_acf_error(
                    data['real']['raw'],
                    data['synth']['raw']
                )
                print(f"  → Price-Vol Asymmetry Error: {metrics['stylized_facts']['price_vol_asym']:.4f}")
                print(f"  → Volume ACF Error: {metrics['stylized_facts']['volume_acf']:.4f}")
            except Exception as e:
                print(f"  → Volume metrics failed: {e}")

        print(f"  → Tail Index Error: {metrics['stylized_facts']['tail_index_error']:.4f}")
        print(f"  → Risk (VaR/ES) Error: {metrics['stylized_facts']['var_es_error']:.4f}")
        print(f"  → Volatility Clustering MAE: {metrics['stylized_facts']['volatility_clustering']:.4f}")

        # Quantitative Finance (New Tier)
        print("Computing Quantitative Finance metrics...")
        # Use log-returns for financial metrics
        real_ret = data['real']['log_returns'] # (N, T, D)
        synth_ret = data['synth']['log_returns']
        
        # Tail Dependence
        metrics['quant_finance'] = {}
        try:
            td_res = calculate_tail_dependence(real_ret, synth_ret, q=0.05)
            metrics['quant_finance'].update(td_res)
            print(f"  → Tail Dep Diff (Low): {td_res['Tail_Dep_Lower_Diff']:.4f}")
        except Exception as e:
            print(f"  → Tail Dep failed: {e}")

        # Hurst Exponent
        try:
            hurst_res = calculate_hurst_metrics(real_ret, synth_ret)
            metrics['quant_finance'].update(hurst_res)
            print(f"  → Hurst Diff: {hurst_res['Hurst_Diff']:.4f}")
        except Exception as e:
            print(f"  → Hurst failed: {e}")

        # Leverage Effect
        try:
            lev_res = calculate_leverage_effect(real_ret, synth_ret)
            metrics['quant_finance'].update(lev_res)
            print(f"  → Leverage Diff: {lev_res['Leverage_Diff']:.4f}")
        except Exception as e:
            print(f"  → Leverage failed: {e}")
            
        # Drawdown Dynamics
        if not is_reparam:
            try:
                # Drawdowns calculated on Price paths (standardized -> re-cumprod or use raw if available?)
                # Ideally use Raw Price paths if they exist, but 'data' dict might not have them readily in 'raw' if scaled.
                # prepare_evaluation_data 'raw' is usually the original input.
                # Let's use data['real']['raw'] which is presumably prices.
                dd_res = calculate_drawdown_stats(data['real']['raw'], data['synth']['raw'])
                metrics['quant_finance'].update(dd_res)
                print(f"  → Drawdown KS Stat: {dd_res['MaxDD_KS_Stat']:.4f}")
            except Exception as e:
                print(f"  → Drawdown failed: {e}")
        else:
            print("  → Drawdown: Skipped (Not applicable for reparameterized space)")


        # Statistician
        print("Computing Statistician metrics...")
        # OPTIMIZATION: Use standardized for Euclidean distance in manifold
        metrics['statistician'] = {
            'alpha_precision': alpha_precision(
                data['real']['standardized'],
                data['synth']['standardized'],
                k=self.config.manifold_k
            ),
            'beta_recall': beta_recall(
                data['real']['standardized'],
                data['synth']['standardized'],
                k=self.config.manifold_k
            ),
        }
        print(f"  → α-Precision: {metrics['statistician']['alpha_precision']:.4f}")
        print(f"  → β-Recall: {metrics['statistician']['beta_recall']:.4f}")
        
        # Integrity Officer
        print("Computing Integrity Officer metrics...")
        # OPTIMIZATION: Use flattened_standardized to avoid volume domination (FIXED BUG)
        dcr_stats = dcr_score(
            data['real']['flattened_standardized'],
            data['synth']['flattened_standardized']
        )
        metrics['integrity'] = {'dcr': dcr_stats}

        mem_ratio = memorization_ratio(
            data['real']['flattened_standardized'],
            data['synth']['flattened_standardized'],
            k=self.config.memorization_k
        )
        metrics['integrity']['memorization_ratio'] = mem_ratio
        print(f"  → DCR: {dcr_stats:.4f}" if isinstance(dcr_stats, float) else f"  → DCR: {dcr_stats}")
        print(f"  → Memorization Ratio: {mem_ratio:.4f}")

        return metrics

    def _run_legacy_metrics(self, data: dict) -> dict:
        """Run Legacy (Source) metrics for comparison."""
        from .core_metrics.legacy import (
            discriminative_score_legacy,
            predictive_score_legacy
        )
        
        metrics = {}
        
        # 1. Discriminative Score (Legacy GRU)
        print("Computing Legacy Discriminative Score (GRU)...")
        # Legacy typically used min-max scaled data [0,1]
        try:
            disc_score = discriminative_score_legacy(
                data['real']['scaled_01'], 
                data['synth']['scaled_01'],
                iterations=self.config.discriminative_iterations
            )
            metrics['discriminative_legacy'] = disc_score
            print(f"  → Discriminative (Legacy): {metrics['discriminative_legacy']:.4f}")
        except Exception as e:
            warnings.warn(f"Legacy Discriminative failed: {e}")
            metrics['discriminative_legacy'] = float('nan')
            
        # 2. Predictive Score (Legacy 1-step)
        print("Computing Legacy Predictive Score (1-step)...")
        try:
            pred_score = predictive_score_legacy(
                data['real']['scaled_01'],
                data['synth']['scaled_01'],
                iterations=self.config.predictive_iterations
            )
            metrics['predictive_legacy_mae'] = pred_score
            print(f"  → Predictive (Legacy): {metrics['predictive_legacy_mae']:.4f}")
        except Exception as e:
            warnings.warn(f"Legacy Predictive failed: {e}")
            metrics['predictive_legacy_mae'] = float('nan')
            
        return metrics
