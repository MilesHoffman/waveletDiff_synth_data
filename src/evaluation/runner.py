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
        # Prepare data formats
        data = prepare_evaluation_data(
            real, synth, 
            exclude_volume=self.config.exclude_volume,
            close_col=self.config.close_col
        )
        
        result = EvaluationResult(space=space)
        
        # === Core Metrics (Tier 1) ===
        print("\n--- Core Metrics (Tier 1) ---")
        result.core_metrics = self._run_core_metrics(data)
        
        # === Advanced Metrics (Tier 2) ===
        if self.config.compute_advanced:
            print("\n--- Advanced Metrics (Tier 2) ---")
            result.advanced_metrics = self._run_advanced_metrics(data)
        
        return result
    
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
        # Returns (score, fake_acc, real_acc)
        disc_score, fake_acc, real_acc = discriminative_score(
            data['real']['scaled_01'], 
            data['synth']['scaled_01'],
            iterations=self.config.discriminative_iterations
        )
        metrics['discriminative'] = disc_score
        metrics['discriminative_fake_acc'] = fake_acc
        metrics['discriminative_real_acc'] = real_acc
        print(f"  → Discriminative: {metrics['discriminative']:.4f} (Real: {real_acc:.2f}, Fake: {fake_acc:.2f})")
        
        # 2. Predictive Utility (TSTR / TRTR)
        print("Computing Predictive Utility (TSTR/TRTR)...")
        tstr, trtr, gap = predictive_utility(
            data['real']['scaled_01'], 
            data['synth']['scaled_01'],
            iterations=self.config.predictive_iterations
        )
        metrics['predictive_tstr'] = tstr
        metrics['predictive_trtr'] = trtr
        metrics['utility_gap'] = gap
        print(f"  → TSTR: {tstr:.4f}, TRTR: {trtr:.4f}, Gap: {gap:.4f}")
        
        # 3. Context-FID
        print("Computing Context-FID...")
        try:
            metrics['context_fid'] = context_fid(
                data['real']['raw'], 
                data['synth']['raw']
            )
            print(f"  → Context-FID: {metrics['context_fid']:.4f}")
        except Exception as e:
            warnings.warn(f"Context-FID failed: {e}")
            metrics['context_fid'] = float('nan')
        
        # 4. Correlation Score
        print("Computing Correlation Score...")
        metrics['correlation'] = correlation_score(
            data['real']['raw'], 
            data['synth']['raw'],
            sample_size=self.config.correlation_sample_size
        )
        print(f"  → Correlation: {metrics['correlation']:.4f}")
        
        # 5. DTW Distance
        print("Computing DTW Distance...")
        dtw_result = dtw_distance(
            data['real']['raw'], 
            data['synth']['raw'],
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
    
    def _run_advanced_metrics(self, data: dict) -> dict:
        """Run Tier 2 metrics."""
        from .advanced_metrics import (
            js_divergence, 
            acf_similarity,
            alpha_precision, 
            beta_recall,
            dcr_score, 
            memorization_ratio
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
        
        # Statistician
        print("Computing Statistician metrics...")
        metrics['statistician'] = {
            'alpha_precision': alpha_precision(
                data['real']['scaled_01'],
                data['synth']['scaled_01'],
                k=self.config.manifold_k
            ),
            'beta_recall': beta_recall(
                data['real']['scaled_01'],
                data['synth']['scaled_01'],
                k=self.config.manifold_k
            ),
        }
        print(f"  → α-Precision: {metrics['statistician']['alpha_precision']:.4f}")
        print(f"  → β-Recall: {metrics['statistician']['beta_recall']:.4f}")
        
        # Integrity Officer
        print("Computing Integrity Officer metrics...")
        dcr_stats = dcr_score(
            data['real']['flattened'],
            data['synth']['flattened']
        )
        mem_ratio = memorization_ratio(
            data['real']['flattened'],
            data['synth']['flattened'],
            k=self.config.memorization_k
        )
        metrics['integrity_officer'] = {
            'dcr': dcr_stats,
            'memorization_ratio': mem_ratio,
        }
        print(f"  → DCR (mean): {dcr_stats['mean']:.4f}")
        print(f"  → Memorization Ratio: {mem_ratio:.4f}")
        
        return metrics
