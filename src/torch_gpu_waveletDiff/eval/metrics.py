import sys
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

# Ensure WaveletDiff_source/src is in path
current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent
wd_source = repo_root / "src" / "copied_waveletDiff" / "src"
if str(wd_source) not in sys.path and wd_source.exists():
    sys.path.append(str(wd_source))
    
# Also add 'evaluation' submodule to path for internal imports (like metric_utils)
wd_eval = wd_source / "evaluation"
if str(wd_eval) not in sys.path and wd_eval.exists():
    sys.path.append(str(wd_eval))

# Imports (Exceptions removed to allow debugging of missing dependencies)
from evaluation.discriminative_metrics import discriminative_score_metrics
from evaluation.predictive_metrics import predictive_score_metrics
try:
    from evaluation.context_fid import Context_FID
except ImportError:
    # Optional dependency, might fail if torch-fidelity not installed
    Context_FID = None
try:
    from evaluation.cross_correlation import CrossCorrelLoss
except ImportError:
    CrossCorrelLoss = None
from evaluation.dtw import dtw_js_divergence_distance
from evaluation.metric_utils import display_scores

try:
    from . import visualizations
    from . import stylized_facts
except ImportError:
    import visualizations
    import stylized_facts

# Baseline metrics from WaveletDiff paper (Stocks dataset, best performers)
# Discriminative: 0.005 (sym)
# Predictive: 0.037 (all)
# Context-FID: 0.016 (bior/rbio)
# Correlational: 0.003 (rbio)
# DTW-JS: 0.106 (sym)
BASELINE_STOCKS = {
    'discriminative': 0.005,
    'predictive': 0.037,
    'context_fid': 0.016,
    'correlation': 0.003,
    'dtw': 0.106
}

INTERPRETATION = {
    'discriminative': "Lower is Better (Closer to 0.0 means indistinguishable)",
    'predictive': "Lower is Better (TSTR MAE)",
    'context_fid': "Lower is Better (Distributional distance)",
    'correlation': "Lower is Better (Temporal dependency diff)",
    'dtw': "Lower is Better (Temporal alignment JS distance)"
}

class MetricsEvaluator:
    def __init__(self, real_data, generated_data, device='cuda'):
        """
        Args:
            real_data (np.ndarray): Real samples [N, T, D]
            generated_data (np.ndarray): Generated samples [N, T, D]
        """
        self.real_data = real_data
        self.generated_data = generated_data
        self.device = device
        
        # Preprocessing: MinMax Scale (as in evaluation.ipynb)
        self._preprocess()
        
    def _preprocess(self):
        # Align sample counts
        n_real = self.real_data.shape[0]
        n_gen = self.generated_data.shape[0]
        n_samples = min(n_real, n_gen)
        
        if n_real > n_samples:
            idx = np.random.choice(n_real, n_samples, replace=False)
            self.real_data = self.real_data[idx]
            
        if n_gen > n_samples:
            idx = np.random.choice(n_gen, n_samples, replace=False)
            self.generated_data = self.generated_data[idx]
            
        # MinMax Scale
        data_min = np.min(self.real_data, axis=(0,1), keepdims=True)
        data_max = np.max(self.real_data, axis=(0,1), keepdims=True)
        
        # Avoid division by zero
        denom = data_max - data_min
        denom[denom == 0] = 1.0
        
        self.real_data_norm = (self.real_data - data_min) / denom
        self.gen_data_norm = (self.generated_data - data_min) / denom
        
    def evaluate_discriminative(self, iterations=5):
        print("Running Discriminative Score...")
        scores = []
        for i in tqdm(range(iterations), desc="Discriminative Score"):
            score, _, _ = discriminative_score_metrics(self.real_data_norm, self.gen_data_norm)
            scores.append(score)
            # print(f"Iter {i}: {score:.4f}") # Suppress individual print if using tqdm
        return np.mean(scores), scores

    def evaluate_predictive(self, iterations=5):
        print("Running Predictive Score...")
        scores = []
        for i in tqdm(range(iterations), desc="Predictive Score"):
            score = predictive_score_metrics(self.real_data_norm, self.gen_data_norm)
            scores.append(score)
            # print(f"Iter {i}: {score:.4f}")
        return np.mean(scores), scores

    def evaluate_context_fid(self, iterations=5):
        print("Running Context-FID Score...")
        scores = []
        for i in tqdm(range(iterations), desc="Context-FID"):
            score = Context_FID(self.real_data_norm, self.gen_data_norm)
            scores.append(score)
            # print(f"Iter {i}: {score:.4f}")
        return np.mean(scores), scores

    def evaluate_correlation(self, iterations=5, sample_size=1000):
        print("Running Correlation Score...")
        scores = []
        x_real = torch.from_numpy(self.real_data_norm).float().to(self.device)
        x_fake = torch.from_numpy(self.gen_data_norm).float().to(self.device)
        
        # Ensure sample_size is valid
        sample_size = min(sample_size, x_real.shape[0])
        
        for i in tqdm(range(iterations), desc="Correlation Score"):
            real_idx = np.random.choice(x_real.shape[0], sample_size, replace=False)
            fake_idx = np.random.choice(x_fake.shape[0], sample_size, replace=False)
            
            # CrossCorrelLoss expects inputs? imports might serve class
            # Inspecting notebook: 
            # corr = CrossCorrelLoss(x_real[real_idx], ...)
            # loss = corr.compute(x_fake[fake_idx])
            
            corr = CrossCorrelLoss(x_real[real_idx], name='CrossCorrelLoss')
            loss = corr.compute(x_fake[fake_idx])
            scores.append(loss.item())
            # print(f"Iter {i}: {loss.item():.4f}")
            
        return np.mean(scores), scores

    def evaluate_dtw(self, iterations=5):
        print("Running DTW Distance...")
        scores = []
        for i in tqdm(range(iterations), desc="DTW Distance"):
            # dtw_js_divergence_distance args: real, fake, n_samples
            res = dtw_js_divergence_distance(self.real_data_norm, self.gen_data_norm, n_samples=100)
            score = res['js_divergence']
            scores.append(score)
            # print(f"Iter {i}: {score:.4f}")
        return np.mean(scores), scores

        return np.mean(scores), scores

    def evaluate_stylized_facts(self):
        print("Calculating Stylized Facts...")
        results = {}
        
        # Kurtosis
        kurt = stylized_facts.calculate_kurtosis(self.real_data_norm, self.gen_data_norm)
        results.update(kurt)
        
        # Leverage Effect
        lev = stylized_facts.leverage_effect(self.real_data_norm, self.gen_data_norm)
        results.update(lev)
        
        return results

    def visualize_all(self, save_dir=None):
        print("Generating Visualizations...")
        feat_names = [f"Feature {i}" for i in range(self.real_data.shape[2])]
        
        # 1. t-SNE
        visualizations.plot_tsne(self.real_data_norm, self.gen_data_norm, save_path=f"{save_path}/tsne.png" if save_dir else None)
        
        # 2. Distributions (exclude last dim if it is volume? User said exclude volume. 
        # Usually Vol is last or first. Assuming last for now or asking user? 
        # User said "exclude volume... since volume is often larger".
        # I'll Assume Volume is the last feature if >1 features, or pass logic to exclude)
        # For now, I'll exclude the feature with highest variance/mean if not specified? 
        # Or just plot all. I'll plot all but maybe mention to user to specify index.
        # User constraint: "exclude volume from the visual comparison for clarity".
        # I'll assume Volume is index 5 (last) based on typical OHLCV + something? 
        # Actually user mentioned 6 dims in previous turn.
        # I'll try to guess volume by max value range?
        # self.real_data is UNNORMALIZED for this check?
        # self._preprocess normalizes. I should use unnormalized for volume check logic or just exclude the one with largest scale.
        
        # Heuristic: Feature with largest max value is likely volume.
        max_vals = np.max(self.real_data, axis=(0,1))
        vol_idx = np.argmax(max_vals)
        visualizations.plot_distributions(self.real_data_norm, self.gen_data_norm, exclude_indices=[vol_idx], save_path=f"{save_path}/dist.png" if save_dir else None)
        
        # 3. Scalogram (single sample, Feature 0 - Close/Price)
        visualizations.plot_scalogram(self.real_data_norm[0, :, 0], self.gen_data_norm[0, :, 0], save_path=f"{save_path}/cwt.png" if save_dir else None)
        
        # 4. Q-Q Plot (Feature 0)
        visualizations.plot_qq(self.real_data_norm, self.gen_data_norm, feature_idx=0, save_path=f"{save_path}/qq.png" if save_dir else None)
        
        # 5. PSD (Feature 0)
        visualizations.plot_psd(self.real_data_norm, self.gen_data_norm, feature_idx=0, save_path=f"{save_path}/psd.png" if save_dir else None)
        
        # 6. ACF (Feature 0)
        visualizations.plot_acf(self.real_data_norm, self.gen_data_norm, feature_idx=0, save_path=f"{save_path}/acf.png" if save_dir else None)

    def print_comparison(self, results):
        print("\n" + "="*80)
        print(f"{'METRIC':<20} | {'YOUR SCORE':<15} | {'BASELINE (STOCKS)':<20} | {'STATUS':<10}")
        print("-" * 80)
        
        for key in BASELINE_STOCKS:
            if key in results:
                score = results[key]
                base = BASELINE_STOCKS[key]
                # Assuming all listed metrics: Lower is Better
                # Diff: score - base. If > 0, worse.
                diff = score - base
                status = "GOOD" if diff <= 0.005 else ("OK" if diff <= 0.02 else "BAD") # Tolerance
                
                print(f"{key:<20} | {score:<15.4f} | {base:<20.4f} | {status:<10}")
                print(f"  -> {INTERPRETATION.get(key, '')}")
        print("="*80 + "\n")
        
        print("STYLIZED FACTS COMPARISON:")
        if 'real_kurtosis' in results:
            print(f"Kurtosis: Real={results['real_kurtosis']:.4f}, Syn={results['syn_kurtosis']:.4f}")
            print("  -> Should match (High kurtosis indicates fat tails)")
            
        if 'real_leverage' in results:
            print(f"Leverage Effect: Real={results['real_leverage']:.4f}, Syn={results['syn_leverage']:.4f}")
            print("  -> Should be negative and similar magnitude.")
        
        if 'utility_mae' in results:
             print(f"Utility Score (TSTR MAE): {results['utility_mae']:.4f}")
             print("  -> Lower is better. Compare with Predictive Score (Train on Real, Test on Real) if available.")


    def run_all(self, iterations=5):
        results = {}
        results['discriminative'] = self.evaluate_discriminative(iterations)[0]
        results['predictive'] = self.evaluate_predictive(iterations)[0]
        results['context_fid'] = self.evaluate_context_fid(iterations)[0]
        if torch.cuda.is_available() or self.device == 'cpu':
             results['correlation'] = self.evaluate_correlation(iterations)[0]
        results['dtw'] = self.evaluate_dtw(iterations)[0]
        
        # Stylized Facts
        sf = self.evaluate_stylized_facts()
        results.update(sf)
        
        self.print_comparison(results)
        
        return results
