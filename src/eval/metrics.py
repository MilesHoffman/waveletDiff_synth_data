import sys
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

# Ensure WaveletDiff_source/src is in path
current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent
wd_source = repo_root / "WaveletDiff_source" / "src"
if str(wd_source) not in sys.path and wd_source.exists():
    sys.path.append(str(wd_source))

try:
    from evaluation.discriminative_metrics import discriminative_score_metrics
except ImportError as e:
    print(f"Warning: Could not import discriminative_metrics: {e}")
    discriminative_score_metrics = None

try:
    from evaluation.predictive_metrics import predictive_score_metrics
except ImportError as e:
    print(f"Warning: Could not import predictive_metrics: {e}")
    predictive_score_metrics = None

try:
    from evaluation.context_fid import Context_FID
except ImportError as e:
    print(f"Warning: Could not import Context_FID: {e}")
    Context_FID = None

try:
    from evaluation.cross_correlation import CrossCorrelLoss
except ImportError as e:
    print(f"Warning: Could not import CrossCorrelLoss: {e}")
    CrossCorrelLoss = None

try:
    from evaluation.dtw import dtw_js_divergence_distance
except ImportError as e:
    print(f"Warning: Could not import dtw: {e}")
    dtw_js_divergence_distance = None

try:
    from evaluation.metric_utils import display_scores
except ImportError as e:
    print(f"Warning: Could not import metric_utils: {e}")

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

    def run_all(self, iterations=5):
        results = {}
        results['discriminative'] = self.evaluate_discriminative(iterations)[0]
        results['predictive'] = self.evaluate_predictive(iterations)[0]
        results['context_fid'] = self.evaluate_context_fid(iterations)[0]
        if torch.cuda.is_available() or self.device == 'cpu':
             results['correlation'] = self.evaluate_correlation(iterations)[0]
        results['dtw'] = self.evaluate_dtw(iterations)[0]
        return results
