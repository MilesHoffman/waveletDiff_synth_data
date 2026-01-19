"""
Inline Evaluation Callback for WaveletDiff Training.

Runs periodic inference and computes deterministic metrics to monitor
synthetic data quality during training.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import acf


class InlineEvaluationCallback(pl.Callback):
    """
    Runs inference and evaluates synthetic data quality every N epochs.
    
    Metrics computed:
    - OHLC Invariants: % of valid OHLCV sequences (High >= Low, etc.)
    - Memorization: Distance to nearest neighbor statistics
    - Tail Fidelity: VaR (99th percentile of returns) comparison
    - Temporal Fidelity: ACF MSE on returns
    - Distribution: Wasserstein distance on marginals
    """
    
    def __init__(
        self,
        data_module,
        eval_every_n_epochs: int = 200,
        n_samples: int = 500,
        ohlcv_indices: dict = None
    ):
        """
        Args:
            data_module: The WaveletTimeSeriesDataModule instance
            eval_every_n_epochs: How often to run evaluation
            n_samples: Number of synthetic samples to generate
            ohlcv_indices: Dict mapping OHLCV columns to indices, e.g. {'open': 0, 'high': 1, 'low': 2, 'close': 3}
                          If None, OHLC invariant check is skipped.
        """
        super().__init__()
        self.data_module = data_module
        self.eval_every = eval_every_n_epochs
        self.n_samples = n_samples
        self.ohlcv_indices = ohlcv_indices
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.eval_every != 0:
            return
            
        print(f"\n{'='*60}")
        print(f"INLINE EVALUATION - Epoch {epoch}")
        print(f"{'='*60}")
        
        # Generate synthetic samples
        synth_wavelet = self._generate_samples(pl_module)
        synth_ts = self.data_module.convert_wavelet_to_timeseries(synth_wavelet)
        synth_ts = synth_ts.numpy()
        
        # Get real samples for comparison
        real_ts = self.data_module.raw_data_tensor[:self.n_samples].numpy()
        
        results = {}
        
        # 1. OHLC Invariants (if applicable)
        if self.ohlcv_indices is not None:
            ohlc_valid = self._check_ohlc_invariants(synth_ts)
            results['OHLC_Valid_Pct'] = ohlc_valid * 100
            print(f"  OHLC Valid:       {ohlc_valid*100:.1f}%")
        
        # 2. Memorization (Geometric Fidelity)
        mem_stats = self._compute_memorization_stats(real_ts, synth_ts)
        results.update(mem_stats)
        print(f"  NN Dist Min:      {mem_stats['NN_Dist_Min']:.4f}")
        print(f"  NN Dist Avg:      {mem_stats['NN_Dist_Avg']:.4f}")
        print(f"  NN Dist Median:   {mem_stats['NN_Dist_Median']:.4f}")
        
        # 3. Tail Fidelity (VaR)
        var_diff = self._compute_var_difference(real_ts, synth_ts)
        results['VaR_99_Diff'] = var_diff
        print(f"  VaR 99 Diff:      {var_diff:.4f}")
        
        # 4. Temporal Fidelity (ACF MSE)
        acf_mse = self._compute_acf_mse(real_ts, synth_ts)
        results['ACF_MSE'] = acf_mse
        print(f"  ACF MSE:          {acf_mse:.6f}")
        
        # 5. Distribution (Wasserstein)
        w_dist = self._compute_wasserstein(real_ts, synth_ts)
        results['Wasserstein_Mean'] = w_dist
        print(f"  Wasserstein:      {w_dist:.4f}")
        
        print(f"{'='*60}\n")
        
        # Log to trainer
        for k, v in results.items():
            pl_module.log(f"eval/{k}", v, prog_bar=False)
    
    def _generate_samples(self, pl_module) -> torch.Tensor:
        """Generate synthetic wavelet samples using DDIM (fast)."""
        pl_module.eval()
        device = pl_module.device
        
        # Get shape from data module
        sample_shape = self.data_module.data_tensor.shape[1:]  # (n_coeffs, n_features)
        
        with torch.no_grad():
            # Start from pure noise
            x_t = torch.randn(self.n_samples, *sample_shape, device=device)
            
            # Simple DDPM reverse process (can be replaced with DDIM for speed)
            T = pl_module.T
            # Use registered buffers directly
            # beta_all corresponds to the beta schedule
            # alpha_bar_all corresponds to cumulative product of alphas
            betas = pl_module.beta_all
            alphas_cumprod = pl_module.alpha_bar_all
            
            # DDIM-style skip (use 50 steps instead of 1000)
            step_size = max(1, T // 50)
            timesteps = list(range(T - 1, -1, -step_size))
            
            for i, t in enumerate(timesteps):
                t_tensor = torch.full((self.n_samples,), t, device=device, dtype=torch.long)
                t_norm = t_tensor.float() / T
                
                # Predict noise
                predicted_noise = pl_module(x_t, t_norm)
                
                # Compute x_{t-1}
                alpha_t = alphas_cumprod[t]
                alpha_prev = alphas_cumprod[max(0, t - step_size)] if t > 0 else torch.tensor(1.0, device=device)
                
                # DDIM update (deterministic)
                x0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                x_t = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * predicted_noise
        
        pl_module.train()
        return x_t.cpu()
    
    def _check_ohlc_invariants(self, synth_ts: np.ndarray) -> float:
        """Check % of samples where OHLC invariants hold."""
        o_idx = self.ohlcv_indices.get('open', 0)
        h_idx = self.ohlcv_indices.get('high', 1)
        l_idx = self.ohlcv_indices.get('low', 2)
        c_idx = self.ohlcv_indices.get('close', 3)
        
        # For each sample, check if High >= max(Open, Close) and Low <= min(Open, Close)
        opens = synth_ts[:, :, o_idx]
        highs = synth_ts[:, :, h_idx]
        lows = synth_ts[:, :, l_idx]
        closes = synth_ts[:, :, c_idx]
        
        # Check invariants across all timesteps
        high_valid = highs >= np.maximum(opens, closes)
        low_valid = lows <= np.minimum(opens, closes)
        high_low_valid = highs >= lows
        
        # A sample is valid if ALL timesteps satisfy ALL invariants
        all_valid = high_valid & low_valid & high_low_valid
        sample_valid = np.all(all_valid, axis=1)
        
        return np.mean(sample_valid)
    
    def _compute_memorization_stats(self, real_ts: np.ndarray, synth_ts: np.ndarray) -> dict:
        """Compute nearest neighbor distance statistics."""
        # Flatten time series: (N, T*D)
        real_flat = real_ts.reshape(real_ts.shape[0], -1)
        synth_flat = synth_ts.reshape(synth_ts.shape[0], -1)
        
        # Find nearest neighbor in real set for each synthetic sample
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(real_flat)
        distances, _ = nbrs.kneighbors(synth_flat)
        distances = distances.flatten()
        
        return {
            'NN_Dist_Min': float(np.min(distances)),
            'NN_Dist_Avg': float(np.mean(distances)),
            'NN_Dist_Median': float(np.median(distances))
        }
    
    def _compute_var_difference(self, real_ts: np.ndarray, synth_ts: np.ndarray) -> float:
        """Compute difference in 99th percentile of returns (Value-at-Risk)."""
        # Calculate returns
        eps = 1e-8
        real_returns = np.diff(real_ts, axis=1) / (real_ts[:, :-1, :] + eps)
        synth_returns = np.diff(synth_ts, axis=1) / (synth_ts[:, :-1, :] + eps)
        
        # Flatten returns
        real_ret_flat = real_returns.flatten()
        synth_ret_flat = synth_returns.flatten()
        
        # 99th percentile (VaR)
        real_var = np.percentile(np.abs(real_ret_flat), 99)
        synth_var = np.percentile(np.abs(synth_ret_flat), 99)
        
        return abs(real_var - synth_var)
    
    def _compute_acf_mse(self, real_ts: np.ndarray, synth_ts: np.ndarray, nlags: int = 20) -> float:
        """Compute MSE between ACF curves of real and synthetic data."""
        n_features = real_ts.shape[2]
        
        def avg_acf(data, nlags):
            """Average ACF across all samples for a single feature."""
            acfs = []
            for sample in data:
                try:
                    acfs.append(acf(sample, nlags=nlags, fft=True))
                except:
                    pass
            return np.mean(acfs, axis=0) if acfs else np.zeros(nlags + 1)
        
        total_mse = 0
        for feat_idx in range(n_features):
            # Calculate returns
            real_ret = np.diff(real_ts[:, :, feat_idx], axis=1)
            synth_ret = np.diff(synth_ts[:, :, feat_idx], axis=1)
            
            real_acf = avg_acf(real_ret, nlags)
            synth_acf = avg_acf(synth_ret, nlags)
            
            total_mse += np.mean((real_acf - synth_acf) ** 2)
        
        return total_mse / n_features
    
    def _compute_wasserstein(self, real_ts: np.ndarray, synth_ts: np.ndarray) -> float:
        """Compute mean Wasserstein distance across features."""
        n_features = real_ts.shape[2]
        
        # Flatten time dimension
        real_flat = real_ts.reshape(-1, n_features)
        synth_flat = synth_ts.reshape(-1, n_features)
        
        w_dists = []
        for feat_idx in range(n_features):
            w_dist = wasserstein_distance(real_flat[:, feat_idx], synth_flat[:, feat_idx])
            w_dists.append(w_dist)
        
        return float(np.mean(w_dists))
