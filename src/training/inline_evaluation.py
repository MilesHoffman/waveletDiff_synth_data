"""
Inline Evaluation Callback for WaveletDiff Training.

Runs periodic inference and computes deterministic metrics to monitor
synthetic data quality during training.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance, skew, kurtosis
from statsmodels.tsa.stattools import acf
import time


class InlineEvaluationCallback(pl.Callback):
    """
    Runs inference and evaluates synthetic data quality every N epochs.
    
    Metrics computed:
    - OHLC Invariants: % of valid OHLCV sequences (High >= Low, etc.)
    - Memorization: Distance to nearest neighbor statistics
    - Tail Fidelity: VaR (99th percentile of returns) comparison
    - Temporal Fidelity: ACF MSE on returns
    - Distribution: Wasserstein distance on marginals
    - Correlation: Frobenius norm of correlation matrix difference
    - Moments: Skewness/Kurtosis drift
    """
    
    def __init__(
        self,
        data_module,
        eval_every_n_epochs: int = 200,
        n_samples: int = 128,
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
            
        print(f"\n> Inline Evaluation Metrics (Every {self.eval_every} epochs)")
        
        # Prepare scale conditioning if available
        scale = None
        if getattr(self.data_module, 'has_conditioning', False):
            atr_pcts = self.data_module.norm_stats['atr_pcts']
            indices = np.random.choice(len(atr_pcts), size=self.n_samples, replace=True)
            scale = torch.FloatTensor(atr_pcts[indices]).to(pl_module.device)

        # Generate synthetic samples
        synth_wavelet = self._generate_samples(pl_module, scale=scale)
        synth_ts_norm = self.data_module.convert_wavelet_to_timeseries(synth_wavelet).cpu().numpy()
        
        # Get real samples (normalized)
        real_ts_norm = self.data_module.raw_data_tensor[:self.n_samples].cpu().numpy()
        
        results = {}
        
        # 1. OHLC Invariants (Check validity in Price space)
        if self.ohlcv_indices is not None:
            ohlc_valid = self._check_ohlc_invariants(synth_ts_norm)
            results['OHLC_Valid_Pct'] = ohlc_valid * 100
        
        # 2. Memorization (Geometric Fidelity in Norm Space)
        mem_stats = self._compute_memorization_stats(real_ts_norm, synth_ts_norm)
        results.update(mem_stats)
        
        # 3. Tail Fidelity (VaR of Body Normalized features as return proxy)
        var_diff = self._compute_var_difference(real_ts_norm, synth_ts_norm)
        results['VaR_Norm_Diff'] = var_diff
        
        # 4. Temporal Fidelity (ACF MSE in Norm Space)
        acf_mse = self._compute_acf_mse(real_ts_norm, synth_ts_norm)
        results['ACF_MSE_Norm'] = acf_mse
        
        # 5. Distribution (Wasserstein in Norm Space)
        w_dist = self._compute_wasserstein(real_ts_norm, synth_ts_norm)
        results['Wasserstein_Norm'] = w_dist
        
        # 6. Correlation Matrix Norm (Structure Check)
        corr_diff = self._compute_correlation_matrix_diff(real_ts_norm, synth_ts_norm)
        results['Corr_Norm_Diff'] = corr_diff
        
        # 7. Moment Drift (Gaussianity Check)
        moments = self._compute_moment_drift(real_ts_norm, synth_ts_norm)
        results.update(moments)
        
        # --- Structured Output ---
        print(f"  • Structural Fidelity")
        if self.ohlcv_indices is not None:
             print(f"    OHLC Valid:      {results['OHLC_Valid_Pct']:.1f}%")
        print(f"    Corr Norm Diff:  {results['Corr_Norm_Diff']:.4f}")
        print(f"    Wasserstein:     {results['Wasserstein_Norm']:.4f}")

        print(f"\n  • Stylized Facts / Moments")
        print(f"    Skew Drift:      {moments['Skew_Drift']:.4f}")
        print(f"    Kurtosis Drift:  {moments['Kurt_Drift']:.4f}")
        print(f"    VaR Diff:        {results['VaR_Norm_Diff']:.4f}")
        print(f"    ACF MSE:         {results['ACF_MSE_Norm']:.4f}")

        print(f"\n  • Memorization / Privacy")
        print(f"    NN Dist Min:     {results['NN_Dist_Min']:.4f}")
        print(f"    NN Dist Avg:     {results['NN_Dist_Avg']:.4f}")
        
        print("-" * 80)
        
        # Log to trainer
        for k, v in results.items():
            pl_module.log(f"eval/{k}", v, prog_bar=False)
    
    def _generate_samples(self, pl_module, scale=None) -> torch.Tensor:
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
            
            # DDIM-style skip (use 20 steps for speed)
            step_size = max(1, T // 20)
            timesteps = list(range(T - 1, -1, -step_size))
            total_steps = len(timesteps)
            
            for i, t in enumerate(timesteps):
                t_tensor = torch.full((self.n_samples,), t, device=device, dtype=torch.long)
                t_norm = t_tensor.float() / T
                
                # Predict noise
                predicted_noise = pl_module(x_t, t_norm, scale=scale)
                
                # Compute x_{t-1}
                alpha_t = alphas_cumprod[t]
                alpha_prev = alphas_cumprod[max(0, t - step_size)] if t > 0 else torch.tensor(1.0, device=device)
                
                # DDIM update (deterministic)
                x0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                x_t = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * predicted_noise
        
        pl_module.train()
        return x_t.cpu()
    
    def _check_ohlc_invariants(self, synth_ts_norm: np.ndarray) -> float:
        """
        Check OHLC invariants in reconstructed price space.
        Verifies High >= Open/Close/Low and Low <= Open/Close/High.
        """
        # Reconstruct OHLCV using random anchor/atr context
        synth_ohlcv = self.data_module.inverse_normalize(synth_ts_norm, sample_indices=None)
        
        # Channels: [Open, High, Low, Close, Volume]
        open_p = synth_ohlcv[..., 0]
        high_p = synth_ohlcv[..., 1]
        low_p = synth_ohlcv[..., 2]
        close_p = synth_ohlcv[..., 3]
        
        # Check geometric invariants (with tiny epsilon for float stability)
        eps = 1e-7
        h_ge_o = (high_p >= open_p - eps)
        h_ge_c = (high_p >= close_p - eps)
        l_le_o = (low_p <= open_p + eps)
        l_le_c = (low_p <= close_p + eps)
        h_ge_l = (high_p >= low_p - eps)
        
        all_valid = h_ge_o & h_ge_c & l_le_o & l_le_c & h_ge_l
        return np.mean(all_valid)
    
    def _sanitize_data(self, data: np.ndarray) -> np.ndarray:
        """Replace Infs/NaNs with finite values to prevent sklearn errors."""
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=1e9, neginf=-1e9)
        return np.clip(data, -1e9, 1e9)

    def _compute_memorization_stats(self, real_ts: np.ndarray, synth_ts: np.ndarray) -> dict:
        """Compute nearest neighbor distance statistics."""
        real_ts = self._sanitize_data(real_ts)
        synth_ts = self._sanitize_data(synth_ts)
        
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
    
    def _compute_var_difference(self, real_ts_norm: np.ndarray, synth_ts_norm: np.ndarray) -> float:
        """Compute VaR (99th percentile) difference of body_norm (price return proxy)."""
        real_ts_norm = self._sanitize_data(real_ts_norm)
        synth_ts_norm = self._sanitize_data(synth_ts_norm)
        
        # In reparameterized space, body_norm (index 1) is (Close-Open)/Anchor/ATR_pct
        # which is a very close proxy to returns.
        real_body = real_ts_norm[:, :, 1].flatten()
        synth_body = synth_ts_norm[:, :, 1].flatten()
        
        real_var = np.percentile(np.abs(real_body), 99)
        synth_var = np.percentile(np.abs(synth_body), 99)
        
        return abs(real_var - synth_var)
    
    def _compute_acf_mse(self, real_ts_norm: np.ndarray, synth_ts_norm: np.ndarray, nlags: int = 20) -> float:
        """Compute MSE between ACF curves in norm space."""
        real_ts_norm = self._sanitize_data(real_ts_norm)
        synth_ts_norm = self._sanitize_data(synth_ts_norm)
        
        n_features = real_ts_norm.shape[2]
        
        max_acf_samples = 64
        
        def avg_acf(data, nlags):
            """Average ACF across subsampled data for a single feature."""
            if len(data) > max_acf_samples:
                idx = np.random.choice(len(data), max_acf_samples, replace=False)
                data = data[idx]
            acfs = []
            for sample in data:
                try:
                    if np.var(sample) < 1e-9:
                        acfs.append(np.zeros(nlags + 1))
                        continue
                    a = acf(sample, nlags=nlags, fft=True)
                    if len(a) < nlags + 1:
                        a = np.pad(a, (0, nlags + 1 - len(a)))
                    acfs.append(a[:nlags + 1])
                except Exception:
                    acfs.append(np.zeros(nlags + 1))
            return np.nan_to_num(np.mean(acfs, axis=0)) if acfs else np.zeros(nlags + 1)
        
        total_mse = 0
        for feat_idx in range(n_features):
            real_acf = avg_acf(real_ts_norm[:, :, feat_idx], nlags)
            synth_acf = avg_acf(synth_ts_norm[:, :, feat_idx], nlags)
            total_mse += np.mean((real_acf - synth_acf) ** 2)
        
        return total_mse / n_features
    
    def _compute_wasserstein(self, real_ts: np.ndarray, synth_ts: np.ndarray) -> float:
        """Compute mean Wasserstein distance across features."""
        real_ts = self._sanitize_data(real_ts)
        synth_ts = self._sanitize_data(synth_ts)
        
        n_features = real_ts.shape[2]
        
        # Flatten time dimension
        real_flat = real_ts.reshape(-1, n_features)
        synth_flat = synth_ts.reshape(-1, n_features)
        
        w_dists = []
        for feat_idx in range(n_features):
            w_dist = wasserstein_distance(real_flat[:, feat_idx], synth_flat[:, feat_idx])
            w_dists.append(w_dist)
        
        return float(np.mean(w_dists))
    
    def _compute_correlation_matrix_diff(self, real_ts: np.ndarray, synth_ts: np.ndarray) -> float:
        """Compute Frobenius norm of difference between correlation matrices."""
        real_ts = self._sanitize_data(real_ts)
        synth_ts = self._sanitize_data(synth_ts)
        
        # Flatten pairs: (N*T, D)
        real_flat = real_ts.reshape(-1, real_ts.shape[2])
        synth_flat = synth_ts.reshape(-1, synth_ts.shape[2])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_real = np.corrcoef(real_flat, rowvar=False)
            corr_synth = np.corrcoef(synth_flat, rowvar=False)
        
        corr_real = np.nan_to_num(corr_real)
        corr_synth = np.nan_to_num(corr_synth)
        
        if not np.all(np.isfinite(corr_real)): corr_real[:] = 0
        if not np.all(np.isfinite(corr_synth)): corr_synth[:] = 0
        
        return np.linalg.norm(corr_real - corr_synth)

    def _compute_moment_drift(self, real_ts: np.ndarray, synth_ts: np.ndarray) -> dict:
        """Compute average drift in Skewness and Kurtosis."""
        real_ts = self._sanitize_data(real_ts)
        synth_ts = self._sanitize_data(synth_ts)
        
        real_flat = real_ts.reshape(-1, real_ts.shape[2])
        synth_flat = synth_ts.reshape(-1, real_ts.shape[2])
        
        real_skew = skew(real_flat, axis=0)
        synth_skew = skew(synth_flat, axis=0)
        
        real_kurt = kurtosis(real_flat, axis=0)
        synth_kurt = kurtosis(synth_flat, axis=0)
        
        skew_diff = np.nan_to_num(np.abs(real_skew - synth_skew))
        kurt_diff = np.nan_to_num(np.abs(real_kurt - synth_kurt))
        
        return {
            'Skew_Drift': float(np.mean(skew_diff)),
            'Kurt_Drift': float(np.mean(kurt_diff))
        }
