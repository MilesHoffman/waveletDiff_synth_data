"""
Inline Evaluation Callback for WaveletDiff Training.

Runs periodic zero-split inference and computes deterministic metrics 
(DCR, NNDR, Fréchet Distance, EVT Tail Index) on reconstructed OHLC prices
using EMA weights to monitor synthetic data quality and detect memorization.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.neighbors import NearestNeighbors
import scipy.linalg

class InlineEvaluationCallback(pl.Callback):
    def __init__(
        self,
        data_module,
        eval_every_n_epochs: int = 200,
        n_samples: int = 500,
        ohlcv_indices: dict = None
    ):
        super().__init__()
        self.data_module = data_module
        self.eval_every = eval_every_n_epochs
        self.n_samples = n_samples
        self.ohlcv_indices = ohlcv_indices
        
        # Caches for uniform extraction conditions
        self.eval_indices = None
        self.ref_indices = None
        self.eval_conditions = None
        
    def _find_ema_callback(self, trainer):
        for callback in trainer.callbacks:
            if callback.__class__.__name__ == 'EMACallback':
                return callback
        return None

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.eval_every != 0:
            return
            
        print(f"\n> Zero-Split Inline Evaluation (Epoch {epoch})")
        
        # 1. EMA Weight Swap
        ema_callback = self._find_ema_callback(trainer)
        original_state_dict = None
        
        if ema_callback is not None and ema_callback.ema_model is not None:
            print("  [EMA] Temporarily swapping to EMA weights for evaluation...")
            # We must be careful to copy since we'll restore
            original_state_dict = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
            # Extract underlying module state
            ema_state = ema_callback.ema_model.module.state_dict()
            pl_module.load_state_dict(ema_state)
            
        # 2. Extract Consistent Conditions (Uniformly Spaced) & Reference Set
        if self.eval_indices is None or self.ref_indices is None:
            if getattr(self.data_module, 'has_conditioning', False):
                total_samples = len(self.data_module.norm_stats['anchors'])
            else:
                total_samples = len(self.data_module.raw_data_tensor)
                
            # Need 2 * n_samples to create orthogonal Target and Reference sets
            # Target (0 to n_samples-1) for conditioning/generation, Reference (n_samples to 2*n_samples-1) for benchmarking
            all_indices = np.linspace(0, total_samples - 1, self.n_samples * 2, dtype=int)
            self.eval_indices = all_indices[:self.n_samples]
            self.ref_indices = all_indices[self.n_samples:]
            
            if getattr(self.data_module, 'has_path_sig_conditioning', False):
                sigs = self.data_module.path_sig_tensor
                self.eval_conditions = sigs[self.eval_indices].to(pl_module.device)

        # Ensure conditions are on the correct device if already cached
        conditions = None
        if self.eval_conditions is not None:
            conditions = self.eval_conditions.to(pl_module.device)

        # 3. Generate DDIM-50 samples
        print("  [DDIM] Generating synthetic samples (50 steps) uniformly distributed...")
        synth_wavelet = self._generate_samples(pl_module, conditions=conditions)
        synth_ts_norm = self.data_module.convert_wavelet_to_timeseries(synth_wavelet).cpu().numpy()
        
        # 4. Reconstruct OHLC Space
        real_ts_norm_target = self.data_module.raw_data_tensor[self.eval_indices].cpu().numpy()
        real_ts_norm_ref = self.data_module.raw_data_tensor[self.ref_indices].cpu().numpy()
        
        # Need to reconstruct to physical OHLCV to compute physical structure distances
        synth_ohlcv = self.data_module.inverse_normalize(synth_ts_norm, sample_indices=self.eval_indices)
        real_ohlcv_target = self.data_module.inverse_normalize(real_ts_norm_target, sample_indices=self.eval_indices)
        real_ohlcv_ref = self.data_module.inverse_normalize(real_ts_norm_ref, sample_indices=self.ref_indices)
        
        # Extract strictly OHLC columns
        if self.ohlcv_indices is not None:
            o_idx, h_idx, l_idx, c_idx = self.ohlcv_indices['open'], self.ohlcv_indices['high'], self.ohlcv_indices['low'], self.ohlcv_indices['close']
            synth_ohlc = synth_ohlcv[:, :, [o_idx, h_idx, l_idx, c_idx]]
            real_ohlc_target = real_ohlcv_target[:, :, [o_idx, h_idx, l_idx, c_idx]]
            real_ohlc_ref = real_ohlcv_ref[:, :, [o_idx, h_idx, l_idx, c_idx]]
        else:
            synth_ohlc = synth_ohlcv
            real_ohlc_target = real_ohlcv_target
            real_ohlc_ref = real_ohlcv_ref
            
        results = {}
        
        # Metric 1: OHLC Valid Pct
        if self.ohlcv_indices is not None:
            results['OHLC_Valid_Pct'] = self._check_ohlc_invariants(synth_ohlc) * 100.0

        # Metric 2: DCR & NNDR (Benchmarked)
        mem_stats = self._compute_ohlc_memorization_stats(real_ohlc_target, synth_ohlc, real_ohlc_ref)
        results.update(mem_stats)
        
        # Metric 3: Context-FID
        results['Context_FID'] = self._compute_training_frechet_distance(real_ohlc_target, synth_ohlc)
        
        # Metric 4: EVT Tail Index (Hill)
        evt_stats = self._compute_evt_tail_drift(real_ohlc_target, synth_ohlc)
        results.update(evt_stats)
        
        # Restore Weights
        if original_state_dict is not None:
            print("  [EMA] Restoring original training weights...")
            curr_device = pl_module.device
            restored_state = {k: v.to(curr_device) for k, v in original_state_dict.items()}
            pl_module.load_state_dict(restored_state)
            
        # Logging
        print(f"  • Structural Fidelity")
        if self.ohlcv_indices is not None:
            print(f"    OHLC Valid Pct:  {results['OHLC_Valid_Pct']:.1f}%")
        print(f"    Context-FID:     {results['Context_FID']:.4f}")
        print(f"\n  • Memorization (Privacy) vs Baseline")
        print(f"    S→R DCR (5th):   {results['Synth_to_Real_DCR']:.4f}  [vs Real: {results['Real_to_Real_DCR']:.4f}]")
        print(f"    S→R NNDR (Avg):  {results['Synth_to_Real_NNDR']:.4f}  [vs Real: {results['Real_to_Real_NNDR']:.4f}]")
        print(f"\n  • Fat Tail Drift (EVT)")
        print(f"    Real Tail Index: {results['Real_Tail_Index']:.4f}")
        print(f"    Synth Tail Index:{results['Synth_Tail_Index']:.4f}")
        print(f"    Tail Index Diff: {results['Tail_Index_Diff']:.4f}")
        print("-" * 80)
        
        for k, v in results.items():
            pl_module.log(f"eval/{k}", v, prog_bar=False)

    def _generate_samples(self, pl_module, conditions=None) -> torch.Tensor:
        pl_module.eval()
        from .diffusion_process import DiffusionTrainer
        
        pl_module.input_dim = self.data_module.get_input_dim()
        pl_module.num_features = self.data_module.get_wavelet_info()['n_features']
        
        config_obj = getattr(pl_module, 'config', getattr(pl_module, 'hparams', {}))
        sampling_method = config_obj.get('sampling', {}).get('eval_method', config_obj.get('sampling', {}).get('method', 'ddpm'))
        use_ddim = (sampling_method == 'ddim')
        
        original_ddim_steps = getattr(pl_module, 'ddim_steps', None)
        if use_ddim:
            pl_module.ddim_steps = 50
        
        trainer_util = DiffusionTrainer(pl_module)
        
        x_t = trainer_util.generate_samples(
            n_samples=self.n_samples, 
            use_ddim=use_ddim, 
            sampling_method=sampling_method,
            conditions=conditions, 
            show_progress=False
        )
        
        pl_module.ddim_steps = original_ddim_steps
        pl_module.train()
        return x_t.cpu()

    def _sanitize(self, data: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=1e9, neginf=-1e9)
        return np.clip(data, -1e9, 1e9)

    def _check_ohlc_invariants(self, synth_ohlc: np.ndarray) -> float:
        # Assumes input is [N, T, 4] with Open, High, Low, Close
        open_p, high_p, low_p, close_p = synth_ohlc[..., 0], synth_ohlc[..., 1], synth_ohlc[..., 2], synth_ohlc[..., 3]
        eps = 1e-7
        valid = (high_p >= open_p - eps) & (high_p >= close_p - eps) & (low_p <= open_p + eps) & (low_p <= close_p + eps) & (high_p >= low_p - eps)
        return float(np.mean(valid))

    def _compute_ohlc_memorization_stats(self, real_ohlc_target: np.ndarray, synth_ohlc: np.ndarray, real_ohlc_ref: np.ndarray) -> dict:
        real_flat_target = self._sanitize(real_ohlc_target).reshape(real_ohlc_target.shape[0], -1)
        real_flat_ref = self._sanitize(real_ohlc_ref).reshape(real_ohlc_ref.shape[0], -1)
        synth_flat = self._sanitize(synth_ohlc).reshape(synth_ohlc.shape[0], -1)
        
        # Fit NN model on the Reference training set pool
        # Need k=3 for the Real-to-Real benchmark to ensure we can skip self-matches
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(real_flat_ref)
        
        # 1. Synthetic Samples: Raw math (Do NOT filter 0, we WANT to see memorization!)
        synth_distances, _ = nbrs.kneighbors(synth_flat)
        s_d1 = synth_distances[:, 0]
        s_d2 = synth_distances[:, 1]
        synth_dcr = s_d1
        synth_nndr = s_d1 / (s_d2 + 1e-8)
        
        # 2. Real Target Samples: Filter out self-matches to get a true diversity baseline
        real_distances, _ = nbrs.kneighbors(real_flat_target)
        r_valid_d1 = []
        r_valid_d2 = []
        
        for i in range(len(real_distances)):
            row = real_distances[i]
            # Filter out points that are too close (distance ~ 0) to avoid self-referencing
            non_self = row[row > 1e-6]
            if len(non_self) < 2:
                # Fallback to standard neighbors if the sample is extremely unique or data is very sparse
                r_valid_d1.append(row[0])
                r_valid_d2.append(row[1])
            else:
                r_valid_d1.append(non_self[0])
                r_valid_d2.append(non_self[1])
        
        r_valid_d1 = np.array(r_valid_d1)
        r_valid_d2 = np.array(r_valid_d2)
        real_dcr = r_valid_d1
        real_nndr = r_valid_d1 / (r_valid_d2 + 1e-8)
        
        return {
            'Synth_to_Real_DCR': float(np.percentile(synth_dcr, 5)),
            'Real_to_Real_DCR': float(np.percentile(real_dcr, 5)),
            'Synth_to_Real_NNDR': float(np.mean(synth_nndr)),
            'Real_to_Real_NNDR': float(np.mean(real_nndr))
        }

    def _compute_training_frechet_distance(self, real_ohlc: np.ndarray, synth_ohlc: np.ndarray) -> float:
        real_flat = self._sanitize(real_ohlc).reshape(real_ohlc.shape[0], -1)
        synth_flat = self._sanitize(synth_ohlc).reshape(synth_ohlc.shape[0], -1)
        
        mu_r = np.mean(real_flat, axis=0)
        mu_s = np.mean(synth_flat, axis=0)
        
        if real_flat.shape[0] < 2:
            return 0.0
            
        sigma_r = np.cov(real_flat, rowvar=False)
        sigma_s = np.cov(synth_flat, rowvar=False)
        
        diff = mu_r - mu_s
        
        covmean, _ = scipy.linalg.sqrtm(sigma_r.dot(sigma_s), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_r.shape[0]) * 1e-6
            covmean = scipy.linalg.sqrtm((sigma_r + offset).dot(sigma_s + offset))
            
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma_r) + np.trace(sigma_s) - 2 * tr_covmean
        return float(max(0.0, fid))

    def _compute_evt_tail_drift(self, real_ohlc: np.ndarray, synth_ohlc: np.ndarray) -> dict:
        # Assuming OHLC arrays, close is at index 3
        real_c = real_ohlc[..., 3]
        synth_c = synth_ohlc[..., 3]
        
        eps = 1e-8
        real_ret = np.log((real_c[:, 1:] + eps) / (real_c[:, :-1] + eps)).flatten()
        synth_ret = np.log((synth_c[:, 1:] + eps) / (synth_c[:, :-1] + eps)).flatten()
        
        def hill_estimator(data, threshold_pct=95):
            data = np.abs(data)
            data = data[data > 0]
            if len(data) < 10: return 0.0
            threshold = np.percentile(data, threshold_pct)
            tail_data = data[data > threshold]
            if len(tail_data) < 2: return 0.0
            
            log_data = np.log(tail_data)
            log_thresh = np.log(threshold)
            return 1.0 / np.mean(log_data - log_thresh)
            
        real_alpha = hill_estimator(real_ret)
        synth_alpha = hill_estimator(synth_ret)
        
        return {
            'Real_Tail_Index': float(real_alpha),
            'Synth_Tail_Index': float(synth_alpha),
            'Tail_Index_Diff': float(abs(real_alpha - synth_alpha))
        }
