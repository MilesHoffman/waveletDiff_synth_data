"""
Inline Evaluation Callback for WaveletDiff Training.

Runs periodic zero-split inference and computes deterministic metrics
(NNAA, Fréchet Distance, EVT Tail Index) on reconstructed OHLC prices
using EMA weights to monitor synthetic data quality and detect memorization.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from scipy.stats import skew as _scipy_skew
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
            original_state_dict = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
            ema_state = ema_callback.ema_model.module.state_dict()
            pl_module.load_state_dict(ema_state)

        # 2. Extract Consistent Conditions (Uniformly Spaced) & Reference Set
        if self.eval_indices is None or self.ref_indices is None:
            if getattr(self.data_module, 'has_path_sig_conditioning', False):
                total_samples = len(self.data_module.norm_stats['anchors'])
            else:
                total_samples = len(self.data_module.raw_data_tensor)

            # Two disjoint halves: Target for conditioning/generation, Reference for benchmarking
            all_indices = np.linspace(0, total_samples - 1, self.n_samples * 2, dtype=int)
            self.eval_indices = all_indices[:self.n_samples]
            self.ref_indices = all_indices[self.n_samples:]

            if getattr(self.data_module, 'has_path_sig_conditioning', False):
                sigs = self.data_module.path_sig_tensor
                self.eval_conditions = sigs[self.eval_indices].to(pl_module.device)

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

        synth_ohlcv = self.data_module.inverse_normalize(synth_ts_norm, sample_indices=self.eval_indices)
        real_ohlcv_target = self.data_module.inverse_normalize(real_ts_norm_target, sample_indices=self.eval_indices)
        real_ohlcv_ref = self.data_module.inverse_normalize(real_ts_norm_ref, sample_indices=self.ref_indices)

        # Extract strictly OHLC columns — output order is always [Open, High, Low, Close]
        if self.ohlcv_indices is not None:
            o_idx, h_idx, l_idx, c_idx = (
                self.ohlcv_indices['open'], self.ohlcv_indices['high'],
                self.ohlcv_indices['low'],  self.ohlcv_indices['close']
            )
            synth_ohlc      = synth_ohlcv[:, :, [o_idx, h_idx, l_idx, c_idx]]
            real_ohlc_target = real_ohlcv_target[:, :, [o_idx, h_idx, l_idx, c_idx]]
            real_ohlc_ref    = real_ohlcv_ref[:,   :, [o_idx, h_idx, l_idx, c_idx]]
        else:
            synth_ohlc       = synth_ohlcv
            real_ohlc_target = real_ohlcv_target
            real_ohlc_ref    = real_ohlcv_ref

        results = {}

        # Metric 1: OHLC Valid Pct
        if self.ohlcv_indices is not None:
            results['OHLC_Valid_Pct'] = self._check_ohlc_invariants(synth_ohlc) * 100.0

        # Metric 2: NNAA — Nearest Neighbor Adversarial Accuracy (memorization / privacy)
        nnaa_stats = self._compute_nnaa(real_ohlc_target, synth_ohlc, real_ohlc_ref)
        results.update(nnaa_stats)

        # Metric 3: Context-FID (both paths use 12-dim summary features)
        results['Synth_to_Real_CFID'] = self._compute_training_frechet_distance(real_ohlc_target, synth_ohlc)
        results['Real_to_Real_CFID']  = self._compute_training_frechet_distance(real_ohlc_target, real_ohlc_ref)

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
        print(f"    S→R Context-FID: {results['Synth_to_Real_CFID']:.4f}  [vs Real: {results['Real_to_Real_CFID']:.4f}]")
        print(f"\n  • Memorization (Privacy) — NNAA  [Yale et al. 2019]")
        print(f"    Train Acc:       {results['NNAA_Train_Acc']:.4f}  (ideal: 0.50)")
        print(f"    Test Acc:        {results['NNAA_Test_Acc']:.4f}  (ideal: 0.50)")
        print(f"    Privacy Loss:    {results['NNAA_Privacy_Loss']:.4f}  (ideal: 0.00, high=memorizing)")
        print(f"\n  • Fat Tail Drift (EVT)")
        print(f"    Real Tail Index: {results['Real_Tail_Index']:.4f}")
        print(f"    Synth Tail Index:{results['Synth_Tail_Index']:.4f}")
        print(f"    Tail Index Diff: {results['Tail_Index_Diff']:.4f}")
        print("-" * 80)

        for k, v in results.items():
            pl_module.log(f"eval/{k}", v, prog_bar=False)

    # ── Sample Generation ──────────────────────────────────────────────────────

    def _generate_samples(self, pl_module, conditions=None) -> torch.Tensor:
        pl_module.eval()
        from .diffusion_process import DiffusionTrainer

        pl_module.input_dim = self.data_module.get_input_dim()
        pl_module.num_features = self.data_module.get_wavelet_info()['n_features']

        config_obj = getattr(pl_module, 'config', getattr(pl_module, 'hparams', {}))
        sampling_method = config_obj.get('sampling', {}).get(
            'eval_method', config_obj.get('sampling', {}).get('method', 'ddpm')
        )
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

    # ── Shared Feature Projection ──────────────────────────────────────────────

    def _sanitize(self, data: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=1e9, neginf=-1e9)
        return np.clip(data, -1e9, 1e9)

    def _to_summary_features(self, ohlc: np.ndarray) -> np.ndarray:
        """
        Project [N, T, C] OHLC array to [N, C*3] distributional summary.

        Per channel computes [mean, std, skew] across the time axis, producing
        a compact 12-dim descriptor for C=4 that is stable for covariance
        estimation and avoids the curse of dimensionality in the raw T*C flat space.
        """
        ohlc = self._sanitize(ohlc)
        parts = []
        for c in range(ohlc.shape[2]):
            ch = ohlc[:, :, c]
            parts.append(ch.mean(axis=1, keepdims=True))
            parts.append(ch.std(axis=1, keepdims=True))
            
            # Scipy skew returns NaN for constant data (variance=0); replace with 0.0
            skew_vals = _scipy_skew(ch, axis=1)
            skew_vals = np.nan_to_num(skew_vals, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(skew_vals.reshape(-1, 1))
        return np.hstack(parts).astype(np.float64)

    # ── Structural Invariants ──────────────────────────────────────────────────

    def _check_ohlc_invariants(self, synth_ohlc: np.ndarray) -> float:
        """Fraction of candles satisfying High≥Open,Close and Low≤Open,Close."""
        open_p, high_p, low_p, close_p = (
            synth_ohlc[..., 0], synth_ohlc[..., 1],
            synth_ohlc[..., 2], synth_ohlc[..., 3]
        )
        eps = 1e-7
        valid = (
            (high_p >= open_p  - eps) & (high_p >= close_p - eps) &
            (low_p  <= open_p  + eps) & (low_p  <= close_p + eps) &
            (high_p >= low_p   - eps)
        )
        return float(np.mean(valid))

    # ── Memorization: NNAA ────────────────────────────────────────────────────

    def _compute_nnaa(
        self,
        real_ohlc_target: np.ndarray,
        synth_ohlc: np.ndarray,
        real_ohlc_ref: np.ndarray
    ) -> dict:
        """
        Nearest Neighbor Adversarial Accuracy (Yale et al., AIDR 2019).

        A 1-NN adversary attempts to classify sequences as real or synthetic.
        Accuracy near 0.5 means model output is indistinguishable from real data.
        Privacy Loss = |AA_train - AA_test|; high values indicate memorization.

        Both paths operate on 12-dim per-channel [mean, std, skew] descriptors
        to remain discriminative without suffering from high-dim distance collapse.

        Args:
            real_ohlc_target: [N, T, C] physical OHLC, conditioning / target set.
            synth_ohlc:       [N, T, C] generated OHLC sequences.
            real_ohlc_ref:    [N, T, C] held-out real set (disjoint from target).

        Returns:
            NNAA_Train_Acc:    adversarial accuracy against training real pool.
            NNAA_Test_Acc:     adversarial accuracy against held-out real pool.
            NNAA_Privacy_Loss: |train_acc - test_acc|; high = memorization risk.
        """
        real_feat  = self._to_summary_features(real_ohlc_target)
        ref_feat   = self._to_summary_features(real_ohlc_ref)
        synth_feat = self._to_summary_features(synth_ohlc)

        def _adversarial_accuracy(
            pool_real: np.ndarray, pool_synth: np.ndarray
        ) -> float:
            """
            Build a labeled pool [real=1, synth=0] and run 1-NN on itself.
            We exclude the query point's exact self-match by fitting on the full
            pool and using leave-one-out semantics — acceptable here because
            real and synth have disjoint origins and the pool is the query set.
            """
            X = np.vstack([pool_real, pool_synth])
            y = np.concatenate([
                np.ones(len(pool_real), dtype=np.float32),
                np.zeros(len(pool_synth), dtype=np.float32)
            ])
            # k=2: skip self (d=0) by using second neighbor when distance is zero
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
            dists, indices = nbrs.kneighbors(X)
            # Use k=1 neighbor; if it's the self-match (d≈0) fall back to k=2
            nn_idx = np.where(dists[:, 0] < 1e-10, indices[:, 1], indices[:, 0])
            y_pred = y[nn_idx]
            return float(np.mean(y_pred == y))

        aa_train = _adversarial_accuracy(real_feat,  synth_feat)
        aa_test  = _adversarial_accuracy(ref_feat,   synth_feat)

        return {
            'NNAA_Train_Acc':    float(aa_train),
            'NNAA_Test_Acc':     float(aa_test),
            'NNAA_Privacy_Loss': float(abs(aa_train - aa_test)),
        }

    # ── Context-FID ───────────────────────────────────────────────────────────

    def _compute_training_frechet_distance(
        self, real_ohlc: np.ndarray, synth_ohlc: np.ndarray
    ) -> float:
        """
        Fréchet Distance on 12-dim per-channel distributional summaries.

        Using raw T*C flat vectors produces a spuriously high Real-to-Real
        baseline (~433) due to finite-sample covariance noise accumulating over
        all dims. Projecting to [mean, std, skew] per channel gives a stable
        12*12 covariance estimable from n=250 samples with room to spare.

        Both synth and real paths pass through _to_summary_features so the
        metric space is identical for both comparisons.
        """
        real_flat  = self._to_summary_features(real_ohlc)
        synth_flat = self._to_summary_features(synth_ohlc)

        if real_flat.shape[0] < 2:
            return 0.0

        feat_std   = real_flat.std(axis=0) + 1e-8
        real_flat  = real_flat  / feat_std
        synth_flat = synth_flat / feat_std

        mu_r = np.mean(real_flat,  axis=0)
        mu_s = np.mean(synth_flat, axis=0)

        sigma_r = np.cov(real_flat,  rowvar=False)
        sigma_s = np.cov(synth_flat, rowvar=False)

        # Tikhonov regularization
        eps = 1e-4
        sigma_r += np.eye(sigma_r.shape[0]) * eps
        sigma_s += np.eye(sigma_s.shape[0]) * eps

        diff = mu_r - mu_s
        covmean, _ = scipy.linalg.sqrtm(sigma_r.dot(sigma_s), disp=False)

        if not np.isfinite(covmean).all():
            covmean = scipy.linalg.sqrtm(
                (sigma_r + np.eye(sigma_r.shape[0]) * 1e-3).dot(
                    sigma_s + np.eye(sigma_s.shape[0]) * 1e-3
                )
            )

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_r) + np.trace(sigma_s) - 2 * np.trace(covmean)
        return float(max(0.0, fid))

    # ── EVT Fat-Tail ──────────────────────────────────────────────────────────

    def _compute_evt_tail_drift(
        self, real_ohlc: np.ndarray, synth_ohlc: np.ndarray
    ) -> dict:
        """
        Hill estimator for the tail index (α) of the close-price log-return distribution.

        Requires inverse normalization to have been applied — if norm_stats is None
        the data is in normalized ≈zero-mean space and log-returns are meaningless.
        Index 3 = Close is guaranteed by the [O,H,L,C] column-select in on_train_epoch_end.
        """
        zero = {'Real_Tail_Index': 0.0, 'Synth_Tail_Index': 0.0, 'Tail_Index_Diff': 0.0}
        if getattr(self.data_module, 'norm_stats', None) is None:
            return zero

        eps = 1e-8
        # Ensure prices are non-negative before logarithm to prevent NaNs
        real_c  = np.clip(real_ohlc[..., 3], 0.0, None)
        synth_c = np.clip(synth_ohlc[..., 3], 0.0, None)

        real_ret  = np.log((real_c[:,  1:] + eps) / (real_c[:,  :-1] + eps)).flatten()
        synth_ret = np.log((synth_c[:, 1:] + eps) / (synth_c[:, :-1] + eps)).flatten()

        def hill_estimator(data, threshold_pct=95):
            data = np.abs(data)
            data = data[data > 0]
            if len(data) < 10:
                return 0.0
            threshold = np.percentile(data, threshold_pct)
            tail_data = data[data > threshold]
            if len(tail_data) < 2:
                return 0.0
            log_data   = np.log(tail_data)
            log_thresh = np.log(threshold)
            return 1.0 / np.mean(log_data - log_thresh)

        real_alpha  = hill_estimator(real_ret)
        synth_alpha = hill_estimator(synth_ret)

        return {
            'Real_Tail_Index':  float(real_alpha),
            'Synth_Tail_Index': float(synth_alpha),
            'Tail_Index_Diff':  float(abs(real_alpha - synth_alpha))
        }
