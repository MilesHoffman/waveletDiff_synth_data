"""
Inline Evaluation Callback for WaveletDiff Training.

Runs periodic zero-split inference and computes deterministic metrics
(NNAA, Fréchet Distance, EVT Tail Index) on reconstructed OHLC prices
using EMA weights to monitor synthetic data quality and detect memorization.

When a `test_data_module` is provided the callback performs two full
evaluation phases per trigger:
  Phase 1 (Train) — conditioned on training signatures, benchmarked against
                    an interleaved training reference set.
  Phase 2 (Test)  — conditioned on unseen test signatures, benchmarked against
                    an interleaved test reference set.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from scipy.stats import skew as _scipy_skew
from sklearn.neighbors import NearestNeighbors
import scipy.linalg

from evaluation.core_metrics.discriminative import discriminative_score
from evaluation.preprocessing import prepare_evaluation_data


class InlineEvaluationCallback(pl.Callback):
    def __init__(
        self,
        data_module,
        eval_every_n_epochs: int = 200,
        n_samples: int = 500,
        ohlcv_indices: dict = None,
        test_data_module=None,
    ):
        super().__init__()
        self.data_module = data_module
        self.test_data_module = test_data_module
        self.eval_every = eval_every_n_epochs
        self.n_samples = n_samples
        self.ohlcv_indices = ohlcv_indices

        # Cached interleaved indices (computed once on first eval)
        self._train_eval_idx = None
        self._train_ref_idx = None
        self._test_eval_idx = None
        self._test_ref_idx = None

    # ── Utility ────────────────────────────────────────────────────────────────

    def _find_ema_callback(self, trainer):
        for callback in trainer.callbacks:
            if callback.__class__.__name__ == 'EMACallback':
                return callback
        return None

    def _get_interleaved_indices(self, total_samples: int):
        """
        Return two perfectly interleaved, disjoint index arrays that together
        cover the full temporal extent of the dataset.

        Picks `2 * n_samples` evenly spaced points in [0, total_samples-1],
        then separates even/odd positions to create two disjoint sets that
        share the exact same temporal distribution.
        """
        n_total = min(self.n_samples * 2, total_samples)
        all_idx = np.linspace(0, total_samples - 1, n_total, dtype=int)
        return all_idx[0::2], all_idx[1::2]

    def _resolve_total_samples(self, dm) -> int:
        if getattr(dm, 'has_path_sig_conditioning', False):
            return len(dm.norm_stats['anchors'])
        return len(dm.raw_data_tensor)

    def _extract_ohlc(self, dm, raw_tensor_subset, indices):
        """Inverse-normalize a raw tensor slice and extract OHLC columns."""
        ohlcv = dm.inverse_normalize(raw_tensor_subset, sample_indices=indices)
        if self.ohlcv_indices is not None:
            cols = [self.ohlcv_indices[k] for k in ('open', 'high', 'low', 'close')]
            return ohlcv[:, :, cols]
        return ohlcv

    # ── Main Epoch Hook ────────────────────────────────────────────────────────

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.eval_every != 0:
            return

        print(f"\n> Inline Evaluation (Epoch {epoch})")

        # 1. EMA Weight Swap
        ema_callback = self._find_ema_callback(trainer)
        original_state_dict = None

        if ema_callback is not None and ema_callback.ema_model is not None:
            print("  [EMA] Swapping to EMA weights...")
            original_state_dict = {
                k: v.cpu().clone() for k, v in pl_module.state_dict().items()
            }
            pl_module.load_state_dict(ema_callback.ema_model.module.state_dict())

        # 2. Compute/cache interleaved indices for each dataset
        if self._train_eval_idx is None:
            total_train = self._resolve_total_samples(self.data_module)
            self._train_eval_idx, self._train_ref_idx = self._get_interleaved_indices(total_train)

        if self.test_data_module is not None and self._test_eval_idx is None:
            total_test = self._resolve_total_samples(self.test_data_module)
            self._test_eval_idx, self._test_ref_idx = self._get_interleaved_indices(total_test)

        # 3. Phase 1 — Training data evaluation
        print("\n  ── Phase 1: Training Distribution ──")
        self._run_eval_phase(
            pl_module=pl_module,
            source_dm=self.data_module,
            eval_idx=self._train_eval_idx,
            ref_idx=self._train_ref_idx,
            log_prefix="Train",
            label="Train",
        )

        # 4. Phase 2 — Test data evaluation (if available)
        if self.test_data_module is not None:
            print("\n  ── Phase 2: Test Distribution (Out-of-Sample) ──")
            self._run_eval_phase(
                pl_module=pl_module,
                source_dm=self.test_data_module,
                eval_idx=self._test_eval_idx,
                ref_idx=self._test_ref_idx,
                log_prefix="Test",
                label="Test",
            )

        # 5. Restore training weights
        if original_state_dict is not None:
            print("\n  [EMA] Restoring training weights...")
            restored = {k: v.to(pl_module.device) for k, v in original_state_dict.items()}
            pl_module.load_state_dict(restored)

        print("-" * 80)

    # ── Phase Runner ───────────────────────────────────────────────────────────

    def _run_eval_phase(self, pl_module, source_dm, eval_idx, ref_idx, log_prefix, label):
        """
        Execute one complete evaluation pass for a given data module.

        Generates `n_samples` sequences conditioned on `source_dm`'s path
        signatures (at `eval_idx`), then computes NNAA, Context-FID, and EVT
        metrics against the disjoint reference set at `ref_idx`.
        """
        has_sig = getattr(source_dm, 'has_path_sig_conditioning', False)

        # Build conditions from source dataset's evaluation indices
        conditions = None
        if has_sig:
            conditions = source_dm.path_sig_tensor[eval_idx].to(pl_module.device)

        # Generate synthetic samples conditioned on source signatures
        print(f"  [DDIM] Generating {label}-conditioned samples ({len(eval_idx)} samples)...")
        synth_wavelet = self._generate_samples(pl_module, conditions=conditions)
        synth_ts_norm = source_dm.convert_wavelet_to_timeseries(synth_wavelet).cpu().numpy()

        # Reconstruct physical OHLC for all three sets
        real_ts_norm_target = source_dm.raw_data_tensor[eval_idx].cpu().numpy()
        real_ts_norm_ref = source_dm.raw_data_tensor[ref_idx].cpu().numpy()

        synth_ohlc = self._extract_ohlc(source_dm, synth_ts_norm, eval_idx)
        real_ohlc_target = self._extract_ohlc(source_dm, real_ts_norm_target, eval_idx)
        real_ohlc_ref = self._extract_ohlc(source_dm, real_ts_norm_ref, ref_idx)

        results = {}

        # Metric 1: OHLC Structural Invariants
        if self.ohlcv_indices is not None:
            results['OHLC_Valid_Pct'] = self._check_ohlc_invariants(synth_ohlc) * 100.0

        # Metric 2: NNAA memorization
        results.update(self._compute_nnaa(real_ohlc_target, synth_ohlc, real_ohlc_ref))

        # Metric 3: Context-FID
        results['Synth_to_Real_CFID'] = self._compute_training_frechet_distance(real_ohlc_target, synth_ohlc)
        results['Real_to_Real_CFID'] = self._compute_training_frechet_distance(real_ohlc_target, real_ohlc_ref)

        # Metric 4: EVT Tail Index
        results.update(self._compute_evt_tail_drift(real_ohlc_target, synth_ohlc, source_dm))

        # Metric 5: Discriminative Score
        print(f"  [Discriminative] Computing Discriminative Score (2000 iterations)...")
        eval_data = prepare_evaluation_data(
            real_ohlc_target, 
            synth_ohlc,
            exclude_volume=True,
            close_col=3,
            is_reparam=False
        )
        disc_score, fake_acc, real_acc = discriminative_score(
            eval_data['real']['standardized'],
            eval_data['synth']['standardized'],
            iterations=2000,
            compile_model=True
        )
        results['Discriminative_Score'] = disc_score
        results['Discriminative_Fake_Acc'] = fake_acc
        results['Discriminative_Real_Acc'] = real_acc

        # Console output
        print(f"  • [{label}] Structural Fidelity")
        if self.ohlcv_indices is not None:
            print(f"    OHLC Valid Pct:  {results['OHLC_Valid_Pct']:.1f}%")
        print(f"    S→R Context-FID: {results['Synth_to_Real_CFID']:.4f}  [vs Real: {results['Real_to_Real_CFID']:.4f}]")
        print(f"\n  • [{label}] Memorization (Privacy) — NNAA  [Yale et al. 2019]")
        print(f"    Train Acc:       {results['NNAA_Train_Acc']:.4f}  (ideal: 0.50)")
        print(f"    Test Acc:        {results['NNAA_Test_Acc']:.4f}  (ideal: 0.50)")
        print(f"    Privacy Loss:    {results['NNAA_Privacy_Loss']:.4f}  (ideal: 0.00, high=memorizing)")
        print(f"\n  • [{label}] Fat Tail Drift (EVT)")
        print(f"    Real Tail Index: {results['Real_Tail_Index']:.4f}")
        print(f"    Synth Tail Index:{results['Synth_Tail_Index']:.4f}")
        print(f"    Tail Index Diff: {results['Tail_Index_Diff']:.4f}")

        print(f"\n  • [{label}] Discriminative Model (RNN/LSTM)")
        print(f"    Discriminative Score: {results['Discriminative_Score']:.4f}  (ideal: 0.50, low=indistinguishable)")
        print(f"    Real Accuracy:        {results['Discriminative_Real_Acc']:.4f}")
        print(f"    Fake Accuracy:        {results['Discriminative_Fake_Acc']:.4f}")

        # PL logging with prefixed keys
        for k, v in results.items():
            pl_module.log(f"eval/{log_prefix}_{k}", v, prog_bar=False)

    # ── Sample Generation ──────────────────────────────────────────────────────

    def _generate_samples(self, pl_module, conditions=None) -> torch.Tensor:
        n = conditions.shape[0] if conditions is not None else self.n_samples
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
            n_samples=n,
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

        Args:
            real_ohlc_target: [N, T, C] physical OHLC, conditioning / target set.
            synth_ohlc:       [N, T, C] generated OHLC sequences.
            real_ohlc_ref:    [N, T, C] held-out real set (disjoint from target).
        """
        real_feat  = self._to_summary_features(real_ohlc_target)
        ref_feat   = self._to_summary_features(real_ohlc_ref)
        synth_feat = self._to_summary_features(synth_ohlc)

        def _adversarial_accuracy(pool_real: np.ndarray, pool_synth: np.ndarray) -> float:
            X = np.vstack([pool_real, pool_synth])
            X = X / (X.std(axis=0) + 1e-8)  # Standardize to prevent magnitude domination
            y = np.concatenate([
                np.ones(len(pool_real), dtype=np.float32),
                np.zeros(len(pool_synth), dtype=np.float32)
            ])
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
            dists, indices = nbrs.kneighbors(X)
            nn_idx = np.where(dists[:, 0] < 1e-10, indices[:, 1], indices[:, 0])
            return float(np.mean(y[nn_idx] == y))

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

        Projecting to [mean, std, skew] per channel gives a stable 12×12
        covariance estimable from n=250 samples with room to spare.
        """
        real_flat  = self._to_summary_features(real_ohlc)
        synth_flat = self._to_summary_features(synth_ohlc)

        if real_flat.shape[0] < 2:
            return 0.0

        feat_std   = real_flat.std(axis=0) + 1e-8
        real_flat  = real_flat  / feat_std
        synth_flat = synth_flat / feat_std

        mu_r, mu_s = np.mean(real_flat, axis=0), np.mean(synth_flat, axis=0)
        sigma_r = np.cov(real_flat,  rowvar=False) + np.eye(real_flat.shape[1]) * 1e-4
        sigma_s = np.cov(synth_flat, rowvar=False) + np.eye(synth_flat.shape[1]) * 1e-4

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
        self, real_ohlc: np.ndarray, synth_ohlc: np.ndarray, dm=None
    ) -> dict:
        """
        Hill estimator for the tail index (α) of the close-price log-return distribution.

        Requires inverse normalization to have been applied — if norm_stats is None
        the data is in normalized ≈zero-mean space and log-returns are meaningless.
        Index 3 = Close is guaranteed by the [O,H,L,C] column-select in _run_eval_phase.
        """
        source_dm = dm if dm is not None else self.data_module
        zero = {'Real_Tail_Index': 0.0, 'Synth_Tail_Index': 0.0, 'Tail_Index_Diff': 0.0}
        if getattr(source_dm, 'norm_stats', None) is None:
            return zero

        eps = 1e-8
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
            return 1.0 / np.mean(np.log(tail_data) - np.log(threshold))

        real_alpha  = hill_estimator(real_ret)
        synth_alpha = hill_estimator(synth_ret)

        return {
            'Real_Tail_Index':  float(real_alpha),
            'Synth_Tail_Index': float(synth_alpha),
            'Tail_Index_Diff':  float(abs(real_alpha - synth_alpha)),
        }
