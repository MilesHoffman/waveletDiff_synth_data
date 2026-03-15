"""
Wavelet Time Series Data Module for PyTorch Lightning.

Handles wavelet decomposition/reconstruction and OHLCV inverse normalization
for the SOTA 22-feature pipeline.
"""

import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import pywt

from .loaders import (
    load_ett_data, load_fmri_data, load_exchange_rate_data,
    load_stocks_data, load_eeg_data
)


class WaveletTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, config=None, data_tensor: torch.Tensor = None, **kwargs):
        """
        WaveletTimeSeriesDataModule for loading time series datasets
        and converting to wavelet coefficients.

        Args:
            config: Configuration dict containing all parameters
            data_tensor: Pre-loaded data tensor (if None, loads from config)
        """
        super().__init__()

        if config is None:
            raise ValueError("WaveletTimeSeriesDataModule now requires 'config' to be provided.")

        self.dataset_name = config['dataset']['name']
        self.seq_len = config['dataset']['seq_len']
        self.batch_size = config['training']['batch_size']
        self.data_dir = config['data']['data_dir']
        self.wavelet_type = config['wavelet']['type']
        self.num_levels = config['wavelet']['levels']
        self.normalize_data = config['data']['normalize_data']
        self.mode = kwargs.get('mode', 'symmetric')

        # Path signature config
        self.past_days = config.get('conditioning', {}).get('past_days', 200)

        # Load raw time series data
        if data_tensor is not None:
            self.raw_data_tensor, self.norm_stats = data_tensor, None
        elif self.dataset_name is not None:
            self.raw_data_tensor, self.norm_stats = self._load_dataset(
                self.dataset_name, seq_len=self.seq_len, normalize_data=self.normalize_data)
        else:
            raise ValueError("Either dataset_name or data_tensor must be provided")

        print("Raw Data Tensor Shape:", self.raw_data_tensor.shape)

        # Convert to wavelet coefficients
        self.data_tensor, self.wavelet_info = self._convert_to_wavelet_coefficients()

        # Path signature conditioning
        self.has_path_sig_conditioning = False
        self.path_sig_tensor = None
        self.path_sig_dim = 0

        if (self.norm_stats is not None
                and self.norm_stats.get('path_signatures') is not None):
            sigs = self.norm_stats['path_signatures']
            self.path_sig_tensor = torch.FloatTensor(sigs)
            self.path_sig_dim = sigs.shape[1]
            self.has_path_sig_conditioning = True
            print(f"Path signature conditioning enabled: dim={self.path_sig_dim}, "
                  f"past_days={self.norm_stats.get('past_days', 'N/A')}")

        # Move dataset to GPU RAM if available
        self.data_on_gpu = torch.cuda.is_available()
        if self.data_on_gpu:
            self.data_tensor = self.data_tensor.cuda()
            if self.path_sig_tensor is not None:
                self.path_sig_tensor = self.path_sig_tensor.cuda()
            print("Dataset moved to GPU RAM for faster training")

        # Create dataset with conditioning if available
        if self.has_path_sig_conditioning:
            self.dataset = TensorDataset(self.data_tensor, self.path_sig_tensor)
        else:
            self.dataset = TensorDataset(self.data_tensor)

        print(f"Converted {self.raw_data_tensor.shape} time series to {self.data_tensor.shape} wavelet coefficients")
        print(f"Wavelet: {self.wavelet_type}, Levels: {self.wavelet_info['levels']}")

    def _load_dataset(self, dataset_name: str, seq_len: int, normalize_data: bool = True) -> torch.Tensor:
        """Load dataset based on the dataset name."""
        dataset_name = dataset_name.lower()

        if dataset_name.startswith("ett"):
            raw_data, norm_stats = load_ett_data(dataset_name, self.data_dir, seq_len=seq_len, normalize_data=normalize_data)
        elif dataset_name == "fmri":
            raw_data, norm_stats = load_fmri_data(self.data_dir, seq_len=seq_len, normalize_data=normalize_data)
        elif dataset_name == "exchange_rate":
            raw_data, norm_stats = load_exchange_rate_data(self.data_dir, seq_len=seq_len, normalize_data=normalize_data)
        elif dataset_name == "stocks":
            raw_data, norm_stats = load_stocks_data(
                self.data_dir, seq_len=seq_len,
                normalize_data=normalize_data,
                past_days=self.past_days
            )
        elif dataset_name == "eeg":
            raw_data, norm_stats = load_eeg_data(self.data_dir, seq_len=seq_len, normalize_data=normalize_data)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return raw_data, norm_stats

    def _convert_to_wavelet_coefficients(self) -> tuple[torch.Tensor, dict]:
        """
        Convert time series data to wavelet coefficients.

        Returns:
            wavelet_tensor: Shape [n_samples, n_level_dim, n_features]
            wavelet_info: Dictionary with reconstruction information
        """
        raw_data = self.raw_data_tensor.numpy()
        n_samples, seq_len, n_features = raw_data.shape

        if self.wavelet_type == "auto":
            if seq_len <= 32:
                self.wavelet_type = 'db2'
            elif seq_len <= 64:
                self.wavelet_type = 'db4'
            elif seq_len <= 128:
                self.wavelet_type = 'db6'
            else:
                self.wavelet_type = 'db8'

        if self.num_levels == "auto":
            self.num_levels = int(np.clip(pywt.dwt_max_level(seq_len, self.wavelet_type), 3, 7))

        print(f"Converting to wavelet coefficients with {self.num_levels} levels...")

        sample_signal = raw_data[0, :, 0]
        sample_coeffs = pywt.wavedec(sample_signal, self.wavelet_type, level=self.num_levels, mode=self.mode)
        coeffs_shapes = [c.shape for c in sample_coeffs]
        level_dims = [np.prod(shape) for shape in coeffs_shapes]
        total_coeffs_per_feature = sum(level_dims)

        print(f"Coefficient shapes per level: {coeffs_shapes}")
        print(f"Level dimensions: {level_dims}")
        print(f"Total coefficients per feature: {total_coeffs_per_feature}")

        wavelet_coeffs = np.zeros((n_samples, total_coeffs_per_feature, n_features))

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                signal = raw_data[sample_idx, :, feature_idx]
                coeffs = pywt.wavedec(signal, self.wavelet_type, level=self.num_levels, mode=self.mode)
                coeffs_flat = np.concatenate([c.flatten() for c in coeffs])
                wavelet_coeffs[sample_idx, :, feature_idx] = coeffs_flat

        wavelet_tensor = torch.FloatTensor(wavelet_coeffs)
        level_start_indices = [0] + list(np.cumsum(level_dims[:-1]))

        wavelet_info = {
            'levels': self.num_levels,
            'coeffs_shapes': coeffs_shapes,
            'level_dims': level_dims,
            'level_start_indices': level_start_indices,
            'n_features': n_features,
            'original_shape': (n_samples, seq_len, n_features),
            'wavelet_shape': wavelet_tensor.shape,
            'wavelet_type': self.wavelet_type,
            'mode': self.mode,
            'total_coeffs_per_feature': total_coeffs_per_feature
        }

        return wavelet_tensor, wavelet_info

    def convert_wavelet_to_timeseries(self, wavelet_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Convert wavelet coefficients back to time series.

        Args:
            wavelet_coeffs: Shape [n_samples, n_level_dim, n_features]

        Returns:
            reconstructed_ts: Shape [n_samples, seq_len, n_features]
        """
        if isinstance(wavelet_coeffs, torch.Tensor):
            wavelet_coeffs = wavelet_coeffs.detach().cpu().numpy()

        n_samples, n_level_dim, n_features = wavelet_coeffs.shape
        coeffs_shapes = self.wavelet_info['coeffs_shapes']
        level_dims = self.wavelet_info['level_dims']
        level_start_indices = self.wavelet_info['level_start_indices']
        original_seq_len = self.wavelet_info['original_shape'][1]

        expected_n_features = self.wavelet_info['n_features']
        expected_n_level_dim = self.wavelet_info['total_coeffs_per_feature']

        if n_features != expected_n_features:
            raise ValueError(f"Feature dimension mismatch: expected {expected_n_features}, got {n_features}")
        if n_level_dim != expected_n_level_dim:
            raise ValueError(f"Level dimension mismatch: expected {expected_n_level_dim}, got {n_level_dim}")

        reconstructed_signals = []

        for sample_idx in range(n_samples):
            sample_features = []

            for feature_idx in range(n_features):
                coeffs_flat = wavelet_coeffs[sample_idx, :, feature_idx]
                coeffs = []
                for level_idx, (shape, dim, start_idx) in enumerate(zip(coeffs_shapes, level_dims, level_start_indices)):
                    end_idx = start_idx + dim
                    coeff = coeffs_flat[start_idx:end_idx].reshape(shape)
                    coeffs.append(coeff)

                reconstructed = pywt.waverec(coeffs, self.wavelet_type, mode=self.mode)

                if len(reconstructed) > original_seq_len:
                    reconstructed = reconstructed[:original_seq_len]
                elif len(reconstructed) < original_seq_len:
                    pad_length = original_seq_len - len(reconstructed)
                    reconstructed = np.pad(reconstructed, (0, pad_length), mode='constant', constant_values=0)

                sample_features.append(reconstructed)

            sample_reconstructed = np.stack(sample_features, axis=1)
            reconstructed_signals.append(sample_reconstructed)

        return torch.FloatTensor(np.stack(reconstructed_signals))

    def get_input_dim(self) -> int:
        """Get the input dimension for the model (number of wavelet coefficients)."""
        return self.data_tensor.shape[1]

    def get_wavelet_info(self) -> dict:
        """Get wavelet transformation information."""
        return self.wavelet_info

    def inverse_normalize(self, data: np.ndarray, sample_indices: np.ndarray = None,
                          fixed_anchor: float = None) -> np.ndarray:
        """
        Inverse normalization to convert generated samples back to original scale.

        For reparameterized OHLC data, reconstructs O, H, L, C, V from
        the SOTA 22-feature normalized representation.

        Args:
            data: Normalized data of shape (n_samples, seq_len, n_features)
            sample_indices: Optional indices to select specific anchor values.
            fixed_anchor: Optional fixed price anchor (e.g. 100.0).

        Returns:
            Denormalized data in original scale: (n_samples, seq_len, 5) for OHLCV
        """
        if self.norm_stats is None:
            return data

        data = data.copy()

        if self.norm_stats.get('reparameterized', False):
            return self._inverse_reparameterize_ohlc(data, sample_indices, fixed_anchor=fixed_anchor)

        mean = self.norm_stats['mean']
        std = self.norm_stats['std']
        data = data * std + mean

        if self.norm_stats.get('volume_log_transformed', False):
            data[..., 4] = np.maximum(0, np.expm1(data[..., 4]))

        return data

    def _inverse_reparameterize_ohlc(self, data: np.ndarray,
                                      sample_indices: np.ndarray = None,
                                      fixed_anchor: float = None) -> np.ndarray:
        """
        Inverse reparameterization for SOTA 22-feature OHLC data.

        Reconstructs OHLCV from Log-Return based features using
        Robust Scaling inverse and sequential price chaining.

        Feature Index Map:
            [0] overnight_gap (Robust Scaled log-return)
            [1] intraday_return (Robust Scaled log-return)
            [3] normalized_range (Robust Scaled ratio)
            [4] wick_high_ratio [0,1]
            [5] wick_low_ratio [0,1]
            [21] vol_log_dev (Log-Deviation from Rolling Median)
        """
        n_samples = data.shape[0]
        seq_len = data.shape[1]

        # ── Resolve Price Anchors ──
        if fixed_anchor is not None:
            anchors = np.full((n_samples,), fixed_anchor)
        elif sample_indices is not None:
            anchors = self.norm_stats['anchors'][sample_indices]
        else:
            all_anchors = self.norm_stats['anchors']
            indices = np.random.choice(len(all_anchors), size=n_samples, replace=True)
            anchors = all_anchors[indices]

        # ── Resolve Volume Anchors ──
        if sample_indices is not None:
            vol_medians = self.norm_stats['vol_medians'][sample_indices]
            vol_iqrs = self.norm_stats['vol_iqrs'][sample_indices]
        else:
            _indices = indices if sample_indices is None and fixed_anchor is not None else np.random.choice(
                len(self.norm_stats['vol_medians']), size=n_samples, replace=True)
            vol_medians = self.norm_stats['vol_medians'][_indices]
            vol_iqrs = self.norm_stats['vol_iqrs'][_indices]

        # ── Retrieve Robust Scaling Stats for Inverse ──
        rs = self.norm_stats['robust_scales']
        gap_med, gap_iqr = rs['overnight_gap']['median'], rs['overnight_gap']['iqr']
        idr_med, idr_iqr = rs['intraday_return']['median'], rs['intraday_return']['iqr']
        nr_med, nr_iqr = rs['normalized_range']['median'], rs['normalized_range']['iqr']

        # ── Extract & Unscale Structural Features ──
        gap_scaled = data[..., 0]
        idr_scaled = data[..., 1]
        range_scaled = data[..., 3]
        wick_high_ratio = np.clip(data[..., 4], 0.0, 1.0)
        wick_low_ratio = np.clip(data[..., 5], 0.0, 1.0)
        vol_log_dev = np.clip(data[..., 21], -10.0, 10.0)

        # Invert Robust Scaling to raw Log-Returns
        gap_log_return = (gap_scaled * gap_iqr) + gap_med
        intraday_log_return = (idr_scaled * idr_iqr) + idr_med
        normalized_range = (range_scaled * nr_iqr) + nr_med
        normalized_range = np.maximum(normalized_range, 0.0)

        # ── Reconstruct Prices via Sequential Chaining ──
        open_prices = np.zeros((n_samples, seq_len))
        close_prices = np.zeros((n_samples, seq_len))
        high_prices = np.zeros((n_samples, seq_len))
        low_prices = np.zeros((n_samples, seq_len))

        for t in range(seq_len):
            if t == 0:
                prev_close = anchors
            else:
                prev_close = close_prices[:, t - 1]

            open_prices[:, t] = prev_close * np.exp(gap_log_return[:, t])
            close_prices[:, t] = open_prices[:, t] * np.exp(intraday_log_return[:, t])

            total_range = prev_close * normalized_range[:, t]
            total_range = np.maximum(total_range, 0.0)

            max_oc = np.maximum(open_prices[:, t], close_prices[:, t])
            min_oc = np.minimum(open_prices[:, t], close_prices[:, t])

            high_prices[:, t] = max_oc + wick_high_ratio[:, t] * total_range
            low_prices[:, t] = min_oc - wick_low_ratio[:, t] * total_range

        # ── Volume Reconstruction: Log-Deviation Inverse ──
        vol_medians_exp = vol_medians.reshape(-1, 1)
        vol_iqrs_exp = vol_iqrs.reshape(-1, 1)

        log_volume = (vol_log_dev * vol_iqrs_exp) + vol_medians_exp
        volume = np.exp(log_volume) - 1e-10
        volume = np.maximum(volume, 0.0)

        ohlcv = np.stack([
            open_prices,
            high_prices,
            low_prices,
            close_prices,
            volume
        ], axis=-1)

        return ohlcv.astype(np.float32)

    def train_dataloader(self):
        if getattr(self, 'data_on_gpu', False):
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True
            )
        else:
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
                drop_last=True
            )
