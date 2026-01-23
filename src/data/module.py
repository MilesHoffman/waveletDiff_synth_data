"""
Wavelet Time Series Data Module for PyTorch Lightning.

Revised from the original wavelet_transformer/wavelet_data_module.py
to fit the new modular architecture.
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
        WaveletTimeSeriesDataModule for loading time series datasets and converting to wavelet coefficients.
        
        Args:
            config: Configuration dict containing all parameters
            data_tensor: Pre-loaded data tensor (if None, loads from config)
            **kwargs: Individual parameters for backward compatibility
        """
        super().__init__()
        
        # Enforce config usage
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

        # Load raw time series data
        if data_tensor is not None:
            self.raw_data_tensor, self.norm_stats = data_tensor, None
        elif self.dataset_name is not None:
            self.raw_data_tensor, self.norm_stats = self._load_dataset(self.dataset_name, seq_len=self.seq_len, normalize_data=self.normalize_data)
        else:
            raise ValueError("Either dataset_name or data_tensor must be provided")

        print("Raw Data Tensor Shape:", self.raw_data_tensor.shape)

        # Convert to wavelet coefficients
        self.data_tensor, self.wavelet_info = self._convert_to_wavelet_coefficients()
        
        # Create ATR tensor for conditioning (if using reparameterized stocks data)
        if self.norm_stats is not None and self.norm_stats.get('reparameterized', False):
            self.atr_tensor = torch.FloatTensor(self.norm_stats['atr_pcts'])
            self.has_conditioning = True
            print(f"ATR conditioning enabled: {len(self.atr_tensor)} samples")
        else:
            self.atr_tensor = None
            self.has_conditioning = False
        
        # Move dataset to GPU RAM if available (eliminates PCIe transfer overhead)
        self.data_on_gpu = torch.cuda.is_available()
        if self.data_on_gpu:
            self.data_tensor = self.data_tensor.cuda()
            if self.atr_tensor is not None:
                self.atr_tensor = self.atr_tensor.cuda()
            print("Dataset moved to GPU RAM for faster training")
        
        # Create dataset with conditioning if available
        if self.has_conditioning:
            self.dataset = TensorDataset(self.data_tensor, self.atr_tensor)
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
            raw_data, norm_stats = load_stocks_data(self.data_dir, seq_len=seq_len, normalize_data=normalize_data)
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

        # Auto-detect wavelet type if not specified
        if self.wavelet_type == "auto":
            if seq_len <= 32:
                self.wavelet_type = 'db2'
            elif seq_len <= 64:
                self.wavelet_type = 'db4'
            elif seq_len <= 128:
                self.wavelet_type = 'db6'
            else:
                self.wavelet_type = 'db8'

        # Auto-detect levels if not specified
        if self.num_levels == "auto":
            self.num_levels = int(np.clip(pywt.dwt_max_level(seq_len, self.wavelet_type), 3, 7))
        
        print(f"Converting to wavelet coefficients with {self.num_levels} levels...")
        
        # Get coefficient shapes by decomposing a sample signal
        sample_signal = raw_data[0, :, 0]
        sample_coeffs = pywt.wavedec(sample_signal, self.wavelet_type, level=self.num_levels, mode=self.mode)
        coeffs_shapes = [c.shape for c in sample_coeffs]
        level_dims = [np.prod(shape) for shape in coeffs_shapes]
        total_coeffs_per_feature = sum(level_dims)

        print(f"Coefficient shapes per level: {coeffs_shapes}")
        print(f"Level dimensions: {level_dims}")
        print(f"Total coefficients per feature: {total_coeffs_per_feature}")

        # Initialize output array [n_samples, total_coeffs_per_feature, n_features]
        wavelet_coeffs = np.zeros((n_samples, total_coeffs_per_feature, n_features))

        # Process each sample and feature
        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                signal = raw_data[sample_idx, :, feature_idx]
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(signal, self.wavelet_type, level=self.num_levels, mode=self.mode)
                
                # Flatten and store coefficients
                coeffs_flat = np.concatenate([c.flatten() for c in coeffs])
                wavelet_coeffs[sample_idx, :, feature_idx] = coeffs_flat

        # Convert to tensor
        wavelet_tensor = torch.FloatTensor(wavelet_coeffs)

        # Calculate level start indices for easy access
        level_start_indices = [0] + list(np.cumsum(level_dims[:-1]))
        
        # Store reconstruction information
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

        # Verify dimensions match
        expected_n_features = self.wavelet_info['n_features']
        expected_n_level_dim = self.wavelet_info['total_coeffs_per_feature']
        
        if n_features != expected_n_features:
            raise ValueError(f"Feature dimension mismatch: expected {expected_n_features}, got {n_features}")
        if n_level_dim != expected_n_level_dim:
            raise ValueError(f"Level dimension mismatch: expected {expected_n_level_dim}, got {n_level_dim}")
        
        reconstructed_signals = []

        for sample_idx in range(n_samples):
            sample_features = []
            
            # Process each feature
            for feature_idx in range(n_features):
                coeffs_flat = wavelet_coeffs[sample_idx, :, feature_idx]
                
                # Reconstruct coefficient structure
                coeffs = []
                for level_idx, (shape, dim, start_idx) in enumerate(zip(coeffs_shapes, level_dims, level_start_indices)):
                    end_idx = start_idx + dim
                    coeff = coeffs_flat[start_idx:end_idx].reshape(shape)
                    coeffs.append(coeff)
                
                # Perform inverse wavelet transform
                reconstructed = pywt.waverec(coeffs, self.wavelet_type, mode=self.mode)
                
                # Ensure the reconstructed signal has the correct length
                if len(reconstructed) > original_seq_len:
                    reconstructed = reconstructed[:original_seq_len]
                elif len(reconstructed) < original_seq_len:
                    # Pad if necessary (should rarely happen with proper wavelet settings)
                    pad_length = original_seq_len - len(reconstructed)
                    reconstructed = np.pad(reconstructed, (0, pad_length), mode='constant', constant_values=0)
                
                sample_features.append(reconstructed)
            
            # Stack features: [seq_len, n_features]
            sample_reconstructed = np.stack(sample_features, axis=1)
            reconstructed_signals.append(sample_reconstructed)
        
        # Return [n_samples, seq_len, n_features]
        return torch.FloatTensor(np.stack(reconstructed_signals))

    def get_input_dim(self) -> int:
        """Get the input dimension for the model (number of wavelet coefficients)."""
        return self.data_tensor.shape[1]

    def get_wavelet_info(self) -> dict:
        """Get wavelet transformation information."""
        return self.wavelet_info

    def inverse_normalize(self, data: np.ndarray, sample_indices: np.ndarray = None) -> np.ndarray:
        """
        Inverse normalization to convert generated samples back to original scale.
        
        For reparameterized OHLC data, reconstructs O, H, L, C, V from normalized features.
        
        Args:
            data: Normalized data of shape (n_samples, seq_len, n_features)
            sample_indices: Optional indices to select specific anchor/atr_pct values.
                           If None and reparameterized, samples from stored distributions.
            
        Returns:
            Denormalized data in original scale: (n_samples, seq_len, 5) for OHLCV
        """
        if self.norm_stats is None:
            return data
        
        data = data.copy()
        
        if self.norm_stats.get('reparameterized', False):
            return self._inverse_reparameterize_ohlc(data, sample_indices)
        
        mean = self.norm_stats['mean']
        std = self.norm_stats['std']
        data = data * std + mean
        
        if self.norm_stats.get('volume_log_transformed', False):
            data[..., 4] = np.maximum(0, np.expm1(data[..., 4]))
        
        return data
    
    def _inverse_reparameterize_ohlc(self, data: np.ndarray, sample_indices: np.ndarray = None) -> np.ndarray:
        """
        Inverse reparameterization for OHLC data.
        
        Reconstructs OHLCV from: [open_norm, body_norm, wick_high_norm, wick_low_norm, volume_norm]
        """
        n_samples = data.shape[0]
        
        if sample_indices is not None:
            anchors = self.norm_stats['anchors'][sample_indices]
            atr_pcts = self.norm_stats['atr_pcts'][sample_indices]
        else:
            all_anchors = self.norm_stats['anchors']
            all_atr_pcts = self.norm_stats['atr_pcts']
            indices = np.random.choice(len(all_anchors), size=n_samples, replace=True)
            anchors = all_anchors[indices]
            atr_pcts = all_atr_pcts[indices]
        
        anchors = anchors.reshape(-1, 1, 1)
        atr_pcts = atr_pcts.reshape(-1, 1, 1)
        
        open_norm = data[..., 0:1]
        body_norm = data[..., 1:2]
        wick_high_norm = data[..., 2:3]
        wick_low_norm = data[..., 3:4]
        volume_norm = data[..., 4:5]
        
        open_pct = open_norm * atr_pcts
        body_pct = body_norm * atr_pcts
        wick_high_pct = np.maximum(0, wick_high_norm * atr_pcts)
        wick_low_pct = np.maximum(0, wick_low_norm * atr_pcts)
        
        vol_smas = None
        if 'vol_smas' in self.norm_stats:
            if sample_indices is not None:
                vol_smas = self.norm_stats['vol_smas'][sample_indices]
            else:
                vol_smas = self.norm_stats['vol_smas'][indices]
        
        anchors = anchors.reshape(-1, 1, 1)
        atr_pcts = atr_pcts.reshape(-1, 1, 1)
        
        open_norm = data[..., 0:1]
        body_norm = data[..., 1:2]
        wick_high_norm = data[..., 2:3]
        wick_low_norm = data[..., 3:4]
        volume_norm = data[..., 4:5]
        
        open_pct = open_norm * atr_pcts
        body_pct = body_norm * atr_pcts
        wick_high_pct = np.maximum(0, wick_high_norm * atr_pcts)
        wick_low_pct = np.maximum(0, wick_low_norm * atr_pcts)
        
        open_prices = anchors + (open_pct / 100.0) * anchors
        close_prices = open_prices + (body_pct / 100.0) * anchors
        
        max_oc = np.maximum(open_prices, close_prices)
        min_oc = np.minimum(open_prices, close_prices)
        
        high_prices = max_oc + (wick_high_pct / 100.0) * anchors
        low_prices = min_oc - (wick_low_pct / 100.0) * anchors
        
        # Volume reconstruction: exp(V_norm) * SMA_20
        # V_norm = log((Volume + eps) / (SMA + eps))
        # Volume ≈ exp(V_norm) * SMA
        if self.norm_stats.get('volume_type') == 'log_ratio_sma' and vol_smas is not None:
            volume = np.exp(volume_norm) * vol_smas[..., np.newaxis]
            volume = np.maximum(0, volume)
        else:
            # Fallback for old method (shouldn't be reached with new loader)
            vol_mean = self.norm_stats.get('volume_mean', 0)
            vol_std = self.norm_stats.get('volume_std', 1)
            volume_log = volume_norm * vol_std + vol_mean
            volume = np.maximum(0, np.expm1(volume_log))
        
        ohlcv = np.concatenate([
            open_prices,
            high_prices,
            low_prices,
            close_prices,
            volume
        ], axis=-1)
        
        return ohlcv.astype(np.float32)

    def train_dataloader(self):
        # When data is on GPU, use num_workers=0 (no CPU workers needed)
        # When data is on CPU, use persistent workers for speed
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

