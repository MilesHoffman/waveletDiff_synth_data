import numpy as np
import jax.numpy as jnp
from src.data.module import WaveletTimeSeriesDataModule

class JAXDataBridge:
    """
    Bridges the PyTorch-based WaveletTimeSeriesDataModule to JAX/Keras 3.
    Splits flattened coefficients into level-wise lists.
    """
    def __init__(self, data_module: WaveletTimeSeriesDataModule):
        self.dm = data_module
        self.wavelet_info = data_module.get_wavelet_info()
        self.level_dims = self.wavelet_info['level_dims']
        self.level_start_indices = self.wavelet_info['level_start_indices']

    def _split_levels(self, flat_batch):
        # flat_batch: (batch, total_coeffs, channels)
        levels = []
        for i, (start_idx, dim) in enumerate(zip(self.level_start_indices, self.level_dims)):
            end_idx = start_idx + dim
            # Extract and convert to jnp
            level_data = jnp.array(flat_batch[:, start_idx:end_idx, :])
            levels.append(level_data)
        return levels

    def get_iterator(self):
        """Returns an infinite generator yielding JAX-friendly level lists."""
        pt_loader = self.dm.train_dataloader()
        while True:
            for batch in pt_loader:
                # batch[0] is the flattened tensor from TensorDataset
                flat_data = batch[0].numpy()
                yield self._split_levels(flat_data)

    def get_level_dims(self):
        return self.level_dims
