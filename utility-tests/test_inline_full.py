
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import numpy as np

from training.inline_evaluation import InlineEvaluationCallback

_N_FEATURES = 5
_SEQ_LEN    = 24
_N_WAVELET  = 32  # flat wavelet coeff dim used by mock
_N_EVAL     = 10


class MockDataModule:
    """Minimal data module stub exercising the full inline evaluation pipeline."""

    def __init__(self):
        self.raw_data_tensor = torch.randn(20, _SEQ_LEN, _N_FEATURES)
        self.norm_stats = None
        self.has_path_sig_conditioning = False

    def convert_wavelet_to_timeseries(self, wavelet: torch.Tensor) -> torch.Tensor:
        # wavelet: [N, _N_WAVELET, _N_FEATURES] → [N, _SEQ_LEN, _N_FEATURES]
        return torch.randn(wavelet.shape[0], _SEQ_LEN, _N_FEATURES)

    def inverse_normalize(self, data: np.ndarray, sample_indices=None) -> np.ndarray:
        return data

    def get_input_dim(self) -> int:
        return _N_WAVELET

    def get_wavelet_info(self) -> dict:
        return {'n_features': _N_FEATURES}


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def forward(self, x, t, scale=None, conditions=None):
        return torch.randn_like(x)

    def log(self, k, v, prog_bar=False):
        print(f"  Log: {k} = {v:.6f}")


class MockTrainer:
    def __init__(self):
        self.current_epoch = 199  # triggers eval at epoch 200
        self.callbacks = []


def _mock_generate_samples(self, pl_module, conditions=None) -> torch.Tensor:
    """Bypass DiffusionTrainer entirely — return plausible random wavelet coeffs."""
    return torch.randn(self.n_samples, _N_WAVELET, _N_FEATURES)


def test_full_flow():
    dm = MockDataModule()
    ohlcv_indices = {'open': 0, 'high': 1, 'low': 2, 'close': 3}
    callback = InlineEvaluationCallback(
        dm, n_samples=_N_EVAL, eval_every_n_epochs=200, ohlcv_indices=ohlcv_indices
    )
    # Patch the diffusion sampling to bypass DiffusionTrainer
    callback._generate_samples = lambda pl_module, conditions=None: _mock_generate_samples(
        callback, pl_module, conditions
    )

    model   = MockModel()
    trainer = MockTrainer()

    print("Testing on_train_epoch_end (Full Evaluation)...")
    callback.on_train_epoch_end(trainer, model)
    print("\nTest finished successfully.")


if __name__ == "__main__":
    test_full_flow()
