
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(),'src'))

import torch
import numpy as np
import time

from training.inline_evaluation import InlineEvaluationCallback

class MockDataModule:
    def __init__(self):
        # 5 features: [open_norm, body_norm, wick_high_norm, wick_low_norm, volume_norm]
        self.data_tensor = torch.randn(10, 24, 5)
    
    def convert_wavelet_to_timeseries(self, wavelet):
        return wavelet
    
    def inverse_normalize(self, data, sample_indices=None):
        return data

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 100
        self.register_buffer('beta_all', torch.linspace(1e-4, 0.02, 100))
        self.register_buffer('alpha_bar_all', torch.cumprod(1 - self.beta_all, dim=0))
        self.device = torch.device('cpu')
    
    def forward(self, x, t):
        # Simulate some processing time for first few steps
        time.sleep(0.01)
        return torch.randn_like(x)
    
    def log(self, k, v, prog_bar=False):
        print(f"Log: {k} = {v}")

class MockTrainer:
    def __init__(self):
        self.current_epoch = 199 # So +1 = 200

def test_full_flow():
    dm = MockDataModule()
    # ohlcv_indices={...} triggers the invariant check
    callback = InlineEvaluationCallback(dm, n_samples=10, eval_every_n_epochs=200, ohlcv_indices={'open':0})
    model = MockModel()
    trainer = MockTrainer()
    
    print("Testing on_train_epoch_end (Full Evaluation)...")
    callback.on_train_epoch_end(trainer, model)
    print("Test finished.")

if __name__ == "__main__":
    test_full_flow()
