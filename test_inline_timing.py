
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(),'src'))

import torch
import numpy as np
import time

# Mocking the modules that InlineEvaluationCallback imports from src
# Since InlineEvaluationCallback is in src/training, we need to make sure 
# its own relative imports or standard imports work.
# Actually it just imports standard libs and pl.

from training.inline_evaluation import InlineEvaluationCallback

class MockDataModule:
    def __init__(self):
        self.data_tensor = torch.randn(10, 24, 5)
        self.raw_data_tensor = torch.randn(500, 24, 5)
    
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
        # Simulate some processing time
        time.sleep(0.01)
        return torch.randn_like(x)

def test_timing():
    dm = MockDataModule()
    callback = InlineEvaluationCallback(dm, n_samples=10)
    model = MockModel()
    
    print("Starting generation test...")
    callback._generate_samples(model)
    print("Generation test finished.")

if __name__ == "__main__":
    test_timing()
