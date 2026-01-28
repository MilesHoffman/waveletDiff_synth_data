"""
Predictive Utility - TSTR/TRTR benchmark for downstream task usefulness.

TSTR: Train on Synthetic, Test on Real
TRTR: Train on Real, Test on Real (baseline)
Utility Gap: |TSTR - TRTR| (lower is better)

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm
from typing import Tuple


class _Predictor(nn.Module):
    """RNN-based predictor model for Multi-step prediction."""

    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(_Predictor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        # Predict based on the last hidden state
        y_hat = self.fc(outputs[:, -1, :])
        return y_hat


def _train_predictor(
    train_data: np.ndarray,
    test_data: np.ndarray,
    iterations: int = 5000,
    batch_size: int = 128
) -> float:
    """
    Train predictor on train_data and evaluate on test_data (Multi-step).
    
    Returns MAE score over 5-step horizon.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    no, seq_len, dim = train_data.shape
    hidden_dim = max(dim, 32)
    horizon = 5  # HARDENING: Multi-step prediction

    # Ensure sequence length is sufficient
    if seq_len <= horizon:
        raise ValueError(f"Sequence length ({seq_len}) must be greater than horizon ({horizon})")

    # Input dim is D, Output is horizon * D (flattened) or just horizon (if univariate)
    # We predict the next 'horizon' values for all features? Usually TSTR sums over features.
    # Let's keep it simple: Predict next 5 steps of ALL features.
    
    input_seq_len = seq_len - horizon
    output_dim = horizon * dim

    model = _Predictor(
        input_dim=dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim
    ).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def prepare_batch(data, batch_indices):
        # x: [batch, input_seq_len, dim]
        # y: [batch, horizon * dim] (flattened)
        
        X_mb = []
        Y_mb = []
        
        for i in batch_indices:
            # Input: 0 to T-H
            x_seq = data[i, :input_seq_len, :]
            # Target: T-H to T
            y_seq = data[i, input_seq_len:, :]
            
            X_mb.append(x_seq)
            Y_mb.append(y_seq.flatten())
            
        X_tensor = torch.tensor(np.array(X_mb), dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(np.array(Y_mb), dtype=torch.float32).to(device)

        return X_tensor, Y_tensor

    # Training
    model.train()
    for _ in tqdm(range(iterations), desc="Predictive Training (Multi-step)", leave=False):
        batch_indices = np.random.choice(len(train_data), batch_size, replace=False)
        X_mb, Y_mb = prepare_batch(train_data, batch_indices)

        optimizer.zero_grad()
        y_pred = model(X_mb)
        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()

    # Evaluation on test data
    model.eval()
    test_no = len(test_data)
    batch_indices = np.arange(test_no)
    
    # Process in chunks to avoid OOM
    eval_batch_size = 256
    mae_total = 0
    
    with torch.no_grad():
        for start_idx in range(0, test_no, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, test_no)
            indices = np.arange(start_idx, end_idx)
            
            X_mb, Y_mb = prepare_batch(test_data, indices)
            pred_Y = model(X_mb)
            
            mae_total += criterion(pred_Y, Y_mb).item() * len(indices)

    return mae_total / test_no


def predictive_utility(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    iterations: int = 5000,
    batch_size: int = 128
) -> Tuple[float, float, float]:
    """
    Compute TSTR and TRTR predictive scores.
    
    Args:
        real_data: Real data of shape (N, T, D), should be scaled to [0,1]
        synth_data: Synthetic data of shape (N, T, D), should be scaled to [0,1]
        iterations: Number of training iterations
        batch_size: Batch size for training
        
    Returns:
        Tuple of (TSTR, TRTR, Utility Gap)
        - TSTR: Train on Synthetic, Test on Real (MAE)
        - TRTR: Train on Real, Test on Real (MAE baseline)
        - Utility Gap: |TSTR - TRTR| (lower is better)
    """
    # TSTR: Train on Synthetic, Test on Real
    tstr = _train_predictor(synth_data, real_data, iterations, batch_size)
    
    # TRTR: Train on Real, Test on Real (80/20 split)
    n = len(real_data)
    split_idx = int(n * 0.8)
    idx = np.random.permutation(n)
    train_real = real_data[idx[:split_idx]]
    test_real = real_data[idx[split_idx:]]
    trtr = _train_predictor(train_real, test_real, iterations, batch_size)
    
    utility_gap = abs(tstr - trtr)
    
    return float(tstr), float(trtr), float(utility_gap)
