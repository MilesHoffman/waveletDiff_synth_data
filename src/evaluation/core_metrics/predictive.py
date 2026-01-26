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
    """GRU-based predictor for one-step ahead prediction."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        y_hat_logit = self.fc(outputs)
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat


def _train_predictor(
    train_data: np.ndarray,
    test_data: np.ndarray,
    iterations: int = 5000,
    batch_size: int = 128
) -> float:
    """
    Train predictor on train_data and evaluate on test_data.
    
    Returns MAE score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    no, seq_len, dim = train_data.shape
    hidden_dim = max(dim // 2, 2)

    model = _Predictor(
        input_dim=(dim - 1) if dim > 1 else 1, 
        hidden_dim=hidden_dim
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    def prepare_batch(data, batch_indices):
        if dim > 1:
            X_mb = [
                torch.tensor(data[i][:-1, :dim - 1], dtype=torch.float32)
                for i in batch_indices
            ]
            Y_mb = [
                torch.tensor(data[i][1:, dim - 1:], dtype=torch.float32)
                for i in batch_indices
            ]
            X_tensor = nn.utils.rnn.pad_sequence(X_mb, batch_first=True).to(device)
            Y_tensor = nn.utils.rnn.pad_sequence(Y_mb, batch_first=True).to(device)
        else:
            window_size = 20
            X_mb = [
                torch.tensor(data[i][:-window_size], dtype=torch.float32).unsqueeze(-1)
                for i in batch_indices
            ]
            Y_mb = [
                torch.tensor(data[i][window_size:], dtype=torch.float32).unsqueeze(-1)
                for i in batch_indices
            ]
            X_tensor = torch.stack(X_mb).squeeze(-1).to(device)
            Y_tensor = torch.stack(Y_mb).squeeze(-1).to(device)
        return X_tensor, Y_tensor

    # Training
    model.train()
    for _ in tqdm(range(iterations), desc="Predictive Training", leave=False):
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
    X_mb, Y_mb = prepare_batch(test_data, batch_indices)

    with torch.no_grad():
        pred_Y = model(X_mb)

    mae_total = 0
    for i in range(test_no):
        pred = pred_Y[i].cpu().numpy()
        target = Y_mb[i].cpu().numpy()
        mae_total += mean_absolute_error(target, pred)

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
