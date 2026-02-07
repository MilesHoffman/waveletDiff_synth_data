"""
Legacy Core Metrics - Re-implementation of original benchmarks.

Includes:
- Discriminative Score (1-layer GRU)
- Predictive Score (1-step prediction)

These are provided for comparison with the "hardened" metrics in the main pipeline.
Reference: TimeGAN Codebase
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm.auto import tqdm
from typing import Tuple

# --- UTILS ---

def batch_generator(data, time, batch_size):
    """Mini-batch generator."""
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb

def extract_time(data):
    """Returns Maximum sequence length and each sequence length.
    
    Args:
      - data: original data
    
    Returns:
      - time: each sequence length
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))
        
    return time, max_seq_len

# --- 1. LEGACY DISCRIMINATIVE SCORE ---

class LegacyDiscriminator(nn.Module):
    """Original 1-layer GRU Discriminator."""
    def __init__(self, input_dim, hidden_dim):
        super(LegacyDiscriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Note: Original implementation might not pack sequences?
        # But let's assume standard GRU usage
        _, hidden = self.rnn(x)
        # hidden is (num_layers, batch, hidden_dim) -> (1, B, H)
        logits = self.fc(hidden[-1])
        return logits

def discriminative_score_legacy(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    iterations: int = 2000,
    batch_size: int = 128
) -> float:
    """
    Compute discriminative score using original 1-layer GRU architecture.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    no, seq_len, dim = real_data.shape
    hidden_dim = max(dim // 2, 2)
    
    # Train/Test Split (Simple 80/20)
    idx = np.random.permutation(len(real_data))
    train_idx = idx[:int(len(real_data)*0.8)]
    test_idx = idx[int(len(real_data)*0.8):]
    
    train_x = real_data[train_idx]
    test_x = real_data[test_idx]
    train_x_hat = synth_data[train_idx]
    test_x_hat = synth_data[test_idx]
    
    # Model
    model = LegacyDiscriminator(dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for _ in range(iterations):
        # Batching
        b_idx_real = np.random.choice(len(train_x), batch_size)
        b_idx_fake = np.random.choice(len(train_x_hat), batch_size)
        
        X_real = torch.tensor(train_x[b_idx_real], dtype=torch.float32).to(device)
        X_fake = torch.tensor(train_x_hat[b_idx_fake], dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        
        y_real = model(X_real)
        y_fake = model(X_fake)
        
        loss_real = criterion(y_real, torch.ones_like(y_real))
        loss_fake = criterion(y_fake, torch.zeros_like(y_fake))
        loss = loss_real + loss_fake
        
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_real = torch.tensor(test_x, dtype=torch.float32).to(device)
        test_fake = torch.tensor(test_x_hat, dtype=torch.float32).to(device)
        
        pred_real = torch.sigmoid(model(test_real)).cpu().numpy()
        pred_fake = torch.sigmoid(model(test_fake)).cpu().numpy()
        
    y_pred = np.concatenate([pred_real, pred_fake]).squeeze()
    y_true = np.concatenate([np.ones(len(pred_real)), np.zeros(len(pred_fake))])
    
    acc = accuracy_score(y_true, y_pred > 0.5)
    
    return np.abs(0.5 - acc)


# --- 2. LEGACY PREDICTIVE SCORE ---

class LegacyPredictor(nn.Module):
    """Original 1-layer GRU Predictor (1-step ahead)."""
    def __init__(self, input_dim, hidden_dim):
        super(LegacyPredictor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        # Prediction based on last hidden state -> 1 step ahead
        y_hat_logit = self.fc(outputs) 
        # Note: Original implementation sometimes used sigmoid if data [0,1]?
        # Let's stick to linear output for MAE unless original had sigmoid.
        # Checking implementation: "y_hat = torch.sigmoid(y_hat_logit)" was in source.
        # We will assume data is scaled [0,1].
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat

def predictive_score_legacy(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    iterations: int = 5000,
    batch_size: int = 128
) -> float:
    """
    Compute predictive MAE using original 1-step ahead prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no, seq_len, dim = real_data.shape
    hidden_dim = max(dim // 2, 2)
    
    # Model: Predict next step based on history
    # Input: [B, T-1, D], Output: [B, T-1, D] (shifted)
    
    class LegacyOneStepPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, input_dim) # Predict all features
            
        def forward(self, x):
            # x: [B, T, D]
            outputs, _ = self.rnn(x)
            y_hat = torch.sigmoid(self.fc(outputs))
            return y_hat

    model = LegacyOneStepPredictor(dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.L1Loss()
    
    # Train on Synthetic
    model.train()
    for _ in range(iterations):
        idx = np.random.choice(len(synth_data), batch_size)
        data = torch.tensor(synth_data[idx], dtype=torch.float32).to(device)
        
        # Predict next step:
        # Input: X[:, :-1, :]
        # Target: X[:, 1:, :] 
        X_mb = data[:, :-1, :]
        Y_mb = data[:, 1:, :]
        
        optimizer.zero_grad()
        y_pred = model(X_mb)
        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()
        
    # Test on Real
    model.eval()
    with torch.no_grad():
        X_real = torch.tensor(real_data[:, :-1, :], dtype=torch.float32).to(device)
        Y_real = torch.tensor(real_data[:, 1:, :], dtype=torch.float32).to(device)
        
        pred_Y = model(X_real)
        mae = criterion(pred_Y, Y_real).item()
    
    return mae
