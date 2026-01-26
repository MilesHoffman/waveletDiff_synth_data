"""
Discriminative Score - Post-hoc RNN classifier for Real vs Fake.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from typing import Tuple


def _extract_time(data: np.ndarray) -> Tuple[list, int]:
    """Returns Maximum sequence length and each sequence length."""
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def _train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data."""
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    )


def _batch_generator(data, time, batch_size):
    """Mini-batch generator."""
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb


class _Discriminator(nn.Module):
    """Discriminator model for distinguishing between real and synthetic time-series data."""

    def __init__(self, input_dim, hidden_dim):
        super(_Discriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed_input)
        logits = self.fc(hidden[-1])
        return logits


def discriminative_score(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    iterations: int = 2000,
    batch_size: int = 128
) -> float:
    """
    Compute discriminative score using post-hoc RNN classifier.
    
    Args:
        real_data: Real data of shape (N, T, D), should be scaled to [0,1]
        synth_data: Synthetic data of shape (N, T, D), should be scaled to [0,1]
        iterations: Number of training iterations
        batch_size: Batch size for training
        
    Returns:
        Discriminative score: |0.5 - accuracy|
        (Closer to 0 means classifier is confused, which is better)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no, seq_len, dim = np.asarray(real_data).shape
    hidden_dim = max(dim // 2, 2)

    ori_time, _ = _extract_time(real_data)
    gen_time, _ = _extract_time(synth_data)
    
    (train_x, train_x_hat, test_x, test_x_hat,
     train_t, train_t_hat, test_t, test_t_hat) = _train_test_divide(
        real_data, synth_data, ori_time, gen_time
    )

    discriminator = _Discriminator(dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.parameters())

    # Training
    discriminator.train()
    for _ in tqdm(range(iterations), desc="Discriminative Training", leave=False):
        X_mb, T_mb = _batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = _batch_generator(train_x_hat, train_t_hat, batch_size)

        X_mb = torch.tensor(np.array(X_mb), dtype=torch.float32).to(device)
        T_mb = torch.tensor(np.array(T_mb), dtype=torch.long).to(device)
        X_hat_mb = torch.tensor(np.array(X_hat_mb), dtype=torch.float32).to(device)
        T_hat_mb = torch.tensor(np.array(T_hat_mb), dtype=torch.long).to(device)

        optimizer.zero_grad()

        logits_real = discriminator(X_mb, T_mb)
        logits_fake = discriminator(X_hat_mb, T_hat_mb)

        loss_real = criterion(logits_real, torch.ones_like(logits_real))
        loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake))
        loss = loss_real + loss_fake

        loss.backward()
        optimizer.step()

    # Evaluation
    discriminator.eval()
    test_x_tensor = torch.tensor(np.array(test_x), dtype=torch.float32).to(device)
    test_t_tensor = torch.tensor(np.array(test_t), dtype=torch.long).to(device)
    test_x_hat_tensor = torch.tensor(np.array(test_x_hat), dtype=torch.float32).to(device)
    test_t_hat_tensor = torch.tensor(np.array(test_t_hat), dtype=torch.long).to(device)

    with torch.no_grad():
        y_pred_real = torch.sigmoid(discriminator(test_x_tensor, test_t_tensor)).cpu().numpy()
        y_pred_fake = torch.sigmoid(discriminator(test_x_hat_tensor, test_t_hat_tensor)).cpu().numpy()

    y_pred_final = np.concatenate((y_pred_real, y_pred_fake), axis=0).squeeze()
    y_label_final = np.concatenate([
        np.ones(len(y_pred_real)),
        np.zeros(len(y_pred_fake)),
    ])

    acc = accuracy_score(y_label_final, y_pred_final > 0.5)
    score = np.abs(0.5 - acc)
    
    return float(score)
