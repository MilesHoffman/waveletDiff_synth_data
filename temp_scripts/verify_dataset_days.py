
import sys
import os
import torch
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.loaders import load_stocks_data
from data.module import WaveletTimeSeriesDataModule

def verify_dataset_days():
    from utils.config import INTERNAL_DEFAULTS
    
    # 1. Use actual dataset path if possible, but for a 1-off we'll generate SPY-like data
    # (actually let's check the current workspace for a CSV)
    # Looking at my earlier runs, there's likely a CSV.
    # Actually, I'll just check the day distribution of any stocks-like sequence.
    
    # Let's create a date range of 30 days including weekends.
    dates = pd.date_range("2024-01-01", "2024-01-30")
    df = pd.DataFrame({'Date': dates})
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Filter only trading days (SPY behavior)
    trading_days = df[~df['is_weekend']]
    print("Trading Days - Day of Week Distributions:")
    print(trading_days['day_of_week'].value_counts().sort_index())
    
    # Calculate transitions
    vals = trading_days['day_of_week'].values
    transitions = []
    for i in range(len(vals)-1):
        diff = (vals[i+1] - vals[i]) % 7
        transitions.append(diff)
    
    print("\nAdjacency (days difference):")
    counts = pd.Series(transitions).value_counts()
    print(counts)
    
    # Analysis for 5-day cycle:
    # If Friday=4, Monday=0. The model see 4 -> 0.
    # sin(4/5 * 2PI) = sin(1.6PI) = -0.95
    # sin(0/5 * 2PI) = sin(0) = 0
    # Gap is 0.95
    
    # Analysis for 7-day cycle:
    # sin(4/7 * 2PI) = sin(1.14PI) = -0.43
    # Gap is 0.43
    
    print("\nCyclic encoding check (max_val=5 for 0..4):")
    for d in range(5):
        s = np.sin(2 * np.pi * d / 5.0)
        c = np.cos(2 * np.pi * d / 5.0)
        print(f"Day {d}: sin={s:.2f}, cos={c:.2f}")

    print("\nCyclic encoding check (max_val=7 for 0..4):")
    for d in range(5):
        s = np.sin(2 * np.pi * d / 7.0)
        c = np.cos(2 * np.pi * d / 7.0)
        print(f"Day {d}: sin={s:.2f}, cos={c:.2f}")
    
if __name__ == "__main__":
    verify_dataset_days()
