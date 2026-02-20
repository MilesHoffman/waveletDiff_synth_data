import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from evaluation.advanced_metrics.financial_metrics import (
    calculate_tail_dependence,
    calculate_hurst_metrics,
    calculate_leverage_effect,
    calculate_drawdown_stats
)

def test_financial_metrics():
    print("Testing Financial Metrics...")
    
    # Create dummy data (N=100, T=50, D=3)
    # Real: Random Walk
    # Synth: Random Walk + Noise
    np.random.seed(42)
    real = np.cumsum(np.random.randn(100, 50, 3), axis=1) + 100
    synth = np.cumsum(np.random.randn(100, 50, 3), axis=1) + 100
    
    # 1. Tail Dependence
    print("1. Testing Tail Dependence...")
    try:
        td = calculate_tail_dependence(real, synth)
        print("   PASS:", td)
    except Exception as e:
        print("   FAIL:", e)
        
    # 2. Hurst
    print("2. Testing Hurst...")
    try:
        hurst = calculate_hurst_metrics(real, synth)
        print("   PASS:", hurst)
    except Exception as e:
        print("   FAIL:", e)
        
    # 3. Leverage
    print("3. Testing Leverage...")
    try:
        lev = calculate_leverage_effect(real, synth)
        print("   PASS:", lev)
    except Exception as e:
        print("   FAIL:", e)
        
    # 4. Drawdowns
    print("4. Testing Drawdowns...")
    try:
        dd = calculate_drawdown_stats(real, synth)
        print("   PASS:", dd)
    except Exception as e:
        print("   FAIL:", e)

if __name__ == "__main__":
    test_financial_metrics()
