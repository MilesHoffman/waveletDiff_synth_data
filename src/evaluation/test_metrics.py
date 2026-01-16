
import numpy as np
import sys
import os

# Add current directory to path to import advanced_metrics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_metrics import calculate_memorization_ratio, calculate_diversity_metrics, calculate_fld

def test_metrics():
    print("Generating dummy data...")
    # Real data: 100 samples, 50 time steps, 4 features
    real = np.random.randn(100, 50, 4)
    # Synthetic: same shape, slightly different distribution
    synth = np.random.randn(100, 50, 4) + 0.1
    
    # Test Memorization
    print("Testing Memorization Ratio...")
    mem_ratio = calculate_memorization_ratio(real, synth)
    print(f"Memorization Ratio: {mem_ratio}")
    assert 0 <= mem_ratio <= 1, "Memorization ratio out of bounds"
    
    # Test Diversity
    print("Testing Diversity (Coverage)...")
    div = calculate_diversity_metrics(real, synth, k=5)
    print(f"Diversity: {div}")
    assert "Coverage" in div, "Coverage missing"
    assert 0 <= div["Coverage"] <= 1, "Coverage out of bounds"
    
    # Test FLD
    print("Testing FLD...")
    fld = calculate_fld(real, synth)
    print(f"FLD: {fld}")
    assert isinstance(fld, float), "FLD not a float"
    
    print("\n✅ All smoke tests passed!")

if __name__ == "__main__":
    test_metrics()
