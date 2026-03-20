import os
import sys
import torch

# Add src to python path so imports work
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
sys.path.insert(0, src_path)

from models.attention import CrossLevelAttention

def test_pyramidal_attention():
    batch_size = 4
    # Mock multi-resolution wavelet levels
    # Level 0 (Approx): seq=16, dim=256
    # Level 1 (Detail): seq=32, dim=128
    # Level 2 (Detail): seq=64, dim=128
    level_shapes = [
        (batch_size, 16, 256),
        (batch_size, 32, 128),
        (batch_size, 64, 128)
    ]
    time_embed_dim = 64
    level_embed_dims = [shape[2] for shape in level_shapes]
    
    print("Instantiating CrossLevelAttention (Pyramidal)...")
    attn = CrossLevelAttention(
        level_embed_dims=level_embed_dims,
        common_dim=128,
        num_heads=4,
        dropout=0.1,
        time_embed_dim=time_embed_dim,
        attention_mode="cross_only"
    )
    
    level_embeddings = [torch.randn(shape) for shape in level_shapes]
    time_embed = torch.randn(batch_size, time_embed_dim)
    
    print("Running Pyramidal Forward Pass...")
    outputs = attn(level_embeddings, time_embed)
    
    assert len(outputs) == len(level_embeddings), "Output list length mismatch"
    for i, (out, inp) in enumerate(zip(outputs, level_embeddings)):
        assert out.shape == inp.shape, f"Level {i} shape mismatch: expected {inp.shape}, got {out.shape}"
        print(f"Level {i} temporal shape preserved: {out.shape}")
        
    print("\nRunning get_cross_level_attention_weights Hook...")
    attn_matrix = attn.get_cross_level_attention_weights(level_embeddings, time_embed)
    assert attn_matrix.shape == (3, 3), f"Attention matrix shape mismatch: expected (3, 3), got {attn_matrix.shape}"
    print(f"\nSUCCESS: Pyramidal Attention implemented and verified correctly. Matrix Shape: {attn_matrix.shape}")

if __name__ == "__main__":
    test_pyramidal_attention()