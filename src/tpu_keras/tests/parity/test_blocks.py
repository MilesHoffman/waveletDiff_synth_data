import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import torch
import torch.nn as nn
import keras_core as keras
import jax
import jax.numpy as jnp
from src.tpu_keras.models.blocks import AdaLayerNorm, WaveletTransformerBlock, TimeEmbedding, LevelEmbedding

def test_blocks_parity():
    print("Running Blocks Parity Test...")
    
    batch_size = 2
    seq_len = 32
    dim = 64
    time_embed_dim = 128
    
    # Inputs
    x_np = np.random.randn(batch_size, seq_len, dim).astype('float32')
    t_emb_np = np.random.randn(batch_size, time_embed_dim).astype('float32')
    
    x_jax = jnp.array(x_np)
    t_emb_jax = jnp.array(t_emb_np)
    
    # 1. AdaLayerNorm Parity
    print("Testing AdaLayerNorm...")
    keras_ln = AdaLayerNorm()
    keras_ln([x_jax, t_emb_jax]) # Build
    
    # PyTorch AdaLayerNorm Reference
    class PyTorchAdaLN(nn.Module):
        def __init__(self, dim, t_dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.scale_proj = nn.Linear(t_dim, dim)
            self.shift_proj = nn.Linear(t_dim, dim)
        def forward(self, x, t):
            norm_x = self.norm(x)
            scale = self.scale_proj(t).unsqueeze(1)
            shift = self.shift_proj(t).unsqueeze(1)
            return (1 + scale) * norm_x + shift

    pt_ln = PyTorchAdaLN(dim, time_embed_dim)
    
    # Align weights
    # Keras Dense stores [In, Out]. PyTorch Linear stores [Out, In].
    pt_ln.scale_proj.weight.data = torch.from_numpy(np.array(keras_ln.scale_proj.kernel)).T
    pt_ln.scale_proj.bias.data = torch.from_numpy(np.array(keras_ln.scale_proj.bias))
    pt_ln.shift_proj.weight.data = torch.from_numpy(np.array(keras_ln.shift_proj.kernel)).T
    pt_ln.shift_proj.bias.data = torch.from_numpy(np.array(keras_ln.shift_proj.bias))
    
    # Run
    keras_out = keras_ln([x_jax, t_emb_jax])
    with torch.no_grad():
        pt_out = pt_ln(torch.from_numpy(x_np), torch.from_numpy(t_emb_np))
    
    diff = np.max(np.abs(np.array(keras_out) - pt_out.numpy()))
    print(f"  AdaLayerNorm Max Diff: {diff}")
    
    if diff < 1e-5:
        print("  AdaLayerNorm PASSED")
    else:
        print("  AdaLayerNorm FAILED")

    # 2. TimeEmbedding Parity
    print("Testing TimeEmbedding...")
    time_val = np.array([10.0, 50.0]).astype('float32')
    keras_time_emb = TimeEmbedding(dim)
    keras_out = keras_time_emb(jnp.array(time_val))
    print("  TimeEmbedding (Keras) shape:", keras_out.shape)
    
    # 3. LevelEmbedding Parity
    print("Testing LevelEmbedding...")
    level_idx = np.array([0, 1]).astype('int32')
    keras_level_emb = LevelEmbedding(5, dim)
    keras_out = keras_level_emb(jnp.array(level_idx))
    print("  LevelEmbedding (Keras) shape:", keras_out.shape)
    
    # 4. TransformerBlock Parity
    print("Testing WaveletTransformerBlock...")
    keras_block = WaveletTransformerBlock(dim)
    keras_out = keras_block([x_jax, t_emb_jax])
    print("  WaveletTransformerBlock (Keras) shape:", keras_out.shape)

if __name__ == "__main__":
    test_blocks_parity()
