"""
Core Keras 3 Layers for WaveletDiff.
"""

import math
import numpy as np
import keras
from keras import layers, ops

class TimeEmbedding(layers.Layer):
    """Sinusoidal time embedding for diffusion timesteps."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.half_dim = embed_dim // 2
        
        # Precompute frequency term
        # exp(-log(10000) / (half_dim - 1) * i)
        emb = math.log(10000) / (self.half_dim - 1)
        self.emb_freq = ops.exp(ops.arange(self.half_dim, dtype="float32") * -emb)
        
        # MLP for processing
        self.mlp = keras.Sequential([
            layers.Dense(embed_dim * 4, activation="silu"),
            layers.Dense(embed_dim)
        ])

    def call(self, t):
        # t: [batch_size], range [0, 1]
        t = ops.cast(t, "float32") * 1000.0
        t = ops.expand_dims(t, -1) # [B, 1]
        
        # Sinusoidal encoding
        # emb: [B, half_dim]
        emb = t * ops.expand_dims(self.emb_freq, 0)
        emb = ops.concatenate([ops.sin(emb), ops.cos(emb)], axis=-1)
        
        return self.mlp(emb)


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding to match PyTorch implementation."""
    def __init__(self, embed_dim, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Precompute pe matrix
        # pe: [max_len, embed_dim]
        pe = np.zeros((max_len, embed_dim), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2, dtype=np.float32) * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Register as non-trainable weight (Buffer)
        # This ensures compatibility with JAX/TF/Torch backends
        self.pe = self.add_weight(
            name="pe_buffer",
            shape=(max_len, embed_dim),
            initializer=keras.initializers.Constant(pe),
            trainable=False,
            dtype="float32"
        )

    def call(self, x):
        # x: [B, SeqLen, EmbedDim]
        seq_len = ops.shape(x)[1]
        
        # Slice pe to seq_len
        # Expand dims to broadcast batch: [1, SeqLen, EmbedDim]
        # self.pe is a Variable, slicing works
        pe_slice = self.pe[:seq_len, :]
        pe_slice = ops.expand_dims(pe_slice, 0)
        
        return x + pe_slice


class AdaLayerNorm(layers.Layer):
    """Adaptive Layer Normalization conditioned on time embedding."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.norm = layers.LayerNormalization(axis=-1, center=False, scale=False)
        
        # Predict scale and shift from time_embed
        self.ada_lin = layers.Dense(2 * embed_dim)

    def build(self, input_shape):
        # Initialize ada_lin to identity-like behavior
        # Dense creates kernel and bias.
        # We want bias to be [1...1, 0...0] (scale=1, shift=0).
        # We want kernel to be near zero.
        pass

    def call(self, x, time_embed):
        # x: [B, ..., embed_dim]
        # time_embed: [B, time_embed_dim]
        
        x_norm = self.norm(x)
        
        ada_params = self.ada_lin(time_embed) # [B, 2*embed_dim]
        scale, shift = ops.split(ada_params, 2, axis=-1)
        
        # Expand dims for broadcasting
        # Handle arbitrary rank input x (usually [B, S, D] or [B, D] if pooled)
        # scale/shift are [B, D]. 
        # We need to expand to match x's rank.
        
        # If x is Rank 3 [B, S, D], we need [B, 1, D]
        # If x is Rank 2 [B, D], we keep [B, D]
        
        # Robust expansion:
        while len(scale.shape) < len(x.shape):
             scale = ops.expand_dims(scale, 1)
             shift = ops.expand_dims(shift, 1)
        
        return scale * x_norm + shift


class WaveletTransformerBlock(layers.Layer):
    """Transformer block with time conditioning."""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = AdaLayerNorm(embed_dim)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.dropout1 = layers.Dropout(dropout)
        
        self.norm2 = AdaLayerNorm(embed_dim)
        # PyTorch impl uses: Linear -> GELU -> Dropout -> Linear -> Dropout
        # Our Keras layers usually default to relu, so explicit gelu is good.
        self.mlp = keras.Sequential([
            layers.Dense(embed_dim * 4, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout) 
        ])

    def call(self, x, time_embed, training=None):
        # x: [B, S, D]
        # time_embed: [B, T_D]
        
        # Self Attention
        # Note: PyTorch impl applies Norm BEFORE attention (Pre-Norm)
        # x_norm1 = self.norm1(x, time_embed)
        # attn_out = self.attn(x_norm1, x_norm1, x_norm1)
        # x = x + self.dropout(attn_out)
        
        x_norm1 = self.norm1(x, time_embed)
        # Pass use_causal_mask=False? It's bidirectional for coefficients usually? 
        # Source uses nn.MultiheadAttention default (bidirectional).
        attn_out = self.attn(x_norm1, x_norm1, training=training)
        x = x + self.dropout1(attn_out, training=training)
        
        # MLP
        x_norm2 = self.norm2(x, time_embed)
        mlp_out = self.mlp(x_norm2, training=training)
        x = x + mlp_out
        
        return x
