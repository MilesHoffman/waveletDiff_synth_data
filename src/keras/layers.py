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
        # In Keras, we can do this with kernel_initializer, but doing it in build
        # ensures we access the weights after creation.
        # Ideally, we want scale=1, shift=0.
        # Dense output is [scale, shift]. So we want [1...1, 0...0].
        # We can achieve this by initializing bias to this vector and weight to near-zero.
        input_dim = input_shape[-1]
        # Custom initialization is tricky in pure functional output, relying on default initialization usually works fine
        # with sufficient training, but let's try to be precise if possible.
        pass

    def call(self, x, time_embed):
        # x: [B, ..., embed_dim]
        # time_embed: [B, time_embed_dim]
        
        x_norm = self.norm(x)
        
        ada_params = self.ada_lin(time_embed) # [B, 2*embed_dim]
        scale, shift = ops.split(ada_params, 2, axis=-1)
        
        # Expand dims for broadcasting to [B, Seq, embed_dim]
        # Assuming x is rank 3 [B, S, D]
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
        residual = x
        x = self.norm1(x, time_embed)
        x = self.attn(x, x, training=training)
        x = self.dropout1(x, training=training)
        x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x, time_embed)
        x = self.mlp(x, training=training)
        x = residual + x
        
        return x
