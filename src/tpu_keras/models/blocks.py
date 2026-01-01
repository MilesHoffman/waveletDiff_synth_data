import keras
from keras import ops
import jax.numpy as jnp
import math

class TimeEmbedding(keras.layers.Layer):
    """Sinusoidal time embedding followed by MLP."""
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mlp = keras.Sequential([
            keras.layers.Dense(dim * 4, activation='gelu'),
            keras.layers.Dense(dim)
        ])

    def call(self, time):
        # time: (batch,)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = ops.exp(jnp.arange(half_dim, dtype='float32') * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = ops.concatenate([ops.sin(embeddings), ops.cos(embeddings)], axis=-1)
        return self.mlp(embeddings)

class LevelEmbedding(keras.layers.Layer):
    """Learned embedding for wavelet levels."""
    def __init__(self, num_levels, dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding = keras.layers.Embedding(num_levels, dim)

    def call(self, level_idx):
        # level_idx: (batch,) or (1,)
        return self.embedding(level_idx)

class AdaLayerNorm(keras.layers.Layer):
    """Adaptive Layer Normalization conditioned on time and level embedding."""
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[0][-1]
        self.layernorm = keras.layers.LayerNormalization(epsilon=self.epsilon, center=False, scale=False)
        self.scale_proj = keras.layers.Dense(dim)
        self.shift_proj = keras.layers.Dense(dim)
        super().build(input_shape)

    def call(self, inputs):
        x, emb = inputs # emb could be time_emb or time_emb + level_emb
        norm_x = self.layernorm(x)
        scale = self.scale_proj(emb)
        shift = self.shift_proj(emb)
        scale = ops.expand_dims(scale, axis=1)
        shift = ops.expand_dims(shift, axis=1)
        return (1.0 + scale) * norm_x + shift

class WaveletTransformerBlock(keras.layers.Layer):
    """Transformer block with AdaLayerNorm for conditioning."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        
        self.norm1 = AdaLayerNorm()
        self.attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        
        self.norm2 = AdaLayerNorm()
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='gelu'),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(dim),
            keras.layers.Dropout(dropout)
        ])
        
        self.res_dropout = keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        # inputs can be [x, emb] or [x, emb, mask]
        x, emb = inputs[0], inputs[1]
        mask = inputs[2] if len(inputs) > 2 else None
        
        # Attention path
        x_norm1 = self.norm1([x, emb])
        attn_out = self.attn(x_norm1, x_norm1, attention_mask=mask, training=training)
        x = x + self.res_dropout(attn_out, training=training)
        
        # MLP path
        x_norm2 = self.norm2([x, emb])
        mlp_out = self.mlp(x_norm2, training=training)
        x = x + mlp_out
        
        return x
