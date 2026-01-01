import keras
from keras import ops
import jax.numpy as jnp
from src.tpu_keras.models.blocks import WaveletTransformerBlock, AdaLayerNorm

class CrossLevelAttention(keras.layers.Layer):
    """
    Cross-level attention mechanism for multi-scale wavelet coefficients.
    Aggregates coefficients into level-wise tokens, applies attention, and expands.
    """
    def __init__(self, level_embed_dims, common_dim=None, num_heads=8, dropout=0.1, 
                 attention_mode="all_to_all", **kwargs):
        super().__init__(**kwargs)
        self.level_embed_dims = level_embed_dims
        self.num_levels = len(level_embed_dims)
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_mode = attention_mode
        self.common_dim = common_dim if common_dim is not None else max(level_embed_dims)

    def build(self, input_shape):
        # input_shape: list of shapes for each level's embeddings
        self.level_aggregators = []
        self.level_projections = []
        self.level_expanders = []
        self.cross_norms = []
        self.cross_level_gates = []
        
        for embed_dim in self.level_embed_dims:
            # Aggregator (Attention Pooling)
            agg = keras.Sequential([
                keras.layers.Dense(embed_dim // 2, activation='gelu'),
                keras.layers.Dense(1)
            ])
            self.level_aggregators.append(agg)
            
            # Projections
            self.level_projections.append(keras.layers.Dense(self.common_dim))
            self.level_expanders.append(keras.layers.Dense(embed_dim))
            
            # AdaLayerNorm and Gates
            self.cross_norms.append(AdaLayerNorm())
            self.cross_level_gates.append(keras.Sequential([
                keras.layers.Dense(embed_dim, activation='sigmoid')
            ]))

        self.cross_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.common_dim // self.num_heads, dropout=self.dropout
        )
        
        self.level_pos_emb = self.add_weight(
            shape=(self.num_levels, self.common_dim),
            initializer='truncated_normal',
            name='level_pos_emb'
        )
        
        super().build(input_shape)

    def call(self, inputs, training=False):
        # inputs: [level_embeddings_list, time_emb]
        level_embeddings, time_emb = inputs[0], inputs[1]
        
        # Step 1: Aggregate
        level_reprs = []
        for i in range(self.num_levels):
            x_level = level_embeddings[i] # (batch, seq, dim)
            attn_weights = self.level_aggregators[i](x_level) # (batch, seq, 1)
            attn_weights = ops.softmax(attn_weights, axis=1)
            
            level_repr = ops.sum(x_level * attn_weights, axis=1) # (batch, dim)
            level_repr = self.level_projections[i](level_repr) # (batch, common_dim)
            level_repr = level_repr + self.level_pos_emb[i]
            level_reprs.append(level_repr)
            
        # Step 2: Cross-Attention
        # (batch, num_levels, common_dim)
        level_stack = ops.stack(level_reprs, axis=1)
        
        # JAX MultiHeadAttention expects (batch, query_seq, dim)
        attended_stack = self.cross_attention(level_stack, level_stack, training=training)
        
        # Step 3: Expand and Gate
        outputs = []
        for i in range(self.num_levels):
            # attended_stack[:, i, :] -> (batch, common_dim)
            h_cross = attended_stack[:, i, :]
            h_expanded = self.level_expanders[i](h_cross) # (batch, embed_dim)
            h_expanded = ops.expand_dims(h_expanded, axis=1) # Broadcast to seq
            
            # AdaLayerNorm
            h_norm = self.cross_norms[i]([h_expanded, time_emb])
            
            # Gating
            gate_in = ops.concatenate([level_embeddings[i], ops.expand_dims(time_emb, 1) * ops.ones_like(level_embeddings[i])], axis=-1)
            # Simplified gate: project concat [x, t]
            gate = self.cross_level_gates[i](gate_in)
            
            outputs.append(level_embeddings[i] + gate * h_norm)
            
        return outputs

class WaveletLevelTransformer(keras.layers.Layer):
    """Transformer for a single wavelet level."""
    def __init__(self, dim, num_layers=4, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [WaveletTransformerBlock(dim, num_heads) for _ in range(num_layers)]

    def call(self, inputs, training=False):
        x, time_emb = inputs
        for block in self.blocks:
            x = block([x, time_emb], training=training)
        return x
