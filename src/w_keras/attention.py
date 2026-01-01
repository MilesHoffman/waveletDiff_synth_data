"""
Cross-Level Attention Mechanism (Keras 3).
"""

import keras
from keras import layers, ops
from .layers import AdaLayerNorm

class CrossLevelAttention(layers.Layer):
    """
    Keras 3 implementation of Cross-Level Attention.
    Supports 'all_to_all' and 'cross_only' modes.
    """
    def __init__(self, level_embed_dims, common_dim=None, num_heads=8, dropout=0.1, 
                 time_embed_dim=64, attention_mode="all_to_all", **kwargs):
        super().__init__(**kwargs)
        self.level_embed_dims = level_embed_dims
        self.num_levels = len(level_embed_dims)
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.attention_mode = attention_mode
        self.common_dim = common_dim if common_dim else max(level_embed_dims)
        
        # 1. Aggregators
        self.aggregators = []
        for dim in level_embed_dims:
            agg = keras.Sequential([
                layers.Dense(dim // 2, activation="gelu"),
                layers.Dense(1),
                layers.Softmax(axis=1) # Attention weights over sequence length
            ])
            self.aggregators.append(agg)
            
        # 2. Projections to Common Dim
        self.projections = [layers.Dense(self.common_dim) for _ in level_embed_dims]
        
        # 3. Attention Mechanism
        if attention_mode == "all_to_all":
            self.cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.common_dim // num_heads, dropout=dropout)
        else:
            self.cross_attn_layers = [
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.common_dim // num_heads, dropout=dropout)
                for _ in range(self.num_levels)
            ]
            
        # 4. Expanders
        self.expanders = [layers.Dense(dim) for dim in level_embed_dims]
        
        # 5. AdaLayerNorm & Gates
        self.cross_norm = [AdaLayerNorm(dim) for dim in level_embed_dims]
        self.gates = []
        for dim in level_embed_dims:
            gate = keras.Sequential([
                layers.Dense(dim, activation="sigmoid")
            ])
            self.gates.append(gate)
            
        # Level Positional Embeddings
        self.level_pos_emb = self.add_weight(
            shape=(self.num_levels, self.common_dim),
            initializer="random_normal",
            trainable=True,
            name="level_pos_emb"
        )

    def call(self, level_embeddings, time_embed, training=None):
        # level_embeddings: List of [B, S_i, D_i]
        
        # Step 1: Aggregate to Level Representations [B, CommonDim]
        level_reprs = []
        for i, emb in enumerate(level_embeddings):
            # Attention pooling
            # weights: [B, S, 1]
            attn_weights = self.aggregators[i](emb)
            # Weighted sum: [B, D_i]
            pooled = ops.sum(emb * attn_weights, axis=1)
            
            # Project
            repr_vector = self.projections[i](pooled)
            
            # Add Level Pos Emb
            repr_vector = repr_vector + self.level_pos_emb[i]
            level_reprs.append(repr_vector)
            
        # Stack: [B, NumLevels, CommonDim]
        level_stack = ops.stack(level_reprs, axis=1)
        
        # Step 2: Cross Attention
        if self.attention_mode == "all_to_all":
            # Self-attention on the level stack
            cross_attended = self.cross_attn(level_stack, level_stack, training=training)
            # Unstack back to list of [B, CommonDim]
            # Keras ops.unstack not strictly standard in all backends, split is safer
            cross_attended_list = ops.split(cross_attended, self.num_levels, axis=1)
            cross_attended_list = [ops.squeeze(x, axis=1) for x in cross_attended_list]
            
        else: # cross_only
            cross_attended_list = []
            for i in range(self.num_levels):
                query = ops.expand_dims(level_stack[:, i, :], 1) # [B, 1, D]
                
                # Key/Value is everything ELSE
                # This is tricky with static indexing in JAX if num_levels is small relying on python loops is fine
                others = []
                for j in range(self.num_levels):
                    if i != j:
                        others.append(level_stack[:, j, :])
                
                if not others: # Single level case
                    cross_attended_list.append(level_stack[:, i, :])
                    continue
                    
                key_value = ops.stack(others, axis=1) # [B, NumLevels-1, D]
                
                out = self.cross_attn_layers[i](query, key_value, training=training)
                cross_attended_list.append(ops.squeeze(out, axis=1))

        # Step 3: Expand and Gate
        output_embeddings = []
        for i, (attended_repr, original_emb) in enumerate(zip(cross_attended_list, level_embeddings)):
            # Expand: [B, D_i]
            expanded = self.expanders[i](attended_repr)
            
            # Broadcast to sequence length: [B, S_i, D_i]
            # Broadcast to sequence length using Tile
            seq_len = ops.shape(original_emb)[1]
            expanded = ops.expand_dims(expanded, 1)
            # Tile: [1 (Batch), seq_len, 1 (Dim)]
            expanded = ops.tile(expanded, [1, seq_len, 1])
            
            # AdaNorm
            normed = self.cross_norm[i](expanded, time_embed)
            
            # Gate Input
            # Gate needs concatenation of original and time. 
            # PyTorch implementation: Linear(embed_dim + time_dim) -> embed_dim
            # But the gate in __init__ is defined as Dense(dim). 
            # Wait, the PyTorch code says:
            # gate = nn.Sequential(nn.Linear(embed_dim + time_embed_dim, embed_dim), nn.Sigmoid())
            # My init just did Dense(dim). I need to fix logic or inputs.
            # Let's verify input. 
            # "gate_input = torch.cat([original_emb, time_embed_expanded], dim=-1)"
            # So input dim is D + T.
            
            time_expanded = ops.expand_dims(time_embed, 1)
            # Tile: [1 (Batch), seq_len, 1 (TimeDim)]
            time_expanded = ops.tile(time_expanded, [1, seq_len, 1])
            
            gate_in = ops.concatenate([original_emb, time_expanded], axis=-1)
            gate_val = self.gates[i](gate_in)
            
            out = original_emb + gate_val * normed
            output_embeddings.append(out)
            
        return output_embeddings
