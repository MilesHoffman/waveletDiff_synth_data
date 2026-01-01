import keras
from keras import ops
import jax.numpy as jnp
from src.tpu_keras.models.blocks import TimeEmbedding, LevelEmbedding
from src.tpu_keras.models.attention import CrossLevelAttention, WaveletLevelTransformer

class WaveletDiffusionTransformer(keras.Model):
    """
    Keras 3 implementation of the Wavelet Diffusion Transformer.
    Optimized for TPU/XLA.
    """
    def __init__(self, 
                 input_dim, 
                 model_dim=128, 
                 num_levels=3, 
                 num_layers_per_level=2,
                 num_heads=8,
                 dropout=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_levels = num_levels
        
        # 1. Embeddings
        self.time_embed = TimeEmbedding(model_dim)
        self.level_embed = LevelEmbedding(num_levels + 1, model_dim)
        
        # 2. Input Projections (mapping wavelet coeffs to model_dim)
        self.input_projections = [
            keras.layers.Dense(model_dim) for _ in range(num_levels + 1)
        ]
        
        # 3. Level-specific Transformers
        self.level_transformers = [
            WaveletLevelTransformer(model_dim, num_layers=num_layers_per_level, num_heads=num_heads)
            for _ in range(num_levels + 1)
        ]
        
        # 4. Cross-Level Attention
        self.cross_attention = CrossLevelAttention(
            level_embed_dims=[model_dim] * (num_levels + 1),
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 5. Output Projections (mapping back to input_dim)
        self.output_projections = [
            keras.layers.Dense(input_dim) for _ in range(num_levels + 1)
        ]

    def call(self, inputs, training=False):
        """
        Args:
            inputs: [level_coeffs_list, t]
                level_coeffs_list: List of tensors [A_L, D_L, D_{L-1}, ..., D_1]
                t: (batch,)
        """
        coeffs, t = inputs[0], inputs[1]
        
        # a. Conditionings
        t_emb = self.time_embed(t)
        
        # b. Embed each level and apply level-specific transformers
        level_hiddens = []
        for i in range(len(coeffs)):
            x_i = self.input_projections[i](coeffs[i])
            l_emb = self.level_embed(ops.cast(i, 'int32'))
            # Condition on time + level metadata? 
            # In source, level embeddings are often added or used in AdaLN
            # We'll add level embedding to the time embedding for conditioning
            cond_emb = t_emb + l_emb
            
            x_i = self.level_transformers[i]([x_i, cond_emb], training=training)
            level_hiddens.append(x_i)
            
        # c. Cross-Level Attention
        level_hiddens = self.cross_attention([level_hiddens, t_emb], training=training)
        
        # d. Local refinement (Post-attention transformer blocks?) 
        # Source often has more blocks here. For now, we project to output.
        
        outputs = []
        for i in range(len(coeffs)):
            out_i = self.output_projections[i](level_hiddens[i])
            outputs.append(out_i)
            
        return outputs
