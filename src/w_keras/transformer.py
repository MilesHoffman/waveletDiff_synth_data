"""
WaveletDiffusionTransformer (Keras 3).
"""

import keras
from keras import layers, ops, Model
from .layers import TimeEmbedding, WaveletTransformerBlock
from .attention import CrossLevelAttention

class WaveletLevelTransformer(layers.Layer):
    """Transformer for a single wavelet level."""
    def __init__(self, level_dim, num_features, embed_dim=128, num_heads=8, num_layers=4, 
                 time_embed_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_proj = layers.Dense(embed_dim)
        
        # Positional Encoding (Fixed Sinusoidal)
        # In Keras we can use a standard Embedding layer or compute on fly. 
        # Since level_dim is fixed per level, we can learn it or use fixed.
        # PyTorch uses fixed. Let's use Learnabale for Keras simplicity or stick to fixed?
        # Let's use Learnable "Embedding" style for simplicity and power.
        self.pos_emb = layers.Embedding(input_dim=level_dim, output_dim=embed_dim)
        
        self.blocks = [
            WaveletTransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ]
        
        self.output_proj = layers.Dense(num_features)
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, time_embed, return_embeddings=False, training=None):
        # x: [B, level_dim, num_features]
        b, s, f = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        
        # Embed
        x = self.input_proj(x) # [B, S, E]
        
        # Pos Emb
        positions = ops.arange(0, s, dtype="int32")
        # Broadcast positions to batch? No, Embedding layer handles it if we pass indices or add.
        # Better: x + pos_emb(positions)
        pe = self.pos_emb(positions) # [S, E]
        x = x + ops.expand_dims(pe, 0)
        
        x = self.dropout(x, training=training)
        
        for block in self.blocks:
            x = block(x, time_embed, training=training)
            
        if return_embeddings:
            return x
            
        return self.output_proj(x)
        
    def final_projection(self, x):
        return self.output_proj(x)


class WaveletDiffusionTransformer(Model):
    """
    Main Model.
    Overrides train_step for custom diffusion logic.
    """
    def __init__(self, data_config, model_config, **kwargs):
        super().__init__(**kwargs)
        
        # Unpack Configs
        self.level_dims = data_config['level_dims'] # List of ints
        self.num_features = data_config['n_features']
        self.level_starts = data_config['level_start_indices']
        
        embed_dim = model_config.get('embed_dim', 256)
        num_heads = model_config.get('num_heads', 8)
        num_layers = model_config.get('num_layers', 8)
        time_embed_dim = model_config.get('time_embed_dim', 128)
        dropout = model_config.get('dropout', 0.1)
        self.prediction_target = model_config.get('prediction_target', 'noise')
        
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Level Transformers
        self.level_transformers = []
        embed_dims_list = []
        for i, dim in enumerate(self.level_dims):
            # Level 0 gets double capacity
            e_dim = embed_dim * 2 if i == 0 else embed_dim
            n_lay = num_layers + 2 if i == 0 else num_layers
            
            sub_model = WaveletLevelTransformer(
                level_dim=dim,
                num_features=self.num_features,
                embed_dim=e_dim,
                num_heads=num_heads,
                num_layers=n_lay,
                time_embed_dim=time_embed_dim,
                dropout=dropout
            )
            self.level_transformers.append(sub_model)
            embed_dims_list.append(e_dim)
            
        # Cross Attention
        self.use_cross = model_config.get('use_cross_level_attention', True)
        if self.use_cross:
            self.cross_attn = CrossLevelAttention(
                level_embed_dims=embed_dims_list,
                num_heads=num_heads,
                time_embed_dim=time_embed_dim,
                attention_mode="cross_only" # or all_to_all
            )
            
        # Noise Schedule (Fixed buffers)
        # We need to register these buffers or store them. 
        # In Keras, we can store as non-trainable weights.
        self.T = 1000
        beta = ops.linspace(0.0001, 0.02, self.T)
        alpha = 1.0 - beta
        alpha_bar = ops.cumprod(alpha, axis=0)
        
        self.alpha_bar = self.add_weight(
            name="alpha_bar",
            shape=(self.T,),
            initializer="zeros",
            trainable=False
        )
        self.alpha_bar.assign(alpha_bar)

    def call(self, inputs):
        # inputs is tuple: (x, t) normally.
        # But Keras .fit passes one x.
        # We'll split it or expect inputs structure if used manually.
        # For training, we override train_step so this call() is for INFERENCE mainly.
        # Let's assume call receives (x_t, t_norm)
        
        x_all, t_norm = inputs
        
        # Time Embed
        time_embed = self.time_embedding(t_norm)
        
        # Split x_all into levels
        # x_all: [B, TotalCoeffs, Features]
        
        level_outputs = []
        
        if self.use_cross:
            level_embeddings = []
            
            # 1. Get Embeddings
            for i, (start, dim) in enumerate(zip(self.level_starts, self.level_dims)):
                level_data = x_all[:, start:start+dim, :]
                emb = self.level_transformers[i].call(level_data, time_embed, return_embeddings=True)
                level_embeddings.append(emb)
                
            # 2. Cross Attn
            cross_embeddings = self.cross_attn(level_embeddings, time_embed)
            
            # 3. Final Projection
            for i, emb in enumerate(cross_embeddings):
                out = self.level_transformers[i].final_projection(emb)
                level_outputs.append(out)
                
        else:
            for i, (start, dim) in enumerate(zip(self.level_starts, self.level_dims)):
                level_data = x_all[:, start:start+dim, :]
                out = self.level_transformers[i].call(level_data, time_embed)
                level_outputs.append(out)
                
        # Concat
        return ops.concatenate(level_outputs, axis=1)

    def train_step(self, data):
        # Data is just x_0 (batch of wavelet coeffs)
        x_0 = data
        batch_size = ops.shape(x_0)[0]
        
        # 1. Sample Time
        t = keras.random.randint(minval=0, maxval=self.T, shape=(batch_size,))
        t_norm = ops.cast(t, "float32") / float(self.T)
        
        # 2. Add Noise (Forward Process)
        noise = keras.random.normal(shape=ops.shape(x_0))
        
        # Get alpha_bar_t
        # Gather is tricky for batch. 
        # t is [B]. We want alpha_bar[t].
        alpha_bar_t = ops.take(self.alpha_bar, t, axis=0) # [B]
        # Reshape for broadcast: [B, 1, 1]
        alpha_bar_t = ops.reshape(alpha_bar_t, (batch_size, 1, 1))
        
        sqrt_alpha = ops.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha = ops.sqrt(1.0 - alpha_bar_t)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        # 3. Predict
        with tf.GradientTape() as tape:
            # Prediction
            pred_noise = self((x_t, t_norm), training=True)
            
            # Loss (MSE)
            # Custom logic: weighted MSE? Or simple MSE for now.
            if self.prediction_target == 'noise':
                target = noise
            else:
                target = x_0
                
            loss = ops.mean(ops.square(target - pred_noise))
            # We can add energy terms here later
            
        # 4. Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # 5. Update
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {"loss": loss}
