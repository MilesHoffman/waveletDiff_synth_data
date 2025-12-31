
# Dry run verification script
import sys
import os

# Set backend
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORM_NAME"] = "cpu" # Force CPU for dry run local check if no TPU

import keras
import numpy as np
from src.w_keras import transformer as ktrans

# Mock Config
data_config = {
    'level_dims': [12, 12, 12],
    'n_features': 5,
    'level_start_indices': [0, 12, 24]
}

model_config = {
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 8,
    'time_embed_dim': 128,
    'dropout': 0.1,
    'prediction_target': 'noise',
    'use_cross_level_attention': True,
    'energy_weight': 0.05
}

# 1. Instantiate
print("Instantiating model...")
model = ktrans.WaveletDiffusionTransformer(data_config, model_config)

# 2. Build
print("Building model...")
# Input: x_0 [B, 36, 5], t [B]
B = 2
x_0 = np.random.randn(B, 36, 5).astype(np.float32)
t = np.random.rand(B).astype(np.float32)

# Call (Inference)
out = model((x_0, t))
print(f"Output shape: {out.shape}")
assert out.shape == (B, 36, 5)

# 3. Compile
print("Compiling model...")
model.compile(optimizer='adam', loss=model.loss_fn)

# 4. Train Step (Dry Run)
print("Running train_step...")
loss = model.train_step(x_0)
print(f"Loss: {loss}")

print("Verification Successful!")
