"""
High-level interface for Keras/TPU training.
Mirrors src.train.trainer API for consistency.
"""

import os
import keras
import jax
from . import data as kdata
from . import transformer as ktrans
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

def setup_environment():
    """Sets up JAX backend and hardware policies."""
    os.environ["KERAS_BACKEND"] = "jax"
    
    # Check Hardware
    try:
        devices = jax.devices()
        device_type = devices[0].platform.upper()
        print(f"🚀 Hardware Detected: {device_type} (Count: {len(devices)})")
        
        if device_type == 'TPU':
            keras.mixed_precision.set_global_policy("mixed_bfloat16")
            print("✅ Optimization: Precision set to 'mixed_bfloat16' (TPU Native)")
        elif device_type == 'GPU':
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("✅ Optimization: Precision set to 'mixed_float16'")
    except Exception as e:
        print(f"Hardware detection failed: {e}")

def get_dataloader(data_path, batch_size, seq_len):
    """
    Returns (ds, info). 
    """
    return kdata.load_dataset(data_path, batch_size, seq_len)

def init_model(info, config):
    """
    Initializes and Compiles the Keras Model.
    """
    # 1. Init
    model_config = {
        'embed_dim': config['EMBED_DIM'],
        'num_heads': config['NUM_HEADS'],
        'num_layers': config['NUM_LAYERS'],
        'dropout': config['DROPOUT'],
        'time_embed_dim': config.get('TIME_EMBED_DIM', 128),
        'prediction_target': config.get('PREDICTION_TARGET', 'noise'),
        'use_cross_level_attention': config.get('USE_CROSS_LEVEL_ATTENTION', True)
    }
    
    model = ktrans.WaveletDiffusionTransformer(info, model_config)
    
    # 2. Compile (XLA)
    optimizer = keras.optimizers.AdamW(
        learning_rate=config['LEARNING_RATE'],
        weight_decay=config.get('WEIGHT_DECAY', 1e-5),
        clipnorm=1.0 # Soft global clipping
    )
    
    model.compile(optimizer=optimizer, jit_compile=True)
    
    # 3. Build (Optional, for summary)
    # We can pass a dummy batch if we want to print summary immediately, 
    # but lazy build on first fit is standard in Keras 3.
    
    return model

def train_loop(model, dataset, config):
    """
    Runs model.fit() with callbacks.
    """
    callbacks = []
    
    # Checkpoint
    if config.get('CHECKPOINT_DIR'):
        os.makedirs(config['CHECKPOINT_DIR'], exist_ok=True)
        ckpt_path = os.path.join(config['CHECKPOINT_DIR'], "weights_{epoch:02d}.weights.h5")
        callbacks.append(ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=False,
            save_weights_only=True,
            verbose=1
        ))
        
    # LR Scheduler
    callbacks.append(ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ))
    
    # Logging
    callbacks.append(CSVLogger("training_log.csv"))
    
    # Train
    # Note: dataset is assumed to be finite (no .repeat()) so steps_per_epoch is None by default
    # unless config overrides.
    
    history = model.fit(
        dataset,
        epochs=config['EPOCHS'],
        steps_per_epoch=config.get('STEPS_PER_EPOCH'), # Can be None
        callbacks=callbacks,
        verbose=1
    )
    
    return history
