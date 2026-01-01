
import os
import sys
import traceback
import numpy as np

# Set backend before importing keras
os.environ['KERAS_BACKEND'] = 'jax'

import keras
import tensorflow as tf

# Add src to path so we can import w_keras
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from w_keras import trainer_interface as trainer
from w_keras import transformer
from w_keras import loss

def main():
    # Redirect output to file to avoid encoding issues
    sys.stdout = open('verify_output.log', 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    
    print("Verifying WaveletDiff Keras Build...")
    print(f"Keras version: {keras.__version__}")
    print(f"Backend: {keras.backend.backend()}")

    # Mock Config
    BATCH_SIZE = 4
    SEQ_LEN = 24
    CONFIG = {
        'EMBED_DIM': 32, # Reduced for speed
        'NUM_HEADS': 2,
        'NUM_LAYERS': 2,
        'DROPOUT': 0.1,
        'TIME_EMBED_DIM': 16,
        'BATCH_SIZE': BATCH_SIZE,
        'PREDICTION_TARGET': 'noise',
        'USE_CROSS_LEVEL_ATTENTION': True,
        'LEARNING_RATE': 1e-4,
        'WEIGHT_DECAY': 1e-5,
        'EPOCHS': 1,
        'STEPS_PER_EPOCH': 10,
        'CHECKPOINT_DIR': './tmp_checkpoints',
        'ENERGY_WEIGHT': 0.05,
        'WAVELET_TYPE': 'db2'
    }

    # Mock Info (based on db2 and seq_len 24)
    # db2 with mode='periodization' and levels... 
    # Let's assume a generic level structure for verification.
    # If 24 -> level dims might be like [4, 4, 8, 12] or similar depending on decomposition.
    # But init_model needs 'level_dims'.
    # For db2 and len 24, typically max level is 2 or 3.
    # Let's say inputs are split into 3 levels with dims [6, 6, 12] summing to 24.
    # Actually, let's use the code's own logic if possible, or just mock it.
    # Mock Info
    level_dims = [6, 6, 12]
    # start indices: [0, 6, 12]
    level_starts = [0]
    current = 0
    for d in level_dims[:-1]:
        current += d
        level_starts.append(current)
        
    info = {
        'level_dims': level_dims, 
        'n_features': 1,
        'level_start_indices': level_starts,
        'n_samples': 100
    }

    print("Initializing Model...")
    try:
        model = trainer.init_model(info, CONFIG)
        print("Model initialized.")
    except Exception as e:
        print(f"FAILED to initialize model: {e}")
        with open('error_log.txt', 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(1)

    print("Running Dummy Forward Pass...")
    # Input shape: ((batch, seq_len, 1), (batch, 1)) -> (x, t)
    # The model expects specific inputs.
    # WaveletDiffusionTransformer call(inputs): inputs is [x_levels..., t]
    # Wait, trainer_interface.init_model returns `WaveletDiffusionTransformer`.
    # Its input signature in `call` is `inputs`.
    # Let's check `transformer.py` call method signature.
    
    # Generate dummy data
    # x represents the wavelet coefficients for each level.
    # dims are [6, 6, 12] * batch * features(1)
    
    x_inputs = []
    for dim in info['level_dims']:
        # Shape: (Batch, Dim, Features=1) ?? 
        # Verified transformer logic: It projects input_dim to embed_dim.
        # w_keras/data.py produces inputs as a list of tensors?
        # Let's verify trainer_interface or transformer.py inputs.
        x_inputs.append(np.random.normal(size=(BATCH_SIZE, dim, 1)).astype('float32'))
    
    # Time embedding
    t_input = np.random.uniform(size=(BATCH_SIZE, 1)).astype('float32')
    
    # Transformer expects (x_all, t) where x_all is concatenated levels
    x_concat = np.concatenate(x_inputs, axis=1)
    
    # Use Dict inputs matching updated pipeline
    inputs = {'x': x_concat, 't': t_input}
    
    try:
        # Run forward
        output = model(inputs)
        print("Forward pass successful.")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"FAILED forward pass: {e}")
        with open('error_log.txt', 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(1)

    print("Verifying Loss Function...")
    try:
        loss_fn = loss.WaveletLoss(info['level_dims'], 
                                   level_start_indices=info['level_start_indices'],
                                   use_energy_term=True, 
                                   energy_weight=0.1)
        # Dummy targets same shape as inputs (noise prediction)
        targets = x_concat
        
        # Loss call(y_true, y_pred)
        l = loss_fn(targets, output)
        print(f"Loss value: {l}")
    except Exception as e:
        print(f"FAILED loss calculation: {e}")
        with open('error_log.txt', 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(1)
        
    print("\nSUCCESS: Build Verification Passed!")

if __name__ == "__main__":
    main()
