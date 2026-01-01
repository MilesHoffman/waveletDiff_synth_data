import os
import sys
sys.path.append('src')

import tensorflow as tf
from w_keras import trainer_interface

# Mock config
config = {
    'PREDICTION_TARGET': 'noise',
    'EMBED_DIM': 256,
    'NUM_HEADS': 8,
    'NUM_LAYERS': 4,
    'DROPOUT': 0.1,
    'LEARNING_RATE': 1e-4,
    'EPOCHS': 1,
    'BATCH_SIZE': 4 
}

# Path to data (assuming user paths)
data_path = r'c:\Users\Miles\personal\repos\waveletDiff_synth_data\src\copied_waveletDiff\data\stocks\stock_data.csv' 
# Just use dummy CSV if needed or try to find existing one. 
# Attempting to verify with existing logic if possible.
# If CSV doesn't exist, we might need to mock prepare_wavelet_data.

# Let's verify what load_dataset returns
print("Loading dataset...")
try:
    ds, info = trainer_interface.get_dataloader(
        data_path, 
        batch_size=4, 
        seq_len=32, 
        config=config
    )
    
    print("Inspecting element spec...")
    print(ds.element_spec)
    
    print("Taking one batch...")
    for batch in ds.take(1):
        inputs, targets = batch
        print(f"Inputs type: {type(inputs)}")
        if isinstance(inputs, dict):
            print(f"Keys: {inputs.keys()}")
            print(f"x shape: {inputs['x'].shape}, dtype: {inputs['x'].dtype}")
            print(f"t shape: {inputs['t'].shape}, dtype: {inputs['t'].dtype}")
        else:
            print(f"Inputs shape: {inputs.shape}")
            
        print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")
        
except Exception as e:
    print(f"Error: {e}")
