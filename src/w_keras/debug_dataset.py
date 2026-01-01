import os
import sys
sys.path.append('src')

import tensorflow as tf
import numpy as np
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

data_path = r'c:\Users\Miles\personal\repos\waveletDiff_synth_data\src\copied_waveletDiff\data\stocks\stock_data.csv' 

print("Loading dataset...")
try:
    ds, info = trainer_interface.get_dataloader(
        data_path, 
        batch_size=4, 
        seq_len=32, 
        config=config
    )
    
    print(f"Element Spec: {ds.element_spec}")
    
    print("Taking one batch...")
    for batch in ds.take(1):
        inputs, targets = batch
        print(f"Inputs type: {type(inputs)}")
        
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                print(f"Key '{k}' shape: {v.shape}, dtype: {v.dtype}")
        elif isinstance(inputs, (tuple, list)):
             for i, v in enumerate(inputs):
                print(f"Index {i} shape: {v.shape}, dtype: {v.dtype}")
        else:
            print(f"Single input shape: {inputs.shape}")
            
        print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")
        
except Exception as e:
    import traceback
    traceback.print_exc()
