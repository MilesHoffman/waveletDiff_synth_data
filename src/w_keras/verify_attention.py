
import os
import sys
import numpy as np

os.environ['KERAS_BACKEND'] = 'jax'
import keras
from keras import ops

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from w_keras.attention import CrossLevelAttention

def main():
    print("Verifying CrossLevelAttention...")
    
    B = 4
    level_dims = [64, 32, 32] # 3 levels
    seq_lens = [6, 6, 12]
    time_dim = 16
    
    # Mock inputs
    level_embeddings = []
    for dim, seq in zip(level_dims, seq_lens):
        shape = (B, seq, dim)
        level_embeddings.append(np.random.normal(size=shape).astype('float32'))
        
    time_embed = np.random.normal(size=(B, time_dim)).astype('float32')
    
    print(f"Inputs: Levels={[s.shape for s in level_embeddings]}, Time={time_embed.shape}")
    
    layer = CrossLevelAttention(
        level_embed_dims=level_dims,
        num_heads=4,
        time_embed_dim=time_dim,
        attention_mode='all_to_all' # Test this mode
    )
    
    try:
        out = layer(level_embeddings, time_embed)
        print("Success!")
        for i, o in enumerate(out):
             print(f"Out {i}: {o.shape}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
