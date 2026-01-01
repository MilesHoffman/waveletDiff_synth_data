import jax
import jax.numpy as jnp
import keras
from flax.training import train_state
import optax
import time
import numpy as np
from src.tpu_keras.models.train_utils import create_train_step, create_sample_fn

class TPUTrainer:
    """
    High-performance TPU Trainer for Keras 3 / JAX WaveletDiff.
    Optimized to eliminate host-TPU synchronization.
    """
    def __init__(self, 
                 model, 
                 scheduler, 
                 loss_fn, 
                 learning_rate=1e-4, 
                 steps_per_epoch=1000,
                 log_interval_percent=1):
        self.model = model
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.steps_per_epoch = steps_per_epoch
        self.log_interval = max(1, (steps_per_epoch * log_interval_percent) // 100)
        
        # Optimizer
        self.tx = optax.adamw(learning_rate)
        
        # Build model if not built (required for stateless_call)
        if not model.built:
            print("Model not built. Building with dummy inputs...")
            # Infer input shapes from loss_fn.level_dims
            # Input is [coeffs_list, t]
            # coeffs_list items shape: (batch, dim, input_dim) -> input_dim=1 usually
            # We assume input_dim=1 as per typical time series usage (or infer from model)
            input_dim = getattr(model, 'input_dim', 1) 
            
            # Create dummy inputs
            dummy_batch_size = 1
            dummy_coeffs = [
                jnp.zeros((dummy_batch_size, d, input_dim)) 
                for d in self.loss_fn.level_dims
            ]
            dummy_t = jnp.zeros((dummy_batch_size,), dtype="int32")
            
            # Build
            model.build([dummy_coeffs, dummy_t])
            print("Model built successfully.")

        # Extract trainable parameter values from Keras model
        # Keras 3 Variables have a .value property which returns the backend tensor (jax array)
        self.params = [v.value for v in model.trainable_variables]
        
        # Train state
        self.state = train_state.TrainState.create(
            apply_fn=None, # Not used directly, we use stateless_call in train_step
            params=self.params, 
            tx=self.tx
        )
        
        # Compilation
        self.train_step = create_train_step(model, scheduler, loss_fn, self.tx)
        self.sample_fn = create_sample_fn(model, scheduler)
        
        self.key = jax.random.PRNGKey(0)

    def train_epoch(self, dataloader, epoch):
        print(f"Starting Epoch {epoch}...")
        epoch_start = time.time()
        
        total_loss = 0.0
        step_count = 0
        
        for batch in dataloader:
            # batch is expected to be a list of wavelet coefficients
            # Convert to jnp
            batch_jnp = [jnp.array(c) for c in batch]
            
            self.state, loss, self.key = self.train_step(self.state, batch_jnp, self.key)
            
            total_loss += loss
            step_count += 1
            
            if step_count % self.log_interval == 0:
                # 1% Logging
                avg_loss = total_loss / self.log_interval
                print(f"  Step {step_count}/{self.steps_per_epoch} | Avg Loss: {avg_loss:.6f}")
                total_loss = 0.0
                
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    def generate_samples(self, shape_list):
        print("Generating samples...")
        samples = self.sample_fn(self.state.params, shape_list, self.key)
        # Shift key
        self.key, _ = jax.random.split(self.key)
        return samples
