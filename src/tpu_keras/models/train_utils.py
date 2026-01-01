import jax
import jax.numpy as jnp
from keras import ops

def create_train_step(model, scheduler, loss_fn, optimizer):
    """Creates a JIT-compiled training step for Keras 3 + JAX."""
    
    # Capture non-trainable variables (assumed constant for this architecture)
    # If using BatchNormalization, this would need to obtain updated values.
    non_trainable_weights = [v.value for v in model.non_trainable_variables]

    @jax.jit
    def train_step(state, batch, key):
        # batch: List of wavelet coefficients [A_L, D_L, ..., D_1]
        
        # 1. Setup Data
        coeffs = batch
        batch_size = coeffs[0].shape[0]
        
        # 2. Sample time steps
        key, t_key, n_key = jax.random.split(key, 3)
        t = jax.random.randint(t_key, (batch_size,), 0, scheduler.num_steps)
        
        # 3. Add noise to each level
        noisy_coeffs = []
        noises = []
        for c in coeffs:
            noise = jax.random.normal(n_key, c.shape)
            n_key, _ = jax.random.split(n_key)
            noisy_c = scheduler.q_sample(c, t, noise)
            noisy_coeffs.append(noisy_c)
            noises.append(noise)
            
        # 4. Predict & Loss
        def loss_forward(trainable_params):
            # Use stateless_call for Keras 3 compatibility
            # Returns: (outputs, non_trainable_variables, losses)
            preds, _, _ = model.stateless_call(
                trainable_params, 
                non_trainable_weights, 
                [noisy_coeffs, t], 
                training=True
            )
            
            # Ensure output is list (if model returns single tensor, stateless_call returns single tensor in outputs position)
            # But WaveletDiffusionTransformer returns a list.
            if not isinstance(preds, (list, tuple)):
                preds = [preds]
                
            loss = loss_fn(noises, preds)
            return loss
            
        loss, grads = jax.value_and_grad(loss_forward)(state.params)
        
        # 5. Update
        state = state.apply_gradients(grads=grads)
        
        return state, loss, key
        
    return train_step

def create_sample_fn(model, scheduler):
    """Creates a sample function using jax.lax.scan."""
    
    non_trainable_weights = [v.value for v in model.non_trainable_variables]
    
    @jax.jit
    def sample(params, shape_list, key):
        # params: trainable parameters (state.params)
        
        # Initial noise for each level
        key, *n_keys = jax.random.split(key, len(shape_list) + 1)
        coeffs = [jax.random.normal(nk, shape) for nk, shape in zip(n_keys, shape_list)]
        
        def scan_fn(carry_coeffs, t_idx):
            # t_idx goes from num_steps-1 down to 0
            curr_coeffs, curr_key = carry_coeffs
            t = ops.ones((shape_list[0][0],), dtype='int32') * t_idx
            
            # Predict noise using stateless_call
            preds, _, _ = model.stateless_call(
                params, 
                non_trainable_weights, 
                [curr_coeffs, t], 
                training=False
            )
            
            # Step back
            next_coeffs = []
            new_key, step_key = jax.random.split(curr_key)
            
            for i in range(len(curr_coeffs)):
                z = jax.random.normal(step_key, curr_coeffs[i].shape) if t_idx > 0 else 0
                
                # Equation: x_{t-1} = ...
                x_0_pred = scheduler.predict_start_from_noise(curr_coeffs[i], t, preds[i])
                mean, log_var = scheduler.q_posterior(x_0_pred, curr_coeffs[i], t)
                
                next_c = mean + ops.exp(0.5 * log_var) * z
                next_coeffs.append(next_c)
                
            return (next_coeffs, new_key), None

        # Scan over timesteps
        timesteps = jnp.arange(scheduler.num_steps - 1, -1, -1)
        (final_coeffs, _), _ = jax.lax.scan(scan_fn, (coeffs, key), timesteps)
        
        return final_coeffs

    return sample
