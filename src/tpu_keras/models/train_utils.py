import jax
import jax.numpy as jnp
from keras_core import ops

def create_train_step(model, scheduler, loss_fn, optimizer):
    """Creates a JIT-compiled training step."""
    
    @jax.jit
    def train_step(state, batch, key):
        # batch: (batch_size, seq_len, channels)
        # We need to decompose batch into wavelet levels first
        # For efficiency, this might happen in the data loader, 
        # but let's assume raw data for now and use model's DWT if it has one.
        
        # 1. Decompose (using model's internal DWT or external)
        # Assuming batch is already wavelet coefficients list for now?
        # Let's assume the caller handles DWT for now or we add it here.
        coeffs = batch # List of [A_L, D_L, ..., D_1]
        
        # 2. Sample time steps
        batch_size = coeffs[0].shape[0]
        key, t_key, n_key = jax.random.split(key, 3)
        t = jax.random.randint(t_key, (batch_size,), 0, scheduler.num_steps)
        
        # 3. Add noise to each level
        noisy_coeffs = []
        noises = []
        for c in coeffs:
            noise = jax.random.normal(n_key, c.shape) # Same n_key or split? Usually split.
            n_key, _ = jax.random.split(n_key)
            noisy_c = scheduler.q_sample(c, t, noise)
            noisy_coeffs.append(noisy_c)
            noises.append(noise)
            
        # 4. Predict
        def loss_forward(params):
            preds = model.apply(params, [noisy_coeffs, t], training=True)
            loss = loss_fn(noises, preds) # Predicting noise
            return loss
            
        grad_fn = jax.value_and_grad(loss_forward)
        loss, grads = grad_fn(state.params)
        
        # 5. Update
        state = state.apply_gradients(grads=grads)
        
        return state, loss, key
        
    return train_step

def create_sample_fn(model, scheduler):
    """Creates a sample function using jax.lax.scan."""
    
    @jax.jit
    def sample(params, shape_list, key):
        # shape_list: list of shapes for each level [(batch, n_i, c), ...]
        
        # Initial noise for each level
        key, *n_keys = jax.random.split(key, len(shape_list) + 1)
        coeffs = [jax.random.normal(nk, shape) for nk, shape in zip(n_keys, shape_list)]
        
        def scan_fn(carry_coeffs, t_idx):
            # t_idx goes from num_steps-1 down to 0
            curr_coeffs, curr_key = carry_coeffs
            t = ops.ones((shape_list[0][0],), dtype='int32') * t_idx
            
            # Predict noise
            preds = model.apply(params, [curr_coeffs, t], training=False)
            
            # Step back
            next_coeffs = []
            new_key, step_key = jax.random.split(curr_key)
            
            for i in range(len(curr_coeffs)):
                # Simplified DDPM step
                z = jax.random.normal(step_key, curr_coeffs[i].shape) if t_idx > 0 else 0
                
                # Equation: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_prod_t) * eps) + sigma_t * z
                # This logic is inside DiffusionScheduler or we inline it.
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
