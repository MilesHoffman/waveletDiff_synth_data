import jax
import jax.numpy as jnp
from keras import ops

class DiffusionScheduler:
    """
    JAX-native Diffusion Scheduler for WaveletDiff.
    Handles noise schedules and sampling steps.
    """
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        self.num_steps = num_steps
        
        if schedule_type == 'linear':
            self.betas = jnp.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0]), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # Below: clip because posterior variance is 0 at t=0
        self.posterior_log_variance_clipped = jnp.log(jnp.maximum(self.posterior_variance, 1e-20))
        self.posterior_mean_coef1 = self.betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * jnp.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(self, steps, s=0.008):
        """Cosine schedule as proposed in IDDPM."""
        t = jnp.linspace(0, steps, steps + 1)
        alphas_cumprod = jnp.cos(((t / steps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return jnp.clip(betas, 0, 0.999)

    def q_sample(self, x_start, t, noise):
        """Diffuses x_start up to step t."""
        # x_start: (batch, seq, dim)
        # t: (batch,)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """Recovers x_0 estimate from x_t and predicted noise."""
        return (
            self.sqrt_recip_alphas_cumprod[t][:, None, None] * x_t - 
            self.sqrt_recipm1_alphas_cumprod[t][:, None, None] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Calculates the mean and variance of the posterior q(x_{t-1} | x_t, x_0)."""
        mean = (
            self.posterior_mean_coef1[t][:, None, None] * x_start +
            self.posterior_mean_coef2[t][:, None, None] * x_t
        )
        log_variance = self.posterior_log_variance_clipped[t][:, None, None]
        return mean, log_variance
