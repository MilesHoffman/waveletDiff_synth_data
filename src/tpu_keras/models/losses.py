import jax.numpy as jnp
from keras import ops

class WaveletLoss:
    """
    Balanced loss for multi-level wavelet coefficients.
    Supports level weighting and energy preservation.
    """
    def __init__(self, 
                 level_dims, 
                 strategy="coefficient_weighted", 
                 approximation_weight=2.0,
                 use_energy_term=False,
                 energy_weight=0.1):
        self.level_dims = level_dims
        self.strategy = strategy
        self.approximation_weight = approximation_weight
        self.use_energy_term = use_energy_term
        self.energy_weight = energy_weight
        
        self.level_weights = self._initialize_weights()

    def _initialize_weights(self):
        num_levels = len(self.level_dims)
        if self.strategy == "standard":
            return [1.0] * num_levels
        
        weights = []
        for i, dim in enumerate(self.level_dims):
            if i == 0: # Approximation
                weights.append(self.approximation_weight / dim)
            else:
                weights.append(1.0 / dim)
        
        total = sum(weights)
        return [w / total for w in weights]

    def __call__(self, targets, predictions):
        # targets/predictions are lists of tensors [A_L, D_L, ..., D_1]
        
        recon_loss = 0.0
        for i in range(len(targets)):
            # MSE for this level
            level_mse = ops.mean(ops.square(targets[i] - predictions[i]))
            recon_loss += self.level_weights[i] * level_mse
            
        if not self.use_energy_term:
            return recon_loss
            
        # Energy loss (sum of variances)
        energy_loss = 0.0
        for i in range(len(targets)):
            # energy = sum of squares per sample/feature
            # Here we just use the level-wise MSE as a proxy or calculate exact energy diff
            target_energy = ops.sum(ops.square(targets[i]), axis=1) # (batch, channels)
            pred_energy = ops.sum(ops.square(predictions[i]), axis=1)
            energy_loss += ops.mean(ops.abs(target_energy - pred_energy))
            
        return recon_loss + self.energy_weight * energy_loss
