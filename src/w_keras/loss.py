"""
WaveletLoss implementation (Keras 3).
Matches src.models.wavelet_losses.py
"""

import keras
from keras import ops

class WaveletLoss(keras.losses.Loss):
    def __init__(self, 
                 level_dims, 
                 level_start_indices, 
                 strategy="coefficient_weighted", 
                 approximation_weight=2.0, 
                 use_energy_term=False, 
                 energy_weight=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.level_dims = level_dims
        self.level_start_indices = level_start_indices
        self.strategy = strategy
        self.approximation_weight = approximation_weight
        self.use_energy_term = use_energy_term
        self.energy_weight = energy_weight
        
        # Compute weights based on strategy
        # Weights for each level
        self.level_weights = []
        num_levels = len(level_dims)
        
        if strategy == "coefficient_weighted":
            # Weight = 1.0 / num_coeffs (Inverse Linear)
            # Level 0 gets extra approximation_weight
            for i, dim in enumerate(level_dims):
                w = 1.0 / float(dim)
                if i == 0:
                    w *= approximation_weight
                self.level_weights.append(w)
                
            # Normalize weights to sum to 1.0 (or total_weight in PyTorch)
            # Source uses total_weight = sum([1/dim ...]) and divides by it.
            total_weight = sum(self.level_weights)
            self.level_weights = [w / total_weight for w in self.level_weights]
            
        elif strategy == "approximation_heavy":
            for i in range(num_levels):
                if i == 0:
                    self.level_weights.append(approximation_weight)
                else:
                    self.level_weights.append(1.0)
            
            # Normalize? Source usually does. Let's normalize here too for consistency.
            total_weight = sum(self.level_weights)
            self.level_weights = [w / total_weight for w in self.level_weights]
        else:
            # Default ones (Uniform)
            self.level_weights = [1.0 / num_levels] * num_levels
            
        print(f"WaveletLoss initialized. Weights: {self.level_weights}, Energy: {use_energy_term} ({energy_weight})")

    def get_level_losses(self, y_true, y_pred):
        """Returns list of losses per level."""
        losses = []
        for i, (start, dim) in enumerate(zip(self.level_start_indices, self.level_dims)):
            # Slice: [B, LevelDim, Features]
            yt = y_true[:, start:start+dim, :]
            yp = y_pred[:, start:start+dim, :]
            
            # MSE for this level
            mse = ops.mean(ops.square(yt - yp))
            losses.append(mse)
        return losses

    def call(self, y_true, y_pred):
        # 1. Reconstruction Loss (Weighted Sum of Level MSEs)
        level_losses = self.get_level_losses(y_true, y_pred)
        
        recon_loss = 0.0
        for l_loss, weight in zip(level_losses, self.level_weights):
            recon_loss += l_loss * weight
            
        # 2. Energy Term (Optional)
        # Energy = sum(squared coeffs)
        # We want predicted energy to match target energy
        if self.use_energy_term and self.energy_weight > 0:
            energy_true = ops.mean(ops.square(y_true), axis=1) # [B, F] - Mean energy across sequence
            energy_pred = ops.mean(ops.square(y_pred), axis=1) # [B, F]
            
            # Loss is MSE of energies
            energy_loss = ops.mean(ops.square(energy_true - energy_pred))
            
            return recon_loss + self.energy_weight * energy_loss
            
        return recon_loss
