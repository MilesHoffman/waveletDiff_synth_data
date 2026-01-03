"""Hyperparameter search space definitions with toggleable parameters."""

from typing import Dict, Any
from optuna.trial import Trial


class WaveletDiffSearchSpace:
    """
    Configurable search space for WaveletDiff hyperparameters.
    
    Each hyperparameter can be enabled/disabled via flags.
    """
    
    def __init__(self, tune_flags: Dict[str, bool]):
        """
        Initialize search space with tuning flags.
        
        Args:
            tune_flags: Dict mapping hyperparameter names to boolean flags
                       True = tune this parameter, False = use default
        """
        self.tune_flags = tune_flags
    
    def suggest_hyperparameters(
        self, 
        trial: Trial,
        defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on enabled flags.
        
        Args:
            trial: Optuna trial object
            defaults: Default values for parameters not being tuned
            
        Returns:
            Dict of hyperparameter values (tuned or default)
        """
        params = {}
        
        # Learning Rate
        if self.tune_flags.get('learning_rate', False):
            params['learning_rate'] = trial.suggest_float(
                'learning_rate', 1e-5, 1e-3, log=True
            )
        else:
            params['learning_rate'] = defaults.get('learning_rate', 2e-4)
        
        # Max LR (OneCycleLR)
        if self.tune_flags.get('max_lr', False):
            params['max_lr'] = trial.suggest_float(
                'max_lr', 5e-4, 5e-3, log=True
            )
        else:
            params['max_lr'] = defaults.get('max_lr', 1e-3)
        
        # Weight Decay
        if self.tune_flags.get('weight_decay', False):
            params['weight_decay'] = trial.suggest_float(
                'weight_decay', 1e-6, 1e-4, log=True
            )
        else:
            params['weight_decay'] = defaults.get('weight_decay', 1e-5)
        
        # Embedding Dimension
        if self.tune_flags.get('embed_dim', False):
            params['embed_dim'] = trial.suggest_categorical(
                'embed_dim', [128, 256, 384, 512]
            )
        else:
            params['embed_dim'] = defaults.get('embed_dim', 256)
        
        # Number of Attention Heads
        if self.tune_flags.get('num_heads', False):
            params['num_heads'] = trial.suggest_categorical(
                'num_heads', [4, 8, 16]
            )
        else:
            params['num_heads'] = defaults.get('num_heads', 8)
        
        # Number of Transformer Layers
        if self.tune_flags.get('num_layers', False):
            params['num_layers'] = trial.suggest_categorical(
                'num_layers', [4, 6, 8, 10, 12]
            )
        else:
            params['num_layers'] = defaults.get('num_layers', 8)
        
        # Dropout
        if self.tune_flags.get('dropout', False):
            params['dropout'] = trial.suggest_float(
                'dropout', 0.0, 0.3
            )
        else:
            params['dropout'] = defaults.get('dropout', 0.1)
        
        # Batch Size
        if self.tune_flags.get('batch_size', False):
            params['batch_size'] = trial.suggest_categorical(
                'batch_size', [128, 256, 512, 1024]
            )
        else:
            params['batch_size'] = defaults.get('batch_size', 512)
        
        # OneCycleLR pct_start
        if self.tune_flags.get('pct_start', False):
            params['pct_start'] = trial.suggest_float(
                'pct_start', 0.2, 0.4
            )
        else:
            params['pct_start'] = defaults.get('pct_start', 0.3)
        
        # Gradient Clipping Norm
        if self.tune_flags.get('grad_clip_norm', False):
            params['grad_clip_norm'] = trial.suggest_float(
                'grad_clip_norm', 0.5, 2.0
            )
        else:
            params['grad_clip_norm'] = defaults.get('grad_clip_norm', 1.0)
        
        # Time Embedding Dimension
        if self.tune_flags.get('time_embed_dim', False):
            params['time_embed_dim'] = trial.suggest_categorical(
                'time_embed_dim', [64, 128, 256]
            )
        else:
            params['time_embed_dim'] = defaults.get('time_embed_dim', 128)
        
        # Ensure num_heads divides embed_dim
        if params['embed_dim'] % params['num_heads'] != 0:
            params['embed_dim'] = (params['embed_dim'] // params['num_heads']) * params['num_heads']
            trial.set_user_attr('adjusted_embed_dim', params['embed_dim'])
        
        return params
    
    @staticmethod
    def get_all_flags() -> Dict[str, bool]:
        """Get dictionary of all available hyperparameters with default False."""
        return {
            'learning_rate': False,
            'max_lr': False,
            'weight_decay': False,
            'embed_dim': False,
            'num_heads': False,
            'num_layers': False,
            'dropout': False,
            'batch_size': False,
            'pct_start': False,
            'grad_clip_norm': False,
            'time_embed_dim': False,
        }
    
    @staticmethod
    def get_recommended_flags() -> Dict[str, bool]:
        """Get recommended high-impact hyperparameters to tune."""
        return {
            'learning_rate': True,
            'max_lr': True,
            'weight_decay': True,
            'embed_dim': True,
            'num_heads': False,
            'num_layers': True,
            'dropout': True,
            'batch_size': True,
            'pct_start': False,
            'grad_clip_norm': False,
            'time_embed_dim': False,
        }
