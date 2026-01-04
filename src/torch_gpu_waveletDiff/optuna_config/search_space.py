"""Hyperparameter search space definitions with toggleable parameters."""

from typing import Dict, Any
from optuna.trial import Trial


class WaveletDiffSearchSpace:
    """
    Configurable search space for WaveletDiff hyperparameters.
    
    Each hyperparameter can be enabled/disabled via flags.
    """
    
    def __init__(self, tune_flags: Dict[str, bool], param_ranges: Dict[str, Any] = None):
        """
        Initialize search space with tuning flags and optional custom ranges.
        
        Args:
            tune_flags: Dict mapping hyperparameter names to boolean flags
            param_ranges: Dict defining min/max or choices for each parameter
        """
        self.tune_flags = tune_flags
        self.ranges = param_ranges or {}
    
    def suggest_hyperparameters(
        self, 
        trial: Trial,
        defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on enabled flags and configured ranges.
        """
        params = {}
        
        # Helper for float ranges
        def suggest_float(name, low, high, log=True):
            r = self.ranges.get(name, {})
            return trial.suggest_float(
                name, 
                r.get('min', low), 
                r.get('max', high), 
                log=log
            )
            
        # Helper for categorical ranges
        def suggest_cat(name, default_choices):
            choices = self.ranges.get(name, default_choices)
            # Ensure it's a list
            if not isinstance(choices, list):
                choices = default_choices
            return trial.suggest_categorical(name, choices)

        # Learning Rate
        if self.tune_flags.get('learning_rate', False):
            params['learning_rate'] = suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        else:
            params['learning_rate'] = defaults.get('learning_rate', 2e-4)
        
        # Max LR (OneCycleLR)
        if self.tune_flags.get('max_lr', False):
            params['max_lr'] = suggest_float('max_lr', 5e-4, 5e-3, log=True)
        else:
            params['max_lr'] = defaults.get('max_lr', 1e-3)
        
        # Weight Decay
        if self.tune_flags.get('weight_decay', False):
            params['weight_decay'] = suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        else:
            params['weight_decay'] = defaults.get('weight_decay', 1e-5)
        
        # Embedding Dimension
        if self.tune_flags.get('embed_dim', False):
            params['embed_dim'] = suggest_cat('embed_dim', [128, 256, 384, 512])
        else:
            params['embed_dim'] = defaults.get('embed_dim', 256)
        
        # Number of Attention Heads
        if self.tune_flags.get('num_heads', False):
            params['num_heads'] = suggest_cat('num_heads', [4, 8, 16])
        else:
            params['num_heads'] = defaults.get('num_heads', 8)
        
        # Number of Transformer Layers
        if self.tune_flags.get('num_layers', False):
            params['num_layers'] = suggest_cat('num_layers', [4, 6, 8, 10, 12])
        else:
            params['num_layers'] = defaults.get('num_layers', 8)
        
        # Dropout
        if self.tune_flags.get('dropout', False):
            params['dropout'] = suggest_float('dropout', 0.0, 0.3, log=False)
        else:
            params['dropout'] = defaults.get('dropout', 0.1)
        
        # Batch Size
        if self.tune_flags.get('batch_size', False):
            params['batch_size'] = suggest_cat('batch_size', [128, 256, 512, 1024])
        else:
            params['batch_size'] = defaults.get('batch_size', 512)
        
        # OneCycleLR pct_start
        if self.tune_flags.get('pct_start', False):
            params['pct_start'] = suggest_float('pct_start', 0.2, 0.4, log=False)
        else:
            params['pct_start'] = defaults.get('pct_start', 0.3)
        
        # Gradient Clipping Norm
        if self.tune_flags.get('grad_clip_norm', False):
            params['grad_clip_norm'] = suggest_float('grad_clip_norm', 0.5, 2.0, log=False)
        else:
            params['grad_clip_norm'] = defaults.get('grad_clip_norm', 1.0)
        
        # Time Embedding Dimension
        if self.tune_flags.get('time_embed_dim', False):
            params['time_embed_dim'] = suggest_cat('time_embed_dim', [64, 128, 256])
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
