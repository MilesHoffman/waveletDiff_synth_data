"""Configuration loading and management utilities."""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

INTERNAL_DEFAULTS = {
    "training": {
        "epochs": 5000,
        "batch_size": 512,
        "save_model": True,
        "log_every_n_epochs": 1
    },
    "model": {
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 8,
        "time_embed_dim": 128,
        "dropout": 0.1,
        "prediction_target": "noise"
    },
    "attention": {
        "use_cross_level_attention": True
    },
    "energy": {
        "weight": 0.0
    },
    "noise": {
        "schedule": "exponential"
    },
    "wavelet": {
        "type": "db2",
        "levels": "auto"
    },
    "sampling": {
        "method": "ddpm",
        "ddim_eta": 0.0,
        "ddim_steps": None
    },
    "data": {
        "normalize_data": True,
        "data_dir": "../data"
    },
    "optimizer": {
        "scheduler_type": "onecycle",
        "warmup_epochs": 50,
        "lr": 0.0002
    },
    "dataset": {
        "name": "stocks",
        "seq_len": 24
    },
    "evaluation": {
        "num_samples": 20000
    },
    "paths": {
        "output_dir": "../outputs"
    }
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_dataset_config(dataset_name: str, config_dir: str = "configs") -> Dict[str, Any]:
    """Load configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        config_dir: Directory containing config files
        
    Returns:
        Complete configuration dictionary with dataset-specific overrides
    """
    # Start with internal defaults
    config = INTERNAL_DEFAULTS.copy()
    
    # Load default config file if it exists
    default_config_path = os.path.join(config_dir, "default.yaml")
    if os.path.exists(default_config_path):
        try:
            yaml_config = load_config(default_config_path)
            config = merge_configs(config, yaml_config)
        except Exception as e:
            print(f"Warning: Could not load default config from {default_config_path}: {e}")
    
    # Load dataset-specific config if it exists
    dataset_config_path = os.path.join(config_dir, "datasets", f"{dataset_name}.yaml")
    if os.path.exists(dataset_config_path):
        try:
            dataset_config = load_config(dataset_config_path)
            config = merge_configs(config, dataset_config)
        except Exception as e:
            print(f"Warning: Could not load dataset config from {dataset_config_path}: {e}")
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the config file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


class ConfigManager:
    """Configuration manager class for handling configs throughout training."""
    
    def __init__(self, config_dir: str = "../configs"):
        self.config_dir = config_dir
        self.config = None
    
    def load(self, dataset_name: Optional[str] = None, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration with optional dataset-specific and manual overrides.
        
        Args:
            dataset_name: Name of dataset for dataset-specific config
            config_overrides: Manual configuration overrides
            
        Returns:
            Complete configuration dictionary
        """
        if dataset_name:
            self.config = load_dataset_config(dataset_name, self.config_dir)
        else:
            # Start with internal defaults
            self.config = INTERNAL_DEFAULTS.copy()
            default_config_path = os.path.join(self.config_dir, "default.yaml")
            if os.path.exists(default_config_path):
                try:
                    yaml_config = load_config(default_config_path)
                    self.config = merge_configs(self.config, yaml_config)
                except Exception as e:
                    print(f"Warning: Could not load default config from {default_config_path}: {e}")
        
        # Apply manual overrides if provided
        if config_overrides:
            self.config = merge_configs(self.config, config_overrides)
        
        return self.config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value (e.g., 'model.embed_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update(self, key_path: str, value: Any) -> None:
        """Update a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value
            value: New value to set
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the final value
        config_ref[keys[-1]] = value



