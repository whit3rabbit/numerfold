from typing import Dict, Optional, Union
import os
import yaml
from dataclasses import dataclass, asdict
import torch

@dataclass
class PipelineConfig:
    """Configuration class for NumerAIFold pipeline."""
    # Data configuration
    data_version: str = "v5.0"
    feature_set: str = "small"
    era_col: str = "era"
    
    # Model architecture
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 3
    
    # Feature domain settings
    n_clusters: int = 10
    force_phase1: bool = False
    skip_phase1: bool = False
    domains_save_path: str = 'feature_domains_data.csv'
    
    # Pipeline settings
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    confidence_threshold: float = 0.5
    save_model: bool = True
    base_path: str = '.'

def get_default_pipeline_config() -> PipelineConfig:
    """
    Get default pipeline configuration.
    
    Returns:
        PipelineConfig: Default configuration object
    """
    return PipelineConfig()

def validate_pipeline_config(config: PipelineConfig) -> Dict[str, str]:
    """
    Validate pipeline configuration settings.
    
    Args:
        config: PipelineConfig object to validate
        
    Returns:
        Dict[str, str]: Dictionary of validation errors, empty if valid
    """
    errors = {}
    
    # Validate data configuration
    if not isinstance(config.data_version, str):
        errors['data_version'] = f"Expected string, got {type(config.data_version)}"
    
    if config.feature_set not in ['small', 'medium', 'all']:
        errors['feature_set'] = f"feature_set must be one of ['small', 'medium', 'all'], got {config.feature_set}"
    
    # Validate model architecture
    if config.embed_dim <= 0:
        errors['embed_dim'] = f"embed_dim must be positive, got {config.embed_dim}"
    
    if config.num_layers <= 0:
        errors['num_layers'] = f"num_layers must be positive, got {config.num_layers}"
    
    if config.num_heads <= 0:
        errors['num_heads'] = f"num_heads must be positive, got {config.num_heads}"
    
    if not 0 <= config.dropout < 1:
        errors['dropout'] = f"dropout must be between 0 and 1, got {config.dropout}"
    
    # Validate training parameters
    if config.batch_size <= 0:
        errors['batch_size'] = f"batch_size must be positive, got {config.batch_size}"
    
    if config.epochs <= 0:
        errors['epochs'] = f"epochs must be positive, got {config.epochs}"
    
    if config.learning_rate <= 0:
        errors['learning_rate'] = f"learning_rate must be positive, got {config.learning_rate}"
    
    if config.weight_decay < 0:
        errors['weight_decay'] = f"weight_decay must be non-negative, got {config.weight_decay}"
    
    # Validate domain settings
    if config.n_clusters <= 1:
        errors['n_clusters'] = f"n_clusters must be greater than 1, got {config.n_clusters}"
    
    # Validate paths
    if not os.path.exists(config.base_path):
        errors['base_path'] = f"base_path does not exist: {config.base_path}"
    
    return errors

def configure_pipeline(
    config_path: Optional[str] = None,
    **kwargs
) -> PipelineConfig:
    """
    Configure the NumerAIFold pipeline.
    
    Args:
        config_path: Optional path to YAML configuration file
        **kwargs: Optional overrides for configuration values
        
    Returns:
        PipelineConfig: Configured pipeline settings
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Start with default configuration
    config = get_default_pipeline_config()
    
    # Load from file if provided
    if config_path is not None:
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Update only valid fields from file
                for key, value in file_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {str(e)}")
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Validate the configuration
    errors = validate_pipeline_config(config)
    if errors:
        raise ValueError(f"Invalid configuration: {errors}")
    
    return config

def save_pipeline_config(config: PipelineConfig, save_path: str) -> None:
    """
    Save pipeline configuration to a YAML file.
    
    Args:
        config: PipelineConfig object to save
        save_path: Path to save the configuration file
    """
    # Convert dataclass to dictionary
    config_dict = asdict(config)
    
    # Save to YAML
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def load_pipeline_config(config_path: str) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        PipelineConfig: Loaded configuration
        
    Raises:
        ValueError: If configuration file is invalid
    """
    return configure_pipeline(config_path=config_path)