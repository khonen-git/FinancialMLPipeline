"""Configuration helper functions for Hydra configs."""

import logging
from pathlib import Path
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> DictConfig:
    """Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Loaded configuration as DictConfig
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def get_asset_config(cfg: DictConfig) -> Dict[str, Any]:
    """Extract asset configuration from Hydra config.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Asset configuration dictionary
    """
    if "asset" not in cfg:
        raise ValueError("Asset configuration not found in config")
    
    return OmegaConf.to_container(cfg.asset, resolve=True)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations.
    
    Args:
        *configs: Variable number of DictConfig objects
        
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(*configs)


def save_config(cfg: DictConfig, output_path: Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        cfg: Configuration to save
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path)
    logger.info(f"Configuration saved to {output_path}")

