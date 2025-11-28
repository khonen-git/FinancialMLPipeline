#!/usr/bin/env python3
"""Validate Hydra configuration.

Usage:
    python scripts/validate_config.py experiment=eurusd_scalping
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def validate_config(cfg: DictConfig):
    """Validate configuration.
    
    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Configuration validation")
    print("=" * 80)
    
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Basic validation
    errors = []
    
    # Check required fields
    if not cfg.get('experiment'):
        errors.append("Missing 'experiment' section")
    
    if not cfg.get('assets'):
        errors.append("Missing 'assets' section")
    
    if not cfg.get('session'):
        errors.append("Missing 'session' section")
    
    if not cfg.get('labeling'):
        errors.append("Missing 'labeling' section")
    
    # Check session times
    if cfg.get('session'):
        try:
            from datetime import time
            session_end = cfg.session.session_end
            # Validate time format
        except Exception as e:
            errors.append(f"Invalid session time format: {e}")
    
    # Check labeling params
    if cfg.get('labeling', {}).get('triple_barrier'):
        tb = cfg.labeling.triple_barrier
        if tb.tp_ticks <= 0:
            errors.append("tp_ticks must be > 0")
        if tb.sl_ticks <= 0:
            errors.append("sl_ticks must be > 0")
    
    # Report
    if errors:
        print("\n❌ Configuration validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\n✅ Configuration is valid!")


if __name__ == '__main__':
    validate_config()

