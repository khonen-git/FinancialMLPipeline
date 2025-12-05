"""Data loading and cleaning module.

Handles:
- Loading CSV/Parquet files
- Schema detection and validation
- Data cleaning
"""

import logging
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig

from src.data.schema_detection import SchemaDetector

logger = logging.getLogger(__name__)


def load_and_clean_data(cfg: DictConfig) -> pd.DataFrame:
    """Load and clean tick data.
    
    Args:
        cfg: Hydra configuration with data.dukascopy section
        
    Returns:
        Cleaned DataFrame with timestamp as index
        
    Raises:
        ValueError: If required config is missing or file not found
    """
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    
    # Require filename in config
    if 'filename' not in cfg.data.dukascopy:
        raise ValueError("Missing required config: data.dukascopy.filename")
    filename = cfg.data.dukascopy.filename
    data_path = Path(cfg.data.dukascopy.raw_dir) / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Require format in config
    if 'format' not in cfg.data.dukascopy:
        raise ValueError("Missing required config: data.dukascopy.format")
    file_format = cfg.data.dukascopy.format
    if file_format == 'auto':
        file_format = 'csv' if str(data_path).endswith('.csv') else 'parquet'
    
    # Load data
    if file_format == 'csv':
        logger.info(f"Loading CSV: {data_path}")
        ticks = pd.read_csv(data_path)
        # Ensure timestamp column is datetime
        if 'timestamp' in ticks.columns:
            ticks['timestamp'] = pd.to_datetime(ticks['timestamp'], unit='ms', utc=True)
    else:
        logger.info(f"Loading Parquet: {data_path}")
        ticks = pd.read_parquet(data_path)
    
    logger.info(f"Loaded {len(ticks)} ticks from {data_path}")
    
    # Step 2: Schema detection and cleaning (before setting index)
    logger.info("Step 2: Schema detection and cleaning")
    detector = SchemaDetector(cfg.data.dukascopy)
    ticks = detector.validate_and_clean(ticks)
    
    # Set timestamp as index after validation
    if 'timestamp' in ticks.columns:
        ticks = ticks.set_index('timestamp')
    
    logger.info(f"Cleaned data: {len(ticks)} ticks remaining")
    
    return ticks

