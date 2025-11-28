"""Data cleaning and preprocessing.

Cleans Dukascopy tick data:
- Remove outliers
- Handle duplicates
- Fill gaps
- Normalize timestamps
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def clean_ticks(
    ticks: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """Clean tick data.
    
    Args:
        ticks: Raw tick DataFrame
        config: Cleaning configuration
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning {len(ticks)} ticks")
    
    initial_count = len(ticks)
    
    # Remove duplicates
    if config.get('remove_duplicates', True):
        ticks = ticks.drop_duplicates(subset=['timestamp'], keep='first')
        logger.info(f"Removed {initial_count - len(ticks)} duplicate timestamps")
    
    # Remove zero spreads
    if config.get('remove_zero_spread', True):
        spread = ticks['askPrice'] - ticks['bidPrice']
        zero_spread_mask = spread <= 0
        ticks = ticks[~zero_spread_mask]
        logger.info(f"Removed {zero_spread_mask.sum()} zero spread ticks")
    
    # Remove outliers (price spikes)
    if config.get('remove_outliers', True):
        ticks = remove_price_outliers(ticks, config)
    
    # Sort by timestamp
    ticks = ticks.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Cleaning complete: {len(ticks)} ticks remaining")
    
    return ticks


def remove_price_outliers(
    ticks: pd.DataFrame,
    config: dict,
    price_col: str = 'bidPrice'
) -> pd.DataFrame:
    """Remove price outliers using rolling z-score.
    
    Args:
        ticks: Tick DataFrame
        config: Configuration
        price_col: Price column to check
        
    Returns:
        DataFrame without outliers
    """
    window = config.get('outlier_window', 100)
    threshold = config.get('outlier_threshold', 5.0)
    
    # Compute rolling mean and std
    rolling_mean = ticks[price_col].rolling(window=window, min_periods=1).mean()
    rolling_std = ticks[price_col].rolling(window=window, min_periods=1).std()
    
    # Z-score
    z_score = np.abs((ticks[price_col] - rolling_mean) / (rolling_std + 1e-10))
    
    # Filter
    outlier_mask = z_score > threshold
    n_outliers = outlier_mask.sum()
    
    if n_outliers > 0:
        logger.info(f"Removed {n_outliers} price outliers (z-score > {threshold})")
    
    return ticks[~outlier_mask]


def normalize_timestamps(
    ticks: pd.DataFrame
) -> pd.DataFrame:
    """Normalize timestamps to datetime.
    
    Args:
        ticks: Tick DataFrame
        
    Returns:
        DataFrame with normalized timestamps
    """
    if not pd.api.types.is_datetime64_any_dtype(ticks['timestamp']):
        ticks['timestamp'] = pd.to_datetime(ticks['timestamp'], unit='ms')
    
    # Set as index
    ticks = ticks.set_index('timestamp')
    
    return ticks
