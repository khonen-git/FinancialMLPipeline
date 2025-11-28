"""Data cleaning and normalization module.

According to DATA_HANDLING.md:
- Convert timestamps to UTC
- Sort chronologically
- Drop corrupted rows
- Ensure askPrice > bidPrice
- Compute derived fields (midPrice, spread)
"""

import logging
from typing import Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and normalize raw tick data."""
    
    def __init__(self, config: dict):
        """Initialize data cleaner.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.timezone = config.get('timezone', 'UTC')
    
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Clean raw tick data.
        
        Args:
            df: Raw DataFrame with tick data
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning stats dict)
        """
        logger.info(f"Starting cleaning. Input rows: {len(df)}")
        
        initial_rows = len(df)
        stats = {'initial_rows': initial_rows}
        
        # Convert timestamp from epoch ms to UTC datetime
        df = self._convert_timestamps(df)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.debug("Sorted by timestamp")
        
        # Drop corrupted rows
        df, corruption_stats = self._drop_corrupted_rows(df)
        stats.update(corruption_stats)
        
        # Enforce strictly increasing timestamps
        df = self._enforce_strictly_increasing_timestamps(df)
        stats['after_timestamp_dedup'] = len(df)
        
        # Compute derived fields
        df = self._compute_derived_fields(df)
        
        stats['final_rows'] = len(df)
        stats['rows_dropped'] = initial_rows - stats['final_rows']
        stats['drop_rate'] = stats['rows_dropped'] / initial_rows if initial_rows > 0 else 0
        
        logger.info(
            f"Cleaning complete. Final rows: {stats['final_rows']} "
            f"({stats['drop_rate']*100:.2f}% dropped)"
        )
        
        return df, stats
    
    def _convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert epoch milliseconds to UTC datetime.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with converted timestamps
        """
        df = df.copy()
        
        if df['timestamp'].dtype == 'int64':
            # Assume epoch milliseconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        elif not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            # Try to convert
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Ensure UTC
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        return df
    
    def _drop_corrupted_rows(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Drop corrupted rows (NaN, inf, askPrice <= bidPrice).
        
        Args:
            df: DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, stats dict)
        """
        df = df.copy()
        stats = {}
        
        initial_rows = len(df)
        
        # Drop NaN in required columns
        required_cols = ['timestamp', 'askPrice', 'bidPrice']
        df = df.dropna(subset=required_cols)
        stats['nan_dropped'] = initial_rows - len(df)
        
        # Drop infinite values
        df = df[~np.isinf(df['askPrice'])]
        df = df[~np.isinf(df['bidPrice'])]
        stats['inf_dropped'] = initial_rows - stats['nan_dropped'] - len(df)
        
        # Ensure askPrice > bidPrice
        df = df[df['askPrice'] > df['bidPrice']]
        stats['invalid_spread_dropped'] = (
            initial_rows - stats['nan_dropped'] - stats['inf_dropped'] - len(df)
        )
        
        # Drop negative volumes if present
        if 'askVolume' in df.columns:
            df = df[df['askVolume'] >= 0]
        if 'bidVolume' in df.columns:
            df = df[df['bidVolume'] >= 0]
        stats['negative_volume_dropped'] = (
            initial_rows - stats['nan_dropped'] - stats['inf_dropped'] 
            - stats['invalid_spread_dropped'] - len(df)
        )
        
        return df, stats
    
    def _enforce_strictly_increasing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keep first occurrence.
        
        Args:
            df: DataFrame sorted by timestamp
            
        Returns:
            DataFrame with unique timestamps
        """
        df = df.copy()
        
        # Drop duplicate timestamps (keep first)
        duplicates_before = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        duplicates_dropped = duplicates_before - len(df)
        
        if duplicates_dropped > 0:
            logger.debug(f"Dropped {duplicates_dropped} duplicate timestamps")
        
        return df
    
    def _compute_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute midPrice and spread.
        
        Args:
            df: DataFrame with askPrice and bidPrice
            
        Returns:
            DataFrame with derived fields
        """
        df = df.copy()
        
        df['midPrice'] = (df['askPrice'] + df['bidPrice']) / 2.0
        df['spread'] = df['askPrice'] - df['bidPrice']
        
        return df


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned data as Parquet.
    
    Args:
        df: Cleaned DataFrame
        output_path: Output file path
    """
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"Cleaned data saved to {output_path}")

