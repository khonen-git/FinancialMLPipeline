"""Bar construction using pandas.

Supports:
- Tick bars (every N ticks)
- Volume bars (every N volume units)
- Dollar bars (future)

According to ARCH_DATA_PIPELINE.md and DATA_HANDLING.md.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BarBuilder:
    """Build bars from tick data."""
    
    def __init__(self, config: dict):
        """Initialize bar builder.
        
        Args:
            config: Bar configuration from Hydra
        """
        self.config = config
    
    def build_tick_bars(
        self,
        df: pd.DataFrame,
        tick_size: int,
        use_bid_ask: bool = True
    ) -> pd.DataFrame:
        """Build tick bars (every N ticks).
        
        Args:
            df: Tick data with timestamp, bid, ask prices
            tick_size: Number of ticks per bar
            use_bid_ask: If True, build separate bid/ask OHLC
            
        Returns:
            DataFrame with OHLC bars
        """
        logger.info(f"Building {tick_size}-tick bars from {len(df)} ticks")
        
        if len(df) == 0:
            raise ValueError("Empty DataFrame provided")
        
        # Group into chunks of tick_size
        df = df.reset_index(drop=True)
        df['bar_id'] = df.index // tick_size
        
        if use_bid_ask:
            bars = self._build_bid_ask_ohlc(df)
        else:
            bars = self._build_mid_ohlc(df)
        
        # Add metadata
        bars['tick_count'] = df.groupby('bar_id').size()
        bars['bar_duration_sec'] = (
            df.groupby('bar_id')['timestamp'].apply(
                lambda x: (x.max() - x.min()).total_seconds()
            )
        )
        
        logger.info(f"Created {len(bars)} bars")
        
        return bars
    
    def _build_bid_ask_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build separate bid and ask OHLC.
        
        Args:
            df: Tick data with bar_id
            
        Returns:
            DataFrame with bid/ask OHLC columns
        """
        agg_dict = {
            'timestamp': 'last',
            # Bid OHLC
            'bidPrice': ['first', 'max', 'min', 'last'],
            # Ask OHLC
            'askPrice': ['first', 'max', 'min', 'last'],
            # Mid and spread
            'midPrice': ['first', 'last', 'mean'],
            'spread': ['mean', 'std', 'min', 'max'],
        }
        
        # Add volume if present
        if 'bidVolume' in df.columns:
            agg_dict['bidVolume'] = 'sum'
        if 'askVolume' in df.columns:
            agg_dict['askVolume'] = 'sum'
        
        bars = df.groupby('bar_id').agg(agg_dict)
        
        # Flatten multi-level columns
        bars.columns = [
            '_'.join(col) if isinstance(col, tuple) else col
            for col in bars.columns
        ]
        
        # Rename to standard format
        bars = bars.rename(columns={
            'bidPrice_first': 'bid_open',
            'bidPrice_max': 'bid_high',
            'bidPrice_min': 'bid_low',
            'bidPrice_last': 'bid_close',
            'askPrice_first': 'ask_open',
            'askPrice_max': 'ask_high',
            'askPrice_min': 'ask_low',
            'askPrice_last': 'ask_close',
            'timestamp_last': 'timestamp',
        })
        
        return bars.reset_index(drop=True)
    
    def _build_mid_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build OHLC using midPrice.
        
        Args:
            df: Tick data with bar_id and midPrice
            
        Returns:
            DataFrame with mid OHLC
        """
        agg_dict = {
            'timestamp': 'last',
            'midPrice': ['first', 'max', 'min', 'last'],
            'spread': ['mean', 'std'],
        }
        
        if 'bidVolume' in df.columns and 'askVolume' in df.columns:
            df['total_volume'] = df['bidVolume'] + df['askVolume']
            agg_dict['total_volume'] = 'sum'
        
        bars = df.groupby('bar_id').agg(agg_dict)
        
        bars.columns = ['_'.join(col) for col in bars.columns]
        bars = bars.rename(columns={
            'midPrice_first': 'open',
            'midPrice_max': 'high',
            'midPrice_min': 'low',
            'midPrice_last': 'close',
            'timestamp_last': 'timestamp',
        })
        
        return bars.reset_index(drop=True)
    
    def build_volume_bars(
        self,
        df: pd.DataFrame,
        target_volume: float,
        volume_column: str = 'bidVolume'
    ) -> pd.DataFrame:
        """Build volume bars (every N volume units).
        
        Args:
            df: Tick data with volume column
            target_volume: Target cumulative volume per bar
            volume_column: Volume column to use
            
        Returns:
            DataFrame with volume-based OHLC bars
        """
        logger.info(f"Building volume bars (target: {target_volume})")
        
        if volume_column not in df.columns:
            raise ValueError(f"Volume column '{volume_column}' not found")
        
        if df[volume_column].sum() == 0:
            raise ValueError("Total volume is zero, cannot build volume bars")
        
        # Compute cumulative volume and assign bar IDs
        df = df.copy()
        df['cum_volume'] = df[volume_column].cumsum()
        df['bar_id'] = (df['cum_volume'] // target_volume).astype(int)
        
        # Build OHLC using bid/ask
        bars = self._build_bid_ask_ohlc(df)
        
        # Volume per bar
        bars['volume'] = df.groupby('bar_id')[volume_column].sum().values
        bars['tick_count'] = df.groupby('bar_id').size().values
        
        logger.info(f"Created {len(bars)} volume bars")
        
        return bars


def build_bars_from_config(
    df: pd.DataFrame,
    bar_config: dict,
    bar_type: str
) -> pd.DataFrame:
    """Build bars according to configuration.
    
    Args:
        df: Clean tick data
        bar_config: Bar configuration from Hydra
        bar_type: Type of bar ('tick100', 'tick1000', 'volume_bar', etc.)
        
    Returns:
        DataFrame with bars
    """
    builder = BarBuilder(bar_config)
    
    if bar_type.startswith('tick'):
        tick_size = int(bar_type.replace('tick', ''))
        return builder.build_tick_bars(df, tick_size)
    
    elif bar_type == 'volume_bar':
        if not bar_config.get('volume', {}).get('enabled', False):
            raise ValueError("Volume bars not enabled in config")
        target_volume = bar_config['volume']['target_volume']
        return builder.build_volume_bars(df, target_volume)
    
    else:
        raise ValueError(f"Unknown bar type: {bar_type}")

