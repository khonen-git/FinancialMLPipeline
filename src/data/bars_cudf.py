"""Bar construction using cuDF (GPU-accelerated)."""

import logging
import pandas as pd
import numpy as np
from typing import Optional

try:
    import cudf
    _cudf_available = True
except ImportError:
    _cudf_available = False
    cudf = None

logger = logging.getLogger(__name__)


class BarBuilderCuDF:
    """Bar builder using cuDF for GPU acceleration."""
    
    def __init__(self, config: dict):
        """Initialize bar builder.
        
        Args:
            config: Bar configuration
        """
        if not _cudf_available:
            raise ImportError("cuDF is not installed. Please install it to use BarBuilderCuDF.")
        
        self.config = config
        if 'type' not in config:
            raise ValueError("Missing required config: type")
        if 'threshold' not in config:
            raise ValueError("Missing required config: threshold")
        self.bar_type = config['type']
        self.threshold = config['threshold']
    
    def build_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Build bars from ticks using cuDF.
        
        Args:
            ticks: Tick DataFrame with bid/ask prices and volumes
            
        Returns:
            DataFrame with OHLC bars
        """
        logger.info(f"Building {self.bar_type} bars (threshold={self.threshold}) [CUDF/GPU]")
        
        # Convert pandas to cuDF (preserve timestamp from index)
        ticks_cudf = cudf.from_pandas(ticks.reset_index())
        # Ensure timestamp column exists
        if 'timestamp' not in ticks_cudf.columns and 'index' in ticks_cudf.columns:
            ticks_cudf = ticks_cudf.rename(columns={'index': 'timestamp'})
        
        if self.bar_type == 'tick':
            bars_cudf = self._build_tick_bars(ticks_cudf)
        elif self.bar_type == 'volume':
            bars_cudf = self._build_volume_bars(ticks_cudf)
        elif self.bar_type == 'dollar':
            bars_cudf = self._build_dollar_bars(ticks_cudf)
        else:
            raise ValueError(f"Unknown bar type: {self.bar_type}")
        
        # Convert back to pandas
        bars_df = bars_cudf.to_pandas()
        if 'timestamp' in bars_df.columns:
            bars_df = bars_df.set_index('timestamp')
        
        logger.info(f"Built {len(bars_df)} bars [CUDF/GPU]")
        
        return bars_df
    
    def _build_tick_bars(self, ticks_cudf: cudf.DataFrame) -> cudf.DataFrame:
        """Build tick bars using cuDF."""
        # Add bar_id column
        ticks_cudf = ticks_cudf.reset_index(drop=True)
        ticks_cudf['bar_id'] = ticks_cudf.index // self.threshold
        
        # Group by bar_id and aggregate (cuDF uses different syntax)
        bars_cudf = ticks_cudf.groupby('bar_id', as_index=False).agg({
            'timestamp': 'first',
            'bidPrice': ['first', 'max', 'min', 'last'],
            'askPrice': ['first', 'max', 'min', 'last'],
        })
        
        # Flatten column names (cuDF returns MultiIndex)
        if isinstance(bars_cudf.columns, pd.MultiIndex):
            bars_cudf.columns = [
                'bar_id', 'timestamp', 'bid_open', 'bid_high', 'bid_low', 'bid_close',
                'ask_open', 'ask_high', 'ask_low', 'ask_close'
            ]
        else:
            # Fallback if columns are already flat
            bars_cudf.columns = [
                'bar_id', 'timestamp', 'bid_open', 'bid_high', 'bid_low', 'bid_close',
                'ask_open', 'ask_high', 'ask_low', 'ask_close'
            ]
        
        # Add tick_count
        tick_counts = ticks_cudf.groupby('bar_id').size().reset_index(name='tick_count')
        bars_cudf = bars_cudf.merge(tick_counts, on='bar_id', how='left')
        
        # Add volume columns if available
        if 'bidVolume' in ticks_cudf.columns:
            bid_vol_sums = ticks_cudf.groupby('bar_id')['bidVolume'].sum().reset_index(name='bidVolume_sum')
            bars_cudf = bars_cudf.merge(bid_vol_sums, on='bar_id', how='left')
        
        if 'askVolume' in ticks_cudf.columns:
            ask_vol_sums = ticks_cudf.groupby('bar_id')['askVolume'].sum().reset_index(name='askVolume_sum')
            bars_cudf = bars_cudf.merge(ask_vol_sums, on='bar_id', how='left')
        
        # Spread statistics
        ticks_cudf['spread'] = ticks_cudf['askPrice'] - ticks_cudf['bidPrice']
        spread_stats = ticks_cudf.groupby('bar_id')['spread'].agg(['mean', 'std']).reset_index()
        spread_stats.columns = ['bar_id', 'spread_mean', 'spread_std']
        bars_cudf = bars_cudf.merge(spread_stats, on='bar_id', how='left')
        
        # Drop bar_id and sort by timestamp
        bars_cudf = bars_cudf.drop(columns=['bar_id']).sort_values('timestamp')
        
        return bars_cudf
    
    def _build_volume_bars(self, ticks_cudf: cudf.DataFrame) -> cudf.DataFrame:
        """Build volume bars using cuDF."""
        if 'bidVolume' not in ticks_cudf.columns:
            logger.warning("Volume data not available, falling back to tick bars")
            return self._build_tick_bars(ticks_cudf)
        
        # Compute cumulative volume
        ticks_cudf['total_volume'] = ticks_cudf['bidVolume'] + ticks_cudf.get('askVolume', 0).fillna(0)
        ticks_cudf['cumulative_volume'] = ticks_cudf['total_volume'].cumsum()
        
        # Assign bar_id
        ticks_cudf['bar_id'] = (ticks_cudf['cumulative_volume'] // self.threshold).astype(int)
        
        # Group and aggregate
        return self._build_tick_bars(ticks_cudf)
    
    def _build_dollar_bars(self, ticks_cudf: cudf.DataFrame) -> cudf.DataFrame:
        """Build dollar bars using cuDF."""
        if 'bidVolume' not in ticks_cudf.columns:
            logger.warning("Volume data not available, falling back to tick bars")
            return self._build_tick_bars(ticks_cudf)
        
        # Compute dollar volume
        ticks_cudf['dollar_volume'] = (
            ticks_cudf['bidPrice'] * ticks_cudf['bidVolume'] +
            ticks_cudf['askPrice'] * ticks_cudf.get('askVolume', 0).fillna(0)
        )
        
        # Cumulative dollar volume
        ticks_cudf['cumulative_dollar'] = ticks_cudf['dollar_volume'].cumsum()
        
        # Assign bar_id
        ticks_cudf['bar_id'] = (ticks_cudf['cumulative_dollar'] // self.threshold).astype(int)
        
        # Group and aggregate
        return self._build_tick_bars(ticks_cudf)

