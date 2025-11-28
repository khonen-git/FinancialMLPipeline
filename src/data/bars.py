"""Bar construction from tick data.

Implements multiple bar types:
- Tick bars
- Volume bars
- Dollar bars
- Time bars (optional)
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BarBuilder:
    """Constructs bars from tick data."""
    
    def __init__(self, config: dict):
        """Initialize bar builder.
        
        Args:
            config: Bar configuration
        """
        self.config = config
        self.bar_type = config.get('type', 'tick')
        self.threshold = config.get('threshold', 1000)
    
    def build_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Build bars from ticks.
        
        Args:
            ticks: Tick DataFrame with bid/ask prices and volumes
            
        Returns:
            DataFrame with OHLC bars
        """
        logger.info(f"Building {self.bar_type} bars (threshold={self.threshold})")
        
        if self.bar_type == 'tick':
            bars = self._build_tick_bars(ticks)
        elif self.bar_type == 'volume':
            bars = self._build_volume_bars(ticks)
        elif self.bar_type == 'dollar':
            bars = self._build_dollar_bars(ticks)
        else:
            raise ValueError(f"Unknown bar type: {self.bar_type}")
        
        logger.info(f"Built {len(bars)} bars")
        
        return bars
    
    def _build_tick_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Build tick bars (every N ticks).
        
        Args:
            ticks: Tick DataFrame
            
        Returns:
            OHLC bars
        """
        bars = []
        
        for i in range(0, len(ticks), self.threshold):
            chunk = ticks.iloc[i:i + self.threshold]
            
            # Only create bar if we have exactly the threshold (or it's the last chunk with enough ticks)
            if len(chunk) < self.threshold and i + len(chunk) < len(ticks):
                continue
            
            if len(chunk) == 0:
                continue
            
            # For consistency, only create bars with at least threshold ticks
            # (don't create partial bars at the end)
            if len(chunk) < self.threshold:
                continue
            
            bar = self._aggregate_chunk(chunk)
            bars.append(bar)
        
        # Create DataFrame and set timestamp as index
        bars_df = pd.DataFrame(bars)
        if 'timestamp' in bars_df.columns and not bars_df.empty:
            bars_df = bars_df.set_index('timestamp')
        return bars_df
    
    def _build_volume_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Build volume bars (cumulative volume threshold).
        
        Args:
            ticks: Tick DataFrame
            
        Returns:
            OHLC bars
        """
        if 'bidVolume' not in ticks.columns:
            logger.warning("Volume data not available, falling back to tick bars")
            return self._build_tick_bars(ticks)
        
        bars = []
        current_chunk = []
        cumulative_volume = 0
        
        for idx, row in ticks.iterrows():
            current_chunk.append(row)
            cumulative_volume += row['bidVolume'] + row.get('askVolume', 0)
            
            if cumulative_volume >= self.threshold:
                bar = self._aggregate_chunk(pd.DataFrame(current_chunk))
                bars.append(bar)
                
                current_chunk = []
                cumulative_volume = 0
        
        # Last chunk
        if len(current_chunk) > 0:
            bar = self._aggregate_chunk(pd.DataFrame(current_chunk))
            bars.append(bar)
        
        return pd.DataFrame(bars)
    
    def _build_dollar_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Build dollar bars (cumulative dollar volume threshold).
        
        Args:
            ticks: Tick DataFrame
            
        Returns:
            OHLC bars
        """
        if 'bidVolume' not in ticks.columns:
            logger.warning("Volume data not available, falling back to tick bars")
            return self._build_tick_bars(ticks)
        
        bars = []
        current_chunk = []
        cumulative_dollar = 0
        
        for idx, row in ticks.iterrows():
            current_chunk.append(row)
            
            # Dollar volume = price * volume
            dollar_vol = row['bidPrice'] * row['bidVolume']
            if 'askVolume' in row:
                dollar_vol += row['askPrice'] * row['askVolume']
            
            cumulative_dollar += dollar_vol
            
            if cumulative_dollar >= self.threshold:
                bar = self._aggregate_chunk(pd.DataFrame(current_chunk))
                bars.append(bar)
                
                current_chunk = []
                cumulative_dollar = 0
        
        # Last chunk
        if len(current_chunk) > 0:
            bar = self._aggregate_chunk(pd.DataFrame(current_chunk))
            bars.append(bar)
        
        return pd.DataFrame(bars)
    
    def _aggregate_chunk(self, chunk: pd.DataFrame) -> dict:
        """Aggregate tick chunk into OHLC bar.
        
        Args:
            chunk: DataFrame with ticks
            
        Returns:
            Dictionary with bar data
        """
        # Get timestamp - should be datetime already from index
        timestamp = chunk.index[0]
        if not isinstance(timestamp, pd.Timestamp):
            # Convert if needed
            timestamp = pd.Timestamp(timestamp)
        
        bar = {
            'timestamp': timestamp,
            'bid_open': chunk['bidPrice'].iloc[0],
            'bid_high': chunk['bidPrice'].max(),
            'bid_low': chunk['bidPrice'].min(),
            'bid_close': chunk['bidPrice'].iloc[-1],
            'ask_open': chunk['askPrice'].iloc[0],
            'ask_high': chunk['askPrice'].max(),
            'ask_low': chunk['askPrice'].min(),
            'ask_close': chunk['askPrice'].iloc[-1],
            'tick_count': len(chunk)
        }
        
        # Volume aggregation if available
        if 'bidVolume' in chunk.columns:
            bar['bidVolume_sum'] = chunk['bidVolume'].sum()
        
        if 'askVolume' in chunk.columns:
            bar['askVolume_sum'] = chunk['askVolume'].sum()
        
        # Spread statistics
        spread = chunk['askPrice'] - chunk['bidPrice']
        bar['spread_mean'] = spread.mean()
        bar['spread_std'] = spread.std()
        
        return bar

