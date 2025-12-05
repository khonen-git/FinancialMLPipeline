"""Bar construction using Polars (faster than pandas)."""

import logging
import polars as pl
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class BarBuilderPolars:
    """Bar builder using Polars for performance."""
    
    def __init__(self, config: dict):
        """Initialize bar builder.
        
        Args:
            config: Bar configuration
        """
        self.config = config
        if 'type' not in config:
            raise ValueError("Missing required config: type")
        if 'threshold' not in config:
            raise ValueError("Missing required config: threshold")
        self.bar_type = config['type']
        self.threshold = config['threshold']
    
    def build_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Build bars from ticks using Polars.
        
        Args:
            ticks: Tick DataFrame with bid/ask prices and volumes
            
        Returns:
            DataFrame with OHLC bars
        """
        logger.info(f"Building {self.bar_type} bars (threshold={self.threshold}) [POLARS]")
        
        # Convert pandas to Polars (preserve timestamp from index)
        ticks_pl = pl.from_pandas(ticks.reset_index())
        # Ensure timestamp column exists
        if 'timestamp' not in ticks_pl.columns and 'index' in ticks_pl.columns:
            ticks_pl = ticks_pl.rename({'index': 'timestamp'})
        
        if self.bar_type == 'tick':
            bars_pl = self._build_tick_bars(ticks_pl)
        elif self.bar_type == 'volume':
            bars_pl = self._build_volume_bars(ticks_pl)
        elif self.bar_type == 'dollar':
            bars_pl = self._build_dollar_bars(ticks_pl)
        else:
            raise ValueError(f"Unknown bar type: {self.bar_type}")
        
        # Convert back to pandas
        bars_df = bars_pl.to_pandas()
        if 'timestamp' in bars_df.columns:
            bars_df = bars_df.set_index('timestamp')
        
        logger.info(f"Built {len(bars_df)} bars [POLARS]")
        
        return bars_df
    
    def _build_tick_bars(self, ticks_pl: pl.DataFrame) -> pl.DataFrame:
        """Build tick bars using Polars."""
        # Add bar_id column
        ticks_pl = ticks_pl.with_columns([
            (pl.int_range(pl.len()).alias("row_id") // self.threshold).alias("bar_id")
        ])
        
        # Group by bar_id and aggregate
        bars_pl = ticks_pl.group_by("bar_id").agg([
            pl.first("timestamp").alias("timestamp"),
            pl.first("bidPrice").alias("bid_open"),
            pl.max("bidPrice").alias("bid_high"),
            pl.min("bidPrice").alias("bid_low"),
            pl.last("bidPrice").alias("bid_close"),
            pl.first("askPrice").alias("ask_open"),
            pl.max("askPrice").alias("ask_high"),
            pl.min("askPrice").alias("ask_low"),
            pl.last("askPrice").alias("ask_close"),
            pl.count().alias("tick_count"),
        ])
        
        # Add volume columns if available
        if "bidVolume" in ticks_pl.columns:
            bars_pl = bars_pl.join(
                ticks_pl.group_by("bar_id").agg([
                    pl.sum("bidVolume").alias("bidVolume_sum"),
                ]),
                on="bar_id"
            )
        
        if "askVolume" in ticks_pl.columns:
            bars_pl = bars_pl.join(
                ticks_pl.group_by("bar_id").agg([
                    pl.sum("askVolume").alias("askVolume_sum"),
                ]),
                on="bar_id"
            )
        
        # Spread statistics
        spread = ticks_pl.with_columns([
            (pl.col("askPrice") - pl.col("bidPrice")).alias("spread")
        ])
        
        spread_stats = spread.group_by("bar_id").agg([
            pl.mean("spread").alias("spread_mean"),
            pl.std("spread").alias("spread_std"),
        ])
        
        bars_pl = bars_pl.join(spread_stats, on="bar_id")
        
        # Drop bar_id and sort by timestamp
        bars_pl = bars_pl.drop("bar_id").sort("timestamp")
        
        return bars_pl
    
    def _build_volume_bars(self, ticks_pl: pl.DataFrame) -> pl.DataFrame:
        """Build volume bars using Polars."""
        if "bidVolume" not in ticks_pl.columns:
            logger.warning("Volume data not available, falling back to tick bars")
            return self._build_tick_bars(ticks_pl)
        
        # Compute cumulative volume
        ticks_pl = ticks_pl.with_columns([
            (pl.col("bidVolume") + pl.col("askVolume").fill_null(0)).alias("total_volume")
        ])
        
        ticks_pl = ticks_pl.with_columns([
            pl.cumsum("total_volume").alias("cumulative_volume")
        ])
        
        # Assign bar_id based on cumulative volume threshold
        ticks_pl = ticks_pl.with_columns([
            (pl.col("cumulative_volume") // self.threshold).alias("bar_id")
        ])
        
        # Group and aggregate (same as tick bars)
        return self._build_tick_bars(ticks_pl)
    
    def _build_dollar_bars(self, ticks_pl: pl.DataFrame) -> pl.DataFrame:
        """Build dollar bars using Polars."""
        if "bidVolume" not in ticks_pl.columns:
            logger.warning("Volume data not available, falling back to tick bars")
            return self._build_tick_bars(ticks_pl)
        
        # Compute dollar volume
        ticks_pl = ticks_pl.with_columns([
            (pl.col("bidPrice") * pl.col("bidVolume") + 
             pl.col("askPrice") * pl.col("askVolume").fill_null(0)).alias("dollar_volume")
        ])
        
        # Cumulative dollar volume
        ticks_pl = ticks_pl.with_columns([
            pl.cumsum("dollar_volume").alias("cumulative_dollar")
        ])
        
        # Assign bar_id
        ticks_pl = ticks_pl.with_columns([
            (pl.col("cumulative_dollar") // self.threshold).alias("bar_id")
        ])
        
        # Group and aggregate
        return self._build_tick_bars(ticks_pl)

