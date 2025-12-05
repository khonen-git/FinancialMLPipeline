"""Optimized bar construction with Numba acceleration.

This module provides optimized bar construction using:
- Numba JIT compilation for critical loops
- Vectorized NumPy operations
- Reduced pandas overhead
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from numba import jit, prange

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _aggregate_chunk_numba(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    has_volumes: bool
) -> Tuple[float, float, float, float, float, float, float, float, int, float, float, float, float]:
    """Aggregate tick chunk into OHLC bar (Numba-accelerated).
    
    Returns:
        (bid_open, bid_high, bid_low, bid_close,
         ask_open, ask_high, ask_low, ask_close,
         tick_count, bid_vol_sum, ask_vol_sum, spread_mean, spread_std)
    """
    n = len(bid_prices)
    
    # OHLC for bid
    bid_open = bid_prices[0]
    bid_close = bid_prices[n - 1]
    bid_high = bid_prices[0]
    bid_low = bid_prices[0]
    
    # OHLC for ask
    ask_open = ask_prices[0]
    ask_close = ask_prices[n - 1]
    ask_high = ask_prices[0]
    ask_low = ask_prices[0]
    
    # Compute high/low and spread stats
    spread_sum = 0.0
    spread_sum_sq = 0.0
    bid_vol_sum = 0.0
    ask_vol_sum = 0.0
    
    for i in prange(n):
        # High/Low
        if bid_prices[i] > bid_high:
            bid_high = bid_prices[i]
        if bid_prices[i] < bid_low:
            bid_low = bid_prices[i]
            
        if ask_prices[i] > ask_high:
            ask_high = ask_prices[i]
        if ask_prices[i] < ask_low:
            ask_low = ask_prices[i]
        
        # Spread
        spread = ask_prices[i] - bid_prices[i]
        spread_sum += spread
        spread_sum_sq += spread * spread
        
        # Volumes
        if has_volumes:
            bid_vol_sum += bid_volumes[i]
            ask_vol_sum += ask_volumes[i]
    
    # Spread statistics
    spread_mean = spread_sum / n if n > 0 else 0.0
    spread_variance = (spread_sum_sq / n) - (spread_mean * spread_mean) if n > 0 else 0.0
    spread_std = np.sqrt(spread_variance) if spread_variance > 0 else 0.0
    
    return (
        bid_open, bid_high, bid_low, bid_close,
        ask_open, ask_high, ask_low, ask_close,
        n, bid_vol_sum, ask_vol_sum, spread_mean, spread_std
    )


@jit(nopython=True, cache=True)
def _build_tick_bars_numba(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    timestamps: np.ndarray,
    threshold: int,
    has_volumes: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray]:
    """Build tick bars using Numba (vectorized).
    
    Returns:
        Tuple of arrays: (timestamps, bid_open, bid_high, bid_low, bid_close,
                         ask_open, ask_high, ask_low, ask_close,
                         tick_count, bid_vol_sum, ask_vol_sum, spread_mean, spread_std)
    """
    n_ticks = len(bid_prices)
    n_bars = (n_ticks // threshold) + (1 if n_ticks % threshold >= threshold else 0)
    
    # Pre-allocate arrays
    bar_timestamps = np.empty(n_bars, dtype=np.int64)
    bid_opens = np.empty(n_bars, dtype=np.float64)
    bid_highs = np.empty(n_bars, dtype=np.float64)
    bid_lows = np.empty(n_bars, dtype=np.float64)
    bid_closes = np.empty(n_bars, dtype=np.float64)
    ask_opens = np.empty(n_bars, dtype=np.float64)
    ask_highs = np.empty(n_bars, dtype=np.float64)
    ask_lows = np.empty(n_bars, dtype=np.float64)
    ask_closes = np.empty(n_bars, dtype=np.float64)
    tick_counts = np.empty(n_bars, dtype=np.int32)
    bid_vol_sums = np.empty(n_bars, dtype=np.float64)
    ask_vol_sums = np.empty(n_bars, dtype=np.float64)
    spread_means = np.empty(n_bars, dtype=np.float64)
    spread_stds = np.empty(n_bars, dtype=np.float64)
    
    bar_idx = 0
    for i in range(0, n_ticks, threshold):
        end_idx = min(i + threshold, n_ticks)
        chunk_size = end_idx - i
        
        if chunk_size < threshold:
            break  # Skip incomplete bars
        
        # Extract chunk
        chunk_bid = bid_prices[i:end_idx]
        chunk_ask = ask_prices[i:end_idx]
        chunk_bid_vol = bid_volumes[i:end_idx] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
        chunk_ask_vol = ask_volumes[i:end_idx] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
        
        # Aggregate
        result = _aggregate_chunk_numba(
            chunk_bid, chunk_ask, chunk_bid_vol, chunk_ask_vol, has_volumes
        )
        
        bid_open, bid_high, bid_low, bid_close, ask_open, ask_high, ask_low, ask_close, \
        tick_count, bid_vol_sum, ask_vol_sum, spread_mean, spread_std = result
        
        # Store
        bar_timestamps[bar_idx] = timestamps[i]
        bid_opens[bar_idx] = bid_open
        bid_highs[bar_idx] = bid_high
        bid_lows[bar_idx] = bid_low
        bid_closes[bar_idx] = bid_close
        ask_opens[bar_idx] = ask_open
        ask_highs[bar_idx] = ask_high
        ask_lows[bar_idx] = ask_low
        ask_closes[bar_idx] = ask_close
        tick_counts[bar_idx] = tick_count
        bid_vol_sums[bar_idx] = bid_vol_sum
        ask_vol_sums[bar_idx] = ask_vol_sum
        spread_means[bar_idx] = spread_mean
        spread_stds[bar_idx] = spread_std
        
        bar_idx += 1
    
    # Trim to actual size
    return (
        bar_timestamps[:bar_idx],
        bid_opens[:bar_idx], bid_highs[:bar_idx], bid_lows[:bar_idx], bid_closes[:bar_idx],
        ask_opens[:bar_idx], ask_highs[:bar_idx], ask_lows[:bar_idx], ask_closes[:bar_idx],
        tick_counts[:bar_idx],
        bid_vol_sums[:bar_idx], ask_vol_sums[:bar_idx],
        spread_means[:bar_idx], spread_stds[:bar_idx]
    )


@jit(nopython=True, cache=True)
def _build_volume_bars_numba(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    has_volumes: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Build volume bars using Numba."""
    n_ticks = len(bid_prices)
    
    # Estimate max bars (conservative)
    max_bars = n_ticks // 10  # Rough estimate
    
    # Pre-allocate arrays
    bar_timestamps = np.empty(max_bars, dtype=np.int64)
    bid_opens = np.empty(max_bars, dtype=np.float64)
    bid_highs = np.empty(max_bars, dtype=np.float64)
    bid_lows = np.empty(max_bars, dtype=np.float64)
    bid_closes = np.empty(max_bars, dtype=np.float64)
    ask_opens = np.empty(max_bars, dtype=np.float64)
    ask_highs = np.empty(max_bars, dtype=np.float64)
    ask_lows = np.empty(max_bars, dtype=np.float64)
    ask_closes = np.empty(max_bars, dtype=np.float64)
    tick_counts = np.empty(max_bars, dtype=np.int32)
    bid_vol_sums = np.empty(max_bars, dtype=np.float64)
    ask_vol_sums = np.empty(max_bars, dtype=np.float64)
    spread_means = np.empty(max_bars, dtype=np.float64)
    spread_stds = np.empty(max_bars, dtype=np.float64)
    
    bar_idx = 0
    chunk_start = 0
    cumulative_volume = 0.0
    
    for i in prange(n_ticks):
        vol = bid_volumes[i] + ask_volumes[i] if has_volumes else 0.0
        cumulative_volume += vol
        
        if cumulative_volume >= threshold:
            chunk_size = i - chunk_start + 1
            
            # Extract chunk
            chunk_bid = bid_prices[chunk_start:i+1]
            chunk_ask = ask_prices[chunk_start:i+1]
            chunk_bid_vol = bid_volumes[chunk_start:i+1] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
            chunk_ask_vol = ask_volumes[chunk_start:i+1] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
            
            # Aggregate
            result = _aggregate_chunk_numba(
                chunk_bid, chunk_ask, chunk_bid_vol, chunk_ask_vol, has_volumes
            )
            
            bid_open, bid_high, bid_low, bid_close, ask_open, ask_high, ask_low, ask_close, \
            tick_count, bid_vol_sum, ask_vol_sum, spread_mean, spread_std = result
            
            # Store
            bar_timestamps[bar_idx] = timestamps[chunk_start]
            bid_opens[bar_idx] = bid_open
            bid_highs[bar_idx] = bid_high
            bid_lows[bar_idx] = bid_low
            bid_closes[bar_idx] = bid_close
            ask_opens[bar_idx] = ask_open
            ask_highs[bar_idx] = ask_high
            ask_lows[bar_idx] = ask_low
            ask_closes[bar_idx] = ask_close
            tick_counts[bar_idx] = tick_count
            bid_vol_sums[bar_idx] = bid_vol_sum
            ask_vol_sums[bar_idx] = ask_vol_sum
            spread_means[bar_idx] = spread_mean
            spread_stds[bar_idx] = spread_std
            
            bar_idx += 1
            chunk_start = i + 1
            cumulative_volume = 0.0
    
    # Last chunk
    if chunk_start < n_ticks:
        chunk_size = n_ticks - chunk_start
        chunk_bid = bid_prices[chunk_start:]
        chunk_ask = ask_prices[chunk_start:]
        chunk_bid_vol = bid_volumes[chunk_start:] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
        chunk_ask_vol = ask_volumes[chunk_start:] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
        
        result = _aggregate_chunk_numba(
            chunk_bid, chunk_ask, chunk_bid_vol, chunk_ask_vol, has_volumes
        )
        
        bid_open, bid_high, bid_low, bid_close, ask_open, ask_high, ask_low, ask_close, \
        tick_count, bid_vol_sum, ask_vol_sum, spread_mean, spread_std = result
        
        if bar_idx < max_bars:
            bar_timestamps[bar_idx] = timestamps[chunk_start]
            bid_opens[bar_idx] = bid_open
            bid_highs[bar_idx] = bid_high
            bid_lows[bar_idx] = bid_low
            bid_closes[bar_idx] = bid_close
            ask_opens[bar_idx] = ask_open
            ask_highs[bar_idx] = ask_high
            ask_lows[bar_idx] = ask_low
            ask_closes[bar_idx] = ask_close
            tick_counts[bar_idx] = tick_count
            bid_vol_sums[bar_idx] = bid_vol_sum
            ask_vol_sums[bar_idx] = ask_vol_sum
            spread_means[bar_idx] = spread_mean
            spread_stds[bar_idx] = spread_std
            bar_idx += 1
    
    # Trim to actual size
    return (
        bar_timestamps[:bar_idx],
        bid_opens[:bar_idx], bid_highs[:bar_idx], bid_lows[:bar_idx], bid_closes[:bar_idx],
        ask_opens[:bar_idx], ask_highs[:bar_idx], ask_lows[:bar_idx], ask_closes[:bar_idx],
        tick_counts[:bar_idx],
        bid_vol_sums[:bar_idx], ask_vol_sums[:bar_idx],
        spread_means[:bar_idx], spread_stds[:bar_idx]
    )


@jit(nopython=True, cache=True)
def _build_dollar_bars_numba(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    has_volumes: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Build dollar bars using Numba."""
    n_ticks = len(bid_prices)
    
    # Estimate max bars
    max_bars = n_ticks // 10
    
    # Pre-allocate arrays
    bar_timestamps = np.empty(max_bars, dtype=np.int64)
    bid_opens = np.empty(max_bars, dtype=np.float64)
    bid_highs = np.empty(max_bars, dtype=np.float64)
    bid_lows = np.empty(max_bars, dtype=np.float64)
    bid_closes = np.empty(max_bars, dtype=np.float64)
    ask_opens = np.empty(max_bars, dtype=np.float64)
    ask_highs = np.empty(max_bars, dtype=np.float64)
    ask_lows = np.empty(max_bars, dtype=np.float64)
    ask_closes = np.empty(max_bars, dtype=np.float64)
    tick_counts = np.empty(max_bars, dtype=np.int32)
    bid_vol_sums = np.empty(max_bars, dtype=np.float64)
    ask_vol_sums = np.empty(max_bars, dtype=np.float64)
    spread_means = np.empty(max_bars, dtype=np.float64)
    spread_stds = np.empty(max_bars, dtype=np.float64)
    
    bar_idx = 0
    chunk_start = 0
    cumulative_dollar = 0.0
    
    for i in prange(n_ticks):
        if has_volumes:
            dollar_vol = bid_prices[i] * bid_volumes[i] + ask_prices[i] * ask_volumes[i]
        else:
            dollar_vol = 0.0
        cumulative_dollar += dollar_vol
        
        if cumulative_dollar >= threshold:
            chunk_size = i - chunk_start + 1
            
            # Extract chunk
            chunk_bid = bid_prices[chunk_start:i+1]
            chunk_ask = ask_prices[chunk_start:i+1]
            chunk_bid_vol = bid_volumes[chunk_start:i+1] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
            chunk_ask_vol = ask_volumes[chunk_start:i+1] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
            
            # Aggregate
            result = _aggregate_chunk_numba(
                chunk_bid, chunk_ask, chunk_bid_vol, chunk_ask_vol, has_volumes
            )
            
            bid_open, bid_high, bid_low, bid_close, ask_open, ask_high, ask_low, ask_close, \
            tick_count, bid_vol_sum, ask_vol_sum, spread_mean, spread_std = result
            
            # Store
            bar_timestamps[bar_idx] = timestamps[chunk_start]
            bid_opens[bar_idx] = bid_open
            bid_highs[bar_idx] = bid_high
            bid_lows[bar_idx] = bid_low
            bid_closes[bar_idx] = bid_close
            ask_opens[bar_idx] = ask_open
            ask_highs[bar_idx] = ask_high
            ask_lows[bar_idx] = ask_low
            ask_closes[bar_idx] = ask_close
            tick_counts[bar_idx] = tick_count
            bid_vol_sums[bar_idx] = bid_vol_sum
            ask_vol_sums[bar_idx] = ask_vol_sum
            spread_means[bar_idx] = spread_mean
            spread_stds[bar_idx] = spread_std
            
            bar_idx += 1
            chunk_start = i + 1
            cumulative_dollar = 0.0
    
    # Last chunk
    if chunk_start < n_ticks:
        chunk_size = n_ticks - chunk_start
        chunk_bid = bid_prices[chunk_start:]
        chunk_ask = ask_prices[chunk_start:]
        chunk_bid_vol = bid_volumes[chunk_start:] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
        chunk_ask_vol = ask_volumes[chunk_start:] if has_volumes else np.zeros(chunk_size, dtype=np.float64)
        
        result = _aggregate_chunk_numba(
            chunk_bid, chunk_ask, chunk_bid_vol, chunk_ask_vol, has_volumes
        )
        
        bid_open, bid_high, bid_low, bid_close, ask_open, ask_high, ask_low, ask_close, \
        tick_count, bid_vol_sum, ask_vol_sum, spread_mean, spread_std = result
        
        if bar_idx < max_bars:
            bar_timestamps[bar_idx] = timestamps[chunk_start]
            bid_opens[bar_idx] = bid_open
            bid_highs[bar_idx] = bid_high
            bid_lows[bar_idx] = bid_low
            bid_closes[bar_idx] = bid_close
            ask_opens[bar_idx] = ask_open
            ask_highs[bar_idx] = ask_high
            ask_lows[bar_idx] = ask_low
            ask_closes[bar_idx] = ask_close
            tick_counts[bar_idx] = tick_count
            bid_vol_sums[bar_idx] = bid_vol_sum
            ask_vol_sums[bar_idx] = ask_vol_sum
            spread_means[bar_idx] = spread_mean
            spread_stds[bar_idx] = spread_std
            bar_idx += 1
    
    # Trim to actual size
    return (
        bar_timestamps[:bar_idx],
        bid_opens[:bar_idx], bid_highs[:bar_idx], bid_lows[:bar_idx], bid_closes[:bar_idx],
        ask_opens[:bar_idx], ask_highs[:bar_idx], ask_lows[:bar_idx], ask_closes[:bar_idx],
        tick_counts[:bar_idx],
        bid_vol_sums[:bar_idx], ask_vol_sums[:bar_idx],
        spread_means[:bar_idx], spread_stds[:bar_idx]
    )


class BarBuilderOptimized:
    """Optimized bar builder with Numba acceleration."""
    
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
        """Build bars from ticks (optimized version).
        
        Args:
            ticks: Tick DataFrame with bid/ask prices and volumes
            
        Returns:
            DataFrame with OHLC bars
        """
        logger.info(f"Building {self.bar_type} bars (threshold={self.threshold}) [OPTIMIZED]")
        
        # Convert to numpy arrays for Numba
        bid_prices = ticks['bidPrice'].values
        ask_prices = ticks['askPrice'].values
        
        has_volumes = 'bidVolume' in ticks.columns and 'askVolume' in ticks.columns
        if has_volumes:
            bid_volumes = ticks['bidVolume'].values
            ask_volumes = ticks['askVolume'].values
        else:
            bid_volumes = np.zeros(len(ticks), dtype=np.float64)
            ask_volumes = np.zeros(len(ticks), dtype=np.float64)
        
        # Convert timestamps to int64 (nanoseconds since epoch)
        if isinstance(ticks.index, pd.DatetimeIndex):
            timestamps = ticks.index.astype(np.int64).values
        else:
            timestamps = pd.to_datetime(ticks.index).astype(np.int64).values
        
        # Call appropriate Numba function
        if self.bar_type == 'tick':
            results = _build_tick_bars_numba(
                bid_prices, ask_prices, bid_volumes, ask_volumes,
                timestamps, self.threshold, has_volumes
            )
        elif self.bar_type == 'volume':
            if not has_volumes:
                logger.warning("Volume data not available, falling back to tick bars")
                results = _build_tick_bars_numba(
                    bid_prices, ask_prices, bid_volumes, ask_volumes,
                    timestamps, self.threshold, has_volumes
                )
            else:
                results = _build_volume_bars_numba(
                    bid_prices, ask_prices, bid_volumes, ask_volumes,
                    timestamps, self.threshold, has_volumes
                )
        elif self.bar_type == 'dollar':
            if not has_volumes:
                logger.warning("Volume data not available, falling back to tick bars")
                results = _build_tick_bars_numba(
                    bid_prices, ask_prices, bid_volumes, ask_volumes,
                    timestamps, self.threshold, has_volumes
                )
            else:
                results = _build_dollar_bars_numba(
                    bid_prices, ask_prices, bid_volumes, ask_volumes,
                    timestamps, self.threshold, has_volumes
                )
        else:
            raise ValueError(f"Unknown bar type: {self.bar_type}")
        
        # Unpack results
        (bar_timestamps, bid_opens, bid_highs, bid_lows, bid_closes,
         ask_opens, ask_highs, ask_lows, ask_closes,
         tick_counts, bid_vol_sums, ask_vol_sums, spread_means, spread_stds) = results
        
        # Convert timestamps back to DatetimeIndex
        bar_timestamps_dt = pd.to_datetime(bar_timestamps)
        
        # Build DataFrame
        bars_df = pd.DataFrame({
            'bid_open': bid_opens,
            'bid_high': bid_highs,
            'bid_low': bid_lows,
            'bid_close': bid_closes,
            'ask_open': ask_opens,
            'ask_high': ask_highs,
            'ask_low': ask_lows,
            'ask_close': ask_closes,
            'tick_count': tick_counts,
        }, index=bar_timestamps_dt)
        
        # Add optional columns
        if has_volumes:
            bars_df['bidVolume_sum'] = bid_vol_sums
            bars_df['askVolume_sum'] = ask_vol_sums
        
        bars_df['spread_mean'] = spread_means
        bars_df['spread_std'] = spread_stds
        
        logger.info(f"Built {len(bars_df)} bars [OPTIMIZED]")
        
        return bars_df

