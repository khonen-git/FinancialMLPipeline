"""Optimized triple-barrier labeling with Numba acceleration.

This module provides an optimized version of compute_triple_barrier using Numba JIT
for the critical barrier checking loop.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np
from numba import jit, prange

from .session_calendar import SessionCalendar
from ..utils.helpers import convert_ticks_to_price

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _check_barriers_numba(
    bid_highs: np.ndarray,
    bid_lows: np.ndarray,
    tp_level: float,
    sl_level: float
) -> tuple[int, int]:
    """Check barriers using Numba (faster than numpy argmax for small arrays).
    
    Args:
        bid_highs: Array of bid high prices
        bid_lows: Array of bid low prices
        tp_level: Take-profit level
        sl_level: Stop-loss level
    
    Returns:
        Tuple of (tp_first_idx, sl_first_idx) or (-1, -1) if not hit
    """
    n = len(bid_highs)
    tp_first_idx = -1
    sl_first_idx = -1
    
    # Find first TP hit
    for i in prange(n):
        if bid_highs[i] >= tp_level:
            tp_first_idx = i
            break
    
    # Find first SL hit
    for i in prange(n):
        if bid_lows[i] <= sl_level:
            sl_first_idx = i
            break
    
    return tp_first_idx, sl_first_idx


@jit(nopython=True, cache=True)
def _compute_barriers_batch_numba(
    bar_indices: np.ndarray,
    effective_horizons: np.ndarray,
    bid_highs_all: np.ndarray,
    bid_lows_all: np.ndarray,
    bid_closes_all: np.ndarray,
    ask_closes_all: np.ndarray,
    tp_distances: np.ndarray,
    sl_distances: np.ndarray,
    min_horizon_bars: int,
    n_prices: int,
    labels: np.ndarray,
    exit_bar_indices: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    pnls: np.ndarray,
    effective_horizons_out: np.ndarray
) -> int:
    """Compute barriers for a batch of events using Numba.
    
    This function processes multiple events in a single Numba-compiled loop,
    avoiding Python overhead.
    
    Args:
        bar_indices: Array of start bar indices for each event
        effective_horizons: Array of effective horizons for each event
        bid_highs_all: All bid high prices (full array)
        bid_lows_all: All bid low prices (full array)
        bid_closes_all: All bid close prices (full array)
        ask_closes_all: All ask close prices (full array)
        tp_distances: Array of TP distances for each event
        sl_distances: Array of SL distances for each event
        min_horizon_bars: Minimum horizon threshold
        n_prices: Total number of price bars
        labels: Output array for labels (1=TP, -1=SL, 0=time)
        exit_bar_indices: Output array for exit bar indices
        entry_prices: Output array for entry prices
        exit_prices: Output array for exit prices
        pnls: Output array for PnL
        effective_horizons_out: Output array for effective horizons
    
    Returns:
        Number of valid labels created
    """
    n_events = len(bar_indices)
    valid_count = 0
    
    for i in prange(n_events):
        bar_idx_start = bar_indices[i]
        effective_horizon = effective_horizons[i]
        tp_distance = tp_distances[i]
        sl_distance = sl_distances[i]
        
        # Skip if effective horizon too short
        if effective_horizon < min_horizon_bars:
            continue
        
        # Bounds check
        if bar_idx_start >= n_prices:
            continue
        
        # Entry prices
        entry_price_ask = ask_closes_all[bar_idx_start]
        entry_price_bid = bid_closes_all[bar_idx_start]
        
        # Barrier levels
        tp_level = entry_price_bid + tp_distance
        sl_level = entry_price_bid - sl_distance
        
        # Scan window
        end_bar_idx = min(bar_idx_start + effective_horizon, n_prices - 1)
        start_slice = bar_idx_start + 1
        end_slice = end_bar_idx + 1
        
        if start_slice >= end_slice or end_slice > n_prices:
            continue
        
        # Extract slice
        slice_size = end_slice - start_slice
        bid_highs = bid_highs_all[start_slice:end_slice]
        bid_lows = bid_lows_all[start_slice:end_slice]
        
        # Check barriers
        tp_first_idx, sl_first_idx = _check_barriers_numba(bid_highs, bid_lows, tp_level, sl_level)
        
        # Determine exit
        if tp_first_idx >= 0 and (sl_first_idx < 0 or tp_first_idx <= sl_first_idx):
            # TP hit
            label = 1
            exit_bar_idx = start_slice + tp_first_idx
        elif sl_first_idx >= 0:
            # SL hit
            label = -1
            exit_bar_idx = start_slice + sl_first_idx
        else:
            # Time barrier
            label = 0
            exit_bar_idx = end_bar_idx
        
        # Exit price
        exit_price = bid_closes_all[exit_bar_idx]
        pnl = exit_price - entry_price_ask
        
        # Store results
        labels[valid_count] = label
        exit_bar_indices[valid_count] = exit_bar_idx
        entry_prices[valid_count] = entry_price_ask
        exit_prices[valid_count] = exit_price
        pnls[valid_count] = pnl
        effective_horizons_out[valid_count] = effective_horizon
        
        valid_count += 1
    
    return valid_count


def compute_triple_barrier_optimized(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    tp_distance: float,
    sl_distance: float,
    max_horizon_bars: int,
    session_calendar: SessionCalendar,
    min_horizon_bars: int = 10,
    avg_bar_duration_sec: Optional[float] = None,
) -> pd.DataFrame:
    """Optimized version of compute_triple_barrier using Numba.
    
    This function pre-computes session horizons and uses Numba for barrier checking.
    
    Args:
        events: DataFrame with columns: timestamp, bar_index
        prices: Bar DataFrame with bid/ask columns
        tp_distance: Take-profit distance
        sl_distance: Stop-loss distance
        max_horizon_bars: Maximum horizon
        session_calendar: SessionCalendar instance
        min_horizon_bars: Minimum horizon
        avg_bar_duration_sec: Average bar duration
    
    Returns:
        DataFrame with labels
    """
    logger.info(
        f"Computing triple barriers (OPTIMIZED) for {len(events)} events. "
        f"TP={tp_distance}, SL={sl_distance}, max_horizon={max_horizon_bars}"
    )
    
    if avg_bar_duration_sec is None:
        if len(prices) > 1:
            time_diff = prices.index[-1] - prices.index[0]
            avg_bar_duration_sec = time_diff.total_seconds() / len(prices)
        else:
            avg_bar_duration_sec = 60.0
    
    # Pre-compute session horizons for all events (Python overhead, but done once)
    n_events = len(events)
    bar_indices = np.empty(n_events, dtype=np.int32)
    effective_horizons = np.empty(n_events, dtype=np.int32)
    
    valid_events = []
    skipped_count = 0
    
    for idx, event_tuple in enumerate(events.itertuples()):
        t0 = event_tuple.timestamp
        bar_idx_start = event_tuple.bar_index
        
        # Ensure timezone-aware timestamp
        if isinstance(t0, pd.Timestamp):
            if t0.tz is None:
                # Assume UTC if naive
                t0 = t0.tz_localize('UTC')
            elif t0.tz != pd.Timestamp.now('UTC').tz:
                # Convert to UTC if different timezone
                t0 = t0.tz_convert('UTC')
        
        # Compute effective horizon
        bars_to_session_end = session_calendar.bars_until_session_end(
            t0, avg_bar_duration_sec
        )
        effective_horizon_bars = min(max_horizon_bars, bars_to_session_end)
        
        if effective_horizon_bars < min_horizon_bars:
            skipped_count += 1
            continue
        
        bar_indices[len(valid_events)] = bar_idx_start
        effective_horizons[len(valid_events)] = effective_horizon_bars
        valid_events.append((idx, t0, bar_idx_start))
    
    n_valid = len(valid_events)
    if n_valid == 0:
        logger.warning("No valid events after filtering")
        return pd.DataFrame()
    
    # Trim arrays to valid size
    bar_indices = bar_indices[:n_valid]
    effective_horizons = effective_horizons[:n_valid]
    
    # Extract price arrays (once, for Numba)
    bid_highs_all = prices['bid_high'].values
    bid_lows_all = prices['bid_low'].values
    bid_closes_all = prices['bid_close'].values
    ask_closes_all = prices['ask_close'].values
    
    # TP/SL distances (same for all events in this call)
    tp_distances = np.full(n_valid, tp_distance, dtype=np.float64)
    sl_distances = np.full(n_valid, sl_distance, dtype=np.float64)
    
    # Pre-allocate output arrays
    labels = np.empty(n_valid, dtype=np.int32)
    exit_bar_indices = np.empty(n_valid, dtype=np.int32)
    entry_prices = np.empty(n_valid, dtype=np.float64)
    exit_prices = np.empty(n_valid, dtype=np.float64)
    pnls = np.empty(n_valid, dtype=np.float64)
    effective_horizons_out = np.empty(n_valid, dtype=np.int32)
    
    # Call Numba function
    valid_count = _compute_barriers_batch_numba(
        bar_indices, effective_horizons,
        bid_highs_all, bid_lows_all, bid_closes_all, ask_closes_all,
        tp_distances, sl_distances,
        min_horizon_bars, len(prices),
        labels, exit_bar_indices, entry_prices, exit_prices, pnls, effective_horizons_out
    )
    
    # Build results DataFrame
    results = []
    for i in range(valid_count):
        orig_idx, t0, bar_idx_start = valid_events[i]
        exit_bar_idx = exit_bar_indices[i]
        exit_timestamp = prices.index[exit_bar_idx]
        
        # Map label to barrier_hit string
        label_val = labels[i]
        if label_val == 1:
            barrier_hit = 'tp'
        elif label_val == -1:
            barrier_hit = 'sl'
        else:
            barrier_hit = 'time'
        
        results.append({
            'event_start': t0,
            'event_end': exit_timestamp,
            'label': label_val,
            'bar_index_start': bar_idx_start,
            'bar_index_end': exit_bar_idx,
            'entry_price': entry_prices[i],
            'exit_price': exit_prices[i],
            'pnl': pnls[i],
            'barrier_hit': barrier_hit,
            'effective_horizon': effective_horizons_out[i],
        })
    
    logger.info(
        f"Triple barrier (OPTIMIZED) complete. Labels: {len(results)}, "
        f"Skipped: {skipped_count}"
    )
    
    return pd.DataFrame(results)

