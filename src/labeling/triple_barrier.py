"""Triple barrier labeling with session-aware logic.

Implements Lopez de Prado's triple barrier method with:
- Upper barrier (take profit)
- Lower barrier (stop loss)
- Time barrier (session-aware, no overnight)

According to ARCH_DATA_PIPELINE.md §6 and validated in DEVBOOK.
"""

import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from .session_calendar import SessionCalendar

logger = logging.getLogger(__name__)


def compute_triple_barrier(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    tp_distance: float,
    sl_distance: float,
    max_horizon_bars: int,
    session_calendar: SessionCalendar,
    min_horizon_bars: int = 10,
    avg_bar_duration_sec: Optional[float] = None
) -> pd.DataFrame:
    """Compute triple-barrier labels for long-only events with session-aware logic.
    
    Args:
        events: DataFrame with columns ['timestamp', 'bar_index']
            - timestamp: event start time
            - bar_index: index in prices DataFrame
        prices: DataFrame with columns ['timestamp', 'bid_close', 'ask_close', 'bid_high', 'bid_low']
        tp_distance: Take profit distance in price units
        sl_distance: Stop loss distance in price units
        max_horizon_bars: Maximum holding period in bars
        session_calendar: SessionCalendar instance for session-aware logic
        min_horizon_bars: Minimum horizon required (no-trade zone threshold)
        avg_bar_duration_sec: Average bar duration in seconds (for session calc)
        
    Returns:
        DataFrame with columns:
            - event_start: original event timestamp
            - event_end: exit timestamp (t1)
            - label: {-1, 0, 1} for SL/time/TP
            - bar_index_start: start bar index
            - bar_index_end: exit bar index
            - exit_price: actual exit price (bid)
            - pnl: profit/loss
            - barrier_hit: 'tp' | 'sl' | 'time'
    
    Notes:
        - Entry price (long): askPrice at t0
        - Exit price: bidPrice at t1
        - TP barrier: bid >= entry_price + tp_distance
        - SL barrier: bid <= entry_price - sl_distance
        - Time barrier: capped by session_end (no overnight)
        - If effective_horizon < min_horizon_bars: event is SKIPPED (NaN label)
    """
    logger.info(
        f"Computing triple barriers for {len(events)} events. "
        f"TP={tp_distance}, SL={sl_distance}, max_horizon={max_horizon_bars}"
    )
    
    if avg_bar_duration_sec is None:
        # Estimate from prices DataFrame (timestamp is in index)
        if len(prices) > 1:
            time_diff = (prices.index[-1] - prices.index[0])
            avg_bar_duration_sec = time_diff.total_seconds() / len(prices)
        else:
            avg_bar_duration_sec = 60.0  # fallback: 1 minute
    
    results = []
    skipped_count = 0
    
    for idx, event in events.iterrows():
        t0 = event['timestamp']
        bar_idx_start = event['bar_index']
        
        # Session-aware horizon calculation
        bars_to_session_end = session_calendar.bars_until_session_end(
            t0, avg_bar_duration_sec
        )
        
        # Effective horizon = min of max_horizon and bars until session end
        effective_horizon_bars = min(max_horizon_bars, bars_to_session_end)
        
        # NO-TRADE ZONE: skip if effective horizon < min_horizon_bars
        if effective_horizon_bars < min_horizon_bars:
            logger.debug(
                f"Event at {t0} skipped: effective_horizon={effective_horizon_bars} "
                f"< min_horizon={min_horizon_bars}"
            )
            skipped_count += 1
            continue
        
        # Entry price = ask at t0
        entry_price = prices.iloc[bar_idx_start]['ask_close']
        
        # Scan forward up to effective_horizon_bars
        end_bar_idx = min(bar_idx_start + effective_horizon_bars, len(prices) - 1)
        
        # Get future price slice
        future_prices = prices.iloc[bar_idx_start + 1 : end_bar_idx + 1]
        
        if len(future_prices) == 0:
            # No future data available
            skipped_count += 1
            continue
        
        # Check barriers
        label, barrier_hit, exit_bar_idx = _check_barriers(
            entry_price=entry_price,
            tp_distance=tp_distance,
            sl_distance=sl_distance,
            future_prices=future_prices,
            start_bar_idx=bar_idx_start
        )
        
        # Exit price = bid at exit time
        exit_price = prices.iloc[exit_bar_idx]['bid_close']
        exit_timestamp = prices.index[exit_bar_idx]  # timestamp is in index now
        
        # Compute PnL
        pnl = exit_price - entry_price
        
        results.append({
            'event_start': t0,
            'event_end': exit_timestamp,
            'label': label,
            'bar_index_start': bar_idx_start,
            'bar_index_end': exit_bar_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'barrier_hit': barrier_hit,
            'effective_horizon': effective_horizon_bars,
        })
    
    logger.info(
        f"Triple barrier complete. Labels: {len(results)}, "
        f"Skipped (no-trade zone): {skipped_count}"
    )
    
    return pd.DataFrame(results)


def _check_barriers(
    entry_price: float,
    tp_distance: float,
    sl_distance: float,
    future_prices: pd.DataFrame,
    start_bar_idx: int
) -> Tuple[int, str, int]:
    """Check which barrier is hit first.
    
    Args:
        entry_price: Entry price (ask at t0)
        tp_distance: Take profit distance
        sl_distance: Stop loss distance
        future_prices: DataFrame slice of future bars
        start_bar_idx: Starting bar index
        
    Returns:
        Tuple of (label, barrier_hit, exit_bar_idx)
            label: 1 (TP), -1 (SL), 0 (time)
            barrier_hit: 'tp' | 'sl' | 'time'
            exit_bar_idx: index in original prices DataFrame
    """
    tp_level = entry_price + tp_distance
    sl_level = entry_price - sl_distance
    
    # For each bar, check if TP or SL hit using bid high/low
    for i, (idx, row) in enumerate(future_prices.iterrows()):
        # TP check: bid_high >= tp_level
        if row['bid_high'] >= tp_level:
            return 1, 'tp', idx
        
        # SL check: bid_low <= sl_level
        if row['bid_low'] <= sl_level:
            return -1, 'sl', idx
    
    # Time barrier hit (neither TP nor SL)
    last_idx = future_prices.index[-1]
    return 0, 'time', last_idx


def convert_ticks_to_price(
    ticks: int,
    tick_size: float
) -> float:
    """Convert tick distance to price distance.
    
    Args:
        ticks: Number of ticks
        tick_size: Size of one tick (from asset config)
        
    Returns:
        Price distance
    """
    return ticks * tick_size


def compute_labels_from_config(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    labeling_config: dict,
    session_config: dict,
    asset_config: dict,
    avg_bar_duration_sec: Optional[float] = None
) -> pd.DataFrame:
    """Compute triple barrier labels from Hydra configuration.
    
    Args:
        events: Events DataFrame
        prices: Prices DataFrame
        labeling_config: Labeling configuration from Hydra
        session_config: Session configuration from Hydra
        asset_config: Asset configuration (for tick_size)
        avg_bar_duration_sec: Average bar duration
        
    Returns:
        Labels DataFrame
    """
    # Create session calendar
    session_calendar = SessionCalendar(session_config)
    
    # Parse distance mode
    distance_mode = labeling_config.get('distance_mode', 'ticks')
    
    if distance_mode == 'ticks':
        tick_size = asset_config['tick_size']
        tp_distance = convert_ticks_to_price(
            labeling_config['tp_ticks'], tick_size
        )
        sl_distance = convert_ticks_to_price(
            labeling_config['sl_ticks'], tick_size
        )
    elif distance_mode == 'absolute':
        tp_distance = labeling_config['tp_absolute']
        sl_distance = labeling_config['sl_absolute']
    elif distance_mode == 'volatility':
        # Would need to compute from data
        raise NotImplementedError("Volatility-based barriers not yet implemented")
    else:
        raise ValueError(f"Unknown distance_mode: {distance_mode}")
    
    # Get horizon parameters
    max_horizon_bars = labeling_config['max_horizon_bars']
    min_horizon_bars = labeling_config['min_horizon_bars']
    
    # Compute labels
    labels = compute_triple_barrier(
        events=events,
        prices=prices,
        tp_distance=tp_distance,
        sl_distance=sl_distance,
        max_horizon_bars=max_horizon_bars,
        session_calendar=session_calendar,
        min_horizon_bars=min_horizon_bars,
        avg_bar_duration_sec=avg_bar_duration_sec
    )
    
    return labels


class TripleBarrierLabeler:
    """Class wrapper for triple barrier labeling functionality.
    
    Provides an object-oriented interface to the triple barrier functions.
    """
    
    def __init__(self, config: dict, session_calendar: SessionCalendar):
        """Initialize labeler with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - tp_ticks: Take profit in ticks
                - sl_ticks: Stop loss in ticks
                - max_horizon_bars: Maximum holding period
                - min_horizon_bars: Minimum horizon (no-trade threshold)
                - distance_mode: 'ticks' or 'absolute'
                - tick_size: Tick size for 'ticks' mode (default 0.0001)
            session_calendar: SessionCalendar instance
        """
        self.config = config
        self.calendar = session_calendar
        
        self.tp_ticks = config.get('tp_ticks', 100)
        self.sl_ticks = config.get('sl_ticks', 100)
        self.max_horizon_bars = config.get('max_horizon_bars', 50)
        self.min_horizon_bars = config.get('min_horizon_bars', 10)
        self.distance_mode = config.get('distance_mode', 'ticks')
        self.tick_size = config.get('tick_size', 0.0001)
        
        # Compute distances
        if self.distance_mode == 'ticks':
            self.tp_distance = self.tp_ticks * self.tick_size
            self.sl_distance = self.sl_ticks * self.tick_size
        else:
            self.tp_distance = self.tp_ticks  # Assume absolute
            self.sl_distance = self.sl_ticks
    
    def label_dataset(
        self,
        bars: pd.DataFrame,
        event_indices: pd.Index,
        avg_bar_duration_sec: Optional[float] = None
    ) -> pd.DataFrame:
        """Label a dataset using triple barrier method.
        
        Args:
            bars: DataFrame with OHLC bars
            event_indices: Indices to label (from features DataFrame)
            avg_bar_duration_sec: Average bar duration (optional)
            
        Returns:
            DataFrame with labels
        """
        # Ensure event_indices is DatetimeIndex
        if not isinstance(event_indices, pd.DatetimeIndex):
            if hasattr(bars, 'index') and isinstance(bars.index, pd.DatetimeIndex):
                # Use bars index which should be datetime
                event_indices = bars.index[event_indices] if len(event_indices) <= len(bars) else bars.index
            else:
                raise ValueError("Cannot convert event_indices to DatetimeIndex. Bars must have DatetimeIndex.")
        
        # Create events DataFrame
        events = pd.DataFrame({
            'timestamp': event_indices,
            'bar_index': range(len(event_indices))
        })
        
        # Use compute_triple_barrier function
        labels = compute_triple_barrier(
            events=events,
            prices=bars,
            tp_distance=self.tp_distance,
            sl_distance=self.sl_distance,
            max_horizon_bars=self.max_horizon_bars,
            session_calendar=self.calendar,
            min_horizon_bars=self.min_horizon_bars,
            avg_bar_duration_sec=avg_bar_duration_sec
        )
        
        return labels
    
    def _compute_effective_horizon_bars(
        self,
        dt: pd.Timestamp,
        bar_duration_minutes: int = 5
    ) -> int:
        """Compute effective horizon considering session end.
        
        Args:
            dt: Event start time
            bar_duration_minutes: Bar duration in minutes
            
        Returns:
            Effective horizon in bars
        """
        # Time until session end in minutes
        time_until_session = self.calendar.time_until_session_end(dt, unit='minutes')
        
        # Convert to bars
        bars_until_session = int(time_until_session / bar_duration_minutes)
        
        # Effective horizon is minimum
        return min(self.max_horizon_bars, bars_until_session)
    
    def _label_single_event(
        self,
        event_time: pd.Timestamp,
        event_idx: int,
        bars: pd.DataFrame
    ) -> Optional[dict]:
        """Label a single event (for testing).
        
        Args:
            event_time: Event start time
            event_idx: Event index in bars
            bars: Bars DataFrame
            
        Returns:
            Dictionary with label info or None if skipped
        """
        # Check if too close to session end
        effective_horizon = self._compute_effective_horizon_bars(event_time)
        
        if effective_horizon < self.min_horizon_bars:
            return None  # Skip event
        
        # Entry price (ask)
        entry_price = bars.loc[event_time, 'ask_close']
        
        # TP/SL barriers
        tp_price = entry_price + self.tp_distance
        sl_price = entry_price - self.sl_distance
        
        # Scan forward
        for i in range(event_idx + 1, min(event_idx + 1 + effective_horizon, len(bars))):
            current_time = bars.index[i]
            bid_high = bars.iloc[i]['bid_high']
            bid_low = bars.iloc[i]['bid_low']
            bid_close = bars.iloc[i]['bid_close']
            
            # Check both barriers
            tp_hit = bid_high >= tp_price
            sl_hit = bid_low <= sl_price
            
            # If both hit in same bar, TP takes priority (conservative assumption)
            # In reality, we'd need intra-bar data to know which was hit first
            if tp_hit:
                return {
                    'label': 1,
                    'barrier_hit': 'tp',
                    'pnl': tp_price - entry_price,
                    'exit_time': current_time
                }
            
            if sl_hit:
                return {
                    'label': -1,
                    'barrier_hit': 'sl',
                    'pnl': sl_price - entry_price,
                    'exit_time': current_time
                }
        
        # Time barrier hit
        exit_idx = min(event_idx + effective_horizon, len(bars) - 1)
        exit_time = bars.index[exit_idx]
        exit_price = bars.iloc[exit_idx]['bid_close']
        
        return {
            'label': 0,
            'barrier_hit': 'time',
            'pnl': exit_price - entry_price,
            'exit_time': exit_time
        }

