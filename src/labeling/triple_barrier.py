"""Triple-barrier labeling with session-aware logic (long-only, bid/ask FX).

This module implements Lopez de Prado's triple-barrier method for
long-only FX trading on a raw-spread account:

- entries at the **ask**,
- exits at the **bid**,
- TP/SL barriers evaluated on the **bid**,
- holding horizon capped by the session end (no overnight positions).

The logic is described in `ARCH_DATA_PIPELINE.md §6` and is aligned
with the global “no overnight positions” constraint.
"""

import logging
from typing import Optional

import pandas as pd

from .session_calendar import SessionCalendar
from ..utils.helpers import convert_ticks_to_price

logger = logging.getLogger(__name__)


def compute_triple_barrier(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    tp_distance: float,
    sl_distance: float,
    max_horizon_bars: int,
    session_calendar: SessionCalendar,
    min_horizon_bars: int = 10,
    avg_bar_duration_sec: Optional[float] = None,
) -> pd.DataFrame:
    """Compute triple-barrier labels for long-only FX events.

    Args:
        events: DataFrame with columns:
            - timestamp: event start time
            - bar_index: integer index into prices DataFrame
        prices: Bar DataFrame with at least the following bid/ask columns:
            - bid_close, ask_close, bid_high, bid_low
            The index must be a DatetimeIndex (bar timestamps).
        tp_distance: Take-profit distance in price units (defined on the bid).
        sl_distance: Stop-loss distance in price units (defined on the bid).
        max_horizon_bars: Maximum holding period in number of bars.
        session_calendar: SessionCalendar instance used to enforce session-aware
            horizons (no overnight positions).
        min_horizon_bars: Minimum required horizon in bars. If the effective
            horizon is shorter, the event is ignored (no-trade zone, no label created).
        avg_bar_duration_sec: Average bar duration in seconds. If None, it is
            estimated from the prices index.

    Returns:
        DataFrame with one row per labeled event, containing:
            - event_start: event start timestamp
            - event_end: exit timestamp
            - label: 1 = TP hit, -1 = SL hit, 0 = time barrier
            - bar_index_start: start bar index
            - bar_index_end: exit bar index
            - entry_price: entry price (ask_close at t0)
            - exit_price: exit price (bid_close at t1)
            - pnl: profit/loss = exit_bid - entry_ask
            - barrier_hit: 'tp', 'sl' or 'time'
            - effective_horizon: effective horizon in bars
    """
    logger.info(
        f"Computing triple barriers for {len(events)} events. "
        f"TP={tp_distance}, SL={sl_distance}, max_horizon={max_horizon_bars}"
    )
    
    if avg_bar_duration_sec is None:
        # Estimate average bar duration from the index
        if len(prices) > 1:
            time_diff = prices.index[-1] - prices.index[0]
            avg_bar_duration_sec = time_diff.total_seconds() / len(prices)
        else:
            avg_bar_duration_sec = 60.0  # fallback neutre
    
    results = []
    skipped_count = 0
    
    for idx, event in events.iterrows():
        t0 = event['timestamp']
        bar_idx_start = event['bar_index']
        
        # Effective horizon based exclusively on the SessionCalendar
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
        
        # Long entry at ask
        entry_price_ask = prices.iloc[bar_idx_start]['ask_close']
        entry_price_bid = prices.iloc[bar_idx_start]['bid_close']

        # Barriers defined from the entry bid
        tp_level = entry_price_bid + tp_distance
        sl_level = entry_price_bid - sl_distance
        
        # Scan forward up to effective_horizon_bars
        end_bar_idx = min(bar_idx_start + effective_horizon_bars, len(prices) - 1)
        
        # Future prices slice
        future_prices = prices.iloc[bar_idx_start + 1 : end_bar_idx + 1]
        
        if len(future_prices) == 0:
            # No future data available
            skipped_count += 1
            continue
        
        # Scan forward to find the first barrier hit
        label = 0
        barrier_hit = 'time'
        exit_bar_idx = end_bar_idx

        for i, (_, row) in enumerate(future_prices.iterrows()):
            bar_position = bar_idx_start + 1 + i
            bid_high = row['bid_high']
            bid_low = row['bid_low']

            # TP has priority if TP and SL hit in the same bar
            if bid_high >= tp_level:
                label = 1
                barrier_hit = 'tp'
                exit_bar_idx = bar_position
                break

            if bid_low <= sl_level:
                label = -1
                barrier_hit = 'sl'
                exit_bar_idx = bar_position
                break
        
        # Exit always at bid
        exit_price = prices.iloc[exit_bar_idx]['bid_close']
        
        # PnL computed between entry_ask and exit_bid
        entry_price = entry_price_ask
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


class TripleBarrierLabeler:
    """Object-oriented wrapper around ``compute_triple_barrier``.

    This class does **not** contain any additional business logic:

    - reads the configuration (tp/sl in ticks or absolute),
    - pre-computes TP/SL distances in price units,
    - maps event indices to bar indices,
    - calls ``compute_triple_barrier``.
    """
    
    def __init__(self, config: dict, session_calendar: SessionCalendar):
        """Initialize the labeler from a Hydra-style configuration.

        Args:
            config: Configuration dictionary with keys:
                - tp_ticks: Take profit distance in ticks (for 'ticks' mode)
                - sl_ticks: Stop loss distance in ticks (for 'ticks' mode)
                - tp_quantile: Take profit quantile (for 'mfe_mae' mode)
                - sl_quantile: Stop loss quantile (for 'mfe_mae' mode)
                - max_horizon_bars: Maximum horizon in bars
                - min_horizon_bars: Minimum horizon (no-trade zone threshold)
                - distance_mode: 'ticks' or 'mfe_mae'
                - tick_size: Tick size for 'ticks' mode (default 0.0001)
            session_calendar: SessionCalendar instance for session-aware logic
            
        Note:
            For 'ticks' mode, uses convert_ticks_to_price() to convert ticks to price.
            For 'mfe_mae' mode, TP/SL are computed from MFE/MAE quantiles in the pipeline.
        """
        self.config = config
        self.calendar = session_calendar
        
        # Require distance_mode
        if 'distance_mode' not in config:
            raise ValueError("Missing required config: distance_mode")
        self.distance_mode = config['distance_mode']
        
        # Require max_horizon_bars and min_horizon_bars
        if 'max_horizon_bars' not in config:
            raise ValueError("Missing required config: max_horizon_bars")
        if 'min_horizon_bars' not in config:
            raise ValueError("Missing required config: min_horizon_bars")
        self.max_horizon_bars = config['max_horizon_bars']
        self.min_horizon_bars = config['min_horizon_bars']
        
        # Pre-computed price distances
        # Only use convert_ticks_to_price for 'ticks' mode
        if self.distance_mode == 'ticks':
            # Require tp_ticks, sl_ticks, and tick_size for ticks mode
            if 'tp_ticks' not in config:
                raise ValueError("Missing required config: tp_ticks (required when distance_mode='ticks')")
            if 'sl_ticks' not in config:
                raise ValueError("Missing required config: sl_ticks (required when distance_mode='ticks')")
            if 'tick_size' not in config:
                raise ValueError("Missing required config: tick_size (required when distance_mode='ticks')")
            
            self.tp_ticks = config['tp_ticks']
            self.sl_ticks = config['sl_ticks']
            self.tick_size = config['tick_size']
            self.tp_distance = convert_ticks_to_price(self.tp_ticks, self.tick_size)
            self.sl_distance = convert_ticks_to_price(self.sl_ticks, self.tick_size)
        elif self.distance_mode == 'mfe_mae':
            # MFE/MAE mode: TP/SL will be set dynamically from MFE/MAE quantiles
            # Require mfe_mae config block
            if 'mfe_mae' not in config:
                raise ValueError(
                    "Missing required config: mfe_mae (required when distance_mode='mfe_mae')"
                )
            # TP/SL will be set by pipeline, use temporary values
            # tick_size will be provided by assets config
            self.tp_ticks = config.get('tp_ticks', 100)  # Temporary, will be overridden
            self.sl_ticks = config.get('sl_ticks', 100)  # Temporary, will be overridden
            self.tick_size = config.get('tick_size', 0.00001)  # Temporary, will be overridden
            self.tp_distance = convert_ticks_to_price(self.tp_ticks, self.tick_size)
            self.sl_distance = convert_ticks_to_price(self.sl_ticks, self.tick_size)
        else:
            raise ValueError(
                f"Invalid distance_mode: {self.distance_mode}. "
                f"Must be one of: 'ticks', 'mfe_mae'"
            )
    
    def label_dataset(
        self,
        bars: pd.DataFrame,
        event_indices: pd.Index,
        avg_bar_duration_sec: Optional[float] = None
    ) -> pd.DataFrame:
        """Apply compute_triple_barrier to a set of bars.

        Args:
            bars: DataFrame with DatetimeIndex and required bid/ask columns
                (bid_close, ask_close, bid_high, bid_low). See compute_triple_barrier
                for details.
            event_indices: Indices to label (typically the features DataFrame index)
            avg_bar_duration_sec: Average bar duration in seconds. If None, estimated
                from bars index.

        Returns:
            DataFrame with labels (same format as compute_triple_barrier output),
            with an additional 'bar_timestamp' column for alignment.
        """
        # Normalisation: ensure we work with a DatetimeIndex
        if not isinstance(event_indices, pd.DatetimeIndex):
            if hasattr(bars, 'index') and isinstance(bars.index, pd.DatetimeIndex):
                # Remap if necessary
                event_indices = bars.index[event_indices] if len(event_indices) <= len(bars) else bars.index
            else:
                raise ValueError("Cannot convert event_indices to DatetimeIndex. Bars must have DatetimeIndex.")
        
        # Build events DataFrame (timestamp + bar index)
        bar_positions = [bars.index.get_loc(ts) for ts in event_indices if ts in bars.index]
        valid_timestamps = [ts for ts in event_indices if ts in bars.index]
        
        events = pd.DataFrame({
            'timestamp': valid_timestamps,
            'bar_index': bar_positions
        })
        
        # Single call to the core function
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
        
        # Add a reference bar timestamp for alignment
        if len(labels) > 0 and 'bar_index_start' in labels.columns:
            labels['bar_timestamp'] = labels['bar_index_start'].apply(
                lambda idx: bars.index[idx] if idx < len(bars) else None
            )
            labels = labels.dropna(subset=['bar_timestamp'])
        
        return labels

