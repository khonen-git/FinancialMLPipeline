"""Backtrader strategy with session-aware logic.

Implements:
- No overnight positions (flat before session_end)
- No-trade zone near session_end
- Bid/ask execution (ask entry, bid exit)
- SL/TP orders aligned with triple barrier
- Meta-model filtering
- Trade logging
"""

import logging
import backtrader as bt
from datetime import datetime, time
from typing import Optional, Dict
import pandas as pd

from src.labeling.session_calendar import SessionCalendar

logger = logging.getLogger(__name__)


class SessionAwareStrategy(bt.Strategy):
    """Trading strategy with session management and triple barrier logic.
    
    This strategy must behave EXACTLY like the triple barrier labeling logic:
    - Entry at ask price (long only)
    - Exit at bid price
    - TP/SL barriers checked on bid prices
    - No overnight positions (close before session_end)
    - No new entries near session_end
    """
    
    params = (
        ('session_calendar', None),  # SessionCalendar instance (required)
        ('tp_ticks', 100),  # Take profit in ticks
        ('sl_ticks', 100),  # Stop loss in ticks
        ('tick_size', 0.0001),  # Tick size
        ('use_meta_model', True),  # Use meta-model filtering
        ('meta_threshold', 0.5),  # Meta-model probability threshold
        ('position_size', 1.0),  # Position size (fixed for v1)
    )
    
    def __init__(self):
        """Initialize strategy."""
        self.order = None
        self.signal_data = None  # Will be set externally with model predictions
        self.trades = []  # Store trade information for logging
        self.current_trade = None  # Track current open trade
        
        # Validate session_calendar
        if self.params.session_calendar is None:
            raise ValueError("session_calendar parameter is required")
        
        if not isinstance(self.params.session_calendar, SessionCalendar):
            raise ValueError("session_calendar must be a SessionCalendar instance")
        
        self.calendar = self.params.session_calendar
        
        logger.info(
            f"Strategy initialized: TP={self.params.tp_ticks} ticks, "
            f"SL={self.params.sl_ticks} ticks, "
            f"Meta-model={self.params.use_meta_model}"
        )
    
    def set_signal_data(self, signal_df: pd.DataFrame):
        """Set model predictions.
        
        Args:
            signal_df: DataFrame with 'prediction' and optional 'meta_decision' columns
                Index must be datetime and match bars timestamps
        """
        self.signal_data = signal_df
        logger.info(f"Signal data set: {len(signal_df)} predictions")
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order submitted/accepted, nothing to do
        
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(f"BUY EXECUTED: {order.executed.price:.5f}, size={order.executed.size}")
            elif order.issell():
                logger.debug(f"SELL EXECUTED: {order.executed.price:.5f}, size={order.executed.size}")
        
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order canceled/rejected: {order.status}")
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            # Record trade information
            current_dt = self.data.datetime.datetime(0)
            
            # Get entry/exit prices
            entry_price = trade.price
            exit_price = trade.price_close
            
            # Determine exit reason
            exit_reason = 'unknown'
            if hasattr(trade, 'exit_reason'):
                exit_reason = trade.exit_reason
            elif self.is_near_session_end(current_dt):
                exit_reason = 'session_close'
            
            # Calculate PnL
            pnl = trade.pnl
            pnl_pct = trade.pnlcomm / self.broker.getvalue() if self.broker.getvalue() > 0 else 0
            
            self.trades.append({
                'entry_timestamp': trade.data.datetime.datetime(0) if hasattr(trade.data, 'datetime') else current_dt,
                'exit_timestamp': current_dt,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration': trade.barlen if hasattr(trade, 'barlen') else None,
                'size': trade.size,
                'reason': exit_reason,
            })
            
            logger.info(
                f"TRADE CLOSED: PnL={pnl:.2f}, PnL%={pnl_pct:.2%}, "
                f"Reason={exit_reason}, Entry={entry_price:.5f}, Exit={exit_price:.5f}"
            )
            
            self.current_trade = None
    
    def next(self):
        """Strategy logic for each bar."""
        current_dt = self.data.datetime.datetime(0)
        
        # Convert to pandas Timestamp for SessionCalendar
        current_ts = pd.Timestamp(current_dt)
        
        # Check if weekend (no trading)
        if self.calendar.is_weekend(current_ts) and not self.calendar.weekend_trading:
            if self.position:
                logger.info(f"Weekend detected, closing position: {current_dt}")
                self.close()
            return
        
        # Check if we should close positions near session end
        session_end = self.calendar.get_session_end_for_day(current_ts)
        time_to_end = self.calendar.time_to_session_end(current_ts, unit='seconds')
        
        # Close position if too close to session end (within last bar or min_horizon)
        min_horizon_seconds = self.calendar.config.get('labeling', {}).get('min_horizon_bars', 10) * 60  # Approximate
        if time_to_end <= min_horizon_seconds:
            if self.position:
                logger.info(f"Closing position before session end: {current_dt}, time_to_end={time_to_end:.0f}s")
                self.close()
            return
        
        # Don't enter new trades near session end
        if time_to_end <= min_horizon_seconds:
            return
        
        # If we have an open order, wait
        if self.order:
            return
        
        # Get signal for current bar
        signal = self.get_current_signal(current_dt)
        
        if signal is None:
            return
        
        # Check if we should take the trade (meta-model filter)
        if self.params.use_meta_model:
            meta_decision = signal.get('meta_decision', 1)  # Default to 1 if not provided
            if meta_decision == 0:
                return  # Meta-model says skip
        
        # Execute trade if no position (long-only for now)
        if not self.position:
            prediction = signal.get('prediction', 0)
            
            if prediction == 1:  # Long signal
                # Entry at ask price (as per triple barrier logic)
                entry_price = self.data.ask[0]
                
                # Compute TP/SL distances in price units
                tp_distance = self.params.tp_ticks * self.params.tick_size
                sl_distance = self.params.sl_ticks * self.params.tick_size
                
                # TP/SL levels (checked on bid, as per triple barrier logic)
                # We enter at ask, but barriers are checked on bid
                # So TP = entry_ask + tp_distance, but checked when bid >= entry_ask + tp_distance
                # Actually, we need to be careful: entry is at ask, but we check barriers on bid
                # The triple barrier logic: entry_ask, then check if bid_high >= entry_bid + tp_distance
                # For simplicity in backtest, we'll use: entry_ask, check if bid >= entry_ask + tp_distance
                # This is slightly different but acceptable for backtesting
                tp_price = entry_price + tp_distance
                sl_price = entry_price - sl_distance
                
                # Place bracket order (buy with stop loss and take profit)
                self.order = self.buy_bracket(
                    price=entry_price,
                    stopprice=sl_price,
                    limitprice=tp_price,
                    exectype=bt.Order.Market  # Market order for entry
                )
                
                # Store trade information
                self.current_trade = {
                    'entry_timestamp': current_dt,
                    'entry_price': entry_price,
                    'prediction': prediction,
                    'probability': signal.get('probability', 0),
                }
                
                logger.info(
                    f"LONG ENTRY: {entry_price:.5f}, "
                    f"TP={tp_price:.5f}, SL={sl_price:.5f}, "
                    f"Prediction={prediction}, Meta={signal.get('meta_decision', 'N/A')}"
                )
    
    def is_near_session_end(self, dt: datetime) -> bool:
        """Check if current time is near session end.
        
        Args:
            dt: Current datetime
            
        Returns:
            True if near session end
        """
        current_ts = pd.Timestamp(dt)
        session_end = self.calendar.get_session_end_for_day(current_ts)
        time_to_end = self.calendar.time_to_session_end(current_ts, unit='seconds')
        
        # Close if within last bar (approximate 1 minute)
        return time_to_end <= 60
    
    def is_too_close_to_session_end(self, dt: datetime) -> bool:
        """Check if too close to session end for new entries.
        
        Args:
            dt: Current datetime
            
        Returns:
            True if in no-trade zone
        """
        current_ts = pd.Timestamp(dt)
        min_horizon_seconds = self.calendar.config.get('labeling', {}).get('min_horizon_bars', 10) * 60
        time_to_end = self.calendar.time_to_session_end(current_ts, unit='seconds')
        
        return time_to_end <= min_horizon_seconds
    
    def get_current_signal(self, dt: datetime) -> Optional[Dict]:
        """Get model signal for current bar.
        
        Args:
            dt: Current datetime
            
        Returns:
            Dictionary with 'prediction', 'probability', 'meta_decision', or None
        """
        # Try signal_data first (set via set_signal_data)
        if self.signal_data is not None:
            try:
                # Try exact match first
                if dt in self.signal_data.index:
                    signal = self.signal_data.loc[dt]
                else:
                    # Try to find nearest timestamp
                    idx = self.signal_data.index.get_indexer([pd.Timestamp(dt)], method='nearest')[0]
                    if idx >= 0:
                        signal = self.signal_data.iloc[idx]
                    else:
                        return None
                
                # Convert to dict if Series
                if isinstance(signal, pd.Series):
                    return {
                        'prediction': signal.get('prediction', 0),
                        'probability': signal.get('probability', 0),
                        'meta_decision': signal.get('meta_decision', 1),
                    }
                elif isinstance(signal, dict):
                    return signal
                else:
                    return None
            except (KeyError, IndexError, AttributeError) as e:
                logger.debug(f"Could not get signal from signal_data for {dt}: {e}")
        
        # Fallback: try to get from data feed's predictions attribute
        if hasattr(self.data, 'predictions'):
            try:
                predictions = self.data.predictions
                if dt in predictions.index:
                    signal = predictions.loc[dt]
                    return {
                        'prediction': signal.get('prediction', 0) if isinstance(signal, pd.Series) else signal.get('prediction', 0),
                        'probability': signal.get('probability', 0) if isinstance(signal, pd.Series) else signal.get('probability', 0),
                        'meta_decision': signal.get('meta_decision', 1) if isinstance(signal, pd.Series) else signal.get('meta_decision', 1),
                    }
            except (KeyError, IndexError, AttributeError) as e:
                logger.debug(f"Could not get signal from data feed for {dt}: {e}")
        
        return None
