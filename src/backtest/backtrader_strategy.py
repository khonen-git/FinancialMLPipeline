"""Backtrader strategy with session-aware logic.

Implements:
- No overnight positions (flat before session_end)
- No-trade zone near session_end
- Bid/ask execution
- SL/TP orders
- Meta-model filtering
"""

import logging
import backtrader as bt
from datetime import datetime, time

logger = logging.getLogger(__name__)


class SessionAwareStrategy(bt.Strategy):
    """Trading strategy with session management."""
    
    params = (
        ('session_end', time(21, 55)),  # UTC
        ('friday_end', time(20, 0)),  # UTC
        ('min_time_to_session_end_bars', 10),
        ('tp_ticks', 100),
        ('sl_ticks', 100),
        ('tick_size', 0.0001),
        ('use_meta_model', True),
        ('meta_threshold', 0.5),
    )
    
    def __init__(self):
        """Initialize strategy."""
        self.order = None
        self.signal_data = None  # Will be set externally with model predictions
    
    def set_signal_data(self, signal_df):
        """Set model predictions.
        
        Args:
            signal_df: DataFrame with 'prediction' and 'meta_proba' columns
        """
        self.signal_data = signal_df
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(f"BUY EXECUTED: {order.executed.price:.5f}")
            elif order.issell():
                logger.debug(f"SELL EXECUTED: {order.executed.price:.5f}")
        
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order canceled/rejected: {order.status}")
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            logger.info(f"TRADE CLOSED: PnL={trade.pnl:.2f}, PnL%={trade.pnlcomm:.2%}")
    
    def next(self):
        """Strategy logic for each bar."""
        current_dt = self.data.datetime.datetime()
        
        # Check if we should close positions near session end
        if self.is_near_session_end(current_dt):
            if self.position:
                logger.info(f"Closing position before session end: {current_dt}")
                self.close()
            return
        
        # Don't enter new trades near session end
        if self.is_too_close_to_session_end(current_dt):
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
            meta_proba = signal.get('meta_proba', 0)
            if meta_proba < self.params.meta_threshold:
                return
        
        # Execute trade if no position
        if not self.position:
            prediction = signal.get('prediction', 0)
            
            if prediction == 1:  # Long signal
                entry_price = self.data.ask[0]  # Enter at ask
                
                # Compute SL/TP
                tp_price = entry_price + self.params.tp_ticks * self.params.tick_size
                sl_price = entry_price - self.params.sl_ticks * self.params.tick_size
                
                # Place bracket order
                self.order = self.buy_bracket(
                    price=entry_price,
                    stopprice=sl_price,
                    limitprice=tp_price
                )
                
                logger.info(
                    f"LONG ENTRY: {entry_price:.5f}, "
                    f"TP={tp_price:.5f}, SL={sl_price:.5f}"
                )
    
    def is_near_session_end(self, dt: datetime) -> bool:
        """Check if current time is near session end.
        
        Args:
            dt: Current datetime
            
        Returns:
            True if near session end
        """
        current_time = dt.time()
        
        # Friday early close
        if dt.weekday() == 4:  # Friday
            return current_time >= self.params.friday_end
        
        # Regular session end
        return current_time >= self.params.session_end
    
    def is_too_close_to_session_end(self, dt: datetime) -> bool:
        """Check if too close to session end for new entries.
        
        Args:
            dt: Current datetime
            
        Returns:
            True if in no-trade zone
        """
        # Simplified: use session_end check
        # In reality, we'd compute bars_to_session_end
        return self.is_near_session_end(dt)
    
    def get_current_signal(self, dt: datetime) -> dict:
        """Get model signal for current bar.
        
        Args:
            dt: Current datetime
            
        Returns:
            Dictionary with 'prediction' and 'meta_proba', or None
        """
        if self.signal_data is None:
            return None
        
        # Match current bar datetime with signal_data index
        try:
            signal = self.signal_data.loc[dt]
            return {
                'prediction': signal.get('prediction', 0),
                'meta_proba': signal.get('meta_proba', 0)
            }
        except KeyError:
            return None

