"""Unit tests for Triple Barrier labeling."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling.triple_barrier import TripleBarrierLabeler
from src.labeling.session_calendar import SessionCalendar


class TestTripleBarrierLabeler(unittest.TestCase):
    """Test TripleBarrierLabeler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'tp_ticks': 10,
            'sl_ticks': 10,
            'max_horizon_bars': 50,
            'min_horizon_bars': 5,
            'distance_mode': 'ticks',
            'tick_size': 0.0001
        }
        
        session_config = {
            'session_start': "00:00",
            'session_end': "21:55",
            'friday_end': "20:00",
            'weekend_trading': False
        }
        self.calendar = SessionCalendar(session_config)
        
        self.labeler = TripleBarrierLabeler(self.config, self.calendar)
    
    def test_initialization(self):
        """Test labeler initialization."""
        self.assertEqual(self.labeler.tp_ticks, 10)
        self.assertEqual(self.labeler.sl_ticks, 10)
        self.assertEqual(self.labeler.max_horizon_bars, 50)
        self.assertIsNotNone(self.labeler.calendar)
    
    def test_compute_effective_horizon_regular(self):
        """Test effective horizon computation for regular case."""
        dt = pd.Timestamp('2024-01-09 10:00:00', tz='UTC')
        bars_remaining = self.labeler._compute_effective_horizon_bars(dt, bar_duration_minutes=5)
        
        # Should be max_horizon_bars since we're far from session end
        self.assertEqual(bars_remaining, 50)
    
    def test_compute_effective_horizon_near_session_end(self):
        """Test effective horizon when near session end."""
        dt = pd.Timestamp('2024-01-09 21:30:00', tz='UTC')
        bars_remaining = self.labeler._compute_effective_horizon_bars(dt, bar_duration_minutes=5)
        
        # Should be limited by session end (25min / 5min = 5 bars)
        self.assertEqual(bars_remaining, 5)
    
    def test_compute_effective_horizon_past_session(self):
        """Test effective horizon past session end."""
        dt = pd.Timestamp('2024-01-09 22:30:00', tz='UTC')
        bars_remaining = self.labeler._compute_effective_horizon_bars(dt, bar_duration_minutes=5)
        
        # Should be 0 (past session)
        self.assertEqual(bars_remaining, 0)
    
    def test_label_single_event_tp_hit(self):
        """Test labeling when TP is hit."""
        # Create simple bar data
        dates = pd.date_range('2024-01-09 10:00', periods=20, freq='5min', tz='UTC')
        
        # Entry at ask_close = 1.0705
        # TP = 1.0705 + 50*0.0001 = 1.0755 (50 ticks configured in labeler)
        # Make bid_high go up high enough to hit TP
        bid_high = [1.0700 + i * 0.005 for i in range(20)]  # Big jumps to hit 1.0755
        
        bars = pd.DataFrame({
            'timestamp': dates,
            'bid_open': 1.0700,
            'bid_high': bid_high,
            'bid_low': 1.0695,
            'bid_close': 1.0700,
            'ask_open': 1.0705,
            'ask_high': 1.0715,
            'ask_low': 1.0700,
            'ask_close': 1.0705
        })
        bars = bars.set_index('timestamp')
        
        # Label single event at first bar
        event_time = dates[0]
        label = self.labeler._label_single_event(
            event_time=event_time,
            event_idx=0,
            bars=bars
        )
        
        # Should hit TP (label = 1)
        self.assertEqual(label['label'], 1)
        self.assertEqual(label['barrier_hit'], 'tp')
        self.assertGreater(label['pnl'], 0)
    
    def test_label_single_event_sl_hit(self):
        """Test labeling when SL is hit."""
        dates = pd.date_range('2024-01-09 10:00', periods=20, freq='5min', tz='UTC')
        
        # Entry at ask_close = 1.0705
        # SL = 1.0705 - 50*0.0001 = 1.0655 (50 ticks configured in labeler)
        # Make bid_low go down enough to hit SL
        bid_low = [1.0700 - i * 0.005 for i in range(20)]  # Big drops to hit 1.0655
        
        bars = pd.DataFrame({
            'timestamp': dates,
            'bid_open': 1.0700,
            'bid_high': 1.0705,
            'bid_low': bid_low,
            'bid_close': 1.0695,
            'ask_open': 1.0705,
            'ask_high': 1.0710,
            'ask_low': 1.0700,
            'ask_close': 1.0705
        })
        bars = bars.set_index('timestamp')
        
        event_time = dates[0]
        label = self.labeler._label_single_event(
            event_time=event_time,
            event_idx=0,
            bars=bars
        )
        
        # Should hit SL (label = -1)
        self.assertEqual(label['label'], -1)
        self.assertEqual(label['barrier_hit'], 'sl')
        self.assertLess(label['pnl'], 0)
    
    def test_label_single_event_time_barrier(self):
        """Test labeling when time barrier is hit."""
        dates = pd.date_range('2024-01-09 10:00', periods=60, freq='5min', tz='UTC')
        
        # Prices stay flat, no TP/SL hit
        bars = pd.DataFrame({
            'timestamp': dates,
            'bid_open': 1.0700,
            'bid_high': 1.0701,
            'bid_low': 1.0699,
            'bid_close': 1.0700,
            'ask_open': 1.0705,
            'ask_high': 1.0706,
            'ask_low': 1.0704,
            'ask_close': 1.0705
        })
        bars = bars.set_index('timestamp')
        
        event_time = dates[0]
        label = self.labeler._label_single_event(
            event_time=event_time,
            event_idx=0,
            bars=bars
        )
        
        # Should hit time barrier (label = 0)
        self.assertEqual(label['label'], 0)
        self.assertEqual(label['barrier_hit'], 'time')
    
    def test_skip_event_near_session_end(self):
        """Test skipping events too close to session end."""
        # Event very close to session end
        dates = pd.date_range('2024-01-09 21:50', periods=10, freq='1min', tz='UTC')
        
        bars = pd.DataFrame({
            'timestamp': dates,
            'bid_open': 1.0700,
            'bid_high': 1.0701,
            'bid_low': 1.0699,
            'bid_close': 1.0700,
            'ask_open': 1.0705,
            'ask_high': 1.0706,
            'ask_low': 1.0704,
            'ask_close': 1.0705
        })
        bars = bars.set_index('timestamp')
        
        event_time = dates[0]
        label = self.labeler._label_single_event(
            event_time=event_time,
            event_idx=0,
            bars=bars
        )
        
        # Should be None (event skipped)
        self.assertIsNone(label)


class TestTripleBarrierEdgeCases(unittest.TestCase):
    """Test edge cases for Triple Barrier."""
    
    def test_entry_exit_prices(self):
        """Test entry at ask and exit at bid."""
        config = {
            'tp_ticks': 10,
            'sl_ticks': 10,
            'max_horizon_bars': 50,
            'min_horizon_bars': 5,
            'distance_mode': 'ticks',
            'tick_size': 0.0001
        }
        
        session_config = {
            'session_start': "00:00",
            'session_end': "23:59",
            'friday_end': "23:59",
            'weekend_trading': False
        }
        calendar = SessionCalendar(session_config)
        labeler = TripleBarrierLabeler(config, calendar)
        
        dates = pd.date_range('2024-01-09 10:00', periods=10, freq='5min', tz='UTC')
        
        # Entry: ask_close = 1.0705
        # TP: bid_high >= 1.0715 (1.0705 + 10*0.0001 with labeler tp_ticks=10)
        # SL: bid_low <= 1.0695 (1.0705 - 10*0.0001)
        # Exit at bid_high = 1.0720
        
        bars = pd.DataFrame({
            'timestamp': dates,
            'bid_open': 1.0700,
            'bid_high': [1.0700, 1.0705, 1.0710, 1.0720, 1.0720, 1.0720, 1.0720, 1.0720, 1.0720, 1.0720],
            'bid_low': 1.0696,  # Changed to 1.0696 to NOT hit SL at 1.0695
            'bid_close': 1.0700,
            'ask_open': 1.0705,
            'ask_high': 1.0715,
            'ask_low': 1.0700,
            'ask_close': [1.0705, 1.0705, 1.0705, 1.0705, 1.0705, 1.0705, 1.0705, 1.0705, 1.0705, 1.0705]
        })
        bars = bars.set_index('timestamp')
        
        event_time = dates[0]
        label = labeler._label_single_event(
            event_time=event_time,
            event_idx=0,
            bars=bars
        )
        
        # Entry: 1.0705, Exit should be at TP: 1.0715
        # PnL should be positive (TP - entry = 1.0715 - 1.0705 = 0.001)
        self.assertEqual(label['label'], 1)  # Should hit TP
        self.assertAlmostEqual(label['pnl'], 0.001, places=4)


if __name__ == '__main__':
    unittest.main()

