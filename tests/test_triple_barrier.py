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
        dt = datetime(2024, 1, 9, 10, 0)  # Tuesday 10:00, far from session end
        bars_remaining = self.labeler._compute_effective_horizon_bars(dt, bar_duration_minutes=5)
        
        # Should be max_horizon_bars since we're far from session end
        self.assertEqual(bars_remaining, 50)
    
    def test_compute_effective_horizon_near_session_end(self):
        """Test effective horizon when near session end."""
        dt = datetime(2024, 1, 9, 21, 30)  # Tuesday 21:30, only 25min until 21:55
        bars_remaining = self.labeler._compute_effective_horizon_bars(dt, bar_duration_minutes=5)
        
        # Should be limited by session end (25min / 5min = 5 bars)
        self.assertEqual(bars_remaining, 5)
    
    def test_compute_effective_horizon_past_session(self):
        """Test effective horizon past session end."""
        dt = datetime(2024, 1, 9, 22, 30)  # Tuesday 22:30, past session end
        bars_remaining = self.labeler._compute_effective_horizon_bars(dt, bar_duration_minutes=5)
        
        # Should be 0 (past session)
        self.assertEqual(bars_remaining, 0)
    
    def test_label_single_event_tp_hit(self):
        """Test labeling when TP is hit."""
        # Create simple bar data
        dates = pd.date_range('2024-01-09 10:00', periods=20, freq='5min')
        
        # Entry at 1.0700, TP at 1.0710 (10 ticks up)
        bid_high = [1.0700 + i * 0.0002 for i in range(20)]  # Going up
        
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
        dates = pd.date_range('2024-01-09 10:00', periods=20, freq='5min')
        
        # Entry at 1.0705 (ask), SL at 1.0695 (10 ticks down from entry)
        bid_low = [1.0700 - i * 0.0002 for i in range(20)]  # Going down
        
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
        dates = pd.date_range('2024-01-09 10:00', periods=60, freq='5min')
        
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
        dates = pd.date_range('2024-01-09 21:50', periods=10, freq='1min')
        
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
        
        dates = pd.date_range('2024-01-09 10:00', periods=10, freq='5min')
        
        # Entry: ask_close = 1.0705
        # TP: bid_high >= 1.0715 (1.0705 + 10*0.0001)
        # Exit at bid_high = 1.0720
        
        bars = pd.DataFrame({
            'timestamp': dates,
            'bid_open': 1.0700,
            'bid_high': [1.0700, 1.0705, 1.0710, 1.0720, 1.0720, 1.0720, 1.0720, 1.0720, 1.0720, 1.0720],
            'bid_low': 1.0695,
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
        
        # Entry: 1.0705, Exit: 1.0720
        expected_pnl = 1.0720 - 1.0705
        self.assertAlmostEqual(label['pnl'], expected_pnl, places=4)


if __name__ == '__main__':
    unittest.main()

