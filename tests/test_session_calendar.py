"""Unit tests for SessionCalendar."""

import unittest
from datetime import datetime, time
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling.session_calendar import SessionCalendar


class TestSessionCalendar(unittest.TestCase):
    """Test SessionCalendar functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'session_start': "00:00",
            'session_end': "21:55",
            'friday_end': "20:00",
            'weekend_trading': False
        }
        self.calendar = SessionCalendar(self.config)
    
    def test_initialization(self):
        """Test calendar initialization."""
        self.assertEqual(self.calendar.session_start, time(0, 0))
        self.assertEqual(self.calendar.session_end, time(21, 55))
        self.assertEqual(self.calendar.friday_end, time(20, 0))
        self.assertFalse(self.calendar.weekend_trading)
    
    def test_is_weekend_saturday(self):
        """Test weekend detection for Saturday."""
        saturday = datetime(2024, 1, 6, 12, 0)  # Saturday
        self.assertTrue(self.calendar.is_weekend(saturday))
    
    def test_is_weekend_sunday(self):
        """Test weekend detection for Sunday."""
        sunday = datetime(2024, 1, 7, 12, 0)  # Sunday
        self.assertTrue(self.calendar.is_weekend(sunday))
    
    def test_is_weekend_weekday(self):
        """Test weekend detection for weekday."""
        monday = datetime(2024, 1, 8, 12, 0)  # Monday
        self.assertFalse(self.calendar.is_weekend(monday))
    
    def test_get_session_end_regular_day(self):
        """Test session end for regular weekday."""
        tuesday = datetime(2024, 1, 9, 12, 0)  # Tuesday
        session_end = self.calendar.get_session_end(tuesday)
        self.assertEqual(session_end.time(), time(21, 55))
    
    def test_get_session_end_friday(self):
        """Test session end for Friday."""
        friday = datetime(2024, 1, 12, 12, 0)  # Friday
        session_end = self.calendar.get_session_end(friday)
        self.assertEqual(session_end.time(), time(20, 0))
    
    def test_is_near_session_end_far(self):
        """Test not near session end."""
        dt = datetime(2024, 1, 9, 10, 0)  # Tuesday 10:00
        self.assertFalse(self.calendar.is_near_session_end(dt, threshold_minutes=60))
    
    def test_is_near_session_end_close(self):
        """Test near session end."""
        dt = datetime(2024, 1, 9, 21, 30)  # Tuesday 21:30 (25 min before 21:55)
        self.assertTrue(self.calendar.is_near_session_end(dt, threshold_minutes=30))
    
    def test_is_near_session_end_past(self):
        """Test past session end."""
        dt = datetime(2024, 1, 9, 22, 30)  # Tuesday 22:30 (past 21:55)
        self.assertTrue(self.calendar.is_near_session_end(dt, threshold_minutes=60))
    
    def test_filter_ticks_weekend(self):
        """Test filtering weekend ticks."""
        # Create ticks with weekend dates
        dates = pd.date_range('2024-01-05', '2024-01-08', freq='h')  # Fri-Mon
        ticks = pd.DataFrame({'timestamp': dates, 'bidPrice': 1.0})
        
        filtered = self.calendar.filter_ticks_by_session(ticks)
        
        # Should exclude Saturday and Sunday
        for dt in filtered['timestamp']:
            self.assertNotEqual(dt.weekday(), 5)  # Saturday
            self.assertNotEqual(dt.weekday(), 6)  # Sunday
    
    def test_time_until_session_end_regular(self):
        """Test time calculation until session end."""
        dt = datetime(2024, 1, 9, 20, 0)  # Tuesday 20:00
        minutes = self.calendar.time_until_session_end(dt, unit='minutes')
        self.assertEqual(minutes, 115)  # 1h55min until 21:55
    
    def test_time_until_session_end_friday(self):
        """Test time calculation until Friday session end."""
        dt = datetime(2024, 1, 12, 18, 0)  # Friday 18:00
        minutes = self.calendar.time_until_session_end(dt, unit='minutes')
        self.assertEqual(minutes, 120)  # 2h until 20:00


class TestSessionCalendarEdgeCases(unittest.TestCase):
    """Test edge cases for SessionCalendar."""
    
    def test_midnight_session_end(self):
        """Test session ending at midnight."""
        config = {
            'session_start': "00:00",
            'session_end': "23:59",
            'friday_end': "23:59",
            'weekend_trading': False
        }
        calendar = SessionCalendar(config)
        
        dt = datetime(2024, 1, 9, 23, 30)
        self.assertTrue(calendar.is_near_session_end(dt, threshold_minutes=30))
    
    def test_weekend_trading_enabled(self):
        """Test with weekend trading enabled."""
        config = {
            'session_start': "00:00",
            'session_end': "21:55",
            'friday_end': "21:55",
            'weekend_trading': True
        }
        calendar = SessionCalendar(config)
        
        saturday = datetime(2024, 1, 6, 12, 0)
        self.assertFalse(calendar.is_weekend(saturday))  # Should not filter


if __name__ == '__main__':
    unittest.main()

