"""Unit tests for session calendar."""

import pytest
import pandas as pd
from src.labeling.session_calendar import SessionCalendar


@pytest.mark.unit
class TestSessionCalendar:
    """Test session calendar functionality."""
    
    def test_session_calendar_init(self):
        """Test SessionCalendar initialization."""
        from datetime import time
        
        config = {
            'timezone': 'UTC',
            'session_start': '00:00',
            'session_end': '21:55',
            'friday_end': '20:00',
            'weekend_trading': False,
        }
        
        calendar = SessionCalendar(config)
        
        assert calendar is not None
        assert calendar.session_start == time(0, 0)
        assert calendar.session_end == time(21, 55)
    
    def test_is_trading_allowed(self, sample_session_calendar):
        """Test is_trading_allowed method."""
        # Monday 10:00 UTC should be trading hour
        dt = pd.Timestamp('2024-01-08 10:00:00', tz='UTC')
        # Check if trading is allowed (not weekend and within session)
        assert sample_session_calendar.is_weekend(dt) == False
        
        # Monday 22:00 UTC should be outside trading hours
        dt = pd.Timestamp('2024-01-08 22:00:00', tz='UTC')
        # After session end
        session_end = sample_session_calendar.get_session_end_for_day(dt)
        assert dt > session_end
    
    def test_is_weekend(self, sample_session_calendar):
        """Test is_weekend method."""
        # Saturday
        dt = pd.Timestamp('2024-01-06 10:00:00', tz='UTC')
        assert sample_session_calendar.is_weekend(dt) == True
        
        # Monday
        dt = pd.Timestamp('2024-01-08 10:00:00', tz='UTC')
        assert sample_session_calendar.is_weekend(dt) == False
    
    def test_friday_early_close(self, sample_session_calendar):
        """Test Friday early close."""
        # Friday 20:00 UTC should be at or after early close
        dt = pd.Timestamp('2024-01-05 20:00:00', tz='UTC')
        session_end = sample_session_calendar.get_session_end_for_day(dt)
        assert dt >= session_end
        
        # Friday 19:00 UTC should still be before early close
        dt = pd.Timestamp('2024-01-05 19:00:00', tz='UTC')
        session_end = sample_session_calendar.get_session_end_for_day(dt)
        assert dt < session_end

