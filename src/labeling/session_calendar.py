"""Session calendar for no-overnight constraint.

Manages:
- Session start/end times
- Friday early close
- Weekend handling
- Bars/time until session end

Critical for session-aware triple barrier labeling.
"""

import logging
from datetime import time, datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SessionCalendar:
    """Manage trading session times and no-overnight constraint."""
    
    def __init__(self, config: dict):
        """Initialize session calendar.
        
        Args:
            config: Session configuration from Hydra
        """
        self.config = config
        if 'timezone' not in config:
            raise ValueError("Missing required config: timezone")
        self.timezone = config['timezone']
        
        # Parse session times
        self.session_start = self._parse_time(config['session_start'])
        self.session_end = self._parse_time(config['session_end'])
        self.friday_end = self._parse_time(config['friday_end'])
        self.weekend_trading = config.get('weekend_trading', False)
        
        logger.info(
            f"SessionCalendar initialized: "
            f"start={self.session_start}, end={self.session_end}, "
            f"friday_end={self.friday_end}"
        )
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string 'HH:MM' to time object.
        
        Args:
            time_str: Time string like '21:55'
            
        Returns:
            time object
        """
        hour, minute = map(int, time_str.split(':'))
        return time(hour, minute)
    
    def is_weekend(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is on weekend.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if Saturday (5) or Sunday (6)
        """
        return timestamp.weekday() >= 5
    
    def get_session_end_for_day(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        """Get session end time for given timestamp's day.
        
        Args:
            timestamp: Timestamp
            
        Returns:
            Session end timestamp (Friday uses friday_end)
        """
        date = timestamp.date()
        weekday = timestamp.weekday()
        
        if weekday == 4:  # Friday
            session_end_time = self.friday_end
        else:
            session_end_time = self.session_end
        
        # Combine date with session end time
        session_end = pd.Timestamp.combine(date, session_end_time)
        
        # Ensure UTC
        if session_end.tz is None:
            session_end = session_end.tz_localize(self.timezone)
        
        return session_end
    
    def time_to_session_end(
        self,
        timestamp: pd.Timestamp,
        unit: str = 'seconds'
    ) -> float:
        """Calculate time remaining until session end.
        
        Args:
            timestamp: Current timestamp
            unit: 'seconds' | 'minutes' | 'hours'
            
        Returns:
            Time to session end in specified unit
        """
        session_end = self.get_session_end_for_day(timestamp)
        
        if timestamp > session_end:
            # Already past session end
            return 0.0
        
        delta = session_end - timestamp
        
        if unit == 'seconds':
            return delta.total_seconds()
        elif unit == 'minutes':
            return delta.total_seconds() / 60
        elif unit == 'hours':
            return delta.total_seconds() / 3600
        else:
            raise ValueError(f"Unknown unit: {unit}")
    
    def bars_until_session_end(
        self,
        timestamp: pd.Timestamp,
        bar_duration_sec: float
    ) -> int:
        """Calculate number of bars until session end.
        
        Args:
            timestamp: Current timestamp
            bar_duration_sec: Average duration of one bar in seconds
            
        Returns:
            Number of bars until session end (rounded down)
        """
        time_remaining_sec = self.time_to_session_end(timestamp, unit='seconds')
        
        if time_remaining_sec <= 0:
            return 0
        
        if bar_duration_sec <= 0:
            raise ValueError("Bar duration must be positive")
        
        return int(time_remaining_sec / bar_duration_sec)
    
    def is_near_session_end(
        self,
        timestamp: pd.Timestamp,
        min_horizon_bars: int,
        bar_duration_sec: float
    ) -> bool:
        """Check if timestamp is too close to session end.
        
        Args:
            timestamp: Current timestamp
            min_horizon_bars: Minimum horizon required
            bar_duration_sec: Average bar duration in seconds
            
        Returns:
            True if < min_horizon_bars remaining
        """
        bars_remaining = self.bars_until_session_end(timestamp, bar_duration_sec)
        return bars_remaining < min_horizon_bars
    
    def is_trading_allowed(self, timestamp: pd.Timestamp) -> bool:
        """Check if trading is allowed at given timestamp.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if within session and not weekend
        """
        # Weekend check
        if not self.weekend_trading and self.is_weekend(timestamp):
            return False
        
        # Session time check
        current_time = timestamp.time()
        session_end_time = (
            self.friday_end if timestamp.weekday() == 4 else self.session_end
        )
        
        if current_time < self.session_start or current_time > session_end_time:
            return False
        
        return True
    
    def filter_session_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out ticks outside trading sessions.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering ticks by session. Input: {len(df)} ticks")
        
        initial_len = len(df)
        
        # Remove weekend ticks if not allowed
        if not self.weekend_trading:
            df = df[df['timestamp'].apply(lambda x: not self.is_weekend(x))]
            weekend_dropped = initial_len - len(df)
            logger.debug(f"Dropped {weekend_dropped} weekend ticks")
        
        # Remove ticks outside session hours
        df = df[df['timestamp'].apply(self.is_trading_allowed)]
        
        final_len = len(df)
        logger.info(
            f"Session filter complete. Output: {final_len} ticks "
            f"({initial_len - final_len} dropped)"
        )
        
        return df
    
    def get_session_end(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        """Alias for get_session_end_for_day() for backward compatibility.
        
        Args:
            timestamp: Timestamp
            
        Returns:
            Session end timestamp
        """
        return self.get_session_end_for_day(timestamp)
    
    def time_until_session_end(
        self,
        timestamp: pd.Timestamp,
        unit: str = 'minutes'
    ) -> float:
        """Alias for time_to_session_end() for backward compatibility.
        
        Args:
            timestamp: Current timestamp
            unit: 'minutes' or 'seconds'
            
        Returns:
            Time until session end in specified unit
        """
        return self.time_to_session_end(timestamp, unit=unit)
    
    def is_near_session_end(
        self,
        timestamp: pd.Timestamp,
        threshold_minutes: int = 30
    ) -> bool:
        """Check if timestamp is near session end.
        
        Args:
            timestamp: Timestamp to check
            threshold_minutes: Minutes before session end
            
        Returns:
            True if within threshold of session end
        """
        time_remaining = self.time_to_session_end(timestamp, unit='minutes')
        return time_remaining <= threshold_minutes
    
    def filter_ticks_by_session(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Filter ticks to remove weekends and out-of-session times.
        
        Args:
            ticks: DataFrame with 'timestamp' column
            
        Returns:
            Filtered DataFrame
        """
        if 'timestamp' not in ticks.columns:
            raise ValueError("ticks DataFrame must have 'timestamp' column")
        
        # Remove weekends unless weekend_trading is enabled
        if not self.weekend_trading:
            if pd.api.types.is_datetime64_any_dtype(ticks['timestamp']):
                mask = ticks['timestamp'].apply(lambda x: not self.is_weekend(x))
                ticks = ticks[mask]
            else:
                # If timestamp is index
                if pd.api.types.is_datetime64_any_dtype(ticks.index):
                    mask = ticks.index.map(lambda x: not self.is_weekend(x))
                    ticks = ticks[mask]
        
        return ticks


def compute_average_bar_duration(bars: pd.DataFrame) -> float:
    """Compute average bar duration in seconds.
    
    Args:
        bars: DataFrame with 'bar_duration_sec' column
        
    Returns:
        Average duration in seconds
    """
    if 'bar_duration_sec' not in bars.columns:
        raise ValueError("bars DataFrame must have 'bar_duration_sec' column")
    
    return bars['bar_duration_sec'].mean()

