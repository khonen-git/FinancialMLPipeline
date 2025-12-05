"""Pytest fixtures for FinancialMLPipeline tests.

Provides reusable fixtures for test data generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


@pytest.fixture
def sample_ticks():
    """Generate sample tick data (EURUSD-like)."""
    np.random.seed(42)
    n_ticks = 1000
    base_price = 1.1000
    tick_size = 0.00001  # 1 pip = 10 ticks for EURUSD
    
    dates = pd.date_range(
        start='2024-01-02 00:00:00',
        periods=n_ticks,
        freq='1s',
        tz='UTC'
    )
    
    # Generate realistic price movements
    price_changes = np.random.randn(n_ticks) * tick_size * 5
    bid_prices = base_price + np.cumsum(price_changes)
    ask_prices = bid_prices + tick_size * 5  # 5 tick spread
    
    ticks = pd.DataFrame({
        'timestamp': dates,
        'bidPrice': bid_prices,
        'askPrice': ask_prices,
        'bidVolume': np.random.uniform(0.5, 1.5, n_ticks),
        'askVolume': np.random.uniform(0.5, 1.5, n_ticks),
    })
    
    ticks = ticks.set_index('timestamp')
    return ticks


@pytest.fixture
def sample_bars():
    """Generate sample OHLC bars."""
    np.random.seed(42)
    n_bars = 100
    
    dates = pd.date_range(
        start='2024-01-02 00:00:00',
        periods=n_bars,
        freq='5min',
        tz='UTC'
    )
    
    base_price = 1.1000
    price_changes = np.random.randn(n_bars) * 0.0001
    
    bars = pd.DataFrame({
        'timestamp': dates,
        'bid_open': base_price + np.cumsum(price_changes),
        'bid_high': base_price + np.cumsum(price_changes) + np.abs(np.random.randn(n_bars) * 0.0001),
        'bid_low': base_price + np.cumsum(price_changes) - np.abs(np.random.randn(n_bars) * 0.0001),
        'bid_close': base_price + np.cumsum(price_changes) + np.random.randn(n_bars) * 0.00005,
        'ask_open': base_price + np.cumsum(price_changes) + 0.00005,
        'ask_high': base_price + np.cumsum(price_changes) + 0.00005 + np.abs(np.random.randn(n_bars) * 0.0001),
        'ask_low': base_price + np.cumsum(price_changes) + 0.00005 - np.abs(np.random.randn(n_bars) * 0.0001),
        'ask_close': base_price + np.cumsum(price_changes) + 0.00005 + np.random.randn(n_bars) * 0.00005,
        'tick_count': 100,
        'spread_mean': 0.00005,
        'spread_std': 0.00001,
    })
    
    bars = bars.set_index('timestamp')
    return bars


@pytest.fixture
def sample_session_calendar():
    """Create a sample session calendar."""
    from src.labeling.session_calendar import SessionCalendar
    
    config = {
        'timezone': 'UTC',
        'session_start': '00:00',
        'session_end': '21:55',
        'friday_end': '20:00',
        'weekend_trading': False,
    }
    
    return SessionCalendar(config)


@pytest.fixture
def small_tick_data():
    """Small synthetic tick dataset for fast unit tests."""
    np.random.seed(42)
    n_ticks = 500
    
    dates = pd.date_range(
        start='2024-01-02 00:00:00',
        periods=n_ticks,
        freq='1s',
        tz='UTC'
    )
    
    base_price = 1.1000
    price_changes = np.random.randn(n_ticks) * 0.00001
    
    ticks = pd.DataFrame({
        'timestamp': dates,
        'bidPrice': base_price + np.cumsum(price_changes),
        'askPrice': base_price + np.cumsum(price_changes) + 0.00005,
        'bidVolume': np.random.uniform(0.5, 1.5, n_ticks),
        'askVolume': np.random.uniform(0.5, 1.5, n_ticks),
    })
    
    ticks = ticks.set_index('timestamp')
    return ticks


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test isolation."""
    output_dir = tmp_path / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

