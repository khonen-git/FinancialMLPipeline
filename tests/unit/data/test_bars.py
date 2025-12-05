"""Unit tests for bar construction."""

import pytest
import pandas as pd
import numpy as np
from src.data.bars import BarBuilder


@pytest.mark.unit
class TestBarBuilder:
    """Test BarBuilder functionality."""
    
    def test_tick_bars_construction(self, sample_ticks):
        """Test tick bars construction."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(sample_ticks)
        
        # Should have approximately 10 bars (1000 ticks / 100 threshold)
        assert len(bars) == 10
        
        # Check OHLC structure
        assert 'bid_open' in bars.columns
        assert 'bid_high' in bars.columns
        assert 'bid_low' in bars.columns
        assert 'bid_close' in bars.columns
        assert 'ask_close' in bars.columns
    
    def test_tick_bars_ohlc_logic(self, sample_ticks):
        """Test OHLC logic for tick bars."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(sample_ticks)
        
        # For each bar, high >= low
        for _, bar in bars.iterrows():
            assert bar['bid_high'] >= bar['bid_low']
            assert bar['ask_high'] >= bar['ask_low']
            
            # Open and close should be within high/low
            assert bar['bid_high'] >= bar['bid_open']
            assert bar['bid_low'] <= bar['bid_open']
            assert bar['bid_high'] >= bar['bid_close']
            assert bar['bid_low'] <= bar['bid_close']
    
    def test_tick_bars_metadata(self, sample_ticks):
        """Test bar metadata (tick count, spread)."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(sample_ticks)
        
        # Check metadata columns
        assert 'tick_count' in bars.columns
        assert 'spread_mean' in bars.columns
        
        # Tick count should be 100 for each bar (except maybe last)
        for count in bars['tick_count'].iloc[:-1]:
            assert count == 100
    
    def test_volume_bars_construction(self, sample_ticks):
        """Test volume bars construction."""
        config = {'type': 'volume', 'threshold': 500}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(sample_ticks)
        
        # Should have some bars
        assert len(bars) > 0
        
        # Check volume columns
        assert 'bidVolume_sum' in bars.columns or 'volume' in bars.columns
    
    def test_dollar_bars_construction(self, sample_ticks):
        """Test dollar bars construction."""
        config = {'type': 'dollar', 'threshold': 100000}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(sample_ticks)
        
        # Should have some bars
        assert len(bars) > 0
    
    def test_empty_ticks(self):
        """Test with empty tick data."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        empty_ticks = pd.DataFrame(columns=['bidPrice', 'askPrice', 'bidVolume', 'askVolume'])
        bars = builder.build_bars(empty_ticks)
        
        assert len(bars) == 0
    
    def test_few_ticks(self):
        """Test with fewer ticks than threshold."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        dates = pd.date_range('2024-01-09 10:00', periods=50, freq='1s', tz='UTC')
        ticks = pd.DataFrame({
            'timestamp': dates,
            'bidPrice': 1.0700,
            'askPrice': 1.0705,
            'bidVolume': 100,
            'askVolume': 100
        })
        ticks = ticks.set_index('timestamp')
        
        bars = builder.build_bars(ticks)
        
        # Should have 0 bars (not enough ticks)
        assert len(bars) == 0
    
    def test_bar_timestamps(self, sample_ticks):
        """Test that bars have correct timestamps."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(sample_ticks)
        
        # Timestamps should be in order
        assert bars.index.is_monotonic_increasing
        
        # First bar timestamp should be >= first tick timestamp
        assert bars.index[0] >= sample_ticks.index[0]

