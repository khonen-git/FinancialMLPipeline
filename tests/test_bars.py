"""Unit tests for bar construction."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bars import BarBuilder


class TestBarBuilder(unittest.TestCase):
    """Test BarBuilder functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample tick data
        dates = pd.date_range('2024-01-09 10:00', periods=1000, freq='1s')
        
        self.ticks = pd.DataFrame({
            'timestamp': dates,
            'bidPrice': np.random.uniform(1.0695, 1.0705, 1000),
            'askPrice': np.random.uniform(1.0700, 1.0710, 1000),
            'bidVolume': np.random.uniform(100, 1000, 1000),
            'askVolume': np.random.uniform(100, 1000, 1000)
        })
        self.ticks = self.ticks.set_index('timestamp')
    
    def test_tick_bars_construction(self):
        """Test tick bars (every N ticks)."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(self.ticks)
        
        # Should have 10 bars (1000 ticks / 100 threshold)
        self.assertEqual(len(bars), 10)
        
        # Check OHLC structure
        self.assertIn('bid_open', bars.columns)
        self.assertIn('bid_high', bars.columns)
        self.assertIn('bid_low', bars.columns)
        self.assertIn('bid_close', bars.columns)
        self.assertIn('ask_close', bars.columns)
    
    def test_tick_bars_ohlc_logic(self):
        """Test OHLC logic for tick bars."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(self.ticks)
        
        # For each bar, high >= low
        for idx, bar in bars.iterrows():
            self.assertGreaterEqual(bar['bid_high'], bar['bid_low'])
            self.assertGreaterEqual(bar['ask_high'], bar['ask_low'])
        
        # Open and close should be within high/low
        for idx, bar in bars.iterrows():
            self.assertGreaterEqual(bar['bid_high'], bar['bid_open'])
            self.assertLessEqual(bar['bid_low'], bar['bid_open'])
            self.assertGreaterEqual(bar['bid_high'], bar['bid_close'])
            self.assertLessEqual(bar['bid_low'], bar['bid_close'])
    
    def test_volume_bars_construction(self):
        """Test volume bars."""
        config = {'type': 'volume', 'threshold': 50000}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(self.ticks)
        
        # Should have some bars
        self.assertGreater(len(bars), 0)
        
        # Check volume columns
        self.assertIn('bidVolume_sum', bars.columns)
        self.assertIn('askVolume_sum', bars.columns)
    
    def test_dollar_bars_construction(self):
        """Test dollar bars."""
        config = {'type': 'dollar', 'threshold': 100000}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(self.ticks)
        
        # Should have some bars
        self.assertGreater(len(bars), 0)
    
    def test_bar_metadata(self):
        """Test bar metadata (tick count, spread)."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(self.ticks)
        
        # Check metadata
        self.assertIn('tick_count', bars.columns)
        self.assertIn('spread_mean', bars.columns)
        self.assertIn('spread_std', bars.columns)
        
        # Tick count should be 100 for each bar (except maybe last)
        for count in bars['tick_count'].iloc[:-1]:
            self.assertEqual(count, 100)
    
    def test_spread_calculation(self):
        """Test spread calculation in bars."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        bars = builder.build_bars(self.ticks)
        
        # Spread should be positive
        for spread in bars['spread_mean']:
            self.assertGreater(spread, 0)


class TestBarBuilderEdgeCases(unittest.TestCase):
    """Test edge cases for bar construction."""
    
    def test_empty_ticks(self):
        """Test with empty tick data."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        empty_ticks = pd.DataFrame(columns=['bidPrice', 'askPrice', 'bidVolume', 'askVolume'])
        bars = builder.build_bars(empty_ticks)
        
        self.assertEqual(len(bars), 0)
    
    def test_few_ticks(self):
        """Test with fewer ticks than threshold."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        dates = pd.date_range('2024-01-09 10:00', periods=50, freq='1s')
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
        self.assertEqual(len(bars), 0)
    
    def test_volume_bars_no_volume_data(self):
        """Test volume bars fallback when no volume data."""
        config = {'type': 'volume', 'threshold': 5000}
        builder = BarBuilder(config)
        
        # Ticks without volume
        dates = pd.date_range('2024-01-09 10:00', periods=1000, freq='1s')
        ticks = pd.DataFrame({
            'timestamp': dates,
            'bidPrice': np.random.uniform(1.0695, 1.0705, 1000),
            'askPrice': np.random.uniform(1.0700, 1.0710, 1000)
        })
        ticks = ticks.set_index('timestamp')
        
        # Should fallback to tick bars
        bars = builder.build_bars(ticks)
        self.assertGreater(len(bars), 0)


if __name__ == '__main__':
    unittest.main()

