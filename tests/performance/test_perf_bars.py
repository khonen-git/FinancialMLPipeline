"""Performance tests for bar construction."""

import pytest
import time
from src.data.bars import BarBuilder
from tests.conftest import sample_ticks, small_tick_data


@pytest.mark.perf
class TestBarPerformance:
    """Performance tests for bar construction."""
    
    def test_tick_bars_performance_small(self, small_tick_data):
        """Test tick bars construction performance on small dataset."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        start = time.time()
        bars = builder.build_bars(small_tick_data)
        elapsed = time.time() - start
        
        # Should complete in < 1 second for 500 ticks
        assert elapsed < 1.0, f"Bar construction took {elapsed:.2f}s, threshold: 1s"
        assert len(bars) > 0
    
    def test_tick_bars_performance_medium(self, sample_ticks):
        """Test tick bars construction performance on medium dataset."""
        config = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config)
        
        start = time.time()
        bars = builder.build_bars(sample_ticks)
        elapsed = time.time() - start
        
        # Should complete in < 2 seconds for 1000 ticks
        assert elapsed < 2.0, f"Bar construction took {elapsed:.2f}s, threshold: 2s"
        assert len(bars) > 0
    
    def test_volume_bars_performance(self, sample_ticks):
        """Test volume bars construction performance."""
        config = {'type': 'volume', 'threshold': 500}
        builder = BarBuilder(config)
        
        start = time.time()
        bars = builder.build_bars(sample_ticks)
        elapsed = time.time() - start
        
        # Should complete in < 3 seconds
        assert elapsed < 3.0, f"Volume bar construction took {elapsed:.2f}s, threshold: 3s"
        assert len(bars) > 0
    
    def test_dollar_bars_performance(self, sample_ticks):
        """Test dollar bars construction performance."""
        config = {'type': 'dollar', 'threshold': 100000}
        builder = BarBuilder(config)
        
        start = time.time()
        bars = builder.build_bars(sample_ticks)
        elapsed = time.time() - start
        
        # Should complete in < 3 seconds
        assert elapsed < 3.0, f"Dollar bar construction took {elapsed:.2f}s, threshold: 3s"
        assert len(bars) > 0

