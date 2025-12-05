"""Performance tests for feature engineering."""

import pytest
import time
from src.features.price import create_price_features
from src.features.microstructure import create_microstructure_features
from tests.conftest import sample_bars


@pytest.mark.perf
class TestFeaturePerformance:
    """Performance tests for feature engineering."""
    
    def test_price_features_performance(self, sample_bars):
        """Test price feature engineering performance."""
        config = {
            'returns_lookbacks': [1, 5, 10, 20],
            'volatility_lookbacks': [10, 20, 50],
            'range_lookbacks': [10, 20],
        }
        
        start = time.time()
        features = create_price_features(sample_bars, config)
        elapsed = time.time() - start
        
        # Should complete in < 1 second for 100 bars
        assert elapsed < 1.0, f"Price feature engineering took {elapsed:.2f}s, threshold: 1s"
        assert len(features) == len(sample_bars)
    
    def test_microstructure_features_performance(self, sample_bars):
        """Test microstructure feature engineering performance."""
        config = {
            'spread_stats_lookbacks': [5, 10, 20],
            'order_flow_lookbacks': [5, 10, 20],
        }
        
        start = time.time()
        features = create_microstructure_features(sample_bars, config)
        elapsed = time.time() - start
        
        # Should complete in < 1 second for 100 bars
        assert elapsed < 1.0, f"Microstructure feature engineering took {elapsed:.2f}s, threshold: 1s"
        assert len(features) == len(sample_bars)
    
    def test_combined_features_performance(self, sample_bars):
        """Test combined feature engineering performance."""
        price_config = {
            'returns_lookbacks': [1, 5, 10],
            'volatility_lookbacks': [10, 20],
            'range_lookbacks': [10],
        }
        micro_config = {
            'spread_stats_lookbacks': [5, 10],
            'order_flow_lookbacks': [5, 10],
        }
        
        start = time.time()
        price_features = create_price_features(sample_bars, price_config)
        micro_features = create_microstructure_features(sample_bars, micro_config)
        elapsed = time.time() - start
        
        # Should complete in < 2 seconds for combined features
        assert elapsed < 2.0, f"Combined feature engineering took {elapsed:.2f}s, threshold: 2s"
        assert len(price_features) == len(sample_bars)
        assert len(micro_features) == len(sample_bars)

