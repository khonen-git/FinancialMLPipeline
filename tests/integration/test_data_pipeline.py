"""Integration tests for data pipeline."""

import pytest
import pandas as pd
from src.data.bars import BarBuilder
from src.features.price import create_price_features
from src.features.microstructure import create_microstructure_features


@pytest.mark.integration
class TestDataPipeline:
    """Test complete data pipeline: ticks → bars → features."""
    
    def test_tick_to_bars_to_features(self, sample_ticks):
        """Test complete pipeline: ticks → bars → features."""
        # Step 1: Build bars
        config_bars = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config_bars)
        bars = builder.build_bars(sample_ticks)
        
        assert len(bars) > 0
        assert 'bid_close' in bars.columns
        
        # Step 2: Build price features
        config_features = {
            'returns_lookbacks': [1, 5],
            'volatility_lookbacks': [10],
            'range_lookbacks': [10],
        }
        price_features = create_price_features(bars, config_features)
        
        assert len(price_features) == len(bars)
        assert not price_features.empty
        
        # Step 3: Build microstructure features
        config_micro = {
            'spread_stats_lookbacks': [5],
            'order_flow_lookbacks': [5],
        }
        micro_features = create_microstructure_features(bars, config_micro)
        
        assert len(micro_features) == len(bars)
        
        # Step 4: Combine features
        all_features = pd.concat([price_features, micro_features], axis=1)
        
        assert len(all_features) == len(bars)
        assert len(all_features.columns) > 0
    
    def test_pipeline_index_consistency(self, sample_ticks):
        """Test that indices remain consistent through pipeline."""
        # Build bars
        config_bars = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config_bars)
        bars = builder.build_bars(sample_ticks)
        
        # Build features
        config_features = {
            'returns_lookbacks': [1],
            'volatility_lookbacks': [10],
            'range_lookbacks': [],
        }
        features = create_price_features(bars, config_features)
        
        # Indices should align
        assert features.index.equals(bars.index)
    
    def test_pipeline_no_data_loss(self, sample_ticks):
        """Test that no data is lost in pipeline."""
        # Build bars
        config_bars = {'type': 'tick', 'threshold': 100}
        builder = BarBuilder(config_bars)
        bars = builder.build_bars(sample_ticks)
        
        # Build features
        config_features = {
            'returns_lookbacks': [1],
            'volatility_lookbacks': [],
            'range_lookbacks': [],
        }
        features = create_price_features(bars, config_features)
        
        # Should have same number of rows (after accounting for NaN from rolling)
        # Features may have fewer non-NaN rows due to lookback windows
        assert len(features) == len(bars)

