"""Unit tests for Time Series Cross-Validation."""

import pytest
import pandas as pd
import numpy as np
from src.validation.tscv import TimeSeriesCV


@pytest.mark.unit
class TestTimeSeriesCV:
    """Test TimeSeriesCV functionality."""
    
    def test_tscv_init(self):
        """Test TimeSeriesCV initialization."""
        cv = TimeSeriesCV(
            n_splits=5,
            test_size=1000,
            gap=50
        )
        
        assert cv.n_splits == 5
        assert cv.test_size == 1000
        assert cv.gap == 50
    
    def test_tscv_split_basic(self):
        """Test basic TSCV split."""
        cv = TimeSeriesCV(
            n_splits=3,
            test_size=100,
            gap=10
        )
        
        # Create sample data
        n_samples = 1000
        X = pd.DataFrame(np.random.randn(n_samples, 10))
        y = pd.Series(np.random.choice([-1, 0, 1], n_samples))
        
        splits = list(cv.split(X, y))
        
        # Should have 3 splits
        assert len(splits) == 3
        
        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # Train should come before test
            assert train_idx[-1] < test_idx[0]
    
    def test_tscv_get_n_splits(self):
        """Test get_n_splits method."""
        cv = TimeSeriesCV(n_splits=5)
        
        assert cv.get_n_splits() == 5

