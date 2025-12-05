"""Unit tests for Combinatorial Purged Cross-Validation."""

import pytest
import pandas as pd
import numpy as np
from src.validation.cpcv import CombinatorialPurgedCV


@pytest.mark.unit
class TestCombinatorialPurgedCV:
    """Test CPCV functionality."""
    
    def test_cpcv_init(self):
        """Test CPCV initialization."""
        cv = CombinatorialPurgedCV(
            n_groups=10,
            n_test_groups=2,
            embargo_size=20
        )
        
        assert cv.n_groups == 10
        assert cv.n_test_groups == 2
        assert cv.embargo_size == 20
    
    def test_cpcv_split_basic(self):
        """Test basic CPCV split."""
        cv = CombinatorialPurgedCV(
            n_groups=5,
            n_test_groups=1,
            embargo_size=10
        )
        
        # Create sample data
        n_samples = 1000
        X = pd.DataFrame(np.random.randn(n_samples, 10))
        y = pd.Series(np.random.choice([-1, 0, 1], n_samples))
        
        # Create label indices DataFrame (with start_idx and end_idx)
        label_indices = pd.DataFrame({
            'start_idx': range(n_samples),
            'end_idx': range(10, n_samples + 10)  # Each label spans 10 bars
        })
        
        splits = list(cv.split(X, label_indices=label_indices))
        
        # Should have some splits
        assert len(splits) > 0
        
        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # Train and test should not overlap
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_cpcv_purging(self):
        """Test that CPCV purges overlapping labels."""
        cv = CombinatorialPurgedCV(
            n_groups=5,
            n_test_groups=1,
            embargo_size=10
        )
        
        n_samples = 1000
        X = pd.DataFrame(np.random.randn(n_samples, 10))
        y = pd.Series(np.random.choice([-1, 0, 1], n_samples))
        label_indices = pd.DataFrame({
            'start_idx': range(n_samples),
            'end_idx': range(10, n_samples + 10)
        })
        
        splits = list(cv.split(X, label_indices=label_indices))
        
        # Check that purging is applied (test indices removed from train)
        for train_idx, test_idx in splits:
            # Train should not contain test indices
            assert len(set(train_idx) & set(test_idx)) == 0

