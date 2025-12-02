"""Tests for cross-validation implementations.

Tests both TimeSeriesCV and CombinatorialPurgedCV to ensure they work correctly.
"""

import pytest
import pandas as pd
import numpy as np
from src.validation.tscv import TimeSeriesCV
from src.validation.cpcv import CombinatorialPurgedCV


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    }, index=dates)
    
    # Create label_indices with some overlap
    label_indices = pd.DataFrame({
        'start_idx': np.arange(n_samples),
        'end_idx': np.arange(n_samples) + 10  # Each label spans 10 bars
    }, index=dates)
    
    return X, label_indices


def test_time_series_cv_basic(sample_data):
    """Test basic TimeSeriesCV functionality."""
    X, label_indices = sample_data
    
    cv = TimeSeriesCV(
        n_splits=3,
        test_size=100,
        gap=10
    )
    
    splits = list(cv.split(X))
    
    assert len(splits) == 3, "Should generate 3 folds"
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        assert len(train_idx) > 0, f"Fold {fold}: train should not be empty"
        assert len(test_idx) > 0, f"Fold {fold}: test should not be empty"
        assert train_idx.max() < test_idx.min(), f"Fold {fold}: train should be before test"
        assert len(np.intersect1d(train_idx, test_idx)) == 0, f"Fold {fold}: no overlap"


def test_time_series_cv_gap(sample_data):
    """Test that TimeSeriesCV correctly applies gap."""
    X, label_indices = sample_data
    
    cv = TimeSeriesCV(
        n_splits=2,
        test_size=100,
        gap=20
    )
    
    splits = list(cv.split(X))
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        # Check that gap is respected: no samples between train end and test start
        train_end = train_idx.max()
        test_start = test_idx.min()
        gap_size = test_start - train_end - 1
        assert gap_size >= cv.gap, f"Fold {fold}: gap should be at least {cv.gap}, got {gap_size}"


def test_cpcv_basic(sample_data):
    """Test basic CombinatorialPurgedCV functionality."""
    X, label_indices = sample_data
    
    cv = CombinatorialPurgedCV(
        n_groups=10,
        n_test_groups=2,
        embargo_size=5,
        max_combinations=5  # Limit to 5 folds for testing
    )
    
    splits = list(cv.split(X, label_indices=label_indices))
    
    assert len(splits) <= 5, "Should respect max_combinations"
    assert len(splits) > 0, "Should generate at least one fold"
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        assert len(train_idx) > 0, f"Fold {fold}: train should not be empty"
        assert len(test_idx) > 0, f"Fold {fold}: test should not be empty"
        assert len(np.intersect1d(train_idx, test_idx)) == 0, f"Fold {fold}: no overlap"


def test_cpcv_purging(sample_data):
    """Test that CPCV correctly purges overlapping labels."""
    X, label_indices = sample_data
    
    cv = CombinatorialPurgedCV(
        n_groups=8,
        n_test_groups=2,
        embargo_size=0,  # No embargo for this test
        max_combinations=3
    )
    
    splits = list(cv.split(X, label_indices=label_indices))
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        # Check that no training label overlaps with test labels
        test_start = label_indices.iloc[test_idx]['start_idx'].min()
        test_end = label_indices.iloc[test_idx]['end_idx'].max()
        
        train_starts = label_indices.iloc[train_idx]['start_idx'].values
        train_ends = label_indices.iloc[train_idx]['end_idx'].values
        
        # No overlap: train_end < test_start OR train_start > test_end
        overlaps = (train_ends >= test_start) & (train_starts <= test_end)
        assert not overlaps.any(), f"Fold {fold}: Found overlapping labels after purging"


def test_cpcv_embargo(sample_data):
    """Test that CPCV correctly applies embargo."""
    X, label_indices = sample_data
    
    cv = CombinatorialPurgedCV(
        n_groups=10,
        n_test_groups=2,
        embargo_size=10,  # 10 bars embargo
        max_combinations=2
    )
    
    splits = list(cv.split(X, label_indices=label_indices))
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        # Find contiguous runs in test_idx
        test_sorted = np.sort(test_idx)
        if len(test_sorted) > 0:
            # Check that no train sample is within embargo_size after test
            test_max = test_sorted.max()
            embargo_end = test_max + 10
            
            # Train samples should not be in [test_max+1, embargo_end]
            train_in_embargo = train_idx[(train_idx > test_max) & (train_idx <= embargo_end)]
            assert len(train_in_embargo) == 0, f"Fold {fold}: Found train samples in embargo zone"


def test_cpcv_n_splits():
    """Test that get_n_splits returns correct number."""
    cv = CombinatorialPurgedCV(
        n_groups=10,
        n_test_groups=2,
        max_combinations=None
    )
    
    # C(10, 2) = 45
    assert cv.get_n_splits() == 45
    
    cv_limited = CombinatorialPurgedCV(
        n_groups=10,
        n_test_groups=2,
        max_combinations=10
    )
    
    assert cv_limited.get_n_splits() == 10


def test_cpcv_custom_groups(sample_data):
    """Test CPCV with custom non-consecutive group IDs."""
    X, label_indices = sample_data
    
    # Create custom groups with non-consecutive IDs
    n_samples = len(X)
    custom_groups = np.zeros(n_samples, dtype=int)
    group_size = n_samples // 5
    for i in range(5):
        start = i * group_size
        end = (i + 1) * group_size if i < 4 else n_samples
        custom_groups[start:end] = [10, 20, 30, 40, 50][i]  # Non-consecutive IDs
    
    cv = CombinatorialPurgedCV(
        n_groups=5,
        n_test_groups=1,
        embargo_size=0,
        max_combinations=3
    )
    
    # Should work with custom groups (will be remapped internally)
    splits = list(cv.split(X, groups=custom_groups, label_indices=label_indices))
    
    assert len(splits) > 0, "Should generate at least one fold"
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        assert len(train_idx) > 0, f"Fold {fold}: train should not be empty"
        assert len(test_idx) > 0, f"Fold {fold}: test should not be empty"
        assert len(np.intersect1d(train_idx, test_idx)) == 0, f"Fold {fold}: no overlap"

