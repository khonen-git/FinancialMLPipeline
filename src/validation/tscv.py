"""Simple time-series cross-validation with gap.

Wrapper around sklearn.model_selection.TimeSeriesSplit with optional gap.
This is a baseline CV method for benchmarking against CPCV (Combinatorial Purged CV).
"""

import logging
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """Simple time-series cross-validation splitter with optional gap.
    
    This is a wrapper around sklearn's TimeSeriesSplit with an optional gap parameter.
    It serves as a baseline for benchmarking against more sophisticated methods
    like Combinatorial Purged Cross-Validation (CPCV).
    
    Unlike CPCV, this method:
    - Does not perform purging based on label intervals
    - Does not apply embargo
    - Simply splits data temporally with an optional gap between train and test
    """
    
    def __init__(
        self,
        n_splits: int,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """Initialize CV splitter.
        
        Args:
            n_splits: Number of CV folds
            test_size: Number of samples in each test set. If None, uses default
                sklearn behavior (test_size = n_samples // (n_splits + 1))
            gap: Number of samples to skip between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        
        # Use sklearn's TimeSeriesSplit internally
        self._sklearn_cv = SklearnTimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap
        )
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        label_indices: Optional[pd.DataFrame] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits.
        
        Args:
            X: Feature DataFrame (with datetime index, ordered temporally)
            y: Optional target Series (for sklearn compatibility)
            label_indices: Optional DataFrame (ignored, kept for API compatibility)
            groups: Optional pre-assigned groups (for sklearn compatibility, ignored)
            
        Yields:
            Tuple of (train_indices, test_indices) as integer arrays usable with X.iloc[...]
        """
        # Convert to numpy array for sklearn compatibility
        # sklearn expects array-like, not DataFrame
        X_array = np.arange(len(X))
        
        # Use sklearn's splitter
        for train_idx, test_idx in self._sklearn_cv.split(X_array, y, groups):
            logger.info(
                f"Fold: train={len(train_idx)} samples "
                f"[{train_idx[0]}:{train_idx[-1]+1}], "
                f"test={len(test_idx)} samples [{test_idx[0]}:{test_idx[-1]+1}], "
                f"gap={self.gap}"
            )
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits.
        
        Args:
            X: Optional data (for sklearn compatibility)
            y: Optional target (for sklearn compatibility)
            groups: Optional groups (for sklearn compatibility)
            
        Returns:
            Number of splits
        """
        return self.n_splits
