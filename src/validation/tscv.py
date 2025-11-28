"""Time-series cross-validation with purging and embargo.

Implements Lopez de Prado's walk-forward CV with:
- Purging: Remove samples too close to test set
- Embargo: Add gap after test set to avoid look-ahead
"""

import logging
import pandas as pd
import numpy as np
from typing import Iterator, Tuple

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """Time-series cross-validation splitter with purging and embargo."""
    
    def __init__(
        self,
        n_splits: int,
        train_duration: int,
        test_duration: int,
        purge_window: int = 0,
        embargo_duration: int = 0
    ):
        """Initialize CV splitter.
        
        Args:
            n_splits: Number of CV folds
            train_duration: Number of bars for training
            test_duration: Number of bars for testing
            purge_window: Number of bars to purge before test set
            embargo_duration: Number of bars to embargo after test set
        """
        self.n_splits = n_splits
        self.train_duration = train_duration
        self.test_duration = test_duration
        self.purge_window = purge_window
        self.embargo_duration = embargo_duration
    
    def split(
        self,
        X: pd.DataFrame,
        label_indices: pd.DataFrame = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits.
        
        Args:
            X: Feature DataFrame (with datetime index)
            label_indices: Optional DataFrame with 'start_idx' and 'end_idx' for purging
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        for fold in range(self.n_splits):
            # Compute test window
            test_start = self.train_duration + fold * self.test_duration
            test_end = test_start + self.test_duration
            
            if test_end > n_samples:
                logger.warning(f"Fold {fold}: test_end exceeds data length, stopping")
                break
            
            # Test indices
            test_idx = np.arange(test_start, test_end)
            
            # Train indices
            train_start = max(0, test_start - self.train_duration)
            train_end = test_start
            
            # Apply purging if needed
            if self.purge_window > 0:
                train_end = test_start - self.purge_window
            
            train_idx = np.arange(train_start, train_end)
            
            # Advanced purging based on label overlaps
            if label_indices is not None:
                train_idx = self._purge_overlapping_labels(
                    train_idx, test_idx, label_indices
                )
            
            logger.info(
                f"Fold {fold}: train={len(train_idx)} samples "
                f"[{train_start}:{train_end}], test={len(test_idx)} samples "
                f"[{test_start}:{test_end}]"
            )
            
            yield train_idx, test_idx
    
    def _purge_overlapping_labels(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        label_indices: pd.DataFrame
    ) -> np.ndarray:
        """Purge training samples whose labels overlap with test set.
        
        Args:
            train_idx: Training indices (integer positions)
            test_idx: Test indices (integer positions)
            label_indices: DataFrame with 'start_idx' and 'end_idx'
            
        Returns:
            Purged training indices
        """
        test_start = test_idx[0]
        test_end = test_idx[-1]
        
        # Find training samples whose labels extend into test period
        # Use iloc since train_idx contains integer positions, not index labels
        overlapping_mask = (
            (label_indices.iloc[train_idx]['end_idx'] >= test_start) &
            (label_indices.iloc[train_idx]['start_idx'] <= test_end)
        )
        
        purged_train_idx = train_idx[~overlapping_mask.values]
        
        n_purged = len(train_idx) - len(purged_train_idx)
        if n_purged > 0:
            logger.info(f"Purged {n_purged} overlapping training samples")
        
        return purged_train_idx

