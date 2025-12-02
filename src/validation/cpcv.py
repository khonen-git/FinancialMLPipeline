"""Combinatorial Purged Cross-Validation (CPCV) with Embargo.

Implements López de Prado's Combinatorial Purged Cross-Validation:
- Partitions time series into N temporal groups (blocks)
- For each fold, selects k groups as test (combinatorial)
- Uses all other groups as training candidates
- Applies embargo after each test block
- Applies purging based on label intervals to prevent label overlap

This method allows training data to come both before and after test blocks,
while ensuring no information leakage through proper purging and embargo.
"""

import logging
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Iterator, Tuple, Optional
from math import comb

logger = logging.getLogger(__name__)


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation according to López de Prado.
    
    This CV method:
    - Partitions data into N temporal groups
    - Generates folds by selecting k groups as test (combinatorial)
    - Allows training on data before and after test (unlike walk-forward)
    - Applies purging to remove overlapping labels
    - Applies embargo after each test block
    
    This creates a richer distribution of out-of-sample performance estimates
    compared to simple walk-forward validation.
    """
    
    def __init__(
        self,
        n_groups: int,
        n_test_groups: int,
        embargo_size: int = 0,
        max_combinations: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """Initialize CPCV splitter.
        
        Args:
            n_groups: Number of temporal groups (blocks) to partition the series into
            n_test_groups: Number of groups used as test for each fold (k)
            embargo_size: Size of embargo (in number of bars) after each test block
            max_combinations: If not None, limit the number of test group combinations
                to sample. If C(n_groups, n_test_groups) is too large, this allows
                random subsampling of combinations.
            random_state: Seed for random subsampling of combinations
            
        Raises:
            ValueError: If n_test_groups is invalid
        """
        if n_test_groups <= 0 or n_test_groups >= n_groups:
            raise ValueError(
                f"n_test_groups must be between 1 and n_groups-1, "
                f"got n_test_groups={n_test_groups}, n_groups={n_groups}"
            )
        
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_size = int(embargo_size)
        self.max_combinations = max_combinations
        self.random_state = random_state
    
    def _make_groups(self, n_samples: int) -> np.ndarray:
        """Create temporal groups by partitioning the series into contiguous blocks.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            Array of group IDs (0 to n_groups-1) for each sample
        """
        # Base size of each group
        base_size = n_samples // self.n_groups
        remainder = n_samples % self.n_groups
        
        group_ids = np.empty(n_samples, dtype=int)
        start = 0
        gid = 0
        
        for g in range(self.n_groups):
            # Distribute remainder across first groups
            size = base_size + (1 if g < remainder else 0)
            end = start + size
            group_ids[start:end] = gid
            start = end
            gid += 1
        
        return group_ids
    
    def _iter_test_group_sets(self) -> Iterator[Tuple[int, ...]]:
        """Generate combinations of test groups.
        
        Yields all combinations C(n_groups, n_test_groups), optionally
        subsampled if max_combinations is set.
        
        Yields:
            Tuple of group IDs to use as test for this fold
        """
        all_combos = list(combinations(range(self.n_groups), self.n_test_groups))
        total_combos = len(all_combos)
        
        # Subsample if needed
        if self.max_combinations is not None and self.max_combinations < total_combos:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(total_combos, size=self.max_combinations, replace=False)
            idx.sort()
            all_combos = [all_combos[i] for i in idx]
            logger.info(
                f"Subsampling {self.max_combinations} combinations from {total_combos} total"
            )
        
        for combo in all_combos:
            yield combo
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        label_indices: Optional[pd.DataFrame] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits for CPCV.
        
        Args:
            X: Feature DataFrame, temporally ordered (ascending index)
            y: Optional target Series (for sklearn compatibility, not used)
            label_indices: Optional DataFrame with 'start_idx' and 'end_idx' columns
                for label-based purging. Must be aligned with X (same index and order).
                These should be integer bar indices.
            groups: Optional pre-assigned groups (length = len(X)).
                If None, groups are created automatically via _make_groups.
                
        Yields:
            Tuple of (train_indices, test_indices) as integer arrays usable with X.iloc[...]
            
        Raises:
            ValueError: If groups length doesn't match X length or number of unique groups
                doesn't match n_groups
        """
        n_samples = len(X)
        if n_samples == 0:
            return
        
        # Determine groups
        if groups is None:
            groups = self._make_groups(n_samples)
        else:
            groups = np.asarray(groups)
            if groups.shape[0] != n_samples:
                raise ValueError(f"len(groups) must equal len(X), got {len(groups)} vs {len(X)}")
        
        unique_groups = np.unique(groups)
        if len(unique_groups) != self.n_groups:
            raise ValueError(
                f"n_groups={self.n_groups}, but {len(unique_groups)} unique groups found. "
                f"If providing custom groups, ensure they contain exactly n_groups unique values."
            )
        
        # Remap group IDs to 0..n_groups-1 if needed
        # This handles cases where user provides groups with non-consecutive IDs (e.g., [10, 20, 30])
        if not np.array_equal(unique_groups, np.arange(self.n_groups)):
            # Create mapping from original IDs to 0..n_groups-1
            mapping = {old_id: new_id for new_id, old_id in enumerate(unique_groups)}
            groups_mapped = np.vectorize(mapping.get)(groups)
            logger.debug(
                f"Remapped group IDs {unique_groups.tolist()} to {list(range(self.n_groups))}"
            )
        else:
            groups_mapped = groups
        
        # Pre-compute: indices for each group (for fast lookup)
        # Use mapped groups (0..n_groups-1) for indexing
        group_to_indices = {
            g: np.flatnonzero(groups_mapped == g) for g in range(self.n_groups)
        }
        
        # Pre-compute label arrays if provided (avoid repeated .iloc calls)
        if label_indices is not None:
            # Assume label_indices is aligned with X
            start_arr = label_indices["start_idx"].to_numpy()
            end_arr = label_indices["end_idx"].to_numpy()
        else:
            start_arr = end_arr = None
        
        # Iterate over all combinations of test groups
        for fold_idx, test_group_set in enumerate(self._iter_test_group_sets()):
            # 1) Build test_idx: union of all test groups
            test_idx_list = [group_to_indices[g] for g in test_group_set]
            if len(test_idx_list) == 0:
                continue
            
            test_idx = np.concatenate(test_idx_list)
            test_idx.sort()
            
            # 2) Build train candidates: all groups not in test
            is_test = np.zeros(n_samples, dtype=bool)
            is_test[test_idx] = True
            is_train_candidate = ~is_test
            
            # 3) Apply embargo (if requested)
            if self.embargo_size > 0:
                embargo_mask = np.zeros(n_samples, dtype=bool)
                
                # Find contiguous runs in test_idx
                # Example: [10, 11, 12, 30, 31] → runs: (10-12), (30-31)
                if len(test_idx) > 0:
                    run_start = test_idx[0]
                    prev = test_idx[0]
                    
                    for idx in test_idx[1:]:
                        if idx == prev + 1:
                            # Same run, continue
                            prev = idx
                        else:
                            # End of run [run_start, prev]
                            start_emb = prev + 1
                            end_emb = min(prev + self.embargo_size, n_samples - 1)
                            if start_emb <= end_emb:
                                embargo_mask[start_emb : end_emb + 1] = True
                            # New run
                            run_start = idx
                            prev = idx
                    
                    # Last run
                    start_emb = prev + 1
                    end_emb = min(prev + self.embargo_size, n_samples - 1)
                    if start_emb <= end_emb:
                        embargo_mask[start_emb : end_emb + 1] = True
                
                is_train_candidate &= ~embargo_mask
            
            train_candidate_idx = np.flatnonzero(is_train_candidate)
            
            # 4) Apply label-based purging (if label_indices provided)
            if label_indices is not None and len(train_candidate_idx) > 0:
                # Period covered by test labels (union of all test label intervals)
                test_start_labels = start_arr[test_idx].min()
                test_end_labels = end_arr[test_idx].max()
                
                # Keep training samples whose label intervals don't overlap
                # [test_start_labels, test_end_labels]
                train_start_i = start_arr[train_candidate_idx]
                train_end_i = end_arr[train_candidate_idx]
                
                # Overlap condition: [start_i, end_i] overlaps [test_start, test_end]
                # if: end_i >= test_start AND start_i <= test_end
                overlap = (
                    (train_end_i >= test_start_labels) &
                    (train_start_i <= test_end_labels)
                )
                train_idx = train_candidate_idx[~overlap]
                
                n_purged = len(train_candidate_idx) - len(train_idx)
                if n_purged > 0:
                    logger.debug(
                        f"Fold {fold_idx} (test groups {test_group_set}): "
                        f"Purged {n_purged} training samples with overlapping labels "
                        f"(test label range: [{test_start_labels}, {test_end_labels}])"
                    )
            else:
                train_idx = train_candidate_idx
            
            # Skip empty folds
            if len(train_idx) == 0 or len(test_idx) == 0:
                logger.warning(
                    f"Fold {fold_idx} (test groups {test_group_set}): "
                    f"Empty train or test set, skipping"
                )
                continue
            
            logger.info(
                f"Fold {fold_idx} (test groups {test_group_set}): "
                f"train={len(train_idx)} samples, test={len(test_idx)} samples"
            )
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits.
        
        Args:
            X: Optional data (for sklearn compatibility)
            y: Optional target (for sklearn compatibility)
            groups: Optional groups (for sklearn compatibility)
            
        Returns:
            Number of splits (theoretical or subsampled)
        """
        total = comb(self.n_groups, self.n_test_groups)
        if self.max_combinations is not None:
            return min(total, self.max_combinations)
        return total
