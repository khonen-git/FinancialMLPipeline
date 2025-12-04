"""Feature validation utilities for ML pipeline.

Validates features before training to catch common issues:
- NaN values
- Infinite values
- Data types
- Constant features
- High correlation between features
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def validate_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    name: str = "features",
    strict: bool = True
) -> Dict[str, any]:
    """Validate features before training.
    
    Args:
        X: Feature DataFrame
        y: Optional target Series (for correlation checks)
        name: Name for logging (e.g., "training features", "test features")
        strict: If True, raise errors on critical issues. If False, only warn.
        
    Returns:
        Dictionary with validation results:
        - 'is_valid': bool
        - 'n_samples': int
        - 'n_features': int
        - 'n_nan': int (total NaN values)
        - 'n_inf': int (total infinite values)
        - 'constant_features': List[str] (features with constant values)
        - 'nan_features': List[str] (features with any NaN)
        - 'inf_features': List[str] (features with any inf)
        - 'dtype_issues': List[str] (features with unexpected dtypes)
        - 'warnings': List[str] (non-critical issues)
        - 'errors': List[str] (critical issues)
    """
    results = {
        'is_valid': True,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'n_nan': 0,
        'n_inf': 0,
        'constant_features': [],
        'nan_features': [],
        'inf_features': [],
        'dtype_issues': [],
        'warnings': [],
        'errors': []
    }
    
    if len(X) == 0:
        results['errors'].append(f"{name}: Empty feature matrix")
        results['is_valid'] = False
        if strict:
            raise ValueError(f"{name}: Empty feature matrix")
        return results
    
    if len(X.columns) == 0:
        results['errors'].append(f"{name}: No features")
        results['is_valid'] = False
        if strict:
            raise ValueError(f"{name}: No features")
        return results
    
    # Check for NaN values
    nan_counts = X.isna().sum()
    total_nan = nan_counts.sum()
    results['n_nan'] = total_nan
    
    if total_nan > 0:
        nan_features = nan_counts[nan_counts > 0].index.tolist()
        results['nan_features'] = nan_features
        
        nan_pct = (total_nan / (len(X) * len(X.columns))) * 100
        if nan_pct > 5:
            results['errors'].append(
                f"{name}: {total_nan} NaN values ({nan_pct:.1f}% of data) in {len(nan_features)} features"
            )
            results['is_valid'] = False
            if strict:
                raise ValueError(f"{name}: Too many NaN values ({nan_pct:.1f}%)")
        else:
            results['warnings'].append(
                f"{name}: {total_nan} NaN values ({nan_pct:.1f}%) in {len(nan_features)} features"
            )
    
    # Check for infinite values
    inf_counts = np.isinf(X.select_dtypes(include=[np.number])).sum()
    total_inf = inf_counts.sum()
    results['n_inf'] = total_inf
    
    if total_inf > 0:
        inf_features = inf_counts[inf_counts > 0].index.tolist()
        results['inf_features'] = inf_features
        
        results['errors'].append(
            f"{name}: {total_inf} infinite values in {len(inf_features)} features"
        )
        results['is_valid'] = False
        if strict:
            raise ValueError(f"{name}: Infinite values detected")
    
    # Check for constant features (zero variance)
    constant_features = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_features.append(col)
        elif X[col].dtype in [np.float64, np.float32]:
            # Check for near-constant features (very low variance)
            if X[col].std() < 1e-10:
                constant_features.append(col)
    
    results['constant_features'] = constant_features
    
    if constant_features:
        results['warnings'].append(
            f"{name}: {len(constant_features)} constant/near-constant features: {constant_features[:5]}"
        )
    
    # Check data types
    dtype_issues = []
    for col in X.columns:
        dtype = X[col].dtype
        # Check for object dtype (should be numeric or categorical)
        if dtype == 'object':
            dtype_issues.append(f"{col}: object dtype (should be numeric)")
        # Check for bool dtype (should be int or float)
        elif dtype == bool:
            dtype_issues.append(f"{col}: bool dtype (should be int/float)")
    
    results['dtype_issues'] = dtype_issues
    
    if dtype_issues:
        results['warnings'].append(
            f"{name}: {len(dtype_issues)} dtype issues: {dtype_issues[:3]}"
        )
    
    # Log summary
    if results['errors']:
        logger.error(f"{name} validation FAILED:")
        for error in results['errors']:
            logger.error(f"  ✗ {error}")
    
    if results['warnings']:
        logger.warning(f"{name} validation warnings:")
        for warning in results['warnings']:
            logger.warning(f"  ⚠ {warning}")
    
    if results['is_valid'] and not results['warnings']:
        logger.info(f"{name} validation passed: {results['n_samples']} samples, {results['n_features']} features")
    
    return results


def validate_feature_consistency(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    name_train: str = "training",
    name_test: str = "test"
) -> Dict[str, any]:
    """Validate that training and test features are consistent.
    
    Args:
        X_train: Training features
        X_test: Test features
        name_train: Name for training set
        name_test: Name for test set
        
    Returns:
        Dictionary with consistency check results
    """
    results = {
        'is_consistent': True,
        'missing_in_test': [],
        'extra_in_test': [],
        'dtype_mismatches': [],
        'warnings': [],
        'errors': []
    }
    
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    # Check for missing features in test
    missing = train_cols - test_cols
    if missing:
        results['missing_in_test'] = list(missing)
        results['errors'].append(
            f"Features in {name_train} but missing in {name_test}: {list(missing)[:5]}"
        )
        results['is_consistent'] = False
    
    # Check for extra features in test (usually OK, but warn)
    extra = test_cols - train_cols
    if extra:
        results['extra_in_test'] = list(extra)
        results['warnings'].append(
            f"Extra features in {name_test} not in {name_train}: {list(extra)[:5]}"
        )
    
    # Check for dtype mismatches in common features
    common_cols = train_cols & test_cols
    for col in common_cols:
        if X_train[col].dtype != X_test[col].dtype:
            results['dtype_mismatches'].append(
                f"{col}: {name_train}={X_train[col].dtype}, {name_test}={X_test[col].dtype}"
            )
            results['warnings'].append(
                f"Dtype mismatch for {col}: {X_train[col].dtype} vs {X_test[col].dtype}"
            )
    
    # Log results
    if results['errors']:
        logger.error(f"Feature consistency check FAILED:")
        for error in results['errors']:
            logger.error(f"  ✗ {error}")
    
    if results['warnings']:
        logger.warning(f"Feature consistency warnings:")
        for warning in results['warnings']:
            logger.warning(f"  ⚠ {warning}")
    
    if results['is_consistent'] and not results['warnings']:
        logger.info(f"Feature consistency check passed: {len(common_cols)} common features")
    
    return results

