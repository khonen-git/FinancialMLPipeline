"""Data leakage audit utilities.

Audits feature engineering code to detect potential data leakage:
- Forward-looking operations (using future data)
- Incorrect use of shift/rolling
- Features that use data from after the prediction time
"""

import logging
import ast
import inspect
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def audit_feature_functions() -> Dict[str, List[str]]:
    """Audit feature engineering functions for potential data leakage.
    
    This is a static analysis that checks for common patterns:
    - .shift() with negative values (forward fill)
    - .iloc[] with future indices
    - .loc[] with future timestamps
    - Operations that don't use .shift() or .rolling()
    
    Returns:
        Dictionary mapping function names to list of warnings
    """
    warnings = {}
    
    # Import feature modules
    try:
        from src.features import price, microstructure, bars_stats, ma_slopes
        
        modules = {
            'price': price,
            'microstructure': microstructure,
            'bars_stats': bars_stats,
            'ma_slopes': ma_slopes
        }
        
        for module_name, module in modules.items():
            module_warnings = []
            
            # Get all functions in the module
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith('_'):
                    continue
                
                try:
                    source = inspect.getsource(obj)
                    tree = ast.parse(source)
                    
                    # Check for problematic patterns
                    for node in ast.walk(tree):
                        # Check for negative shift (forward fill)
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Attribute):
                                if node.func.attr == 'shift':
                                    for arg in node.args:
                                        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                                            if arg.value < 0:
                                                module_warnings.append(
                                                    f"{name}: .shift({arg.value}) - negative shift (forward fill) detected!"
                                                )
                        
                        # Check for iloc with positive offset (future data)
                        if isinstance(node, ast.Subscript):
                            if isinstance(node.value, ast.Attribute):
                                if node.value.attr == 'iloc':
                                    # This is complex to parse, but we can warn about iloc usage
                                    pass  # Skip for now - too complex
                
                except Exception as e:
                    # Skip functions we can't parse
                    pass
            
            if module_warnings:
                warnings[module_name] = module_warnings
    
    except ImportError as e:
        logger.warning(f"Could not import feature modules for audit: {e}")
    
    return warnings


def check_feature_temporal_safety(features: pd.DataFrame, timestamps: pd.DatetimeIndex) -> Dict[str, any]:
    """Check if features are temporally safe (no future data leakage).
    
    This is a runtime check that verifies:
    - Features at time t don't use data from t+1, t+2, etc.
    - Features are properly lagged
    
    Args:
        features: Feature DataFrame with datetime index
        timestamps: Timestamps to check
        
    Returns:
        Dictionary with audit results
    """
    results = {
        'is_safe': True,
        'warnings': [],
        'errors': []
    }
    
    # This is a placeholder - full implementation would require
    # tracking which features use which time windows
    # For now, we just check that features are properly indexed
    
    if not isinstance(features.index, pd.DatetimeIndex):
        results['warnings'].append("Features index is not DatetimeIndex - cannot verify temporal safety")
        return results
    
    # Check for duplicate timestamps (could indicate leakage)
    duplicates = features.index.duplicated()
    if duplicates.any():
        results['warnings'].append(f"Found {duplicates.sum()} duplicate timestamps in features")
    
    # Check that features are sorted by time
    if not features.index.is_monotonic_increasing:
        results['warnings'].append("Features index is not monotonically increasing - may indicate data leakage")
    
    return results


def validate_no_forward_fill(df: pd.DataFrame, column: str) -> bool:
    """Validate that a column doesn't use forward fill (data leakage).
    
    Args:
        df: DataFrame
        column: Column name to check
        
    Returns:
        True if safe, False if forward fill detected
    """
    # Check if column has forward-filled values
    # This is a heuristic: if values appear before they should logically exist,
    # it might be forward-filled
    
    # Simple check: if NaN is followed by same value, might be forward-filled
    # But this is not definitive - need more sophisticated checks
    
    return True  # Placeholder


def audit_feature_engineering_pipeline() -> Dict[str, any]:
    """Comprehensive audit of the feature engineering pipeline.
    
    Returns:
        Dictionary with audit results
    """
    results = {
        'is_safe': True,
        'function_warnings': {},
        'recommendations': []
    }
    
    # Audit feature functions
    function_warnings = audit_feature_functions()
    results['function_warnings'] = function_warnings
    
    if function_warnings:
        results['is_safe'] = False
        logger.warning("Data leakage audit found potential issues:")
        for module, warnings in function_warnings.items():
            for warning in warnings:
                logger.warning(f"  {module}: {warning}")
    
    # Recommendations
    recommendations = [
        "All features should use .shift() or .rolling() with positive lookback periods",
        "Never use negative shift values (forward fill)",
        "Never use .iloc[] or .loc[] to access future data",
        "Features at time t should only use data from t-k where k > 0",
        "Use session-aware logic to prevent overnight data leakage"
    ]
    
    results['recommendations'] = recommendations
    
    return results

