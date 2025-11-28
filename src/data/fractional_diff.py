"""Fractional differencing for stationary features.

Implements Lopez de Prado's fractional differencing to achieve
stationarity while preserving memory.
"""

import logging
import pandas as pd
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)


def get_weights(d: float, size: int) -> np.ndarray:
    """Compute fractional differencing weights.
    
    Args:
        d: Differencing order (0 < d < 1)
        size: Number of weights
        
    Returns:
        Array of weights
    """
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    
    return np.array(w)


def frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-3
) -> pd.Series:
    """Apply fractional differencing to a series.
    
    Args:
        series: Time series to difference
        d: Differencing order (0 < d < 1)
        threshold: Weight threshold for truncation
        
    Returns:
        Fractionally differenced series
    """
    if not 0 < d < 1:
        raise ValueError(f"d must be in (0, 1), got {d}")
    
    # Compute weights
    weights = get_weights(d, len(series))
    
    # Truncate small weights
    weights = weights[np.abs(weights) > threshold]
    
    logger.debug(f"Using {len(weights)} weights for d={d}")
    
    # Apply convolution
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(len(weights), len(series)):
        result.iloc[i] = np.dot(weights, series.iloc[i - len(weights):i][::-1])
    
    return result


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5
) -> pd.Series:
    """Apply fixed-width fractional differencing (FFD).
    
    More efficient version with fixed window.
    
    Args:
        series: Time series to difference
        d: Differencing order
        threshold: Weight threshold
        
    Returns:
        Fractionally differenced series
    """
    # Compute weights up to threshold
    weights = []
    w = 1.0
    weights.append(w)
    
    k = 1
    while abs(w) > threshold:
        w = -w * (d - k + 1) / k
        weights.append(w)
        k += 1
    
    weights = np.array(weights)
    width = len(weights)
    
    logger.info(f"FFD with d={d}, width={width}")
    
    # Apply FFD
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(width, len(series)):
        result.iloc[i] = np.dot(weights, series.iloc[i - width:i][::-1])
    
    return result


def apply_frac_diff_to_features(
    features: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """Apply fractional differencing to specified features.
    
    Args:
        features: Feature DataFrame
        config: Configuration with 'd' and 'columns' to difference
        
    Returns:
        DataFrame with differenced features
    """
    if not config.get('enabled', False):
        return features
    
    d = config.get('d', 0.5)
    columns = config.get('columns', [])
    
    if len(columns) == 0:
        logger.info("No columns specified for fractional differencing")
        return features
    
    logger.info(f"Applying fractional differencing (d={d}) to {len(columns)} columns")
    
    result = features.copy()
    
    for col in columns:
        if col in features.columns:
            result[f'{col}_fracdiff'] = frac_diff_ffd(features[col], d)
            logger.debug(f"Differenced {col}")
        else:
            logger.warning(f"Column {col} not found in features")
    
    return result

