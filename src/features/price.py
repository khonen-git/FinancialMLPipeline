"""Price-based feature engineering.

Features:
- Log returns
- Rolling volatility
- Bar ranges
- Multi-timeframe aggregates
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_returns(
    prices: pd.Series,
    lookbacks: list[int]
) -> pd.DataFrame:
    """Compute log returns for multiple lookback periods.
    
    Args:
        prices: Price series (typically 'close')
        lookbacks: List of lookback periods
        
    Returns:
        DataFrame with return columns
    """
    features = pd.DataFrame(index=prices.index)
    
    for lb in lookbacks:
        col_name = f'ret_{lb}'
        features[col_name] = np.log(prices / prices.shift(lb))
    
    return features


def compute_volatility(
    returns: pd.Series,
    lookbacks: list[int]
) -> pd.DataFrame:
    """Compute rolling volatility.
    
    Args:
        returns: Return series
        lookbacks: List of rolling window sizes
        
    Returns:
        DataFrame with volatility columns
    """
    features = pd.DataFrame(index=returns.index)
    
    for lb in lookbacks:
        col_name = f'vol_{lb}'
        features[col_name] = returns.rolling(window=lb).std()
    
    return features


def compute_bar_ranges(
    bars: pd.DataFrame,
    lookbacks: list[int],
    price_col: str = 'close'
) -> pd.DataFrame:
    """Compute bar range statistics.
    
    Args:
        bars: OHLC bars
        lookbacks: List of lookback periods
        price_col: Price column to use for range calculation
        
    Returns:
        DataFrame with range features
    """
    features = pd.DataFrame(index=bars.index)
    
    # Current bar range
    features['bar_range'] = bars['high'] - bars['low']
    
    # Rolling average range
    for lb in lookbacks:
        col_name = f'avg_range_{lb}'
        features[col_name] = features['bar_range'].rolling(window=lb).mean()
    
    return features


def create_price_features(
    bars: pd.DataFrame,
    config: dict,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Create all price-based features from configuration.
    
    Args:
        bars: OHLC bars DataFrame
        config: Features configuration
        price_col: Price column to use
        
    Returns:
        DataFrame with all price features
    """
    logger.info("Creating price-based features")
    
    if 'close' not in bars.columns and 'bid_close' in bars.columns:
        price_col = 'bid_close'
    
    prices = bars[price_col]
    
    feature_dfs = []
    
    # Returns
    if 'returns_lookbacks' not in config:
        raise ValueError("Missing required config: returns_lookbacks")
    returns_lookbacks = config['returns_lookbacks']
    returns_df = compute_returns(prices, returns_lookbacks)
    feature_dfs.append(returns_df)
    
    # Volatility (needs returns)
    if 'volatility_lookbacks' not in config:
        raise ValueError("Missing required config: volatility_lookbacks")
    vol_lookbacks = config['volatility_lookbacks']
    ret_1 = np.log(prices / prices.shift(1))
    vol_df = compute_volatility(ret_1, vol_lookbacks)
    feature_dfs.append(vol_df)
    
    # Bar ranges
    if 'high' in bars.columns and 'low' in bars.columns:
        if 'range_lookbacks' not in config:
            raise ValueError("Missing required config: range_lookbacks")
        range_lookbacks = config['range_lookbacks']
        range_df = compute_bar_ranges(bars, range_lookbacks)
        feature_dfs.append(range_df)
    
    # Combine all
    all_features = pd.concat(feature_dfs, axis=1)
    
    logger.info(f"Created {len(all_features.columns)} price features")
    
    return all_features

