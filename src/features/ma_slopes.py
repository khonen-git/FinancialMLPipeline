"""Moving Average slopes features.

Based on proven methodology from previous project:
- Compute MA of different periods
- Calculate slopes: (MA_t - MA_t-n) / n
- Despite correlation, these are effective directional features
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def create_ma_slope_features(bars: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """Create moving average slope features.
    
    Args:
        bars: DataFrame with OHLC data (must have 'close' column)
        periods: List of MA periods (default: [5, 10, 20, 50])
        
    Returns:
        DataFrame with MA slope features
    """
    if periods is None:
        periods = [5, 10, 20, 50]
    
    features = pd.DataFrame(index=bars.index)
    
    # Use mid-price (average of bid_close and ask_close)
    if 'bid_close' in bars.columns and 'ask_close' in bars.columns:
        price = (bars['bid_close'] + bars['ask_close']) / 2
    else:
        price = bars['close']
    
    for period in periods:
        # Moving average
        ma = price.rolling(window=period, min_periods=period).mean()
        
        # MA slope: (MA_t - MA_t-n) / n
        # This measures the rate of change of the MA
        ma_slope = (ma - ma.shift(period)) / period
        
        # Normalize by typical price movement (volatility)
        # This makes slopes comparable across different volatility regimes
        volatility = price.pct_change().rolling(window=period).std()
        ma_slope_norm = ma_slope / (price * volatility.replace(0, np.nan))
        
        features[f'ma_{period}'] = ma
        features[f'ma_{period}_slope'] = ma_slope
        features[f'ma_{period}_slope_norm'] = ma_slope_norm
        
        # Distance from price to MA (normalized)
        features[f'dist_to_ma_{period}'] = (price - ma) / ma
    
    logger.info(f"Created {len(features.columns)} MA slope features for periods {periods}")
    
    return features


def create_ma_cross_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Create MA crossover features.
    
    Args:
        bars: DataFrame with OHLC data
        
    Returns:
        DataFrame with crossover features
    """
    features = pd.DataFrame(index=bars.index)
    
    # Use mid-price
    if 'bid_close' in bars.columns and 'ask_close' in bars.columns:
        price = (bars['bid_close'] + bars['ask_close']) / 2
    else:
        price = bars['close']
    
    # Common MA periods
    ma_5 = price.rolling(window=5, min_periods=5).mean()
    ma_10 = price.rolling(window=10, min_periods=10).mean()
    ma_20 = price.rolling(window=20, min_periods=20).mean()
    ma_50 = price.rolling(window=50, min_periods=50).mean()
    
    # Crossover signals (1 if fast > slow, -1 otherwise)
    features['ma_cross_5_10'] = np.where(ma_5 > ma_10, 1, -1)
    features['ma_cross_10_20'] = np.where(ma_10 > ma_20, 1, -1)
    features['ma_cross_20_50'] = np.where(ma_20 > ma_50, 1, -1)
    
    # Strength of crossover (distance between MAs)
    features['ma_cross_5_10_strength'] = (ma_5 - ma_10) / ma_10
    features['ma_cross_10_20_strength'] = (ma_10 - ma_20) / ma_20
    features['ma_cross_20_50_strength'] = (ma_20 - ma_50) / ma_50
    
    logger.info(f"Created {len(features.columns)} MA crossover features")
    
    return features

