"""Features for HMM training (macro and micro regimes).

Macro: slow market regimes (trend, volatility)
Micro: microstructure regimes (order flow, spread)
"""

import logging
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compute_trend_features(
    prices: pd.Series,
    window: int = 50
) -> pd.DataFrame:
    """Compute trend slope and strength.
    
    Args:
        prices: Price series
        window: Lookback window for regression
        
    Returns:
        DataFrame with trend_slope and trend_strength
    """
    features = pd.DataFrame(index=prices.index)
    
    def rolling_linregress(series):
        """Compute rolling linear regression."""
        slopes = []
        r_values = []
        
        for i in range(len(series)):
            if i < window:
                slopes.append(np.nan)
                r_values.append(np.nan)
                continue
            
            y = series.iloc[i - window + 1 : i + 1].values
            x = np.arange(len(y))
            
            if len(y) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                slopes.append(slope)
                r_values.append(r_value ** 2)  # R²
            else:
                slopes.append(np.nan)
                r_values.append(np.nan)
        
        return pd.Series(slopes, index=series.index), pd.Series(r_values, index=series.index)
    
    features['trend_slope'], features['trend_strength'] = rolling_linregress(prices)
    
    return features


def create_macro_hmm_features(
    bars: pd.DataFrame,
    config: dict,
    price_col: str = 'close'
) -> pd.DataFrame:
    """Create features for macro HMM (slow regimes).
    
    Uses longer timeframe or higher bars for slow regime detection.
    
    Args:
        bars: OHLC bars
        config: HMM macro configuration
        price_col: Price column
        
    Returns:
        DataFrame with macro features
    """
    logger.info("Creating macro HMM features")
    
    if 'bid_close' in bars.columns:
        price_col = 'bid_close'
    
    prices = bars[price_col]
    
    # Long-horizon returns
    ret_long = np.log(prices / prices.shift(50))
    
    # Long-horizon volatility
    vol_long = ret_long.rolling(window=50).std()
    
    # Trend features
    trend_df = compute_trend_features(prices, window=50)
    
    # Combine
    features = pd.DataFrame({
        'ret_long': ret_long,
        'vol_long': vol_long,
        'trend_slope': trend_df['trend_slope'],
        'trend_strength': trend_df['trend_strength'],
    })
    
    logger.info(f"Created {len(features.columns)} macro HMM features")
    
    return features


def create_micro_hmm_features(
    bars: pd.DataFrame,
    ticks: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """Create features for micro HMM (microstructure regimes).
    
    Args:
        bars: OHLC bars
        ticks: Original tick data (for order flow)
        config: HMM micro configuration
        
    Returns:
        DataFrame with micro features
    """
    logger.info("Creating micro HMM features")
    
    features = pd.DataFrame(index=bars.index)
    
    # Spread
    if 'spread_mean' in bars.columns:
        features['spread'] = bars['spread_mean']
        features['spread_change'] = features['spread'] - features['spread'].shift(1)
    
    # Tick direction (from bar closes)
    if 'bid_close' in bars.columns:
        tick_dir = compute_tick_direction_from_bars(bars['bid_close'])
        features['tick_direction'] = tick_dir
    
    # Order flow imbalance (simplified: use tick direction as proxy)
    features['of_imbalance'] = features.get('tick_direction', 0)
    
    # If actual volume available
    if 'bidVolume_sum' in bars.columns and 'askVolume_sum' in bars.columns:
        bid_vol = bars['bidVolume_sum']
        ask_vol = bars['askVolume_sum']
        total_vol = bid_vol + ask_vol
        
        features['of_imbalance'] = np.where(
            total_vol > 0,
            (bid_vol - ask_vol) / total_vol,
            0
        )
    
    logger.info(f"Created {len(features.columns)} micro HMM features")
    
    return features


def compute_tick_direction_from_bars(prices: pd.Series) -> pd.Series:
    """Compute tick direction from bar close prices.
    
    Args:
        prices: Close price series
        
    Returns:
        Series with +1 (up), -1 (down), 0 (flat)
    """
    price_change = prices.diff()
    
    direction = pd.Series(0, index=prices.index, dtype=int)
    direction[price_change > 0] = 1
    direction[price_change < 0] = -1
    
    return direction

