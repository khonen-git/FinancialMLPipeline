"""Microstructure features.

Features:
- Bid/ask imbalance
- Spread level and changes
- Tick direction
- Order flow imbalance
- Signed volume
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_spread_features(
    bars: pd.DataFrame,
    lookbacks: list[int]
) -> pd.DataFrame:
    """Compute spread-based features.
    
    Args:
        bars: Bars with 'spread_mean' or similar
        lookbacks: List of lookback windows
        
    Returns:
        DataFrame with spread features
    """
    features = pd.DataFrame(index=bars.index)
    
    if 'spread_mean' in bars.columns:
        spread = bars['spread_mean']
    elif 'spread' in bars.columns:
        spread = bars['spread']
    else:
        # Compute from bid/ask
        if 'ask_close' in bars.columns and 'bid_close' in bars.columns:
            spread = bars['ask_close'] - bars['bid_close']
        else:
            logger.warning("No spread data available")
            return features
    
    # Current spread
    features['spread'] = spread
    
    # Spread change
    features['spread_change'] = spread - spread.shift(1)
    
    # Rolling spread stats
    for lb in lookbacks:
        features[f'spread_mean_{lb}'] = spread.rolling(window=lb).mean()
        features[f'spread_std_{lb}'] = spread.rolling(window=lb).std()
    
    return features


def compute_tick_direction(prices: pd.Series) -> pd.Series:
    """Compute tick direction (+1 up, -1 down, 0 flat).
    
    Args:
        prices: Price series
        
    Returns:
        Series with tick direction
    """
    price_change = prices.diff()
    
    direction = pd.Series(0, index=prices.index)
    direction[price_change > 0] = 1
    direction[price_change < 0] = -1
    
    return direction


def compute_order_flow_imbalance(
    bars: pd.DataFrame,
    lookbacks: list[int]
) -> pd.DataFrame:
    """Compute order flow imbalance.
    
    Args:
        bars: Bars with volume data
        lookbacks: List of lookback windows
        
    Returns:
        DataFrame with order flow features
    """
    features = pd.DataFrame(index=bars.index)
    
    # Simple version: use tick direction as proxy
    if 'bid_close' in bars.columns:
        tick_dir = compute_tick_direction(bars['bid_close'])
        features['tick_direction'] = tick_dir
        
        # Rolling imbalance
        for lb in lookbacks:
            features[f'of_imbalance_{lb}'] = tick_dir.rolling(window=lb).mean()
    
    # If we have actual volume data
    if 'bidVolume_sum' in bars.columns and 'askVolume_sum' in bars.columns:
        bid_vol = bars['bidVolume_sum']
        ask_vol = bars['askVolume_sum']
        total_vol = bid_vol + ask_vol
        
        # Avoid division by zero
        imbalance = np.where(
            total_vol > 0,
            (bid_vol - ask_vol) / total_vol,
            0
        )
        features['volume_imbalance'] = imbalance
        
        for lb in lookbacks:
            features[f'vol_imbalance_{lb}'] = pd.Series(imbalance).rolling(window=lb).mean()
    
    return features


def create_microstructure_features(
    bars: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """Create all microstructure features from configuration.
    
    Args:
        bars: OHLC bars DataFrame
        config: Features configuration
        
    Returns:
        DataFrame with microstructure features
    """
    logger.info("Creating microstructure features")
    
    feature_dfs = []
    
    # Spread features
    if 'spread_stats_lookbacks' not in config:
        raise ValueError("Missing required config: spread_stats_lookbacks")
    spread_lookbacks = config['spread_stats_lookbacks']
    spread_df = compute_spread_features(bars, spread_lookbacks)
    feature_dfs.append(spread_df)
    
    # Order flow features
    if 'order_flow_lookbacks' not in config:
        raise ValueError("Missing required config: order_flow_lookbacks")
    of_lookbacks = config['order_flow_lookbacks']
    of_df = compute_order_flow_imbalance(bars, of_lookbacks)
    feature_dfs.append(of_df)
    
    # Combine
    all_features = pd.concat(feature_dfs, axis=1)
    
    logger.info(f"Created {len(all_features.columns)} microstructure features")
    
    return all_features

