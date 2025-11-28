"""Bar statistics features.

Features derived from bar metadata:
- Tick count per bar
- Bar duration
- Bar volume
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def create_bar_stats_features(
    bars: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """Create bar statistics features.
    
    Args:
        bars: Bars DataFrame
        config: Features configuration
        
    Returns:
        DataFrame with bar statistics features
    """
    logger.info("Creating bar statistics features")
    
    features = pd.DataFrame(index=bars.index)
    
    # Tick count (if available)
    if config.get('include_tick_count', True) and 'tick_count' in bars.columns:
        features['tick_count'] = bars['tick_count']
        features['tick_count_norm'] = (
            bars['tick_count'] / bars['tick_count'].rolling(20).mean()
        )
    
    # Bar duration
    if 'bar_duration_sec' in bars.columns:
        features['bar_duration'] = bars['bar_duration_sec']
    
    # Volume (if available and config enabled)
    if config.get('include_volume', True):
        if 'bidVolume_sum' in bars.columns:
            features['volume'] = bars['bidVolume_sum'] + bars.get('askVolume_sum', 0)
        elif 'volume' in bars.columns:
            features['volume'] = bars['volume']
    
    logger.info(f"Created {len(features.columns)} bar statistics features")
    
    return features

