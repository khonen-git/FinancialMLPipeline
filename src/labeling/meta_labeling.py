"""Meta-labeling for trade filtering.

Meta-label = 1 if taking the trade would be profitable, 0 otherwise.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def create_meta_labels(
    base_predictions: pd.Series,
    triple_barrier_labels: pd.DataFrame
) -> pd.Series:
    """Create meta-labels from base model predictions and actual outcomes.
    
    Args:
        base_predictions: Base model directional predictions (1, -1)
        triple_barrier_labels: DataFrame with 'label' and 'pnl' columns
        
    Returns:
        Series with meta-labels (0 or 1)
            1 = trade would be profitable
            0 = trade would not be profitable
    """
    logger.info(f"Creating meta-labels for {len(triple_barrier_labels)} events")
    
    # Meta-label logic:
    # If base model predicts +1 (long), meta-label = 1 if pnl > 0
    # (we could extend to shorts later)
    
    meta_labels = pd.Series(index=triple_barrier_labels.index, dtype=int)
    
    for idx, row in triple_barrier_labels.iterrows():
        # For now, assuming base model always predicts long (+1)
        # Meta-label = 1 if pnl > 0
        meta_labels[idx] = 1 if row['pnl'] > 0 else 0
    
    positive_rate = meta_labels.mean()
    logger.info(f"Meta-labels created. Positive rate: {positive_rate:.2%}")
    
    return meta_labels


def add_meta_label_features(
    features: pd.DataFrame,
    base_model_proba: pd.Series,
    regime_features: pd.DataFrame,
    volatility_features: pd.DataFrame,
    spread_features: pd.DataFrame
) -> pd.DataFrame:
    """Combine features for meta-label model training.
    
    Args:
        features: Base features
        base_model_proba: Base model probability output
        regime_features: HMM regime features
        volatility_features: Volatility features
        spread_features: Spread/cost features
        
    Returns:
        Combined features for meta-model
    """
    meta_features = features.copy()
    meta_features['base_proba'] = base_model_proba
    
    # Add regime features
    if regime_features is not None:
        meta_features = meta_features.join(regime_features, how='left')
    
    # Add volatility
    if volatility_features is not None:
        meta_features = meta_features.join(volatility_features, how='left')
    
    # Add spread
    if spread_features is not None:
        meta_features = meta_features.join(spread_features, how='left')
    
    return meta_features

