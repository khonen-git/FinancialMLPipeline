"""Meta-labeling for trade filtering.

Meta-label = 1 if taking the trade would be profitable, 0 otherwise.
Long-only strategy: we only consider long positions.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def create_meta_labels(
    triple_barrier_labels: pd.DataFrame
) -> pd.Series:
    """Create meta-labels from triple barrier outcomes (PnL-based).
    
    Meta-labeling for long-only strategy:
    - Meta-label = 1 if PnL > 0 (profitable trade)
    - Meta-label = 0 if PnL <= 0 (unprofitable trade)
    
    This includes ALL triple barrier outcomes (label 1, 0, -1):
    - label=1 (TP hit): PnL > 0 → meta-label=1
    - label=-1 (SL hit): PnL < 0 → meta-label=0
    - label=0 (time barrier): PnL can be positive or negative → meta-label based on PnL
    
    Args:
        triple_barrier_labels: DataFrame with 'label' and 'pnl' columns.
            Must contain all labels (1, 0, -1) with PnL calculated for each.
        
    Returns:
        Series with meta-labels (0 or 1), indexed by triple_barrier_labels index
            1 = trade would be profitable (PnL > 0)
            0 = trade would not be profitable (PnL <= 0)
    """
    logger.info(f"Creating meta-labels for {len(triple_barrier_labels)} events")
    
    if 'pnl' not in triple_barrier_labels.columns:
        raise ValueError("triple_barrier_labels must contain 'pnl' column")
    
    # Meta-label: 1 if PnL > 0, 0 otherwise
    # This works for all barrier types (TP, SL, Time)
    meta_labels = (triple_barrier_labels['pnl'] > 0).astype(int)
    
    # Log distribution
    positive_count = meta_labels.sum()
    positive_rate = meta_labels.mean()
    logger.info(
        f"Meta-labels created: {positive_count}/{len(meta_labels)} positive "
        f"({positive_rate:.2%}), {len(meta_labels) - positive_count} negative "
        f"({1 - positive_rate:.2%})"
    )
    
    # Log breakdown by barrier type
    if 'label' in triple_barrier_labels.columns:
        for barrier_label in [1, 0, -1]:
            mask = triple_barrier_labels['label'] == barrier_label
            if mask.sum() > 0:
                barrier_meta = meta_labels[mask]
                barrier_positive_rate = barrier_meta.mean()
                barrier_name = {1: 'TP', 0: 'Time', -1: 'SL'}[barrier_label]
                logger.info(
                    f"  {barrier_name} barrier (label={barrier_label}): "
                    f"{barrier_meta.sum()}/{len(barrier_meta)} positive "
                    f"({barrier_positive_rate:.2%})"
                )
    
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

