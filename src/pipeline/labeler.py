"""Labeling module.

Handles:
- Triple barrier labeling
- Meta-labeling
- Label filtering and statistics
"""

import logging
import pandas as pd
from omegaconf import DictConfig
import mlflow

from src.labeling.triple_barrier import TripleBarrierLabeler
from src.labeling.meta_labeling import create_meta_labels

logger = logging.getLogger(__name__)


def create_labels(
    bars: pd.DataFrame,
    all_features: pd.DataFrame,
    calendar,
    cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, TripleBarrierLabeler]:
    """Create triple barrier labels and meta-labels.
    
    Args:
        bars: DataFrame with OHLCV bars
        all_features: DataFrame with all features
        calendar: SessionCalendar instance
        cfg: Hydra configuration with labeling and assets sections
        
    Returns:
        Tuple of (labels_df, labels_df_all, labeler)
        - labels_df: Labels with label=0 filtered out (for primary model)
        - labels_df_all: All labels including label=0 (for meta-labeling)
        - labeler: TripleBarrierLabeler instance
    """
    logger.info("Step 7: Triple barrier labeling")
    
    # Get assets config (handle both structures)
    assets_config = dict(cfg.assets) if 'assets' in cfg else {}
    # Handle nested structure: assets.asset.symbol vs assets.symbol
    if 'asset' in assets_config:
        assets_config = assets_config['asset']
    
    # Get tick_size from assets config
    tick_size = None
    if 'tick_size' in assets_config:
        tick_size = assets_config['tick_size']
    elif hasattr(cfg.assets, 'tick_size'):
        tick_size = cfg.assets.tick_size
    elif hasattr(cfg.assets, 'asset') and hasattr(cfg.assets.asset, 'tick_size'):
        tick_size = cfg.assets.asset.tick_size
    
    # Prepare triple barrier config (add tick_size if in ticks mode)
    triple_barrier_config = dict(cfg.labeling.triple_barrier)
    if triple_barrier_config.get('distance_mode') == 'ticks' and 'tick_size' not in triple_barrier_config:
        if tick_size is None:
            raise ValueError("tick_size not found in assets config (required for distance_mode='ticks')")
        triple_barrier_config['tick_size'] = tick_size
    
    # Create triple barrier labels
    labeler = TripleBarrierLabeler(
        triple_barrier_config,
        calendar,
        bars=bars,
        assets_config=assets_config
    )
    
    labels_df = labeler.label_dataset(bars, all_features.index)
    logger.info(f"Created {len(labels_df)} labels (before filtering)")
    
    # Log label distribution BEFORE filtering
    if 'label' in labels_df.columns:
        label_counts = labels_df['label'].value_counts()
        logger.info(f"Label distribution BEFORE filtering:")
        for label_val, count in label_counts.items():
            pct = count / len(labels_df) * 100
            logger.info(f"  Label {label_val}: {count} ({pct:.1f}%)")
        mlflow.log_metric('n_labels_tp', label_counts.get(1, 0))
        mlflow.log_metric('n_labels_sl', label_counts.get(-1, 0))
        mlflow.log_metric('n_labels_time', label_counts.get(0, 0))
    
    # Save complete labels (with label=0) for meta-labeling BEFORE filtering
    labels_df_all = labels_df.copy()  # Complete labels: 1, 0, -1
    
    # DROP label=0 (time barrier) - keep only TP (+1) and SL (-1)
    # Binary classification for direction prediction only (Primary model)
    labels_before = len(labels_df)
    labels_df = labels_df[labels_df['label'] != 0].copy()
    labels_after = len(labels_df)
    logger.info(f"Dropped label=0: {labels_before} → {labels_after} ({labels_after/labels_before*100:.1f}% retained)")
    logger.info(f"Binary classification: +1 (TP) vs -1 (SL) only (Primary model)")
    logger.info(f"Meta-labeling will use all {len(labels_df_all)} labels (including label=0)")
    
    # Log distribution AFTER filtering
    if len(labels_df) > 0 and 'label' in labels_df.columns:
        label_counts_after = labels_df['label'].value_counts()
        logger.info(f"Label distribution AFTER filtering:")
        for label_val, count in label_counts_after.items():
            pct = count / len(labels_df) * 100
            logger.info(f"  Label {label_val}: {count} ({pct:.1f}%)")
    
    return labels_df, labels_df_all, labeler

