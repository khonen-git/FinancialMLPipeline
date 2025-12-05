"""Feature engineering module.

Handles:
- Price-based features
- Microstructure features
- Bar statistics features
- MA slopes and cross features
- Feature validation
- Data leakage audit (optional)
"""

import logging
import pandas as pd
from omegaconf import DictConfig

from src.features.price import create_price_features
from src.features.microstructure import create_microstructure_features
from src.features.bars_stats import create_bar_stats_features
from src.features.ma_slopes import create_ma_slope_features, create_ma_cross_features
from src.utils.feature_validation import validate_features
from src.utils.data_leakage_audit import audit_feature_engineering_pipeline

logger = logging.getLogger(__name__)


def engineer_features(bars: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Create all features from bars.
    
    Args:
        bars: DataFrame with OHLCV bars
        cfg: Hydra configuration with features section
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Step 5: Feature engineering")
    
    # Price features
    price_features = create_price_features(bars, cfg.features)
    
    # Microstructure features
    micro_features = create_microstructure_features(bars, cfg.features)
    
    # Bar statistics features
    bar_features = create_bar_stats_features(bars, cfg.features)
    
    # MA slopes and cross features (proven from previous project)
    ma_slope_features = create_ma_slope_features(bars, periods=[5, 10, 20, 50])
    ma_cross_features = create_ma_cross_features(bars)
    
    # Combine all features
    all_features = pd.concat([
        price_features, 
        micro_features, 
        bar_features,
        ma_slope_features,
        ma_cross_features
    ], axis=1)
    
    logger.info(f"Created {len(all_features.columns)} total features")
    
    # Optional: Data leakage audit (only in debug mode or first run)
    if cfg.experiment.get('audit_data_leakage', False):
        logger.info("Step 5b: Auditing feature engineering for data leakage")
        leakage_audit = audit_feature_engineering_pipeline()
        if not leakage_audit['is_safe']:
            logger.warning("Data leakage audit found potential issues - review feature engineering")
        import mlflow
        mlflow.log_param('data_leakage_audit_safe', leakage_audit['is_safe'])
    
    # Warn if MFE/MAE features are enabled (data leakage)
    mfe_mae_enabled = cfg.features.get('mfe_mae', {}).get('enabled', False) if 'features' in cfg else False
    if mfe_mae_enabled:
        logger.warning(
            "⚠️ MFE/MAE features are disabled to prevent data leakage. "
            "MFE/MAE uses future data and should only be used for TP/SL parameter selection, "
            "not as model features. Use distance_mode='mfe_mae' in labeling.triple_barrier instead."
        )
    
    return all_features

