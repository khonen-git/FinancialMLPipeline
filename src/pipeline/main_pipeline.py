"""Main ML pipeline orchestration.

Orchestrates:
1. Data loading and cleaning
2. Bar construction
3. Feature engineering
4. Labeling (triple barrier + meta)
5. Model training (HMM + RF/GB)
6. Validation (TSCV)
7. Backtesting
8. Risk analysis
9. Reporting
"""

import logging
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow

from src.utils.logging_config import setup_logging
from src.data.schema_detection import SchemaDetector
from src.data.bars import BarBuilder
from src.labeling.session_calendar import SessionCalendar
from src.labeling.triple_barrier import TripleBarrierLabeler
from src.labeling.meta_labeling import create_meta_labels
from src.features.price import create_price_features
from src.features.microstructure import create_microstructure_features
from src.features.bars_stats import create_bar_stats_features
from src.features.hmm_features import create_macro_hmm_features, create_micro_hmm_features
from src.models.hmm_macro import MacroHMM
from src.models.hmm_micro import MicroHMM
from src.models.rf_cpu import RandomForestCPU
from src.validation.tscv import TimeSeriesCV
from src.validation.cpcv import CombinatorialPurgedCV
from src.backtest.runner import run_backtest
from src.risk.monte_carlo import run_monte_carlo_simulation, analyze_prop_firm_constraints
from src.reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def run_pipeline(cfg: DictConfig):
    """Run full ML pipeline.
    
    Args:
        cfg: Hydra configuration
    """
    setup_logging(cfg.runtime.log_level)
    
    logger.info("=" * 80)
    logger.info(f"Starting pipeline: {cfg.experiment.name}")
    logger.info("=" * 80)
    
    # Setup MLflow
    if 'mlflow' not in cfg or 'tracking_uri' not in cfg.mlflow:
        raise ValueError("Missing required config: mlflow.tracking_uri")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    
    with mlflow.start_run():
        # Log config
        mlflow.log_params({
            'asset': cfg.assets.symbol,
            'bars_type': cfg.data.bars.type,
            'tp_ticks': cfg.labeling.triple_barrier.tp_ticks,
            'sl_ticks': cfg.labeling.triple_barrier.sl_ticks
        })
        
        # Step 1: Load data
        logger.info("Step 1: Loading data")
        # Require filename in config
        if 'filename' not in cfg.data.dukascopy:
            raise ValueError("Missing required config: data.dukascopy.filename")
        filename = cfg.data.dukascopy.filename
        data_path = Path(cfg.data.dukascopy.raw_dir) / filename
        
        # Require format in config
        if 'format' not in cfg.data.dukascopy:
            raise ValueError("Missing required config: data.dukascopy.format")
        file_format = cfg.data.dukascopy.format
        if file_format == 'auto':
            file_format = 'csv' if str(data_path).endswith('.csv') else 'parquet'
        
        if file_format == 'csv':
            logger.info(f"Loading CSV: {data_path}")
            ticks = pd.read_csv(data_path)
            # Ensure timestamp column is datetime
            if 'timestamp' in ticks.columns:
                ticks['timestamp'] = pd.to_datetime(ticks['timestamp'], unit='ms', utc=True)
        else:
            ticks = pd.read_parquet(data_path)
        
        logger.info(f"Loaded {len(ticks)} ticks from {data_path}")
        
        # Step 2: Schema detection and cleaning (before setting index)
        logger.info("Step 2: Schema detection and cleaning")
        detector = SchemaDetector(cfg.data.dukascopy)
        ticks = detector.validate_and_clean(ticks)
        
        # Set timestamp as index after validation
        if 'timestamp' in ticks.columns:
            ticks = ticks.set_index('timestamp')
        
        # Step 3: Session calendar
        logger.info("Step 3: Initializing session calendar")
        calendar = SessionCalendar(cfg.session)
        
        # Step 4: Bar construction
        logger.info("Step 4: Bar construction")
        bar_builder = BarBuilder(cfg.data.bars)
        bars = bar_builder.build_bars(ticks)
        logger.info(f"Built {len(bars)} bars")
        
        # Step 5: Feature engineering
        logger.info("Step 5: Feature engineering")
        price_features = create_price_features(bars, cfg.features)
        micro_features = create_microstructure_features(bars, cfg.features)
        bar_features = create_bar_stats_features(bars, cfg.features)
        
        # Add MA slopes features (proven from previous project)
        from src.features.ma_slopes import create_ma_slope_features, create_ma_cross_features
        ma_slope_features = create_ma_slope_features(bars, periods=[5, 10, 20, 50])
        ma_cross_features = create_ma_cross_features(bars)
        
        all_features = pd.concat([
            price_features, 
            micro_features, 
            bar_features,
            ma_slope_features,
            ma_cross_features
        ], axis=1)
        logger.info(f"Created {len(all_features.columns)} total features")
        
        # MFE/MAE should NOT be used as features (data leakage - uses future data)
        # It should only be used for TP/SL parameter selection
        # Check if someone tried to enable it as features and warn them
        mfe_mae_enabled = cfg.features.get('mfe_mae', {}).get('enabled', False)
        if mfe_mae_enabled:
            logger.warning(
                "⚠️ MFE/MAE features are disabled to prevent data leakage. "
                "MFE/MAE uses future data and should only be used for TP/SL parameter selection, "
                "not as model features. Use distance_mode='mfe_mae' in labeling.triple_barrier instead."
            )
        
        # Step 6: HMM regime detection (optional)
        logger.info("Step 6: HMM regime detection")
        if cfg.models.get('hmm', {}).get('macro', {}).get('enabled', False):
            logger.info("Training Macro HMM")
            macro_features = create_macro_hmm_features(bars, cfg.models.hmm.macro)
            macro_hmm = MacroHMM(cfg.models.hmm.macro)
            macro_hmm.fit(macro_features)
            macro_regimes = macro_hmm.predict(macro_features)
            all_features['macro_regime'] = macro_regimes
        else:
            logger.info("Macro HMM disabled - skipping")
        
        if cfg.models.get('hmm', {}).get('micro', {}).get('enabled', False):
            logger.info("Training Micro HMM")
            micro_features_hmm = create_micro_hmm_features(bars, ticks, cfg.models.hmm.micro)
            micro_hmm = MicroHMM(cfg.models.hmm.micro)
            micro_hmm.fit(micro_features_hmm)
            micro_regimes = micro_hmm.predict(micro_features_hmm)
            all_features['micro_regime'] = micro_regimes
        else:
            logger.info("Micro HMM disabled - skipping")
        
        # Step 7: Labeling
        logger.info("Step 7: Triple barrier labeling")
        # Pass bars and assets config for MFE/MAE mode (if needed)
        labeler = TripleBarrierLabeler(
            cfg.labeling.triple_barrier, 
            calendar,
            bars=bars,
            assets_config=dict(cfg.assets)
        )
        labels_df = labeler.label_dataset(bars, all_features.index)
        
        # Log MFE/MAE parameters if used
        if hasattr(labeler, 'mfe_quantile_val') and labeler.mfe_quantile_val is not None:
            mlflow.log_params({
                'tp_ticks_mfe_mae': labeler.tp_ticks,
                'sl_ticks_mfe_mae': labeler.sl_ticks,
                'mfe_quantile': labeler.mfe_quantile_val,
                'mae_quantile': labeler.mae_quantile_val,
                'mfe_mae_horizon_bars': cfg.labeling.triple_barrier.mfe_mae.horizon_bars
            })
        
        logger.info(f"Created {len(labels_df)} labels (before filtering)")
        mlflow.log_metric('n_labels', len(labels_df))
        
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
        
        mlflow.log_metric('n_labels_binary', labels_after)
        
        # Step 8: Merge features and labels
        logger.info("Step 8: Merging features and labels")
        # Use bar_timestamp for accurate alignment with features
        if 'bar_timestamp' in labels_df.columns:
            labels_indexed = labels_df.set_index('bar_timestamp')
            logger.info(f"Using bar_timestamp for merge. Features: {len(all_features)}, Labels: {len(labels_indexed)}")
            
            # Diagnostic: check for exact matches
            common_idx = all_features.index.intersection(labels_indexed.index)
            logger.info(f"Common timestamps before dropna: {len(common_idx)}")
            if len(common_idx) > 0:
                logger.info(f"First common timestamp: {common_idx[0]}")
        elif 'event_start' in labels_df.columns:
            labels_indexed = labels_df.set_index('event_start')
            logger.warning("Using event_start for merge (may cause alignment issues)")
        else:
            labels_indexed = labels_df
        
        dataset = all_features.join(labels_indexed, how='inner')
        logger.info(f"After join: {len(dataset)} samples")
        
        # Check for NaN and drop columns that are entirely NaN
        if len(dataset) > 0:
            nan_counts = dataset.isna().sum()
            all_nan_cols = nan_counts[nan_counts == len(dataset)].index.tolist()
            if all_nan_cols:
                logger.warning(f"Dropping {len(all_nan_cols)} columns with all NaN: {all_nan_cols}")
                dataset = dataset.drop(columns=all_nan_cols)
            
            remaining_nan = dataset.isna().sum()
            if remaining_nan.max() > 0:
                logger.info(f"Columns with some NaN (will be dropped): {remaining_nan[remaining_nan > 0].to_dict()}")
        
        # Drop rows with any NaN in features or label
        dataset = dataset.dropna()
        logger.info(f"After dropna: {len(dataset)} samples")
        
        if len(dataset) > 0:
            X = dataset.drop(columns=['label', 'pnl', 'barrier_hit', 'event_start', 'event_end', 'bar_index_start', 'bar_index_end'], errors='ignore')
            y = dataset['label']
            logger.info(f"Merge successful! {len(dataset)} samples aligned")
        else:
            logger.warning(f"Empty merge! Features index sample: {all_features.index[:3].tolist()}, Labels index sample: {labels_indexed.index[:3].tolist() if len(labels_indexed) > 0 else 'empty'}")
            X = all_features.iloc[:0]  # Empty DataFrame with columns
            y = pd.Series([], dtype=int)
        
        logger.info(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
        
        # Step 9: Cross-validation setup
        if 'cv_type' not in cfg.validation:
            raise ValueError("Missing required config: validation.cv_type")
        cv_type = cfg.validation.cv_type
        logger.info(f"Step 9: Setting up {cv_type} cross-validation")
        
        # Prepare label_indices for advanced purging (avoid label overlap)
        label_indices = pd.DataFrame({
            'start_idx': labels_df.set_index('bar_timestamp').loc[dataset.index, 'bar_index_start'].values,
            'end_idx': labels_df.set_index('bar_timestamp').loc[dataset.index, 'bar_index_end'].values
        }, index=dataset.index)
        
        # Initialize CV splitter based on type
        if cv_type == 'cpcv':
            # Combinatorial Purged Cross-Validation
            if 'n_groups' not in cfg.validation:
                raise ValueError("Missing required config: validation.n_groups (required when cv_type='cpcv')")
            if 'n_test_groups' not in cfg.validation:
                raise ValueError("Missing required config: validation.n_test_groups (required when cv_type='cpcv')")
            
            cv = CombinatorialPurgedCV(
                n_groups=cfg.validation.n_groups,
                n_test_groups=cfg.validation.n_test_groups,
                embargo_size=cfg.validation.get('embargo_duration', 0),
                max_combinations=cfg.validation.get('max_combinations', None),
                random_state=cfg.experiment.get('seed', None)
            )
            logger.info(
                f"CPCV: {cv.n_groups} groups, {cv.n_test_groups} test groups per fold, "
                f"embargo={cv.embargo_size} bars"
            )
        else:
            # Simple time-series CV (baseline, sklearn-based)
            if 'n_splits' not in cfg.validation:
                raise ValueError("Missing required config: validation.n_splits (required when cv_type='time_series')")
            if 'test_size' not in cfg.validation and 'test_duration' not in cfg.validation:
                raise ValueError(
                    "Missing required config: validation.test_size or validation.test_duration "
                    "(required when cv_type='time_series')"
                )
            
            test_size = cfg.validation.get('test_size', cfg.validation.get('test_duration', None))
            cv = TimeSeriesCV(
                n_splits=cfg.validation.n_splits,
                test_size=test_size,
                gap=cfg.validation.get('gap', 0)
            )
            logger.info(
                f"TimeSeriesCV (baseline): {cv.n_splits} folds, "
                f"test_size={cv.test_size}, gap={cv.gap} bars"
            )
        
        # Step 10: Train Primary Model (direction prediction: 1 vs -1)
        logger.info("Step 10: Training Primary Random Forest (direction: +1 vs -1)")
        accuracy = 0.0  # Initialize
        accuracies = []
        primary_models = []  # Store models for meta-labeling
        
        if len(X) > 0:
            # Pass label_indices to enable advanced purging
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, label_indices=label_indices)):
                logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
                
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train Primary Model (direction: 1 vs -1)
                primary_model = RandomForestCPU(cfg.models.random_forest)
                primary_model.fit(X_train, y_train)
                primary_models.append(primary_model)
                
                # Evaluate with multiple metrics (not just accuracy)
                y_pred = primary_model.predict(X_test)
                fold_accuracy = (y_pred == y_test).mean()
                accuracies.append(fold_accuracy)
                
                # Calculate precision, recall, F1 (aligned with trading)
                from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
                
                # For binary classification (+1 vs -1), use pos_label=1
                precision = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
                
                logger.info(f"Fold {fold} Primary Model metrics:")
                logger.info(f"  Accuracy:  {fold_accuracy:.2%}")
                logger.info(f"  Precision: {precision:.2%} (TP when predicted +1)")
                logger.info(f"  Recall:    {recall:.2%}")
                logger.info(f"  F1:        {f1:.2%}")
                
                mlflow.log_metric(f'fold_{fold}_primary_accuracy', fold_accuracy)
                mlflow.log_metric(f'fold_{fold}_primary_precision', precision)
                mlflow.log_metric(f'fold_{fold}_primary_recall', recall)
                mlflow.log_metric(f'fold_{fold}_primary_f1', f1)
            
            # Calculate mean accuracy across all folds
            if accuracies:
                accuracy = np.mean(accuracies)
                logger.info(f"Mean Primary Model accuracy across {len(accuracies)} folds: {accuracy:.2%}")
                mlflow.log_metric('mean_primary_cv_accuracy', accuracy)
        else:
            logger.warning("No data available for training, skipping model training")
        
        # Step 10b: Meta-labeling (PnL-based filtering)
        logger.info("Step 10b: Meta-labeling (PnL-based trade filtering)")
        
        # Prepare complete dataset with all labels (including label=0)
        # Align labels_df_all with features using bar_timestamp
        if len(labels_df_all) > 0:
            if 'bar_timestamp' in labels_df_all.columns:
                labels_all_indexed = labels_df_all.set_index('bar_timestamp')
            elif 'event_start' in labels_df_all.columns:
                labels_all_indexed = labels_df_all.set_index('event_start')
            else:
                logger.warning("Cannot align complete labels for meta-labeling, skipping")
                labels_all_indexed = pd.DataFrame()
        else:
            labels_all_indexed = pd.DataFrame()
        
        if len(labels_all_indexed) > 0 and len(X) > 0:
            # Merge features with complete labels (all barrier types)
            dataset_all = all_features.join(labels_all_indexed, how='inner')
            logger.info(f"Meta-labeling dataset: {len(dataset_all)} samples (including label=0)")
            
            # Create meta-labels: 1 if PnL > 0, 0 otherwise
            if 'pnl' in dataset_all.columns and 'label' in dataset_all.columns:
                # Create meta-labels from complete labels (including label=0)
                meta_labels = create_meta_labels(dataset_all[['label', 'pnl']])
                
                # Prepare meta-features: base features + primary model probabilities
                # For each fold, we need to get primary model probabilities on the full dataset
                # We'll use the last trained primary model for simplicity (or average across folds)
                logger.info("Preparing meta-features (base features + primary model probabilities)")
                
                # Use the last primary model to get probabilities on full dataset
                if primary_models:
                    last_primary_model = primary_models[-1]
                    # Get primary model probabilities for all samples
                    # Note: We need to align with the complete dataset
                    X_all = dataset_all[all_features.columns]
                    
                    # Get probabilities for class +1 (TP)
                    primary_proba = last_primary_model.predict_proba(X_all)
                    # For binary classification, proba shape is (n_samples, 2)
                    # Classes are ordered by sklearn: typically [-1, 1] or [1, -1]
                    # We want the probability of class +1
                    classes = last_primary_model.model.classes_
                    if len(classes) == 2:
                        # Find which column corresponds to class +1
                        pos_class_idx = np.where(classes == 1)[0]
                        if len(pos_class_idx) > 0:
                            primary_proba_pos = primary_proba[:, pos_class_idx[0]]
                        else:
                            # Should not happen, but fallback
                            logger.warning(f"Class +1 not found in primary model classes {classes}, using second column")
                            primary_proba_pos = primary_proba[:, 1]
                    else:
                        logger.warning(f"Unexpected number of classes in primary model: {len(classes)}, using first column")
                        primary_proba_pos = primary_proba[:, 0]
                    
                    # Create meta-features
                    meta_features = X_all.copy()
                    meta_features['primary_proba'] = primary_proba_pos
                    
                    # Align meta_labels with meta_features
                    common_idx = meta_features.index.intersection(meta_labels.index)
                    if len(common_idx) > 0:
                        meta_features_aligned = meta_features.loc[common_idx]
                        meta_labels_aligned = meta_labels.loc[common_idx]
                        
                        logger.info(f"Meta-labeling: {len(meta_features_aligned)} samples, "
                                  f"{meta_labels_aligned.sum()} positive ({meta_labels_aligned.mean():.2%})")
                        
                        # Train Meta Model with cross-validation
                        logger.info("Training Meta Model (PnL-based filtering) with cross-validation")
                        
                        # Prepare label_indices for meta-labeling (same structure)
                        label_indices_meta = pd.DataFrame({
                            'start_idx': labels_all_indexed.loc[common_idx, 'bar_index_start'].values,
                            'end_idx': labels_all_indexed.loc[common_idx, 'bar_index_end'].values
                        }, index=common_idx)
                        
                        meta_accuracies = []
                        for fold, (train_idx, test_idx) in enumerate(cv.split(meta_features_aligned, label_indices=label_indices_meta)):
                            logger.info(f"Meta Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
                            
                            X_meta_train = meta_features_aligned.iloc[train_idx]
                            X_meta_test = meta_features_aligned.iloc[test_idx]
                            y_meta_train = meta_labels_aligned.iloc[train_idx]
                            y_meta_test = meta_labels_aligned.iloc[test_idx]
                            
                            # Train Meta Model
                            meta_model = RandomForestCPU(cfg.models.random_forest)
                            meta_model.fit(X_meta_train, y_meta_train)
                            
                            # Evaluate Meta Model
                            y_meta_pred = meta_model.predict(X_meta_test)
                            meta_accuracy = (y_meta_pred == y_meta_test).mean()
                            meta_accuracies.append(meta_accuracy)
                            
                            # Meta model metrics (precision is key for trading)
                            meta_precision = precision_score(y_meta_test, y_meta_pred, average='binary', pos_label=1, zero_division=0)
                            meta_recall = recall_score(y_meta_test, y_meta_pred, average='binary', pos_label=1, zero_division=0)
                            meta_f1 = f1_score(y_meta_test, y_meta_pred, average='binary', pos_label=1, zero_division=0)
                            
                            logger.info(f"Meta Fold {fold} metrics:")
                            logger.info(f"  Accuracy:  {meta_accuracy:.2%}")
                            logger.info(f"  Precision: {meta_precision:.2%} (key metric: avoid bad trades)")
                            logger.info(f"  Recall:    {meta_recall:.2%}")
                            logger.info(f"  F1:        {meta_f1:.2%}")
                            
                            mlflow.log_metric(f'fold_{fold}_meta_accuracy', meta_accuracy)
                            mlflow.log_metric(f'fold_{fold}_meta_precision', meta_precision)
                            mlflow.log_metric(f'fold_{fold}_meta_recall', meta_recall)
                            mlflow.log_metric(f'fold_{fold}_meta_f1', meta_f1)
                        
                        if meta_accuracies:
                            mean_meta_accuracy = np.mean(meta_accuracies)
                            logger.info(f"Mean Meta Model accuracy: {mean_meta_accuracy:.2%}")
                            mlflow.log_metric('mean_meta_cv_accuracy', mean_meta_accuracy)
                    else:
                        logger.warning("No common indices between meta_features and meta_labels")
                else:
                    logger.warning("No primary models available for meta-labeling")
            else:
                logger.warning("PnL column missing in complete labels, skipping meta-labeling")
        else:
            logger.warning("Cannot prepare meta-labeling dataset, skipping")
        
        # Step 11: Backtesting
        logger.info("Step 11: Backtesting")
        bt_results = None
        
        # Check if backtesting is enabled
        if cfg.backtest.get('enabled', True):
            # Prepare predictions for backtesting
            # We'll use the last trained primary model and meta model (if available)
            if len(X) > 0 and primary_models:
                logger.info("Preparing predictions for backtesting")
                
                # Use the last primary model for predictions
                last_primary_model = primary_models[-1]
                
                # Get primary model predictions on full dataset
                primary_predictions = last_primary_model.predict(X)
                primary_proba = last_primary_model.predict_proba(X)
                
                # Get probability of class +1
                classes = last_primary_model.model.classes_
                if len(classes) == 2:
                    pos_class_idx = np.where(classes == 1)[0]
                    if len(pos_class_idx) > 0:
                        primary_proba_pos = primary_proba[:, pos_class_idx[0]]
                    else:
                        primary_proba_pos = primary_proba[:, 1] if primary_proba.shape[1] > 1 else primary_proba[:, 0]
                else:
                    primary_proba_pos = primary_proba[:, 0]
                
                # Create predictions DataFrame
                predictions_df = pd.DataFrame({
                    'prediction': primary_predictions,
                    'probability': primary_proba_pos,
                }, index=X.index)
                
                # Add meta-model predictions if available
                # For now, we'll use a simple threshold on primary probability
                # In a full implementation, we'd use the trained meta-model
                if cfg.models.meta_model.get('enabled', False):
                    # Meta decision: 1 if probability > threshold, 0 otherwise
                    meta_threshold = cfg.backtest.get('meta_model', {}).get('threshold', 0.5)
                    predictions_df['meta_decision'] = (primary_proba_pos >= meta_threshold).astype(int)
                    logger.info(f"Meta-model filtering enabled: threshold={meta_threshold}")
                else:
                    predictions_df['meta_decision'] = 1  # Take all trades
                
                # Align bars with predictions (use common timestamps)
                bars_for_backtest = bars.loc[bars.index.intersection(predictions_df.index)]
                predictions_aligned = predictions_df.loc[bars_for_backtest.index]
                
                if len(bars_for_backtest) > 0 and len(predictions_aligned) > 0:
                    logger.info(f"Running backtest on {len(bars_for_backtest)} bars with {len(predictions_aligned)} predictions")
                    
                    try:
                        bt_results = run_backtest(
                            bars=bars_for_backtest,
                            predictions=predictions_aligned,
                            session_calendar=calendar,
                            config=dict(cfg.backtest),
                            labeling_config=dict(cfg.labeling.triple_barrier),
                            assets_config=dict(cfg.assets)
                        )
                        
                        # Log backtest metrics to MLflow
                        mlflow.log_metric('backtest_total_trades', bt_results['total_trades'])
                        mlflow.log_metric('backtest_win_rate', bt_results['win_rate'])
                        mlflow.log_metric('backtest_sharpe_ratio', bt_results['sharpe_ratio'])
                        mlflow.log_metric('backtest_max_drawdown', bt_results['max_drawdown'])
                        mlflow.log_metric('backtest_total_pnl', bt_results['total_pnl'])
                        mlflow.log_metric('backtest_total_return', bt_results['total_return'])
                        
                        # Save trade log and equity curve to MLflow
                        if cfg.backtest.logs.get('save_trades', True) and len(bt_results['trade_log']) > 0:
                            trade_log_path = Path('backtest_trade_log.csv')
                            bt_results['trade_log'].to_csv(trade_log_path, index=False)
                            mlflow.log_artifact(str(trade_log_path))
                        
                        if cfg.backtest.logs.get('save_equity', True) and len(bt_results['equity_curve']) > 0:
                            equity_path = Path('backtest_equity_curve.csv')
                            bt_results['equity_curve'].to_csv(equity_path, index=False)
                            mlflow.log_artifact(str(equity_path))
                        
                        logger.info("Backtesting completed successfully")
                    except Exception as e:
                        logger.error(f"Backtesting failed: {e}", exc_info=True)
                        bt_results = {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
                else:
                    logger.warning("No aligned data for backtesting, skipping")
                    bt_results = {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
            else:
                logger.warning("No models available for backtesting, skipping")
                bt_results = {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        else:
            logger.info("Backtesting disabled in config")
            bt_results = {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        # Step 12: Risk analysis
        logger.info("Step 12: Risk analysis")
        mc_results = run_monte_carlo_simulation(
            labels_df[['pnl']],
            cfg.risk,
            n_simulations=cfg.risk.mc_simulations
        )
        
        mlflow.log_metric('prob_ruin', mc_results['prob_ruin'])
        mlflow.log_metric('prob_profit_target', mc_results['prob_profit_target'])
        
        # Step 13: Generate report
        logger.info("Step 13: Generating report")
        results = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_labels': len(labels_df),
            'metrics': {
                'accuracy': accuracy,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'auc': 0
            },
            'backtest': bt_results if bt_results else {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0},
            'risk': mc_results
        }
        
        report_gen = ReportGenerator(Path('templates'))
        report_path = Path(cfg.reporting.output_dir) / f"{cfg.experiment.name}_report.html"
        report_gen.generate_report(results, report_path, dict(cfg))
        
        mlflow.log_artifact(str(report_path))
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)


if __name__ == '__main__':
    run_pipeline()

