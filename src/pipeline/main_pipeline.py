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
import io
from typing import Optional, Dict, Any, Tuple
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow

from src.utils.logging_config import setup_logging
from src.utils.feature_validation import validate_features, validate_feature_consistency
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
from src.models.model_factory import create_random_forest
from src.validation.tscv import TimeSeriesCV
from src.validation.cpcv import CombinatorialPurgedCV
from src.backtest.runner import run_backtest
from src.risk.monte_carlo import run_monte_carlo_simulation, analyze_prop_firm_constraints
from src.reporting.report_generator import ReportGenerator
from src.benchmarks.benchmark_runner import BenchmarkRunner
from src.benchmarks.baselines import BuyAndHold, RandomStrategy, MovingAverageCrossover, RSIStrategy

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def run_pipeline(cfg: DictConfig) -> None:
    """Run full ML pipeline.
    
    Args:
        cfg: Hydra configuration
    """
    log_level = cfg.runtime.get('log_level', 'INFO') if 'runtime' in cfg else 'INFO'
    
    # Setup MLflow first to get tracking URI
    if 'mlflow' not in cfg or 'tracking_uri' not in cfg.mlflow:
        logger.warning("MLflow config not found or incomplete, using default local tracking")
        tracking_uri = 'file:./mlruns'
        experiment_name = cfg.experiment.name
    else:
        tracking_uri = cfg.mlflow.tracking_uri
        experiment_name = cfg.mlflow.get('experiment_name', cfg.experiment.name)
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Setup logging with MLflow handler (logs will be saved to MLflow artifacts)
    mlflow_log_handler = setup_logging(log_level, mlflow_log_handler=None)
    
    logger.info("=" * 80)
    logger.info(f"Starting pipeline: {cfg.experiment.name}")
    logger.info("=" * 80)
    
    with mlflow.start_run():
        # Log model backend
        model_backend = cfg.models.model.get('backend', 'cpu')
        mlflow.log_param('model_backend', model_backend)
        mlflow.log_param('available_backends', ','.join(available_backends))
        
        # Log basic config (TP/SL will be logged later based on distance_mode)
        asset_symbol = cfg.assets.get('symbol', cfg.assets.get('asset', {}).get('symbol', 'UNKNOWN'))
        triple_barrier_cfg = cfg.labeling.get('triple_barrier', {}) if 'labeling' in cfg else {}
        distance_mode = triple_barrier_cfg.get('distance_mode', 'ticks')
        
        mlflow.log_params({
            'asset': asset_symbol,
            'bars_type': cfg.data.bars.type,
            'distance_mode': distance_mode
        })
        
        # Log TP/SL from config if using 'ticks' mode
        if distance_mode == 'ticks':
            mlflow.log_params({
                'tp_ticks': triple_barrier_cfg.get('tp_ticks', 0),
                'sl_ticks': triple_barrier_cfg.get('sl_ticks', 0)
            })
        
        # Step 1-2: Load and clean data
        from src.pipeline.data_loader import load_and_clean_data
        ticks = load_and_clean_data(cfg)
        
        # Step 3-4: Build bars
        from src.pipeline.bar_builder import build_bars
        bars, calendar = build_bars(ticks, cfg)
        
        # Step 5: Feature engineering
        from src.pipeline.feature_engineer import engineer_features
        all_features = engineer_features(bars, cfg)
        
        # Step 6: HMM regime detection (optional)
        logger.info("Step 6: HMM regime detection")
        macro_hmm: Optional[MacroHMM] = None
        micro_hmm: Optional[MicroHMM] = None
        
        if cfg.models.get('hmm', {}).get('macro', {}).get('enabled', False):
            logger.info("Training Macro HMM")
            from src.models.hmm_macro import MacroHMM
            from src.features.hmm_features import create_macro_hmm_features
            macro_features = create_macro_hmm_features(bars, cfg.models.hmm.macro)
            macro_hmm = MacroHMM(cfg.models.hmm.macro)
            macro_hmm.fit(macro_features)
            macro_regimes = macro_hmm.predict(macro_features)
            all_features['macro_regime'] = macro_regimes
        else:
            logger.info("Macro HMM disabled - skipping")
        
        if cfg.models.get('hmm', {}).get('micro', {}).get('enabled', False):
            logger.info("Training Micro HMM")
            from src.models.hmm_micro import MicroHMM
            from src.features.hmm_features import create_micro_hmm_features
            micro_features_hmm = create_micro_hmm_features(bars, ticks, cfg.models.hmm.micro)
            micro_hmm = MicroHMM(cfg.models.hmm.micro)
            micro_hmm.fit(micro_features_hmm)
            micro_regimes = micro_hmm.predict(micro_features_hmm)
            all_features['micro_regime'] = micro_regimes
        else:
            logger.info("Micro HMM disabled - skipping")
        
        # Step 7: Labeling
        from src.pipeline.labeler import create_labels
        labels_df, labels_df_all, labeler = create_labels(bars, all_features, calendar, cfg)
        
        # Log MFE/MAE parameters if used (from labeler)
        if hasattr(labeler, 'mfe_quantile_val') and labeler.mfe_quantile_val is not None:
            mlflow.log_params({
                'tp_ticks_mfe_mae': labeler.tp_ticks,
                'sl_ticks_mfe_mae': labeler.sl_ticks,
                'mfe_quantile': labeler.mfe_quantile_val,
                'mae_quantile': labeler.mae_quantile_val,
                'mfe_mae_horizon_bars': cfg.labeling.triple_barrier.mfe_mae.horizon_bars
            })
        
        mlflow.log_metric('n_labels', len(labels_df_all))
        mlflow.log_metric('n_labels_binary', len(labels_df))
        
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
        
        # Step 8b: Validate features before training
        logger.info("Step 8b: Validating features")
        if len(X) > 0:
            validation_results = validate_features(
                X, y=y, name="training features", strict=True
            )
            mlflow.log_metric('n_features_validated', validation_results['n_features'])
            mlflow.log_metric('n_nan_features', validation_results['n_nan'])
            mlflow.log_metric('n_inf_features', validation_results['n_inf'])
            mlflow.log_metric('n_constant_features', len(validation_results['constant_features']))
            
            if not validation_results['is_valid']:
                raise ValueError(f"Feature validation failed: {validation_results['errors']}")
        else:
            logger.warning("Skipping feature validation: empty feature matrix")
        
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
        meta_models = []  # Store meta-models for backtesting
        
        if len(X) > 0:
            # Pass label_indices to enable advanced purging
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, label_indices=label_indices)):
                logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
                
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Validate feature consistency between train and test
                if fold == 0:  # Only check once to avoid log spam
                    consistency_results = validate_feature_consistency(
                        X_train, X_test, name_train=f"fold_{fold}_train", name_test=f"fold_{fold}_test"
                    )
                    if not consistency_results['is_consistent']:
                        logger.error(f"Feature consistency check failed in fold {fold}")
                        # Don't raise error, but log it - this is a warning
                
                # Validate test features
                if len(X_test) > 0:
                    test_validation = validate_features(
                        X_test, y=y_test, name=f"fold_{fold}_test", strict=False
                    )
                
                # Train Primary Model (direction: 1 vs -1)
                # Use factory to create CPU or GPU model based on config
                import time
                fit_start = time.time()
                primary_model = create_random_forest(dict(cfg.models.model))
                primary_model.fit(X_train, y_train)
                fit_time = time.time() - fit_start
                primary_models.append(primary_model)
                
                # Log training time per fold
                mlflow.log_metric(f'fold_{fold}_fit_time', fit_time)
                logger.info(f"Fold {fold}: Model training took {fit_time:.2f}s")
                
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
                    # CRITICAL: Use the same feature columns as during training
                    # X was created by dropping label columns from dataset, so we need to
                    # replicate that same logic for dataset_all
                    label_cols = ['label', 'pnl', 'barrier_hit', 'event_start', 'event_end', 'bar_index_start', 'bar_index_end']
                    # Get feature columns by excluding label columns and any other non-feature columns
                    feature_cols = [col for col in dataset_all.columns if col not in label_cols and col in all_features.columns]
                    # Ensure we use the exact same columns as X (from training)
                    # If X.columns is available, use it; otherwise use feature_cols
                    if len(X.columns) > 0:
                        # Use only columns that exist in both X and dataset_all
                        X_all_cols = [col for col in X.columns if col in dataset_all.columns]
                        X_all = dataset_all[X_all_cols]
                    else:
                        X_all = dataset_all[feature_cols]
                    
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
                            # Use factory to create CPU or GPU model based on config
                            meta_model = create_random_forest(dict(cfg.models.model))
                            meta_model.fit(X_meta_train, y_meta_train)
                            meta_models.append(meta_model)  # Store for backtesting
                            
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
            # Check if we need to load separate backtest data (out-of-sample)
            backtest_data_file = cfg.backtest.get('data', {}).get('filename', None)
            use_separate_backtest_data = backtest_data_file is not None
            
            if use_separate_backtest_data:
                logger.info(f"Loading separate backtest data: {backtest_data_file}")
                # Load backtest data (e.g., 2024 for out-of-sample testing)
                backtest_data_path = Path(cfg.data.dukascopy.raw_dir) / backtest_data_file
                backtest_file_format = cfg.backtest.get('data', {}).get('format', cfg.data.dukascopy.format)
                if backtest_file_format == 'auto':
                    backtest_file_format = 'csv' if str(backtest_data_path).endswith('.csv') else 'parquet'
                
                if backtest_file_format == 'csv':
                    backtest_ticks = pd.read_csv(backtest_data_path)
                    if 'timestamp' in backtest_ticks.columns:
                        backtest_ticks['timestamp'] = pd.to_datetime(backtest_ticks['timestamp'], unit='ms', utc=True)
                else:
                    backtest_ticks = pd.read_parquet(backtest_data_path)
                
                logger.info(f"Loaded {len(backtest_ticks)} ticks for backtesting from {backtest_data_path}")
                
                # Clean backtest data
                from src.data.schema_detection import SchemaDetector
                detector = SchemaDetector(cfg.data.dukascopy)
                backtest_ticks = detector.validate_and_clean(backtest_ticks)
                if 'timestamp' in backtest_ticks.columns:
                    backtest_ticks = backtest_ticks.set_index('timestamp')
                
                # Build bars for backtest
                from src.data.bars import BarBuilder
                bar_builder = BarBuilder(cfg.data.bars)
                backtest_bars = bar_builder.build_bars(backtest_ticks)
                logger.info(f"Built {len(backtest_bars)} bars for backtesting")
                
                # Build features for backtest (same feature engineering as training)
                backtest_price_features = create_price_features(backtest_bars, cfg.features)
                backtest_micro_features = create_microstructure_features(backtest_bars, cfg.features)
                backtest_bar_features = create_bar_stats_features(backtest_bars, cfg.features)
                from src.features.ma_slopes import create_ma_slope_features, create_ma_cross_features
                backtest_ma_slope_features = create_ma_slope_features(backtest_bars, periods=cfg.features.get('ma_periods', [5, 10, 20, 50]))
                backtest_ma_cross_features = create_ma_cross_features(backtest_bars)
                
                backtest_all_features = pd.concat([
                    backtest_price_features,
                    backtest_micro_features,
                    backtest_bar_features,
                    backtest_ma_slope_features,
                    backtest_ma_cross_features
                ], axis=1)
                
                # Add HMM regimes if enabled (predict on backtest data)
                if macro_hmm is not None:
                    logger.info("Predicting Macro HMM regimes on backtest data")
                    from src.features.hmm_features import create_macro_hmm_features
                    backtest_macro_features = create_macro_hmm_features(backtest_bars, cfg.models.hmm.macro)
                    backtest_macro_regimes = macro_hmm.predict(backtest_macro_features)
                    backtest_all_features['macro_regime'] = backtest_macro_regimes
                
                if micro_hmm is not None:
                    logger.info("Predicting Micro HMM regimes on backtest data")
                    from src.features.hmm_features import create_micro_hmm_features
                    backtest_micro_features_hmm = create_micro_hmm_features(backtest_bars, backtest_ticks, cfg.models.hmm.micro)
                    backtest_micro_regimes = micro_hmm.predict(backtest_micro_features_hmm)
                    backtest_all_features['micro_regime'] = backtest_micro_regimes
                
                # Prepare backtest dataset (drop NaN)
                backtest_all_features = backtest_all_features.dropna()
                logger.info(f"Backtest features: {len(backtest_all_features)} samples, {len(backtest_all_features.columns)} features")
                
                # Use the last trained primary model for predictions on backtest data
                if primary_models:
                    last_primary_model = primary_models[-1]
                    
                    # CRITICAL: Use only features that exist in both training and backtest
                    # Some features (like volume) may be NaN and dropped in one but not the other
                    training_features = set(X.columns)
                    backtest_features = set(backtest_all_features.columns)
                    common_features = sorted(list(training_features & backtest_features))
                    missing_in_backtest = training_features - backtest_features
                    
                    if missing_in_backtest:
                        logger.warning(
                            f"Features in training but missing in backtest: {list(missing_in_backtest)}. "
                            f"These will be excluded from backtest predictions."
                        )
                    
                    if len(common_features) == 0:
                        raise ValueError("No common features between training and backtest!")
                    
                    # Use only common features
                    backtest_X = backtest_all_features[common_features].dropna()
                    
                    # Validate backtest features consistency with training
                    logger.info("Validating backtest features consistency")
                    # Create a subset of X with only common features for comparison
                    X_common = X[common_features]
                    backtest_consistency = validate_feature_consistency(
                        X_common, backtest_X, name_train="training", name_test="backtest"
                    )
                    if not backtest_consistency['is_consistent']:
                        logger.error("Backtest features are inconsistent with training features!")
                        # This is critical - raise error
                        raise ValueError(f"Backtest feature inconsistency: {backtest_consistency['errors']}")
                    
                    # Validate backtest features quality
                    backtest_validation = validate_features(
                        backtest_X, name="backtest features", strict=False
                    )
                    
                    logger.info(f"Generating predictions on {len(backtest_X)} backtest samples using {len(common_features)} common features")
                    backtest_primary_predictions = last_primary_model.predict(backtest_X)
                    backtest_primary_proba = last_primary_model.predict_proba(backtest_X)
                    
                    # Get probability of class +1
                    classes = last_primary_model.model.classes_
                    if len(classes) == 2:
                        pos_class_idx = np.where(classes == 1)[0]
                        if len(pos_class_idx) > 0:
                            backtest_primary_proba_pos = backtest_primary_proba[:, pos_class_idx[0]]
                        else:
                            backtest_primary_proba_pos = backtest_primary_proba[:, 1] if backtest_primary_proba.shape[1] > 1 else backtest_primary_proba[:, 0]
                    else:
                        backtest_primary_proba_pos = backtest_primary_proba[:, 0]
                    
                    # Create predictions DataFrame for backtest
                    backtest_predictions_df = pd.DataFrame({
                        'prediction': backtest_primary_predictions,
                        'probability': backtest_primary_proba_pos,
                    }, index=backtest_X.index)
                    
                    # Add meta-model predictions if available
                    if cfg.models.meta_model.get('enabled', False) and meta_models:
                        logger.info("Using trained meta-model for backtest filtering")
                        last_meta_model = meta_models[-1]
                        
                        # Prepare meta-features: base features + primary model probabilities
                        backtest_meta_features = backtest_X.copy()
                        backtest_meta_features['primary_proba'] = backtest_primary_proba_pos
                        
                        # Get meta-model predictions
                        backtest_meta_predictions = last_meta_model.predict(backtest_meta_features)
                        backtest_predictions_df['meta_decision'] = backtest_meta_predictions
                        logger.info(f"Meta-model filtering: {backtest_meta_predictions.sum()}/{len(backtest_meta_predictions)} trades approved")
                    else:
                        backtest_predictions_df['meta_decision'] = 1
                        if cfg.models.meta_model.get('enabled', False):
                            logger.warning("Meta-model enabled but no trained models available, taking all trades")
                    
                    # Align backtest bars with predictions
                    backtest_bars_aligned = backtest_bars.loc[backtest_bars.index.intersection(backtest_predictions_df.index)]
                    backtest_predictions_aligned = backtest_predictions_df.loc[backtest_bars_aligned.index]
                    
                    if len(backtest_bars_aligned) > 0 and len(backtest_predictions_aligned) > 0:
                        logger.info(f"Running backtest on {len(backtest_bars_aligned)} bars (out-of-sample)")
                        
                        try:
                            bt_results = run_backtest(
                                bars=backtest_bars_aligned,
                                predictions=backtest_predictions_aligned,
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
                            mlflow.log_param('backtest_data_file', backtest_data_file)
                            
                            # Save trade log and equity curve to MLflow
                            if cfg.backtest.logs.get('save_trades', True) and len(bt_results['trade_log']) > 0:
                                trade_log_path = Path('backtest_trade_log.csv')
                                bt_results['trade_log'].to_csv(trade_log_path, index=False)
                                mlflow.log_artifact(str(trade_log_path))
                            
                            if cfg.backtest.logs.get('save_equity', True) and len(bt_results['equity_curve']) > 0:
                                equity_path = Path('backtest_equity_curve.csv')
                                bt_results['equity_curve'].to_csv(equity_path, index=False)
                                mlflow.log_artifact(str(equity_path))
                            
                            logger.info("Out-of-sample backtesting completed successfully")
                        except Exception as e:
                            logger.error(f"Backtesting failed: {e}", exc_info=True)
                            bt_results = {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
                    else:
                        logger.warning("No aligned data for out-of-sample backtesting, skipping")
                        bt_results = {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
                else:
                    logger.warning("No models available for out-of-sample backtesting, skipping")
                    bt_results = {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
            else:
                # Original behavior: backtest on same data as training
                # Prepare predictions for backtesting
                # We'll use the last trained primary model and meta model (if available)
                if len(X) > 0 and primary_models:
                    logger.info("Preparing predictions for backtesting (in-sample)")
                    
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
                    if cfg.models.meta_model.get('enabled', False) and meta_models:
                        logger.info("Using trained meta-model for backtest filtering (in-sample)")
                        last_meta_model = meta_models[-1]
                        
                        # Prepare meta-features: base features + primary model probabilities
                        meta_features_bt = X.copy()
                        meta_features_bt['primary_proba'] = primary_proba_pos
                        
                        # Get meta-model predictions
                        meta_predictions = last_meta_model.predict(meta_features_bt)
                        predictions_df['meta_decision'] = meta_predictions
                        logger.info(f"Meta-model filtering: {meta_predictions.sum()}/{len(meta_predictions)} trades approved")
                    else:
                        predictions_df['meta_decision'] = 1  # Take all trades
                        if cfg.models.meta_model.get('enabled', False):
                            logger.warning("Meta-model enabled but no trained models available, taking all trades")
                    
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
        
        # Step 11b: Benchmark Comparisons
        benchmark_results = None
        if cfg.get('benchmarks', {}).get('enabled', False):
            logger.info("=" * 80)
            logger.info("Step 11b: Running benchmark strategies")
            logger.info("=" * 80)
            
            try:
                # Get bars for benchmarking (use backtest bars if available, otherwise training bars)
                benchmark_bars = backtest_bars_aligned if 'backtest_bars_aligned' in locals() and len(backtest_bars_aligned) > 0 else bars_for_backtest if 'bars_for_backtest' in locals() and len(bars_for_backtest) > 0 else bars
                
                if len(benchmark_bars) == 0:
                    logger.warning("No bars available for benchmarking, skipping")
                else:
                    # Initialize benchmark runner
                    benchmark_runner = BenchmarkRunner(
                        bars=benchmark_bars,
                        session_calendar=calendar,
                        config=dict(cfg),
                        labeling_config=dict(cfg.labeling.triple_barrier),
                        assets_config=dict(cfg.assets)
                    )
                    
                    # Build enabled strategies
                    strategies = {}
                    benchmark_config = cfg.benchmarks
                    
                    if benchmark_config.get('strategies', {}).get('buy_and_hold', {}).get('enabled', False):
                        strategies['buy_and_hold'] = BuyAndHold()
                    
                    if benchmark_config.get('strategies', {}).get('random', {}).get('enabled', False):
                        random_config = benchmark_config.get('strategies', {}).get('random', {})
                        strategies['random'] = RandomStrategy(
                            seed=random_config.get('seed', 42),
                            long_only=random_config.get('long_only', True)
                        )
                    
                    if benchmark_config.get('strategies', {}).get('ma_crossover', {}).get('enabled', False):
                        ma_config = benchmark_config.get('strategies', {}).get('ma_crossover', {})
                        strategies['ma_crossover'] = MovingAverageCrossover(
                            short_period=ma_config.get('short_period', 10),
                            long_period=ma_config.get('long_period', 50)
                        )
                    
                    if benchmark_config.get('strategies', {}).get('rsi', {}).get('enabled', False):
                        rsi_config = benchmark_config.get('strategies', {}).get('rsi', {})
                        strategies['rsi'] = RSIStrategy(
                            period=rsi_config.get('period', 14),
                            oversold=rsi_config.get('oversold', 30),
                            overbought=rsi_config.get('overbought', 70)
                        )
                    
                    if len(strategies) > 0:
                        # Run benchmark comparisons
                        use_extended = benchmark_config.get('metrics', {}).get('extended', True)
                        benchmark_comparison = benchmark_runner.compare_strategies(
                            strategies=strategies,
                            use_extended_metrics=use_extended
                        )
                        
                        # Log benchmark results to MLflow
                        for _, row in benchmark_comparison.iterrows():
                            strategy_name = row['strategy']
                            mlflow.log_metric(f"benchmark_{strategy_name}_sharpe", row['sharpe_ratio'])
                            mlflow.log_metric(f"benchmark_{strategy_name}_return", row['total_return'])
                            mlflow.log_metric(f"benchmark_{strategy_name}_max_drawdown", row['max_drawdown'])
                            mlflow.log_metric(f"benchmark_{strategy_name}_win_rate", row['win_rate'])
                            mlflow.log_metric(f"benchmark_{strategy_name}_total_trades", row['total_trades'])
                            
                            if use_extended:
                                if 'sortino_ratio' in row:
                                    mlflow.log_metric(f"benchmark_{strategy_name}_sortino", row['sortino_ratio'])
                                if 'calmar_ratio' in row:
                                    mlflow.log_metric(f"benchmark_{strategy_name}_calmar", row['calmar_ratio'])
                                if 'profit_factor' in row:
                                    mlflow.log_metric(f"benchmark_{strategy_name}_profit_factor", row['profit_factor'])
                        
                        # Save comparison CSV
                        comparison_path = Path('benchmark_comparison.csv')
                        benchmark_comparison.to_csv(comparison_path, index=False)
                        mlflow.log_artifact(str(comparison_path))
                        
                        # Compare model to baselines if we have model backtest results
                        if 'bt_results' in locals() and bt_results and 'trade_log' in bt_results:
                            logger.info("Comparing model to baseline strategies")
                            comparison_summary = benchmark_runner.compare_model_to_baselines(
                                model_results=bt_results,
                                baseline_strategies=strategies,
                                use_statistical_tests=benchmark_config.get('statistical_tests', {}).get('enabled', True),
                                significance_level=benchmark_config.get('statistical_tests', {}).get('significance_level', 0.05)
                            )
                            
                            # Log statistical test results
                            for baseline_name, test_results in comparison_summary.get('statistical_tests', {}).items():
                                if 't_test' in test_results:
                                    mlflow.log_metric(f"stat_test_{baseline_name}_t_pvalue", test_results['t_test']['p_value'])
                                    mlflow.log_metric(f"stat_test_{baseline_name}_t_significant", float(test_results['t_test_significant']))
                                if 'mann_whitney' in test_results:
                                    mlflow.log_metric(f"stat_test_{baseline_name}_mw_pvalue", test_results['mann_whitney']['p_value'])
                                    mlflow.log_metric(f"stat_test_{baseline_name}_mw_significant", float(test_results['mann_whitney_significant']))
                            
                            benchmark_results = comparison_summary
                        
                        logger.info("Benchmark comparisons completed successfully")
                    else:
                        logger.warning("No benchmark strategies enabled in config")
            except Exception as e:
                logger.error(f"Benchmarking failed: {e}", exc_info=True)
                benchmark_results = None
        
        # Step 12: Risk analysis
        logger.info("Step 12: Risk analysis")
        # Get Monte Carlo config (handle different structures)
        risk_config = cfg.get('risk', {})
        mc_config = risk_config.get('monte_carlo', {}) if isinstance(risk_config, dict) else {}
        if hasattr(risk_config, 'monte_carlo'):
            mc_config = dict(risk_config.monte_carlo)
        n_sims = mc_config.get('n_sims', 1000) if isinstance(mc_config, dict) else getattr(risk_config.monte_carlo, 'n_sims', 1000) if hasattr(risk_config, 'monte_carlo') else 1000
        
        # Only run if enabled or if we have backtest trades
        if mc_config.get('enabled', False):
            # Prepare trade outcomes for Monte Carlo
            if 'pnl' in labels_df.columns and len(labels_df) > 0:
                trade_outcomes = labels_df[['pnl']].copy()
                mc_results = run_monte_carlo_simulation(
                    trade_outcomes,
                    risk_config,
                    n_simulations=n_sims
                )
                mlflow.log_metric('prob_ruin', mc_results.get('prob_ruin', 0))
                mlflow.log_metric('prob_profit_target', mc_results.get('prob_profit_target', 0))
            else:
                logger.info("Monte Carlo simulation: no PnL data available, skipping")
                mc_results = {'prob_ruin': 0, 'prob_profit_target': 0}
        else:
            logger.info("Monte Carlo simulation disabled in config")
            mc_results = {'prob_ruin': 0, 'prob_profit_target': 0}
        
        # Step 13: Generate report
        logger.info("Step 13: Generating report")
        reporting_config = cfg.get('reporting', {})
        if reporting_config.get('enabled', True):
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
                'backtest': bt_results if 'bt_results' in locals() and bt_results else {'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0},
                'risk': mc_results
            }
            
            report_gen = ReportGenerator(Path('templates'))
            output_dir = reporting_config.get('output_dir', 'outputs/reports')
            report_path = Path(output_dir) / f"{cfg.experiment.name}_report.html"
            report_gen.generate_report(results, report_path, dict(cfg))
        else:
            logger.info("Reporting disabled in config")
        
        mlflow.log_artifact(str(report_path))
        
        # Save logs to MLflow artifacts (as per ARCH_INFRA.md §9)
        if mlflow_log_handler is not None:
            log_content = mlflow_log_handler.getvalue()
            if log_content:
                log_path = Path('pipeline.log')
                log_path.write_text(log_content, encoding='utf-8')
                mlflow.log_artifact(str(log_path))
                logger.info("Pipeline logs saved to MLflow artifacts")
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)


if __name__ == '__main__':
    run_pipeline()

