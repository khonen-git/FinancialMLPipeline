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
from src.features.price import create_price_features
from src.features.microstructure import create_microstructure_features
from src.features.bars_stats import create_bar_stats_features
from src.features.hmm_features import create_macro_hmm_features, create_micro_hmm_features
from src.models.hmm_macro import MacroHMM
from src.models.hmm_micro import MicroHMM
from src.models.rf_cpu import RandomForestCPU
from src.validation.tscv import TimeSeriesCV
from src.validation.cpcv import CombinatorialPurgedCV
from src.backtest.backtrader_strategy import SessionAwareStrategy
from src.backtest.data_feed import create_backtrader_feed
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
    tracking_uri = cfg.get('mlflow', {}).get('tracking_uri', './mlruns')
    mlflow.set_tracking_uri(tracking_uri)
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
        # Use filename from config if specified, otherwise construct from symbol
        filename = cfg.data.dukascopy.get('filename', f"{cfg.assets.symbol}.parquet")
        data_path = Path(cfg.data.dukascopy.raw_dir) / filename
        
        # Support both CSV and Parquet formats
        file_format = cfg.data.dukascopy.get('format', 'auto')
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
        
        # Add MFE/MAE features (Maximum Favorable/Adverse Excursion)
        from src.features.mfe_mae import compute_mfe_mae
        if cfg.features.get('mfe_mae', {}).get('enabled', False):
            horizon = cfg.features.mfe_mae.get('horizon_bars', 32)
            quantile = cfg.features.mfe_mae.get('quantile', 0.5)
            mfe_mae_features = compute_mfe_mae(bars, horizon_bars=horizon, quantile=quantile)
        else:
            mfe_mae_features = pd.DataFrame(index=bars.index)
        
        all_features = pd.concat([
            price_features, 
            micro_features, 
            bar_features,
            ma_slope_features,
            ma_cross_features,
            mfe_mae_features
        ], axis=1)
        logger.info(f"Created {len(all_features.columns)} total features")
        
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
        labeler = TripleBarrierLabeler(cfg.labeling.triple_barrier, calendar)
        labels_df = labeler.label_dataset(bars, all_features.index)
        
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
        
        # DROP label=0 (time barrier) - keep only TP (+1) and SL (-1)
        # Binary classification for direction prediction only
        labels_before = len(labels_df)
        labels_df = labels_df[labels_df['label'] != 0].copy()
        labels_after = len(labels_df)
        logger.info(f"Dropped label=0: {labels_before} → {labels_after} ({labels_after/labels_before*100:.1f}% retained)")
        logger.info(f"Binary classification: +1 (TP) vs -1 (SL) only")
        
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
        cv_type = cfg.validation.get('cv_type', 'time_series')
        logger.info(f"Step 9: Setting up {cv_type} cross-validation")
        
        # Prepare label_indices for advanced purging (avoid label overlap)
        label_indices = pd.DataFrame({
            'start_idx': labels_df.set_index('bar_timestamp').loc[dataset.index, 'bar_index_start'].values,
            'end_idx': labels_df.set_index('bar_timestamp').loc[dataset.index, 'bar_index_end'].values
        }, index=dataset.index)
        
        # Initialize CV splitter based on type
        if cv_type == 'cpcv':
            # Combinatorial Purged Cross-Validation
            cv = CombinatorialPurgedCV(
                n_groups=cfg.validation.get('n_groups', 10),
                n_test_groups=cfg.validation.get('n_test_groups', 2),
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
            cv = TimeSeriesCV(
                n_splits=cfg.validation.n_splits,
                test_size=cfg.validation.get('test_duration', None),
                gap=cfg.validation.get('gap', 0)
            )
            logger.info(
                f"TimeSeriesCV (baseline): {cv.n_splits} folds, "
                f"test_size={cv.test_size}, gap={cv.gap} bars"
            )
        
        # Step 10: Train model with cross-validation
        logger.info("Step 10: Training Random Forest with cross-validation")
        accuracy = 0.0  # Initialize
        accuracies = []
        
        if len(X) > 0:
            # Pass label_indices to enable advanced purging
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, label_indices=label_indices)):
                logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
                
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train
                rf_model = RandomForestCPU(cfg.models.random_forest)
                rf_model.fit(X_train, y_train)
                
                # Evaluate with multiple metrics (not just accuracy)
                y_pred = rf_model.predict(X_test)
                fold_accuracy = (y_pred == y_test).mean()
                accuracies.append(fold_accuracy)
                
                # Calculate precision, recall, F1 (aligned with trading)
                from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
                
                # For binary classification (+1 vs -1), use pos_label=1
                precision = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
                
                logger.info(f"Fold {fold} metrics:")
                logger.info(f"  Accuracy:  {fold_accuracy:.2%}")
                logger.info(f"  Precision: {precision:.2%} (TP when predicted +1)")
                logger.info(f"  Recall:    {recall:.2%}")
                logger.info(f"  F1:        {f1:.2%}")
                
                mlflow.log_metric(f'fold_{fold}_accuracy', fold_accuracy)
                mlflow.log_metric(f'fold_{fold}_precision', precision)
                mlflow.log_metric(f'fold_{fold}_recall', recall)
                mlflow.log_metric(f'fold_{fold}_f1', f1)
            
            # Calculate mean accuracy across all folds
            if accuracies:
                accuracy = np.mean(accuracies)
                logger.info(f"Mean accuracy across {len(accuracies)} folds: {accuracy:.2%}")
                mlflow.log_metric('mean_cv_accuracy', accuracy)
        else:
            logger.warning("No data available for training, skipping model training")
        
        # Step 11: Backtesting (simplified placeholder)
        logger.info("Step 11: Backtesting")
        # bt_results = run_backtest(bars, y_pred, cfg.backtest)
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
            'backtest': bt_results,
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

