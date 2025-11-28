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
        
        all_features = pd.concat([price_features, micro_features, bar_features], axis=1)
        logger.info(f"Created {len(all_features.columns)} features")
        
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
        
        logger.info(f"Created {len(labels_df)} labels")
        mlflow.log_metric('n_labels', len(labels_df))
        
        # Step 8: Merge features and labels
        logger.info("Step 8: Merging features and labels")
        # Align indexes: labels use event_start, features use bar timestamp
        labels_indexed = labels_df.set_index('event_start') if 'event_start' in labels_df.columns else labels_df
        dataset = all_features.join(labels_indexed, how='inner')
        dataset = dataset.dropna()
        
        if len(dataset) > 0:
            X = dataset.drop(columns=['label', 'pnl', 'barrier_hit'], errors='ignore')
            y = dataset['label']
        else:
            logger.warning(f"Empty merge! Features index sample: {all_features.index[:3].tolist()}, Labels index sample: {labels_indexed.index[:3].tolist() if len(labels_indexed) > 0 else 'empty'}")
            X = all_features.iloc[:0]  # Empty DataFrame with columns
            y = pd.Series([], dtype=int)
        
        logger.info(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
        
        # Step 9: Time-series CV
        logger.info("Step 9: Time-series cross-validation")
        tscv = TimeSeriesCV(
            n_splits=cfg.validation.n_splits,
            train_duration=cfg.validation.train_duration,
            test_duration=cfg.validation.test_duration,
            purge_window=cfg.validation.purge_window,
            embargo_duration=cfg.validation.embargo_duration
        )
        
        # Step 10: Train model (simplified, only one fold for demo)
        logger.info("Step 10: Training Random Forest")
        accuracy = 0.0  # Initialize
        
        if len(X) > 0:
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
                
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train
                rf_model = RandomForestCPU(cfg.models.random_forest)
                rf_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = rf_model.predict(X_test)
                accuracy = (y_pred == y_test).mean()
                
                logger.info(f"Fold {fold} accuracy: {accuracy:.2%}")
                mlflow.log_metric(f'fold_{fold}_accuracy', accuracy)
                
                # Only use first fold for demo
                break
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

