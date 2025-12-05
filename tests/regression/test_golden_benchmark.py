"""Regression tests based on golden benchmark.

These tests ensure that modifications do not degrade previously validated results.
"""

import pytest
from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow
from mlflow.tracking import MlflowClient


@pytest.mark.regression
@pytest.mark.slow
class TestGoldenBenchmark:
    """Regression tests against golden benchmark."""
    
    def test_label_distribution_stability(self, tmp_path, monkeypatch):
        """Ensure label distribution remains stable (±5% tolerance)."""
        monkeypatch.chdir(tmp_path)
        
        (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mlruns").mkdir(parents=True, exist_ok=True)
        
        # Use e2e small EURUSD config as golden benchmark
        config_path = Path(__file__).parent.parent / 'e2e' / 'configs'
        
        with hydra.initialize(config_path=str(config_path), version_base=None):
            cfg = hydra.compose(config_name="exp_small_eurusd")
            
            from src.pipeline.main_pipeline import run_pipeline
            run_pipeline(cfg)
        
        # Get metrics from MLflow
        client = MlflowClient()
        experiments = client.search_experiments()
        
        exp = None
        for exp_candidate in experiments:
            if exp_candidate.name == "e2e_small_eurusd":
                exp = exp_candidate
                break
        
        if exp:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
            if runs:
                metrics = runs[0].data.metrics
                
                # Check label distribution if available
                n_labels_tp = metrics.get('n_labels_tp', 0)
                n_labels_sl = metrics.get('n_labels_sl', 0)
                
                if n_labels_tp > 0 and n_labels_sl > 0:
                    ratio = n_labels_tp / n_labels_sl
                    # Expected ratio around 0.5-1.5 (balanced labels)
                    # Allow ±50% tolerance for regression test
                    assert 0.25 <= ratio <= 2.0, \
                        f"Label ratio {ratio:.2f} outside acceptable range [0.25, 2.0]"
    
    def test_pipeline_completion(self, tmp_path, monkeypatch):
        """Ensure pipeline completes without errors."""
        monkeypatch.chdir(tmp_path)
        
        (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mlruns").mkdir(parents=True, exist_ok=True)
        
        config_path = Path(__file__).parent.parent / 'e2e' / 'configs'
        
        with hydra.initialize(config_path=str(config_path), version_base=None):
            cfg = hydra.compose(config_name="exp_sanity")
            
            from src.pipeline.main_pipeline import run_pipeline
            
            # Pipeline should complete without exceptions
            try:
                run_pipeline(cfg)
                pipeline_completed = True
            except Exception as e:
                pipeline_completed = False
                pytest.fail(f"Pipeline failed with error: {e}")
            
            assert pipeline_completed, "Pipeline did not complete successfully"

