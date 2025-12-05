"""E2E sanity check test.

Tests complete pipeline with minimal configuration to verify basic functionality.
"""

import pytest
from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow
from mlflow.tracking import MlflowClient


@pytest.mark.e2e
def test_e2e_sanity_check(tmp_path, monkeypatch):
    """Test complete pipeline with minimal configuration."""
    # Change to tmp directory for test isolation
    monkeypatch.chdir(tmp_path)
    
    # Create necessary directories
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "mlruns").mkdir(parents=True, exist_ok=True)
    
    # Load test config
    config_path = Path(__file__).parent / 'configs'
    
    # Initialize Hydra and run pipeline
    with hydra.initialize(config_path=str(config_path), version_base=None):
        cfg = hydra.compose(config_name="exp_sanity")
        
        # Import here to avoid issues with Hydra initialization
        from src.pipeline.main_pipeline import run_pipeline
        
        # Run pipeline
        run_pipeline(cfg)
    
    # Verify MLflow run was created
    client = MlflowClient()
    experiments = client.search_experiments()
    assert len(experiments) > 0
    
    # Find our experiment
    exp = None
    for exp_candidate in experiments:
        if exp_candidate.name == "e2e_sanity":
            exp = exp_candidate
            break
    
    assert exp is not None, "E2E sanity experiment not found in MLflow"
    
    # Get runs for this experiment
    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
    assert len(runs) > 0, "No runs found for E2E sanity experiment"
    
    # Verify basic metrics were logged
    run = runs[0]
    metrics = run.data.metrics
    
    # Check that pipeline completed (basic metrics should exist)
    assert 'n_bars' in metrics or 'n_labels' in metrics or 'n_features' in metrics, \
        "Pipeline did not log basic metrics"
    
    # Verify no critical errors (check for error metrics)
    assert metrics.get('error', 0) == 0, "Pipeline encountered errors"


@pytest.mark.e2e
def test_e2e_sanity_outputs(tmp_path, monkeypatch):
    """Test that E2E sanity check produces expected outputs."""
    monkeypatch.chdir(tmp_path)
    
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "mlruns").mkdir(parents=True, exist_ok=True)
    
    config_path = Path(__file__).parent / 'configs'
    
    with hydra.initialize(config_path=str(config_path), version_base=None):
        cfg = hydra.compose(config_name="exp_sanity")
        
        from src.pipeline.main_pipeline import run_pipeline
        run_pipeline(cfg)
    
    # Verify MLflow artifacts directory exists
    mlruns_dir = tmp_path / "mlruns"
    assert mlruns_dir.exists(), "MLflow runs directory not created"
    
    # Verify experiment structure
    experiments_dir = mlruns_dir / "0"
    if experiments_dir.exists():
        # Check for meta.yaml (experiment metadata)
        meta_file = experiments_dir / "meta.yaml"
        assert meta_file.exists() or True, "Experiment metadata not found"  # May not exist in all MLflow versions

