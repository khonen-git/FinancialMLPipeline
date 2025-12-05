"""Unit tests for labeler module."""

import pytest
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from unittest.mock import patch, MagicMock
from src.pipeline.labeler import create_labels
from tests.conftest import sample_bars, sample_session_calendar


@pytest.mark.unit
class TestLabeler:
    """Test labeler module."""
    
    def test_create_labels_basic(self, sample_bars, sample_session_calendar):
        """Test basic label creation."""
        # Create sample features
        features = pd.DataFrame(
            index=sample_bars.index,
            data={'feature1': range(len(sample_bars))}
        )
        
        cfg = OmegaConf.create({
            'assets': {
                'tick_size': 0.00001,
                'symbol': 'EURUSD'
            },
            'labeling': {
                'triple_barrier': {
                    'distance_mode': 'ticks',
                    'tp_ticks': 50,
                    'sl_ticks': 50,
                    'max_horizon_bars': 20,
                    'min_horizon_bars': 10,
                    'tick_size': 0.00001
                }
            }
        })
        
        # Mock MLflow to avoid tracking issues in tests
        with patch('src.pipeline.labeler.mlflow') as mock_mlflow:
            mock_mlflow.log_metric = MagicMock()
            
            labels_df, labels_df_all, labeler = create_labels(
                sample_bars, features, sample_session_calendar, cfg
            )
            
            assert labeler is not None
            # Labels may be empty if no events are created, but function should complete
            assert isinstance(labels_df, pd.DataFrame)
            assert isinstance(labels_df_all, pd.DataFrame)
    
    def test_create_labels_missing_tick_size(self, sample_bars, sample_session_calendar):
        """Test error when tick_size is missing."""
        features = pd.DataFrame(
            index=sample_bars.index,
            data={'feature1': range(len(sample_bars))}
        )
        
        cfg = OmegaConf.create({
            'assets': {
                'symbol': 'EURUSD'
                # Missing tick_size
            },
            'labeling': {
                'triple_barrier': {
                    'distance_mode': 'ticks',
                    'tp_ticks': 50,
                    'sl_ticks': 50,
                    'max_horizon_bars': 20,
                    'min_horizon_bars': 10
                }
            }
        })
        
        with pytest.raises(ValueError, match="tick_size not found"):
            create_labels(sample_bars, features, sample_session_calendar, cfg)

