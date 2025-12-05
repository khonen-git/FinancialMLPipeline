"""Unit tests for data loader module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from unittest.mock import patch, MagicMock
from src.pipeline.data_loader import load_and_clean_data


@pytest.mark.unit
class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_and_clean_data_csv(self, tmp_path):
        """Test loading CSV data."""
        # Create test CSV file
        csv_file = tmp_path / "test_data.csv"
        dates = pd.date_range('2024-01-01', periods=100, freq='1s', tz='UTC')
        test_data = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'bidPrice': np.random.uniform(1.09, 1.11, 100),
            'askPrice': np.random.uniform(1.09, 1.11, 100),
            'bidVolume': np.random.uniform(0.5, 1.5, 100),
            'askVolume': np.random.uniform(0.5, 1.5, 100),
        })
        test_data.to_csv(csv_file, index=False)
        
        # Create config
        cfg = OmegaConf.create({
            'data': {
                'dukascopy': {
                    'raw_dir': str(tmp_path),
                    'filename': 'test_data.csv',
                    'format': 'csv'
                }
            }
        })
        
        # Mock SchemaDetector to avoid file reading issues
        with patch('src.pipeline.data_loader.SchemaDetector') as mock_detector:
            mock_detector_instance = MagicMock()
            # validate_and_clean should return data with timestamp as column (not index)
            cleaned_data = test_data.copy()
            # Ensure timestamp is datetime
            cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['timestamp'], unit='ms', utc=True)
            mock_detector_instance.validate_and_clean.return_value = cleaned_data
            mock_detector.return_value = mock_detector_instance
            
            result = load_and_clean_data(cfg)
            
            assert len(result) == 100
            # Index should be DatetimeIndex after load_and_clean_data sets it
            assert hasattr(result.index, 'tz') or isinstance(result.index, pd.DatetimeIndex)
    
    def test_load_and_clean_data_missing_filename(self):
        """Test error when filename is missing."""
        cfg = OmegaConf.create({
            'data': {
                'dukascopy': {
                    'raw_dir': 'data/raw',
                    'format': 'csv'
                }
            }
        })
        
        with pytest.raises(ValueError, match="Missing required config: data.dukascopy.filename"):
            load_and_clean_data(cfg)
    
    def test_load_and_clean_data_file_not_found(self):
        """Test error when file doesn't exist."""
        cfg = OmegaConf.create({
            'data': {
                'dukascopy': {
                    'raw_dir': 'data/raw',
                    'filename': 'nonexistent.csv',
                    'format': 'csv'
                }
            }
        })
        
        with pytest.raises(FileNotFoundError):
            load_and_clean_data(cfg)
    
    def test_load_and_clean_data_missing_format(self, tmp_path):
        """Test error when format is missing."""
        # Create a test file first
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("timestamp,bidPrice,askPrice\n1000000,1.10,1.1001\n")
        
        cfg = OmegaConf.create({
            'data': {
                'dukascopy': {
                    'raw_dir': str(tmp_path),
                    'filename': 'test.csv'
                    # Missing format
                }
            }
        })
        
        with pytest.raises(ValueError, match="Missing required config: data.dukascopy.format"):
            load_and_clean_data(cfg)

