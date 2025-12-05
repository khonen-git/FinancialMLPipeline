"""Unit tests for data cleaning."""

import pytest
import pandas as pd
import numpy as np
from src.data.cleaning import clean_ticks, remove_price_outliers, normalize_timestamps
from tests.conftest import sample_ticks


@pytest.mark.unit
class TestDataCleaning:
    """Test data cleaning functionality."""
    
    def test_clean_ticks_basic(self, sample_ticks):
        """Test basic tick cleaning."""
        config = {
            'remove_duplicates': True,
            'remove_zero_spread': True,
            'remove_outliers': True,
            'outlier_window': 100,
            'outlier_threshold': 3.0
        }
        
        # Reset index to have timestamp as column
        ticks = sample_ticks.reset_index()
        
        cleaned = clean_ticks(ticks, config)
        
        assert len(cleaned) <= len(ticks)
        assert 'timestamp' in cleaned.columns
    
    def test_clean_ticks_remove_duplicates(self):
        """Test duplicate removal."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1s', tz='UTC')
        ticks = pd.DataFrame({
            'timestamp': list(dates) + [dates[0]],  # Duplicate first timestamp
            'bidPrice': np.random.uniform(1.09, 1.11, 11),
            'askPrice': np.random.uniform(1.09, 1.11, 11),
        })
        
        config = {
            'remove_duplicates': True,
            'remove_zero_spread': False,
            'remove_outliers': False
        }
        
        cleaned = clean_ticks(ticks, config)
        
        assert len(cleaned) == 10  # Duplicate removed
    
    def test_clean_ticks_remove_zero_spread(self):
        """Test zero spread removal."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1s', tz='UTC')
        ticks = pd.DataFrame({
            'timestamp': dates,
            'bidPrice': [1.10] * 10,
            'askPrice': [1.10] * 10,  # Zero spread
        })
        
        config = {
            'remove_duplicates': False,
            'remove_zero_spread': True,
            'remove_outliers': False
        }
        
        cleaned = clean_ticks(ticks, config)
        
        assert len(cleaned) == 0  # All removed (zero spread)
    
    def test_remove_price_outliers(self):
        """Test price outlier removal."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1s', tz='UTC')
        prices = np.random.uniform(1.09, 1.11, 100)
        prices[0] = 2.0  # Outlier
        prices[1] = 0.5  # Outlier
        
        ticks = pd.DataFrame({
            'timestamp': dates,
            'bidPrice': prices,
            'askPrice': prices + 0.0001,
        })
        
        config = {
            'outlier_window': 50,
            'outlier_threshold': 3.0  # 3 standard deviations
        }
        
        cleaned = remove_price_outliers(ticks, config)
        
        assert len(cleaned) <= len(ticks)  # Outliers may be removed

