"""Unit tests for schema detection."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema_detection import SchemaDetector


class TestSchemaDetector(unittest.TestCase):
    """Test SchemaDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'expected_columns': ['timestamp', 'askPrice', 'bidPrice', 'askVolume', 'bidVolume'],
            'timestamp_unit': 'ms'
        }
        self.detector = SchemaDetector(self.config)
    
    def test_valid_schema(self):
        """Test detection of valid schema."""
        df = pd.DataFrame({
            'timestamp': [1672610641067, 1672610678111, 1672610684112],
            'askPrice': [1.07092, 1.07092, 1.07092],
            'bidPrice': [1.0697, 1.06974, 1.0697],
            'askVolume': [900.0, 900.0, 900.0],
            'bidVolume': [900.0, 900.0, 900.0]
        })
        
        is_valid, errors = self.detector.validate_schema_simple(df)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_missing_columns(self):
        """Test detection of missing columns."""
        df = pd.DataFrame({
            'timestamp': [1672610641067, 1672610678111],
            'askPrice': [1.07092, 1.07092],
            # Missing bidPrice (required)
        })
        
        is_valid, errors = self.detector.validate_schema_simple(df)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_invalid_timestamps(self):
        """Test detection of invalid timestamps."""
        df = pd.DataFrame({
            'timestamp': ['invalid', 'invalid', 'invalid'],
            'askPrice': [1.07092, 1.07092, 1.07092],
            'bidPrice': [1.0697, 1.06974, 1.0697],
            'askVolume': [900.0, 900.0, 900.0],
            'bidVolume': [900.0, 900.0, 900.0]
        })
        
        is_valid, errors = self.detector.validate_schema_simple(df)
        
        self.assertFalse(is_valid)
    
    def test_validate_and_clean(self):
        """Test validate and clean functionality."""
        df = pd.DataFrame({
            'timestamp': [1672610641067, 1672610678111, 1672610684112, 1672610685871],
            'askPrice': [1.07092, 1.07092, 1.07092, 1.07092],
            'bidPrice': [1.0697, 1.06974, 1.0697, 1.06974],
            'askVolume': [900.0, 900.0, 900.0, 900.0],
            'bidVolume': [900.0, 900.0, 900.0, 900.0]
        })
        
        cleaned = self.detector.validate_and_clean(df)
        
        # Should have datetime index
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned.index))
        
        # Should have all required columns
        for col in ['askPrice', 'bidPrice', 'askVolume', 'bidVolume']:
            self.assertIn(col, cleaned.columns)
    
    def test_negative_prices(self):
        """Test detection of negative prices."""
        df = pd.DataFrame({
            'timestamp': [1672610641067, 1672610678111, 1672610684112],
            'askPrice': [1.07092, -1.07092, 1.07092],  # Negative price
            'bidPrice': [1.0697, 1.06974, 1.0697],
            'askVolume': [900.0, 900.0, 900.0],
            'bidVolume': [900.0, 900.0, 900.0]
        })
        
        cleaned = self.detector.validate_and_clean(df)
        
        # Should remove negative prices
        self.assertEqual(len(cleaned), 2)  # One row removed
    
    def test_negative_spread(self):
        """Test detection of negative spreads (bid > ask)."""
        df = pd.DataFrame({
            'timestamp': [1672610641067, 1672610678111, 1672610684112],
            'askPrice': [1.07092, 1.06900, 1.07092],  # ask < bid on 2nd row
            'bidPrice': [1.0697, 1.07000, 1.0697],
            'askVolume': [900.0, 900.0, 900.0],
            'bidVolume': [900.0, 900.0, 900.0]
        })
        
        cleaned = self.detector.validate_and_clean(df)
        
        # Should remove negative spread rows
        self.assertLess(len(cleaned), 3)


class TestSchemaDetectorEdgeCases(unittest.TestCase):
    """Test edge cases for schema detection."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        config = {
            'expected_columns': ['timestamp', 'askPrice', 'bidPrice', 'askVolume', 'bidVolume'],
            'timestamp_unit': 'ms'
        }
        detector = SchemaDetector(config)
        
        # Empty DataFrame - need to specify dtypes for validation to work
        df = pd.DataFrame({
            'timestamp': pd.Series([], dtype='int64'),
            'askPrice': pd.Series([], dtype='float64'),
            'bidPrice': pd.Series([], dtype='float64'),
            'askVolume': pd.Series([], dtype='float64'),
            'bidVolume': pd.Series([], dtype='float64')
        })
        
        is_valid, errors = detector.validate_schema_simple(df)
        
        # Empty DataFrame with correct types should be valid
        self.assertTrue(is_valid)
    
    def test_extra_columns(self):
        """Test with extra columns (should be okay)."""
        config = {
            'expected_columns': ['timestamp', 'askPrice', 'bidPrice', 'askVolume', 'bidVolume'],
            'timestamp_unit': 'ms'
        }
        detector = SchemaDetector(config)
        
        df = pd.DataFrame({
            'timestamp': [1672610641067, 1672610678111],
            'askPrice': [1.07092, 1.07092],
            'bidPrice': [1.0697, 1.06974],
            'askVolume': [900.0, 900.0],
            'bidVolume': [900.0, 900.0],
            'extra_column': [1, 2]  # Extra column
        })
        
        is_valid, errors = detector.validate_schema_simple(df)
        
        # Should still be valid (extra columns are okay)
        self.assertTrue(is_valid)


if __name__ == '__main__':
    unittest.main()

