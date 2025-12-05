"""Unit tests for schema detection."""

import pytest
import pandas as pd
from src.data.schema_detection import SchemaDetector, SchemaValidationResult


@pytest.mark.unit
class TestSchemaDetector:
    """Test schema detection functionality."""
    
    def test_schema_detector_init(self):
        """Test SchemaDetector initialization."""
        config = {
            'schema': {}
        }
        
        detector = SchemaDetector(config)
        
        assert detector is not None
        assert detector.REQUIRED_COLUMNS == ['timestamp', 'askPrice', 'bidPrice']
    
    def test_validate_schema_valid(self):
        """Test validation with valid schema."""
        config = {'schema': {}}
        detector = SchemaDetector(config)
        
        ticks = pd.DataFrame({
            'timestamp': [1000000, 2000000, 3000000],
            'bidPrice': [1.10, 1.11, 1.12],
            'askPrice': [1.1001, 1.1101, 1.1201],
            'bidVolume': [1.0, 1.0, 1.0],
            'askVolume': [1.0, 1.0, 1.0],
        })
        
        result = detector.validate_schema(ticks, file_format='csv')
        
        assert result.is_valid == True
    
    def test_validate_schema_missing_columns(self):
        """Test validation with missing required columns."""
        config = {'schema': {}}
        detector = SchemaDetector(config)
        
        ticks = pd.DataFrame({
            'timestamp': [1000000, 2000000],
            'bidPrice': [1.10, 1.11],
            # Missing askPrice
        })
        
        result = detector.validate_schema(ticks, file_format='csv')
        
        assert result.is_valid == False
        assert len(result.missing_columns) > 0
    
    def test_validate_and_clean(self):
        """Test validate_and_clean method."""
        config = {'schema': {}}
        detector = SchemaDetector(config)
        
        ticks = pd.DataFrame({
            'timestamp': [1000000, 2000000, 3000000],
            'bidPrice': [1.10, 1.11, 1.12],
            'askPrice': [1.1001, 1.1101, 1.1201],
        })
        
        cleaned = detector.validate_and_clean(ticks)
        
        assert len(cleaned) == len(ticks)
        # validate_and_clean may return data with timestamp as index or column
        assert 'timestamp' in cleaned.columns or 'timestamp' == cleaned.index.name

