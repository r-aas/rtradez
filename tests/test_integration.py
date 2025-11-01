"""Integration tests for RTradez workflows."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from rtradez.data_sources import RTradezDataManager
from rtradez.utils.dataset_combiner import DatasetCombiner
from rtradez.utils.temporal_alignment import TemporalAligner, FrequencyType
from rtradez.validation.time_series_cv import TimeSeriesValidation
from rtradez.methods.strategies import OptionsStrategy


class TestIntegratedWorkflows:
    """Test integrated RTradez workflows."""
    
    def test_data_integration_workflow(self, sample_datasets):
        """Test complete data integration workflow."""
        # Test temporal alignment
        aligner = TemporalAligner(
            target_frequency=FrequencyType.DAILY,
            alignment_method='outer'
        )
        
        aligned_datasets = aligner.align_datasets(sample_datasets)
        assert len(aligned_datasets) > 0
        
        # Test dataset combination
        combiner = DatasetCombiner()
        combined_data = combiner.combine_datasets(aligned_datasets)
        assert not combined_data.empty
        
    def test_strategy_validation_workflow(self, sample_data):
        """Test strategy training and validation workflow."""
        X, y = sample_data['X'], sample_data['y']
        
        # Create and train strategy
        strategy = OptionsStrategy(strategy_type='iron_condor')
        strategy.fit(X, y)
        
        # Validate with time series CV
        validator = TimeSeriesValidation()
        results = validator.validate_model(strategy, X, y)
        
        assert 'detailed_results' in results
        assert 'summary' in results
        
    @patch('requests.get')
    def test_end_to_end_pipeline(self, mock_get, sample_data, mock_api_responses):
        """Test end-to-end pipeline from data fetch to validation."""
        # Mock API responses
        mock_get.return_value.json.return_value = mock_api_responses['fred']
        mock_get.return_value.raise_for_status.return_value = None
        
        # Test data fetching (mocked)
        data_manager = RTradezDataManager()
        
        # Test strategy pipeline
        X, y = sample_data['X'], sample_data['y']
        strategy = OptionsStrategy(strategy_type='iron_condor')
        
        # Fit and predict
        strategy.fit(X, y)
        predictions = strategy.predict(X)
        score = strategy.score(X, y)
        
        assert predictions is not None
        assert isinstance(score, (int, float))
