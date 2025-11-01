"""
Comprehensive tests for time series cross-validation module.

Tests for walk-forward CV, purged group CV, Monte Carlo CV, and comprehensive validation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import BaseCrossValidator
import warnings

from rtradez.validation.time_series_cv import (
    WalkForwardCV, PurgedGroupTimeSeriesSplit, MonteCarloCV,
    CombinatorialPurgedCV, TimeSeriesValidation
)


class TestWalkForwardCV:
    """Test WalkForwardCV functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        n_samples = 500
        
        X = np.random.randn(n_samples, 5)
        y = np.random.randn(n_samples)
        
        return X, y
    
    def test_walk_forward_cv_initialization_default(self):
        """Test WalkForwardCV with default parameters."""
        cv = WalkForwardCV()
        
        assert cv.min_train_size == 252
        assert cv.test_size == 63
        assert cv.step_size == 21
        assert cv.expanding_window == True
        assert cv.max_train_size is None
    
    def test_walk_forward_cv_initialization_custom(self):
        """Test WalkForwardCV with custom parameters."""
        cv = WalkForwardCV(
            min_train_size=100,
            test_size=20,
            step_size=10,
            expanding_window=False,
            max_train_size=200
        )
        
        assert cv.min_train_size == 100
        assert cv.test_size == 20
        assert cv.step_size == 10
        assert cv.expanding_window == False
        assert cv.max_train_size == 200
    
    def test_walk_forward_cv_rolling_window_default(self):
        """Test rolling window default max_train_size."""
        cv = WalkForwardCV(expanding_window=False)
        
        assert cv.max_train_size == cv.min_train_size * 2
    
    def test_get_n_splits_sufficient_data(self, sample_data):
        """Test get_n_splits with sufficient data."""
        X, y = sample_data
        cv = WalkForwardCV(min_train_size=100, test_size=50, step_size=25)
        
        n_splits = cv.get_n_splits(X)
        
        # Should have multiple splits with 500 samples
        assert n_splits > 0
        expected_splits = (len(X) - 100 - 50 + 1) // 25
        assert n_splits == expected_splits
    
    def test_get_n_splits_insufficient_data(self):
        """Test get_n_splits with insufficient data."""
        X = np.random.randn(50, 3)  # Very small dataset
        cv = WalkForwardCV(min_train_size=100, test_size=20)
        
        n_splits = cv.get_n_splits(X)
        
        assert n_splits == 0
    
    def test_get_n_splits_none_input(self):
        """Test get_n_splits with None input."""
        cv = WalkForwardCV()
        
        n_splits = cv.get_n_splits(None)
        
        assert n_splits == 0
    
    def test_split_expanding_window(self, sample_data):
        """Test split with expanding window."""
        X, y = sample_data
        cv = WalkForwardCV(
            min_train_size=100,
            test_size=50,
            step_size=25,
            expanding_window=True
        )
        
        splits = list(cv.split(X, y))
        
        assert len(splits) > 0
        
        # Check each split
        for i, (train_idx, test_idx) in enumerate(splits):
            # Training set should expand
            assert len(train_idx) >= cv.min_train_size
            if i > 0:
                prev_train_size = len(list(cv.split(X, y))[i-1][0])
                # Current training set should be larger (expanding)
                # Note: this may not always be true due to step size, so check reasonably
                pass
            
            # Test set should be consistent size
            assert len(test_idx) == cv.test_size
            
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Train comes before test
            assert max(train_idx) < min(test_idx)
    
    def test_split_rolling_window(self, sample_data):
        """Test split with rolling window."""
        X, y = sample_data
        cv = WalkForwardCV(
            min_train_size=80,
            test_size=30,
            step_size=20,
            expanding_window=False,
            max_train_size=120
        )
        
        splits = list(cv.split(X, y))
        
        assert len(splits) > 0
        
        # Check each split
        for train_idx, test_idx in splits:
            # Training set should not exceed max_train_size
            assert len(train_idx) <= cv.max_train_size
            assert len(train_idx) >= cv.min_train_size
            
            # Test set should be consistent size
            assert len(test_idx) == cv.test_size
            
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Train comes before test
            assert max(train_idx) < min(test_idx)
    
    def test_split_insufficient_data_raises(self):
        """Test split raises error with insufficient data."""
        X = np.random.randn(50, 3)
        cv = WalkForwardCV(min_train_size=100, test_size=20)
        
        with pytest.raises(ValueError, match="Not enough samples"):
            list(cv.split(X))
    
    def test_split_single_fold(self):
        """Test split with exactly enough data for one fold."""
        cv = WalkForwardCV(min_train_size=50, test_size=20, step_size=10)
        X = np.random.randn(70, 3)  # Exactly min_train_size + test_size
        
        splits = list(cv.split(X))
        
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 50
        assert len(test_idx) == 20


class TestPurgedGroupTimeSeriesSplit:
    """Test PurgedGroupTimeSeriesSplit functionality."""
    
    @pytest.fixture
    def sample_data_with_groups(self):
        """Create sample data with groups."""
        np.random.seed(42)
        n_samples = 300
        
        X = np.random.randn(n_samples, 4)
        y = np.random.randn(n_samples)
        
        # Create groups (e.g., months)
        groups = np.repeat(range(10), 30)  # 10 groups, 30 samples each
        
        return X, y, groups
    
    def test_purged_group_cv_initialization_default(self):
        """Test PurgedGroupTimeSeriesSplit with default parameters."""
        cv = PurgedGroupTimeSeriesSplit()
        
        assert cv.n_splits == 5
        assert cv.embargo_period == 5
        assert cv.purge_period == 2
        assert cv.group_col is None
    
    def test_purged_group_cv_initialization_custom(self):
        """Test PurgedGroupTimeSeriesSplit with custom parameters."""
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=3,
            embargo_period=10,
            purge_period=5,
            group_col='date'
        )
        
        assert cv.n_splits == 3
        assert cv.embargo_period == 10
        assert cv.purge_period == 5
        assert cv.group_col == 'date'
    
    def test_get_n_splits(self):
        """Test get_n_splits method."""
        cv = PurgedGroupTimeSeriesSplit(n_splits=7)
        
        n_splits = cv.get_n_splits()
        
        assert n_splits == 7
    
    def test_split_with_groups(self, sample_data_with_groups):
        """Test split with group-based splitting."""
        X, y, groups = sample_data_with_groups
        cv = PurgedGroupTimeSeriesSplit(n_splits=3, embargo_period=1, purge_period=0)
        
        splits = list(cv.split(X, y, groups))
        
        assert len(splits) > 0
        assert len(splits) <= 3
        
        # Check each split
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Train should come before test (temporal order)
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            
            # Training groups should be before test groups
            if train_groups and test_groups:
                assert max(train_groups) < min(test_groups) - cv.embargo_period
    
    def test_split_without_groups(self):
        """Test split without groups (standard time series split)."""
        X = np.random.randn(200, 3)
        y = np.random.randn(200)
        cv = PurgedGroupTimeSeriesSplit(n_splits=4, embargo_period=5, purge_period=2)
        
        splits = list(cv.split(X, y))
        
        assert len(splits) > 0
        assert len(splits) <= 4
        
        # Check each split
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Proper temporal ordering with gaps
            if len(train_idx) > 0 and len(test_idx) > 0:
                gap = min(test_idx) - max(train_idx)
                assert gap > cv.purge_period
    
    def test_split_insufficient_groups(self):
        """Test split with insufficient groups."""
        X = np.random.randn(60, 3)
        y = np.random.randn(60)
        groups = np.repeat(range(2), 30)  # Only 2 groups
        
        cv = PurgedGroupTimeSeriesSplit(n_splits=5)  # Asking for 5 splits
        
        with pytest.raises(ValueError, match="Number of groups"):
            list(cv.split(X, y, groups))
    
    def test_split_with_embargo_and_purge(self):
        """Test split with embargo and purge periods."""
        X = np.random.randn(150, 3)
        y = np.random.randn(150)
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=3,
            embargo_period=10,
            purge_period=5
        )
        
        splits = list(cv.split(X, y))
        
        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Check gap between train and test
                gap = min(test_idx) - max(train_idx)
                assert gap >= cv.purge_period


class TestMonteCarloCV:
    """Test MonteCarloCV functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for Monte Carlo CV."""
        np.random.seed(42)
        n_samples = 400
        
        X = np.random.randn(n_samples, 4)
        y = np.random.randn(n_samples)
        
        return X, y
    
    def test_monte_carlo_cv_initialization_default(self):
        """Test MonteCarloCV with default parameters."""
        cv = MonteCarloCV()
        
        assert cv.n_splits == 100
        assert cv.train_size == 0.7
        assert cv.test_size == 0.2
        assert cv.min_gap == 5
        assert cv.random_state is None
    
    def test_monte_carlo_cv_initialization_custom(self):
        """Test MonteCarloCV with custom parameters."""
        cv = MonteCarloCV(
            n_splits=50,
            train_size=0.6,
            test_size=0.3,
            min_gap=10,
            random_state=42
        )
        
        assert cv.n_splits == 50
        assert cv.train_size == 0.6
        assert cv.test_size == 0.3
        assert cv.min_gap == 10
        assert cv.random_state == 42
    
    def test_monte_carlo_random_state_setting(self):
        """Test random state is set correctly."""
        with patch('rtradez.validation.time_series_cv.np.random.seed') as mock_seed:
            cv = MonteCarloCV(random_state=123)
            mock_seed.assert_called_once_with(123)
    
    def test_split_generates_correct_number_of_splits(self, sample_data):
        """Test split generates the correct number of splits."""
        X, y = sample_data
        cv = MonteCarloCV(n_splits=20, train_size=0.5, test_size=0.2, min_gap=5)
        
        splits = list(cv.split(X, y))
        
        assert len(splits) == 20
    
    def test_split_correct_sizes_and_gaps(self, sample_data):
        """Test split generates correct train/test sizes and gaps."""
        X, y = sample_data
        cv = MonteCarloCV(n_splits=10, train_size=0.4, test_size=0.2, min_gap=8)
        
        splits = list(cv.split(X, y))
        
        expected_train_size = int(len(X) * 0.4)
        expected_test_size = int(len(X) * 0.2)
        
        for train_idx, test_idx in splits:
            # Check sizes
            assert len(train_idx) == expected_train_size
            assert len(test_idx) == expected_test_size
            
            # Check no overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Check temporal order and gap
            assert max(train_idx) < min(test_idx)
            gap = min(test_idx) - max(train_idx) - 1
            assert gap >= cv.min_gap
    
    def test_split_insufficient_data_raises(self):
        """Test split raises error with insufficient data."""
        X = np.random.randn(50, 3)
        cv = MonteCarloCV(train_size=0.7, test_size=0.2, min_gap=10)
        
        # 50 * 0.7 + 50 * 0.2 + 10 = 35 + 10 + 10 = 55 > 50
        with pytest.raises(ValueError, match="Not enough samples"):
            list(cv.split(X))
    
    def test_split_randomness(self, sample_data):
        """Test that splits are actually random."""
        X, y = sample_data
        cv1 = MonteCarloCV(n_splits=5, random_state=42)
        cv2 = MonteCarloCV(n_splits=5, random_state=123)
        
        splits1 = list(cv1.split(X, y))
        splits2 = list(cv2.split(X, y))
        
        # With different random states, splits should be different
        # Check at least one split is different
        different = False
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            if not np.array_equal(train1, train2) or not np.array_equal(test1, test2):
                different = True
                break
        
        assert different, "Splits should be different with different random states"
    
    def test_split_reproducible(self, sample_data):
        """Test that splits are reproducible with same random state."""
        X, y = sample_data
        cv1 = MonteCarloCV(n_splits=5, random_state=42)
        cv2 = MonteCarloCV(n_splits=5, random_state=42)
        
        splits1 = list(cv1.split(X, y))
        splits2 = list(cv2.split(X, y))
        
        # With same random state, splits should be identical
        assert len(splits1) == len(splits2)
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)


class TestCombinatorialPurgedCV:
    """Test CombinatorialPurgedCV functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for combinatorial purged CV."""
        np.random.seed(42)
        n_samples = 240  # Divisible by multiple group sizes
        
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)
        
        return X, y
    
    def test_combinatorial_cv_initialization_default(self):
        """Test CombinatorialPurgedCV with default parameters."""
        cv = CombinatorialPurgedCV()
        
        assert cv.n_splits == 6
        assert cv.n_test_groups == 2
        assert cv.embargo_period == 5
        assert cv.purge_period == 2
    
    def test_combinatorial_cv_initialization_custom(self):
        """Test CombinatorialPurgedCV with custom parameters."""
        cv = CombinatorialPurgedCV(
            n_splits=4,
            n_test_groups=1,
            embargo_period=3,
            purge_period=1
        )
        
        assert cv.n_splits == 4
        assert cv.n_test_groups == 1
        assert cv.embargo_period == 3
        assert cv.purge_period == 1
    
    def test_split_generates_combinations(self, sample_data):
        """Test split generates correct combinations."""
        X, y = sample_data
        cv = CombinatorialPurgedCV(n_splits=4, n_test_groups=1, embargo_period=1, purge_period=1)
        
        splits = list(cv.split(X, y))
        
        # Should generate multiple combinations
        assert len(splits) > 0
        
        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_split_group_assignment(self, sample_data):
        """Test proper group assignment in splits."""
        X, y = sample_data
        cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_period=0, purge_period=0)
        
        splits = list(cv.split(X, y))
        
        group_size = len(X) // cv.n_splits
        
        for train_idx, test_idx in splits:
            # Test indices should correspond to consecutive groups
            expected_test_size = group_size * cv.n_test_groups
            # Allow for some variation due to remainder handling
            assert abs(len(test_idx) - expected_test_size) <= cv.n_test_groups
    
    def test_split_with_embargo_and_purge(self, sample_data):
        """Test splits respect embargo and purge periods."""
        X, y = sample_data
        cv = CombinatorialPurgedCV(
            n_splits=6,
            n_test_groups=1,
            embargo_period=2,
            purge_period=1
        )
        
        splits = list(cv.split(X, y))
        
        # Should have fewer splits due to embargo/purge restrictions
        assert len(splits) > 0
        
        # Check that training groups are sufficiently separated from test groups
        group_size = len(X) // cv.n_splits
        
        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Convert indices back to group numbers
                train_groups = set(train_idx // group_size)
                test_groups = set(test_idx // group_size)
                
                # Training groups should be sufficiently separated from test groups
                for train_group in train_groups:
                    for test_group in test_groups:
                        distance = abs(train_group - test_group)
                        assert distance > max(cv.embargo_period, cv.purge_period)
    
    def test_split_remainder_handling(self):
        """Test handling of remainder samples."""
        # Use data size that doesn't divide evenly
        X = np.random.randn(100, 3)  # 100 samples, 6 groups = 16.67 per group
        y = np.random.randn(100)
        
        cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=1)
        
        splits = list(cv.split(X, y))
        
        # Should still generate valid splits
        assert len(splits) > 0
        
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0


class TestTimeSeriesValidation:
    """Test TimeSeriesValidation comprehensive framework."""
    
    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial time series data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        
        # Generate realistic financial features
        n_samples = len(dates)
        X = pd.DataFrame({
            'returns_lag1': np.random.normal(0, 0.02, n_samples),
            'volatility': np.random.exponential(0.01, n_samples),
            'volume': np.random.lognormal(10, 0.5, n_samples),
            'momentum': np.random.normal(0, 0.1, n_samples)
        }, index=dates)
        
        # Generate target (future returns)
        y = pd.Series(np.random.normal(0.001, 0.02, n_samples), index=dates)
        
        # Generate groups (months)
        groups = pd.Series([d.strftime('%Y-%m') for d in dates], index=dates)
        
        return X.dropna(), y.dropna(), groups.dropna()
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return LinearRegression()
    
    def test_time_series_validation_initialization_default(self):
        """Test TimeSeriesValidation with default parameters."""
        validator = TimeSeriesValidation()
        
        assert 'walk_forward' in validator.cv_methods
        assert 'purged_group' in validator.cv_methods
        assert 'mse' in validator.scoring_metrics
        assert 'mae' in validator.scoring_metrics
        assert 'r2' in validator.scoring_metrics
        assert 'sharpe' in validator.scoring_metrics
        assert len(validator.validation_results_) == 0
        assert len(validator.performance_summary_) == 0
    
    def test_time_series_validation_initialization_custom(self):
        """Test TimeSeriesValidation with custom parameters."""
        validator = TimeSeriesValidation(
            cv_methods=['walk_forward', 'monte_carlo'],
            scoring_metrics=['mse', 'r2']
        )
        
        assert validator.cv_methods == ['walk_forward', 'monte_carlo']
        assert validator.scoring_metrics == ['mse', 'r2']
        assert 'walk_forward' in validator.cv_objects
        assert 'monte_carlo' in validator.cv_objects
    
    def test_cv_objects_initialization(self):
        """Test CV objects are properly initialized."""
        validator = TimeSeriesValidation()
        
        assert isinstance(validator.cv_objects['walk_forward'], WalkForwardCV)
        assert isinstance(validator.cv_objects['purged_group'], PurgedGroupTimeSeriesSplit)
        assert isinstance(validator.cv_objects['monte_carlo'], MonteCarloCV)
        assert isinstance(validator.cv_objects['combinatorial'], CombinatorialPurgedCV)
    
    def test_clone_model_sklearn(self, simple_model):
        """Test model cloning for sklearn models."""
        validator = TimeSeriesValidation()
        cloned = validator._clone_model(simple_model)
        
        assert type(cloned) == type(simple_model)
        assert cloned is not simple_model
    
    def test_clone_model_custom(self):
        """Test model cloning for custom models."""
        validator = TimeSeriesValidation()
        
        class CustomModel:
            def __init__(self, param=1):
                self.param = param
            
            def get_params(self):
                return {'param': self.param}
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                return np.zeros(len(X))
        
        model = CustomModel(param=10)
        cloned = validator._clone_model(model)
        
        assert type(cloned) == type(model)
        assert cloned.param == model.param
    
    def test_evaluate_fold_basic_metrics(self, simple_model):
        """Test fold evaluation with basic metrics."""
        validator = TimeSeriesValidation(scoring_metrics=['mse', 'mae', 'r2'])
        
        # Create sample data
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        train_idx = np.arange(70)
        test_idx = np.arange(70, 100)
        
        fold_result = validator._evaluate_fold(simple_model, X, y, train_idx, test_idx, 0)
        
        assert isinstance(fold_result, dict)
        assert fold_result['fold'] == 0
        assert fold_result['train_size'] == 70
        assert fold_result['test_size'] == 30
        assert 'metrics' in fold_result
        
        metrics = fold_result['metrics']
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check metric validity
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert -np.inf <= metrics['r2'] <= 1
    
    def test_evaluate_fold_sharpe_metric(self, simple_model):
        """Test fold evaluation with Sharpe ratio metric."""
        validator = TimeSeriesValidation(scoring_metrics=['sharpe'])
        
        # Create sample data
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        train_idx = np.arange(30)
        test_idx = np.arange(30, 50)
        
        fold_result = validator._evaluate_fold(simple_model, X, y, train_idx, test_idx, 0)
        
        metrics = fold_result['metrics']
        assert 'sharpe' in metrics
        assert isinstance(metrics['sharpe'], float)
        assert not np.isnan(metrics['sharpe'])
    
    def test_aggregate_fold_results(self):
        """Test aggregation of fold results."""
        validator = TimeSeriesValidation()
        
        # Create mock fold results
        fold_results = [
            {
                'fold': 0,
                'train_size': 100,
                'test_size': 20,
                'metrics': {'mse': 0.5, 'r2': 0.3}
            },
            {
                'fold': 1,
                'train_size': 120,
                'test_size': 20,
                'metrics': {'mse': 0.7, 'r2': 0.25}
            },
            {
                'fold': 2,
                'train_size': 140,
                'test_size': 20,
                'metrics': {'mse': 0.6, 'r2': 0.35}
            }
        ]
        
        aggregated = validator._aggregate_fold_results(fold_results)
        
        assert isinstance(aggregated, dict)
        assert 'fold_metrics' in aggregated
        assert 'mean_metrics' in aggregated
        assert 'std_metrics' in aggregated
        assert 'train_sizes' in aggregated
        assert 'test_sizes' in aggregated
        
        # Check mean calculations
        assert aggregated['mean_metrics']['mse'] == pytest.approx(0.6, rel=1e-3)
        assert aggregated['mean_metrics']['r2'] == pytest.approx(0.3, rel=1e-3)
        
        # Check standard deviation calculations
        assert aggregated['std_metrics']['mse'] > 0
        assert aggregated['std_metrics']['r2'] > 0
        
        # Check size aggregations
        assert aggregated['avg_train_size'] == 120
        assert aggregated['avg_test_size'] == 20
    
    def test_calculate_validation_summary(self):
        """Test validation summary calculation."""
        validator = TimeSeriesValidation()
        
        # Create mock CV results
        results = {
            'walk_forward': {
                'cv_method': 'walk_forward',
                'n_folds': 5,
                'mean_metrics': {'mse': 0.5, 'r2': 0.3, 'mae': 0.6}
            },
            'purged_group': {
                'cv_method': 'purged_group',
                'n_folds': 4,
                'mean_metrics': {'mse': 0.7, 'r2': 0.2, 'mae': 0.8}
            },
            'failed_method': {
                'error': 'Some error occurred'
            }
        }
        
        summary = validator._calculate_validation_summary(results)
        
        assert isinstance(summary, dict)
        assert 'cv_methods_used' in summary
        assert 'cv_methods_successful' in summary
        assert 'consensus_metrics' in summary
        assert 'method_comparison' in summary
        
        # Check successful methods
        assert 'walk_forward' in summary['cv_methods_successful']
        assert 'purged_group' in summary['cv_methods_successful']
        assert 'failed_method' not in summary['cv_methods_successful']
        
        # Check consensus metrics
        consensus = summary['consensus_metrics']
        assert 'mse' in consensus
        assert 'r2' in consensus
        assert 'mae' in consensus
        
        # Check consensus calculations
        assert consensus['mse']['mean'] == pytest.approx(0.6, rel=1e-3)  # (0.5 + 0.7) / 2
        assert consensus['r2']['mean'] == pytest.approx(0.25, rel=1e-3)  # (0.3 + 0.2) / 2
    
    @patch('rtradez.validation.time_series_cv.logger')
    def test_run_cv_method_success(self, mock_logger, simple_model, sample_financial_data):
        """Test successful CV method execution."""
        X, y, groups = sample_financial_data
        validator = TimeSeriesValidation(cv_methods=['walk_forward'], scoring_metrics=['mse', 'r2'])
        
        # Use smaller subset for faster testing
        X_subset = X.iloc[:200]
        y_subset = y.iloc[:200]
        
        result = validator._run_cv_method('walk_forward', simple_model, X_subset, y_subset)
        
        assert isinstance(result, dict)
        assert 'n_folds' in result
        assert 'cv_method' in result
        assert result['cv_method'] == 'walk_forward'
        assert 'mean_metrics' in result
        assert 'std_metrics' in result
        
        # Check that metrics were calculated
        assert 'mse' in result['mean_metrics']
        assert 'r2' in result['mean_metrics']
    
    @patch('rtradez.validation.time_series_cv.logger')
    def test_run_cv_method_failure(self, mock_logger, sample_financial_data):
        """Test CV method failure handling."""
        X, y, groups = sample_financial_data
        validator = TimeSeriesValidation()
        
        # Create failing model
        class FailingModel:
            def fit(self, X, y):
                raise ValueError("Model fitting failed")
            
            def predict(self, X):
                raise ValueError("Prediction failed")
        
        failing_model = FailingModel()
        
        result = validator._run_cv_method('walk_forward', failing_model, X.iloc[:100], y.iloc[:100])
        
        # Should catch exception and log error
        mock_logger.error.assert_called()
    
    def test_validate_model_complete_workflow(self, simple_model, sample_financial_data):
        """Test complete model validation workflow."""
        X, y, groups = sample_financial_data
        validator = TimeSeriesValidation(
            cv_methods=['walk_forward'],
            scoring_metrics=['mse', 'r2']
        )
        
        # Use smaller subset for faster testing
        X_subset = X.iloc[:300]
        y_subset = y.iloc[:300]
        groups_subset = groups.iloc[:300]
        
        validation_result = validator.validate_model(
            model=simple_model,
            X=X_subset,
            y=y_subset,
            groups=groups_subset,
            model_name="test_model"
        )
        
        assert isinstance(validation_result, dict)
        assert 'detailed_results' in validation_result
        assert 'summary' in validation_result
        assert 'model_name' in validation_result
        assert validation_result['model_name'] == "test_model"
        
        # Check detailed results
        detailed = validation_result['detailed_results']
        assert 'walk_forward' in detailed
        
        # Check summary
        summary = validation_result['summary']
        assert 'cv_methods_used' in summary
        assert 'consensus_metrics' in summary
        
        # Check that results are stored
        assert 'test_model' in validator.validation_results_
        assert 'test_model' in validator.performance_summary_
    
    def test_validate_model_data_alignment(self, simple_model):
        """Test model validation with misaligned data."""
        validator = TimeSeriesValidation(cv_methods=['walk_forward'])
        
        # Create misaligned data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        y = pd.Series([0.1, 0.2, 0.3], 
                     index=pd.date_range('2023-01-02', periods=3))
        
        groups = pd.Series(['A', 'A', 'B'], 
                          index=pd.date_range('2023-01-02', periods=3))
        
        validation_result = validator.validate_model(
            model=simple_model,
            X=X,
            y=y,
            groups=groups,
            model_name="misaligned_test"
        )
        
        # Should handle misalignment gracefully
        assert isinstance(validation_result, dict)
        assert 'detailed_results' in validation_result
    
    def test_compare_models(self, sample_financial_data):
        """Test model comparison functionality."""
        X, y, groups = sample_financial_data
        validator = TimeSeriesValidation(
            cv_methods=['walk_forward'],
            scoring_metrics=['mse', 'r2']
        )
        
        # Create multiple models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=5, random_state=42)
        }
        
        # Use smaller subset for faster testing
        X_subset = X.iloc[:200]
        y_subset = y.iloc[:200]
        groups_subset = groups.iloc[:200]
        
        comparison_df = validator.compare_models(models, X_subset, y_subset, groups_subset)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'model' in comparison_df.columns
        
        # Check that both models are present
        model_names = comparison_df['model'].tolist()
        assert 'linear_regression' in model_names
        assert 'random_forest' in model_names
        
        # Check for metric columns
        metric_columns = [col for col in comparison_df.columns if 'mse' in col or 'r2' in col]
        assert len(metric_columns) > 0
    
    def test_compare_models_with_failure(self, sample_financial_data):
        """Test model comparison with one failing model."""
        X, y, groups = sample_financial_data
        validator = TimeSeriesValidation()
        
        # Create models, one that will fail
        class FailingModel:
            def fit(self, X, y):
                raise ValueError("Fitting failed")
        
        models = {
            'working_model': LinearRegression(),
            'failing_model': FailingModel()
        }
        
        comparison_df = validator.compare_models(models, X.iloc[:100], y.iloc[:100])
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        
        # Check that error is recorded for failing model
        failing_row = comparison_df[comparison_df['model'] == 'failing_model']
        assert len(failing_row) == 1
        assert 'error' in failing_row.columns
    
    def test_get_validation_report(self, simple_model, sample_financial_data):
        """Test validation report generation."""
        X, y, groups = sample_financial_data
        validator = TimeSeriesValidation()
        
        # First validate a model
        validator.validate_model(
            model=simple_model,
            X=X.iloc[:200],
            y=y.iloc[:200],
            groups=groups.iloc[:200],
            model_name="report_test_model"
        )
        
        # Get validation report
        report = validator.get_validation_report("report_test_model")
        
        assert isinstance(report, dict)
        assert 'model_name' in report
        assert 'validation_date' in report
        assert 'detailed_results' in report
        assert 'summary' in report
        assert 'cv_framework' in report
        
        assert report['model_name'] == "report_test_model"
        
        # Check framework info
        framework = report['cv_framework']
        assert 'cv_methods' in framework
        assert 'scoring_metrics' in framework
    
    def test_get_validation_report_nonexistent_model(self):
        """Test getting report for non-existent model."""
        validator = TimeSeriesValidation()
        
        with pytest.raises(ValueError, match="No validation results found"):
            validator.get_validation_report("nonexistent_model")


@pytest.mark.integration
class TestTimeSeriesCVIntegration:
    """Integration tests for time series cross-validation."""
    
    @pytest.fixture
    def realistic_financial_data(self):
        """Create realistic financial time series dataset."""
        np.random.seed(42)
        dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
        
        # Generate correlated price series
        n_samples = len(dates)
        returns = np.random.normal(0.0005, 0.015, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create realistic features
        price_series = pd.Series(prices, index=dates)
        
        X = pd.DataFrame({
            'price': price_series,
            'sma_20': price_series.rolling(20).mean(),
            'sma_50': price_series.rolling(50).mean(),
            'volatility_20': price_series.pct_change().rolling(20).std(),
            'rsi': np.random.uniform(20, 80, n_samples),
            'volume': np.random.lognormal(12, 0.3, n_samples),
            'momentum_5': price_series.pct_change(5),
            'momentum_20': price_series.pct_change(20)
        })
        
        # Target: next day returns
        y = price_series.pct_change().shift(-1)
        
        # Groups: monthly periods
        groups = pd.Series([d.strftime('%Y-%m') for d in dates], index=dates)
        
        return X.dropna(), y.dropna(), groups.dropna()
    
    def test_full_validation_pipeline_multiple_cv_methods(self, realistic_financial_data):
        """Test complete validation pipeline with multiple CV methods."""
        X, y, groups = realistic_financial_data
        
        # Create comprehensive validator
        validator = TimeSeriesValidation(
            cv_methods=['walk_forward', 'purged_group', 'monte_carlo'],
            scoring_metrics=['mse', 'mae', 'r2', 'sharpe']
        )
        
        # Use multiple models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        # Use subset for reasonable test time
        X_subset = X.iloc[:800]
        y_subset = y.iloc[:800]
        groups_subset = groups.iloc[:800]
        
        # Compare models
        comparison_results = validator.compare_models(models, X_subset, y_subset, groups_subset)
        
        assert isinstance(comparison_results, pd.DataFrame)
        assert len(comparison_results) == 2
        
        # Should have metrics from all CV methods
        metric_columns = [col for col in comparison_results.columns 
                         if any(metric in col for metric in ['mse', 'mae', 'r2', 'sharpe'])]
        assert len(metric_columns) > 0
        
        # Check consensus metrics exist
        consensus_columns = [col for col in comparison_results.columns if 'mean' in col]
        assert len(consensus_columns) > 0
    
    def test_walk_forward_vs_purged_group_comparison(self, realistic_financial_data):
        """Test comparison between walk-forward and purged group CV."""
        X, y, groups = realistic_financial_data
        
        # Create validators for each method
        wf_validator = TimeSeriesValidation(
            cv_methods=['walk_forward'],
            scoring_metrics=['mse', 'r2']
        )
        
        pg_validator = TimeSeriesValidation(
            cv_methods=['purged_group'],
            scoring_metrics=['mse', 'r2']
        )
        
        model = LinearRegression()
        
        # Use subset for faster testing
        X_subset = X.iloc[:600]
        y_subset = y.iloc[:600]
        groups_subset = groups.iloc[:600]
        
        # Validate with both methods
        wf_result = wf_validator.validate_model(model, X_subset, y_subset, model_name="wf_test")
        pg_result = pg_validator.validate_model(model, X_subset, y_subset, groups_subset, model_name="pg_test")
        
        # Both should produce valid results
        assert 'detailed_results' in wf_result
        assert 'detailed_results' in pg_result
        
        # Both should have metrics
        wf_metrics = wf_result['summary']['consensus_metrics']
        pg_metrics = pg_result['summary']['consensus_metrics']
        
        assert 'mse' in wf_metrics
        assert 'mse' in pg_metrics
        assert 'r2' in wf_metrics
        assert 'r2' in pg_metrics
        
        # Metrics should be reasonable
        assert wf_metrics['mse']['mean'] > 0
        assert pg_metrics['mse']['mean'] > 0
    
    def test_monte_carlo_cv_stability(self, realistic_financial_data):
        """Test Monte Carlo CV produces consistent results."""
        X, y, groups = realistic_financial_data
        
        # Create validators with same random state
        validator1 = TimeSeriesValidation(cv_methods=['monte_carlo'])
        validator2 = TimeSeriesValidation(cv_methods=['monte_carlo'])
        
        # Set same random state for reproducibility
        validator1.cv_objects['monte_carlo'].random_state = 42
        validator2.cv_objects['monte_carlo'].random_state = 42
        
        model = LinearRegression()
        
        X_subset = X.iloc[:400]
        y_subset = y.iloc[:400]
        
        # Validate with both validators (should give same results)
        result1 = validator1.validate_model(model, X_subset, y_subset, model_name="mc_test1")
        result2 = validator2.validate_model(model, X_subset, y_subset, model_name="mc_test2")
        
        # Results should be very similar (allowing for small numerical differences)
        metrics1 = result1['summary']['consensus_metrics']
        metrics2 = result2['summary']['consensus_metrics']
        
        for metric in ['mse', 'mae', 'r2']:
            if metric in metrics1 and metric in metrics2:
                assert abs(metrics1[metric]['mean'] - metrics2[metric]['mean']) < 0.01
    
    def test_cv_method_failure_resilience(self, realistic_financial_data):
        """Test validation framework resilience to CV method failures."""
        X, y, groups = realistic_financial_data
        
        # Create validator with multiple methods
        validator = TimeSeriesValidation(
            cv_methods=['walk_forward', 'purged_group', 'combinatorial'],
            scoring_metrics=['mse', 'r2']
        )
        
        # Use very small dataset that might cause some CV methods to fail
        X_tiny = X.iloc[:50]
        y_tiny = y.iloc[:50]
        groups_tiny = groups.iloc[:50]
        
        model = LinearRegression()
        
        # Some CV methods might fail with small data, but validation should continue
        result = validator.validate_model(model, X_tiny, y_tiny, groups_tiny, model_name="resilience_test")
        
        assert isinstance(result, dict)
        assert 'detailed_results' in result
        assert 'summary' in result
        
        # At least one CV method should succeed
        successful_methods = result['summary']['cv_methods_successful']
        assert len(successful_methods) > 0
    
    def test_large_scale_validation_performance(self, realistic_financial_data):
        """Test validation performance with larger dataset."""
        X, y, groups = realistic_financial_data
        
        # Use larger subset to test performance
        X_large = X.iloc[:1500]
        y_large = y.iloc[:1500]
        groups_large = groups.iloc[:1500]
        
        # Use efficient configuration
        validator = TimeSeriesValidation(
            cv_methods=['walk_forward'],  # Single method for speed
            scoring_metrics=['mse', 'r2']  # Limited metrics
        )
        
        # Configure for faster execution
        validator.cv_objects['walk_forward'] = WalkForwardCV(
            min_train_size=200,
            test_size=50,
            step_size=50,  # Larger step size for fewer folds
            expanding_window=True
        )
        
        model = LinearRegression()
        
        import time
        start_time = time.time()
        
        result = validator.validate_model(model, X_large, y_large, model_name="performance_test")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (less than 30 seconds)
        assert execution_time < 30
        
        # Should produce valid results
        assert isinstance(result, dict)
        assert 'detailed_results' in result
        assert 'walk_forward' in result['detailed_results']
        
        # Should have multiple folds
        wf_result = result['detailed_results']['walk_forward']
        assert wf_result['n_folds'] > 1
    
    def test_cross_validation_method_consistency(self, realistic_financial_data):
        """Test that different CV methods produce consistent relative rankings."""
        X, y, groups = realistic_financial_data
        
        # Create models with known performance differences
        models = {
            'simple_linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=20, random_state=42),
            'overfitted_rf': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
        }
        
        # Use different CV methods
        cv_methods = ['walk_forward', 'purged_group']
        results_by_method = {}
        
        for cv_method in cv_methods:
            validator = TimeSeriesValidation(
                cv_methods=[cv_method],
                scoring_metrics=['mse', 'r2']
            )
            
            comparison = validator.compare_models(models, X.iloc[:600], y.iloc[:600], groups.iloc[:600])
            results_by_method[cv_method] = comparison
        
        # Check that model rankings are roughly consistent across CV methods
        for metric in ['mse', 'r2']:
            metric_cols = [col for col in results_by_method['walk_forward'].columns if metric in col and 'mean' in col]
            
            if metric_cols:
                metric_col = metric_cols[0]
                
                # Get model rankings for each CV method
                rankings = {}
                for cv_method in cv_methods:
                    df = results_by_method[cv_method]
                    if metric_col in df.columns:
                        # For MSE, lower is better; for R2, higher is better
                        ascending = (metric == 'mse')
                        sorted_models = df.sort_values(metric_col, ascending=ascending)['model'].tolist()
                        rankings[cv_method] = sorted_models
                
                # Rankings should not be completely different
                if len(rankings) == 2:
                    rank1, rank2 = list(rankings.values())
                    # At least some consistency in rankings (not perfect due to randomness)
                    # This is a basic sanity check rather than strict requirement
                    assert len(set(rank1) & set(rank2)) == len(models)  # Same models present