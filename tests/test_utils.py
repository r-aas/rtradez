"""Tests for utility modules."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from rtradez.utils.caching import RTradezCache, cached
from rtradez.utils.experiments import RTradezExperimentTracker
from rtradez.utils.optimization import RTradezOptimizer


@pytest.mark.unit
class TestRTradezCache:
    """Test RTradezCache functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.cache = RTradezCache(cache_dir=self.test_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        assert self.cache.cache_dir == self.test_dir
        assert 'market_data' in self.cache.caches
        assert 'features' in self.cache.caches
        assert 'backtests' in self.cache.caches
        assert 'models' in self.cache.caches
    
    def test_cache_set_get(self):
        """Test basic cache set and get operations."""
        test_data = {'key': 'value', 'number': 42}
        
        # Set data
        self.cache.set('market_data', 'test_key', test_data)
        
        # Get data
        retrieved_data = self.cache.get('market_data', 'test_key')
        
        assert retrieved_data == test_data
    
    def test_cache_miss_returns_none(self):
        """Test cache miss returns None."""
        result = self.cache.get('market_data', 'nonexistent_key')
        assert result is None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        # Set some data
        self.cache.set('market_data', 'test_key', {'data': 'test'})
        assert self.cache.get('market_data', 'test_key') is not None
        
        # Clear cache
        self.cache.clear('market_data')
        assert self.cache.get('market_data', 'test_key') is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Add some data
        self.cache.set('market_data', 'key1', {'data': 1})
        self.cache.set('market_data', 'key2', {'data': 2})
        
        stats = self.cache.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_size_mb' in stats
        assert 'cache_counts' in stats
        assert stats['cache_counts']['market_data'] >= 2
    
    def test_cached_decorator(self):
        """Test @cached decorator functionality."""
        call_count = 0
        
        @cached(cache_type='market_data', expire=3600)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should compute result
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different arguments should compute again
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2


@pytest.mark.unit
class TestRTradezExperimentTracker:
    """Test RTradezExperimentTracker functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.tracker = RTradezExperimentTracker(experiment_name="test_experiment")
    
    @patch('rtradez.utils.experiments.mlflow')
    def test_start_experiment(self, mock_mlflow):
        """Test experiment start."""
        mock_mlflow.set_experiment.return_value = None
        mock_mlflow.start_run.return_value = MagicMock()
        
        self.tracker.start_experiment()
        
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once()
    
    @patch('rtradez.utils.experiments.mlflow')
    def test_log_strategy_config(self, mock_mlflow):
        """Test strategy configuration logging."""
        strategy_type = "iron_condor"
        parameters = {"profit_target": 0.35, "stop_loss": 2.0}
        
        self.tracker.log_strategy_config(strategy_type, parameters)
        
        # Check that parameters were logged
        mock_mlflow.log_param.assert_any_call("strategy_type", strategy_type)
        mock_mlflow.log_param.assert_any_call("strategy.profit_target", 0.35)
        mock_mlflow.log_param.assert_any_call("strategy.stop_loss", 2.0)
    
    @patch('rtradez.utils.experiments.mlflow')
    def test_log_performance_metrics(self, mock_mlflow):
        """Test performance metrics logging."""
        metrics = {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.15,
            "total_return": 0.12
        }
        
        self.tracker.log_performance_metrics(metrics)
        
        # Check that metrics were logged
        mock_mlflow.log_metric.assert_any_call("sharpe_ratio", 1.5)
        mock_mlflow.log_metric.assert_any_call("max_drawdown", -0.15)
        mock_mlflow.log_metric.assert_any_call("total_return", 0.12)
    
    @patch('rtradez.utils.experiments.mlflow')
    def test_end_experiment(self, mock_mlflow):
        """Test experiment end."""
        self.tracker.end_experiment()
        
        mock_mlflow.end_run.assert_called_once()


@pytest.mark.unit
class TestRTradezOptimizer:
    """Test RTradezOptimizer functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimizer = RTradezOptimizer(study_name="test_study")
    
    def test_optimizer_initialization(self):
        """Test optimizer initializes correctly."""
        assert self.optimizer.study_name == "test_study"
        assert hasattr(self.optimizer, 'sampler')
        assert hasattr(self.optimizer, 'pruner')
    
    @patch('rtradez.utils.optimization.optuna')
    def test_optimize_strategy(self, mock_optuna):
        """Test strategy optimization."""
        # Mock optuna study
        mock_study = MagicMock()
        mock_study.best_params = {"profit_target": 0.35, "stop_loss": 2.0}
        mock_study.best_value = 1.5
        mock_optuna.create_study.return_value = mock_study
        
        # Define objective function
        def objective(trial):
            profit_target = trial.suggest_float('profit_target', 0.2, 0.5)
            stop_loss = trial.suggest_float('stop_loss', 1.0, 3.0)
            return profit_target * stop_loss  # Dummy objective
        
        # Define parameter space
        param_space = {
            'profit_target': {'type': 'float', 'low': 0.2, 'high': 0.5},
            'stop_loss': {'type': 'float', 'low': 1.0, 'high': 3.0}
        }
        
        # Run optimization
        result = self.optimizer.optimize_strategy(objective, param_space, n_trials=10)
        
        # Verify study was created and optimized
        mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()
        assert result == mock_study
    
    def test_suggest_parameters(self):
        """Test parameter suggestion functionality."""
        # Mock trial object
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.35
        mock_trial.suggest_int.return_value = 30
        mock_trial.suggest_categorical.return_value = 'iron_condor'
        
        param_space = {
            'profit_target': {'type': 'float', 'low': 0.2, 'high': 0.5},
            'dte': {'type': 'int', 'low': 20, 'high': 40},
            'strategy': {'type': 'categorical', 'choices': ['iron_condor', 'strangle']}
        }
        
        params = self.optimizer._suggest_parameters(mock_trial, param_space)
        
        assert 'profit_target' in params
        assert 'dte' in params
        assert 'strategy' in params
        mock_trial.suggest_float.assert_called_with('profit_target', 0.2, 0.5)
        mock_trial.suggest_int.assert_called_with('dte', 20, 40)
        mock_trial.suggest_categorical.assert_called_with('strategy', ['iron_condor', 'strangle'])


@pytest.mark.integration
class TestCachingIntegration:
    """Integration tests for caching functionality."""
    
    def test_cache_with_real_data(self, sample_features_and_returns):
        """Test caching with real data structures."""
        X, y = sample_features_and_returns
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RTradezCache(cache_dir=temp_dir)
            
            # Cache DataFrame
            cache.set('features', 'test_features', X)
            cached_X = cache.get('features', 'test_features')
            
            # Verify data integrity
            pd.testing.assert_frame_equal(X, cached_X)
            
            # Cache Series
            cache.set('features', 'test_returns', y)
            cached_y = cache.get('features', 'test_returns')
            
            # Verify data integrity
            pd.testing.assert_series_equal(y, cached_y)
    
    def test_cache_performance_improvement(self):
        """Test that caching actually improves performance."""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RTradezCache(cache_dir=temp_dir)
            
            @cached(cache_type='market_data', expire=3600)
            def slow_computation(n):
                # Simulate slow computation
                time.sleep(0.01)  # 10ms delay
                return sum(range(n))
            
            # First call - should be slow
            start_time = time.time()
            result1 = slow_computation(100)
            first_call_time = time.time() - start_time
            
            # Second call - should be fast (cached)
            start_time = time.time()
            result2 = slow_computation(100)
            second_call_time = time.time() - start_time
            
            # Verify results are same and second call is faster
            assert result1 == result2
            assert second_call_time < first_call_time * 0.5  # At least 50% faster