"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from rtradez.methods.strategies import OptionsStrategy


@pytest.mark.unit
class TestOptionsStrategy:
    """Test OptionsStrategy class."""
    
    def test_initialization_with_defaults(self):
        """Test strategy initialization with default parameters."""
        strategy = OptionsStrategy('iron_condor')
        
        assert strategy.strategy_type == 'iron_condor'
        assert hasattr(strategy, 'profit_target')
        assert hasattr(strategy, 'stop_loss')
        assert hasattr(strategy, 'dte_range')
    
    def test_initialization_with_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = OptionsStrategy(
            strategy_type='strangle',
            profit_target=0.5,
            stop_loss=3.0,
            dte_range=(15, 35)
        )
        
        assert strategy.strategy_type == 'strangle'
        assert strategy.profit_target == 0.5
        assert strategy.stop_loss == 3.0
        assert strategy.dte_range == (15, 35)
    
    def test_get_params_sklearn_compatibility(self):
        """Test get_params returns all parameters for sklearn compatibility."""
        strategy = OptionsStrategy(
            strategy_type='iron_condor',
            profit_target=0.35,
            stop_loss=2.0
        )
        
        params = strategy.get_params()
        
        assert isinstance(params, dict)
        assert 'strategy_type' in params
        assert 'profit_target' in params
        assert 'stop_loss' in params
        assert params['strategy_type'] == 'iron_condor'
        assert params['profit_target'] == 0.35
        assert params['stop_loss'] == 2.0
    
    def test_set_params_sklearn_compatibility(self):
        """Test set_params updates parameters for sklearn compatibility."""
        strategy = OptionsStrategy('iron_condor')
        
        result = strategy.set_params(
            strategy_type='strangle',
            profit_target=0.4,
            stop_loss=2.5
        )
        
        assert result is strategy  # Returns self for chaining
        assert strategy.strategy_type == 'strangle'
        assert strategy.profit_target == 0.4
        assert strategy.stop_loss == 2.5
    
    def test_set_params_invalid_parameter_raises_error(self):
        """Test setting invalid parameter raises ValueError."""
        strategy = OptionsStrategy('iron_condor')
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            strategy.set_params(invalid_param=123)
    
    def test_fit_method_exists_and_returns_self(self, sample_features_and_returns):
        """Test fit method exists and returns self."""
        X, y = sample_features_and_returns
        strategy = OptionsStrategy('iron_condor')
        
        result = strategy.fit(X, y)
        
        assert result is strategy
        assert hasattr(strategy, '_fitted')
    
    def test_predict_method_returns_signals(self, sample_features_and_returns):
        """Test predict method returns signal array."""
        X, y = sample_features_and_returns
        strategy = OptionsStrategy('iron_condor')
        
        # Fit first
        strategy.fit(X, y)
        
        # Predict
        signals = strategy.predict(X)
        
        assert isinstance(signals, np.ndarray)
        assert len(signals) == len(X)
        assert signals.dtype in [np.int32, np.int64, np.float32, np.float64]
        assert np.all(np.isin(signals, [0, 1]))  # Binary signals
    
    def test_score_method_returns_sharpe_ratio(self, sample_features_and_returns):
        """Test score method returns Sharpe ratio."""
        X, y = sample_features_and_returns
        strategy = OptionsStrategy('iron_condor')
        
        # Fit and score
        strategy.fit(X, y)
        score = strategy.score(X, y)
        
        assert isinstance(score, (int, float, np.number))
        assert not np.isnan(score)
    
    def test_predict_without_fit_raises_error(self, sample_features_and_returns):
        """Test predict without fit raises appropriate error."""
        X, _ = sample_features_and_returns
        strategy = OptionsStrategy('iron_condor')
        
        with pytest.raises(ValueError, match="must be fitted"):
            strategy.predict(X)
    
    @pytest.mark.parametrize("strategy_type", [
        'iron_condor', 'strangle', 'straddle', 'calendar_spread'
    ])
    def test_different_strategy_types(self, strategy_type, sample_features_and_returns):
        """Test different strategy types work correctly."""
        X, y = sample_features_and_returns
        strategy = OptionsStrategy(strategy_type)
        
        # Should be able to fit and predict for all types
        strategy.fit(X, y)
        signals = strategy.predict(X)
        score = strategy.score(X, y)
        
        assert len(signals) == len(X)
        assert isinstance(score, (int, float, np.number))
    
    def test_strategy_with_sklearn_pipeline(self, sample_features_and_returns):
        """Test strategy works with sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        X, y = sample_features_and_returns
        
        # Create pipeline with strategy
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('strategy', OptionsStrategy('iron_condor'))
        ])
        
        # Should work with sklearn pipeline
        pipeline.fit(X, y)
        signals = pipeline.predict(X)
        score = pipeline.score(X, y)
        
        assert len(signals) == len(X)
        assert isinstance(score, (int, float, np.number))
    
    def test_strategy_with_sklearn_grid_search(self, sample_features_and_returns):
        """Test strategy works with sklearn GridSearchCV."""
        from sklearn.model_selection import GridSearchCV
        
        X, y = sample_features_and_returns
        strategy = OptionsStrategy('iron_condor')
        
        # Define parameter grid
        param_grid = {
            'profit_target': [0.3, 0.4, 0.5],
            'stop_loss': [2.0, 2.5, 3.0]
        }
        
        # Grid search should work
        grid_search = GridSearchCV(
            strategy, 
            param_grid, 
            cv=3, 
            scoring='neg_mean_squared_error'
        )
        
        grid_search.fit(X, y)
        
        assert hasattr(grid_search, 'best_params_')
        assert hasattr(grid_search, 'best_score_')
        assert grid_search.best_params_['profit_target'] in [0.3, 0.4, 0.5]
        assert grid_search.best_params_['stop_loss'] in [2.0, 2.5, 3.0]


@pytest.mark.integration
class TestOptionsStrategyIntegration:
    """Integration tests for OptionsStrategy with real-like data."""
    
    def test_end_to_end_workflow(self, sample_features_and_returns):
        """Test complete end-to-end workflow."""
        X, y = sample_features_and_returns
        
        # Initialize strategy
        strategy = OptionsStrategy(
            strategy_type='iron_condor',
            profit_target=0.35,
            stop_loss=2.0
        )
        
        # Complete workflow
        strategy.fit(X, y)
        signals = strategy.predict(X)
        score = strategy.score(X, y)
        
        # Verify results
        assert strategy._fitted
        assert len(signals) == len(X)
        assert np.all(np.isin(signals, [0, 1]))
        assert isinstance(score, (int, float, np.number))
        assert not np.isnan(score)
    
    @pytest.mark.slow
    def test_optimization_with_large_dataset(self):
        """Test strategy optimization with larger dataset."""
        # Create larger dataset
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.normal(0, 1, n_samples),
            'volatility': np.random.uniform(0.1, 0.5, n_samples),
        })
        
        y = (0.01 * X['rsi'] / 50 - 0.01 + 
             0.005 * X['macd'] + 
             np.random.normal(0, 0.02, n_samples))
        
        strategy = OptionsStrategy('iron_condor')
        
        # Should handle large dataset efficiently
        strategy.fit(X, y)
        signals = strategy.predict(X)
        score = strategy.score(X, y)
        
        assert len(signals) == n_samples
        assert isinstance(score, (int, float, np.number))