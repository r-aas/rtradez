"""Tests for base classes and interfaces."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from rtradez.base import BaseStrategy, BaseTransformer, BaseMetric, BaseEstimator


@pytest.mark.unit
class TestBaseStrategy:
    """Test BaseStrategy class."""
    
    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseStrategy()
    
    def test_sklearn_interface_methods_exist(self):
        """Test that sklearn interface methods are defined."""
        # Create a concrete implementation
        class ConcreteStrategy(BaseStrategy):
            def fit(self, X, y=None):
                return self
            
            def predict(self, X):
                return np.zeros(len(X))
        
        strategy = ConcreteStrategy()
        
        # Test sklearn interface exists
        assert hasattr(strategy, 'fit')
        assert hasattr(strategy, 'predict')
        assert hasattr(strategy, 'score')
        assert hasattr(strategy, 'get_params')
        assert hasattr(strategy, 'set_params')
    
    def test_get_params_returns_dict(self):
        """Test get_params returns a dictionary."""
        class ConcreteStrategy(BaseStrategy):
            def __init__(self, param1=1, param2=2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2
            
            def fit(self, X, y=None):
                return self
            
            def predict(self, X):
                return np.zeros(len(X))
        
        strategy = ConcreteStrategy(param1=10, param2=20)
        params = strategy.get_params()
        
        assert isinstance(params, dict)
        assert 'param1' in params
        assert 'param2' in params
        assert params['param1'] == 10
        assert params['param2'] == 20
    
    def test_set_params_updates_attributes(self):
        """Test set_params updates object attributes."""
        class ConcreteStrategy(BaseStrategy):
            def __init__(self, param1=1, param2=2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2
            
            def fit(self, X, y=None):
                return self
            
            def predict(self, X):
                return np.zeros(len(X))
        
        strategy = ConcreteStrategy(param1=10, param2=20)
        strategy.set_params(param1=100, param2=200)
        
        assert strategy.param1 == 100
        assert strategy.param2 == 200
    
    def test_score_calculates_sharpe_ratio(self, sample_features_and_returns):
        """Test that score method calculates Sharpe ratio."""
        X, y = sample_features_and_returns
        
        class ConcreteStrategy(BaseStrategy):
            def fit(self, X, y=None):
                return self
            
            def predict(self, X):
                # Return simple signal based on first feature
                return (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)
        
        strategy = ConcreteStrategy()
        strategy.fit(X, y)
        score = strategy.score(X, y)
        
        assert isinstance(score, float)
        assert not np.isnan(score)


@pytest.mark.unit
class TestBaseTransformer:
    """Test BaseTransformer class."""
    
    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseTransformer()
    
    def test_sklearn_transformer_interface(self):
        """Test sklearn transformer interface."""
        class ConcreteTransformer(BaseTransformer):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                return X.copy()
        
        transformer = ConcreteTransformer()
        
        assert hasattr(transformer, 'fit')
        assert hasattr(transformer, 'transform')
        assert hasattr(transformer, 'fit_transform')
    
    def test_fit_transform_combines_fit_and_transform(self, sample_features_and_returns):
        """Test fit_transform method."""
        X, _ = sample_features_and_returns
        
        class ConcreteTransformer(BaseTransformer):
            def __init__(self):
                self.fitted = False
            
            def fit(self, X, y=None):
                self.fitted = True
                return self
            
            def transform(self, X):
                if not self.fitted:
                    raise ValueError("Must fit before transform")
                return X * 2  # Simple transformation
        
        transformer = ConcreteTransformer()
        result = transformer.fit_transform(X)
        
        assert transformer.fitted
        assert np.allclose(result, X * 2)


@pytest.mark.unit  
class TestBaseMetric:
    """Test BaseMetric class."""
    
    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseMetric()
    
    def test_calculate_method_exists(self):
        """Test calculate method exists."""
        class ConcreteMetric(BaseMetric):
            def calculate(self, returns, benchmark=None):
                return np.mean(returns)
        
        metric = ConcreteMetric()
        assert hasattr(metric, 'calculate')
    
    def test_calculate_with_sample_data(self, sample_features_and_returns):
        """Test calculate method with sample data."""
        _, returns = sample_features_and_returns
        
        class ConcreteMetric(BaseMetric):
            def calculate(self, returns, benchmark=None):
                return np.mean(returns)
        
        metric = ConcreteMetric()
        result = metric.calculate(returns)
        
        assert isinstance(result, (int, float, np.number))
        assert not np.isnan(result)


@pytest.mark.unit
class TestBaseEstimator:
    """Test BaseEstimator class."""
    
    def test_get_params_deep_parameter(self):
        """Test get_params with deep parameter."""
        class ConcreteEstimator(BaseEstimator):
            def __init__(self, param1=1, param2=2):
                self.param1 = param1
                self.param2 = param2
        
        estimator = ConcreteEstimator(param1=10, param2=20)
        
        # Test shallow params
        params_shallow = estimator.get_params(deep=False)
        assert params_shallow == {'param1': 10, 'param2': 20}
        
        # Test deep params (same result for simple case)
        params_deep = estimator.get_params(deep=True)
        assert params_deep == {'param1': 10, 'param2': 20}
    
    def test_set_params_returns_self(self):
        """Test set_params returns self for chaining."""
        class ConcreteEstimator(BaseEstimator):
            def __init__(self, param1=1):
                self.param1 = param1
        
        estimator = ConcreteEstimator(param1=10)
        result = estimator.set_params(param1=20)
        
        assert result is estimator
        assert estimator.param1 == 20
    
    def test_invalid_parameter_raises_error(self):
        """Test setting invalid parameter raises error."""
        class ConcreteEstimator(BaseEstimator):
            def __init__(self, param1=1):
                self.param1 = param1
        
        estimator = ConcreteEstimator(param1=10)
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            estimator.set_params(invalid_param=100)