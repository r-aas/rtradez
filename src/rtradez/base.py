"""
Base classes for sklearn-like interface in RTradez.

Provides consistent API patterns following sklearn conventions:
- .fit(X, y) for training/learning from data
- .predict(X) for making predictions
- .transform(X) for data transformation
- .score(X, y) for evaluation metrics
- .fit_transform(X, y) for combined fit and transform
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union, Tuple
from sklearn.base import BaseEstimator


class BaseRTradez(BaseEstimator):
    """
    Base class for all RTradez components following sklearn conventions.
    
    Provides common functionality and ensures consistent API across all
    RTradez algorithms, models, and transformers.
    """
    
    def __init__(self):
        """Initialize base class."""
        self.is_fitted_ = False
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray], 
                       reset: bool = True) -> pd.DataFrame:
        """
        Validate input data following sklearn patterns.
        
        Args:
            X: Input data
            reset: Whether to reset fitted state
            
        Returns:
            Validated DataFrame
        """
        if reset:
            self.is_fitted_ = False
            
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be pandas DataFrame or numpy array")
            
        # Store input info (sklearn convention)
        if reset:
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = list(X.columns)
            
        return X
        
    def _check_is_fitted(self):
        """Check if estimator is fitted (sklearn convention)."""
        if not self.is_fitted_:
            raise ValueError("This estimator has not been fitted yet. Call 'fit' first.")
            
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters (sklearn convention)."""
        return super().get_params(deep=deep)
        
    def set_params(self, **params) -> 'BaseRTradez':
        """Set parameters (sklearn convention)."""
        return super().set_params(**params)


class BaseStrategy(BaseRTradez):
    """
    Base class for trading strategies following sklearn patterns.
    
    Strategies follow this pattern:
    - .fit(price_data, options_data) -> learn from market data
    - .predict(market_conditions) -> generate trading signals
    - .score(X, y) -> evaluate strategy performance
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseStrategy':
        """
        Fit strategy to market data.
        
        Args:
            X: Market data (price, volume, volatility, etc.)
            y: Optional target returns or labels
            
        Returns:
            Fitted strategy instance
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals.
        
        Args:
            X: Market conditions
            
        Returns:
            Trading signals (-1: sell, 0: hold, 1: buy)
        """
        pass
        
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Evaluate strategy performance (default: Sharpe ratio).
        
        Args:
            X: Market data
            y: Actual returns
            
        Returns:
            Performance score
        """
        signals = self.predict(X)
        
        # Align signals and returns properly 
        min_length = min(len(signals) - 1, len(y) - 1)
        aligned_signals = signals[:min_length]
        aligned_returns = y.values[1:min_length + 1]
        
        strategy_returns = aligned_signals * aligned_returns
        
        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            return 0.0
            
        return strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)  # Sharpe ratio


class BaseTransformer(BaseRTradez):
    """
    Base class for data transformers following sklearn patterns.
    
    Transformers follow this pattern:
    - .fit(raw_data) -> learn transformation parameters
    - .transform(raw_data) -> apply transformation
    - .fit_transform(raw_data) -> combined fit and transform
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseTransformer':
        """
        Learn transformation parameters.
        
        Args:
            X: Raw data to learn from
            y: Optional target (usually not used)
            
        Returns:
            Fitted transformer instance
        """
        pass
        
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned transformation.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        pass
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step (sklearn convention).
        
        Args:
            X: Data to fit and transform
            y: Optional target
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)


class BaseMetric(BaseRTradez):
    """
    Base class for performance metrics following sklearn patterns.
    
    Metrics follow this pattern:
    - .fit(returns_data) -> learn metric parameters (if needed)
    - .score(actual, predicted) -> compute metric value
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseMetric':
        """
        Fit metric (usually no-op for metrics).
        
        Args:
            X: Historical data
            y: Optional target
            
        Returns:
            Fitted metric instance
        """
        X = self._validate_input(X)
        self.is_fitted_ = True
        return self
        
    @abstractmethod
    def score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Compute metric score.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Metric value
        """
        pass


class BaseEstimator(BaseRTradez):
    """
    Base class for estimators/models following sklearn patterns.
    
    Estimators follow this pattern:
    - .fit(features, target) -> learn model parameters
    - .predict(features) -> make predictions
    - .score(features, target) -> evaluate predictions
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseEstimator':
        """
        Train the estimator.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Fitted estimator instance
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        pass
        
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Evaluate predictions (default: R-squared for regression).
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


# Utility functions for sklearn-like parameter validation
def check_array(array: Union[pd.DataFrame, np.ndarray], 
               ensure_2d: bool = True) -> Union[pd.DataFrame, np.ndarray]:
    """Validate array input following sklearn patterns."""
    if isinstance(array, pd.DataFrame):
        if ensure_2d and array.ndim != 2:
            raise ValueError("Expected 2D array")
        return array
    elif isinstance(array, np.ndarray):
        if ensure_2d and array.ndim != 2:
            raise ValueError("Expected 2D array")
        return array
    else:
        raise ValueError("Expected pandas DataFrame or numpy array")


def check_X_y(X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray]) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate X and y inputs following sklearn patterns."""
    X = check_array(X, ensure_2d=True)
    
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be pandas Series or numpy array")
        
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
        
    return X, y