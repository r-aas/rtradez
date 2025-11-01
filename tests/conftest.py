"""Pytest configuration and fixtures for RTradez tests."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from datetime import datetime, timedelta


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    np.random.seed(42)
    
    # Generate sample time series data
    dates = pd.bdate_range(start='2023-01-01', end='2023-12-31')
    n_samples = len(dates)
    
    # Market data
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_samples))
    volumes = np.random.exponential(1000000, n_samples)
    
    # Feature matrix
    X = pd.DataFrame({
        'price': prices,
        'volume': volumes,
        'returns': np.concatenate([[0], np.diff(np.log(prices))]),
        'volatility': pd.Series(np.random.normal(0.001, 0.02, n_samples)).rolling(20).std(),
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
    }, index=dates)
    
    # Target variable (returns)
    y = pd.Series(X['returns'].values, index=dates)
    
    # Predictions
    y_pred = y + np.random.normal(0, 0.001, len(y))
    
    # Returns for financial metrics
    returns = pd.Series(np.random.normal(0.001, 0.02, n_samples), index=dates)
    
    return {
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'returns': returns,
        'prices': pd.Series(prices, index=dates),
        'volumes': pd.Series(volumes, index=dates),
        'dates': dates,
        
        # Common parameters for different class types
        'init_params': {},
        'strategy_params': {'strategy_type': 'iron_condor'},
        'transformer_params': {},
        'metric_params': {},
        'provider_params': {},
        'validator_params': {},
        
        # Method-specific parameters
        'fit_params': {},
        'predict_params': {},
        'transform_params': {},
        'fetch_params': {
            'symbol': 'SPY',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        },
        
        # Mock API response
        'api_response': {
            'data': [
                {'date': '2023-01-01', 'value': 100},
                {'date': '2023-01-02', 'value': 101},
            ]
        },
        
        # Mock model
        'mock_model': Mock()
    }


@pytest.fixture
def sample_datasets():
    """Provide sample datasets for multi-source testing."""
    dates = pd.bdate_range(start='2023-01-01', end='2023-12-31')
    
    # Daily data
    daily_data = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),
        'volume': np.random.exponential(1000000, len(dates))
    }, index=dates)
    
    # Weekly data
    weekly_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='W')
    weekly_data = pd.DataFrame({
        'economic_indicator': np.random.normal(50, 10, len(weekly_dates))
    }, index=weekly_dates)
    
    # Monthly data
    monthly_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    monthly_data = pd.DataFrame({
        'sentiment': np.random.normal(0, 1, len(monthly_dates))
    }, index=monthly_dates)
    
    return {
        'daily': daily_data,
        'weekly': weekly_data,
        'monthly': monthly_data
    }


@pytest.fixture
def mock_api_responses():
    """Mock API responses for data providers."""
    return {
        'fred': {
            'observations': [
                {'date': '2023-01-01', 'value': '2.5'},
                {'date': '2023-01-02', 'value': '2.6'},
            ]
        },
        'alpha_vantage': {
            'Time Series (Daily)': {
                '2023-01-01': {'4. close': '100.00'},
                '2023-01-02': {'4. close': '101.00'},
            }
        },
        'news_api': {
            'articles': [
                {
                    'title': 'Market update',
                    'description': 'Positive market sentiment',
                    'publishedAt': '2023-01-01T10:00:00Z'
                }
            ]
        }
    }
