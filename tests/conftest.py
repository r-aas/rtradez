"""Pytest configuration and fixtures for RTradez test suite."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple

from rtradez.datasets.core import OptionsDataset
from rtradez.methods.strategies import OptionsStrategy


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Create realistic market data
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100))
    volumes = np.random.lognormal(10, 1, 100)
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': prices,
        'volume': volumes,
        'returns': np.concatenate([[0], np.diff(np.log(prices))]),
        'volatility': np.random.uniform(0.1, 0.5, 100)
    })
    
    return data.set_index('date')


@pytest.fixture
def sample_options_data() -> pd.DataFrame:
    """Create sample options chain data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    strikes = np.arange(90, 111, 2.5)
    
    data = []
    for date in dates:
        spot_price = 100 + np.random.normal(0, 5)
        for strike in strikes:
            # Create call and put data
            for option_type in ['call', 'put']:
                moneyness = strike / spot_price
                if option_type == 'call':
                    intrinsic = max(0, spot_price - strike)
                else:
                    intrinsic = max(0, strike - spot_price)
                
                # Simple implied volatility model
                iv = 0.2 + 0.1 * abs(1 - moneyness)
                
                data.append({
                    'date': date,
                    'strike': strike,
                    'option_type': option_type,
                    'bid': max(0.01, intrinsic + np.random.uniform(0, 2)),
                    'ask': max(0.02, intrinsic + np.random.uniform(1, 3)),
                    'implied_volatility': iv,
                    'delta': np.random.uniform(0, 1) if option_type == 'call' else np.random.uniform(-1, 0),
                    'gamma': np.random.uniform(0, 0.1),
                    'theta': np.random.uniform(-0.1, 0),
                    'vega': np.random.uniform(0, 0.5),
                    'volume': np.random.poisson(10),
                    'open_interest': np.random.poisson(100)
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_and_returns() -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample feature matrix and returns series for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create features
    features = pd.DataFrame({
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.normal(0, 1, n_samples),
        'bb_position': np.random.uniform(0, 1, n_samples),
        'volatility': np.random.uniform(0.1, 0.5, n_samples),
        'volume_ratio': np.random.uniform(0.5, 2, n_samples),
        'price_momentum': np.random.normal(0, 0.02, n_samples)
    })
    
    # Create returns correlated with features
    returns = (
        0.01 * features['rsi'] / 50 - 0.01 +
        0.005 * features['macd'] +
        np.random.normal(0, 0.02, n_samples)
    )
    
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    features.index = dates
    returns.index = dates
    
    return features, returns


@pytest.fixture
def sample_strategy() -> OptionsStrategy:
    """Create a sample options strategy for testing."""
    return OptionsStrategy(
        strategy_type='iron_condor',
        profit_target=0.35,
        stop_loss=2.0,
        dte_range=(20, 40)
    )


@pytest.fixture
def mock_dataset(sample_market_data, sample_options_data) -> OptionsDataset:
    """Create a mock OptionsDataset for testing."""
    dataset = OptionsDataset('TEST', sample_market_data.index[0], sample_market_data.index[-1])
    dataset.market_data = sample_market_data
    dataset.options_data = sample_options_data
    return dataset


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "slow: Slow-running tests")


# Skip slow tests by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--runslow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )