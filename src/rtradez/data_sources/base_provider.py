"""Base data provider interface for consistent data integration."""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from ..utils.caching import cached


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    provider_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_per_minute: int = 60
    cache_duration_hours: int = 24
    free_tier_limit: Optional[int] = None
    requires_registration: bool = False
    cost_per_request: float = 0.0
    data_categories: List[str] = None


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.
    
    Provides consistent interface for:
    - Data fetching with caching
    - Rate limiting
    - Error handling
    - Data standardization
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        Initialize data provider.
        
        Args:
            config: Data source configuration
        """
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.rate_limit = config.rate_limit_per_minute
        self.cache_duration = config.cache_duration_hours * 3600  # Convert to seconds
        
        # Request tracking for rate limiting
        self.request_count = 0
        self.last_reset_time = datetime.now()
    
    @abstractmethod
    def fetch_data(self, 
                   symbol: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   **kwargs) -> pd.DataFrame:
        """
        Fetch data from the provider.
        
        Args:
            symbol: Asset symbol (if applicable)
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Provider-specific parameters
        
        Returns:
            Standardized DataFrame with datetime index
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols/datasets."""
        pass
    
    @abstractmethod
    def get_data_description(self, symbol: str) -> Dict[str, Any]:
        """Get description of available data for symbol."""
        pass
    
    def check_rate_limit(self):
        """Check and enforce rate limiting."""
        now = datetime.now()
        
        # Reset counter every minute
        if (now - self.last_reset_time).seconds >= 60:
            self.request_count = 0
            self.last_reset_time = now
        
        if self.request_count >= self.rate_limit:
            raise Exception(f"Rate limit exceeded for {self.config.provider_name}")
        
        self.request_count += 1
    
    def standardize_data(self, data: pd.DataFrame, 
                        datetime_col: str = 'date',
                        value_cols: List[str] = None) -> pd.DataFrame:
        """
        Standardize data format.
        
        Args:
            data: Raw data from provider
            datetime_col: Name of datetime column
            value_cols: Names of value columns to standardize
        
        Returns:
            Standardized DataFrame
        """
        if data.empty:
            return data
        
        # Ensure datetime index
        if datetime_col in data.columns:
            data[datetime_col] = pd.to_datetime(data[datetime_col])
            data = data.set_index(datetime_col)
        
        # Sort by index
        data = data.sort_index()
        
        # Standardize column names (lowercase, underscore)
        data.columns = [col.lower().replace(' ', '_').replace('-', '_') 
                       for col in data.columns]
        
        # Convert numeric columns
        if value_cols:
            for col in value_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def validate_data_quality(self, data: pd.DataFrame,
                             min_rows: int = 10,
                             max_missing_pct: float = 0.5) -> bool:
        """
        Validate data quality.
        
        Args:
            data: Data to validate
            min_rows: Minimum required rows
            max_missing_pct: Maximum allowed missing data percentage
        
        Returns:
            True if data quality is acceptable
        """
        if len(data) < min_rows:
            warnings.warn(f"Insufficient data: {len(data)} rows (minimum: {min_rows})")
            return False
        
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > max_missing_pct:
            warnings.warn(f"Too much missing data: {missing_pct:.1%} (max: {max_missing_pct:.1%})")
            return False
        
        return True
    
    @cached(cache_type='market_data')
    def get_cached_data(self, cache_key: str, fetch_func, **kwargs) -> pd.DataFrame:
        """
        Get data with caching.
        
        Args:
            cache_key: Unique cache key
            fetch_func: Function to fetch data if not cached
            **kwargs: Arguments for fetch function
        
        Returns:
            Cached or freshly fetched data
        """
        return fetch_func(**kwargs)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this provider."""
        return {
            'provider': self.config.provider_name,
            'requests_this_minute': self.request_count,
            'rate_limit': self.rate_limit,
            'last_request_time': self.last_reset_time,
            'cost_estimate': self.request_count * self.config.cost_per_request,
            'free_tier_remaining': (self.config.free_tier_limit - self.request_count 
                                  if self.config.free_tier_limit else None)
        }


class DataProviderFactory:
    """Factory for creating data providers."""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a data provider class."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, name: str, config: DataSourceConfig) -> BaseDataProvider:
        """Create a data provider instance."""
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        
        return cls._providers[name](config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())


# Data source configurations for easy setup
DEFAULT_CONFIGS = {
    'fred': DataSourceConfig(
        provider_name='Federal Reserve Economic Data',
        base_url='https://api.stlouisfed.org/fred',
        rate_limit_per_minute=120,
        requires_registration=True,
        data_categories=['economic_indicators', 'interest_rates', 'inflation', 'employment']
    ),
    
    'alpha_vantage': DataSourceConfig(
        provider_name='Alpha Vantage',
        base_url='https://www.alphavantage.co/query',
        rate_limit_per_minute=5,  # Free tier
        free_tier_limit=500,  # Per day
        requires_registration=True,
        data_categories=['stocks', 'forex', 'crypto', 'economics', 'sentiment']
    ),
    
    'yahoo_finance': DataSourceConfig(
        provider_name='Yahoo Finance',
        base_url='https://query1.finance.yahoo.com',
        rate_limit_per_minute=200,
        requires_registration=False,
        data_categories=['stocks', 'options', 'futures', 'currencies']
    ),
    
    'financial_modeling_prep': DataSourceConfig(
        provider_name='Financial Modeling Prep',
        base_url='https://financialmodelingprep.com/api',
        rate_limit_per_minute=10,  # Free tier
        free_tier_limit=250,  # Per day
        requires_registration=True,
        data_categories=['fundamentals', 'earnings', 'estimates', 'insider_trading']
    ),
    
    'news_api': DataSourceConfig(
        provider_name='News API',
        base_url='https://newsapi.org/v2',
        rate_limit_per_minute=60,
        free_tier_limit=1000,  # Per day
        requires_registration=True,
        data_categories=['financial_news', 'sentiment']
    ),
    
    'openweather': DataSourceConfig(
        provider_name='OpenWeather',
        base_url='https://api.openweathermap.org/data',
        rate_limit_per_minute=60,
        free_tier_limit=1000,  # Per day
        requires_registration=True,
        data_categories=['weather', 'climate']
    ),
    
    'coinbase': DataSourceConfig(
        provider_name='Coinbase Pro',
        base_url='https://api.exchange.coinbase.com',
        rate_limit_per_minute=10,
        requires_registration=False,
        data_categories=['crypto_spot', 'crypto_options']
    ),
    
    'worldbank': DataSourceConfig(
        provider_name='World Bank',
        base_url='https://api.worldbank.org/v2',
        rate_limit_per_minute=120,
        requires_registration=False,
        data_categories=['global_economics', 'development_indicators']
    )
}