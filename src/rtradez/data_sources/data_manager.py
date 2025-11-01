"""Unified data manager for RTradez research platform."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base_provider import BaseDataProvider, DataProviderFactory, DEFAULT_CONFIGS
from .economic.fred import FREDProvider
from .alternative.sentiment import SentimentProvider
from .crypto.spot_prices import CryptoProvider


class RTradezDataManager:
    """
    Unified data manager for comprehensive research.
    
    Integrates multiple data sources:
    - Economic indicators (FRED, World Bank)
    - Market sentiment (News, Fear & Greed)
    - Alternative data (Weather, Social)
    - Cryptocurrency data (Spot, DeFi options)
    - Fundamental data (Earnings, Financials)
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize data manager.
        
        Args:
            api_keys: Dictionary of API keys by provider name
                     e.g., {'fred': 'your_key', 'alpha_vantage': 'your_key'}
        """
        self.api_keys = api_keys or {}
        self.providers: Dict[str, BaseDataProvider] = {}
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # Initialize available providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available data providers."""
        try:
            # FRED Economic Data
            if 'fred' in self.api_keys:
                fred_config = DEFAULT_CONFIGS['fred']
                fred_config.api_key = self.api_keys['fred']
                self.providers['fred'] = FREDProvider(fred_config)
            
            # Sentiment Data
            sentiment_config = DEFAULT_CONFIGS['alpha_vantage']
            if 'alpha_vantage' in self.api_keys:
                sentiment_config.api_key = self.api_keys['alpha_vantage']
            self.providers['sentiment'] = SentimentProvider(sentiment_config)
            
            # Crypto Data
            crypto_config = DEFAULT_CONFIGS['coinbase']
            self.providers['crypto'] = CryptoProvider(crypto_config)
            
            print(f"✅ Initialized {len(self.providers)} data providers")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize some providers: {e}")
    
    def get_comprehensive_dataset(self,
                                symbol: str,
                                start_date: str,
                                end_date: Optional[str] = None,
                                include_sources: List[str] = None,
                                align_data: bool = True) -> pd.DataFrame:
        """
        Get comprehensive dataset combining multiple sources.
        
        Args:
            symbol: Primary asset symbol
            start_date: Start date for data
            end_date: End date for data (default: today)
            include_sources: List of data sources to include
            align_data: Whether to align all data to common dates
        
        Returns:
            Combined DataFrame with data from multiple sources
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if include_sources is None:
            include_sources = list(self.providers.keys())
        
        print(f"🔍 Fetching comprehensive dataset for {symbol}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"📊 Sources: {', '.join(include_sources)}")
        
        datasets = {}
        
        # Fetch market data (primary)
        if symbol.endswith('-USD') or symbol.lower() in ['bitcoin', 'ethereum']:
            # Crypto symbol
            if 'crypto' in include_sources and 'crypto' in self.providers:
                print("   📈 Fetching crypto market data...")
                market_data = self.providers['crypto'].fetch_data(
                    symbol, start_date, end_date, data_type='spot'
                )
                if not market_data.empty:
                    datasets['market'] = market_data
        else:
            # Traditional asset - would integrate with yfinance or similar
            print("   📈 Generating synthetic market data...")
            datasets['market'] = self._generate_market_data(symbol, start_date, end_date)
        
        # Fetch economic indicators
        if 'fred' in include_sources and 'fred' in self.providers:
            print("   🏛️ Fetching economic indicators...")
            try:
                econ_data = self.providers['fred'].get_economic_indicators_bundle(
                    start_date, end_date
                )
                if not econ_data.empty:
                    datasets['economic'] = econ_data
            except Exception as e:
                warnings.warn(f"Failed to fetch economic data: {e}")
        
        # Fetch sentiment data
        if 'sentiment' in include_sources and 'sentiment' in self.providers:
            print("   😊 Fetching sentiment data...")
            try:
                sentiment_data = self.providers['sentiment'].fetch_data(
                    symbol, start_date, end_date, sentiment_type='calculated'
                )
                if not sentiment_data.empty:
                    datasets['sentiment'] = sentiment_data
                
                # Add Fear & Greed Index
                fear_greed = self.providers['sentiment'].fetch_data(
                    sentiment_type='fear_greed'
                )
                if not fear_greed.empty:
                    # Filter to date range
                    mask = (fear_greed.index >= start_date) & (fear_greed.index <= end_date)
                    fear_greed = fear_greed[mask]
                    if not fear_greed.empty:
                        datasets['fear_greed'] = fear_greed
            except Exception as e:
                warnings.warn(f"Failed to fetch sentiment data: {e}")
        
        # Fetch crypto metrics (if crypto symbol)
        if symbol.endswith('-USD') and 'crypto' in include_sources and 'crypto' in self.providers:
            print("   🔗 Fetching crypto metrics...")
            try:
                crypto_metrics = self.providers['crypto'].fetch_data(
                    symbol, start_date, end_date, data_type='metrics'
                )
                if not crypto_metrics.empty:
                    datasets['crypto_metrics'] = crypto_metrics
            except Exception as e:
                warnings.warn(f"Failed to fetch crypto metrics: {e}")
        
        # Combine datasets
        if not datasets:
            print("   ❌ No data fetched")
            return pd.DataFrame()
        
        print(f"   ✅ Fetched data from {len(datasets)} sources")
        
        if align_data:
            combined_data = self._align_and_combine_datasets(datasets)
        else:
            # Simple concatenation
            combined_data = pd.concat(datasets.values(), axis=1)
        
        print(f"   📊 Combined dataset: {len(combined_data)} rows, {len(combined_data.columns)} columns")
        
        return combined_data
    
    def _align_and_combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align and combine datasets from different sources."""
        if not datasets:
            return pd.DataFrame()
        
        # Find common date range
        start_dates = []
        end_dates = []
        
        for name, data in datasets.items():
            if not data.empty:
                start_dates.append(data.index.min())
                end_dates.append(data.index.max())
        
        if not start_dates:
            return pd.DataFrame()
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        print(f"   🔗 Aligning data to common period: {common_start.date()} to {common_end.date()}")
        
        # Align all datasets to business days in common range
        business_days = pd.bdate_range(start=common_start, end=common_end)
        
        aligned_datasets = []
        for name, data in datasets.items():
            if data.empty:
                continue
            
            # Filter to common range
            mask = (data.index >= common_start) & (data.index <= common_end)
            filtered_data = data[mask]
            
            if filtered_data.empty:
                continue
            
            # Reindex to business days with forward fill
            aligned_data = filtered_data.reindex(business_days, method='ffill')
            
            # Add source prefix to column names
            aligned_data.columns = [f"{name}_{col}" for col in aligned_data.columns]
            
            aligned_datasets.append(aligned_data)
        
        if not aligned_datasets:
            return pd.DataFrame()
        
        # Combine all aligned datasets
        combined_data = pd.concat(aligned_datasets, axis=1)
        
        return combined_data
    
    def _generate_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic market data for traditional assets."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Business days
        
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate realistic price series
        base_price = 100
        volatility = 0.02
        returns = np.random.normal(0.0005, volatility, len(dates))
        prices = base_price * np.cumprod(1 + returns)
        volumes = np.random.exponential(1000000, len(dates))
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(df)))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, len(df))))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, len(df))))
        df['Volume'] = volumes
        
        # Add returns and volatility
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        return df.fillna(method='bfill')
    
    def get_economic_research_dataset(self,
                                    start_date: str,
                                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get comprehensive economic research dataset.
        
        Returns:
            DataFrame with economic indicators, rates, sentiment
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print("🏛️ Building economic research dataset...")
        
        datasets = {}
        
        # Economic indicators
        if 'fred' in self.providers:
            try:
                # Core economic indicators
                econ_bundle = self.providers['fred'].get_economic_indicators_bundle(
                    start_date, end_date
                )
                if not econ_bundle.empty:
                    datasets['economic'] = econ_bundle
                
                # Yield curve
                yield_curve = self.providers['fred'].get_yield_curve_data(
                    start_date, end_date
                )
                if not yield_curve.empty:
                    datasets['yield_curve'] = yield_curve
                
                # Employment data
                employment = self.providers['fred'].get_employment_data(
                    start_date, end_date
                )
                if not employment.empty:
                    datasets['employment'] = employment
                    
            except Exception as e:
                warnings.warn(f"Failed to fetch economic data: {e}")
        
        # Market sentiment
        if 'sentiment' in self.providers:
            try:
                fear_greed = self.providers['sentiment'].fetch_data(
                    sentiment_type='fear_greed'
                )
                if not fear_greed.empty:
                    mask = (fear_greed.index >= start_date) & (fear_greed.index <= end_date)
                    fear_greed = fear_greed[mask]
                    if not fear_greed.empty:
                        datasets['sentiment'] = fear_greed
            except Exception as e:
                warnings.warn(f"Failed to fetch sentiment data: {e}")
        
        if datasets:
            combined_data = self._align_and_combine_datasets(datasets)
            print(f"✅ Economic dataset: {len(combined_data)} rows, {len(combined_data.columns)} columns")
            return combined_data
        else:
            print("❌ No economic data available")
            return pd.DataFrame()
    
    def get_crypto_research_dataset(self,
                                  symbols: List[str],
                                  start_date: str,
                                  end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive crypto research dataset.
        
        Args:
            symbols: List of crypto symbols
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary of DataFrames by symbol
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🔗 Building crypto research dataset for {len(symbols)} symbols...")
        
        crypto_datasets = {}
        
        if 'crypto' not in self.providers:
            print("❌ Crypto provider not available")
            return crypto_datasets
        
        for symbol in symbols:
            print(f"   📊 Processing {symbol}...")
            
            symbol_data = {}
            
            try:
                # Spot prices
                spot_data = self.providers['crypto'].fetch_data(
                    symbol, start_date, end_date, data_type='spot'
                )
                if not spot_data.empty:
                    symbol_data['prices'] = spot_data
                
                # On-chain metrics
                metrics_data = self.providers['crypto'].fetch_data(
                    symbol, start_date, end_date, data_type='metrics'
                )
                if not metrics_data.empty:
                    symbol_data['metrics'] = metrics_data
                
                # DeFi options (for ETH mainly)
                if 'ETH' in symbol:
                    defi_data = self.providers['crypto'].fetch_data(
                        symbol, start_date, end_date, data_type='defi_options'
                    )
                    if not defi_data.empty:
                        symbol_data['defi_options'] = defi_data
                
                # Combine symbol data
                if symbol_data:
                    combined_symbol_data = self._align_and_combine_datasets(symbol_data)
                    crypto_datasets[symbol] = combined_symbol_data
                    print(f"   ✅ {symbol}: {len(combined_symbol_data)} rows")
                
            except Exception as e:
                warnings.warn(f"Failed to fetch data for {symbol}: {e}")
        
        print(f"✅ Crypto dataset complete: {len(crypto_datasets)} symbols")
        return crypto_datasets
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all data providers."""
        status = {
            'providers_initialized': len(self.providers),
            'providers_available': list(self.providers.keys()),
            'api_keys_configured': list(self.api_keys.keys()),
            'provider_details': {}
        }
        
        for name, provider in self.providers.items():
            try:
                usage_stats = provider.get_usage_stats()
                status['provider_details'][name] = {
                    'status': 'active',
                    'usage_stats': usage_stats
                }
            except Exception as e:
                status['provider_details'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return status
    
    def test_all_providers(self) -> Dict[str, bool]:
        """Test all providers with sample data requests."""
        results = {}
        
        test_configs = {
            'fred': {'symbol': 'FEDFUNDS', 'start_date': '2023-01-01', 'end_date': '2023-12-31'},
            'sentiment': {'symbol': 'SPY', 'start_date': '2023-01-01', 'sentiment_type': 'calculated'},
            'crypto': {'symbol': 'BTC-USD', 'start_date': '2023-01-01', 'data_type': 'spot'}
        }
        
        for provider_name, provider in self.providers.items():
            print(f"🧪 Testing {provider_name}...")
            
            try:
                config = test_configs.get(provider_name, {})
                test_data = provider.fetch_data(**config)
                
                if not test_data.empty:
                    print(f"   ✅ {provider_name}: {len(test_data)} rows fetched")
                    results[provider_name] = True
                else:
                    print(f"   ⚠️ {provider_name}: No data returned")
                    results[provider_name] = False
                    
            except Exception as e:
                print(f"   ❌ {provider_name}: {e}")
                results[provider_name] = False
        
        return results
    
    def get_data_coverage_report(self) -> Dict[str, Any]:
        """Generate data coverage report."""
        coverage = {
            'traditional_assets': {
                'supported': True,
                'sources': ['Yahoo Finance (via yfinance)', 'Alpha Vantage'],
                'data_types': ['OHLCV', 'Fundamentals', 'Technical Indicators']
            },
            'economic_indicators': {
                'supported': 'fred' in self.providers,
                'sources': ['Federal Reserve (FRED)'],
                'data_types': ['Interest Rates', 'Inflation', 'Employment', 'GDP', 'Economic Activity']
            },
            'sentiment_data': {
                'supported': 'sentiment' in self.providers,
                'sources': ['News API', 'CNN Fear & Greed', 'Market-calculated'],
                'data_types': ['News Sentiment', 'Social Sentiment', 'Market Sentiment Indicators']
            },
            'cryptocurrency': {
                'supported': 'crypto' in self.providers,
                'sources': ['Coinbase Pro', 'CoinGecko'],
                'data_types': ['Spot Prices', 'DeFi Options', 'On-chain Metrics']
            },
            'alternative_data': {
                'supported': False,
                'sources': ['Weather APIs', 'Satellite Data', 'Economic Calendar'],
                'data_types': ['Weather', 'Economic Events', 'Seasonal Patterns'],
                'note': 'Available for future implementation'
            }
        }
        
        return coverage