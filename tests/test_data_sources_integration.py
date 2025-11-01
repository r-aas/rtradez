"""
Comprehensive tests for data sources integration.

Tests for data providers, data manager, and integration workflows.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests
import json

from rtradez.data_sources import (
    BaseProvider, DataManager,
    FredProvider, SentimentProvider, SpotPricesProvider
)


class TestBaseProvider:
    """Test BaseProvider functionality."""
    
    def test_base_provider_interface(self):
        """Test BaseProvider abstract interface."""
        # Should not be able to instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseProvider()
    
    def test_concrete_provider_implementation(self):
        """Test concrete provider implementation."""
        class TestProvider(BaseProvider):
            def __init__(self):
                super().__init__(
                    provider_name="Test Provider",
                    base_url="https://api.test.com",
                    api_key="test_key"
                )
            
            def fetch_data(self, symbol, start_date, end_date, **kwargs):
                return pd.DataFrame({
                    'date': pd.date_range(start_date, end_date, freq='D'),
                    'value': np.random.randn(10)
                })
            
            def validate_response(self, response):
                return response is not None
        
        provider = TestProvider()
        
        assert provider.provider_name == "Test Provider"
        assert provider.base_url == "https://api.test.com"
        assert provider.api_key == "test_key"
        
        # Test fetch_data implementation
        data = provider.fetch_data("TEST", "2023-01-01", "2023-01-10")
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        class RateLimitedProvider(BaseProvider):
            def __init__(self):
                super().__init__(
                    provider_name="Rate Limited",
                    rate_limit_calls=2,
                    rate_limit_period=1.0  # 2 calls per second
                )
            
            def fetch_data(self, symbol, start_date, end_date, **kwargs):
                return {"data": "test"}
            
            def validate_response(self, response):
                return True
        
        provider = RateLimitedProvider()
        
        # First two calls should succeed immediately
        start_time = datetime.now()
        provider.fetch_data("TEST1", "2023-01-01", "2023-01-02")
        provider.fetch_data("TEST2", "2023-01-01", "2023-01-02")
        
        # Third call should be rate limited
        provider.fetch_data("TEST3", "2023-01-01", "2023-01-02")
        end_time = datetime.now()
        
        # Should have taken at least the rate limit period
        elapsed = (end_time - start_time).total_seconds()
        assert elapsed >= 0.5  # Some delay expected
    
    def test_caching_functionality(self):
        """Test data caching functionality."""
        class CachedProvider(BaseProvider):
            def __init__(self):
                super().__init__(
                    provider_name="Cached Provider",
                    enable_cache=True,
                    cache_ttl=60  # 1 minute cache
                )
                self.fetch_count = 0
            
            def fetch_data(self, symbol, start_date, end_date, **kwargs):
                self.fetch_count += 1
                return pd.DataFrame({'value': [self.fetch_count]})
            
            def validate_response(self, response):
                return True
        
        provider = CachedProvider()
        
        # First fetch should hit the API
        data1 = provider.fetch_data("TEST", "2023-01-01", "2023-01-02")
        assert provider.fetch_count == 1
        
        # Second fetch should use cache
        data2 = provider.fetch_data("TEST", "2023-01-01", "2023-01-02")
        assert provider.fetch_count == 1  # No additional API call
        
        # Data should be the same
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_retry_mechanism(self):
        """Test automatic retry mechanism."""
        class UnreliableProvider(BaseProvider):
            def __init__(self):
                super().__init__(
                    provider_name="Unreliable Provider",
                    max_retries=3,
                    retry_delay=0.1
                )
                self.attempt_count = 0
            
            def fetch_data(self, symbol, start_date, end_date, **kwargs):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise requests.exceptions.RequestException("Simulated failure")
                return {"data": "success"}
            
            def validate_response(self, response):
                return True
        
        provider = UnreliableProvider()
        
        # Should succeed after retries
        result = provider.fetch_data("TEST", "2023-01-01", "2023-01-02")
        assert result == {"data": "success"}
        assert provider.attempt_count == 3


class TestFredProvider:
    """Test FRED economic data provider."""
    
    @pytest.fixture
    def fred_provider(self):
        """Create FRED provider instance."""
        return FredProvider(api_key="test_fred_key")
    
    def test_fred_provider_initialization(self, fred_provider):
        """Test FRED provider initialization."""
        assert fred_provider.provider_name == "FRED"
        assert fred_provider.api_key == "test_fred_key"
        assert "fred.stlouisfed.org" in fred_provider.base_url
    
    @patch('requests.get')
    def test_fetch_economic_indicator(self, mock_get, fred_provider):
        """Test fetching economic indicator data."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'observations': [
                {'date': '2023-01-01', 'value': '2.5'},
                {'date': '2023-01-02', 'value': '2.6'},
                {'date': '2023-01-03', 'value': '2.7'}
            ]
        }
        mock_get.return_value = mock_response
        
        data = fred_provider.fetch_economic_data(
            series_id="FEDFUNDS",
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert 'date' in data.columns
        assert 'value' in data.columns
        assert data['value'].dtype == float
    
    @patch('requests.get')
    def test_fred_api_error_handling(self, mock_get, fred_provider):
        """Test FRED API error handling."""
        # Mock API error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            'error_message': 'Bad Request'
        }
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception):
            fred_provider.fetch_economic_data(
                series_id="INVALID",
                start_date="2023-01-01",
                end_date="2023-01-03"
            )
    
    def test_fred_data_validation(self, fred_provider):
        """Test FRED data validation."""
        # Valid response
        valid_response = {
            'observations': [
                {'date': '2023-01-01', 'value': '2.5'}
            ]
        }
        assert fred_provider.validate_response(valid_response) == True
        
        # Invalid response
        invalid_response = {
            'error': 'Invalid series ID'
        }
        assert fred_provider.validate_response(invalid_response) == False
    
    def test_multiple_series_fetch(self, fred_provider):
        """Test fetching multiple economic series."""
        with patch('requests.get') as mock_get:
            # Mock responses for multiple series
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'observations': [
                    {'date': '2023-01-01', 'value': '2.5'},
                    {'date': '2023-01-02', 'value': '2.6'}
                ]
            }
            mock_get.return_value = mock_response
            
            series_list = ["FEDFUNDS", "UNRATE", "CPIAUCSL"]
            data = fred_provider.fetch_multiple_series(
                series_list,
                start_date="2023-01-01",
                end_date="2023-01-02"
            )
            
            assert isinstance(data, dict)
            assert len(data) == 3
            for series_id in series_list:
                assert series_id in data
                assert isinstance(data[series_id], pd.DataFrame)


class TestSentimentProvider:
    """Test sentiment data provider."""
    
    @pytest.fixture
    def sentiment_provider(self):
        """Create sentiment provider instance."""
        return SentimentProvider(
            api_key="test_sentiment_key",
            provider_type="news_api"
        )
    
    def test_sentiment_provider_initialization(self, sentiment_provider):
        """Test sentiment provider initialization."""
        assert sentiment_provider.provider_name == "Sentiment"
        assert sentiment_provider.api_key == "test_sentiment_key"
        assert sentiment_provider.provider_type == "news_api"
    
    @patch('requests.get')
    def test_fetch_news_sentiment(self, mock_get, sentiment_provider):
        """Test fetching news sentiment data."""
        # Mock news API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Positive market outlook',
                    'description': 'Markets show strong performance',
                    'publishedAt': '2023-01-01T10:00:00Z',
                    'sentiment_score': 0.8
                },
                {
                    'title': 'Economic concerns rise',
                    'description': 'Inflation worries impact markets',
                    'publishedAt': '2023-01-01T14:00:00Z',
                    'sentiment_score': -0.3
                }
            ]
        }
        mock_get.return_value = mock_response
        
        sentiment_data = sentiment_provider.fetch_sentiment_data(
            query="market outlook",
            start_date="2023-01-01",
            end_date="2023-01-01"
        )
        
        assert isinstance(sentiment_data, pd.DataFrame)
        assert len(sentiment_data) == 2
        assert 'sentiment_score' in sentiment_data.columns
        assert 'timestamp' in sentiment_data.columns
    
    def test_sentiment_score_calculation(self, sentiment_provider):
        """Test sentiment score calculation."""
        news_text = "The market showed excellent performance with strong gains across all sectors."
        
        # Mock sentiment analysis
        with patch.object(sentiment_provider, '_analyze_text_sentiment') as mock_analyze:
            mock_analyze.return_value = 0.85
            
            score = sentiment_provider.calculate_sentiment_score(news_text)
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0
    
    def test_aggregate_sentiment_by_period(self, sentiment_provider):
        """Test sentiment aggregation by time period."""
        # Sample sentiment data
        sentiment_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=24, freq='H'),
            'sentiment_score': np.random.uniform(-1, 1, 24),
            'volume': np.random.randint(1, 100, 24)
        })
        
        daily_sentiment = sentiment_provider.aggregate_sentiment(
            sentiment_data,
            frequency='daily'
        )
        
        assert isinstance(daily_sentiment, pd.DataFrame)
        assert len(daily_sentiment) == 1  # Should aggregate to 1 day
        assert 'avg_sentiment' in daily_sentiment.columns
        assert 'sentiment_volatility' in daily_sentiment.columns


class TestSpotPricesProvider:
    """Test spot prices provider."""
    
    @pytest.fixture
    def spot_provider(self):
        """Create spot prices provider instance."""
        return SpotPricesProvider(
            provider_type="cryptocurrency",
            api_key="test_crypto_key"
        )
    
    def test_spot_provider_initialization(self, spot_provider):
        """Test spot provider initialization."""
        assert spot_provider.provider_name == "SpotPrices"
        assert spot_provider.provider_type == "cryptocurrency"
        assert spot_provider.api_key == "test_crypto_key"
    
    @patch('requests.get')
    def test_fetch_crypto_prices(self, mock_get, spot_provider):
        """Test fetching cryptocurrency prices."""
        # Mock crypto API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'time_period_start': '2023-01-01T00:00:00.0000000Z',
                    'time_period_end': '2023-01-01T23:59:59.9999999Z',
                    'price_open': 45000.0,
                    'price_high': 46000.0,
                    'price_low': 44500.0,
                    'price_close': 45500.0,
                    'volume_traded': 1000000.0
                }
            ]
        }
        mock_get.return_value = mock_response
        
        price_data = spot_provider.fetch_spot_prices(
            symbol="BTC",
            start_date="2023-01-01",
            end_date="2023-01-01"
        )
        
        assert isinstance(price_data, pd.DataFrame)
        assert len(price_data) == 1
        assert 'open' in price_data.columns
        assert 'high' in price_data.columns
        assert 'low' in price_data.columns
        assert 'close' in price_data.columns
        assert 'volume' in price_data.columns
    
    def test_price_data_validation(self, spot_provider):
        """Test price data validation."""
        # Valid price data
        valid_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [1000, 1500]
        })
        
        assert spot_provider.validate_price_data(valid_data) == True
        
        # Invalid price data (high < low)
        invalid_data = pd.DataFrame({
            'open': [100.0],
            'high': [99.0],  # High less than low
            'low': [101.0],
            'close': [100.0],
            'volume': [1000]
        })
        
        assert spot_provider.validate_price_data(invalid_data) == False
    
    def test_calculate_price_metrics(self, spot_provider):
        """Test price metrics calculation."""
        price_data = pd.DataFrame({
            'open': [100.0, 102.0, 101.0],
            'high': [103.0, 105.0, 104.0],
            'low': [99.0, 101.0, 100.0],
            'close': [102.0, 103.0, 102.0],
            'volume': [1000, 1200, 1100]
        })
        
        metrics = spot_provider.calculate_price_metrics(price_data)
        
        assert isinstance(metrics, dict)
        assert 'volatility' in metrics
        assert 'avg_volume' in metrics
        assert 'price_change_pct' in metrics
        assert 'high_low_spread' in metrics


class TestDataManager:
    """Test DataManager orchestration."""
    
    @pytest.fixture
    def data_manager(self):
        """Create data manager instance."""
        return DataManager(
            cache_enabled=True,
            parallel_requests=True,
            max_workers=4
        )
    
    def test_data_manager_initialization(self, data_manager):
        """Test DataManager initialization."""
        assert data_manager.cache_enabled == True
        assert data_manager.parallel_requests == True
        assert data_manager.max_workers == 4
        assert len(data_manager.providers) == 0
    
    def test_register_provider(self, data_manager):
        """Test registering data providers."""
        fred_provider = FredProvider(api_key="test_key")
        data_manager.register_provider("fred", fred_provider)
        
        assert len(data_manager.providers) == 1
        assert "fred" in data_manager.providers
        assert data_manager.providers["fred"] == fred_provider
    
    def test_fetch_from_multiple_providers(self, data_manager):
        """Test fetching data from multiple providers."""
        # Create mock providers
        provider1 = Mock()
        provider1.fetch_data.return_value = pd.DataFrame({'value1': [1, 2, 3]})
        
        provider2 = Mock()
        provider2.fetch_data.return_value = pd.DataFrame({'value2': [4, 5, 6]})
        
        data_manager.register_provider("provider1", provider1)
        data_manager.register_provider("provider2", provider2)
        
        # Fetch from multiple providers
        requests = [
            {"provider": "provider1", "symbol": "TEST1", "start_date": "2023-01-01", "end_date": "2023-01-03"},
            {"provider": "provider2", "symbol": "TEST2", "start_date": "2023-01-01", "end_date": "2023-01-03"}
        ]
        
        results = data_manager.fetch_multiple(requests)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "provider1" in results
        assert "provider2" in results
        
        # Verify providers were called
        provider1.fetch_data.assert_called_once()
        provider2.fetch_data.assert_called_once()
    
    def test_data_alignment(self, data_manager):
        """Test data alignment across providers."""
        # Data with different date ranges
        data1 = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value1': [1, 2, 3, 4, 5]
        }).set_index('date')
        
        data2 = pd.DataFrame({
            'date': pd.date_range('2023-01-02', periods=4, freq='D'),
            'value2': [10, 20, 30, 40]
        }).set_index('date')
        
        aligned_data = data_manager.align_data([data1, data2])
        
        assert isinstance(aligned_data, pd.DataFrame)
        # Should align to common date range
        assert len(aligned_data) == 4  # Intersection of date ranges
        assert 'value1' in aligned_data.columns
        assert 'value2' in aligned_data.columns
    
    def test_data_quality_validation(self, data_manager):
        """Test data quality validation."""
        # Good quality data
        good_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': np.random.randn(10)
        })
        
        quality_report = data_manager.validate_data_quality(good_data)
        
        assert isinstance(quality_report, dict)
        assert 'completeness' in quality_report
        assert 'consistency' in quality_report
        assert 'accuracy' in quality_report
        assert quality_report['completeness'] > 0.9  # Should be high quality
        
        # Poor quality data (with missing values)
        poor_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': [1, np.nan, 3, np.nan, 5, np.nan, 7, 8, 9, np.nan]
        })
        
        poor_quality_report = data_manager.validate_data_quality(poor_data)
        assert poor_quality_report['completeness'] < 0.8  # Should be poor quality
    
    def test_error_handling_and_fallback(self, data_manager):
        """Test error handling and fallback mechanisms."""
        # Primary provider that fails
        failing_provider = Mock()
        failing_provider.fetch_data.side_effect = Exception("API Error")
        
        # Fallback provider that succeeds
        fallback_provider = Mock()
        fallback_provider.fetch_data.return_value = pd.DataFrame({'value': [1, 2, 3]})
        
        data_manager.register_provider("primary", failing_provider)
        data_manager.register_provider("fallback", fallback_provider)
        
        # Configure fallback
        data_manager.set_fallback_provider("primary", "fallback")
        
        # Fetch should use fallback when primary fails
        result = data_manager.fetch_with_fallback(
            "primary",
            symbol="TEST",
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Verify fallback was used
        fallback_provider.fetch_data.assert_called_once()
    
    def test_caching_across_providers(self, data_manager):
        """Test caching mechanism across providers."""
        provider = Mock()
        provider.fetch_data.return_value = pd.DataFrame({'value': [1, 2, 3]})
        
        data_manager.register_provider("cached_provider", provider)
        
        # First call should hit provider
        result1 = data_manager.fetch_data(
            "cached_provider",
            symbol="TEST",
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        # Second call should use cache
        result2 = data_manager.fetch_data(
            "cached_provider",
            symbol="TEST",
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        # Provider should only be called once
        provider.fetch_data.assert_called_once()
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


@pytest.mark.integration
class TestDataSourcesIntegration:
    """Integration tests for data sources."""
    
    def test_multi_provider_data_pipeline(self):
        """Test complete multi-provider data pipeline."""
        # Create data manager
        manager = DataManager(cache_enabled=True, parallel_requests=True)
        
        # Mock providers
        economic_provider = Mock()
        economic_provider.fetch_data.return_value = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'interest_rate': [2.5, 2.6, 2.7, 2.8, 2.9]
        }).set_index('date')
        
        sentiment_provider = Mock()
        sentiment_provider.fetch_data.return_value = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'sentiment_score': [0.1, 0.3, -0.2, 0.5, 0.8]
        }).set_index('date')
        
        price_provider = Mock()
        price_provider.fetch_data.return_value = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'close_price': [100, 102, 101, 105, 107]
        }).set_index('date')
        
        # Register providers
        manager.register_provider("economic", economic_provider)
        manager.register_provider("sentiment", sentiment_provider)
        manager.register_provider("prices", price_provider)
        
        # Fetch and combine data
        requests = [
            {"provider": "economic", "symbol": "FEDFUNDS", "start_date": "2023-01-01", "end_date": "2023-01-05"},
            {"provider": "sentiment", "symbol": "market", "start_date": "2023-01-01", "end_date": "2023-01-05"},
            {"provider": "prices", "symbol": "SPY", "start_date": "2023-01-01", "end_date": "2023-01-05"}
        ]
        
        results = manager.fetch_multiple(requests)
        
        # Verify all data sources returned results
        assert len(results) == 3
        for provider_name, data in results.items():
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 5
        
        # Align all data sources
        aligned_data = manager.align_data(list(results.values()))
        
        assert isinstance(aligned_data, pd.DataFrame)
        assert len(aligned_data) == 5
        assert 'interest_rate' in aligned_data.columns
        assert 'sentiment_score' in aligned_data.columns
        assert 'close_price' in aligned_data.columns
        
        # Validate data quality
        quality_report = manager.validate_data_quality(aligned_data)
        assert quality_report['completeness'] > 0.9
        assert quality_report['consistency'] > 0.9
    
    def test_real_time_data_streaming(self):
        """Test real-time data streaming capabilities."""
        # Mock streaming provider
        streaming_provider = Mock()
        
        def mock_stream_data():
            """Mock streaming data generator."""
            for i in range(5):
                yield {
                    'timestamp': datetime.now(),
                    'symbol': 'BTC',
                    'price': 45000 + i * 100,
                    'volume': 1000 + i * 50
                }
        
        streaming_provider.stream_data.return_value = mock_stream_data()
        
        manager = DataManager()
        manager.register_provider("streaming", streaming_provider)
        
        # Collect streaming data
        stream_data = []
        for data_point in manager.stream_data("streaming", symbol="BTC"):
            stream_data.append(data_point)
            if len(stream_data) >= 5:
                break
        
        assert len(stream_data) == 5
        for point in stream_data:
            assert 'timestamp' in point
            assert 'symbol' in point
            assert 'price' in point
            assert 'volume' in point
    
    def test_data_source_failover_scenario(self):
        """Test data source failover scenarios."""
        manager = DataManager()
        
        # Primary provider (fails)
        primary = Mock()
        primary.fetch_data.side_effect = requests.exceptions.ConnectionError("Network error")
        
        # Secondary provider (succeeds)
        secondary = Mock()
        secondary.fetch_data.return_value = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3, freq='D'),
            'value': [1, 2, 3]
        })
        
        # Tertiary provider (also succeeds)
        tertiary = Mock()
        tertiary.fetch_data.return_value = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3, freq='D'),
            'value': [4, 5, 6]
        })
        
        manager.register_provider("primary", primary)
        manager.register_provider("secondary", secondary)
        manager.register_provider("tertiary", tertiary)
        
        # Configure failover chain
        manager.set_failover_chain("primary", ["secondary", "tertiary"])
        
        # Should automatically failover to secondary
        result = manager.fetch_with_failover(
            "primary",
            symbol="TEST",
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Verify failover was used
        secondary.fetch_data.assert_called_once()
        tertiary.fetch_data.assert_not_called()  # Should not reach tertiary
    
    def test_data_transformation_pipeline(self):
        """Test data transformation and enrichment pipeline."""
        manager = DataManager()
        
        # Raw data provider
        raw_provider = Mock()
        raw_provider.fetch_data.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
            'price': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 2000, 10)
        })
        
        manager.register_provider("raw_data", raw_provider)
        
        # Define transformation pipeline
        def calculate_returns(data):
            data['returns'] = data['price'].pct_change()
            return data
        
        def calculate_vwap(data):
            data['vwap'] = (data['price'] * data['volume']).cumsum() / data['volume'].cumsum()
            return data
        
        def add_technical_indicators(data):
            data['sma_5'] = data['price'].rolling(5).mean()
            data['volatility'] = data['returns'].rolling(5).std()
            return data
        
        transformations = [calculate_returns, calculate_vwap, add_technical_indicators]
        
        # Fetch and transform data
        raw_data = manager.fetch_data("raw_data", symbol="TEST")
        
        # Apply transformations
        transformed_data = raw_data.copy()
        for transform in transformations:
            transformed_data = transform(transformed_data)
        
        # Verify transformations
        assert 'returns' in transformed_data.columns
        assert 'vwap' in transformed_data.columns
        assert 'sma_5' in transformed_data.columns
        assert 'volatility' in transformed_data.columns
        
        # Validate data quality after transformation
        quality_report = manager.validate_data_quality(transformed_data)
        assert quality_report['completeness'] > 0.7  # Some NaN values expected from indicators