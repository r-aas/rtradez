"""Financial sentiment data integration from multiple sources."""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings
import re

from ..base_provider import BaseDataProvider, DataSourceConfig


class SentimentProvider(BaseDataProvider):
    """
    Financial sentiment data provider.
    
    Integrates multiple sentiment sources:
    - News API for financial news sentiment
    - Reddit sentiment (via PRAW)
    - Fear & Greed Index
    - VIX-based sentiment indicators
    - Alpha Vantage sentiment data
    """
    
    def __init__(self, config: DataSourceConfig):
        """Initialize sentiment provider."""
        super().__init__(config)
        
        # Sentiment indicators we can calculate
        self.calculated_indicators = {
            'vix_sentiment': 'VIX-based market sentiment',
            'put_call_ratio': 'Put/Call ratio sentiment',
            'high_low_index': 'New highs vs new lows sentiment',
            'advance_decline': 'Advance/Decline ratio sentiment'
        }
        
        # Fear & Greed Index (CNN, free)
        self.fear_greed_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    
    def fetch_data(self, 
                   symbol: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   sentiment_type: str = 'news',
                   **kwargs) -> pd.DataFrame:
        """
        Fetch sentiment data.
        
        Args:
            symbol: Asset symbol for sentiment analysis
            start_date: Start date
            end_date: End date
            sentiment_type: Type of sentiment ('news', 'social', 'fear_greed', 'calculated')
            **kwargs: Additional parameters
        
        Returns:
            DataFrame with sentiment data
        """
        if sentiment_type == 'news':
            return self._fetch_news_sentiment(symbol, start_date, end_date, **kwargs)
        elif sentiment_type == 'fear_greed':
            return self._fetch_fear_greed_index()
        elif sentiment_type == 'calculated':
            return self._calculate_market_sentiment(symbol, start_date, end_date)
        elif sentiment_type == 'social':
            return self._fetch_social_sentiment(symbol, start_date, end_date, **kwargs)
        else:
            raise ValueError(f"Unknown sentiment type: {sentiment_type}")
    
    def _fetch_news_sentiment(self, 
                            symbol: Optional[str],
                            start_date: Optional[str],
                            end_date: Optional[str],
                            **kwargs) -> pd.DataFrame:
        """Fetch news sentiment using News API or Alpha Vantage."""
        if self.config.provider_name == 'News API':
            return self._fetch_newsapi_sentiment(symbol, start_date, end_date, **kwargs)
        elif self.config.provider_name == 'Alpha Vantage':
            return self._fetch_alpha_vantage_sentiment(symbol, **kwargs)
        else:
            # Fallback to synthetic sentiment for demo
            return self._generate_synthetic_sentiment(symbol, start_date, end_date)
    
    def _fetch_newsapi_sentiment(self,
                               symbol: Optional[str],
                               start_date: Optional[str], 
                               end_date: Optional[str],
                               **kwargs) -> pd.DataFrame:
        """Fetch news sentiment from News API."""
        if not self.api_key:
            warnings.warn("News API key required for real sentiment data")
            return self._generate_synthetic_sentiment(symbol, start_date, end_date)
        
        self.check_rate_limit()
        
        # Build search query
        query = symbol if symbol else 'stock market finance'
        
        url = f"{self.base_url}/everything"
        params = {
            'q': query,
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100
        }
        
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                return pd.DataFrame()
            
            # Process articles for sentiment
            sentiment_data = []
            for article in articles:
                published_at = pd.to_datetime(article['publishedAt'])
                title = article.get('title', '')
                description = article.get('description', '')
                
                # Simple sentiment analysis (in practice, use proper NLP)
                sentiment_score = self._analyze_text_sentiment(title + ' ' + description)
                
                sentiment_data.append({
                    'date': published_at,
                    'sentiment_score': sentiment_score,
                    'title': title,
                    'source': article.get('source', {}).get('name', 'Unknown')
                })
            
            df = pd.DataFrame(sentiment_data)
            if df.empty:
                return df
            
            # Aggregate by date
            daily_sentiment = df.groupby(df['date'].dt.date).agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).round(3)
            
            daily_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'news_count']
            daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
            
            return self.standardize_data(daily_sentiment)
            
        except Exception as e:
            warnings.warn(f"Failed to fetch news sentiment: {e}")
            return self._generate_synthetic_sentiment(symbol, start_date, end_date)
    
    def _fetch_fear_greed_index(self) -> pd.DataFrame:
        """Fetch CNN Fear & Greed Index (free)."""
        try:
            response = requests.get(self.fear_greed_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            fear_greed_data = data.get('fear_and_greed_historical', {}).get('data', [])
            
            if not fear_greed_data:
                return pd.DataFrame()
            
            df_data = []
            for item in fear_greed_data:
                date = pd.to_datetime(item['x'], unit='ms')
                value = float(item['y'])
                
                # Categorize sentiment
                if value <= 25:
                    category = 'Extreme Fear'
                elif value <= 45:
                    category = 'Fear'
                elif value <= 55:
                    category = 'Neutral'
                elif value <= 75:
                    category = 'Greed'
                else:
                    category = 'Extreme Greed'
                
                df_data.append({
                    'date': date,
                    'fear_greed_index': value,
                    'sentiment_category': category,
                    'sentiment_normalized': (value - 50) / 50  # Normalize to [-1, 1]
                })
            
            df = pd.DataFrame(df_data)
            df = df.set_index('date').sort_index()
            
            return self.standardize_data(df)
            
        except Exception as e:
            warnings.warn(f"Failed to fetch Fear & Greed Index: {e}")
            return pd.DataFrame()
    
    def _calculate_market_sentiment(self,
                                  symbol: Optional[str],
                                  start_date: Optional[str],
                                  end_date: Optional[str]) -> pd.DataFrame:
        """Calculate sentiment indicators from market data."""
        # This would integrate with market data to calculate sentiment
        # For demo, generate synthetic calculated sentiment
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Business days
        
        np.random.seed(42)
        
        # Simulate VIX-based sentiment
        vix_levels = np.random.lognormal(3.0, 0.3, len(dates))
        vix_sentiment = np.where(vix_levels < 20, 1,    # Low VIX = bullish
                                np.where(vix_levels > 30, -1, 0))  # High VIX = bearish
        
        # Simulate put/call ratio
        put_call_ratio = np.random.lognormal(0, 0.2, len(dates))
        pc_sentiment = np.where(put_call_ratio < 0.8, 1,    # Low P/C = bullish
                               np.where(put_call_ratio > 1.2, -1, 0))  # High P/C = bearish
        
        # Combined sentiment
        combined_sentiment = (vix_sentiment + pc_sentiment) / 2
        
        df = pd.DataFrame({
            'date': dates,
            'vix_level': vix_levels,
            'vix_sentiment': vix_sentiment,
            'put_call_ratio': put_call_ratio,
            'put_call_sentiment': pc_sentiment,
            'combined_sentiment': combined_sentiment,
            'sentiment_strength': np.abs(combined_sentiment)
        })
        
        df = df.set_index('date')
        return self.standardize_data(df)
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis of text.
        
        In production, use proper NLP libraries like:
        - VADER sentiment
        - TextBlob
        - spaCy with sentiment models
        - Hugging Face transformers
        """
        if not text:
            return 0.0
        
        text = text.lower()
        
        # Simple positive/negative word lists
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit',
            'bullish', 'optimistic', 'strong', 'growth', 'increase', 'rally', 'surge'
        ]
        
        negative_words = [
            'bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'decline',
            'bearish', 'pessimistic', 'weak', 'decrease', 'crash', 'plunge', 'drop'
        ]
        
        # Count words
        words = re.findall(r'\b\w+\b', text)
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        # Normalize to [-1, 1] range
        sentiment = (positive_count - negative_count) / total_count
        return sentiment
    
    def _generate_synthetic_sentiment(self,
                                    symbol: Optional[str],
                                    start_date: Optional[str],
                                    end_date: Optional[str]) -> pd.DataFrame:
        """Generate synthetic sentiment data for demo purposes."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(symbol or 'market') % 2**32)
        
        # Generate realistic sentiment patterns
        base_sentiment = np.random.normal(0, 0.3, len(dates))
        
        # Add some autocorrelation (sentiment persistence)
        for i in range(1, len(base_sentiment)):
            base_sentiment[i] = 0.7 * base_sentiment[i-1] + 0.3 * base_sentiment[i]
        
        # Add some extreme events
        extreme_events = np.random.choice(len(dates), size=int(len(dates) * 0.05), replace=False)
        base_sentiment[extreme_events] += np.random.choice([-2, 2], size=len(extreme_events))
        
        # Clip to reasonable range
        sentiment_scores = np.clip(base_sentiment, -1, 1)
        
        df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'sentiment_magnitude': np.abs(sentiment_scores),
            'news_count': np.random.poisson(10, len(dates)),
            'data_source': 'synthetic'
        })
        
        df = df.set_index('date')
        return self.standardize_data(df)
    
    def get_sentiment_summary(self,
                            symbol: Optional[str] = None,
                            days: int = 30) -> Dict[str, Any]:
        """
        Get sentiment summary for recent period.
        
        Args:
            symbol: Asset symbol
            days: Number of days to analyze
        
        Returns:
            Sentiment summary statistics
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Get multiple sentiment types
        sentiment_data = {}
        
        for sentiment_type in ['news', 'fear_greed', 'calculated']:
            try:
                data = self.fetch_data(symbol, start_date, end_date, 
                                     sentiment_type=sentiment_type)
                if not data.empty:
                    sentiment_data[sentiment_type] = data
            except Exception as e:
                warnings.warn(f"Failed to fetch {sentiment_type} sentiment: {e}")
        
        # Summarize results
        summary = {
            'symbol': symbol or 'market',
            'period_days': days,
            'data_sources': list(sentiment_data.keys()),
            'sentiment_indicators': {}
        }
        
        for source, data in sentiment_data.items():
            if source == 'news' and 'sentiment_score' in data.columns:
                summary['sentiment_indicators'][f'{source}_avg'] = data['sentiment_score'].mean()
                summary['sentiment_indicators'][f'{source}_recent'] = data['sentiment_score'].iloc[-1]
            elif source == 'fear_greed' and 'sentiment_normalized' in data.columns:
                summary['sentiment_indicators'][f'{source}_avg'] = data['sentiment_normalized'].mean()
                summary['sentiment_indicators'][f'{source}_recent'] = data['sentiment_normalized'].iloc[-1]
            elif source == 'calculated' and 'combined_sentiment' in data.columns:
                summary['sentiment_indicators'][f'{source}_avg'] = data['combined_sentiment'].mean()
                summary['sentiment_indicators'][f'{source}_recent'] = data['combined_sentiment'].iloc[-1]
        
        return summary
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with sentiment data."""
        return ['SPY', 'QQQ', 'IWM', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'market']
    
    def get_data_description(self, symbol: str) -> Dict[str, Any]:
        """Get description of sentiment data."""
        return {
            'symbol': symbol,
            'data_types': ['news_sentiment', 'fear_greed_index', 'calculated_indicators'],
            'description': 'Financial sentiment data from multiple sources',
            'sentiment_range': '[-1, 1] where -1 is very negative, 1 is very positive',
            'update_frequency': 'Daily',
            'sources': ['News API', 'CNN Fear & Greed', 'Market-calculated indicators']
        }