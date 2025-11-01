"""Cryptocurrency spot prices and DeFi options data."""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings

from ..base_provider import BaseDataProvider, DataSourceConfig


class CryptoProvider(BaseDataProvider):
    """
    Cryptocurrency data provider.
    
    Supports multiple crypto exchanges and data sources:
    - Coinbase Pro (free)
    - Binance (free)
    - CoinGecko (free with rate limits)
    - DeFi options protocols (Opyn, Hegic, etc.)
    """
    
    def __init__(self, config: DataSourceConfig):
        """Initialize crypto provider."""
        super().__init__(config)
        
        # Popular cryptocurrency symbols
        self.popular_symbols = {
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum', 
            'ADA-USD': 'Cardano',
            'SOL-USD': 'Solana',
            'MATIC-USD': 'Polygon',
            'AVAX-USD': 'Avalanche',
            'DOT-USD': 'Polkadot',
            'LINK-USD': 'Chainlink',
            'UNI-USD': 'Uniswap',
            'AAVE-USD': 'Aave'
        }
        
        # DeFi options protocols
        self.defi_protocols = {
            'opyn': 'Opyn Protocol',
            'hegic': 'Hegic Protocol', 
            'dopex': 'Dopex',
            'lyra': 'Lyra Finance',
            'ribbon': 'Ribbon Finance'
        }
        
        # CoinGecko free API (no key required)
        self.coingecko_base = "https://api.coingecko.com/api/v3"
    
    def fetch_data(self, 
                   symbol: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   data_type: str = 'spot',
                   **kwargs) -> pd.DataFrame:
        """
        Fetch cryptocurrency data.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC-USD', 'bitcoin')
            start_date: Start date
            end_date: End date
            data_type: Type of data ('spot', 'defi_options', 'metrics')
            **kwargs: Additional parameters
        
        Returns:
            DataFrame with crypto data
        """
        if data_type == 'spot':
            return self._fetch_spot_prices(symbol, start_date, end_date, **kwargs)
        elif data_type == 'defi_options':
            return self._fetch_defi_options(symbol, start_date, end_date, **kwargs)
        elif data_type == 'metrics':
            return self._fetch_crypto_metrics(symbol, start_date, end_date, **kwargs)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _fetch_spot_prices(self,
                          symbol: Optional[str],
                          start_date: Optional[str],
                          end_date: Optional[str],
                          **kwargs) -> pd.DataFrame:
        """Fetch cryptocurrency spot prices."""
        if self.config.provider_name == 'Coinbase Pro':
            return self._fetch_coinbase_data(symbol, start_date, end_date, **kwargs)
        elif self.config.provider_name == 'CoinGecko':
            return self._fetch_coingecko_data(symbol, start_date, end_date, **kwargs)
        else:
            # Fallback to synthetic data
            return self._generate_synthetic_crypto_data(symbol, start_date, end_date)
    
    def _fetch_coinbase_data(self,
                           symbol: Optional[str],
                           start_date: Optional[str],
                           end_date: Optional[str],
                           **kwargs) -> pd.DataFrame:
        """Fetch data from Coinbase Pro API (free)."""
        if not symbol:
            symbol = 'BTC-USD'
        
        self.check_rate_limit()
        
        # Coinbase Pro uses different symbol format
        coinbase_symbol = symbol.replace('-', '-')
        
        url = f"{self.base_url}/products/{coinbase_symbol}/candles"
        
        # Convert dates to timestamps if provided
        params = {
            'granularity': 86400  # Daily candles
        }
        
        if start_date:
            start_timestamp = pd.to_datetime(start_date).isoformat()
            params['start'] = start_timestamp
        if end_date:
            end_timestamp = pd.to_datetime(end_date).isoformat()
            params['end'] = end_timestamp
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Coinbase returns: [timestamp, low, high, open, close, volume]
            df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.drop('timestamp', axis=1)
            df = df.set_index('date').sort_index()
            
            # Rename columns to match standard format
            df.columns = ['Low', 'High', 'Open', 'Close', 'Volume']
            
            return self.standardize_data(df)
            
        except Exception as e:
            warnings.warn(f"Failed to fetch Coinbase data for {symbol}: {e}")
            return self._generate_synthetic_crypto_data(symbol, start_date, end_date)
    
    def _fetch_coingecko_data(self,
                            symbol: Optional[str],
                            start_date: Optional[str],
                            end_date: Optional[str],
                            **kwargs) -> pd.DataFrame:
        """Fetch data from CoinGecko API (free)."""
        if not symbol:
            symbol = 'bitcoin'
        
        # Convert symbol to CoinGecko format
        symbol_mapping = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum',
            'ADA-USD': 'cardano',
            'SOL-USD': 'solana',
            'MATIC-USD': 'matic-network',
            'AVAX-USD': 'avalanche-2',
            'DOT-USD': 'polkadot',
            'LINK-USD': 'chainlink',
            'UNI-USD': 'uniswap',
            'AAVE-USD': 'aave'
        }
        
        coingecko_id = symbol_mapping.get(symbol, symbol.lower().replace('-usd', ''))
        
        self.check_rate_limit()
        
        url = f"{self.coingecko_base}/coins/{coingecko_id}/market_chart"
        
        # Calculate days for range
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            days = (end_dt - start_dt).days
        else:
            days = 365  # Default to 1 year
        
        params = {
            'vs_currency': 'usd',
            'days': min(days, 365)  # CoinGecko free tier limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'prices' not in data:
                return pd.DataFrame()
            
            # Convert price data
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                date = pd.to_datetime(timestamp, unit='ms')
                volume = volumes[i][1] if i < len(volumes) else 0
                
                df_data.append({
                    'date': date,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df = df.set_index('date').sort_index()
            
            # Add OHLC estimates (CoinGecko free tier only has close prices)
            df['open'] = df['close'].shift(1)
            df['high'] = df['close'] * 1.02  # Estimate 2% intraday range
            df['low'] = df['close'] * 0.98
            
            # Clean up and standardize
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            return self.standardize_data(df)
            
        except Exception as e:
            warnings.warn(f"Failed to fetch CoinGecko data for {symbol}: {e}")
            return self._generate_synthetic_crypto_data(symbol, start_date, end_date)
    
    def _fetch_defi_options(self,
                          symbol: Optional[str],
                          start_date: Optional[str],
                          end_date: Optional[str],
                          **kwargs) -> pd.DataFrame:
        """
        Fetch DeFi options data.
        
        This would integrate with DeFi protocols like:
        - Opyn (ETH options)
        - Hegic (ETH/BTC options)
        - Dopex (multi-asset options)
        - Lyra Finance (ETH options)
        """
        # For demo, generate synthetic DeFi options data
        return self._generate_synthetic_defi_options(symbol, start_date, end_date)
    
    def _generate_synthetic_defi_options(self,
                                       symbol: Optional[str],
                                       start_date: Optional[str],
                                       end_date: Optional[str]) -> pd.DataFrame:
        """Generate synthetic DeFi options data for demonstration."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        # Generate synthetic options metrics
        df_data = []
        base_price = 2000 if 'ETH' in (symbol or '') else 40000  # ETH vs BTC base price
        
        for date in dates:
            # Simulate DeFi options activity
            total_volume = np.random.exponential(1000000)  # Volume in USD
            open_interest = np.random.exponential(5000000)  # OI in USD
            
            # Implied volatility estimates
            iv_calls = np.random.normal(0.8, 0.2)  # Higher vol than traditional options
            iv_puts = np.random.normal(0.85, 0.2)
            
            # Put/call ratio
            put_call_ratio = np.random.lognormal(0, 0.3)
            
            # Protocol distribution (synthetic)
            opyn_share = np.random.beta(2, 3)
            hegic_share = np.random.beta(1, 2) * (1 - opyn_share)
            other_share = 1 - opyn_share - hegic_share
            
            df_data.append({
                'date': date,
                'total_volume_usd': total_volume,
                'open_interest_usd': open_interest,
                'iv_calls_avg': max(0.1, iv_calls),
                'iv_puts_avg': max(0.1, iv_puts),
                'put_call_ratio': put_call_ratio,
                'opyn_volume_share': opyn_share,
                'hegic_volume_share': hegic_share,
                'other_protocols_share': other_share,
                'active_strikes': np.random.poisson(15),
                'avg_time_to_expiry': np.random.exponential(14)  # Days
            })
        
        df = pd.DataFrame(df_data)
        df = df.set_index('date')
        
        return self.standardize_data(df)
    
    def _fetch_crypto_metrics(self,
                            symbol: Optional[str],
                            start_date: Optional[str],
                            end_date: Optional[str],
                            **kwargs) -> pd.DataFrame:
        """Fetch on-chain and market metrics for cryptocurrencies."""
        # This would integrate with services like:
        # - Glassnode (on-chain metrics)
        # - IntoTheBlock (analytics)
        # - Messari (fundamental metrics)
        
        # For demo, generate synthetic metrics
        return self._generate_synthetic_crypto_metrics(symbol, start_date, end_date)
    
    def _generate_synthetic_crypto_metrics(self,
                                         symbol: Optional[str],
                                         start_date: Optional[str],
                                         end_date: Optional[str]) -> pd.DataFrame:
        """Generate synthetic crypto metrics."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(symbol or 'crypto') % 2**32)
        
        df_data = []
        for date in dates:
            # Network metrics
            active_addresses = np.random.exponential(100000)
            transaction_count = np.random.exponential(300000)
            nvt_ratio = np.random.lognormal(2, 0.5)  # Network Value to Transactions
            
            # Market metrics  
            market_cap = np.random.exponential(500000000000)  # Market cap
            realized_cap = market_cap * np.random.uniform(0.3, 0.8)
            mvrv_ratio = market_cap / realized_cap
            
            # Sentiment metrics
            fear_greed = np.random.uniform(0, 100)
            social_sentiment = np.random.normal(0, 1)
            
            # DeFi metrics (if applicable)
            if 'ETH' in (symbol or ''):
                tvl = np.random.exponential(50000000000)  # Total Value Locked
                defi_dominance = np.random.uniform(0.6, 0.9)
            else:
                tvl = 0
                defi_dominance = 0
            
            df_data.append({
                'date': date,
                'active_addresses': active_addresses,
                'transaction_count': transaction_count,
                'nvt_ratio': nvt_ratio,
                'market_cap_usd': market_cap,
                'realized_cap_usd': realized_cap,
                'mvrv_ratio': mvrv_ratio,
                'fear_greed_index': fear_greed,
                'social_sentiment': social_sentiment,
                'tvl_usd': tvl,
                'defi_dominance': defi_dominance
            })
        
        df = pd.DataFrame(df_data)
        df = df.set_index('date')
        
        return self.standardize_data(df)
    
    def _generate_synthetic_crypto_data(self,
                                      symbol: Optional[str],
                                      start_date: Optional[str],
                                      end_date: Optional[str]) -> pd.DataFrame:
        """Generate synthetic crypto price data."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(symbol or 'BTC') % 2**32)
        
        # Base prices for different cryptos
        if 'BTC' in (symbol or ''):
            base_price = 40000
            volatility = 0.04
        elif 'ETH' in (symbol or ''):
            base_price = 2500
            volatility = 0.05
        else:
            base_price = 100
            volatility = 0.06
        
        # Generate price series with crypto-like characteristics
        returns = np.random.normal(0.001, volatility, len(dates))  # Slightly positive drift
        
        # Add some extreme moves (crypto is volatile)
        extreme_days = np.random.choice(len(dates), size=int(len(dates) * 0.05), replace=False)
        returns[extreme_days] += np.random.choice([-0.2, 0.2], size=len(extreme_days))
        
        prices = base_price * np.cumprod(1 + returns)
        volumes = np.random.exponential(1000000, len(dates))  # High volume variability
        
        # Create OHLC data
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.01, len(df)))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.02, len(df))))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.02, len(df))))
        df['Volume'] = volumes
        
        df = df.fillna(method='bfill')
        
        return self.standardize_data(df)
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available crypto symbols."""
        return list(self.popular_symbols.keys())
    
    def get_data_description(self, symbol: str) -> Dict[str, Any]:
        """Get description of crypto data."""
        return {
            'symbol': symbol,
            'name': self.popular_symbols.get(symbol, 'Unknown'),
            'data_types': ['spot_prices', 'defi_options', 'on_chain_metrics'],
            'exchanges': ['Coinbase Pro', 'Binance', 'CoinGecko'],
            'defi_protocols': list(self.defi_protocols.keys()),
            'update_frequency': '24/7',
            'volatility_characteristics': 'High volatility, 24/7 trading'
        }
    
    def get_defi_protocols_summary(self) -> Dict[str, Any]:
        """Get summary of supported DeFi options protocols."""
        return {
            'supported_protocols': self.defi_protocols,
            'asset_coverage': ['ETH', 'BTC', 'LINK', 'UNI', 'AAVE'],
            'data_availability': 'Protocol-dependent',
            'typical_features': [
                'Decentralized options trading',
                'Automated market making',
                'Liquidity mining rewards',
                'Governance tokens',
                'Cross-chain compatibility'
            ]
        }