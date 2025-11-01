"""Federal Reserve Economic Data (FRED) integration."""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings

from ..base_provider import BaseDataProvider, DataSourceConfig


class FREDProvider(BaseDataProvider):
    """
    Federal Reserve Economic Data provider.
    
    Provides access to 800,000+ economic time series from:
    - Federal Reserve
    - Bureau of Labor Statistics  
    - Census Bureau
    - Other government agencies
    
    Free API with registration required.
    """
    
    def __init__(self, config: DataSourceConfig):
        """Initialize FRED provider."""
        super().__init__(config)
        
        # Popular economic indicators
        self.popular_series = {
            # Interest Rates & Monetary Policy
            'FEDFUNDS': 'Effective Federal Funds Rate',
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'DGS2': '2-Year Treasury Constant Maturity Rate',
            'DGS3MO': '3-Month Treasury Constant Maturity Rate',
            'T10Y2Y': '10-Year Treasury Minus 2-Year Treasury',
            'T10Y3M': '10-Year Treasury Minus 3-Month Treasury',
            
            # Inflation
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
            'CPILFESL': 'Core CPI (Less Food & Energy)',
            'DFEDTARU': 'Federal Reserve Target Rate - Upper Limit',
            'T5YIE': '5-Year Breakeven Inflation Rate',
            
            # Employment
            'UNRATE': 'Unemployment Rate',
            'PAYEMS': 'All Employees: Total Nonfarm Payrolls',
            'CIVPART': 'Labor Force Participation Rate',
            'EMRATIO': 'Employment-Population Ratio',
            
            # Economic Activity
            'GDP': 'Gross Domestic Product',
            'GDPC1': 'Real Gross Domestic Product',
            'INDPRO': 'Industrial Production Index',
            'HOUST': 'Housing Starts',
            'RRSFS': 'Real Retail Sales',
            
            # Financial Markets
            'DEXUSEU': 'US/Euro Foreign Exchange Rate',
            'DEXJPUS': 'Japan/US Foreign Exchange Rate',
            'DEXCHUS': 'China/US Foreign Exchange Rate',
            'VIXCLS': 'CBOE Volatility Index: VIX',
            'WILL5000IND': 'Wilshire 5000 Total Market Index',
            
            # Credit & Banking
            'TOTALSL': 'Total Consumer Loans and Leases Outstanding',
            'DRSFRMACBS': 'Delinquency Rate on Single-Family Residential Mortgages',
            'DRTSCILM': 'Delinquency Rate on Credit Card Loans',
            
            # International
            'BOGZ1FL893164005Q': 'Rest of World; Corporate Equities; Asset',
            'USSTHPI': 'US Total Housing Price Index'
        }
    
    def fetch_data(self, 
                   symbol: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   series_id: Optional[str] = None,
                   **kwargs) -> pd.DataFrame:
        """
        Fetch economic data from FRED.
        
        Args:
            symbol: FRED series ID (e.g., 'FEDFUNDS', 'GDP')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            series_id: Alternative to symbol
            **kwargs: Additional FRED API parameters
        
        Returns:
            DataFrame with economic data
        """
        self.check_rate_limit()
        
        series_id = series_id or symbol
        if not series_id:
            raise ValueError("Must provide series_id or symbol")
        
        if not self.api_key:
            raise ValueError("FRED API key required. Get free key at: https://fred.stlouisfed.org/docs/api/")
        
        # Build request URL
        url = f"{self.base_url}/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        
        # Add any additional parameters
        params.update(kwargs)
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'observations' not in data:
                warnings.warn(f"No observations found for series {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            observations = data['observations']
            df = pd.DataFrame(observations)
            
            if df.empty:
                return df
            
            # Clean and standardize data
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove missing values (marked as '.' in FRED)
            df = df[df['value'].notna()]
            
            # Rename value column to series name
            series_name = self.popular_series.get(series_id, series_id)
            df = df.rename(columns={'value': series_name})
            
            # Set datetime index and select relevant columns
            df = df.set_index('date')[series_name]
            df = df.to_frame()
            
            return self.standardize_data(df, datetime_col=None)
            
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Failed to fetch FRED data for {series_id}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_series(self, 
                            series_ids: List[str],
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch multiple economic series and combine them.
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date
            end_date: End date
        
        Returns:
            Combined DataFrame with all series
        """
        combined_data = pd.DataFrame()
        
        for series_id in series_ids:
            try:
                data = self.fetch_data(series_id, start_date, end_date)
                if not data.empty:
                    if combined_data.empty:
                        combined_data = data
                    else:
                        combined_data = combined_data.join(data, how='outer')
            except Exception as e:
                warnings.warn(f"Failed to fetch {series_id}: {e}")
                continue
        
        return combined_data
    
    def get_economic_indicators_bundle(self,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get bundle of key economic indicators.
        
        Returns:
            DataFrame with major economic indicators
        """
        key_indicators = [
            'FEDFUNDS',  # Fed Funds Rate
            'DGS10',     # 10-Year Treasury
            'UNRATE',    # Unemployment Rate
            'CPIAUCSL',  # CPI Inflation
            'GDP',       # GDP
            'VIXCLS',    # VIX
            'DEXUSEU'    # USD/EUR Exchange Rate
        ]
        
        return self.fetch_multiple_series(key_indicators, start_date, end_date)
    
    def get_yield_curve_data(self,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get complete yield curve data.
        
        Returns:
            DataFrame with yield curve rates
        """
        yield_series = [
            'DGS1MO',   # 1-Month
            'DGS3MO',   # 3-Month
            'DGS6MO',   # 6-Month
            'DGS1',     # 1-Year
            'DGS2',     # 2-Year
            'DGS3',     # 3-Year
            'DGS5',     # 5-Year
            'DGS7',     # 7-Year
            'DGS10',    # 10-Year
            'DGS20',    # 20-Year
            'DGS30'     # 30-Year
        ]
        
        return self.fetch_multiple_series(yield_series, start_date, end_date)
    
    def get_employment_data(self,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get comprehensive employment data.
        
        Returns:
            DataFrame with employment indicators
        """
        employment_series = [
            'UNRATE',    # Unemployment Rate
            'PAYEMS',    # Total Nonfarm Payrolls
            'CIVPART',   # Labor Force Participation Rate
            'EMRATIO',   # Employment-Population Ratio
            'AWHMAN',    # Average Weekly Hours
            'AHETPI'     # Average Hourly Earnings
        ]
        
        return self.fetch_multiple_series(employment_series, start_date, end_date)
    
    def get_available_symbols(self) -> List[str]:
        """Get list of popular FRED series."""
        return list(self.popular_series.keys())
    
    def get_data_description(self, symbol: str) -> Dict[str, Any]:
        """Get description of FRED series."""
        if not self.api_key:
            return {'error': 'API key required'}
        
        url = f"{self.base_url}/series"
        params = {
            'series_id': symbol,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'seriess' in data and data['seriess']:
                series_info = data['seriess'][0]
                return {
                    'id': series_info.get('id'),
                    'title': series_info.get('title'),
                    'units': series_info.get('units'),
                    'frequency': series_info.get('frequency'),
                    'seasonal_adjustment': series_info.get('seasonal_adjustment'),
                    'last_updated': series_info.get('last_updated'),
                    'observation_start': series_info.get('observation_start'),
                    'observation_end': series_info.get('observation_end'),
                    'description': self.popular_series.get(symbol, 'Custom series')
                }
        except Exception as e:
            return {'error': str(e)}
        
        return {'error': 'Series not found'}
    
    def search_series(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for FRED series by text.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
        
        Returns:
            List of matching series information
        """
        if not self.api_key:
            return [{'error': 'API key required'}]
        
        url = f"{self.base_url}/series/search"
        params = {
            'search_text': search_text,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'seriess' in data:
                return [
                    {
                        'id': series.get('id'),
                        'title': series.get('title'),
                        'units': series.get('units'),
                        'frequency': series.get('frequency')
                    }
                    for series in data['seriess']
                ]
        except Exception as e:
            return [{'error': str(e)}]
        
        return []