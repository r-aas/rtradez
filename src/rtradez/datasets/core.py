"""
Core dataset management for options trading data.

This module provides the main OptionsDataset class for handling options chain data,
historical data loading, and data preprocessing using proven open-source libraries.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union, Tuple
import yfinance as yf
import vectorbt as vbt
import quantstats as qs
import mibian  # Black-Scholes pricing


class OptionsDataset:
    """
    Main dataset class for options trading data management.
    
    Handles loading, storing, and preprocessing of options chain data
    from various sources with standardized format and validation.
    """
    
    def __init__(self, symbol: str, data_source: str = 'yahoo'):
        """
        Initialize OptionsDataset.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            data_source: Data source ('yahoo', 'quandl', 'alpha_vantage')
        """
        self.symbol = symbol.upper()
        self.data_source = data_source
        self.options_data: Optional[pd.DataFrame] = None
        self.underlying_data: Optional[pd.DataFrame] = None
        self.expiration_dates: List[datetime] = []
        
    @classmethod
    def from_source(cls, data_source: str, symbol: str, **kwargs) -> 'OptionsDataset':
        """
        Factory method to create dataset from specific data source.
        
        Args:
            data_source: Data source identifier
            symbol: Stock symbol
            **kwargs: Additional source-specific parameters
            
        Returns:
            OptionsDataset instance
        """
        dataset = cls(symbol, data_source)
        
        if data_source.lower() == 'yahoo':
            dataset.load_yahoo_data(**kwargs)
        elif data_source.lower() == 'quandl':
            dataset.load_quandl_data(**kwargs)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
            
        return dataset
        
    def load_yahoo_data(self, period: str = "1y") -> None:
        """
        Load options and underlying data from Yahoo Finance.
        
        Args:
            period: Time period for historical data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        """
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Load underlying stock data
            self.underlying_data = ticker.history(period=period)
            
            # Get available expiration dates
            self.expiration_dates = ticker.options
            
            # Load options chain for each expiration
            options_list = []
            for exp_date in self.expiration_dates[:5]:  # Limit to first 5 expirations
                try:
                    opt_chain = ticker.option_chain(exp_date)
                    
                    # Process calls
                    calls = opt_chain.calls.copy()
                    calls['type'] = 'call'
                    calls['expiration'] = exp_date
                    
                    # Process puts  
                    puts = opt_chain.puts.copy()
                    puts['type'] = 'put'
                    puts['expiration'] = exp_date
                    
                    options_list.extend([calls, puts])
                    
                except Exception as e:
                    print(f"Failed to load options for {exp_date}: {e}")
                    continue
                    
            if options_list:
                self.options_data = pd.concat(options_list, ignore_index=True)
                self._standardize_columns()
                
        except Exception as e:
            raise ValueError(f"Failed to load data for {self.symbol}: {e}")
            
    def load_quandl_data(self, api_key: str, **kwargs) -> None:
        """
        Load data from Quandl (placeholder for future implementation).
        
        Args:
            api_key: Quandl API key
            **kwargs: Additional Quandl parameters
        """
        raise NotImplementedError("Quandl integration coming soon")
        
    def _standardize_columns(self) -> None:
        """Standardize column names and data types."""
        if self.options_data is None:
            return
            
        # Rename columns to standard format
        column_mapping = {
            'lastPrice': 'last_price',
            'lastTradeDate': 'last_trade_date', 
            'impliedVolatility': 'implied_volatility',
            'inTheMoney': 'in_the_money',
            'contractSymbol': 'contract_symbol',
            'contractSize': 'contract_size',
            'openInterest': 'open_interest'
        }
        
        self.options_data = self.options_data.rename(columns=column_mapping)
        
        # Convert data types
        if 'expiration' in self.options_data.columns:
            self.options_data['expiration'] = pd.to_datetime(self.options_data['expiration'])
            
    def get_options_chain(self, expiration_date: Optional[str] = None, 
                         option_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get options chain data with optional filtering.
        
        Args:
            expiration_date: Filter by expiration date (YYYY-MM-DD format)
            option_type: Filter by type ('call' or 'put')
            
        Returns:
            Filtered options chain DataFrame
        """
        if self.options_data is None:
            raise ValueError("No options data loaded")
            
        filtered_data = self.options_data.copy()
        
        if expiration_date:
            exp_date = pd.to_datetime(expiration_date)
            filtered_data = filtered_data[filtered_data['expiration'] == exp_date]
            
        if option_type:
            filtered_data = filtered_data[filtered_data['type'] == option_type.lower()]
            
        return filtered_data
        
    def get_underlying_price(self, date: Optional[str] = None) -> float:
        """
        Get underlying stock price for specific date or latest.
        
        Args:
            date: Date in YYYY-MM-DD format (latest if None)
            
        Returns:
            Stock price
        """
        if self.underlying_data is None:
            raise ValueError("No underlying data loaded")
            
        if date:
            target_date = pd.to_datetime(date)
            try:
                return self.underlying_data.loc[target_date, 'Close']
            except KeyError:
                # Find nearest date
                nearest_idx = self.underlying_data.index.get_indexer([target_date], method='nearest')[0]
                return self.underlying_data.iloc[nearest_idx]['Close']
        else:
            return self.underlying_data['Close'].iloc[-1]
            
    def calculate_moneyness(self, spot_price: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate moneyness (S/K) for all options.
        
        Args:
            spot_price: Current underlying price (latest if None)
            
        Returns:
            Options data with moneyness column
        """
        if self.options_data is None:
            raise ValueError("No options data loaded")
            
        if spot_price is None:
            spot_price = self.get_underlying_price()
            
        result = self.options_data.copy()
        result['moneyness'] = spot_price / result['strike']
        result['spot_price'] = spot_price
        
        return result
        
    def get_time_to_expiration(self, reference_date: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate time to expiration in days for all options.
        
        Args:
            reference_date: Reference date (today if None)
            
        Returns:
            Options data with time_to_expiration column
        """
        if self.options_data is None:
            raise ValueError("No options data loaded")
            
        if reference_date is None:
            ref_date = datetime.now()
        else:
            ref_date = pd.to_datetime(reference_date)
            
        result = self.options_data.copy()
        result['time_to_expiration'] = (result['expiration'] - ref_date).dt.days
        
        return result
        
    def summary(self) -> Dict:
        """
        Get dataset summary statistics.
        
        Returns:
            Dictionary with summary information
        """
        summary_info = {
            'symbol': self.symbol,
            'data_source': self.data_source,
            'options_count': len(self.options_data) if self.options_data is not None else 0,
            'underlying_days': len(self.underlying_data) if self.underlying_data is not None else 0,
            'expiration_dates': len(self.expiration_dates),
            'date_range': None
        }
        
        if self.underlying_data is not None and not self.underlying_data.empty:
            summary_info['date_range'] = {
                'start': self.underlying_data.index.min().strftime('%Y-%m-%d'),
                'end': self.underlying_data.index.max().strftime('%Y-%m-%d')
            }
            
        return summary_info
        
    def __repr__(self) -> str:
        """String representation of the dataset."""
        summary = self.summary()
        return f"OptionsDataset(symbol='{summary['symbol']}', options={summary['options_count']}, underlying_days={summary['underlying_days']})"