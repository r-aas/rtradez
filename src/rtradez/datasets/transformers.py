"""
Sklearn-like data transformers for options trading data.

Provides consistent data preprocessing with sklearn interface:
- .fit(X) -> learn transformation parameters
- .transform(X) -> apply transformation  
- .fit_transform(X) -> combined fit and transform
- .inverse_transform(X) -> reverse transformation
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base import BaseTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ta


class OptionsChainStandardizer(BaseTransformer):
    """
    Standardize options chain data with sklearn-like interface.
    
    Usage:
        scaler = OptionsChainStandardizer()
        clean_data = scaler.fit_transform(raw_options_data)
    """
    
    def __init__(self, fill_method: str = 'forward', 
                 remove_outliers: bool = True,
                 outlier_threshold: float = 3.0):
        """
        Initialize options chain standardizer.
        
        Args:
            fill_method: Method for filling missing values ('forward', 'backward', 'interpolate')
            remove_outliers: Whether to remove statistical outliers
            outlier_threshold: Z-score threshold for outlier removal
        """
        super().__init__()
        self.fill_method = fill_method
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        # Learned parameters (set during fit)
        self.column_means_ = None
        self.column_stds_ = None
        self.outlier_bounds_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OptionsChainStandardizer':
        """
        Learn standardization parameters from options data.
        
        Args:
            X: Raw options chain data
            y: Unused (sklearn compatibility)
            
        Returns:
            Fitted transformer
        """
        X = self._validate_input(X)
        
        # Learn statistics for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.column_means_ = X[numeric_cols].mean()
        self.column_stds_ = X[numeric_cols].std()
        
        # Learn outlier bounds if enabled
        if self.remove_outliers:
            self.outlier_bounds_ = {}
            for col in numeric_cols:
                mean = self.column_means_[col]
                std = self.column_stds_[col]
                self.outlier_bounds_[col] = {
                    'lower': mean - self.outlier_threshold * std,
                    'upper': mean + self.outlier_threshold * std
                }
        
        self.is_fitted_ = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standardization to options data.
        
        Args:
            X: Options data to standardize
            
        Returns:
            Standardized options data
        """
        self._check_is_fitted()
        X = self._validate_input(X, reset=False)
        X_transformed = X.copy()
        
        # Fill missing values
        if self.fill_method == 'forward':
            X_transformed = X_transformed.fillna(method='ffill')
        elif self.fill_method == 'backward':
            X_transformed = X_transformed.fillna(method='bfill')
        elif self.fill_method == 'interpolate':
            X_transformed = X_transformed.interpolate()
            
        # Remove outliers if enabled
        if self.remove_outliers and self.outlier_bounds_:
            for col, bounds in self.outlier_bounds_.items():
                if col in X_transformed.columns:
                    mask = (X_transformed[col] >= bounds['lower']) & (X_transformed[col] <= bounds['upper'])
                    X_transformed = X_transformed[mask]
        
        # Standardize specific options columns
        if 'implied_volatility' in X_transformed.columns:
            # Ensure IV is between 0 and 5 (500%)
            X_transformed['implied_volatility'] = X_transformed['implied_volatility'].clip(0, 5)
            
        if 'time_to_expiration' in X_transformed.columns:
            # Convert to business days if needed
            X_transformed['time_to_expiration'] = pd.to_numeric(X_transformed['time_to_expiration'], errors='coerce')
            
        return X_transformed.dropna()


class TechnicalIndicatorTransformer(BaseTransformer):
    """
    Add technical indicators with sklearn-like interface.
    
    Usage:
        tech = TechnicalIndicatorTransformer(indicators=['rsi', 'macd', 'bollinger'])
        enriched_data = tech.fit_transform(price_data)
    """
    
    def __init__(self, indicators: List[str] = None,
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 bb_period: int = 20,
                 bb_std: int = 2):
        """
        Initialize technical indicator transformer.
        
        Args:
            indicators: List of indicators to calculate
            rsi_period: RSI lookback period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period  
            macd_signal: MACD signal line period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviations
        """
        super().__init__()
        self.indicators = indicators or ['rsi', 'macd', 'bollinger', 'volatility']
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TechnicalIndicatorTransformer':
        """
        Fit transformer (no parameters to learn for technical indicators).
        
        Args:
            X: Price data with OHLCV columns
            y: Unused
            
        Returns:
            Fitted transformer
        """
        X = self._validate_input(X)
        
        # Validate required columns
        required_cols = ['Close']
        if not all(col in X.columns for col in required_cols):
            raise ValueError(f"Input must contain columns: {required_cols}")
            
        self.is_fitted_ = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price data.
        
        Args:
            X: OHLCV price data
            
        Returns:
            Data with added technical indicators
        """
        self._check_is_fitted()
        X = self._validate_input(X, reset=False)
        X_transformed = X.copy()
        
        # Add RSI
        if 'rsi' in self.indicators:
            X_transformed['rsi'] = ta.momentum.RSIIndicator(
                close=X['Close'], 
                window=self.rsi_period
            ).rsi()
            
        # Add MACD
        if 'macd' in self.indicators:
            macd = ta.trend.MACD(
                close=X['Close'],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal
            )
            X_transformed['macd'] = macd.macd()
            X_transformed['macd_signal'] = macd.macd_signal()
            X_transformed['macd_histogram'] = macd.macd_diff()
            
        # Add Bollinger Bands
        if 'bollinger' in self.indicators:
            bb = ta.volatility.BollingerBands(
                close=X['Close'],
                window=self.bb_period,
                window_dev=self.bb_std
            )
            X_transformed['bb_upper'] = bb.bollinger_hband()
            X_transformed['bb_middle'] = bb.bollinger_mavg()
            X_transformed['bb_lower'] = bb.bollinger_lband()
            X_transformed['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            
        # Add volatility measures
        if 'volatility' in self.indicators:
            returns = X['Close'].pct_change()
            X_transformed['realized_vol_10d'] = returns.rolling(10).std() * np.sqrt(252)
            X_transformed['realized_vol_30d'] = returns.rolling(30).std() * np.sqrt(252)
            X_transformed['returns'] = returns
            
        return X_transformed.dropna()


class VolatilitySurfaceTransformer(BaseTransformer):
    """
    Transform options data into volatility surface format with sklearn interface.
    
    Usage:
        vol_surface = VolatilitySurfaceTransformer()
        surface_data = vol_surface.fit_transform(options_chain)
    """
    
    def __init__(self, moneyness_buckets: int = 20,
                 dte_buckets: int = 10,
                 min_dte: int = 7,
                 max_dte: int = 365):
        """
        Initialize volatility surface transformer.
        
        Args:
            moneyness_buckets: Number of moneyness buckets
            dte_buckets: Number of days-to-expiration buckets
            min_dte: Minimum DTE to include
            max_dte: Maximum DTE to include
        """
        super().__init__()
        self.moneyness_buckets = moneyness_buckets
        self.dte_buckets = dte_buckets
        self.min_dte = min_dte
        self.max_dte = max_dte
        
        # Learned parameters
        self.moneyness_edges_ = None
        self.dte_edges_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VolatilitySurfaceTransformer':
        """
        Learn volatility surface structure from options data.
        
        Args:
            X: Options chain data with moneyness, DTE, and IV columns
            y: Unused
            
        Returns:
            Fitted transformer
        """
        X = self._validate_input(X)
        
        # Validate required columns
        required_cols = ['moneyness', 'time_to_expiration', 'implied_volatility']
        if not all(col in X.columns for col in required_cols):
            raise ValueError(f"Input must contain columns: {required_cols}")
            
        # Filter by DTE range
        filtered_data = X[
            (X['time_to_expiration'] >= self.min_dte) & 
            (X['time_to_expiration'] <= self.max_dte)
        ]
        
        # Learn bucket edges
        self.moneyness_edges_ = np.linspace(
            filtered_data['moneyness'].quantile(0.05),
            filtered_data['moneyness'].quantile(0.95),
            self.moneyness_buckets + 1
        )
        
        self.dte_edges_ = np.linspace(
            self.min_dte,
            self.max_dte,
            self.dte_buckets + 1
        )
        
        self.is_fitted_ = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform options data into volatility surface format.
        
        Args:
            X: Options chain data
            
        Returns:
            Volatility surface DataFrame with MultiIndex (moneyness, DTE)
        """
        self._check_is_fitted()
        X = self._validate_input(X, reset=False)
        
        # Filter and prepare data
        filtered_data = X[
            (X['time_to_expiration'] >= self.min_dte) & 
            (X['time_to_expiration'] <= self.max_dte)
        ].copy()
        
        # Assign buckets
        filtered_data['moneyness_bucket'] = pd.cut(
            filtered_data['moneyness'], 
            bins=self.moneyness_edges_,
            labels=False
        )
        
        filtered_data['dte_bucket'] = pd.cut(
            filtered_data['time_to_expiration'],
            bins=self.dte_edges_,
            labels=False
        )
        
        # Aggregate to surface
        surface = filtered_data.groupby(['moneyness_bucket', 'dte_bucket']).agg({
            'implied_volatility': ['mean', 'count'],
            'moneyness': 'mean',
            'time_to_expiration': 'mean'
        }).round(4)
        
        # Flatten column names
        surface.columns = ['_'.join(col).strip() for col in surface.columns]
        
        return surface.reset_index()


class ReturnsTransformer(BaseTransformer):
    """
    Transform price data to returns with sklearn interface.
    
    Usage:
        returns_transformer = ReturnsTransformer(method='log', periods=1)
        returns = returns_transformer.fit_transform(price_data)
    """
    
    def __init__(self, method: str = 'simple', periods: int = 1):
        """
        Initialize returns transformer.
        
        Args:
            method: 'simple' or 'log' returns
            periods: Number of periods for return calculation
        """
        super().__init__()
        self.method = method
        self.periods = periods
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ReturnsTransformer':
        """Fit transformer (no parameters to learn)."""
        X = self._validate_input(X)
        self.is_fitted_ = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform prices to returns.
        
        Args:
            X: Price data
            
        Returns:
            Returns data
        """
        self._check_is_fitted()
        X = self._validate_input(X, reset=False)
        
        if self.method == 'simple':
            returns = X.pct_change(periods=self.periods)
        elif self.method == 'log':
            returns = np.log(X / X.shift(self.periods))
        else:
            raise ValueError("method must be 'simple' or 'log'")
            
        return returns.dropna()
        
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform returns back to prices (requires initial price level).
        
        Args:
            X: Returns data
            
        Returns:
            Price data
        """
        self._check_is_fitted()
        
        if self.method == 'simple':
            # Cumulative product starting from 1
            prices = (1 + X).cumprod()
        elif self.method == 'log':
            # Exponential of cumulative sum
            prices = np.exp(X.cumsum())
        else:
            raise ValueError("method must be 'simple' or 'log'")
            
        return prices