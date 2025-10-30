"""
Options trading strategies with sklearn-like interface.

This module provides pre-built options strategies following sklearn patterns:
- .fit(market_data) -> learn from historical data
- .predict(market_conditions) -> generate trading signals  
- .score(X, y) -> evaluate strategy performance
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base import BaseStrategy


@dataclass
class StrategyConfig:
    """Configuration for options strategies."""
    name: str
    description: str
    parameters: Dict
    risk_profile: str = "medium"
    margin_requirement: float = 0.0


class OptionsStrategy(BaseStrategy):
    """
    Sklearn-like options strategy framework.
    
    Follows sklearn patterns:
    - .fit(market_data) -> learn optimal parameters from historical data
    - .predict(market_conditions) -> generate trading signals
    - .score(X, y) -> evaluate strategy performance
    
    Supports strategies: Iron Condor, Strangle, Straddle, Calendar Spread
    """
    
    STRATEGY_REGISTRY = {
        'iron_condor': {
            'name': 'Iron Condor',
            'description': 'Short put spread + short call spread',
            'risk_profile': 'low',
            'default_params': {
                'put_strike_distance': 5,
                'call_strike_distance': 5,
                'spread_width': 5,
                'dte_entry': 30,
                'dte_exit': 7,
                'profit_target': 0.5,
                'stop_loss': 2.0
            }
        },
        'strangle': {
            'name': 'Short Strangle',
            'description': 'Short out-of-money put and call',
            'risk_profile': 'high',
            'default_params': {
                'put_delta': 0.15,
                'call_delta': 0.15,
                'dte_entry': 30,
                'dte_exit': 7,
                'profit_target': 0.5,
                'stop_loss': 2.5
            }
        },
        'straddle': {
            'name': 'Short Straddle',
            'description': 'Short at-the-money put and call',
            'risk_profile': 'high',
            'default_params': {
                'moneyness_tolerance': 0.02,
                'dte_entry': 30,
                'dte_exit': 7,
                'profit_target': 0.5,
                'stop_loss': 2.5
            }
        },
        'calendar_spread': {
            'name': 'Calendar Spread',
            'description': 'Long back month, short front month',
            'risk_profile': 'medium',
            'default_params': {
                'strike_selection': 'atm',
                'front_dte': 30,
                'back_dte': 60,
                'profit_target': 0.25,
                'stop_loss': 1.5
            }
        }
    }
    
    def __init__(self, strategy_type: str, **params):
        """
        Initialize options strategy.
        
        Args:
            strategy_type: Strategy type from STRATEGY_REGISTRY
            **params: Strategy-specific parameters
        """
        super().__init__()
        
        if strategy_type not in self.STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy_type}")
            
        self.strategy_type = strategy_type
        self.config = self.STRATEGY_REGISTRY[strategy_type].copy()
        
        # Merge default params with user params
        self.params = self.config['default_params'].copy()
        self.params.update(params)
        
        # Set individual parameters as attributes for sklearn compatibility
        for param_name, param_value in self.params.items():
            setattr(self, param_name, param_value)
        
        # Sklearn-like attributes (set after fitting)
        self.optimal_params_ = None
        self.learned_volatility_ = None
        self.historical_performance_ = None
        
    @classmethod
    def create(cls, strategy_type: str, **params) -> 'OptionsStrategy':
        """Factory method to create strategy."""
        return cls(strategy_type, **params)
        
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategy types."""
        return list(cls.STRATEGY_REGISTRY.keys())
        
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        params = {'strategy_type': self.strategy_type}
        params.update(self.params)
        return params
        
    def set_params(self, **parameters):
        """Set parameters for sklearn compatibility."""
        valid_params = set(['strategy_type'] + list(self.params.keys()))
        
        for parameter, value in parameters.items():
            if parameter in valid_params:
                if parameter == 'strategy_type':
                    self.strategy_type = value
                else:
                    self.params[parameter] = value
                    setattr(self, parameter, value)
            else:
                raise ValueError(f"Invalid parameter '{parameter}' for strategy {self.strategy_type}")
        
        return self
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OptionsStrategy':
        """
        Fit strategy to historical market data (sklearn-like interface).
        
        Args:
            X: Market data with columns like ['Close', 'Volume', 'High', 'Low']
            y: Optional target returns for optimization
            
        Returns:
            Fitted strategy instance
        """
        X = self._validate_input(X)
        
        # Learn from historical data
        self.learned_volatility_ = X['Close'].pct_change().std() * np.sqrt(252)
        
        # Optimize parameters based on historical performance
        self.optimal_params_ = self._optimize_parameters(X, y)
        
        # Calculate historical performance metrics
        signals = self._generate_signals(X)
        if y is not None:
            # Align signals and returns properly
            min_length = min(len(signals) - 1, len(y) - 1)
            aligned_signals = signals[:min_length]
            aligned_returns = y.values[1:min_length + 1]
            
            strategy_returns = aligned_signals * aligned_returns
            self.historical_performance_ = {
                'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0,
                'total_return': (1 + strategy_returns).prod() - 1,
                'max_drawdown': self._calculate_max_drawdown(pd.Series(strategy_returns))
            }
        
        self.is_fitted_ = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals (sklearn-like interface).
        
        Args:
            X: Market conditions DataFrame
            
        Returns:
            Trading signals array (-1: sell, 0: hold, 1: buy)
        """
        self._check_is_fitted()
        X = self._validate_input(X, reset=False)
        
        return self._generate_signals(X)
        
    def _optimize_parameters(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict:
        """Optimize strategy parameters based on historical data."""
        # Simple optimization - in practice, use more sophisticated methods
        optimal_params = self.params.copy()
        
        if self.strategy_type == 'iron_condor':
            # Optimize based on realized volatility
            vol = X['Close'].pct_change().rolling(20).std().mean() * np.sqrt(252)
            if vol > 0.25:  # High vol environment
                optimal_params['profit_target'] = 0.6
                optimal_params['stop_loss'] = 1.8
            else:  # Low vol environment
                optimal_params['profit_target'] = 0.4
                optimal_params['stop_loss'] = 2.2
                
        return optimal_params
        
    def _generate_signals(self, X: pd.DataFrame) -> np.ndarray:
        """Generate trading signals based on strategy type."""
        signals = np.zeros(len(X))
        
        if self.strategy_type == 'iron_condor':
            # Enter positions when volatility is high (short vol strategy)
            vol = X['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            vol_threshold = vol.quantile(0.7)  # Top 30% volatility
            signals[vol > vol_threshold] = -1  # Short volatility
            
        elif self.strategy_type in ['strangle', 'straddle']:
            # Similar to iron condor but more aggressive
            vol = X['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            vol_threshold = vol.quantile(0.6)  # Top 40% volatility
            signals[vol > vol_threshold] = -1
            
        elif self.strategy_type == 'calendar_spread':
            # Enter when term structure is steep
            short_vol = X['Close'].pct_change().rolling(10).std() * np.sqrt(252)
            long_vol = X['Close'].pct_change().rolling(60).std() * np.sqrt(252)
            signals[(long_vol - short_vol) > 0.05] = 1  # Long calendar when steep
            
        return signals
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()
        
    def backtest(self, dataset, start_date: str, end_date: str, 
                initial_capital: float = 100000) -> vbt.Portfolio:
        """
        Backtest strategy using VectorBT.
        
        Args:
            dataset: OptionsDataset instance
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital
            
        Returns:
            VectorBT Portfolio object with results
        """
        # Get underlying price data for the period
        underlying_data = dataset.underlying_data.loc[start_date:end_date]['Close']
        
        if self.strategy_type == 'iron_condor':
            return self._backtest_iron_condor(dataset, underlying_data, initial_capital)
        elif self.strategy_type == 'strangle':
            return self._backtest_strangle(dataset, underlying_data, initial_capital)
        elif self.strategy_type == 'straddle':
            return self._backtest_straddle(dataset, underlying_data, initial_capital)
        elif self.strategy_type == 'calendar_spread':
            return self._backtest_calendar_spread(dataset, underlying_data, initial_capital)
        else:
            raise NotImplementedError(f"Backtesting for {self.strategy_type} not implemented")
            
    def _backtest_iron_condor(self, dataset, prices: pd.Series, 
                             initial_capital: float) -> vbt.Portfolio:
        """Backtest Iron Condor using VectorBT vectorized operations."""
        
        # Generate entry signals based on DTE
        entry_signals = self._generate_dte_signals(prices, self.params['dte_entry'])
        
        # Calculate theoretical P&L for iron condor positions
        # This is simplified - in practice you'd use actual options prices
        pnl_series = self._simulate_iron_condor_pnl(prices, entry_signals)
        
        # Create VectorBT portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entry_signals,
            exits=self._generate_exit_signals(prices, pnl_series),
            init_cash=initial_capital,
            fees=0.01,  # $1 per contract
            freq='D'
        )
        
        self.results = portfolio
        return portfolio
        
    def _backtest_strangle(self, dataset, prices: pd.Series,
                          initial_capital: float) -> vbt.Portfolio:
        """Backtest Short Strangle strategy."""
        
        # Generate signals and P&L for strangle
        entry_signals = self._generate_dte_signals(prices, self.params['dte_entry'])
        pnl_series = self._simulate_strangle_pnl(prices, entry_signals)
        
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entry_signals,
            exits=self._generate_exit_signals(prices, pnl_series),
            init_cash=initial_capital,
            fees=0.02,  # $2 per contract (2 legs)
            freq='D'
        )
        
        self.results = portfolio
        return portfolio
        
    def _backtest_straddle(self, dataset, prices: pd.Series,
                          initial_capital: float) -> vbt.Portfolio:
        """Backtest Short Straddle strategy."""
        
        entry_signals = self._generate_dte_signals(prices, self.params['dte_entry'])
        pnl_series = self._simulate_straddle_pnl(prices, entry_signals)
        
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entry_signals,
            exits=self._generate_exit_signals(prices, pnl_series),
            init_cash=initial_capital,
            fees=0.02,
            freq='D'
        )
        
        self.results = portfolio
        return portfolio
        
    def _backtest_calendar_spread(self, dataset, prices: pd.Series,
                                 initial_capital: float) -> vbt.Portfolio:
        """Backtest Calendar Spread strategy."""
        
        entry_signals = self._generate_dte_signals(prices, self.params['front_dte'])
        pnl_series = self._simulate_calendar_pnl(prices, entry_signals)
        
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entry_signals,
            exits=self._generate_exit_signals(prices, pnl_series),
            init_cash=initial_capital,
            fees=0.02,
            freq='D'
        )
        
        self.results = portfolio
        return portfolio
        
    def _generate_dte_signals(self, prices: pd.Series, target_dte: int) -> pd.Series:
        """Generate entry signals based on days to expiration."""
        # Simplified: enter positions every month
        signals = pd.Series(False, index=prices.index)
        
        # Enter on first trading day of each month
        monthly_starts = prices.resample('MS').first().index
        for start in monthly_starts:
            if start in signals.index:
                signals.loc[start] = True
                
        return signals
        
    def _generate_exit_signals(self, prices: pd.Series, pnl_series: pd.Series) -> pd.Series:
        """Generate exit signals based on profit targets and stop losses."""
        exits = pd.Series(False, index=prices.index)
        
        # Exit based on profit target or stop loss
        profit_target = self.params.get('profit_target', 0.5)
        stop_loss = self.params.get('stop_loss', 2.0)
        
        # Simplified exit logic
        exits = (pnl_series >= profit_target) | (pnl_series <= -stop_loss)
        
        return exits
        
    def _simulate_iron_condor_pnl(self, prices: pd.Series, entries: pd.Series) -> pd.Series:
        """Simulate Iron Condor P&L based on underlying movement."""
        pnl = pd.Series(0.0, index=prices.index)
        
        # Simplified simulation: profit from low volatility
        volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
        
        # Iron condor profits when underlying stays within range
        for i, (date, entry) in enumerate(entries.items()):
            if entry and i > 0:
                # Calculate P&L based on realized vs expected volatility
                entry_price = prices.loc[date]
                
                # Simulate 30-day holding period
                end_idx = min(i + 30, len(prices) - 1)
                holding_period = prices.iloc[i:end_idx]
                
                if len(holding_period) > 1:
                    realized_vol = holding_period.pct_change().std() * np.sqrt(252)
                    expected_vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.2
                    
                    # Profit when realized vol < expected vol
                    vol_diff = expected_vol - realized_vol
                    base_pnl = vol_diff * 1000  # Scale factor
                    
                    pnl.iloc[i:end_idx] = base_pnl
                    
        return pnl
        
    def _simulate_strangle_pnl(self, prices: pd.Series, entries: pd.Series) -> pd.Series:
        """Simulate Short Strangle P&L."""
        # Similar to iron condor but higher risk/reward
        pnl = self._simulate_iron_condor_pnl(prices, entries) * 1.5
        return pnl
        
    def _simulate_straddle_pnl(self, prices: pd.Series, entries: pd.Series) -> pd.Series:
        """Simulate Short Straddle P&L."""
        # Higher risk than strangle
        pnl = self._simulate_iron_condor_pnl(prices, entries) * 2.0
        return pnl
        
    def _simulate_calendar_pnl(self, prices: pd.Series, entries: pd.Series) -> pd.Series:
        """Simulate Calendar Spread P&L."""
        # Lower volatility strategy
        pnl = self._simulate_iron_condor_pnl(prices, entries) * 0.8
        return pnl
        
    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics using VectorBT."""
        if self.results is None:
            raise ValueError("No backtest results available")
            
        stats = self.results.stats()
        
        return {
            'total_return': stats['Total Return [%]'],
            'sharpe_ratio': stats['Sharpe Ratio'],
            'max_drawdown': stats['Max. Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'profit_factor': stats.get('Profit Factor', None),
            'total_trades': stats['# Trades'],
            'avg_trade': stats['Avg. Trade [%]'],
        }
        
    def plot_results(self, **kwargs):
        """Plot backtest results using VectorBT."""
        if self.results is None:
            raise ValueError("No backtest results available")
            
        return self.results.plot(**kwargs)
        
    def __repr__(self) -> str:
        """String representation."""
        return f"OptionsStrategy(type='{self.strategy_type}', params={self.params})"