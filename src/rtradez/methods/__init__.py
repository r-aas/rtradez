"""
RTradez Methods Module

Trading strategies and algorithms leveraging proven open-source libraries.

Key Components:
- OptionsStrategy: Main strategy framework using VectorBT for vectorized backtesting
- GreeksCalculator: Options Greeks using Mibian and QuantLib
- StrategyBuilder: Strategy construction utilities
- BacktestEngine: Backtesting framework using BackTrader
"""

from .strategies import OptionsStrategy
from .greeks import GreeksCalculator  
from .backtest import BacktestEngine
from .builder import StrategyBuilder

__all__ = [
    "OptionsStrategy",
    "GreeksCalculator",
    "BacktestEngine", 
    "StrategyBuilder",
]