"""
RTradez Metrics Module

Performance evaluation and risk metrics using proven libraries.

Key Components:
- PerformanceAnalyzer: Main analysis using QuantStats and Empyrical
- RiskMetrics: VaR, CVaR, and risk calculations using PyPortfolioOpt
- OptionsMetrics: Options-specific metrics (Greeks exposure, theta decay)
- ComparisonEngine: Strategy comparison using FFN and PyFolio
"""

from .performance import PerformanceAnalyzer
from .risk import RiskMetrics
from .options import OptionsMetrics
from .comparison import ComparisonEngine

__all__ = [
    "PerformanceAnalyzer",
    "RiskMetrics", 
    "OptionsMetrics",
    "ComparisonEngine",
]