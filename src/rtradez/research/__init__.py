"""Advanced research and analytics modules for RTradez."""

from .regime_detection import MarketRegimeDetector, RegimeBasedStrategy
from .advanced_backtest import AdvancedBacktester, TransactionCostModel
from .greeks_analysis import GreeksAnalyzer, DeltaHedger, GammaScalper
# from .volatility_surface import VolatilitySurface, ImpliedVolatilityModel  # TODO: implement
# from .factor_analysis import FactorAnalyzer, PerformanceAttribution  # TODO: implement 
# from .ensemble_methods import StrategyEnsemble, VotingStrategy, StackingStrategy  # TODO: implement
from .visualization import ResearchVisualizer, InteractivePlotter

__all__ = [
    'MarketRegimeDetector', 'RegimeBasedStrategy',
    'AdvancedBacktester', 'TransactionCostModel', 
    'GreeksAnalyzer', 'DeltaHedger', 'GammaScalper',
    # 'VolatilitySurface', 'ImpliedVolatilityModel',  # TODO: implement
    # 'FactorAnalyzer', 'PerformanceAttribution',  # TODO: implement
    # 'StrategyEnsemble', 'VotingStrategy', 'StackingStrategy',  # TODO: implement
    'ResearchVisualizer', 'InteractivePlotter'
]