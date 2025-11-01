"""
Portfolio risk calculation and monitoring.

Implements Value at Risk (VaR), Expected Shortfall, correlation analysis,
and portfolio-level risk metrics for multi-strategy options trading.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

logger = logging.getLogger(__name__)

class VaRMethod(Enum):
    """VaR calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"

@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var_1d: float  # 1-day VaR
    var_5d: float  # 5-day VaR
    var_21d: float  # 21-day VaR
    expected_shortfall: float  # Expected Shortfall (CVaR)
    confidence_level: float
    method: VaRMethod
    portfolio_volatility: float
    diversification_ratio: float
    warnings: List[str]

@dataclass
class CorrelationAnalysis:
    """Portfolio correlation analysis results."""
    correlation_matrix: np.ndarray
    eigenvalues: np.ndarray
    concentration_ratio: float  # First eigenvalue / sum of eigenvalues
    effective_strategies: float  # Effective number of independent strategies
    max_correlation: float
    avg_correlation: float
    strategy_names: List[str]

class VaRCalculator:
    """Calculates Value at Risk using multiple methodologies."""
    
    def __init__(self, confidence_level: float = 0.05, 
                 lookback_days: int = 252,
                 monte_carlo_simulations: int = 10000):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: VaR confidence level (e.g., 0.05 for 95% VaR)
            lookback_days: Historical lookback period
            monte_carlo_simulations: Number of MC simulations
        """
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
        self.monte_carlo_simulations = monte_carlo_simulations
        
    def calculate_var(self, returns: pd.Series, 
                     method: VaRMethod = VaRMethod.HISTORICAL,
                     portfolio_value: float = 1.0) -> VaRResult:
        """
        Calculate VaR using specified method.
        
        Args:
            returns: Portfolio return series
            method: VaR calculation method
            portfolio_value: Current portfolio value
        """
        warnings_list = []
        
        # Validate inputs
        if len(returns) < 30:
            warnings_list.append("Insufficient data for reliable VaR calculation")
            
        if returns.std() == 0:
            warnings_list.append("Zero volatility in returns")
            return VaRResult(0, 0, 0, 0, self.confidence_level, method, 0, 1.0, warnings_list)
        
        # Calculate VaR based on method
        if method == VaRMethod.HISTORICAL:
            var_1d = self._historical_var(returns, portfolio_value)
        elif method == VaRMethod.PARAMETRIC:
            var_1d = self._parametric_var(returns, portfolio_value)
        elif method == VaRMethod.MONTE_CARLO:
            var_1d = self._monte_carlo_var(returns, portfolio_value)
        elif method == VaRMethod.CORNISH_FISHER:
            var_1d = self._cornish_fisher_var(returns, portfolio_value)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Scale to different time horizons
        daily_vol = returns.std()
        var_5d = var_1d * np.sqrt(5)
        var_21d = var_1d * np.sqrt(21)
        
        # Calculate Expected Shortfall (Conditional VaR)
        expected_shortfall = self._calculate_expected_shortfall(returns, var_1d, portfolio_value)
        
        return VaRResult(
            var_1d=var_1d,
            var_5d=var_5d,
            var_21d=var_21d,
            expected_shortfall=expected_shortfall,
            confidence_level=self.confidence_level,
            method=method,
            portfolio_volatility=daily_vol,
            diversification_ratio=1.0,  # Will be calculated at portfolio level
            warnings=warnings_list
        )
    
    def _historical_var(self, returns: pd.Series, portfolio_value: float) -> float:
        """Calculate historical VaR."""
        # Use most recent data up to lookback limit
        recent_returns = returns.tail(self.lookback_days)
        var_percentile = np.percentile(recent_returns, self.confidence_level * 100)
        return abs(var_percentile * portfolio_value)
    
    def _parametric_var(self, returns: pd.Series, portfolio_value: float) -> float:
        """Calculate parametric (normal) VaR."""
        mean_return = returns.mean()
        volatility = returns.std()
        z_score = stats.norm.ppf(self.confidence_level)
        var_return = mean_return + z_score * volatility
        return abs(var_return * portfolio_value)
    
    def _monte_carlo_var(self, returns: pd.Series, portfolio_value: float) -> float:
        """Calculate Monte Carlo VaR."""
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        scenarios = np.random.normal(mean_return, volatility, self.monte_carlo_simulations)
        
        var_percentile = np.percentile(scenarios, self.confidence_level * 100)
        return abs(var_percentile * portfolio_value)
    
    def _cornish_fisher_var(self, returns: pd.Series, portfolio_value: float) -> float:
        """Calculate Cornish-Fisher VaR (accounts for skewness and kurtosis)."""
        mean_return = returns.mean()
        volatility = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Cornish-Fisher expansion
        z = stats.norm.ppf(self.confidence_level)
        cf_adjustment = (
            z + 
            (z**2 - 1) * skewness / 6 +
            (z**3 - 3*z) * kurtosis / 24 -
            (2*z**3 - 5*z) * skewness**2 / 36
        )
        
        var_return = mean_return + cf_adjustment * volatility
        return abs(var_return * portfolio_value)
    
    def _calculate_expected_shortfall(self, returns: pd.Series, var: float, 
                                    portfolio_value: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var_threshold = -var / portfolio_value  # Convert to return space
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var  # Fallback to VaR if no tail observations
        
        expected_shortfall_return = tail_returns.mean()
        return abs(expected_shortfall_return * portfolio_value)

class CorrelationAnalyzer:
    """Analyzes correlations and diversification in strategy portfolios."""
    
    def __init__(self, min_periods: int = 60, shrinkage_method: str = "ledoit_wolf"):
        """
        Initialize correlation analyzer.
        
        Args:
            min_periods: Minimum periods required for correlation calculation
            shrinkage_method: Covariance matrix shrinkage method
        """
        self.min_periods = min_periods
        self.shrinkage_method = shrinkage_method
        
    def analyze_correlations(self, strategy_returns: pd.DataFrame,
                           strategy_names: Optional[List[str]] = None) -> CorrelationAnalysis:
        """
        Analyze correlations between strategies.
        
        Args:
            strategy_returns: DataFrame with strategy returns (strategies as columns)
            strategy_names: Names of strategies (uses column names if None)
        """
        if strategy_names is None:
            strategy_names = list(strategy_returns.columns)
        
        # Remove strategies with insufficient data
        valid_strategies = []
        valid_returns = []
        
        for i, strategy in enumerate(strategy_names):
            if strategy in strategy_returns.columns:
                strategy_data = strategy_returns[strategy].dropna()
                if len(strategy_data) >= self.min_periods:
                    valid_strategies.append(strategy)
                    valid_returns.append(strategy_data)
        
        if len(valid_strategies) < 2:
            # Return trivial analysis for single strategy
            return CorrelationAnalysis(
                correlation_matrix=np.array([[1.0]]),
                eigenvalues=np.array([1.0]),
                concentration_ratio=1.0,
                effective_strategies=1.0,
                max_correlation=0.0,
                avg_correlation=0.0,
                strategy_names=valid_strategies
            )
        
        # Align returns and calculate correlation matrix
        aligned_returns = pd.DataFrame({name: data for name, data in zip(valid_strategies, valid_returns)})
        aligned_returns = aligned_returns.dropna()
        
        if len(aligned_returns) < self.min_periods:
            logger.warning(f"Only {len(aligned_returns)} aligned observations for correlation analysis")
        
        # Calculate correlation matrix with shrinkage
        if self.shrinkage_method == "ledoit_wolf":
            try:
                cov_estimator = LedoitWolf()
                covariance_matrix = cov_estimator.fit(aligned_returns).covariance_
                std_devs = np.sqrt(np.diag(covariance_matrix))
                correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
            except Exception as e:
                logger.warning(f"Shrinkage correlation failed: {e}, using sample correlation")
                correlation_matrix = aligned_returns.corr().values
        else:
            correlation_matrix = aligned_returns.corr().values
        
        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Calculate concentration ratio (largest eigenvalue proportion)
        concentration_ratio = eigenvalues[0] / np.sum(eigenvalues)
        
        # Effective number of strategies (inverse participation ratio)
        effective_strategies = 1.0 / np.sum((eigenvalues / np.sum(eigenvalues))**2)
        
        # Correlation statistics
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        max_correlation = np.max(np.abs(upper_triangle)) if len(upper_triangle) > 0 else 0.0
        avg_correlation = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        return CorrelationAnalysis(
            correlation_matrix=correlation_matrix,
            eigenvalues=eigenvalues,
            concentration_ratio=concentration_ratio,
            effective_strategies=effective_strategies,
            max_correlation=max_correlation,
            avg_correlation=avg_correlation,
            strategy_names=valid_strategies
        )

class PortfolioRiskCalculator:
    """Comprehensive portfolio risk calculation and monitoring."""
    
    def __init__(self, var_calculator: Optional[VaRCalculator] = None,
                 correlation_analyzer: Optional[CorrelationAnalyzer] = None):
        """Initialize portfolio risk calculator."""
        self.var_calculator = var_calculator or VaRCalculator()
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()
        
    def calculate_portfolio_risk(self, strategy_returns: pd.DataFrame,
                               strategy_weights: Dict[str, float],
                               portfolio_value: float = 1.0) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            strategy_returns: DataFrame with individual strategy returns
            strategy_weights: Dictionary of strategy weights
            portfolio_value: Total portfolio value
        """
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(strategy_returns, strategy_weights)
        
        # VaR calculation
        var_result = self.var_calculator.calculate_var(
            portfolio_returns, 
            VaRMethod.HISTORICAL,
            portfolio_value
        )
        
        # Correlation analysis
        correlation_analysis = self.correlation_analyzer.analyze_correlations(
            strategy_returns, list(strategy_weights.keys())
        )
        
        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(
            strategy_returns, strategy_weights, correlation_analysis
        )
        
        # Update VaR result with diversification ratio
        var_result.diversification_ratio = diversification_ratio
        
        # Risk decomposition
        marginal_var = self._calculate_marginal_var(
            strategy_returns, strategy_weights, portfolio_returns, var_result.var_1d
        )
        
        component_var = {strategy: weight * marginal_var.get(strategy, 0) 
                        for strategy, weight in strategy_weights.items()}
        
        # Portfolio statistics
        portfolio_stats = {
            'portfolio_volatility': portfolio_returns.std() * np.sqrt(252),
            'portfolio_sharpe': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis()
        }
        
        return {
            'var_result': var_result,
            'correlation_analysis': correlation_analysis,
            'diversification_ratio': diversification_ratio,
            'marginal_var': marginal_var,
            'component_var': component_var,
            'portfolio_stats': portfolio_stats,
            'portfolio_returns': portfolio_returns
        }
    
    def _calculate_portfolio_returns(self, strategy_returns: pd.DataFrame,
                                   strategy_weights: Dict[str, float]) -> pd.Series:
        """Calculate weighted portfolio returns."""
        # Align weights with available data
        available_strategies = [s for s in strategy_weights.keys() 
                              if s in strategy_returns.columns]
        
        if not available_strategies:
            return pd.Series(dtype=float)
        
        # Normalize weights for available strategies
        total_weight = sum(strategy_weights[s] for s in available_strategies)
        if total_weight == 0:
            return pd.Series(dtype=float)
        
        normalized_weights = {s: strategy_weights[s] / total_weight 
                            for s in available_strategies}
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(0.0, index=strategy_returns.index)
        for strategy, weight in normalized_weights.items():
            strategy_data = strategy_returns[strategy].fillna(0)
            portfolio_returns += weight * strategy_data
        
        return portfolio_returns.dropna()
    
    def _calculate_diversification_ratio(self, strategy_returns: pd.DataFrame,
                                       strategy_weights: Dict[str, float],
                                       correlation_analysis: CorrelationAnalysis) -> float:
        """Calculate portfolio diversification ratio."""
        available_strategies = [s for s in strategy_weights.keys() 
                              if s in strategy_returns.columns]
        
        if len(available_strategies) <= 1:
            return 1.0
        
        # Calculate individual volatilities
        individual_vols = {}
        for strategy in available_strategies:
            vol = strategy_returns[strategy].std()
            individual_vols[strategy] = vol if not np.isnan(vol) else 0.0
        
        # Weighted average of individual volatilities
        total_weight = sum(strategy_weights[s] for s in available_strategies)
        if total_weight == 0:
            return 1.0
        
        weighted_avg_vol = sum(strategy_weights[s] / total_weight * individual_vols[s] 
                              for s in available_strategies)
        
        # Portfolio volatility
        portfolio_returns = self._calculate_portfolio_returns(strategy_returns, strategy_weights)
        portfolio_vol = portfolio_returns.std()
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_avg_vol / portfolio_vol
    
    def _calculate_marginal_var(self, strategy_returns: pd.DataFrame,
                              strategy_weights: Dict[str, float],
                              portfolio_returns: pd.Series,
                              portfolio_var: float) -> Dict[str, float]:
        """Calculate marginal VaR contribution of each strategy."""
        marginal_var = {}
        
        for strategy in strategy_weights.keys():
            if strategy not in strategy_returns.columns:
                marginal_var[strategy] = 0.0
                continue
            
            strategy_data = strategy_returns[strategy].fillna(0)
            
            # Calculate correlation with portfolio
            aligned_data = pd.DataFrame({
                'portfolio': portfolio_returns,
                'strategy': strategy_data
            }).dropna()
            
            if len(aligned_data) < 10:
                marginal_var[strategy] = 0.0
                continue
            
            correlation = aligned_data.corr().iloc[0, 1]
            strategy_vol = aligned_data['strategy'].std()
            portfolio_vol = aligned_data['portfolio'].std()
            
            if portfolio_vol == 0:
                marginal_var[strategy] = 0.0
            else:
                # Marginal VaR = correlation * strategy_vol / portfolio_vol * portfolio_var
                marginal_var[strategy] = correlation * strategy_vol / portfolio_vol * portfolio_var
        
        return marginal_var
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return abs(drawdown.min())