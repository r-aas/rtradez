"""Greeks analysis and risk management for options strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.stats import norm
import warnings

from ..base import BaseStrategy, BaseTransformer
from .advanced_backtest import OptionsContract, OptionType


@dataclass
class GreeksProfile:
    """Greeks profile for a portfolio or position."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    delta_dollars: float = 0.0
    gamma_dollars: float = 0.0
    theta_dollars: float = 0.0
    vega_dollars: float = 0.0


class BlackScholesCalculator:
    """Black-Scholes option pricing and Greeks calculator."""
    
    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        """
        if T <= 0:
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(0, price)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        if T <= 0:
            return {
                'delta': 1.0 if (option_type == 'call' and S > K) or (option_type == 'put' and S < K) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Common calculations
        phi_d1 = norm.pdf(d1)
        Phi_d1 = norm.cdf(d1)
        Phi_d2 = norm.cdf(d2)
        
        # Delta
        if option_type == 'call':
            delta = Phi_d1
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = phi_d1 / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_common = -(S * phi_d1 * sigma) / (2 * np.sqrt(T))
        if option_type == 'call':
            theta = theta_common - r * K * np.exp(-r * T) * Phi_d2
        else:
            theta = theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert theta to per-day
        theta = theta / 365
        
        # Vega (same for calls and puts)
        vega = S * phi_d1 * np.sqrt(T) / 100  # Per 1% change in volatility
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * Phi_d2 / 100  # Per 1% change in rate
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = 'call', 
                          max_iterations: int = 100, precision: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        """
        if T <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.2
        
        for i in range(max_iterations):
            price = BlackScholesCalculator.black_scholes_price(S, K, T, r, sigma, option_type)
            
            if abs(price - market_price) < precision:
                return sigma
            
            # Calculate vega for Newton-Raphson
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            if vega == 0:
                break
            
            # Newton-Raphson update
            sigma = sigma - (price - market_price) / vega
            
            # Keep sigma positive
            sigma = max(0.001, sigma)
        
        return sigma


class GreeksAnalyzer(BaseTransformer):
    """
    Comprehensive Greeks analysis for options portfolios.
    
    Provides:
    - Portfolio Greeks calculation
    - Risk exposure analysis
    - Greeks-based position sizing
    - Dynamic hedging recommendations
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.05,
                 underlying_price_col: str = 'Close',
                 volatility_method: str = 'historical'):
        """
        Initialize Greeks analyzer.
        
        Args:
            risk_free_rate: Risk-free interest rate
            underlying_price_col: Column name for underlying price
            volatility_method: Method for volatility estimation
        """
        self.risk_free_rate = risk_free_rate
        self.underlying_price_col = underlying_price_col
        self.volatility_method = volatility_method
        self.bs_calculator = BlackScholesCalculator()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'GreeksAnalyzer':
        """Fit Greeks analyzer to historical data."""
        if self.underlying_price_col not in X.columns:
            raise ValueError(f"Column {self.underlying_price_col} not found in data")
        
        # Calculate historical volatility
        returns = X[self.underlying_price_col].pct_change().dropna()
        self.historical_volatility_ = returns.std() * np.sqrt(252)
        
        # Store current price level for Greeks calculations
        self.current_price_ = X[self.underlying_price_col].iloc[-1]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add Greeks analysis to market data."""
        X_transformed = X.copy()
        
        # Add volatility estimate
        if self.volatility_method == 'historical':
            X_transformed['volatility'] = self.historical_volatility_
        else:
            # Use rolling volatility
            returns = X[self.underlying_price_col].pct_change()
            X_transformed['volatility'] = returns.rolling(20).std() * np.sqrt(252)
        
        return X_transformed
    
    def analyze_portfolio_greeks(self, positions: List[Tuple[OptionsContract, int]], 
                               current_price: float, current_vol: float = None) -> GreeksProfile:
        """
        Analyze Greeks for an entire options portfolio.
        
        Args:
            positions: List of (contract, quantity) tuples
            current_price: Current underlying price
            current_vol: Current volatility estimate
        
        Returns:
            Portfolio Greeks profile
        """
        if current_vol is None:
            current_vol = getattr(self, 'historical_volatility_', 0.2)
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        total_delta_dollars = 0
        total_gamma_dollars = 0
        total_theta_dollars = 0
        total_vega_dollars = 0
        
        for contract, quantity in positions:
            # Calculate time to expiration
            # For demo, assume 30 days to expiration
            time_to_expiration = 30 / 365
            
            # Calculate Greeks for this position
            greeks = self.bs_calculator.calculate_greeks(
                S=current_price,
                K=contract.strike,
                T=time_to_expiration,
                r=self.risk_free_rate,
                sigma=current_vol,
                option_type=contract.option_type.value
            )
            
            # Aggregate portfolio Greeks
            position_delta = greeks['delta'] * quantity
            position_gamma = greeks['gamma'] * quantity
            position_theta = greeks['theta'] * quantity
            position_vega = greeks['vega'] * quantity
            position_rho = greeks['rho'] * quantity
            
            total_delta += position_delta
            total_gamma += position_gamma
            total_theta += position_theta
            total_vega += position_vega
            total_rho += position_rho
            
            # Dollar Greeks (Greeks * underlying price * contract multiplier)
            contract_multiplier = 100  # Standard options contract
            total_delta_dollars += position_delta * current_price * contract_multiplier
            total_gamma_dollars += position_gamma * current_price * contract_multiplier
            total_theta_dollars += position_theta * contract_multiplier
            total_vega_dollars += position_vega * contract_multiplier
        
        return GreeksProfile(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho,
            delta_dollars=total_delta_dollars,
            gamma_dollars=total_gamma_dollars,
            theta_dollars=total_theta_dollars,
            vega_dollars=total_vega_dollars
        )
    
    def calculate_hedge_recommendations(self, portfolio_greeks: GreeksProfile,
                                      hedge_instrument: OptionsContract,
                                      current_price: float) -> Dict[str, float]:
        """
        Calculate hedge recommendations to neutralize Greeks exposure.
        
        Args:
            portfolio_greeks: Current portfolio Greeks
            hedge_instrument: Instrument to use for hedging
            current_price: Current underlying price
        
        Returns:
            Hedge recommendations
        """
        # Calculate Greeks for hedge instrument
        time_to_expiration = 30 / 365  # Assume 30 days
        hedge_greeks = self.bs_calculator.calculate_greeks(
            S=current_price,
            K=hedge_instrument.strike,
            T=time_to_expiration,
            r=self.risk_free_rate,
            sigma=getattr(self, 'historical_volatility_', 0.2),
            option_type=hedge_instrument.option_type.value
        )
        
        recommendations = {}
        
        # Delta hedge
        if abs(hedge_greeks['delta']) > 0.01:
            delta_hedge_quantity = -portfolio_greeks.delta / hedge_greeks['delta']
            recommendations['delta_hedge'] = delta_hedge_quantity
        
        # Gamma hedge
        if abs(hedge_greeks['gamma']) > 0.001:
            gamma_hedge_quantity = -portfolio_greeks.gamma / hedge_greeks['gamma']
            recommendations['gamma_hedge'] = gamma_hedge_quantity
        
        # Vega hedge
        if abs(hedge_greeks['vega']) > 0.01:
            vega_hedge_quantity = -portfolio_greeks.vega / hedge_greeks['vega']
            recommendations['vega_hedge'] = vega_hedge_quantity
        
        return recommendations
    
    def analyze_greeks_pnl_attribution(self, initial_greeks: GreeksProfile,
                                     price_change: float, vol_change: float,
                                     time_decay: float) -> Dict[str, float]:
        """
        Attribute P&L to different Greeks exposures.
        
        Args:
            initial_greeks: Greeks at start of period
            price_change: Change in underlying price
            vol_change: Change in volatility
            time_decay: Time elapsed (in days)
        
        Returns:
            P&L attribution by Greek
        """
        pnl_attribution = {}
        
        # Delta P&L: Delta * price change * 100 (contract multiplier)
        pnl_attribution['delta_pnl'] = initial_greeks.delta * price_change * 100
        
        # Gamma P&L: 0.5 * Gamma * (price change)^2 * 100
        pnl_attribution['gamma_pnl'] = 0.5 * initial_greeks.gamma * (price_change ** 2) * 100
        
        # Theta P&L: Theta * time decay * 100
        pnl_attribution['theta_pnl'] = initial_greeks.theta * time_decay * 100
        
        # Vega P&L: Vega * volatility change * 100
        pnl_attribution['vega_pnl'] = initial_greeks.vega * vol_change * 100
        
        # Total P&L
        pnl_attribution['total_pnl'] = sum(pnl_attribution.values())
        
        return pnl_attribution


class DeltaHedger(BaseStrategy):
    """
    Strategy that maintains delta-neutral positions through dynamic hedging.
    """
    
    def __init__(self,
                 rebalance_threshold: float = 0.1,
                 hedge_instrument: str = 'underlying',
                 target_delta: float = 0.0):
        """
        Initialize delta hedger.
        
        Args:
            rebalance_threshold: Delta threshold for rebalancing
            hedge_instrument: Instrument to use for hedging ('underlying', 'options')
            target_delta: Target portfolio delta
        """
        super().__init__()
        self.rebalance_threshold = rebalance_threshold
        self.hedge_instrument = hedge_instrument
        self.target_delta = target_delta
        self.greeks_analyzer = GreeksAnalyzer()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DeltaHedger':
        """Fit delta hedger to historical data."""
        self.greeks_analyzer.fit(X, y)
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate delta hedge signals."""
        if not self._fitted:
            raise ValueError("Must fit before predict")
        
        # This would integrate with portfolio positions to generate hedge signals
        # For demo, return simple signals
        signals = np.zeros(len(X))
        return signals
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Score delta hedging performance."""
        # Calculate how well the strategy maintains delta neutrality
        return 0.0  # Placeholder


class GammaScalper(BaseStrategy):
    """
    Strategy that profits from gamma exposure through scalping.
    """
    
    def __init__(self,
                 scalping_threshold: float = 0.005,
                 gamma_target: float = 1000,
                 rebalance_frequency: int = 1):
        """
        Initialize gamma scalper.
        
        Args:
            scalping_threshold: Price movement threshold for scalping
            gamma_target: Target gamma exposure
            rebalance_frequency: Days between rebalances
        """
        super().__init__()
        self.scalping_threshold = scalping_threshold
        self.gamma_target = gamma_target
        self.rebalance_frequency = rebalance_frequency
        self.greeks_analyzer = GreeksAnalyzer()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'GammaScalper':
        """Fit gamma scalper to historical data."""
        self.greeks_analyzer.fit(X, y)
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate gamma scalping signals."""
        if not self._fitted:
            raise ValueError("Must fit before predict")
        
        # Calculate price changes
        returns = X['Close'].pct_change()
        
        # Generate scalping signals based on price movements
        signals = np.zeros(len(X))
        signals[abs(returns) > self.scalping_threshold] = 1
        
        return signals
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Score gamma scalping performance."""
        predictions = self.predict(X)
        strategy_returns = predictions * y.shift(-1)
        return self._calculate_sharpe_ratio(X, strategy_returns)