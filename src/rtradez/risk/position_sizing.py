"""
Position sizing algorithms for risk-adjusted capital allocation.

Implements various position sizing methodologies including Kelly Criterion,
Fixed Fraction, and volatility-adjusted sizing for options strategies.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import logging
from pydantic import BaseModel, Field, validator, ConfigDict

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    """Position sizing methodology."""
    KELLY = "kelly"
    FIXED_FRACTION = "fixed_fraction"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"

class PositionSizeResult(BaseModel):
    """Result of position sizing calculation with validation."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True
    )
    
    strategy_name: str = Field(..., min_length=1, description="Strategy identifier")
    recommended_size: float = Field(..., ge=0, description="Recommended position size in dollars")
    max_position_value: float = Field(..., ge=0, description="Maximum allowable position value")
    risk_adjusted_size: float = Field(..., ge=0, description="Risk-adjusted position size")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence in the sizing recommendation")
    reasoning: str = Field(..., min_length=1, description="Explanation of sizing logic")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or concerns")
    
    @validator('risk_adjusted_size')
    def risk_adjusted_must_not_exceed_max(cls, v, values):
        if 'max_position_value' in values and v > values['max_position_value']:
            raise ValueError('Risk adjusted size cannot exceed maximum position value')
        return v
    
    @validator('recommended_size')
    def recommended_size_reasonable(cls, v):
        if v > 1e9:  # $1B sanity check
            raise ValueError('Recommended size appears unreasonably large')
        return v

class PositionSizerConfig(BaseModel):
    """Configuration for position sizing algorithms."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    total_capital: float = Field(..., gt=0, description="Total available capital")
    max_risk_per_trade: float = Field(0.02, gt=0, le=1, description="Maximum risk per trade as fraction")
    
    @validator('total_capital')
    def capital_reasonable(cls, v):
        if v < 1000:  # Minimum $1k
            raise ValueError('Total capital should be at least $1,000')
        if v > 1e12:  # $1T sanity check
            raise ValueError('Total capital appears unreasonably large')
        return v

class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms."""
    
    def __init__(self, config: PositionSizerConfig):
        """Initialize position sizer with validated configuration."""
        self.config = config
        
    @abstractmethod
    def calculate_position_size(self, strategy_name: str, 
                              expected_return: float,
                              volatility: float,
                              **kwargs) -> PositionSizeResult:
        """Calculate position size for given strategy parameters."""
        pass
        
    def _validate_inputs(self, expected_return: float, volatility: float) -> List[str]:
        """Validate input parameters and return warnings."""
        warnings = []
        
        if volatility <= 0:
            warnings.append("Volatility must be positive")
        if volatility > 1.0:
            warnings.append("Very high volatility detected")
        if abs(expected_return) > 0.5:
            warnings.append("Unusually high expected return")
            
        return warnings

class KellyConfig(PositionSizerConfig):
    """Configuration for Kelly Criterion position sizing."""
    confidence_threshold: float = Field(0.6, gt=0, le=1, description="Minimum confidence for Kelly sizing")
    max_kelly_fraction: float = Field(0.25, gt=0, le=1, description="Maximum Kelly fraction to prevent over-leverage")

class KellyCriterion(PositionSizer):
    """Kelly Criterion for optimal position sizing."""
    
    def __init__(self, config: KellyConfig):
        """
        Initialize Kelly Criterion sizer.
        
        Args:
            config: Kelly-specific configuration
        """
        super().__init__(config)
        self.confidence_threshold = config.confidence_threshold
        self.max_kelly_fraction = config.max_kelly_fraction
        
    def calculate_position_size(self, strategy_name: str,
                              expected_return: float,
                              volatility: float,
                              win_rate: Optional[float] = None,
                              avg_win: Optional[float] = None,
                              avg_loss: Optional[float] = None,
                              **kwargs) -> PositionSizeResult:
        """
        Calculate Kelly optimal position size.
        
        Args:
            strategy_name: Name of the strategy
            expected_return: Expected return of the strategy
            volatility: Volatility of returns
            win_rate: Historical win rate (optional)
            avg_win: Average winning trade (optional)
            avg_loss: Average losing trade (optional)
        """
        warnings = self._validate_inputs(expected_return, volatility)
        
        # Kelly formula: f = (bp - q) / b
        # where f = fraction of capital, b = odds, p = win probability, q = loss probability
        
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            # Use discrete Kelly formula
            if avg_loss == 0:
                warnings.append("Average loss is zero, using continuous Kelly")
                kelly_fraction = self._continuous_kelly(expected_return, volatility)
            else:
                b = avg_win / abs(avg_loss)  # odds
                p = win_rate
                q = 1 - win_rate
                kelly_fraction = (b * p - q) / b
        else:
            # Use continuous Kelly formula
            kelly_fraction = self._continuous_kelly(expected_return, volatility)
        
        # Apply safety constraints
        kelly_fraction = max(0, min(kelly_fraction, self.max_kelly_fraction))
        
        # Calculate position size
        raw_position_size = kelly_fraction * self.config.total_capital
        
        # Apply maximum risk constraint
        max_risk_position = self.config.max_risk_per_trade * self.config.total_capital / volatility
        final_position_size = min(raw_position_size, max_risk_position)
        
        # Calculate confidence based on Sharpe ratio
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        confidence = min(0.95, max(0.1, abs(sharpe_ratio) / 3.0))
        
        reasoning = f"Kelly fraction: {kelly_fraction:.3f}, Sharpe: {sharpe_ratio:.3f}"
        if final_position_size < raw_position_size:
            reasoning += f", Risk-limited from {raw_position_size:.0f}"
            
        if confidence < self.confidence_threshold:
            warnings.append(f"Low confidence ({confidence:.2f}) in Kelly estimate")
        
        return PositionSizeResult(
            strategy_name=strategy_name,
            recommended_size=final_position_size,
            max_position_value=max_risk_position,
            risk_adjusted_size=final_position_size,
            confidence_level=confidence,
            reasoning=reasoning,
            warnings=warnings
        )
    
    def _continuous_kelly(self, expected_return: float, volatility: float) -> float:
        """Calculate Kelly fraction using continuous formula."""
        if volatility == 0:
            return 0
        return expected_return / (volatility ** 2)

class FixedFractionConfig(PositionSizerConfig):
    """Configuration for Fixed Fraction position sizing."""
    base_fraction: float = Field(0.05, gt=0, le=1, description="Base fraction of capital to risk")
    volatility_target: float = Field(0.15, gt=0, description="Target volatility for scaling")

class FixedFractionSizer(PositionSizer):
    """Fixed fraction position sizing with volatility adjustment."""
    
    def __init__(self, config: FixedFractionConfig):
        """
        Initialize fixed fraction sizer.
        
        Args:
            config: Fixed fraction configuration
        """
        super().__init__(config)
        self.base_fraction = config.base_fraction
        self.volatility_target = config.volatility_target
        
    def calculate_position_size(self, strategy_name: str,
                              expected_return: float,
                              volatility: float,
                              **kwargs) -> PositionSizeResult:
        """Calculate fixed fraction position size with volatility adjustment."""
        warnings = self._validate_inputs(expected_return, volatility)
        
        # Volatility adjustment
        vol_adjustment = self.volatility_target / volatility if volatility > 0 else 1.0
        vol_adjustment = min(2.0, max(0.5, vol_adjustment))  # Cap adjustment
        
        # Calculate position size
        adjusted_fraction = self.base_fraction * vol_adjustment
        raw_position_size = adjusted_fraction * self.config.total_capital
        
        # Apply maximum risk constraint
        max_risk_position = self.config.max_risk_per_trade * self.config.total_capital / volatility
        final_position_size = min(raw_position_size, max_risk_position)
        
        # Confidence based on volatility stability
        confidence = min(0.9, max(0.3, 1.0 - abs(volatility - self.volatility_target)))
        
        reasoning = f"Base fraction: {self.base_fraction:.3f}, Vol adj: {vol_adjustment:.3f}"
        if final_position_size < raw_position_size:
            reasoning += f", Risk-limited from {raw_position_size:.0f}"
        
        return PositionSizeResult(
            strategy_name=strategy_name,
            recommended_size=final_position_size,
            max_position_value=max_risk_position,
            risk_adjusted_size=final_position_size,
            confidence_level=confidence,
            reasoning=reasoning,
            warnings=warnings
        )

class VolatilityAdjustedConfig(PositionSizerConfig):
    """Configuration for Volatility Adjusted position sizing."""
    lookback_window: int = Field(252, gt=0, description="Lookback period for volatility estimation")
    vol_floor: float = Field(0.05, gt=0, le=1, description="Minimum volatility to prevent extreme sizing")

class VolatilityAdjustedSizer(PositionSizer):
    """Position sizing based on inverse volatility weighting."""
    
    def __init__(self, config: VolatilityAdjustedConfig):
        """
        Initialize volatility-adjusted sizer.
        
        Args:
            config: Volatility adjusted configuration
        """
        super().__init__(config)
        self.lookback_window = config.lookback_window
        self.vol_floor = config.vol_floor
        
    def calculate_position_size(self, strategy_name: str,
                              expected_return: float,
                              volatility: float,
                              returns_history: Optional[pd.Series] = None,
                              **kwargs) -> PositionSizeResult:
        """Calculate volatility-adjusted position size."""
        warnings = self._validate_inputs(expected_return, volatility)
        
        # Use historical volatility if available
        if returns_history is not None and len(returns_history) >= 20:
            realized_vol = returns_history.std() * np.sqrt(252)
            vol_to_use = max(realized_vol, self.vol_floor)
            if abs(realized_vol - volatility) > 0.05:
                warnings.append(f"Realized vol ({realized_vol:.3f}) differs from input")
        else:
            vol_to_use = max(volatility, self.vol_floor)
        
        # Inverse volatility weighting
        base_risk = self.config.max_risk_per_trade * self.config.total_capital
        position_size = base_risk / vol_to_use
        
        # Confidence based on volatility estimation quality
        if returns_history is not None:
            vol_stability = 1.0 - returns_history.rolling(21).std().std()
            confidence = min(0.9, max(0.2, vol_stability))
        else:
            confidence = 0.5
        
        reasoning = f"Vol-adjusted sizing: {vol_to_use:.3f} vol, {base_risk:.0f} risk"
        
        return PositionSizeResult(
            strategy_name=strategy_name,
            recommended_size=position_size,
            max_position_value=position_size,
            risk_adjusted_size=position_size,
            confidence_level=confidence,
            reasoning=reasoning,
            warnings=warnings
        )

class MultiStrategyConfig(BaseModel):
    """Configuration for multi-strategy position sizing."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    total_capital: float = Field(..., gt=0, description="Total available capital")
    max_total_risk: float = Field(0.15, gt=0, le=1, description="Maximum total portfolio risk")
    correlation_matrix: Optional[List[List[float]]] = Field(None, description="Strategy correlation matrix")
    
    @validator('correlation_matrix')
    def validate_correlation_matrix(cls, v):
        if v is not None:
            # Convert to numpy and validate
            matrix = np.array(v)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError('Correlation matrix must be square')
            if not np.allclose(matrix, matrix.T):
                raise ValueError('Correlation matrix must be symmetric')
            if not np.all(np.diag(matrix) == 1.0):
                raise ValueError('Correlation matrix diagonal must be 1.0')
        return v

class MultiStrategyPositionSizer:
    """Position sizer for multiple strategies with portfolio-level constraints."""
    
    def __init__(self, config: MultiStrategyConfig):
        """
        Initialize multi-strategy position sizer.
        
        Args:
            config: Multi-strategy configuration
        """
        self.config = config
        self.correlation_matrix = (np.array(config.correlation_matrix) 
                                 if config.correlation_matrix else None)
        self.strategy_sizings: Dict[str, PositionSizeResult] = {}
        
    def add_strategy_sizing(self, result: PositionSizeResult):
        """Add a strategy position sizing result."""
        self.strategy_sizings[result.strategy_name] = result
        
    def optimize_portfolio_allocation(self) -> Dict[str, PositionSizeResult]:
        """Optimize position sizes across all strategies considering correlations."""
        if len(self.strategy_sizings) == 0:
            return {}
        
        strategies = list(self.strategy_sizings.keys())
        individual_sizes = np.array([self.strategy_sizings[s].recommended_size 
                                   for s in strategies])
        
        # If no correlation matrix, assume independence
        if self.correlation_matrix is None:
            correlation_matrix = np.eye(len(strategies))
        else:
            correlation_matrix = self.correlation_matrix
        
        # Calculate portfolio risk
        weights = individual_sizes / self.config.total_capital
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Scale down if portfolio risk exceeds limit
        if portfolio_risk > self.config.max_total_risk:
            scale_factor = self.config.max_total_risk / portfolio_risk
            logger.info(f"Scaling down positions by {scale_factor:.3f} due to portfolio risk")
            
            for strategy_name in strategies:
                result = self.strategy_sizings[strategy_name]
                scaled_size = result.recommended_size * scale_factor
                
                # Update the result
                self.strategy_sizings[strategy_name] = PositionSizeResult(
                    strategy_name=result.strategy_name,
                    recommended_size=scaled_size,
                    max_position_value=result.max_position_value,
                    risk_adjusted_size=scaled_size,
                    confidence_level=result.confidence_level,
                    reasoning=result.reasoning + f", Portfolio scaled: {scale_factor:.3f}",
                    warnings=result.warnings + [f"Portfolio risk adjustment applied"]
                )
        
        return self.strategy_sizings

def create_position_sizer(method: SizingMethod, config: PositionSizerConfig) -> PositionSizer:
    """Factory function to create position sizers."""
    if method == SizingMethod.KELLY:
        if not isinstance(config, KellyConfig):
            # Convert base config to Kelly config
            kelly_config = KellyConfig(
                total_capital=config.total_capital,
                max_risk_per_trade=config.max_risk_per_trade
            )
            return KellyCriterion(kelly_config)
        return KellyCriterion(config)
    elif method == SizingMethod.FIXED_FRACTION:
        if not isinstance(config, FixedFractionConfig):
            # Convert base config to Fixed Fraction config
            ff_config = FixedFractionConfig(
                total_capital=config.total_capital,
                max_risk_per_trade=config.max_risk_per_trade
            )
            return FixedFractionSizer(ff_config)
        return FixedFractionSizer(config)
    elif method == SizingMethod.VOLATILITY_ADJUSTED:
        if not isinstance(config, VolatilityAdjustedConfig):
            # Convert base config to Volatility Adjusted config
            va_config = VolatilityAdjustedConfig(
                total_capital=config.total_capital,
                max_risk_per_trade=config.max_risk_per_trade
            )
            return VolatilityAdjustedSizer(va_config)
        return VolatilityAdjustedSizer(config)
    else:
        raise ValueError(f"Unknown sizing method: {method}")