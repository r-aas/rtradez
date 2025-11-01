"""
Risk limit management and monitoring system.

Implements portfolio-level and position-level risk limits with real-time monitoring
and automatic position sizing adjustments for options trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RiskLimitType(Enum):
    """Types of risk limits."""
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    VAR = "var"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"

class LimitSeverity(Enum):
    """Severity levels for limit breaches."""
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RiskLimit:
    """Individual risk limit definition."""
    limit_type: RiskLimitType
    threshold: float
    severity: LimitSeverity
    description: str
    enabled: bool = True
    lookback_period: Optional[int] = None  # Days for time-based limits

@dataclass
class LimitBreach:
    """Risk limit breach event."""
    limit_type: RiskLimitType
    severity: LimitSeverity
    current_value: float
    threshold: float
    breach_percentage: float
    timestamp: datetime
    affected_strategies: List[str]
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioRiskLimits:
    """Portfolio-level risk limits configuration."""
    max_total_exposure: float = 1.0  # As fraction of capital
    max_position_concentration: float = 0.20  # Max single position size
    max_sector_concentration: float = 0.40  # Max exposure to single sector
    max_daily_var: float = 0.05  # Max daily VaR
    max_drawdown: float = 0.15  # Maximum drawdown limit
    max_leverage: float = 2.0  # Maximum leverage ratio
    min_cash_reserve: float = 0.10  # Minimum cash reserve
    max_correlation_exposure: float = 0.60  # Max correlated position exposure

@dataclass 
class PositionRiskLimits:
    """Individual position risk limits."""
    max_position_size: float = 0.10  # As fraction of capital
    max_delta_exposure: float = 0.05  # Max delta per position
    max_gamma_exposure: float = 0.02  # Max gamma per position
    max_theta_decay: float = 0.01  # Max daily theta decay
    max_vega_exposure: float = 0.03  # Max vega per position
    min_time_to_expiry: int = 7  # Minimum days to expiry
    max_bid_ask_spread: float = 0.10  # Maximum bid-ask spread

class RiskLimitManager:
    """Manages and monitors all risk limits across the portfolio."""
    
    def __init__(self, portfolio_limits: PortfolioRiskLimits,
                 position_limits: PositionRiskLimits,
                 total_capital: float):
        """
        Initialize risk limit manager.
        
        Args:
            portfolio_limits: Portfolio-level risk limits
            position_limits: Position-level risk limits
            total_capital: Total available capital
        """
        self.portfolio_limits = portfolio_limits
        self.position_limits = position_limits
        self.total_capital = total_capital
        
        # Active limits tracking
        self.active_limits: Dict[str, RiskLimit] = {}
        self.breach_history: List[LimitBreach] = []
        self.current_breaches: List[LimitBreach] = []
        
        # Initialize default limits
        self._initialize_default_limits()
        
    def _initialize_default_limits(self):
        """Initialize default risk limits."""
        # Portfolio limits
        self.add_limit(RiskLimit(
            RiskLimitType.DRAWDOWN,
            self.portfolio_limits.max_drawdown,
            LimitSeverity.CRITICAL,
            "Maximum portfolio drawdown limit"
        ))
        
        self.add_limit(RiskLimit(
            RiskLimitType.VAR,
            self.portfolio_limits.max_daily_var,
            LimitSeverity.WARNING,
            "Maximum daily Value at Risk"
        ))
        
        self.add_limit(RiskLimit(
            RiskLimitType.CONCENTRATION,
            self.portfolio_limits.max_position_concentration,
            LimitSeverity.WARNING,
            "Maximum single position concentration"
        ))
        
        self.add_limit(RiskLimit(
            RiskLimitType.LEVERAGE,
            self.portfolio_limits.max_leverage,
            LimitSeverity.CRITICAL,
            "Maximum portfolio leverage"
        ))
        
    def add_limit(self, limit: RiskLimit):
        """Add a new risk limit."""
        limit_key = f"{limit.limit_type.value}_{limit.severity.value}"
        self.active_limits[limit_key] = limit
        logger.info(f"Added risk limit: {limit.description}")
        
    def remove_limit(self, limit_type: RiskLimitType, severity: LimitSeverity):
        """Remove a risk limit."""
        limit_key = f"{limit_type.value}_{severity.value}"
        if limit_key in self.active_limits:
            del self.active_limits[limit_key]
            logger.info(f"Removed risk limit: {limit_type.value}")
            
    def check_portfolio_limits(self, portfolio_positions: Dict[str, Any],
                             portfolio_pnl: pd.Series) -> List[LimitBreach]:
        """Check all portfolio-level risk limits."""
        breaches = []
        
        # Calculate portfolio metrics
        total_exposure = sum(abs(pos.get('market_value', 0)) for pos in portfolio_positions.values())
        exposure_ratio = total_exposure / self.total_capital
        
        # Check drawdown
        if len(portfolio_pnl) > 0:
            peak = portfolio_pnl.expanding().max()
            drawdown = (portfolio_pnl - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            if max_drawdown > self.portfolio_limits.max_drawdown:
                breaches.append(self._create_breach(
                    RiskLimitType.DRAWDOWN,
                    LimitSeverity.CRITICAL,
                    max_drawdown,
                    self.portfolio_limits.max_drawdown,
                    list(portfolio_positions.keys()),
                    "Reduce position sizes or close losing positions"
                ))
        
        # Check total exposure
        if exposure_ratio > self.portfolio_limits.max_total_exposure:
            breaches.append(self._create_breach(
                RiskLimitType.POSITION_SIZE,
                LimitSeverity.WARNING,
                exposure_ratio,
                self.portfolio_limits.max_total_exposure,
                list(portfolio_positions.keys()),
                "Reduce overall position sizes"
            ))
        
        # Check concentration limits
        if portfolio_positions:
            position_values = [abs(pos.get('market_value', 0)) for pos in portfolio_positions.values()]
            max_position = max(position_values) if position_values else 0
            concentration = max_position / self.total_capital
            
            if concentration > self.portfolio_limits.max_position_concentration:
                # Find the largest position
                largest_position = max(portfolio_positions.items(), 
                                     key=lambda x: abs(x[1].get('market_value', 0)))
                
                breaches.append(self._create_breach(
                    RiskLimitType.CONCENTRATION,
                    LimitSeverity.WARNING,
                    concentration,
                    self.portfolio_limits.max_position_concentration,
                    [largest_position[0]],
                    f"Reduce size of largest position: {largest_position[0]}"
                ))
        
        return breaches
    
    def check_position_limits(self, strategy_name: str, position_data: Dict[str, Any]) -> List[LimitBreach]:
        """Check position-level risk limits for a specific strategy."""
        breaches = []
        
        # Extract position metrics
        position_size = abs(position_data.get('market_value', 0))
        delta = position_data.get('delta', 0)
        gamma = position_data.get('gamma', 0)
        theta = position_data.get('theta', 0)
        vega = position_data.get('vega', 0)
        days_to_expiry = position_data.get('days_to_expiry', 365)
        bid_ask_spread = position_data.get('bid_ask_spread', 0)
        
        # Check position size
        size_ratio = position_size / self.total_capital
        if size_ratio > self.position_limits.max_position_size:
            breaches.append(self._create_breach(
                RiskLimitType.POSITION_SIZE,
                LimitSeverity.WARNING,
                size_ratio,
                self.position_limits.max_position_size,
                [strategy_name],
                "Reduce position size"
            ))
        
        # Check Greeks limits
        delta_ratio = abs(delta * position_size) / self.total_capital
        if delta_ratio > self.position_limits.max_delta_exposure:
            breaches.append(self._create_breach(
                RiskLimitType.VAR,  # Using VAR as proxy for Greeks risk
                LimitSeverity.WARNING,
                delta_ratio,
                self.position_limits.max_delta_exposure,
                [strategy_name],
                "Hedge delta exposure or reduce position"
            ))
        
        # Check time to expiry
        if days_to_expiry < self.position_limits.min_time_to_expiry:
            breaches.append(self._create_breach(
                RiskLimitType.VOLATILITY,  # Using volatility as proxy for time risk
                LimitSeverity.WARNING,
                days_to_expiry,
                self.position_limits.min_time_to_expiry,
                [strategy_name],
                "Close or roll position approaching expiry"
            ))
        
        return breaches
    
    def _create_breach(self, limit_type: RiskLimitType, severity: LimitSeverity,
                      current_value: float, threshold: float,
                      affected_strategies: List[str], recommended_action: str) -> LimitBreach:
        """Create a limit breach object."""
        breach_percentage = (current_value - threshold) / threshold * 100
        
        return LimitBreach(
            limit_type=limit_type,
            severity=severity,
            current_value=current_value,
            threshold=threshold,
            breach_percentage=breach_percentage,
            timestamp=datetime.now(),
            affected_strategies=affected_strategies,
            recommended_action=recommended_action
        )
    
    def update_breaches(self, new_breaches: List[LimitBreach]):
        """Update current breaches and log to history."""
        # Clear resolved breaches
        self.current_breaches = [b for b in self.current_breaches 
                               if any(nb.limit_type == b.limit_type and nb.severity == b.severity 
                                     for nb in new_breaches)]
        
        # Add new breaches
        for breach in new_breaches:
            if not any(b.limit_type == breach.limit_type and b.severity == breach.severity 
                      for b in self.current_breaches):
                self.current_breaches.append(breach)
                self.breach_history.append(breach)
                logger.warning(f"Risk limit breach: {breach.limit_type.value} "
                             f"({breach.severity.value}) - {breach.recommended_action}")
    
    def get_active_breaches(self, severity: Optional[LimitSeverity] = None) -> List[LimitBreach]:
        """Get currently active breaches, optionally filtered by severity."""
        if severity is None:
            return self.current_breaches.copy()
        return [b for b in self.current_breaches if b.severity == severity]
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            'total_active_breaches': len(self.current_breaches),
            'critical_breaches': len([b for b in self.current_breaches 
                                    if b.severity == LimitSeverity.CRITICAL]),
            'warning_breaches': len([b for b in self.current_breaches 
                                   if b.severity == LimitSeverity.WARNING]),
            'emergency_breaches': len([b for b in self.current_breaches 
                                     if b.severity == LimitSeverity.EMERGENCY]),
            'breach_types': list(set(b.limit_type.value for b in self.current_breaches)),
            'affected_strategies': list(set(s for b in self.current_breaches 
                                          for s in b.affected_strategies)),
            'total_historical_breaches': len(self.breach_history),
            'last_breach_time': max([b.timestamp for b in self.current_breaches]) 
                              if self.current_breaches else None
        }
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be halted based on current breaches."""
        emergency_breaches = [b for b in self.current_breaches 
                            if b.severity == LimitSeverity.EMERGENCY]
        critical_breaches = [b for b in self.current_breaches 
                           if b.severity == LimitSeverity.CRITICAL]
        
        if emergency_breaches:
            return True, f"Emergency breach: {emergency_breaches[0].limit_type.value}"
        
        if len(critical_breaches) >= 2:
            return True, f"Multiple critical breaches: {len(critical_breaches)}"
        
        return False, "No trading halt required"
    
    def adjust_position_sizes(self, current_positions: Dict[str, float],
                            target_reduction: float = 0.20) -> Dict[str, float]:
        """Automatically adjust position sizes to comply with risk limits."""
        adjusted_positions = current_positions.copy()
        
        # If we have concentration breaches, reduce largest positions
        concentration_breaches = [b for b in self.current_breaches 
                                if b.limit_type == RiskLimitType.CONCENTRATION]
        
        if concentration_breaches:
            # Sort positions by size (largest first)
            sorted_positions = sorted(adjusted_positions.items(), 
                                    key=lambda x: abs(x[1]), reverse=True)
            
            # Reduce largest positions
            for strategy, size in sorted_positions[:3]:  # Top 3 positions
                reduction_factor = 1.0 - target_reduction
                adjusted_positions[strategy] = size * reduction_factor
                logger.info(f"Reduced {strategy} position by {target_reduction:.1%}")
        
        return adjusted_positions