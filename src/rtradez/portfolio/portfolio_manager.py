"""
Central portfolio management system for multi-strategy coordination.

Manages multiple options trading strategies with automatic capital allocation,
risk coordination, and performance tracking across the entire portfolio.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from pydantic import BaseModel, Field, validator, ConfigDict

logger = logging.getLogger(__name__)

class PortfolioStatus(Enum):
    """Portfolio operational status."""
    ACTIVE = "active"
    PAUSED = "paused"
    LIQUIDATING = "liquidating"
    CLOSED = "closed"

class StrategyStatus(Enum):
    """Individual strategy status."""
    ACTIVE = "active"
    PAUSED = "paused"
    REDUCING = "reducing"
    CLOSED = "closed"

class StrategyAllocation(BaseModel):
    """Strategy allocation configuration with validation."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True
    )
    
    strategy_name: str = Field(..., min_length=1, description="Strategy identifier")
    target_allocation: float = Field(..., ge=0, le=1, description="Target percentage of portfolio")
    current_allocation: float = Field(..., ge=0, le=1, description="Current percentage")
    min_allocation: float = Field(0.0, ge=0, le=1, description="Minimum allocation percentage")
    max_allocation: float = Field(1.0, ge=0, le=1, description="Maximum allocation percentage")
    status: StrategyStatus = Field(StrategyStatus.ACTIVE, description="Strategy operational status")
    last_rebalance: Optional[datetime] = Field(None, description="Last rebalancing timestamp")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance tracking")
    
    @validator('target_allocation')
    def target_within_bounds(cls, v, values):
        if 'min_allocation' in values and v < values['min_allocation']:
            raise ValueError('Target allocation below minimum')
        if 'max_allocation' in values and v > values['max_allocation']:
            raise ValueError('Target allocation above maximum')
        return v
    
    @validator('current_allocation')
    def current_within_bounds(cls, v, values):
        if 'min_allocation' in values and v < values['min_allocation']:
            raise ValueError('Current allocation below minimum')
        if 'max_allocation' in values and v > values['max_allocation']:
            raise ValueError('Current allocation above maximum')
        return v
    
    @validator('max_allocation')
    def max_greater_than_min(cls, v, values):
        if 'min_allocation' in values and v < values['min_allocation']:
            raise ValueError('Maximum allocation must be greater than minimum')
        return v

class PortfolioConfig(BaseModel):
    """Portfolio configuration and constraints with validation."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    total_capital: float = Field(..., gt=0, description="Total available capital")
    max_strategies: int = Field(10, gt=0, le=50, description="Maximum number of strategies")
    rebalance_frequency: str = Field("weekly", description="Rebalancing frequency")
    rebalance_threshold: float = Field(0.05, gt=0, le=1, description="Deviation threshold for rebalancing")
    max_correlation_exposure: float = Field(0.60, gt=0, le=1, description="Maximum correlation exposure")
    emergency_stop_drawdown: float = Field(0.20, gt=0, le=1, description="Emergency liquidation drawdown")
    cash_reserve_minimum: float = Field(0.05, ge=0, le=1, description="Minimum cash reserve percentage")
    enable_auto_rebalancing: bool = Field(True, description="Enable automatic rebalancing")
    enable_risk_coordination: bool = Field(True, description="Enable risk management coordination")
    
    @validator('rebalance_frequency')
    def valid_frequency(cls, v):
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly']
        if v not in valid_frequencies:
            raise ValueError(f'Frequency must be one of: {valid_frequencies}')
        return v
    
    @validator('total_capital')
    def capital_reasonable(cls, v):
        if v < 10000:  # Minimum $10k
            raise ValueError('Total capital should be at least $10,000 for portfolio management')
        if v > 1e12:  # $1T sanity check
            raise ValueError('Total capital appears unreasonably large')
        return v

class PortfolioManager:
    """Central portfolio management system."""
    
    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio manager.
        
        Args:
            config: Portfolio configuration and constraints
        """
        self.config = config
        self.status = PortfolioStatus.ACTIVE
        
        # Strategy management
        self.strategies: Dict[str, StrategyAllocation] = {}
        self.strategy_instances: Dict[str, Any] = {}  # Actual strategy objects
        self.strategy_performance: Dict[str, pd.Series] = {}
        
        # Portfolio tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.cash_balance = config.total_capital
        self.invested_capital = 0.0
        
        # Performance metrics
        self.portfolio_returns: pd.Series = pd.Series(dtype=float)
        self.benchmark_returns: Optional[pd.Series] = None
        
        # Risk management integration
        self.risk_manager: Optional[Any] = None
        
        # Callbacks for external integration
        self.rebalance_callbacks: List[Callable] = []
        self.performance_callbacks: List[Callable] = []
        
    def add_strategy(self, strategy_name: str, strategy_instance: Any,
                    target_allocation: float, min_allocation: float = 0.0,
                    max_allocation: float = 0.50) -> bool:
        """
        Add a new strategy to the portfolio.
        
        Args:
            strategy_name: Unique strategy identifier
            strategy_instance: Strategy object with fit/predict interface
            target_allocation: Target allocation percentage (0-1)
            min_allocation: Minimum allocation percentage
            max_allocation: Maximum allocation percentage
        """
        if len(self.strategies) >= self.config.max_strategies:
            logger.error(f"Cannot add strategy: portfolio has maximum {self.config.max_strategies} strategies")
            return False
        
        if strategy_name in self.strategies:
            logger.warning(f"Strategy {strategy_name} already exists")
            return False
        
        if not (0 <= target_allocation <= 1):
            logger.error(f"Invalid target allocation: {target_allocation}")
            return False
        
        # Check if adding this allocation would exceed 100%
        current_total = sum(s.target_allocation for s in self.strategies.values())
        if current_total + target_allocation > 1.0:
            logger.error(f"Total allocation would exceed 100%: {current_total + target_allocation:.2%}")
            return False
        
        # Create strategy allocation
        allocation = StrategyAllocation(
            strategy_name=strategy_name,
            target_allocation=target_allocation,
            current_allocation=0.0,
            min_allocation=min_allocation,
            max_allocation=max_allocation,
            status=StrategyStatus.ACTIVE
        )
        
        self.strategies[strategy_name] = allocation
        self.strategy_instances[strategy_name] = strategy_instance
        self.strategy_performance[strategy_name] = pd.Series(dtype=float)
        
        logger.info(f"Added strategy {strategy_name} with {target_allocation:.1%} target allocation")
        return True
    
    def remove_strategy(self, strategy_name: str, liquidate: bool = True) -> bool:
        """
        Remove a strategy from the portfolio.
        
        Args:
            strategy_name: Strategy to remove
            liquidate: Whether to liquidate positions before removal
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return False
        
        allocation = self.strategies[strategy_name]
        
        if liquidate and allocation.current_allocation > 0:
            # Set status to reducing and let rebalancing handle liquidation
            allocation.status = StrategyStatus.REDUCING
            allocation.target_allocation = 0.0
            logger.info(f"Liquidating strategy {strategy_name}")
            return True
        else:
            # Remove immediately
            del self.strategies[strategy_name]
            del self.strategy_instances[strategy_name]
            del self.strategy_performance[strategy_name]
            logger.info(f"Removed strategy {strategy_name}")
            return True
    
    def update_strategy_allocation(self, strategy_name: str, 
                                 new_target: float) -> bool:
        """Update target allocation for a strategy."""
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return False
        
        if not (0 <= new_target <= 1):
            logger.error(f"Invalid target allocation: {new_target}")
            return False
        
        # Check total allocation constraint
        other_allocations = sum(s.target_allocation for name, s in self.strategies.items() 
                              if name != strategy_name)
        if other_allocations + new_target > 1.0:
            logger.error(f"Total allocation would exceed 100%")
            return False
        
        old_target = self.strategies[strategy_name].target_allocation
        self.strategies[strategy_name].target_allocation = new_target
        
        logger.info(f"Updated {strategy_name} target allocation: {old_target:.1%} -> {new_target:.1%}")
        return True
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate current portfolio metrics."""
        total_invested = sum(s.current_allocation * self.config.total_capital 
                           for s in self.strategies.values())
        
        # Portfolio composition
        composition = {
            name: allocation.current_allocation 
            for name, allocation in self.strategies.items()
        }
        
        # Performance metrics
        if len(self.portfolio_returns) > 0:
            total_return = (1 + self.portfolio_returns).prod() - 1
            volatility = self.portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (self.portfolio_returns.mean() * 252) / (volatility) if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(self.portfolio_returns)
        else:
            total_return = volatility = sharpe_ratio = max_drawdown = 0.0
        
        return {
            'total_capital': self.config.total_capital,
            'invested_capital': total_invested,
            'cash_balance': self.config.total_capital - total_invested,
            'cash_percentage': (self.config.total_capital - total_invested) / self.config.total_capital,
            'num_active_strategies': len([s for s in self.strategies.values() 
                                        if s.status == StrategyStatus.ACTIVE]),
            'strategy_composition': composition,
            'total_return': total_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'status': self.status.value
        }
    
    def update_strategy_performance(self, strategy_name: str, 
                                  performance_data: Dict[str, float]):
        """Update performance data for a strategy."""
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy {strategy_name} not found")
            return
        
        allocation = self.strategies[strategy_name]
        allocation.performance_metrics.update(performance_data)
        
        # Add to performance history
        if 'return' in performance_data:
            timestamp = datetime.now()
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = pd.Series(dtype=float)
            
            self.strategy_performance[strategy_name][timestamp] = performance_data['return']
    
    def calculate_rebalancing_needs(self) -> Dict[str, float]:
        """
        Calculate rebalancing needs for all strategies.
        
        Returns:
            Dictionary of strategy_name -> allocation_change needed
        """
        rebalancing_needs = {}
        
        for name, allocation in self.strategies.items():
            if allocation.status not in [StrategyStatus.ACTIVE, StrategyStatus.REDUCING]:
                continue
            
            target = allocation.target_allocation
            current = allocation.current_allocation
            deviation = abs(target - current)
            
            # Check if rebalancing is needed
            if deviation > self.config.rebalance_threshold:
                rebalancing_needs[name] = target - current
        
        return rebalancing_needs
    
    def execute_rebalancing(self, rebalancing_changes: Optional[Dict[str, float]] = None) -> bool:
        """
        Execute portfolio rebalancing.
        
        Args:
            rebalancing_changes: Optional specific changes to make
        """
        if not self.config.enable_auto_rebalancing:
            logger.info("Auto-rebalancing is disabled")
            return False
        
        if rebalancing_changes is None:
            rebalancing_changes = self.calculate_rebalancing_needs()
        
        if not rebalancing_changes:
            logger.info("No rebalancing needed")
            return True
        
        logger.info(f"Executing rebalancing for {len(rebalancing_changes)} strategies")
        
        # Calculate capital changes
        total_capital = self.config.total_capital
        capital_changes = {}
        
        for strategy_name, allocation_change in rebalancing_changes.items():
            capital_change = allocation_change * total_capital
            capital_changes[strategy_name] = capital_change
            
            # Update current allocation
            if strategy_name in self.strategies:
                self.strategies[strategy_name].current_allocation += allocation_change
                self.strategies[strategy_name].last_rebalance = datetime.now()
        
        # Notify callbacks
        for callback in self.rebalance_callbacks:
            try:
                callback(capital_changes)
            except Exception as e:
                logger.error(f"Error in rebalance callback: {e}")
        
        # Log rebalancing summary
        total_moved = sum(abs(change) for change in capital_changes.values())
        logger.info(f"Rebalancing complete: ${total_moved:,.0f} total capital moved")
        
        return True
    
    def check_risk_limits(self) -> List[str]:
        """Check portfolio-level risk limits."""
        warnings = []
        
        # Check cash reserve
        metrics = self.calculate_portfolio_metrics()
        if metrics['cash_percentage'] < self.config.cash_reserve_minimum:
            warnings.append(f"Cash reserve below minimum: {metrics['cash_percentage']:.1%}")
        
        # Check drawdown
        if metrics['max_drawdown'] > self.config.emergency_stop_drawdown:
            warnings.append(f"Drawdown exceeds emergency limit: {metrics['max_drawdown']:.1%}")
        
        # Check strategy correlation (simplified)
        if len(self.strategy_performance) > 1:
            correlation_matrix = self._calculate_strategy_correlations()
            max_correlation = np.max(np.abs(correlation_matrix - np.eye(len(correlation_matrix))))
            
            if max_correlation > self.config.max_correlation_exposure:
                warnings.append(f"High strategy correlation detected: {max_correlation:.2f}")
        
        return warnings
    
    def pause_portfolio(self, reason: str = "Manual pause"):
        """Pause all portfolio operations."""
        self.status = PortfolioStatus.PAUSED
        logger.warning(f"Portfolio paused: {reason}")
    
    def resume_portfolio(self):
        """Resume portfolio operations."""
        self.status = PortfolioStatus.ACTIVE
        logger.info("Portfolio resumed")
    
    def liquidate_portfolio(self, emergency: bool = False):
        """Liquidate entire portfolio."""
        self.status = PortfolioStatus.LIQUIDATING
        
        # Set all strategies to reducing
        for allocation in self.strategies.values():
            allocation.status = StrategyStatus.REDUCING
            allocation.target_allocation = 0.0
        
        logger.critical(f"Portfolio liquidation initiated (emergency: {emergency})")
        
        if emergency:
            # Force immediate liquidation
            self.execute_rebalancing()
    
    def get_strategy_attribution(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance attribution by strategy."""
        attribution = {}
        
        for name, allocation in self.strategies.items():
            if name not in self.strategy_performance or len(self.strategy_performance[name]) == 0:
                continue
            
            strategy_returns = self.strategy_performance[name]
            current_weight = allocation.current_allocation
            
            # Calculate contribution to portfolio return
            if len(strategy_returns) > 0:
                strategy_total_return = (1 + strategy_returns).prod() - 1
                contribution = strategy_total_return * current_weight
                
                attribution[name] = {
                    'weight': current_weight,
                    'strategy_return': strategy_total_return,
                    'contribution': contribution,
                    'volatility': strategy_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
                        if strategy_returns.std() > 0 else 0
                }
        
        return attribution
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_strategy_correlations(self) -> np.ndarray:
        """Calculate correlation matrix between strategies."""
        # Align all strategy returns
        strategy_data = {}
        for name, returns in self.strategy_performance.items():
            if len(returns) > 10:  # Minimum data requirement
                strategy_data[name] = returns
        
        if len(strategy_data) < 2:
            return np.array([[1.0]])
        
        # Create aligned DataFrame
        aligned_returns = pd.DataFrame(strategy_data).dropna()
        
        if len(aligned_returns) < 10:
            return np.eye(len(strategy_data))
        
        return aligned_returns.corr().values
    
    def add_rebalance_callback(self, callback: Callable[[Dict[str, float]], None]):
        """Add callback for rebalancing events."""
        self.rebalance_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for performance updates."""
        self.performance_callbacks.append(callback)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        metrics = self.calculate_portfolio_metrics()
        attribution = self.get_strategy_attribution()
        rebalancing_needs = self.calculate_rebalancing_needs()
        risk_warnings = self.check_risk_limits()
        
        return {
            'portfolio_metrics': metrics,
            'strategy_attribution': attribution,
            'rebalancing_needs': rebalancing_needs,
            'risk_warnings': risk_warnings,
            'strategy_count': len(self.strategies),
            'active_strategies': [name for name, alloc in self.strategies.items() 
                                if alloc.status == StrategyStatus.ACTIVE],
            'last_update': datetime.now().isoformat()
        }