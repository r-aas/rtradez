"""
Options margin calculation for risk management.

Implements margin requirements for various options strategies following
standard industry practices and regulatory requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Option types."""
    CALL = "call"
    PUT = "put"

class StrategyType(Enum):
    """Options strategy types for margin calculation."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    IRON_CONDOR = "iron_condor"
    STRANGLE = "strangle"
    STRADDLE = "straddle"
    CALENDAR_SPREAD = "calendar_spread"
    VERTICAL_SPREAD = "vertical_spread"

@dataclass
class OptionPosition:
    """Individual option position for margin calculation."""
    option_type: OptionType
    strike: float
    expiry_days: int
    quantity: int  # Positive for long, negative for short
    premium: float
    underlying_price: float
    volatility: float

@dataclass
class MarginRequirement:
    """Margin requirement calculation result."""
    initial_margin: float
    maintenance_margin: float
    buying_power_reduction: float
    strategy_type: StrategyType
    total_premium: float
    max_loss: Optional[float]
    max_profit: Optional[float]
    breakeven_points: List[float]
    risk_metrics: Dict[str, float]
    warnings: List[str]

class OptionsMarginCalculator:
    """Calculates margin requirements for options strategies."""
    
    def __init__(self, account_type: str = "margin", 
                 risk_multiplier: float = 1.0):
        """
        Initialize margin calculator.
        
        Args:
            account_type: Account type ("margin" or "cash")
            risk_multiplier: Risk multiplier for additional safety margin
        """
        self.account_type = account_type
        self.risk_multiplier = risk_multiplier
        
        # Standard margin parameters (can be customized by broker)
        self.equity_margin_rate = 0.50  # 50% for stocks
        self.option_margin_multiplier = 0.20  # 20% for naked options
        self.minimum_margin = 100.0  # Minimum margin requirement
        
    def calculate_strategy_margin(self, positions: List[OptionPosition],
                                strategy_type: StrategyType,
                                underlying_price: float) -> MarginRequirement:
        """
        Calculate margin for an options strategy.
        
        Args:
            positions: List of option positions in the strategy
            strategy_type: Type of options strategy
            underlying_price: Current underlying asset price
        """
        warnings = []
        
        # Validate positions
        if not positions:
            return MarginRequirement(
                initial_margin=0, maintenance_margin=0, buying_power_reduction=0,
                strategy_type=strategy_type, total_premium=0, max_loss=0,
                max_profit=0, breakeven_points=[], risk_metrics={}, warnings=["No positions provided"]
            )
        
        # Calculate based on strategy type
        if strategy_type == StrategyType.IRON_CONDOR:
            return self._calculate_iron_condor_margin(positions, underlying_price)
        elif strategy_type == StrategyType.STRANGLE:
            return self._calculate_strangle_margin(positions, underlying_price)
        elif strategy_type == StrategyType.STRADDLE:
            return self._calculate_straddle_margin(positions, underlying_price)
        elif strategy_type == StrategyType.VERTICAL_SPREAD:
            return self._calculate_vertical_spread_margin(positions, underlying_price)
        elif strategy_type == StrategyType.COVERED_CALL:
            return self._calculate_covered_call_margin(positions, underlying_price)
        elif strategy_type == StrategyType.CASH_SECURED_PUT:
            return self._calculate_cash_secured_put_margin(positions, underlying_price)
        else:
            return self._calculate_generic_margin(positions, strategy_type, underlying_price)
    
    def _calculate_iron_condor_margin(self, positions: List[OptionPosition],
                                    underlying_price: float) -> MarginRequirement:
        """Calculate margin for Iron Condor strategy."""
        if len(positions) != 4:
            return self._calculate_generic_margin(positions, StrategyType.IRON_CONDOR, underlying_price)
        
        # Sort positions by strike price
        sorted_positions = sorted(positions, key=lambda x: x.strike)
        
        # Iron Condor: Short Call Spread + Short Put Spread
        # Margin = max(call spread width, put spread width) * 100 - net credit
        
        call_strikes = [p.strike for p in sorted_positions if p.option_type == OptionType.CALL]
        put_strikes = [p.strike for p in sorted_positions if p.option_type == OptionType.PUT]
        
        if len(call_strikes) != 2 or len(put_strikes) != 2:
            return self._calculate_generic_margin(positions, StrategyType.IRON_CONDOR, underlying_price)
        
        call_spread_width = max(call_strikes) - min(call_strikes)
        put_spread_width = max(put_strikes) - min(put_strikes)
        
        # Net premium received (negative for credit received)
        total_premium = sum(p.premium * p.quantity for p in positions)
        
        # Margin is the maximum spread width minus net credit
        max_spread_width = max(call_spread_width, put_spread_width)
        margin_requirement = (max_spread_width * 100) + total_premium  # Add because premium is negative for credit
        
        # Apply minimum margin
        margin_requirement = max(margin_requirement, self.minimum_margin)
        margin_requirement *= self.risk_multiplier
        
        # Calculate risk metrics
        max_loss = max_spread_width * 100 + total_premium
        max_profit = -total_premium  # Credit received
        
        # Breakeven points
        lower_breakeven = min(put_strikes) - (-total_premium / 100)
        upper_breakeven = max(call_strikes) + (-total_premium / 100)
        
        return MarginRequirement(
            initial_margin=margin_requirement,
            maintenance_margin=margin_requirement * 0.75,
            buying_power_reduction=margin_requirement,
            strategy_type=StrategyType.IRON_CONDOR,
            total_premium=total_premium,
            max_loss=max_loss,
            max_profit=max_profit,
            breakeven_points=[lower_breakeven, upper_breakeven],
            risk_metrics={
                'call_spread_width': call_spread_width,
                'put_spread_width': put_spread_width,
                'max_spread_width': max_spread_width
            },
            warnings=[]
        )
    
    def _calculate_strangle_margin(self, positions: List[OptionPosition],
                                 underlying_price: float) -> MarginRequirement:
        """Calculate margin for Strangle strategy."""
        calls = [p for p in positions if p.option_type == OptionType.CALL]
        puts = [p for p in positions if p.option_type == OptionType.PUT]
        
        total_premium = sum(p.premium * p.quantity for p in positions)
        
        if len(calls) == 1 and len(puts) == 1:
            # Long strangle - just premium paid
            if calls[0].quantity > 0 and puts[0].quantity > 0:
                return MarginRequirement(
                    initial_margin=total_premium,
                    maintenance_margin=0,
                    buying_power_reduction=total_premium,
                    strategy_type=StrategyType.STRANGLE,
                    total_premium=total_premium,
                    max_loss=total_premium,
                    max_profit=float('inf'),
                    breakeven_points=[puts[0].strike - total_premium/100, calls[0].strike + total_premium/100],
                    risk_metrics={},
                    warnings=[]
                )
            
            # Short strangle - margin for naked positions
            else:
                call_margin = self._calculate_naked_option_margin(calls[0], underlying_price)
                put_margin = self._calculate_naked_option_margin(puts[0], underlying_price)
                total_margin = max(call_margin, put_margin) + abs(total_premium)
                
                return MarginRequirement(
                    initial_margin=total_margin * self.risk_multiplier,
                    maintenance_margin=total_margin * 0.75,
                    buying_power_reduction=total_margin,
                    strategy_type=StrategyType.STRANGLE,
                    total_premium=total_premium,
                    max_loss=float('inf'),
                    max_profit=-total_premium,
                    breakeven_points=[puts[0].strike + total_premium/100, calls[0].strike - total_premium/100],
                    risk_metrics={'call_margin': call_margin, 'put_margin': put_margin},
                    warnings=[]
                )
        
        return self._calculate_generic_margin(positions, StrategyType.STRANGLE, underlying_price)
    
    def _calculate_straddle_margin(self, positions: List[OptionPosition],
                                 underlying_price: float) -> MarginRequirement:
        """Calculate margin for Straddle strategy."""
        # Similar to strangle but both options at same strike
        return self._calculate_strangle_margin(positions, underlying_price)
    
    def _calculate_vertical_spread_margin(self, positions: List[OptionPosition],
                                        underlying_price: float) -> MarginRequirement:
        """Calculate margin for vertical spread."""
        if len(positions) != 2:
            return self._calculate_generic_margin(positions, StrategyType.VERTICAL_SPREAD, underlying_price)
        
        # Both options should be same type
        option_types = set(p.option_type for p in positions)
        if len(option_types) != 1:
            return self._calculate_generic_margin(positions, StrategyType.VERTICAL_SPREAD, underlying_price)
        
        sorted_positions = sorted(positions, key=lambda x: x.strike)
        spread_width = sorted_positions[1].strike - sorted_positions[0].strike
        total_premium = sum(p.premium * p.quantity for p in positions)
        
        # For credit spreads, margin is spread width - credit received
        # For debit spreads, margin is just the premium paid
        if total_premium < 0:  # Credit spread
            margin_requirement = (spread_width * 100) + total_premium
            max_loss = margin_requirement
            max_profit = -total_premium
        else:  # Debit spread
            margin_requirement = total_premium
            max_loss = total_premium
            max_profit = (spread_width * 100) - total_premium
        
        margin_requirement = max(margin_requirement, self.minimum_margin)
        margin_requirement *= self.risk_multiplier
        
        return MarginRequirement(
            initial_margin=margin_requirement,
            maintenance_margin=margin_requirement * 0.75,
            buying_power_reduction=margin_requirement,
            strategy_type=StrategyType.VERTICAL_SPREAD,
            total_premium=total_premium,
            max_loss=max_loss,
            max_profit=max_profit,
            breakeven_points=[],
            risk_metrics={'spread_width': spread_width},
            warnings=[]
        )
    
    def _calculate_covered_call_margin(self, positions: List[OptionPosition],
                                     underlying_price: float) -> MarginRequirement:
        """Calculate margin for covered call strategy."""
        # Assumes 100 shares of underlying are owned
        # Margin = Stock margin - call premium received
        
        call_positions = [p for p in positions if p.option_type == OptionType.CALL and p.quantity < 0]
        if not call_positions:
            return self._calculate_generic_margin(positions, StrategyType.COVERED_CALL, underlying_price)
        
        call_premium = sum(p.premium * abs(p.quantity) for p in call_positions)
        stock_value = underlying_price * 100  # 100 shares
        
        if self.account_type == "margin":
            stock_margin = stock_value * self.equity_margin_rate
            net_margin = stock_margin - call_premium
        else:
            net_margin = stock_value - call_premium
        
        net_margin = max(net_margin, self.minimum_margin)
        net_margin *= self.risk_multiplier
        
        return MarginRequirement(
            initial_margin=net_margin,
            maintenance_margin=net_margin * 0.75,
            buying_power_reduction=net_margin,
            strategy_type=StrategyType.COVERED_CALL,
            total_premium=-call_premium,  # Credit received
            max_loss=stock_value + call_premium,  # If stock goes to zero
            max_profit=call_positions[0].strike * 100 - stock_value + call_premium,
            breakeven_points=[underlying_price - call_premium/100],
            risk_metrics={'stock_value': stock_value},
            warnings=[]
        )
    
    def _calculate_cash_secured_put_margin(self, positions: List[OptionPosition],
                                         underlying_price: float) -> MarginRequirement:
        """Calculate margin for cash-secured put strategy."""
        put_positions = [p for p in positions if p.option_type == OptionType.PUT and p.quantity < 0]
        if not put_positions:
            return self._calculate_generic_margin(positions, StrategyType.CASH_SECURED_PUT, underlying_price)
        
        put = put_positions[0]
        put_premium = put.premium * abs(put.quantity)
        
        # Cash required = strike price * 100 - premium received
        cash_required = (put.strike * 100) - put_premium
        
        return MarginRequirement(
            initial_margin=cash_required,
            maintenance_margin=cash_required,
            buying_power_reduction=cash_required,
            strategy_type=StrategyType.CASH_SECURED_PUT,
            total_premium=-put_premium,
            max_loss=cash_required,
            max_profit=put_premium,
            breakeven_points=[put.strike - put_premium/100],
            risk_metrics={'cash_required': cash_required},
            warnings=[]
        )
    
    def _calculate_naked_option_margin(self, position: OptionPosition, 
                                     underlying_price: float) -> float:
        """Calculate margin for naked option position."""
        # Standard formula: 20% of underlying + premium - out-of-the-money amount
        
        premium_value = position.premium * abs(position.quantity)
        underlying_value = underlying_price * 100 * abs(position.quantity)
        
        # Calculate out-of-the-money amount
        if position.option_type == OptionType.CALL:
            otm_amount = max(0, position.strike - underlying_price) * 100 * abs(position.quantity)
        else:  # PUT
            otm_amount = max(0, underlying_price - position.strike) * 100 * abs(position.quantity)
        
        margin = (self.option_margin_multiplier * underlying_value) + premium_value - otm_amount
        return max(margin, self.minimum_margin)
    
    def _calculate_generic_margin(self, positions: List[OptionPosition],
                                strategy_type: StrategyType,
                                underlying_price: float) -> MarginRequirement:
        """Calculate margin for generic/unknown strategy."""
        warnings = ["Using generic margin calculation - strategy not specifically implemented"]
        
        total_margin = 0
        total_premium = 0
        
        for position in positions:
            total_premium += position.premium * position.quantity
            
            if position.quantity > 0:  # Long position
                total_margin += position.premium * position.quantity
            else:  # Short position
                naked_margin = self._calculate_naked_option_margin(position, underlying_price)
                total_margin += naked_margin
        
        total_margin = max(total_margin, self.minimum_margin)
        total_margin *= self.risk_multiplier
        
        return MarginRequirement(
            initial_margin=total_margin,
            maintenance_margin=total_margin * 0.75,
            buying_power_reduction=total_margin,
            strategy_type=strategy_type,
            total_premium=total_premium,
            max_loss=None,
            max_profit=None,
            breakeven_points=[],
            risk_metrics={},
            warnings=warnings
        )
    
    def calculate_portfolio_margin(self, strategy_margins: List[MarginRequirement]) -> Dict[str, float]:
        """Calculate total portfolio margin requirements."""
        total_initial_margin = sum(m.initial_margin for m in strategy_margins)
        total_maintenance_margin = sum(m.maintenance_margin for m in strategy_margins)
        total_buying_power_reduction = sum(m.buying_power_reduction for m in strategy_margins)
        total_premium = sum(m.total_premium for m in strategy_margins)
        
        return {
            'total_initial_margin': total_initial_margin,
            'total_maintenance_margin': total_maintenance_margin,
            'total_buying_power_reduction': total_buying_power_reduction,
            'total_premium': total_premium,
            'margin_utilization': total_buying_power_reduction,  # Assuming this is the key metric
            'number_of_strategies': len(strategy_margins)
        }