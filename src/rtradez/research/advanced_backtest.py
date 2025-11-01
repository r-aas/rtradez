"""Advanced backtesting engine with realistic transaction costs and market impact."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

from ..base import BaseStrategy
from ..utils.caching import cached


class OrderType(Enum):
    """Order types for options trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OptionType(Enum):
    """Option types."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionsContract:
    """Options contract specification."""
    symbol: str
    expiration: str
    strike: float
    option_type: OptionType
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


@dataclass
class Trade:
    """Individual trade record."""
    timestamp: pd.Timestamp
    contract: OptionsContract
    quantity: int  # Positive for buy, negative for sell
    price: float
    order_type: OrderType
    commission: float = 0.0
    slippage: float = 0.0
    trade_id: str = ""


@dataclass
class Position:
    """Current position in an options contract."""
    contract: OptionsContract
    quantity: int
    avg_price: float
    unrealized_pnl: float = 0.0
    market_value: float = 0.0


class TransactionCostModel:
    """Model for calculating realistic transaction costs."""
    
    def __init__(self,
                 commission_per_contract: float = 0.65,
                 commission_per_trade: float = 1.0,
                 bid_ask_spread_model: str = 'linear',
                 slippage_model: str = 'sqrt',
                 min_spread: float = 0.05,
                 max_spread: float = 2.0):
        """
        Initialize transaction cost model.
        
        Args:
            commission_per_contract: Commission per options contract
            commission_per_trade: Fixed commission per trade
            bid_ask_spread_model: Model for bid-ask spreads
            slippage_model: Model for market impact/slippage
            min_spread: Minimum bid-ask spread
            max_spread: Maximum bid-ask spread
        """
        self.commission_per_contract = commission_per_contract
        self.commission_per_trade = commission_per_trade
        self.bid_ask_spread_model = bid_ask_spread_model
        self.slippage_model = slippage_model
        self.min_spread = min_spread
        self.max_spread = max_spread
    
    def calculate_commission(self, quantity: int) -> float:
        """Calculate commission for a trade."""
        return self.commission_per_trade + abs(quantity) * self.commission_per_contract
    
    def estimate_bid_ask_spread(self, contract: OptionsContract) -> float:
        """Estimate bid-ask spread based on contract characteristics."""
        if contract.bid > 0 and contract.ask > 0:
            return contract.ask - contract.bid
        
        # Estimate spread based on price and volume
        mid_price = contract.mid or (contract.bid + contract.ask) / 2 if contract.bid > 0 and contract.ask > 0 else 1.0
        
        if self.bid_ask_spread_model == 'linear':
            # Linear model: higher price = higher spread
            base_spread = mid_price * 0.1  # 10% of mid price
        elif self.bid_ask_spread_model == 'sqrt':
            # Square root model: spread grows slower with price
            base_spread = np.sqrt(mid_price) * 0.2
        else:
            base_spread = mid_price * 0.1
        
        # Adjust for volume/liquidity
        if contract.volume > 0:
            liquidity_factor = max(0.5, 1 - np.log(contract.volume + 1) / 10)
            base_spread *= liquidity_factor
        
        return np.clip(base_spread, self.min_spread, self.max_spread)
    
    def calculate_slippage(self, contract: OptionsContract, quantity: int, order_type: OrderType) -> float:
        """Calculate market impact/slippage."""
        if order_type == OrderType.LIMIT:
            return 0.0  # Assume limit orders don't have slippage
        
        bid_ask_spread = self.estimate_bid_ask_spread(contract)
        
        # Base slippage is half the spread
        base_slippage = bid_ask_spread / 2
        
        # Market impact based on trade size
        if contract.volume > 0:
            trade_ratio = abs(quantity) / max(contract.volume, 1)
            
            if self.slippage_model == 'sqrt':
                impact_factor = np.sqrt(trade_ratio)
            elif self.slippage_model == 'linear':
                impact_factor = trade_ratio
            else:
                impact_factor = np.sqrt(trade_ratio)
            
            # Additional impact for large trades
            market_impact = base_slippage * impact_factor * 0.5
        else:
            market_impact = base_slippage * 0.1  # Minimal impact for unknown volume
        
        return base_slippage + market_impact
    
    def get_execution_price(self, contract: OptionsContract, quantity: int, order_type: OrderType) -> float:
        """Get realistic execution price including costs."""
        mid_price = contract.mid or (contract.bid + contract.ask) / 2 if contract.bid > 0 and contract.ask > 0 else 1.0
        
        if order_type == OrderType.MARKET:
            # Market orders get worse fills
            if quantity > 0:  # Buying
                base_price = contract.ask if contract.ask > 0 else mid_price * 1.05
            else:  # Selling
                base_price = contract.bid if contract.bid > 0 else mid_price * 0.95
        else:
            base_price = mid_price
        
        # Add slippage
        slippage = self.calculate_slippage(contract, quantity, order_type)
        if quantity > 0:  # Buying
            execution_price = base_price + slippage
        else:  # Selling
            execution_price = base_price - slippage
        
        return max(0.01, execution_price)  # Minimum price of $0.01


class AdvancedBacktester:
    """
    Advanced backtesting engine with realistic market conditions.
    
    Features:
    - Realistic transaction costs and slippage
    - Options-specific considerations (expiration, assignment)
    - Multiple order types
    - Position sizing and risk management
    - Detailed performance attribution
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 cost_model: Optional[TransactionCostModel] = None,
                 margin_requirements: Dict[str, float] = None,
                 assignment_probability: float = 0.1):
        """
        Initialize advanced backtester.
        
        Args:
            initial_capital: Starting capital
            cost_model: Transaction cost model
            margin_requirements: Margin requirements by strategy type
            assignment_probability: Probability of early assignment
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.margin_requirements = margin_requirements or {
            'iron_condor': 0.2,
            'strangle': 0.3,
            'straddle': 0.3,
            'calendar_spread': 0.15
        }
        self.assignment_probability = assignment_probability
        
        # Tracking variables
        self.reset()
    
    def reset(self):
        """Reset backtester state."""
        self.current_capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.performance_metrics = {}
        self.current_date = None
    
    def backtest_strategy(self, 
                         strategy: BaseStrategy,
                         market_data: pd.DataFrame,
                         options_data: Optional[pd.DataFrame] = None,
                         position_sizing: Union[float, Callable] = 0.1,
                         rebalance_frequency: str = 'monthly') -> Dict:
        """
        Run backtest with realistic market conditions.
        
        Args:
            strategy: Fitted strategy to backtest
            market_data: Historical market data
            options_data: Historical options chain data (optional)
            position_sizing: Fixed fraction or callable for position sizing
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
        
        Returns:
            Comprehensive backtest results
        """
        self.reset()
        
        # Generate rebalance dates
        rebalance_dates = self._get_rebalance_dates(market_data.index, rebalance_frequency)
        
        for date in market_data.index:
            self.current_date = date
            current_data = market_data.loc[date:date]
            
            # Update position values
            self._update_positions(date, market_data)
            
            # Record equity curve
            total_value = self._calculate_total_value()
            self.equity_curve.append({
                'date': date,
                'capital': self.current_capital,
                'position_value': sum(p.market_value for p in self.positions.values()),
                'total_value': total_value,
                'leverage': self._calculate_leverage()
            })
            
            # Check for strategy signals on rebalance dates
            if date in rebalance_dates:
                self._execute_strategy_signals(strategy, current_data, options_data, position_sizing)
            
            # Handle expiration and assignment
            self._handle_expiration_and_assignment(date)
        
        # Calculate final performance metrics
        self.performance_metrics = self._calculate_performance_metrics()
        
        return {
            'trades': self.trades,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'final_positions': self.positions,
            'performance_metrics': self.performance_metrics,
            'transaction_costs': sum(t.commission + t.slippage for t in self.trades)
        }
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex, frequency: str) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        if frequency == 'daily':
            return list(date_index)
        elif frequency == 'weekly':
            return list(date_index[date_index.weekday == 0])  # Mondays
        elif frequency == 'monthly':
            return list(date_index[date_index.day == 1])  # First of month
        else:
            return list(date_index[::30])  # Every 30 days
    
    def _execute_strategy_signals(self, 
                                 strategy: BaseStrategy,
                                 market_data: pd.DataFrame,
                                 options_data: Optional[pd.DataFrame],
                                 position_sizing: Union[float, Callable]):
        """Execute strategy signals with realistic order execution."""
        # Get strategy signal
        signal = strategy.predict(market_data)
        
        if signal[0] > 0.5:  # Signal to enter position
            # Calculate position size
            if callable(position_sizing):
                size = position_sizing(self.current_capital, market_data)
            else:
                size = position_sizing
            
            # Generate synthetic options contracts for testing
            contracts = self._generate_synthetic_options(market_data, strategy.strategy_type)
            
            # Execute strategy trades
            self._execute_options_strategy(contracts, size, strategy.strategy_type)
    
    def _generate_synthetic_options(self, market_data: pd.DataFrame, strategy_type: str) -> List[OptionsContract]:
        """Generate synthetic options contracts for backtesting."""
        current_price = market_data['Close'].iloc[0]
        
        contracts = []
        
        if strategy_type == 'iron_condor':
            # Generate Iron Condor legs
            put_strike_short = current_price * 0.95
            put_strike_long = current_price * 0.90
            call_strike_short = current_price * 1.05
            call_strike_long = current_price * 1.10
            
            contracts.extend([
                OptionsContract('SPY', '2024-01-19', put_strike_long, OptionType.PUT, 
                              bid=0.50, ask=0.55, volume=100),
                OptionsContract('SPY', '2024-01-19', put_strike_short, OptionType.PUT,
                              bid=1.50, ask=1.55, volume=200),
                OptionsContract('SPY', '2024-01-19', call_strike_short, OptionType.CALL,
                              bid=1.50, ask=1.55, volume=200),
                OptionsContract('SPY', '2024-01-19', call_strike_long, OptionType.CALL,
                              bid=0.50, ask=0.55, volume=100)
            ])
        
        elif strategy_type == 'strangle':
            # Generate Strangle legs
            put_strike = current_price * 0.95
            call_strike = current_price * 1.05
            
            contracts.extend([
                OptionsContract('SPY', '2024-01-19', put_strike, OptionType.PUT,
                              bid=2.00, ask=2.10, volume=150),
                OptionsContract('SPY', '2024-01-19', call_strike, OptionType.CALL,
                              bid=2.00, ask=2.10, volume=150)
            ])
        
        return contracts
    
    def _execute_options_strategy(self, contracts: List[OptionsContract], size: float, strategy_type: str):
        """Execute options strategy trades."""
        if strategy_type == 'iron_condor':
            # Iron Condor: Sell middle strikes, buy outer strikes
            trade_quantities = [-1, 1, 1, -1]  # Short put, long put, long call, short call
        elif strategy_type == 'strangle':
            # Strangle: Sell put and call
            trade_quantities = [-1, -1]  # Short put, short call
        else:
            trade_quantities = [1] * len(contracts)  # Default to buying
        
        # Calculate number of contracts based on position size
        available_capital = self.current_capital * size
        strategy_cost = sum(
            abs(qty) * contract.ask if qty > 0 else abs(qty) * contract.bid
            for contract, qty in zip(contracts, trade_quantities)
        )
        
        if strategy_cost > 0:
            num_spreads = max(1, int(available_capital / strategy_cost))
        else:
            num_spreads = 1
        
        # Execute trades
        for contract, base_qty in zip(contracts, trade_quantities):
            if base_qty != 0:
                quantity = base_qty * num_spreads
                self._execute_trade(contract, quantity, OrderType.MARKET)
    
    def _execute_trade(self, contract: OptionsContract, quantity: int, order_type: OrderType):
        """Execute individual trade with realistic costs."""
        # Get execution price
        execution_price = self.cost_model.get_execution_price(contract, quantity, order_type)
        
        # Calculate costs
        commission = self.cost_model.calculate_commission(quantity)
        slippage_cost = abs(quantity) * self.cost_model.calculate_slippage(contract, quantity, order_type)
        
        # Create trade record
        trade = Trade(
            timestamp=self.current_date,
            contract=contract,
            quantity=quantity,
            price=execution_price,
            order_type=order_type,
            commission=commission,
            slippage=slippage_cost,
            trade_id=f"trade_{len(self.trades)}"
        )
        
        self.trades.append(trade)
        
        # Update positions
        position_key = f"{contract.symbol}_{contract.expiration}_{contract.strike}_{contract.option_type.value}"
        
        if position_key in self.positions:
            # Update existing position
            pos = self.positions[position_key]
            new_quantity = pos.quantity + quantity
            
            if new_quantity == 0:
                # Position closed
                del self.positions[position_key]
            else:
                # Update average price
                total_cost = pos.avg_price * pos.quantity + execution_price * quantity
                pos.quantity = new_quantity
                pos.avg_price = total_cost / new_quantity if new_quantity != 0 else 0
        else:
            # New position
            if quantity != 0:
                self.positions[position_key] = Position(
                    contract=contract,
                    quantity=quantity,
                    avg_price=execution_price
                )
        
        # Update capital
        trade_cost = quantity * execution_price + commission + slippage_cost
        self.current_capital -= trade_cost
    
    def _update_positions(self, date: pd.Timestamp, market_data: pd.DataFrame):
        """Update position values based on current market conditions."""
        current_price = market_data.loc[date, 'Close'] if date in market_data.index else None
        
        for position in self.positions.values():
            # Estimate current option value (simplified)
            if current_price is not None:
                estimated_value = self._estimate_option_value(position.contract, current_price)
                position.market_value = position.quantity * estimated_value
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_price)
    
    def _estimate_option_value(self, contract: OptionsContract, current_price: float) -> float:
        """Estimate current option value (simplified Black-Scholes)."""
        # Simplified intrinsic value calculation
        if contract.option_type == OptionType.CALL:
            intrinsic = max(0, current_price - contract.strike)
        else:
            intrinsic = max(0, contract.strike - current_price)
        
        # Add some time value (simplified)
        time_value = max(0.05, intrinsic * 0.1)
        
        return intrinsic + time_value
    
    def _handle_expiration_and_assignment(self, date: pd.Timestamp):
        """Handle option expiration and early assignment."""
        expired_positions = []
        
        for key, position in self.positions.items():
            # Check for expiration (simplified - check if Friday and near expiration date)
            if date.weekday() == 4:  # Friday
                # Simulate expiration
                if np.random.random() < 0.1:  # 10% chance of expiration this Friday
                    expired_positions.append(key)
        
        # Handle expired positions
        for key in expired_positions:
            position = self.positions[key]
            # Settle at intrinsic value
            settlement_value = max(0, self._estimate_option_value(position.contract, 100))  # Assume $100 settlement
            self.current_capital += position.quantity * settlement_value
            del self.positions[key]
    
    def _calculate_total_value(self) -> float:
        """Calculate total portfolio value."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.current_capital + position_value
    
    def _calculate_leverage(self) -> float:
        """Calculate current leverage ratio."""
        position_value = sum(abs(p.market_value) for p in self.positions.values())
        total_value = self._calculate_total_value()
        return position_value / total_value if total_value > 0 else 0
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        returns = equity_df['total_value'].pct_change().dropna()
        
        total_return = (equity_df['total_value'].iloc[-1] / equity_df['total_value'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        win_rate = (returns > 0).mean()
        
        # Transaction cost analysis
        total_commissions = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'total_commissions': total_commissions,
            'total_slippage': total_slippage,
            'final_capital': equity_df['total_value'].iloc[-1],
            'avg_leverage': equity_df['leverage'].mean() if 'leverage' in equity_df.columns else 0
        }