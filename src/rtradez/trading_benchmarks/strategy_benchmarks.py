"""
Strategy backtesting and performance benchmarking framework.

Provides comprehensive backtesting capabilities for options trading strategies
with detailed performance metrics and risk analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
from pydantic import BaseModel, Field

from ..risk import KellyConfig, KellyCriterion, PositionSizeResult
from ..portfolio.portfolio_manager import PortfolioManager, PortfolioConfig


class StrategyType(Enum):
    """Types of trading strategies."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    COLLAR = "collar"
    CALENDAR_SPREAD = "calendar_spread"


class TradeDirection(Enum):
    """Trade direction indicators."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Trade:
    """Individual trade record."""
    timestamp: datetime
    symbol: str
    strategy_type: StrategyType
    direction: TradeDirection
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int = 1
    commission: float = 0.0
    premium_collected: float = 0.0
    premium_paid: float = 0.0
    strike_price: Optional[float] = None
    expiration: Optional[datetime] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    underlying_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: str = "manual"
    pnl: Optional[float] = None
    
    def calculate_pnl(self) -> float:
        """Calculate profit/loss for the trade."""
        if self.exit_price is None:
            return 0.0
        
        if self.direction == TradeDirection.LONG:
            gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            gross_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        # Add premium effects
        net_pnl = gross_pnl + self.premium_collected - self.premium_paid - self.commission
        self.pnl = net_pnl
        return net_pnl


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=100000, gt=0)
    commission_per_trade: float = Field(default=1.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)  # Basis points
    max_position_size: float = Field(default=0.25, gt=0, le=1.0)  # 25% max per position
    risk_free_rate: float = Field(default=0.02, ge=0)  # 2% annual
    benchmark_symbol: str = "SPY"
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    enable_options_assignment: bool = True
    margin_requirement: float = Field(default=0.5, gt=0, le=1.0)  # 50% margin
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BacktestResults(BaseModel):
    """Comprehensive backtesting results."""
    config: BacktestConfig
    trades: List[Trade]
    daily_returns: pd.Series = Field(exclude=True)
    equity_curve: pd.Series = Field(exclude=True)
    
    # Performance Metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    
    # Trading Metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Risk Metrics
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Options Specific
    avg_iv_rank: Optional[float] = None
    gamma_pnl: Optional[float] = None
    theta_decay: Optional[float] = None
    vega_exposure: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.Series: lambda v: v.to_dict(),
            datetime: lambda v: v.isoformat()
        }


class StrategyBenchmark:
    """Comprehensive strategy backtesting and benchmarking framework."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.daily_returns: pd.Series = pd.Series(dtype=float)
        self.benchmark_data: Optional[pd.Series] = None
        
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the backtest."""
        self.trades.append(trade)
    
    def generate_synthetic_market_data(self, symbol: str = "SPY") -> pd.DataFrame:
        """Generate synthetic market data for backtesting."""
        # Create date range
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        # Generate synthetic price data with realistic characteristics
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0008, 0.016, len(dates))  # ~20% annual vol, 20% annual return
        
        # Add some autocorrelation and volatility clustering
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Slight momentum
            if abs(returns[i-1]) > 0.02:  # Volatility clustering
                returns[i] *= 1.5
        
        # Generate price series
        initial_price = 400.0  # Starting price for SPY-like instrument
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Add some intraday noise for high/low
        highs = prices * (1 + np.random.uniform(0.001, 0.02, len(prices)))
        lows = prices * (1 - np.random.uniform(0.001, 0.02, len(prices)))
        
        # Volume with some correlation to volatility
        volume = np.random.lognormal(15, 0.5, len(dates))  # Base volume
        volume *= (1 + 2 * np.abs(returns))  # Higher volume on big moves
        
        market_data = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volume,
            'returns': returns
        }, index=dates)
        
        return market_data
    
    def generate_synthetic_options_data(self, underlying_data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic options data based on underlying."""
        options_data = []
        
        for date, row in underlying_data.iterrows():
            underlying_price = row['close']
            
            # Generate multiple strikes around ATM
            strikes = np.arange(
                underlying_price * 0.8,
                underlying_price * 1.2,
                underlying_price * 0.02
            )
            
            for strike in strikes:
                # Calculate moneyness
                moneyness = strike / underlying_price
                
                # Synthetic IV with volatility smile
                base_iv = 0.20
                smile_adjustment = 0.05 * (abs(moneyness - 1.0) ** 2)
                iv = base_iv + smile_adjustment
                
                # Simple Black-Scholes-like Greeks approximation
                time_to_expiry = 30 / 365  # Assume 30 DTE
                
                # Simplified Greeks (not exact Black-Scholes)
                delta = 0.5 if abs(moneyness - 1.0) < 0.01 else (1.0 if moneyness < 1.0 else 0.0)
                gamma = 1.0 / (underlying_price * iv * np.sqrt(time_to_expiry))
                theta = -underlying_price * gamma * iv**2 / 2
                vega = underlying_price * np.sqrt(time_to_expiry) * 0.01
                
                # Option price approximation
                intrinsic = max(0, underlying_price - strike)  # Call option
                time_value = iv * underlying_price * np.sqrt(time_to_expiry) * 0.4
                option_price = intrinsic + time_value
                
                options_data.append({
                    'date': date,
                    'underlying_price': underlying_price,
                    'strike': strike,
                    'option_type': 'call',
                    'expiration': date + timedelta(days=30),
                    'iv': iv,
                    'option_price': option_price,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'rho': 0.01,  # Simplified
                    'bid': option_price * 0.98,
                    'ask': option_price * 1.02,
                    'volume': np.random.randint(10, 1000)
                })
        
        return pd.DataFrame(options_data)
    
    def run_simple_covered_call_strategy(self) -> None:
        """Run a simple covered call strategy for demonstration."""
        # Generate market data
        market_data = self.generate_synthetic_market_data()
        options_data = self.generate_synthetic_options_data(market_data)
        
        # Simple covered call strategy
        position_open = False
        entry_date = None
        
        for date, market_row in market_data.iterrows():
            underlying_price = market_row['close']
            
            # Look for options data for this date
            daily_options = options_data[options_data['date'] == date]
            
            if daily_options.empty:
                continue
            
            # If no position, look to open covered call
            if not position_open:
                # Find slightly OTM call to sell
                otm_calls = daily_options[
                    (daily_options['strike'] > underlying_price * 1.02) &
                    (daily_options['strike'] < underlying_price * 1.08)
                ]
                
                if not otm_calls.empty:
                    # Select first available OTM call
                    option = otm_calls.iloc[0]
                    
                    # Create covered call trade (buy stock, sell call)
                    stock_trade = Trade(
                        timestamp=date,
                        symbol="SPY",
                        strategy_type=StrategyType.COVERED_CALL,
                        direction=TradeDirection.LONG,
                        entry_price=underlying_price,
                        quantity=100,
                        commission=self.config.commission_per_trade,
                        underlying_price=underlying_price
                    )
                    
                    call_trade = Trade(
                        timestamp=date,
                        symbol="SPY",
                        strategy_type=StrategyType.COVERED_CALL,
                        direction=TradeDirection.SHORT,
                        entry_price=option['option_price'],
                        quantity=1,
                        commission=self.config.commission_per_trade,
                        premium_collected=option['option_price'] * 100,
                        strike_price=option['strike'],
                        expiration=option['expiration'],
                        implied_volatility=option['iv'],
                        delta=option['delta'],
                        gamma=option['gamma'],
                        theta=option['theta'],
                        vega=option['vega'],
                        underlying_price=underlying_price
                    )
                    
                    self.add_trade(stock_trade)
                    self.add_trade(call_trade)
                    position_open = True
                    entry_date = date
            
            # If position open, check for exit conditions
            elif position_open and entry_date:
                days_held = (date - entry_date).days
                
                # Exit after 20 days or at expiration
                if days_held >= 20:
                    # Close position - sell stock, buy back call
                    # Find the call option to buy back
                    matching_calls = daily_options[
                        daily_options['strike'] == self.trades[-1].strike_price
                    ]
                    
                    if not matching_calls.empty:
                        option = matching_calls.iloc[0]
                        
                        # Close stock position
                        self.trades[-2].exit_price = underlying_price
                        self.trades[-2].exit_timestamp = date
                        self.trades[-2].calculate_pnl()
                        
                        # Close call position
                        self.trades[-1].exit_price = option['option_price']
                        self.trades[-1].exit_timestamp = date
                        self.trades[-1].calculate_pnl()
                        
                        position_open = False
                        entry_date = None
    
    def calculate_performance_metrics(self) -> BacktestResults:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            raise ValueError("No trades to analyze. Run a strategy first.")
        
        # Calculate equity curve
        equity_values = [self.config.initial_capital]
        trade_dates = []
        
        for trade in self.trades:
            if trade.pnl is not None:
                equity_values.append(equity_values[-1] + trade.pnl)
                trade_dates.append(trade.exit_timestamp or trade.timestamp)
        
        # Create full date range equity curve
        full_dates = pd.date_range(self.config.start_date, self.config.end_date, freq='D')
        equity_curve = pd.Series(index=full_dates, dtype=float)
        
        # Forward fill equity values
        current_equity = self.config.initial_capital
        trade_idx = 0
        
        for date in full_dates:
            while trade_idx < len(trade_dates) and trade_dates[trade_idx] <= date:
                current_equity = equity_values[trade_idx + 1]
                trade_idx += 1
            equity_curve[date] = current_equity
        
        # Calculate daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        # Basic performance metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        trading_days = len(daily_returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = daily_returns - (self.config.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Downside deviation for Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / (downside_deviation / np.sqrt(252)) if downside_deviation > 0 else 0
        
        # Drawdown analysis
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Max drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        dd_groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
        dd_durations = drawdown_periods.groupby(dd_groups).sum()
        max_drawdown_duration = dd_durations.max() if len(dd_durations) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading statistics
        completed_trades = [t for t in self.trades if t.pnl is not None]
        total_trades = len(completed_trades)
        
        if total_trades > 0:
            winning_trades = len([t for t in completed_trades if t.pnl > 0])
            losing_trades = len([t for t in completed_trades if t.pnl <= 0])
            win_rate = winning_trades / total_trades
            
            wins = [t.pnl for t in completed_trades if t.pnl > 0]
            losses = [t.pnl for t in completed_trades if t.pnl <= 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0
        
        # Risk metrics
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns) > 0 else 0
        
        # Options-specific metrics
        options_trades = [t for t in self.trades if t.implied_volatility is not None]
        avg_iv_rank = np.mean([t.implied_volatility for t in options_trades]) if options_trades else None
        
        # Store series data
        self.equity_curve = equity_curve
        self.daily_returns = daily_returns
        
        return BacktestResults(
            config=self.config,
            trades=self.trades,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            var_95=var_95,
            cvar_95=cvar_95,
            avg_iv_rank=avg_iv_rank,
            gamma_pnl=None,  # Would need Greeks tracking
            theta_decay=None,
            vega_exposure=None
        )
    
    def run_benchmark_comparison(self, benchmark_symbol: str = "SPY") -> Dict[str, Any]:
        """Compare strategy performance to benchmark."""
        # Generate benchmark data
        benchmark_data = self.generate_synthetic_market_data(benchmark_symbol)
        benchmark_returns = benchmark_data['returns']
        
        if self.daily_returns.empty:
            raise ValueError("No strategy returns calculated. Run calculate_performance_metrics() first.")
        
        # Align dates
        common_dates = self.daily_returns.index.intersection(benchmark_returns.index)
        strategy_returns = self.daily_returns.loc[common_dates]
        bench_returns = benchmark_returns.loc[common_dates]
        
        # Calculate relative metrics
        correlation = strategy_returns.corr(bench_returns)
        
        # Beta calculation
        covariance = np.cov(strategy_returns, bench_returns)[0, 1]
        benchmark_variance = np.var(bench_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation (CAPM)
        strategy_annual_return = strategy_returns.mean() * 252
        benchmark_annual_return = bench_returns.mean() * 252
        alpha = strategy_annual_return - (self.config.risk_free_rate + beta * (benchmark_annual_return - self.config.risk_free_rate))
        
        # Information ratio
        excess_returns = strategy_returns - bench_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        return {
            'benchmark_symbol': benchmark_symbol,
            'correlation': correlation,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'benchmark_annual_return': benchmark_annual_return,
            'excess_return': strategy_annual_return - benchmark_annual_return
        }
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        results = self.calculate_performance_metrics()
        benchmark_comparison = self.run_benchmark_comparison()
        
        report = []
        report.append("📊 TRADING STRATEGY PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Strategy Period: {self.config.start_date.date()} to {self.config.end_date.date()}")
        report.append(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        report.append("")
        
        # Performance Summary
        report.append("🎯 PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Return:      {results.total_return:.2%}")
        report.append(f"Annualized Return: {results.annualized_return:.2%}")
        report.append(f"Volatility:        {results.volatility:.2%}")
        report.append(f"Sharpe Ratio:      {results.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio:     {results.sortino_ratio:.2f}")
        report.append(f"Calmar Ratio:      {results.calmar_ratio:.2f}")
        report.append("")
        
        # Risk Metrics
        report.append("⚠️  RISK ANALYSIS")
        report.append("-" * 30)
        report.append(f"Maximum Drawdown:  {results.max_drawdown:.2%}")
        report.append(f"Max DD Duration:   {results.max_drawdown_duration} days")
        report.append(f"VaR (95%):         {results.var_95:.2%}")
        report.append(f"CVaR (95%):        {results.cvar_95:.2%}")
        report.append("")
        
        # Trading Statistics
        report.append("📈 TRADING STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Trades:      {results.total_trades}")
        report.append(f"Winning Trades:    {results.winning_trades}")
        report.append(f"Losing Trades:     {results.losing_trades}")
        report.append(f"Win Rate:          {results.win_rate:.1%}")
        report.append(f"Average Win:       ${results.avg_win:.2f}")
        report.append(f"Average Loss:      ${results.avg_loss:.2f}")
        report.append(f"Profit Factor:     {results.profit_factor:.2f}")
        report.append(f"Expectancy:        ${results.expectancy:.2f}")
        report.append("")
        
        # Benchmark Comparison
        report.append("🏁 BENCHMARK COMPARISON")
        report.append("-" * 30)
        report.append(f"Benchmark:         {benchmark_comparison['benchmark_symbol']}")
        report.append(f"Alpha:             {benchmark_comparison['alpha']:.2%}")
        report.append(f"Beta:              {benchmark_comparison['beta']:.2f}")
        report.append(f"Correlation:       {benchmark_comparison['correlation']:.2f}")
        report.append(f"Information Ratio: {benchmark_comparison['information_ratio']:.2f}")
        report.append(f"Excess Return:     {benchmark_comparison['excess_return']:.2%}")
        report.append("")
        
        # Options Metrics (if applicable)
        if results.avg_iv_rank is not None:
            report.append("⚙️  OPTIONS ANALYSIS")
            report.append("-" * 30)
            report.append(f"Avg IV Rank:       {results.avg_iv_rank:.1%}")
            report.append("")
        
        return "\n".join(report)