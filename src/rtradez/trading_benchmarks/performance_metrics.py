"""
Trading performance metrics and analysis framework.

Comprehensive performance measurement for trading strategies including
risk-adjusted returns, drawdown analysis, and statistical measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import scipy.stats as stats
from pydantic import BaseModel, Field


class TradingMetrics(BaseModel):
    """Comprehensive trading performance metrics."""
    
    # Return Metrics
    total_return: float
    annualized_return: float
    compound_annual_growth_rate: float
    
    # Risk Metrics  
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    drawdown_frequency: float
    
    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    
    # Distribution Metrics
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk
    var_99: float
    cvar_95: float  # Conditional VaR
    cvar_99: float
    
    # Rolling Metrics
    rolling_sharpe_12m: Optional[float] = None
    rolling_volatility_12m: Optional[float] = None
    rolling_max_dd_12m: Optional[float] = None
    
    # Trading Statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0
    
    # Tail Risk
    tail_ratio: float = 0.0  # 95th percentile / 5th percentile
    gain_pain_ratio: float = 0.0
    
    # Benchmark Relative
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    treynor_ratio: Optional[float] = None


@dataclass
class DrawdownPeriod:
    """Individual drawdown period analysis."""
    start_date: datetime
    end_date: datetime
    trough_date: datetime
    peak_value: float
    trough_value: float
    recovery_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: int
    underwater_days: int


class PerformanceAnalyzer:
    """Comprehensive trading performance analysis framework."""
    
    def __init__(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            returns: Series of daily returns
            benchmark_returns: Optional benchmark returns for comparison
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252
        
        # Calculate cumulative returns and equity curve
        self.cumulative_returns = (1 + self.returns).cumprod()
        self.equity_curve = self.cumulative_returns * 100000  # Assume $100k starting capital
        
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic return and risk metrics."""
        if len(self.returns) == 0:
            return {}
        
        # Return metrics
        total_return = self.cumulative_returns.iloc[-1] - 1
        trading_days = len(self.returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = self.returns.std() * np.sqrt(252)
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'trading_days': trading_days,
            'years': years
        }
    
    def calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive drawdown analysis."""
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max
        
        # Basic drawdown metrics
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Drawdown periods
        drawdown_periods = self._identify_drawdown_periods(self.equity_curve, drawdown)
        
        # Drawdown frequency and duration
        num_drawdowns = len(drawdown_periods)
        total_days = len(self.equity_curve)
        drawdown_frequency = num_drawdowns / (total_days / 252) if total_days > 0 else 0  # Per year
        
        if drawdown_periods:
            max_drawdown_duration = max(dd.duration_days for dd in drawdown_periods)
            avg_drawdown_duration = np.mean([dd.duration_days for dd in drawdown_periods])
            max_recovery_days = max(dd.recovery_days for dd in drawdown_periods if dd.recovery_days > 0)
            avg_recovery_days = np.mean([dd.recovery_days for dd in drawdown_periods if dd.recovery_days > 0])
        else:
            max_drawdown_duration = 0
            avg_drawdown_duration = 0
            max_recovery_days = 0
            avg_recovery_days = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'drawdown_frequency': drawdown_frequency,
            'num_drawdown_periods': num_drawdowns,
            'max_recovery_days': max_recovery_days,
            'avg_recovery_days': avg_recovery_days,
            'drawdown_periods': drawdown_periods,
            'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0
        }
    
    def _identify_drawdown_periods(self, equity_curve: pd.Series, drawdown: pd.Series) -> List[DrawdownPeriod]:
        """Identify individual drawdown periods."""
        periods = []
        in_drawdown = False
        peak_date = None
        peak_value = None
        trough_date = None
        trough_value = None
        
        for date, (equity, dd) in zip(equity_curve.index, zip(equity_curve.values, drawdown.values)):
            if not in_drawdown and dd < 0:
                # Start of new drawdown
                in_drawdown = True
                peak_date = equity_curve.index[equity_curve.index < date][-1] if len(equity_curve.index[equity_curve.index < date]) > 0 else date
                peak_value = equity_curve.expanding().max().loc[date]
                trough_date = date
                trough_value = equity
                
            elif in_drawdown:
                if equity < trough_value:
                    # New trough
                    trough_date = date
                    trough_value = equity
                elif dd >= 0:
                    # Recovery - end of drawdown
                    recovery_date = date
                    recovery_value = equity
                    
                    # Calculate metrics for this period
                    drawdown_pct = (trough_value - peak_value) / peak_value
                    duration_days = (trough_date - peak_date).days
                    recovery_days = (recovery_date - trough_date).days
                    underwater_days = (recovery_date - peak_date).days
                    
                    periods.append(DrawdownPeriod(
                        start_date=peak_date,
                        end_date=recovery_date,
                        trough_date=trough_date,
                        peak_value=peak_value,
                        trough_value=trough_value,
                        recovery_value=recovery_value,
                        drawdown_pct=drawdown_pct,
                        duration_days=duration_days,
                        recovery_days=recovery_days,
                        underwater_days=underwater_days
                    ))
                    
                    in_drawdown = False
        
        return periods
    
    def calculate_risk_adjusted_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        basic = self.calculate_basic_metrics()
        drawdown = self.calculate_drawdown_metrics()
        
        if basic.get('volatility', 0) == 0:
            return {}
        
        # Excess returns over risk-free rate
        excess_returns = self.returns - self.daily_rf_rate
        
        # Sharpe Ratio
        sharpe_ratio = excess_returns.mean() / self.returns.std() * np.sqrt(252) if self.returns.std() > 0 else 0
        
        # Sortino Ratio
        downside_returns = self.returns[self.returns < self.daily_rf_rate]
        downside_dev = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_dev * np.sqrt(252) if downside_dev > 0 else 0
        
        # Calmar Ratio
        max_dd = abs(drawdown['max_drawdown'])
        calmar_ratio = basic['annualized_return'] / max_dd if max_dd > 0 else 0
        
        # Sterling Ratio (average annual return / average max drawdown)
        avg_annual_dd = abs(drawdown['avg_drawdown'])
        sterling_ratio = basic['annualized_return'] / avg_annual_dd if avg_annual_dd > 0 else 0
        
        # Burke Ratio (excess return / sqrt(sum of squared drawdowns))
        squared_dd_sum = np.sum([dd.drawdown_pct ** 2 for dd in drawdown['drawdown_periods']]) if drawdown['drawdown_periods'] else 0
        burke_ratio = excess_returns.mean() * 252 / np.sqrt(squared_dd_sum) if squared_dd_sum > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'sterling_ratio': sterling_ratio,
            'burke_ratio': burke_ratio
        }
    
    def calculate_distribution_metrics(self) -> Dict[str, float]:
        """Calculate return distribution metrics."""
        if len(self.returns) == 0:
            return {}
        
        # Moments
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)
        
        # Value at Risk
        var_95 = np.percentile(self.returns, 5)
        var_99 = np.percentile(self.returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = self.returns[self.returns <= var_95].mean() if (self.returns <= var_95).any() else 0
        cvar_99 = self.returns[self.returns <= var_99].mean() if (self.returns <= var_99).any() else 0
        
        # Tail ratio
        p95 = np.percentile(self.returns, 95)
        p5 = np.percentile(self.returns, 5)
        tail_ratio = abs(p95 / p5) if p5 != 0 else 0
        
        # Gain-Pain ratio (sum of gains / sum of losses)
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        gain_pain_ratio = gains / losses if losses > 0 else float('inf')
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'tail_ratio': tail_ratio,
            'gain_pain_ratio': gain_pain_ratio
        }
    
    def calculate_rolling_metrics(self, window_days: int = 252) -> Dict[str, float]:
        """Calculate rolling performance metrics."""
        if len(self.returns) < window_days:
            return {}
        
        # Rolling Sharpe ratio
        rolling_excess = self.returns.rolling(window_days).mean() - self.daily_rf_rate
        rolling_vol = self.returns.rolling(window_days).std()
        rolling_sharpe = (rolling_excess / rolling_vol * np.sqrt(252)).dropna()
        
        # Rolling volatility
        rolling_volatility = (rolling_vol * np.sqrt(252)).dropna()
        
        # Rolling max drawdown
        rolling_equity = (1 + self.returns).rolling(window_days).apply(lambda x: x.cumprod().iloc[-1], raw=False)
        rolling_max = rolling_equity.rolling(window_days).max()
        rolling_dd = ((rolling_equity - rolling_max) / rolling_max).dropna()
        rolling_max_dd = rolling_dd.rolling(window_days).min()
        
        return {
            'rolling_sharpe_12m': rolling_sharpe.iloc[-1] if len(rolling_sharpe) > 0 else None,
            'rolling_volatility_12m': rolling_volatility.iloc[-1] if len(rolling_volatility) > 0 else None,
            'rolling_max_dd_12m': rolling_max_dd.iloc[-1] if len(rolling_max_dd) > 0 else None,
            'avg_rolling_sharpe': rolling_sharpe.mean() if len(rolling_sharpe) > 0 else None,
            'sharpe_stability': rolling_sharpe.std() if len(rolling_sharpe) > 0 else None
        }
    
    def calculate_benchmark_metrics(self) -> Dict[str, float]:
        """Calculate benchmark-relative metrics."""
        if self.benchmark_returns is None or len(self.benchmark_returns) == 0:
            return {}
        
        # Align returns
        common_dates = self.returns.index.intersection(self.benchmark_returns.index)
        strategy_rets = self.returns.loc[common_dates]
        bench_rets = self.benchmark_returns.loc[common_dates]
        
        if len(strategy_rets) == 0:
            return {}
        
        # Beta and Alpha (CAPM)
        covariance = np.cov(strategy_rets, bench_rets)[0, 1]
        bench_variance = np.var(bench_rets)
        beta = covariance / bench_variance if bench_variance > 0 else 0
        
        strategy_mean = strategy_rets.mean() * 252
        bench_mean = bench_rets.mean() * 252
        alpha = strategy_mean - (self.risk_free_rate + beta * (bench_mean - self.risk_free_rate))
        
        # Information Ratio
        excess_returns = strategy_rets - bench_rets
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Treynor Ratio
        strategy_excess = strategy_mean - self.risk_free_rate
        treynor_ratio = strategy_excess / beta if beta > 0 else 0
        
        # Correlation
        correlation = strategy_rets.corr(bench_rets)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'treynor_ratio': treynor_ratio,
            'correlation': correlation
        }
    
    def calculate_trading_metrics(self, trade_returns: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        if trade_returns is None:
            return {}
        
        trade_returns = np.array(trade_returns)
        
        # Basic trading stats
        total_trades = len(trade_returns)
        winning_trades = np.sum(trade_returns > 0)
        losing_trades = np.sum(trade_returns < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(trade_returns[trade_returns > 0])
        gross_loss = abs(np.sum(trade_returns[trade_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        avg_win = np.mean(trade_returns[trade_returns > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean(trade_returns[trade_returns < 0]) if losing_trades > 0 else 0
        expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)
        
        # Kelly Criterion
        if avg_loss < 0:  # avg_loss should be negative
            kelly_f = win_rate - ((1 - win_rate) / abs(avg_win / avg_loss))
        else:
            kelly_f = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'kelly_criterion': kelly_f,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': np.max(trade_returns) if len(trade_returns) > 0 else 0,
            'worst_trade': np.min(trade_returns) if len(trade_returns) > 0 else 0
        }
    
    def generate_comprehensive_metrics(self, trade_returns: Optional[List[float]] = None) -> TradingMetrics:
        """Generate comprehensive trading metrics."""
        basic = self.calculate_basic_metrics()
        drawdown = self.calculate_drawdown_metrics()
        risk_adjusted = self.calculate_risk_adjusted_metrics()
        distribution = self.calculate_distribution_metrics()
        rolling = self.calculate_rolling_metrics()
        benchmark = self.calculate_benchmark_metrics()
        trading = self.calculate_trading_metrics(trade_returns)
        
        return TradingMetrics(
            # Return metrics
            total_return=basic.get('total_return', 0),
            annualized_return=basic.get('annualized_return', 0),
            compound_annual_growth_rate=basic.get('annualized_return', 0),  # Same as annualized return
            
            # Risk metrics
            volatility=basic.get('volatility', 0),
            downside_deviation=basic.get('downside_deviation', 0),
            max_drawdown=drawdown.get('max_drawdown', 0),
            max_drawdown_duration=drawdown.get('max_drawdown_duration', 0),
            avg_drawdown=drawdown.get('avg_drawdown', 0),
            drawdown_frequency=drawdown.get('drawdown_frequency', 0),
            
            # Risk-adjusted returns
            sharpe_ratio=risk_adjusted.get('sharpe_ratio', 0),
            sortino_ratio=risk_adjusted.get('sortino_ratio', 0),
            calmar_ratio=risk_adjusted.get('calmar_ratio', 0),
            sterling_ratio=risk_adjusted.get('sterling_ratio', 0),
            burke_ratio=risk_adjusted.get('burke_ratio', 0),
            
            # Distribution metrics
            skewness=distribution.get('skewness', 0),
            kurtosis=distribution.get('kurtosis', 0),
            var_95=distribution.get('var_95', 0),
            var_99=distribution.get('var_99', 0),
            cvar_95=distribution.get('cvar_95', 0),
            cvar_99=distribution.get('cvar_99', 0),
            
            # Rolling metrics
            rolling_sharpe_12m=rolling.get('rolling_sharpe_12m'),
            rolling_volatility_12m=rolling.get('rolling_volatility_12m'),
            rolling_max_dd_12m=rolling.get('rolling_max_dd_12m'),
            
            # Trading metrics
            total_trades=trading.get('total_trades', 0),
            win_rate=trading.get('win_rate', 0),
            profit_factor=trading.get('profit_factor', 0),
            expectancy=trading.get('expectancy', 0),
            kelly_criterion=trading.get('kelly_criterion', 0),
            
            # Tail risk
            tail_ratio=distribution.get('tail_ratio', 0),
            gain_pain_ratio=distribution.get('gain_pain_ratio', 0),
            
            # Benchmark relative
            alpha=benchmark.get('alpha'),
            beta=benchmark.get('beta'),
            information_ratio=benchmark.get('information_ratio'),
            tracking_error=benchmark.get('tracking_error'),
            treynor_ratio=benchmark.get('treynor_ratio')
        )
    
    def generate_performance_summary(self) -> str:
        """Generate a formatted performance summary."""
        metrics = self.generate_comprehensive_metrics()
        
        summary = []
        summary.append("📈 COMPREHENSIVE PERFORMANCE ANALYSIS")
        summary.append("=" * 50)
        summary.append("")
        
        # Returns
        summary.append("💰 RETURNS")
        summary.append(f"Total Return:          {metrics.total_return:.2%}")
        summary.append(f"Annualized Return:     {metrics.annualized_return:.2%}")
        summary.append(f"Volatility:            {metrics.volatility:.2%}")
        summary.append("")
        
        # Risk-Adjusted Metrics
        summary.append("⚖️  RISK-ADJUSTED METRICS")
        summary.append(f"Sharpe Ratio:          {metrics.sharpe_ratio:.2f}")
        summary.append(f"Sortino Ratio:         {metrics.sortino_ratio:.2f}")
        summary.append(f"Calmar Ratio:          {metrics.calmar_ratio:.2f}")
        summary.append("")
        
        # Risk Metrics
        summary.append("⚠️  RISK ANALYSIS")
        summary.append(f"Max Drawdown:          {metrics.max_drawdown:.2%}")
        summary.append(f"Max DD Duration:       {metrics.max_drawdown_duration} days")
        summary.append(f"VaR (95%):             {metrics.var_95:.2%}")
        summary.append(f"CVaR (95%):            {metrics.cvar_95:.2%}")
        summary.append("")
        
        # Distribution
        summary.append("📊 RETURN DISTRIBUTION")
        summary.append(f"Skewness:              {metrics.skewness:.2f}")
        summary.append(f"Kurtosis:              {metrics.kurtosis:.2f}")
        summary.append(f"Tail Ratio:            {metrics.tail_ratio:.2f}")
        summary.append("")
        
        # Trading Stats
        if metrics.total_trades > 0:
            summary.append("🎯 TRADING STATISTICS")
            summary.append(f"Total Trades:          {metrics.total_trades}")
            summary.append(f"Win Rate:              {metrics.win_rate:.1%}")
            summary.append(f"Profit Factor:         {metrics.profit_factor:.2f}")
            summary.append(f"Expectancy:            {metrics.expectancy:.4f}")
            summary.append("")
        
        # Benchmark Comparison
        if metrics.alpha is not None:
            summary.append("🏁 vs BENCHMARK")
            summary.append(f"Alpha:                 {metrics.alpha:.2%}")
            summary.append(f"Beta:                  {metrics.beta:.2f}")
            summary.append(f"Information Ratio:     {metrics.information_ratio:.2f}")
            summary.append("")
        
        return "\n".join(summary)