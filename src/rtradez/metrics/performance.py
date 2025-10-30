"""
Performance analysis with sklearn-like interface.

Provides comprehensive performance metrics following sklearn patterns:
- .fit(returns) -> learn baseline parameters  
- .score(y_true, y_pred) -> calculate performance score
- .transform(returns) -> normalize/adjust returns
"""

import pandas as pd
import numpy as np
import quantstats as qs
from typing import Dict, Optional, Union
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base import BaseMetric

warnings.filterwarnings('ignore')


class PerformanceAnalyzer(BaseMetric):
    """
    Comprehensive performance analysis using QuantStats.
    
    Provides standard financial metrics, risk analysis, and reporting
    specifically tailored for options trading strategies.
    """
    
    def __init__(self, benchmark: Optional[pd.Series] = None, 
                 risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer with sklearn-like interface.
        
        Args:
            benchmark: Benchmark returns (e.g., SPY) for comparison
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        super().__init__()
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        
        # Fitted parameters (set during fit)
        self.baseline_sharpe_ = None
        self.baseline_volatility_ = None
        self.benchmark_aligned_ = None
            
    def get_metrics(self) -> Dict:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = qs.stats.comp(self.returns)
        metrics['cagr'] = qs.stats.cagr(self.returns)
        metrics['volatility'] = qs.stats.volatility(self.returns)
        
        # Risk metrics
        metrics['sharpe_ratio'] = qs.stats.sharpe(self.returns)
        metrics['sortino_ratio'] = qs.stats.sortino(self.returns)
        metrics['calmar_ratio'] = qs.stats.calmar(self.returns)
        
        # Drawdown analysis
        metrics['max_drawdown'] = qs.stats.max_drawdown(self.returns)
        
        # Calculate custom average drawdown since qs doesn't have this method
        drawdowns = qs.stats.to_drawdown_series(self.returns)
        negative_drawdowns = drawdowns[drawdowns < 0]
        metrics['avg_drawdown'] = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0
        
        # Custom avg drawdown days calculation
        metrics['avg_drawdown_days'] = self._calculate_avg_drawdown_days(drawdowns)
        
        # Distribution metrics
        metrics['skewness'] = qs.stats.skew(self.returns)
        metrics['kurtosis'] = qs.stats.kurtosis(self.returns)
        metrics['tail_ratio'] = qs.stats.tail_ratio(self.returns)
        
        # Win/loss analysis
        metrics['win_rate'] = qs.stats.win_rate(self.returns)
        metrics['avg_win'] = qs.stats.avg_win(self.returns)
        metrics['avg_loss'] = qs.stats.avg_loss(self.returns)
        metrics['profit_factor'] = qs.stats.profit_factor(self.returns)
        
        # Risk measures
        metrics['var_95'] = qs.stats.value_at_risk(self.returns, confidence=0.95)
        metrics['cvar_95'] = qs.stats.conditional_value_at_risk(self.returns, confidence=0.95)
        
        # Options-specific metrics
        metrics['options_specific'] = self._calculate_options_metrics()
        
        # Benchmark comparison if available
        if self.benchmark is not None:
            metrics['benchmark_comparison'] = self._calculate_benchmark_metrics()
            
        return metrics
        
    def _calculate_options_metrics(self) -> Dict:
        """Calculate options-specific performance metrics."""
        options_metrics = {}
        
        # Theta decay efficiency (higher is better for short strategies)
        daily_returns = self.returns.resample('D').sum()
        positive_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        options_metrics['theta_efficiency'] = positive_days / total_days if total_days > 0 else 0
        
        # Volatility risk premium capture
        # Measures how well strategy captures vol risk premium
        if len(self.returns) > 20:
            rolling_vol = self.returns.rolling(20).std() * np.sqrt(252)
            vol_risk_premium = rolling_vol.mean() - self.returns.std() * np.sqrt(252)
            options_metrics['vol_risk_premium'] = vol_risk_premium
        else:
            options_metrics['vol_risk_premium'] = np.nan
            
        # Drawdown recovery (important for options strategies)
        drawdowns = qs.stats.to_drawdown_series(self.returns)
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for date, dd in drawdowns.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = date
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                if drawdown_start:
                    recovery_days = (date - drawdown_start).days
                    recovery_times.append(recovery_days)
                    
        options_metrics['avg_recovery_days'] = np.mean(recovery_times) if recovery_times else np.nan
        
        return options_metrics
        
    def _calculate_benchmark_metrics(self) -> Dict:
        """Calculate benchmark comparison metrics."""
        benchmark_metrics = {}
        
        # Relative performance
        benchmark_metrics['excess_return'] = qs.stats.comp(self.returns) - qs.stats.comp(self.benchmark)
        benchmark_metrics['tracking_error'] = qs.stats.volatility(self.returns - self.benchmark)
        benchmark_metrics['information_ratio'] = (
            qs.stats.cagr(self.returns) - qs.stats.cagr(self.benchmark)
        ) / benchmark_metrics['tracking_error']
        
        # Beta and correlation
        benchmark_metrics['beta'] = qs.stats.beta(self.returns, self.benchmark)
        benchmark_metrics['correlation'] = self.returns.corr(self.benchmark)
        
        # Up/down capture ratios
        up_market = self.benchmark > 0
        down_market = self.benchmark < 0
        
        if up_market.sum() > 0:
            benchmark_metrics['up_capture'] = (
                self.returns[up_market].mean() / self.benchmark[up_market].mean()
            )
        else:
            benchmark_metrics['up_capture'] = np.nan
            
        if down_market.sum() > 0:
            benchmark_metrics['down_capture'] = (
                self.returns[down_market].mean() / self.benchmark[down_market].mean()
            )
        else:
            benchmark_metrics['down_capture'] = np.nan
            
        return benchmark_metrics
        
    def score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Calculate performance score (Sharpe ratio).
        
        Args:
            y_true: Actual returns
            y_pred: Strategy returns
            
        Returns:
            Sharpe ratio
        """
        if len(y_pred) == 0 or y_pred.std() == 0:
            return 0.0
            
        return y_pred.mean() / y_pred.std() * np.sqrt(252)
        
    def _calculate_avg_drawdown_days(self, drawdowns: pd.Series) -> float:
        """Calculate average drawdown duration in days."""
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdowns.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if start_date:
                    period_days = (date - start_date).days
                    drawdown_periods.append(period_days)
                    
        return np.mean(drawdown_periods) if drawdown_periods else 0
        
    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        return qs.stats.sharpe(self.returns, rf=risk_free_rate)
        
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        return qs.stats.max_drawdown(self.returns)
        
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        return qs.stats.calmar(self.returns)
        
    def rolling_sharpe(self, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        return qs.stats.rolling_sharpe(self.returns, window=window)
        
    def rolling_volatility(self, window: int = 252) -> pd.Series:
        """Calculate rolling volatility."""
        return qs.stats.rolling_volatility(self.returns, window=window)
        
    def generate_report(self, output_file: Optional[str] = None, 
                       title: str = "Strategy Performance Report") -> str:
        """
        Generate comprehensive HTML performance report.
        
        Args:
            output_file: Path to save HTML report
            title: Report title
            
        Returns:
            HTML report as string
        """
        if self.benchmark is not None:
            html_report = qs.reports.html(
                self.returns, 
                benchmark=self.benchmark,
                output=output_file,
                title=title,
                download_filename=output_file
            )
        else:
            html_report = qs.reports.html(
                self.returns,
                output=output_file, 
                title=title,
                download_filename=output_file
            )
            
        return html_report
        
    def plot_returns(self, **kwargs):
        """Plot cumulative returns using QuantStats."""
        if self.benchmark is not None:
            return qs.plots.returns(self.returns, benchmark=self.benchmark, **kwargs)
        else:
            return qs.plots.returns(self.returns, **kwargs)
            
    def plot_drawdown(self, **kwargs):
        """Plot drawdown chart."""
        return qs.plots.drawdown(self.returns, **kwargs)
        
    def plot_rolling_sharpe(self, **kwargs):
        """Plot rolling Sharpe ratio."""
        return qs.plots.rolling_sharpe(self.returns, **kwargs)
        
    def plot_monthly_heatmap(self, **kwargs):
        """Plot monthly returns heatmap."""
        return qs.plots.monthly_heatmap(self.returns, **kwargs)
        
    def __repr__(self) -> str:
        """String representation."""
        total_ret = qs.stats.comp(self.returns)
        sharpe = qs.stats.sharpe(self.returns)
        max_dd = qs.stats.max_drawdown(self.returns)
        
        return f"PerformanceAnalyzer(total_return={total_ret:.2%}, sharpe={sharpe:.2f}, max_dd={max_dd:.2%})"