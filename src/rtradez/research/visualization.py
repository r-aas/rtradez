"""Advanced visualization tools for options research and analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import warnings

from .regime_detection import MarketRegimeDetector
from .greeks_analysis import GreeksProfile, GreeksAnalyzer
from .advanced_backtest import AdvancedBacktester


class ResearchVisualizer:
    """
    Comprehensive visualization suite for options research.
    
    Provides:
    - Performance attribution charts
    - Greeks exposure analysis
    - Market regime visualization
    - Strategy comparison plots
    - Risk analysis dashboards
    """
    
    def __init__(self, style: str = 'plotly_white', figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize research visualizer.
        
        Args:
            style: Plotting style ('plotly_white', 'seaborn', 'matplotlib')
            figure_size: Default figure size
        """
        self.style = style
        self.figure_size = figure_size
        
        # Set matplotlib style
        if style == 'seaborn':
            sns.set_style("whitegrid")
            plt.style.use('seaborn-v0_8')
        
        # Color palettes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'regime_colors': ['#e74c3c', '#f39c12', '#27ae60'],
            'strategy_colors': ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
        }
    
    def plot_performance_attribution(self, 
                                   backtest_results: Dict,
                                   attribution_data: Optional[Dict] = None) -> go.Figure:
        """
        Create comprehensive performance attribution chart.
        
        Args:
            backtest_results: Results from AdvancedBacktester
            attribution_data: Optional Greeks P&L attribution data
        
        Returns:
            Plotly figure with performance attribution
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Equity Curve', 'Drawdown', 'Returns Distribution', 
                          'Greeks P&L Attribution', 'Trade Analysis', 'Risk Metrics'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        equity_curve = backtest_results['equity_curve']
        
        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['total_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # 2. Drawdown
        equity_values = equity_curve['total_value']
        cummax = equity_values.cummax()
        drawdown = (equity_values - cummax) / cummax * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'],
                y=drawdown,
                mode='lines',
                fill='tonexty',
                name='Drawdown %',
                line=dict(color=self.colors['danger'], width=1),
                fillcolor='rgba(214, 39, 40, 0.3)'
            ),
            row=1, col=2
        )
        
        # 3. Returns Distribution
        returns = equity_values.pct_change().dropna()
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=30,
                name='Daily Returns %',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. Greeks P&L Attribution (if provided)
        if attribution_data:
            greek_names = ['Delta P&L', 'Gamma P&L', 'Theta P&L', 'Vega P&L']
            greek_values = [
                attribution_data.get('delta_pnl', 0),
                attribution_data.get('gamma_pnl', 0), 
                attribution_data.get('theta_pnl', 0),
                attribution_data.get('vega_pnl', 0)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=greek_names,
                    y=greek_values,
                    name='Greeks P&L',
                    marker_color=self.colors['strategy_colors']
                ),
                row=2, col=2
            )
        
        # 5. Trade Analysis
        if 'trades' in backtest_results and backtest_results['trades']:
            trades_df = pd.DataFrame([
                {
                    'date': trade.timestamp,
                    'pnl': trade.quantity * trade.price,
                    'commission': trade.commission
                }
                for trade in backtest_results['trades']
            ])
            
            monthly_trades = trades_df.groupby(trades_df['date'].dt.to_period('M')).agg({
                'pnl': 'sum',
                'commission': 'sum'
            }).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=monthly_trades['date'].astype(str),
                    y=monthly_trades['pnl'],
                    name='Monthly P&L',
                    marker_color=self.colors['success']
                ),
                row=3, col=1
            )
        
        # 6. Risk Metrics Summary
        metrics = backtest_results.get('performance_metrics', {})
        metric_names = ['Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Annual Return']
        metric_values = [
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown', 0) * 100,
            metrics.get('win_rate', 0) * 100,
            metrics.get('annual_return', 0) * 100
        ]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name='Risk Metrics',
                marker_color=self.colors['info']
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Comprehensive Performance Attribution Analysis',
            height=1000,
            showlegend=True,
            template=self.style
        )
        
        return fig
    
    def plot_greeks_exposure(self, 
                           greeks_timeline: List[Tuple[pd.Timestamp, GreeksProfile]],
                           price_timeline: Optional[pd.Series] = None) -> go.Figure:
        """
        Visualize Greeks exposure over time.
        
        Args:
            greeks_timeline: List of (timestamp, GreeksProfile) tuples
            price_timeline: Optional underlying price timeline
        
        Returns:
            Plotly figure with Greeks exposure analysis
        """
        if not greeks_timeline:
            return go.Figure()
        
        # Convert to DataFrame
        timeline_data = []
        for timestamp, greeks in greeks_timeline:
            timeline_data.append({
                'date': timestamp,
                'delta': greeks.delta,
                'gamma': greeks.gamma,
                'theta': greeks.theta,
                'vega': greeks.vega,
                'delta_dollars': greeks.delta_dollars,
                'gamma_dollars': greeks.gamma_dollars
            })
        
        df = pd.DataFrame(timeline_data)
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Delta Exposure', 'Gamma Exposure', 'Theta Exposure',
                          'Vega Exposure', 'Delta ($)', 'Underlying Price'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Delta
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['delta'],
                mode='lines+markers',
                name='Delta',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        # Gamma
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['gamma'],
                mode='lines+markers', 
                name='Gamma',
                line=dict(color=self.colors['secondary'])
            ),
            row=1, col=2
        )
        
        # Theta
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['theta'],
                mode='lines+markers',
                name='Theta',
                line=dict(color=self.colors['danger'])
            ),
            row=2, col=1
        )
        
        # Vega
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['vega'],
                mode='lines+markers',
                name='Vega',
                line=dict(color=self.colors['success'])
            ),
            row=2, col=2
        )
        
        # Delta Dollars
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['delta_dollars'],
                mode='lines+markers',
                name='Delta $',
                line=dict(color=self.colors['info'])
            ),
            row=3, col=1
        )
        
        # Underlying Price (if provided)
        if price_timeline is not None:
            fig.add_trace(
                go.Scatter(
                    x=price_timeline.index, y=price_timeline.values,
                    mode='lines',
                    name='Underlying Price',
                    line=dict(color='black', width=2)
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title='Portfolio Greeks Exposure Analysis',
            height=900,
            showlegend=True,
            template=self.style
        )
        
        return fig
    
    def plot_market_regimes(self, 
                          market_data: pd.DataFrame,
                          regime_detector: MarketRegimeDetector) -> go.Figure:
        """
        Visualize market regimes and their characteristics.
        
        Args:
            market_data: Historical market data
            regime_detector: Fitted regime detector
        
        Returns:
            Plotly figure with regime analysis
        """
        # Transform data to include regimes
        data_with_regimes = regime_detector.transform(market_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Price with Regimes', 'Volatility by Regime',
                          'Regime Transition Matrix', 'Regime Statistics'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Price chart colored by regime
        for regime in range(regime_detector.n_regimes):
            regime_data = data_with_regimes[data_with_regimes['regime'] == regime]
            if len(regime_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=regime_data.index,
                        y=regime_data['Close'],
                        mode='markers',
                        name=f"Regime {regime}",
                        marker=dict(
                            color=self.colors['regime_colors'][regime % len(self.colors['regime_colors'])],
                            size=4
                        )
                    ),
                    row=1, col=1
                )
        
        # 2. Volatility by regime
        regime_vol_data = []
        for regime in range(regime_detector.n_regimes):
            regime_data = data_with_regimes[data_with_regimes['regime'] == regime]
            if len(regime_data) > 0:
                vol_data = regime_data['returns'].std() * np.sqrt(252)
                regime_vol_data.append(vol_data)
            else:
                regime_vol_data.append(0)
        
        fig.add_trace(
            go.Bar(
                x=[f'Regime {i}' for i in range(regime_detector.n_regimes)],
                y=regime_vol_data,
                name='Volatility',
                marker_color=self.colors['regime_colors'][:regime_detector.n_regimes]
            ),
            row=1, col=2
        )
        
        # 3. Regime transition matrix
        regime_series = data_with_regimes['regime']
        transition_matrix = np.zeros((regime_detector.n_regimes, regime_detector.n_regimes))
        
        for i in range(len(regime_series) - 1):
            current_regime = int(regime_series.iloc[i])
            next_regime = int(regime_series.iloc[i + 1])
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        transition_matrix = np.nan_to_num(transition_matrix)
        
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix,
                x=[f'To Regime {i}' for i in range(regime_detector.n_regimes)],
                y=[f'From Regime {i}' for i in range(regime_detector.n_regimes)],
                colorscale='Blues',
                showscale=True
            ),
            row=2, col=1
        )
        
        # 4. Regime statistics
        regime_stats = regime_detector.get_regime_statistics(market_data)
        
        fig.add_trace(
            go.Bar(
                x=regime_stats['regime_name'],
                y=regime_stats['frequency'] * 100,
                name='Frequency %',
                marker_color=self.colors['regime_colors'][:len(regime_stats)]
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Market Regime Analysis',
            height=800,
            showlegend=True,
            template=self.style
        )
        
        return fig
    
    def plot_strategy_comparison(self, 
                               strategy_results: Dict[str, Dict],
                               metrics: List[str] = None) -> go.Figure:
        """
        Compare multiple strategies across key metrics.
        
        Args:
            strategy_results: Dict mapping strategy names to results
            metrics: List of metrics to compare
        
        Returns:
            Plotly figure with strategy comparison
        """
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        strategies = list(strategy_results.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{metric.replace("_", " ").title()}' for metric in metrics[:4]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, metric in enumerate(metrics[:4]):
            if i >= len(positions):
                break
                
            row, col = positions[i]
            
            values = []
            for strategy in strategies:
                result = strategy_results[strategy]
                if 'performance_metrics' in result:
                    value = result['performance_metrics'].get(metric, 0)
                    if metric in ['max_drawdown']:
                        value = abs(value) * 100  # Convert to positive percentage
                    elif metric in ['total_return', 'annual_return', 'win_rate']:
                        value = value * 100  # Convert to percentage
                    values.append(value)
                else:
                    values.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    marker_color=self.colors['strategy_colors'][i % len(self.colors['strategy_colors'])]
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Strategy Performance Comparison',
            height=600,
            showlegend=False,
            template=self.style
        )
        
        return fig
    
    def create_research_dashboard(self,
                                backtest_results: Dict,
                                greeks_timeline: Optional[List] = None,
                                regime_data: Optional[Tuple] = None) -> go.Figure:
        """
        Create comprehensive research dashboard.
        
        Args:
            backtest_results: Backtest results
            greeks_timeline: Optional Greeks timeline data
            regime_data: Optional (market_data, regime_detector) tuple
        
        Returns:
            Interactive dashboard figure
        """
        # This would create a comprehensive dashboard
        # For now, return the performance attribution chart
        return self.plot_performance_attribution(backtest_results)


class InteractivePlotter:
    """Interactive plotting utilities for research analysis."""
    
    def __init__(self):
        """Initialize interactive plotter."""
        pass
    
    def create_volatility_surface_3d(self, 
                                   options_data: pd.DataFrame,
                                   date: str) -> go.Figure:
        """
        Create 3D volatility surface plot.
        
        Args:
            options_data: Options chain data with IV, strikes, expiration
            date: Date for surface
        
        Returns:
            3D surface plot of implied volatility
        """
        # Filter data for specific date
        date_data = options_data[options_data['date'] == date]
        
        if date_data.empty:
            return go.Figure()
        
        # Pivot data for surface
        surface_data = date_data.pivot_table(
            index='strike',
            columns='expiration', 
            values='implied_volatility',
            fill_value=np.nan
        )
        
        fig = go.Figure(data=[
            go.Surface(
                z=surface_data.values,
                x=surface_data.columns,
                y=surface_data.index,
                colorscale='Viridis',
                name='Implied Volatility'
            )
        ])
        
        fig.update_layout(
            title=f'Implied Volatility Surface - {date}',
            scene=dict(
                xaxis_title='Expiration',
                yaxis_title='Strike',
                zaxis_title='Implied Volatility'
            ),
            height=600
        )
        
        return fig
    
    def create_interactive_backtest(self, equity_curve: pd.DataFrame) -> go.Figure:
        """
        Create interactive backtest visualization with multiple metrics.
        
        Args:
            equity_curve: Equity curve data
        
        Returns:
            Interactive backtest plot
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Portfolio Value', 'Drawdown', 'Daily Returns'],
            vertical_spacing=0.08
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['total_value'],
                mode='lines',
                name='Portfolio Value',
                hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Drawdown
        returns = equity_curve['total_value'].pct_change()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'],
                y=drawdown,
                mode='lines',
                fill='tonexty',
                name='Drawdown %',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Daily returns
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'][1:],
                y=returns[1:] * 100,
                mode='lines',
                name='Daily Returns %',
                hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Interactive Backtest Analysis',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig