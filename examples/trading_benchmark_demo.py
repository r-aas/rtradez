#!/usr/bin/env python3
"""
RTradez Trading Strategy Benchmark Demonstration.

Shows comprehensive backtesting, performance analysis, and strategy comparison
for options trading strategies.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.rtradez.trading_benchmarks import (
    StrategyBenchmark, BacktestConfig, PerformanceAnalyzer, TradingMetrics
)

console = Console()


def demo_covered_call_backtest():
    """Demonstrate covered call strategy backtesting."""
    
    console.print("\n📈 [bold blue]Covered Call Strategy Backtest[/bold blue]")
    console.print("=" * 50)
    
    # Setup backtest configuration
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=100000,
        commission_per_trade=1.0,
        slippage_bps=5.0,
        benchmark_symbol="SPY"
    )
    
    console.print(f"📅 Period: {config.start_date.date()} to {config.end_date.date()}")
    console.print(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
    console.print(f"🏁 Benchmark: {config.benchmark_symbol}")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Running covered call backtest...", total=None)
        
        # Initialize and run backtest
        backtest = StrategyBenchmark(config)
        
        progress.update(task, description="[cyan]Executing covered call strategy...")
        backtest.run_simple_covered_call_strategy()
        
        progress.update(task, description="[cyan]Calculating performance metrics...")
        results = backtest.calculate_performance_metrics()
        
        progress.update(task, description="[cyan]Running benchmark comparison...")
        benchmark_comparison = backtest.run_benchmark_comparison()
    
    # Display results
    console.print("✅ [bold green]Backtest completed successfully![/bold green]\n")
    
    # Performance summary table
    perf_table = Table(title="📊 Strategy Performance Summary", show_header=True)
    perf_table.add_column("Metric", style="cyan", width=20)
    perf_table.add_column("Value", style="white", width=15)
    perf_table.add_column("Assessment", style="bright_blue", width=15)
    
    # Performance assessment
    def assess_performance(metric, value):
        assessments = {
            'total_return': [(0.15, "Excellent"), (0.08, "Good"), (0.02, "Fair")],
            'sharpe_ratio': [(1.5, "Excellent"), (1.0, "Good"), (0.5, "Fair")],
            'max_drawdown': [(-0.05, "Excellent"), (-0.10, "Good"), (-0.20, "Fair")],
            'win_rate': [(0.7, "Excellent"), (0.6, "Good"), (0.5, "Fair")]
        }
        
        if metric in assessments:
            for threshold, assessment in assessments[metric]:
                if (metric == 'max_drawdown' and value >= threshold) or \
                   (metric != 'max_drawdown' and value >= threshold):
                    return assessment
        return "Poor"
    
    # Add performance rows
    metrics_to_show = [
        ('Total Return', results.total_return, 'total_return'),
        ('Annualized Return', results.annualized_return, 'total_return'),
        ('Volatility', results.volatility, None),
        ('Sharpe Ratio', results.sharpe_ratio, 'sharpe_ratio'),
        ('Sortino Ratio', results.sortino_ratio, 'sharpe_ratio'),
        ('Max Drawdown', results.max_drawdown, 'max_drawdown'),
        ('Win Rate', results.win_rate, 'win_rate'),
        ('Profit Factor', results.profit_factor, None),
    ]
    
    for metric_name, value, assess_key in metrics_to_show:
        if isinstance(value, float):
            if 'Rate' in metric_name or 'Return' in metric_name or 'Drawdown' in metric_name:
                display_value = f"{value:.2%}"
            else:
                display_value = f"{value:.2f}"
        else:
            display_value = str(value)
        
        assessment = assess_performance(assess_key, value) if assess_key else "N/A"
        perf_table.add_row(metric_name, display_value, assessment)
    
    console.print(perf_table)
    console.print()
    
    # Benchmark comparison
    if benchmark_comparison:
        bench_table = Table(title="🏁 vs SPY Benchmark", show_header=True)
        bench_table.add_column("Metric", style="cyan")
        bench_table.add_column("Strategy", style="white")
        bench_table.add_column("Benchmark", style="white")
        bench_table.add_column("Difference", style="green")
        
        strategy_return = results.annualized_return
        benchmark_return = benchmark_comparison['benchmark_annual_return']
        
        bench_table.add_row(
            "Annual Return",
            f"{strategy_return:.2%}",
            f"{benchmark_return:.2%}",
            f"{strategy_return - benchmark_return:+.2%}"
        )
        bench_table.add_row("Alpha", f"{benchmark_comparison['alpha']:.2%}", "0.00%", f"{benchmark_comparison['alpha']:+.2%}")
        bench_table.add_row("Beta", f"{benchmark_comparison['beta']:.2f}", "1.00", f"{benchmark_comparison['beta'] - 1.0:+.2f}")
        bench_table.add_row("Correlation", f"{benchmark_comparison['correlation']:.2f}", "1.00", f"{benchmark_comparison['correlation'] - 1.0:+.2f}")
        
        console.print(bench_table)
        console.print()
    
    # Trading statistics
    if results.total_trades > 0:
        trade_table = Table(title="📊 Trading Statistics", show_header=True)
        trade_table.add_column("Metric", style="cyan")
        trade_table.add_column("Value", style="white")
        
        trade_table.add_row("Total Trades", str(results.total_trades))
        trade_table.add_row("Winning Trades", str(results.winning_trades))
        trade_table.add_row("Losing Trades", str(results.losing_trades))
        trade_table.add_row("Average Win", f"${results.avg_win:.2f}")
        trade_table.add_row("Average Loss", f"${results.avg_loss:.2f}")
        trade_table.add_row("Expectancy", f"${results.expectancy:.2f}")
        
        console.print(trade_table)
        console.print()
    
    # Risk analysis
    risk_table = Table(title="⚠️ Risk Analysis", show_header=True)
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", style="white")
    risk_table.add_column("Interpretation", style="yellow")
    
    risk_metrics = [
        ("Max Drawdown", f"{results.max_drawdown:.2%}", "Largest peak-to-trough decline"),
        ("Max DD Duration", f"{results.max_drawdown_duration} days", "Longest recovery period"),
        ("VaR (95%)", f"{results.var_95:.2%}", "Daily loss exceeded 5% of time"),
        ("CVaR (95%)", f"{results.cvar_95:.2%}", "Average loss when VaR exceeded"),
    ]
    
    for metric, value, interpretation in risk_metrics:
        risk_table.add_row(metric, value, interpretation)
    
    console.print(risk_table)
    
    return results, benchmark_comparison


def demo_performance_analysis():
    """Demonstrate detailed performance analysis."""
    
    console.print("\n🔍 [bold blue]Performance Analysis Demo[/bold blue]")
    console.print("=" * 50)
    
    # Generate synthetic daily returns for demonstration
    np.random.seed(42)
    
    # Strategy returns (slightly outperforming with higher volatility)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    strategy_returns = np.random.normal(0.0008, 0.018, len(dates))  # ~20% annual return, 28% vol
    
    # Add some momentum and volatility clustering
    for i in range(1, len(strategy_returns)):
        strategy_returns[i] += 0.05 * strategy_returns[i-1]  # Slight momentum
        if abs(strategy_returns[i-1]) > 0.03:  # Volatility clustering
            strategy_returns[i] *= 1.3
    
    # Benchmark returns (market-like)
    benchmark_returns = np.random.normal(0.0005, 0.015, len(dates))  # ~12% annual return, 24% vol
    
    strategy_series = pd.Series(strategy_returns, index=dates)
    benchmark_series = pd.Series(benchmark_returns, index=dates)
    
    console.print(f"📊 Analyzing {len(strategy_series)} daily returns")
    console.print(f"📅 Period: {dates[0].date()} to {dates[-1].date()}")
    console.print()
    
    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer(
        returns=strategy_series,
        benchmark_returns=benchmark_series,
        risk_free_rate=0.02
    )
    
    with console.status("[bold green]Calculating comprehensive metrics..."):
        # Generate comprehensive metrics
        metrics = analyzer.generate_comprehensive_metrics()
        
        # Additional analysis
        drawdown_analysis = analyzer.calculate_drawdown_metrics()
        rolling_metrics = analyzer.calculate_rolling_metrics()
    
    # Display comprehensive results
    console.print("✅ [bold green]Analysis completed![/bold green]\n")
    
    # Main performance table
    main_table = Table(title="📈 Comprehensive Performance Metrics", show_header=True)
    main_table.add_column("Category", style="bold cyan", width=18)
    main_table.add_column("Metric", style="cyan", width=20)
    main_table.add_column("Value", style="white", width=12)
    
    # Returns section
    main_table.add_row("📊 Returns", "Total Return", f"{metrics.total_return:.2%}")
    main_table.add_row("", "Annualized Return", f"{metrics.annualized_return:.2%}")
    main_table.add_row("", "Volatility", f"{metrics.volatility:.2%}")
    main_table.add_row("", "Downside Deviation", f"{metrics.downside_deviation:.2%}")
    
    # Risk-Adjusted Returns
    main_table.add_row("⚖️ Risk-Adjusted", "Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    main_table.add_row("", "Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
    main_table.add_row("", "Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
    main_table.add_row("", "Sterling Ratio", f"{metrics.sterling_ratio:.2f}")
    
    # Risk Metrics
    main_table.add_row("⚠️ Risk", "Max Drawdown", f"{metrics.max_drawdown:.2%}")
    main_table.add_row("", "Avg Drawdown", f"{metrics.avg_drawdown:.2%}")
    main_table.add_row("", "VaR (95%)", f"{metrics.var_95:.2%}")
    main_table.add_row("", "CVaR (95%)", f"{metrics.cvar_95:.2%}")
    
    # Distribution
    main_table.add_row("📊 Distribution", "Skewness", f"{metrics.skewness:.2f}")
    main_table.add_row("", "Kurtosis", f"{metrics.kurtosis:.2f}")
    main_table.add_row("", "Tail Ratio", f"{metrics.tail_ratio:.2f}")
    main_table.add_row("", "Gain-Pain Ratio", f"{metrics.gain_pain_ratio:.2f}")
    
    # Benchmark Comparison
    if metrics.alpha is not None:
        main_table.add_row("🏁 vs Benchmark", "Alpha", f"{metrics.alpha:.2%}")
        main_table.add_row("", "Beta", f"{metrics.beta:.2f}")
        main_table.add_row("", "Information Ratio", f"{metrics.information_ratio:.2f}")
        main_table.add_row("", "Tracking Error", f"{metrics.tracking_error:.2%}")
    
    console.print(main_table)
    console.print()
    
    # Drawdown analysis
    dd_table = Table(title="📉 Drawdown Analysis", show_header=True)
    dd_table.add_column("Metric", style="cyan")
    dd_table.add_column("Value", style="white")
    dd_table.add_column("Description", style="dim")
    
    dd_table.add_row("Number of Drawdowns", str(drawdown_analysis['num_drawdown_periods']), "Total drawdown periods")
    dd_table.add_row("Drawdown Frequency", f"{drawdown_analysis['drawdown_frequency']:.1f}/year", "Drawdowns per year")
    dd_table.add_row("Max Duration", f"{drawdown_analysis['max_drawdown_duration']} days", "Longest drawdown period")
    dd_table.add_row("Avg Duration", f"{drawdown_analysis.get('avg_drawdown_duration', 0):.0f} days", "Average drawdown length")
    dd_table.add_row("Max Recovery", f"{drawdown_analysis.get('max_recovery_days', 0)} days", "Longest recovery time")
    dd_table.add_row("Current Drawdown", f"{drawdown_analysis['current_drawdown']:.2%}", "Current drawdown level")
    
    console.print(dd_table)
    console.print()
    
    # Rolling metrics (if available)
    if rolling_metrics.get('rolling_sharpe_12m') is not None:
        rolling_table = Table(title="🔄 Rolling Metrics (12-Month)", show_header=True)
        rolling_table.add_column("Metric", style="cyan")
        rolling_table.add_column("Current", style="white")
        rolling_table.add_column("Average", style="white")
        
        rolling_table.add_row(
            "Sharpe Ratio",
            f"{rolling_metrics['rolling_sharpe_12m']:.2f}",
            f"{rolling_metrics.get('avg_rolling_sharpe', 0):.2f}"
        )
        rolling_table.add_row(
            "Volatility",
            f"{rolling_metrics['rolling_volatility_12m']:.2%}",
            "N/A"
        )
        rolling_table.add_row(
            "Max Drawdown",
            f"{rolling_metrics['rolling_max_dd_12m']:.2%}",
            "N/A"
        )
        
        console.print(rolling_table)
    
    return metrics


def demo_monte_carlo_analysis():
    """Demonstrate Monte Carlo simulation for strategy validation."""
    
    console.print("\n🎲 [bold blue]Monte Carlo Strategy Validation[/bold blue]")
    console.print("=" * 50)
    
    # Generate base strategy returns
    np.random.seed(42)
    base_returns = np.random.normal(0.0008, 0.018, 252)  # 1 year of daily returns
    
    console.print("📊 Base Strategy: 1 year of daily returns")
    console.print("🎲 Running 1000 Monte Carlo simulations")
    console.print("📅 Simulating 252 trading days (1 year)")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Running Monte Carlo simulations...", total=1000)
        
        # Run Monte Carlo simulation
        simulation_results = []
        
        for i in range(1000):
            # Bootstrap resampling
            simulated_returns = np.random.choice(base_returns, size=252, replace=True)
            
            # Calculate cumulative return
            cumulative_return = (1 + simulated_returns).prod() - 1
            simulation_results.append(cumulative_return)
            
            if i % 50 == 0:
                progress.update(task, advance=50)
        
        progress.update(task, completed=1000)
    
    # Analyze results
    simulation_results = np.array(simulation_results)
    
    # Calculate statistics
    mean_return = np.mean(simulation_results)
    std_return = np.std(simulation_results)
    median_return = np.median(simulation_results)
    
    # Confidence intervals
    ci_95_lower = np.percentile(simulation_results, 2.5)
    ci_95_upper = np.percentile(simulation_results, 97.5)
    ci_80_lower = np.percentile(simulation_results, 10)
    ci_80_upper = np.percentile(simulation_results, 90)
    
    # Risk metrics
    var_5 = np.percentile(simulation_results, 5)
    var_1 = np.percentile(simulation_results, 1)
    prob_loss = np.sum(simulation_results < 0) / len(simulation_results)
    prob_large_loss = np.sum(simulation_results < -0.2) / len(simulation_results)
    prob_large_gain = np.sum(simulation_results > 0.3) / len(simulation_results)
    
    console.print("✅ [bold green]Monte Carlo simulation completed![/bold green]\n")
    
    # Results table
    mc_table = Table(title="🎲 Monte Carlo Results (1-Year Simulation)", show_header=True)
    mc_table.add_column("Metric", style="cyan", width=20)
    mc_table.add_column("Value", style="white", width=15)
    mc_table.add_column("Interpretation", style="dim", width=30)
    
    mc_table.add_row("Mean Return", f"{mean_return:.2%}", "Expected annual return")
    mc_table.add_row("Median Return", f"{median_return:.2%}", "50th percentile outcome")
    mc_table.add_row("Std Deviation", f"{std_return:.2%}", "Return volatility")
    mc_table.add_row("95% CI Lower", f"{ci_95_lower:.2%}", "2.5th percentile (worst 2.5%)")
    mc_table.add_row("95% CI Upper", f"{ci_95_upper:.2%}", "97.5th percentile (best 2.5%)")
    mc_table.add_row("80% CI Range", f"{ci_80_lower:.2%} to {ci_80_upper:.2%}", "Middle 80% of outcomes")
    
    console.print(mc_table)
    console.print()
    
    # Risk analysis table
    risk_table = Table(title="⚠️ Risk Probability Analysis", show_header=True)
    risk_table.add_column("Scenario", style="cyan", width=20)
    risk_table.add_column("Probability", style="white", width=15)
    risk_table.add_column("Description", style="dim", width=30)
    
    risk_table.add_row("Any Loss", f"{prob_loss:.1%}", "Probability of negative return")
    risk_table.add_row("Large Loss (>20%)", f"{prob_large_loss:.1%}", "Probability of >20% loss")
    risk_table.add_row("Large Gain (>30%)", f"{prob_large_gain:.1%}", "Probability of >30% gain")
    risk_table.add_row("VaR (95%)", f"{var_5:.2%}", "Loss exceeded 5% of time")
    risk_table.add_row("VaR (99%)", f"{var_1:.2%}", "Loss exceeded 1% of time")
    
    console.print(risk_table)
    console.print()
    
    # Performance distribution insights
    insights_panel = Panel(
        f"[bold cyan]🔍 Monte Carlo Insights[/bold cyan]\n\n"
        f"• [green]Best Case (95th percentile):[/green] {np.percentile(simulation_results, 95):.1%} return\n"
        f"• [yellow]Typical Range (25th-75th):[/yellow] {np.percentile(simulation_results, 25):.1%} to {np.percentile(simulation_results, 75):.1%}\n"
        f"• [red]Worst Case (5th percentile):[/red] {np.percentile(simulation_results, 5):.1%} return\n"
        f"• [blue]Probability of beating 10% return:[/blue] {np.sum(simulation_results > 0.10) / len(simulation_results):.1%}\n"
        f"• [magenta]Return required for top 10%:[/magenta] {np.percentile(simulation_results, 90):.1%}",
        title="📊 Distribution Insights",
        border_style="blue"
    )
    
    console.print(insights_panel)
    
    return simulation_results


def demo_strategy_comparison():
    """Demonstrate comparison between different strategies."""
    
    console.print("\n⚖️  [bold blue]Strategy Comparison Demo[/bold blue]")
    console.print("=" * 50)
    
    # Generate returns for different strategies
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    # Strategy 1: Covered Calls (lower vol, moderate returns)
    covered_call_returns = np.random.normal(0.0006, 0.012, len(dates))  # 15% return, 19% vol
    
    # Strategy 2: Long Calls (higher vol, higher potential returns)
    long_call_returns = np.random.normal(0.0008, 0.025, len(dates))  # 20% return, 40% vol
    
    # Strategy 3: Iron Condor (very low vol, consistent returns)
    iron_condor_returns = np.random.normal(0.0004, 0.008, len(dates))  # 10% return, 13% vol
    
    # SPY Benchmark
    spy_returns = np.random.normal(0.0005, 0.015, len(dates))  # 12% return, 24% vol
    
    strategies = {
        "Covered Call": pd.Series(covered_call_returns, index=dates),
        "Long Call": pd.Series(long_call_returns, index=dates),
        "Iron Condor": pd.Series(iron_condor_returns, index=dates),
        "SPY Benchmark": pd.Series(spy_returns, index=dates)
    }
    
    console.print("📊 Comparing 3 options strategies vs SPY benchmark")
    console.print("📅 Analysis period: 1 year of daily returns")
    console.print()
    
    # Analyze each strategy
    strategy_metrics = {}
    
    with console.status("[bold green]Analyzing strategies..."):
        for name, returns in strategies.items():
            if name != "SPY Benchmark":
                analyzer = PerformanceAnalyzer(returns, strategies["SPY Benchmark"])
            else:
                analyzer = PerformanceAnalyzer(returns)
            
            strategy_metrics[name] = analyzer.generate_comprehensive_metrics()
    
    console.print("✅ [bold green]Strategy analysis completed![/bold green]\n")
    
    # Comparison table
    comparison_table = Table(title="⚖️ Strategy Performance Comparison", show_header=True)
    comparison_table.add_column("Metric", style="cyan", width=18)
    
    for strategy_name in strategies.keys():
        comparison_table.add_column(strategy_name, style="white", width=12)
    
    # Key metrics to compare
    metrics_to_compare = [
        ("Total Return", "total_return", "{:.1%}"),
        ("Annual Return", "annualized_return", "{:.1%}"),
        ("Volatility", "volatility", "{:.1%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.1%}"),
        ("VaR (95%)", "var_95", "{:.1%}"),
        ("Skewness", "skewness", "{:.2f}"),
        ("Kurtosis", "kurtosis", "{:.2f}"),
    ]
    
    for metric_name, metric_attr, format_str in metrics_to_compare:
        row = [metric_name]
        for strategy_name in strategies.keys():
            metrics = strategy_metrics[strategy_name]
            value = getattr(metrics, metric_attr)
            row.append(format_str.format(value))
        comparison_table.add_row(*row)
    
    console.print(comparison_table)
    console.print()
    
    # Risk-adjusted ranking
    ranking_table = Table(title="🏆 Strategy Rankings", show_header=True)
    ranking_table.add_column("Rank", style="cyan")
    ranking_table.add_column("Strategy", style="white")
    ranking_table.add_column("Sharpe Ratio", style="green")
    ranking_table.add_column("Max Drawdown", style="red")
    ranking_table.add_column("Overall Score", style="bright_blue")
    
    # Calculate overall scores (normalized)
    strategy_scores = {}
    for name, metrics in strategy_metrics.items():
        # Simple scoring: 50% Sharpe, 30% Return, 20% Drawdown (inverted)
        sharpe_score = max(0, min(100, metrics.sharpe_ratio * 33.33))  # Scale to 0-100
        return_score = max(0, min(100, metrics.annualized_return * 500))  # Scale to 0-100
        dd_score = max(0, min(100, (0.3 + metrics.max_drawdown) * 333.33))  # Inverted
        
        overall_score = (sharpe_score * 0.5 + return_score * 0.3 + dd_score * 0.2)
        strategy_scores[name] = overall_score
    
    # Sort by overall score
    ranked_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (strategy_name, score) in enumerate(ranked_strategies, 1):
        metrics = strategy_metrics[strategy_name]
        ranking_table.add_row(
            str(i),
            strategy_name,
            f"{metrics.sharpe_ratio:.2f}",
            f"{metrics.max_drawdown:.1%}",
            f"{score:.1f}"
        )
    
    console.print(ranking_table)
    console.print()
    
    # Winner analysis
    winner = ranked_strategies[0][0]
    winner_metrics = strategy_metrics[winner]
    
    winner_panel = Panel(
        f"[bold green]🏆 Best Performing Strategy: {winner}[/bold green]\n\n"
        f"• [cyan]Annual Return:[/cyan] {winner_metrics.annualized_return:.1%}\n"
        f"• [cyan]Sharpe Ratio:[/cyan] {winner_metrics.sharpe_ratio:.2f}\n"
        f"• [cyan]Max Drawdown:[/cyan] {winner_metrics.max_drawdown:.1%}\n"
        f"• [cyan]Volatility:[/cyan] {winner_metrics.volatility:.1%}\n\n"
        f"[dim]This strategy shows the best risk-adjusted performance\n"
        f"considering return, volatility, and drawdown metrics.[/dim]",
        title="🎯 Winner Analysis",
        border_style="green"
    )
    
    console.print(winner_panel)
    
    return strategy_metrics


def main():
    """Main demo function."""
    
    console.print("\n📈 [bold blue]RTradez Trading Benchmarks Demo[/bold blue]")
    console.print("=" * 60)
    
    console.print("\n[bold]This demo showcases comprehensive trading strategy analysis:[/bold]")
    console.print("  1. Strategy Backtesting")
    console.print("  2. Performance Analysis")
    console.print("  3. Monte Carlo Simulation")
    console.print("  4. Strategy Comparison")
    console.print()
    
    try:
        # 1. Covered Call Backtest
        backtest_results, benchmark_comparison = demo_covered_call_backtest()
        
        # 2. Performance Analysis
        performance_metrics = demo_performance_analysis()
        
        # 3. Monte Carlo Analysis
        monte_carlo_results = demo_monte_carlo_analysis()
        
        # 4. Strategy Comparison
        comparison_results = demo_strategy_comparison()
        
        # Final summary
        console.print("\n🎉 [bold green]Trading Benchmarks Demo Completed![/bold green]")
        console.print("=" * 50)
        
        summary_panel = Panel(
            "[bold cyan]🔍 Key Takeaways[/bold cyan]\n\n"
            "• [green]Comprehensive Performance Metrics:[/green] Multiple risk-adjusted measures\n"
            "• [yellow]Monte Carlo Validation:[/yellow] Probabilistic outcome analysis\n"
            "• [blue]Strategy Comparison:[/blue] Objective ranking and selection\n"
            "• [magenta]Risk Analysis:[/magenta] Detailed drawdown and tail risk assessment\n"
            "• [red]Benchmark Comparison:[/red] Alpha and beta analysis vs market\n\n"
            "[dim]RTradez provides institutional-grade backtesting and performance\n"
            "analysis tools for systematic options trading strategies.[/dim]",
            title="📊 Demo Summary",
            border_style="bright_blue"
        )
        
        console.print(summary_panel)
        console.print()
        
        console.print("🚀 [bold]Next Steps:[/bold]")
        console.print("  • Use `rtradez backtest` CLI for your own strategies")
        console.print("  • Analyze real trading data with `rtradez backtest analyze`")
        console.print("  • Compare strategies with `rtradez backtest compare`")
        console.print("  • Run Monte Carlo validation before live trading")
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Demo failed:[/bold red] {str(e)}")
        console.print("\n[dim]This is a demonstration with synthetic data.[/dim]")


if __name__ == "__main__":
    main()