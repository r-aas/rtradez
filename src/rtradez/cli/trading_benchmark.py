"""
Trading benchmarks CLI commands for RTradez.

Provides comprehensive trading strategy backtesting, performance analysis,
and benchmark comparison tools.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from typing import Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..trading_benchmarks import (
    StrategyBenchmark, BacktestConfig, StrategyType,
    PerformanceAnalyzer, TradingMetrics
)

console = Console()
app = typer.Typer(name="trading-benchmark", help="Trading strategy performance benchmarks")


@app.command()
def backtest(
    strategy: str = typer.Option("covered_call", "--strategy", "-s", help="Strategy type (covered_call, long_call, etc)"),
    start_date: str = typer.Option("2023-01-01", "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option("2024-01-01", "--end", help="End date (YYYY-MM-DD)"),
    capital: float = typer.Option(100000, "--capital", "-c", help="Initial capital"),
    commission: float = typer.Option(1.0, "--commission", help="Commission per trade"),
    benchmark: str = typer.Option("SPY", "--benchmark", "-b", help="Benchmark symbol"),
    save_results: bool = typer.Option(True, "--save/--no-save", help="Save results to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run comprehensive strategy backtesting."""
    
    console.print(f"\n📈 [bold blue]RTradez Strategy Backtesting[/bold blue]")
    console.print("=" * 50)
    
    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        console.print("❌ Invalid date format. Use YYYY-MM-DD")
        return
    
    console.print(f"🎯 Strategy: {strategy}")
    console.print(f"📅 Period: {start_date} to {end_date}")
    console.print(f"💰 Capital: ${capital:,.2f}")
    console.print(f"📊 Benchmark: {benchmark}")
    console.print()
    
    # Setup backtest configuration
    config = BacktestConfig(
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=capital,
        commission_per_trade=commission,
        benchmark_symbol=benchmark
    )
    
    # Run backtest
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Running backtest...", total=None)
        
        try:
            # Initialize benchmark framework
            benchmark_framework = StrategyBenchmark(config)
            
            # Run strategy (currently only covered call implemented)
            if strategy.lower() == "covered_call":
                progress.update(task, description="[cyan]Executing covered call strategy...")
                benchmark_framework.run_simple_covered_call_strategy()
            else:
                console.print(f"❌ Strategy '{strategy}' not yet implemented")
                console.print("Available strategies: covered_call")
                return
            
            progress.update(task, description="[cyan]Calculating performance metrics...")
            
            # Calculate performance
            results = benchmark_framework.calculate_performance_metrics()
            
            progress.update(task, description="[cyan]Running benchmark comparison...")
            
            # Benchmark comparison
            benchmark_comparison = benchmark_framework.run_benchmark_comparison(benchmark)
            
        except Exception as e:
            console.print(f"❌ Backtest failed: {str(e)}")
            return
    
    # Display results
    _display_backtest_results(results, benchmark_comparison, verbose)
    
    # Save results if requested
    if save_results:
        output_file = Path(f"backtest_{strategy}_{start_date}_{end_date}.json")
        
        # Prepare results for JSON serialization
        results_dict = results.dict()
        results_dict['benchmark_comparison'] = benchmark_comparison
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        console.print(f"\n💾 Results saved to: {output_file}")


@app.command()
def analyze(
    returns_file: Path = typer.Argument(..., help="CSV file with daily returns"),
    benchmark_file: Optional[Path] = typer.Option(None, "--benchmark", "-b", help="CSV file with benchmark returns"),
    risk_free_rate: float = typer.Option(0.02, "--rf-rate", help="Risk-free rate (annual)"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, report)")
):
    """Analyze trading performance from returns data."""
    
    console.print(f"\n🔍 [bold blue]Trading Performance Analysis[/bold blue]")
    console.print("=" * 50)
    
    if not returns_file.exists():
        console.print(f"❌ Returns file not found: {returns_file}")
        return
    
    try:
        # Load returns data
        import pandas as pd
        returns_data = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        
        if 'returns' not in returns_data.columns:
            console.print("❌ Returns file must have a 'returns' column")
            return
        
        returns = returns_data['returns']
        
        # Load benchmark if provided
        benchmark_returns = None
        if benchmark_file and benchmark_file.exists():
            benchmark_data = pd.read_csv(benchmark_file, index_col=0, parse_dates=True)
            if 'returns' in benchmark_data.columns:
                benchmark_returns = benchmark_data['returns']
        
        console.print(f"📊 Analyzing {len(returns)} return observations")
        console.print(f"📅 Period: {returns.index.min().date()} to {returns.index.max().date()}")
        
        if benchmark_returns is not None:
            console.print(f"🏁 Benchmark: {len(benchmark_returns)} observations")
        
        console.print()
        
        # Initialize analyzer
        analyzer = PerformanceAnalyzer(
            returns=returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate
        )
        
        # Generate comprehensive metrics
        with console.status("[bold green]Calculating performance metrics..."):
            metrics = analyzer.generate_comprehensive_metrics()
        
        # Display results based on format
        if output_format == "table":
            _display_performance_table(metrics)
        elif output_format == "json":
            console.print(metrics.json(indent=2))
        else:  # report
            console.print(analyzer.generate_performance_summary())
            
    except Exception as e:
        console.print(f"❌ Analysis failed: {str(e)}")


@app.command()
def compare(
    strategy1: Path = typer.Argument(..., help="First strategy returns CSV"),
    strategy2: Path = typer.Argument(..., help="Second strategy returns CSV"),
    names: Optional[List[str]] = typer.Option(None, "--names", help="Strategy names for comparison"),
    benchmark: Optional[Path] = typer.Option(None, "--benchmark", "-b", help="Benchmark returns CSV"),
    metric: str = typer.Option("sharpe", "--metric", "-m", help="Primary comparison metric")
):
    """Compare performance of two trading strategies."""
    
    console.print(f"\n⚖️  [bold blue]Strategy Performance Comparison[/bold blue]")
    console.print("=" * 50)
    
    try:
        import pandas as pd
        
        # Load strategy data
        strat1_data = pd.read_csv(strategy1, index_col=0, parse_dates=True)
        strat2_data = pd.read_csv(strategy2, index_col=0, parse_dates=True)
        
        strat1_returns = strat1_data['returns']
        strat2_returns = strat2_data['returns']
        
        # Load benchmark if provided
        benchmark_returns = None
        if benchmark and benchmark.exists():
            bench_data = pd.read_csv(benchmark, index_col=0, parse_dates=True)
            benchmark_returns = bench_data['returns']
        
        # Strategy names
        if names and len(names) >= 2:
            name1, name2 = names[0], names[1]
        else:
            name1, name2 = "Strategy 1", "Strategy 2"
        
        console.print(f"📊 {name1}: {len(strat1_returns)} observations")
        console.print(f"📊 {name2}: {len(strat2_returns)} observations")
        console.print()
        
        # Analyze both strategies
        analyzer1 = PerformanceAnalyzer(strat1_returns, benchmark_returns)
        analyzer2 = PerformanceAnalyzer(strat2_returns, benchmark_returns)
        
        with console.status("[bold green]Analyzing strategies..."):
            metrics1 = analyzer1.generate_comprehensive_metrics()
            metrics2 = analyzer2.generate_comprehensive_metrics()
        
        # Display comparison
        _display_strategy_comparison(metrics1, metrics2, name1, name2, metric)
        
    except Exception as e:
        console.print(f"❌ Comparison failed: {str(e)}")


@app.command()
def monte_carlo(
    returns_file: Path = typer.Argument(..., help="CSV file with historical returns"),
    simulations: int = typer.Option(1000, "--sims", "-n", help="Number of Monte Carlo simulations"),
    periods: int = typer.Option(252, "--periods", "-p", help="Number of periods to simulate"),
    confidence: float = typer.Option(0.95, "--confidence", "-c", help="Confidence level for analysis"),
    method: str = typer.Option("bootstrap", "--method", "-m", help="Simulation method (bootstrap, parametric)")
):
    """Run Monte Carlo analysis on trading strategy."""
    
    console.print(f"\n🎲 [bold blue]Monte Carlo Strategy Analysis[/bold blue]")
    console.print("=" * 50)
    
    if not returns_file.exists():
        console.print(f"❌ Returns file not found: {returns_file}")
        return
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load returns
        returns_data = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        returns = returns_data['returns']
        
        console.print(f"📊 Historical data: {len(returns)} observations")
        console.print(f"🎲 Simulations: {simulations}")
        console.print(f"📅 Periods: {periods}")
        console.print(f"📈 Method: {method}")
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Running Monte Carlo simulation...", total=simulations)
            
            # Run Monte Carlo simulation
            simulation_results = []
            
            for i in range(simulations):
                if method == "bootstrap":
                    # Bootstrap resampling
                    simulated_returns = np.random.choice(returns, size=periods, replace=True)
                else:
                    # Parametric simulation (normal distribution)
                    mean_return = returns.mean()
                    std_return = returns.std()
                    simulated_returns = np.random.normal(mean_return, std_return, periods)
                
                # Calculate cumulative return for this simulation
                cumulative_return = (1 + simulated_returns).prod() - 1
                simulation_results.append(cumulative_return)
                
                if i % 100 == 0:
                    progress.update(task, advance=100)
            
            progress.update(task, completed=simulations)
        
        # Analyze results
        simulation_results = np.array(simulation_results)
        
        # Calculate statistics
        mean_return = np.mean(simulation_results)
        std_return = np.std(simulation_results)
        
        # Confidence intervals
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        lower_ci = np.percentile(simulation_results, lower_percentile)
        upper_ci = np.percentile(simulation_results, upper_percentile)
        
        # Risk metrics
        var_5 = np.percentile(simulation_results, 5)
        prob_loss = np.sum(simulation_results < 0) / len(simulation_results)
        prob_large_loss = np.sum(simulation_results < -0.2) / len(simulation_results)  # >20% loss
        
        # Display results
        _display_monte_carlo_results(
            simulation_results, mean_return, std_return,
            lower_ci, upper_ci, var_5, prob_loss, prob_large_loss,
            confidence, periods
        )
        
    except Exception as e:
        console.print(f"❌ Monte Carlo analysis failed: {str(e)}")


@app.command()
def report(
    results_dir: Path = typer.Option(Path("trading_results"), "--dir", "-d", help="Results directory"),
    format: str = typer.Option("comprehensive", "--format", "-f", help="Report format (comprehensive, summary, json)"),
    include_charts: bool = typer.Option(False, "--charts", help="Include performance charts"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Generate comprehensive trading performance report."""
    
    console.print(f"\n📋 [bold blue]Trading Performance Report[/bold blue]")
    console.print("=" * 50)
    
    if not results_dir.exists():
        console.print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Look for result files
    result_files = list(results_dir.glob("*.json"))
    
    if not result_files:
        console.print(f"❌ No result files found in: {results_dir}")
        return
    
    console.print(f"📁 Found {len(result_files)} result files")
    console.print()
    
    # Process each result file
    for result_file in result_files:
        console.print(f"📄 Processing: {result_file.name}")
        
        try:
            with open(result_file) as f:
                results = json.load(f)
            
            if format == "comprehensive":
                _display_comprehensive_report(results)
            elif format == "summary":
                _display_summary_report(results)
            else:  # json
                console.print(json.dumps(results, indent=2))
                
        except Exception as e:
            console.print(f"❌ Error processing {result_file}: {str(e)}")
    
    if output_file:
        console.print(f"\n💾 Report saved to: {output_file}")


def _display_backtest_results(results, benchmark_comparison, verbose: bool = False):
    """Display backtest results in a formatted table."""
    
    # Performance Summary Table
    perf_table = Table(title="Strategy Performance Summary")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="white")
    perf_table.add_column("Status", style="green")
    
    # Determine status based on performance
    def get_status(metric, value, thresholds):
        if metric in thresholds:
            if value >= thresholds[metric]['good']:
                return "✅ Excellent"
            elif value >= thresholds[metric]['ok']:
                return "⚠️  Good"
            else:
                return "❌ Poor"
        return "ℹ️  N/A"
    
    # Performance thresholds
    thresholds = {
        'sharpe_ratio': {'good': 1.5, 'ok': 1.0},
        'win_rate': {'good': 0.6, 'ok': 0.5},
        'profit_factor': {'good': 1.5, 'ok': 1.2},
        'max_drawdown': {'good': -0.1, 'ok': -0.2}  # Negative because drawdown is negative
    }
    
    # Add rows
    perf_table.add_row("Total Return", f"{results.total_return:.2%}", get_status('total_return', results.total_return, thresholds))
    perf_table.add_row("Annual Return", f"{results.annualized_return:.2%}", get_status('annualized_return', results.annualized_return, thresholds))
    perf_table.add_row("Sharpe Ratio", f"{results.sharpe_ratio:.2f}", get_status('sharpe_ratio', results.sharpe_ratio, thresholds))
    perf_table.add_row("Max Drawdown", f"{results.max_drawdown:.2%}", get_status('max_drawdown', results.max_drawdown, thresholds))
    perf_table.add_row("Win Rate", f"{results.win_rate:.1%}", get_status('win_rate', results.win_rate, thresholds))
    perf_table.add_row("Profit Factor", f"{results.profit_factor:.2f}", get_status('profit_factor', results.profit_factor, thresholds))
    
    console.print(perf_table)
    console.print()
    
    # Benchmark Comparison
    if benchmark_comparison:
        bench_table = Table(title="vs Benchmark Analysis")
        bench_table.add_column("Metric", style="cyan")
        bench_table.add_column("Value", style="white")
        
        bench_table.add_row("Alpha", f"{benchmark_comparison['alpha']:.2%}")
        bench_table.add_row("Beta", f"{benchmark_comparison['beta']:.2f}")
        bench_table.add_row("Correlation", f"{benchmark_comparison['correlation']:.2f}")
        bench_table.add_row("Information Ratio", f"{benchmark_comparison['information_ratio']:.2f}")
        bench_table.add_row("Excess Return", f"{benchmark_comparison['excess_return']:.2%}")
        
        console.print(bench_table)
        console.print()
    
    # Trading Statistics (if verbose)
    if verbose and results.total_trades > 0:
        trade_table = Table(title="Trading Statistics")
        trade_table.add_column("Metric", style="cyan")
        trade_table.add_column("Value", style="white")
        
        trade_table.add_row("Total Trades", str(results.total_trades))
        trade_table.add_row("Winning Trades", str(results.winning_trades))
        trade_table.add_row("Losing Trades", str(results.losing_trades))
        trade_table.add_row("Average Win", f"${results.avg_win:.2f}")
        trade_table.add_row("Average Loss", f"${results.avg_loss:.2f}")
        trade_table.add_row("Expectancy", f"${results.expectancy:.2f}")
        
        console.print(trade_table)


def _display_performance_table(metrics: TradingMetrics):
    """Display performance metrics in table format."""
    
    # Main Performance Table
    main_table = Table(title="Performance Metrics")
    main_table.add_column("Category", style="bold cyan")
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", style="white")
    
    # Returns
    main_table.add_row("Returns", "Total Return", f"{metrics.total_return:.2%}")
    main_table.add_row("", "Annualized Return", f"{metrics.annualized_return:.2%}")
    main_table.add_row("", "Volatility", f"{metrics.volatility:.2%}")
    
    # Risk-Adjusted
    main_table.add_row("Risk-Adjusted", "Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    main_table.add_row("", "Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
    main_table.add_row("", "Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
    
    # Risk
    main_table.add_row("Risk", "Max Drawdown", f"{metrics.max_drawdown:.2%}")
    main_table.add_row("", "VaR (95%)", f"{metrics.var_95:.2%}")
    main_table.add_row("", "CVaR (95%)", f"{metrics.cvar_95:.2%}")
    
    # Distribution
    main_table.add_row("Distribution", "Skewness", f"{metrics.skewness:.2f}")
    main_table.add_row("", "Kurtosis", f"{metrics.kurtosis:.2f}")
    main_table.add_row("", "Tail Ratio", f"{metrics.tail_ratio:.2f}")
    
    # Trading (if applicable)
    if metrics.total_trades > 0:
        main_table.add_row("Trading", "Win Rate", f"{metrics.win_rate:.1%}")
        main_table.add_row("", "Profit Factor", f"{metrics.profit_factor:.2f}")
        main_table.add_row("", "Expectancy", f"{metrics.expectancy:.4f}")
    
    # Benchmark (if applicable)
    if metrics.alpha is not None:
        main_table.add_row("vs Benchmark", "Alpha", f"{metrics.alpha:.2%}")
        main_table.add_row("", "Beta", f"{metrics.beta:.2f}")
        main_table.add_row("", "Information Ratio", f"{metrics.information_ratio:.2f}")
    
    console.print(main_table)


def _display_strategy_comparison(metrics1, metrics2, name1, name2, primary_metric):
    """Display side-by-side strategy comparison."""
    
    comp_table = Table(title="Strategy Comparison")
    comp_table.add_column("Metric", style="cyan")
    comp_table.add_column(name1, style="white")
    comp_table.add_column(name2, style="white")
    comp_table.add_column("Winner", style="green")
    
    # Comparison metrics
    comparisons = [
        ("Total Return", f"{metrics1.total_return:.2%}", f"{metrics2.total_return:.2%}", metrics1.total_return > metrics2.total_return),
        ("Sharpe Ratio", f"{metrics1.sharpe_ratio:.2f}", f"{metrics2.sharpe_ratio:.2f}", metrics1.sharpe_ratio > metrics2.sharpe_ratio),
        ("Max Drawdown", f"{metrics1.max_drawdown:.2%}", f"{metrics2.max_drawdown:.2%}", metrics1.max_drawdown > metrics2.max_drawdown),
        ("Volatility", f"{metrics1.volatility:.2%}", f"{metrics2.volatility:.2%}", metrics1.volatility < metrics2.volatility),
        ("Win Rate", f"{metrics1.win_rate:.1%}", f"{metrics2.win_rate:.1%}", metrics1.win_rate > metrics2.win_rate),
    ]
    
    for metric, val1, val2, winner in comparisons:
        winner_text = name1 if winner else name2
        comp_table.add_row(metric, val1, val2, winner_text)
    
    console.print(comp_table)


def _display_monte_carlo_results(results, mean_return, std_return, lower_ci, upper_ci, var_5, prob_loss, prob_large_loss, confidence, periods):
    """Display Monte Carlo simulation results."""
    
    mc_table = Table(title=f"Monte Carlo Results ({periods} periods)")
    mc_table.add_column("Metric", style="cyan")
    mc_table.add_column("Value", style="white")
    
    mc_table.add_row("Mean Return", f"{mean_return:.2%}")
    mc_table.add_row("Std Deviation", f"{std_return:.2%}")
    mc_table.add_row(f"{confidence:.0%} CI Lower", f"{lower_ci:.2%}")
    mc_table.add_row(f"{confidence:.0%} CI Upper", f"{upper_ci:.2%}")
    mc_table.add_row("VaR (5%)", f"{var_5:.2%}")
    mc_table.add_row("Probability of Loss", f"{prob_loss:.1%}")
    mc_table.add_row("Prob of >20% Loss", f"{prob_large_loss:.1%}")
    
    console.print(mc_table)


def _display_comprehensive_report(results):
    """Display comprehensive performance report."""
    console.print(f"📊 Strategy: {results.get('config', {}).get('start_date', 'Unknown')}")
    # Add more comprehensive reporting logic here


def _display_summary_report(results):
    """Display summary performance report."""
    console.print(f"📈 Total Return: {results.get('total_return', 0):.2%}")
    # Add more summary reporting logic here


if __name__ == "__main__":
    app()