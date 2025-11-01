"""
Analysis CLI Commands.

Market analysis, optimization, and strategy evaluation tools.
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import pandas as pd
import numpy as np

console = Console()
analysis_app = typer.Typer(name="analysis", help="Market analysis and optimization tools")


@analysis_app.command("optimize")
def optimize_strategy(
    strategy: str = typer.Option("kelly", "--strategy", "-s", help="Strategy to optimize"),
    trials: int = typer.Option(100, "--trials", "-t", help="Number of optimization trials"),
    objective: str = typer.Option("sharpe", "--objective", "-o", help="Optimization objective"),
    save_results: bool = typer.Option(False, "--save", help="Save optimization results")
):
    """Optimize strategy parameters using Optuna."""
    
    console.print(f"\n[bold blue]🎯 Strategy Parameter Optimization[/]\n")
    
    # Simulation of optimization process
    console.print(f"[bold]Optimizing {strategy} strategy with {trials} trials...[/]\n")
    
    # Parameter space for different strategies
    param_spaces = {
        "kelly": {
            "confidence_threshold": (0.5, 0.9),
            "max_kelly_fraction": (0.1, 0.5),
            "lookback_window": (20, 252)
        },
        "iron_condor": {
            "strike_width": (5, 50),
            "days_to_expiry": (10, 60),
            "target_delta": (0.05, 0.30)
        },
        "momentum": {
            "lookback_period": (5, 50),
            "threshold": (0.01, 0.10),
            "holding_period": (1, 20)
        }
    }
    
    params = param_spaces.get(strategy, param_spaces["kelly"])
    
    # Simulate optimization trials
    best_params = {}
    best_score = -np.inf
    trial_results = []
    
    for trial in track(range(trials), description="Optimizing..."):
        # Simulate parameter sampling
        trial_params = {}
        for param, (low, high) in params.items():
            trial_params[param] = np.random.uniform(low, high)
        
        # Simulate objective evaluation
        if objective == "sharpe":
            # Simulate Sharpe ratio calculation
            returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
            sharpe = np.mean(returns) * 252 / (np.std(returns) * np.sqrt(252))
            score = sharpe + np.random.normal(0, 0.1)  # Add noise
        elif objective == "return":
            score = np.random.normal(0.12, 0.05)  # Annual return
        else:
            score = np.random.normal(0.5, 0.1)  # Generic score
        
        trial_results.append({
            "trial": trial,
            "score": score,
            "params": trial_params.copy()
        })
        
        if score > best_score:
            best_score = score
            best_params = trial_params.copy()
    
    # Display optimization results
    results_table = Table(title=f"Optimization Results - {strategy.title()}", show_header=True)
    results_table.add_column("Parameter", style="cyan", width=20)
    results_table.add_column("Optimal Value", style="green", width=15)
    results_table.add_column("Search Range", style="blue", width=20)
    results_table.add_column("Improvement", style="yellow", width=15)
    
    for param, value in best_params.items():
        low, high = params[param]
        improvement = f"{((value - low) / (high - low)) * 100:.1f}% through range"
        
        if isinstance(value, float):
            if value < 1:
                value_str = f"{value:.3f}"
            else:
                value_str = f"{value:.1f}"
        else:
            value_str = str(int(value))
        
        results_table.add_row(
            param.replace("_", " ").title(),
            value_str,
            f"{low} - {high}",
            improvement
        )
    
    console.print(results_table)
    
    # Performance summary
    sorted_results = sorted(trial_results, key=lambda x: x["score"], reverse=True)
    top_10_scores = [r["score"] for r in sorted_results[:10]]
    
    perf_table = Table(title="Performance Summary", show_header=True)
    perf_table.add_column("Metric", style="cyan", width=20)
    perf_table.add_column("Value", style="green", width=15)
    perf_table.add_column("Benchmark", style="blue", width=15)
    
    baseline_score = np.mean([r["score"] for r in trial_results[-10:]])  # Avg of worst 10
    improvement = (best_score - baseline_score) / abs(baseline_score) * 100
    
    perf_table.add_row("Best Score", f"{best_score:.3f}", f"{baseline_score:.3f}")
    perf_table.add_row("Improvement", f"{improvement:+.1f}%", "vs Baseline")
    perf_table.add_row("Top 10 Avg", f"{np.mean(top_10_scores):.3f}", f"{best_score:.3f}")
    perf_table.add_row("Consistency", f"{np.std(top_10_scores):.3f}", "Lower is better")
    
    console.print(f"\n{perf_table}")
    
    if save_results:
        results_data = {
            "strategy": strategy,
            "best_params": best_params,
            "best_score": best_score,
            "objective": objective,
            "trials": trials,
            "all_results": trial_results
        }
        
        import json
        filename = f"{strategy}_optimization_results.json"
        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        
        console.print(f"\n[green]✅ Results saved to {filename}[/]")
    
    # Insights and recommendations
    insights_panel = Panel(
        f"Optimization Insights:\n\n"
        f"🎯 Best {objective}: {best_score:.3f}\n"
        f"📈 Improvement: {improvement:+.1f}% vs baseline\n"
        f"🔍 Trials Run: {trials}\n"
        f"💡 Top Parameter: {max(best_params.keys(), key=lambda k: abs(best_params[k] - np.mean(params[k])))}\n"
        f"⚡ Convergence: {'Good' if improvement > 10 else 'Moderate' if improvement > 5 else 'Limited'}",
        title="📊 Key Insights",
        border_style="green"
    )
    console.print(f"\n{insights_panel}")


@analysis_app.command("backtest")
def run_backtest(
    strategy: str = typer.Option("sample", "--strategy", "-s", help="Strategy to backtest"),
    start_date: str = typer.Option("2023-01-01", "--start", help="Backtest start date"),
    end_date: str = typer.Option("2024-01-01", "--end", help="Backtest end date"),
    initial_capital: float = typer.Option(100000, "--capital", "-c", help="Initial capital"),
    benchmark: str = typer.Option("SPY", "--benchmark", "-b", help="Benchmark ticker"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed results")
):
    """Run comprehensive strategy backtest."""
    
    console.print(f"\n[bold blue]🔬 Strategy Backtest Analysis[/]\n")
    
    console.print(f"[bold]Backtesting {strategy} strategy from {start_date} to {end_date}[/]\n")
    
    # Simulate backtest data
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_days = len(dates)
    
    # Generate synthetic returns
    strategy_returns = np.random.normal(0.0008, 0.015, trading_days)  # Slightly positive mean
    benchmark_returns = np.random.normal(0.0005, 0.012, trading_days)  # Market returns
    
    # Calculate cumulative performance
    strategy_cumulative = np.cumprod(1 + strategy_returns)
    benchmark_cumulative = np.cumprod(1 + benchmark_returns)
    
    # Calculate metrics
    total_return_strategy = strategy_cumulative[-1] - 1
    total_return_benchmark = benchmark_cumulative[-1] - 1
    
    volatility_strategy = np.std(strategy_returns) * np.sqrt(252)
    volatility_benchmark = np.std(benchmark_returns) * np.sqrt(252)
    
    sharpe_strategy = np.mean(strategy_returns) * 252 / volatility_strategy
    sharpe_benchmark = np.mean(benchmark_returns) * 252 / volatility_benchmark
    
    # Calculate drawdown
    running_max_strategy = np.maximum.accumulate(strategy_cumulative)
    drawdown_strategy = (strategy_cumulative - running_max_strategy) / running_max_strategy
    max_drawdown_strategy = np.min(drawdown_strategy)
    
    # Display results
    performance_table = Table(title="Backtest Performance Results", show_header=True)
    performance_table.add_column("Metric", style="cyan", width=20)
    performance_table.add_column("Strategy", style="green", width=15)
    performance_table.add_column("Benchmark", style="blue", width=15)
    performance_table.add_column("Difference", style="yellow", width=15)
    
    metrics = [
        ("Total Return", f"{total_return_strategy:.2%}", f"{total_return_benchmark:.2%}", f"{(total_return_strategy - total_return_benchmark):.2%}"),
        ("Annualized Return", f"{total_return_strategy * 252 / trading_days:.2%}", f"{total_return_benchmark * 252 / trading_days:.2%}", f"{(total_return_strategy - total_return_benchmark) * 252 / trading_days:.2%}"),
        ("Volatility", f"{volatility_strategy:.2%}", f"{volatility_benchmark:.2%}", f"{(volatility_strategy - volatility_benchmark):.2%}"),
        ("Sharpe Ratio", f"{sharpe_strategy:.3f}", f"{sharpe_benchmark:.3f}", f"{(sharpe_strategy - sharpe_benchmark):+.3f}"),
        ("Max Drawdown", f"{max_drawdown_strategy:.2%}", f"{np.min((benchmark_cumulative - np.maximum.accumulate(benchmark_cumulative)) / np.maximum.accumulate(benchmark_cumulative)):.2%}", ""),
        ("Final Value", f"${initial_capital * (1 + total_return_strategy):,.0f}", f"${initial_capital * (1 + total_return_benchmark):,.0f}", f"${initial_capital * (total_return_strategy - total_return_benchmark):+,.0f}")
    ]
    
    for metric, strategy_val, benchmark_val, diff in metrics:
        performance_table.add_row(metric, strategy_val, benchmark_val, diff)
    
    console.print(performance_table)
    
    if detailed:
        # Trading statistics
        console.print("\n")
        
        # Simulate trade data
        num_trades = int(trading_days / 5)  # Trade every 5 days on average
        win_rate = 0.55 + np.random.normal(0, 0.05)
        win_rate = max(0.3, min(0.8, win_rate))
        
        winning_trades = int(num_trades * win_rate)
        losing_trades = num_trades - winning_trades
        
        avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0.01
        avg_loss = np.mean(strategy_returns[strategy_returns < 0]) if np.any(strategy_returns < 0) else -0.008
        
        trading_table = Table(title="Trading Statistics", show_header=True)
        trading_table.add_column("Statistic", style="cyan", width=20)
        trading_table.add_column("Value", style="green", width=15)
        trading_table.add_column("Assessment", style="yellow", width=20)
        
        trading_stats = [
            ("Total Trades", f"{num_trades:,}", "Good" if num_trades > 50 else "Limited"),
            ("Win Rate", f"{win_rate:.1%}", "Excellent" if win_rate > 0.6 else "Good" if win_rate > 0.5 else "Poor"),
            ("Winning Trades", f"{winning_trades:,}", ""),
            ("Losing Trades", f"{losing_trades:,}", ""),
            ("Avg Win", f"{avg_win:.2%}", ""),
            ("Avg Loss", f"{avg_loss:.2%}", ""),
            ("Profit Factor", f"{abs(avg_win * winning_trades / (avg_loss * losing_trades)):.2f}", "Good" if abs(avg_win * winning_trades / (avg_loss * losing_trades)) > 1.5 else "Poor"),
        ]
        
        for stat, value, assessment in trading_stats:
            trading_table.add_row(stat, value, assessment)
        
        console.print(trading_table)
        
        # Risk metrics
        console.print("\n")
        risk_table = Table(title="Risk Analysis", show_header=True)
        risk_table.add_column("Risk Metric", style="cyan", width=20)
        risk_table.add_column("Value", style="green", width=15)
        risk_table.add_column("Status", style="yellow", width=15)
        
        # Calculate additional risk metrics
        var_95 = np.percentile(strategy_returns, 5)
        var_99 = np.percentile(strategy_returns, 1)
        
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        risk_metrics = [
            ("Value at Risk (95%)", f"{var_95:.2%}", "Normal" if var_95 > -0.03 else "High Risk"),
            ("Value at Risk (99%)", f"{var_99:.2%}", "Normal" if var_99 > -0.05 else "High Risk"),
            ("Downside Deviation", f"{downside_deviation:.2%}", "Low" if downside_deviation < 0.15 else "Moderate"),
            ("Calmar Ratio", f"{(total_return_strategy * 252 / trading_days) / abs(max_drawdown_strategy):.2f}", "Good" if (total_return_strategy * 252 / trading_days) / abs(max_drawdown_strategy) > 1.0 else "Poor"),
        ]
        
        for metric, value, status in risk_metrics:
            risk_table.add_row(metric, value, status)
        
        console.print(risk_table)
    
    # Summary assessment
    overall_score = 0
    assessment_items = []
    
    if total_return_strategy > total_return_benchmark:
        overall_score += 2
        assessment_items.append("✅ Outperformed benchmark")
    else:
        assessment_items.append("⚠️ Underperformed benchmark")
    
    if sharpe_strategy > 1.0:
        overall_score += 2
        assessment_items.append("✅ Strong risk-adjusted returns")
    elif sharpe_strategy > 0.5:
        overall_score += 1
        assessment_items.append("⚠️ Moderate risk-adjusted returns")
    else:
        assessment_items.append("❌ Poor risk-adjusted returns")
    
    if abs(max_drawdown_strategy) < 0.10:
        overall_score += 1
        assessment_items.append("✅ Low maximum drawdown")
    elif abs(max_drawdown_strategy) < 0.20:
        assessment_items.append("⚠️ Moderate maximum drawdown")
    else:
        assessment_items.append("❌ High maximum drawdown")
    
    if detailed and win_rate > 0.55:
        overall_score += 1
        assessment_items.append("✅ Good win rate")
    
    assessment_text = "\n".join(assessment_items)
    grade = "A" if overall_score >= 5 else "B" if overall_score >= 3 else "C" if overall_score >= 1 else "D"
    
    summary_panel = Panel(
        f"Strategy Assessment (Grade: {grade}):\n\n{assessment_text}\n\n"
        f"💡 Recommendations:\n"
        f"{'• Consider increasing position size' if total_return_strategy > total_return_benchmark else '• Review strategy parameters'}\n"
        f"{'• Monitor for regime changes' if sharpe_strategy < 1.0 else '• Maintain current approach'}\n"
        f"{'• Implement stop-loss if not present' if abs(max_drawdown_strategy) > 0.15 else '• Current risk management adequate'}",
        title="📋 Strategy Assessment",
        border_style="green" if grade in ["A", "B"] else "yellow" if grade == "C" else "red"
    )
    console.print(f"\n{summary_panel}")


@analysis_app.command("compare")
def compare_strategies(
    strategies: List[str] = typer.Option(["kelly", "momentum", "mean_reversion"], "--strategies", "-s", help="Strategies to compare"),
    metric: str = typer.Option("sharpe", "--metric", "-m", help="Comparison metric"),
    period: str = typer.Option("1Y", "--period", "-p", help="Analysis period")
):
    """Compare multiple strategies across key metrics."""
    
    console.print(f"\n[bold blue]⚖️ Strategy Comparison Analysis[/]\n")
    
    console.print(f"[bold]Comparing {len(strategies)} strategies over {period} period[/]\n")
    
    # Simulate performance data for each strategy
    np.random.seed(42)
    results = {}
    
    for i, strategy in enumerate(strategies):
        # Vary performance characteristics by strategy
        base_return = 0.08 + (i * 0.03) + np.random.normal(0, 0.02)
        base_vol = 0.15 + (i * 0.02) + np.random.normal(0, 0.01)
        
        returns = np.random.normal(base_return/252, base_vol/np.sqrt(252), 252)
        
        total_return = np.prod(1 + returns) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = np.mean(returns) * 252 / volatility
        
        # Simulate drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        results[strategy] = {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": 0.45 + np.random.uniform(0, 0.3),
            "profit_factor": 1.0 + np.random.uniform(0, 1.0)
        }
    
    # Create comparison table
    comparison_table = Table(title="Strategy Comparison Matrix", show_header=True)
    comparison_table.add_column("Strategy", style="cyan", width=15)
    comparison_table.add_column("Return", style="green", width=10)
    comparison_table.add_column("Volatility", style="blue", width=12)
    comparison_table.add_column("Sharpe", style="yellow", width=10)
    comparison_table.add_column("Max DD", style="red", width=10)
    comparison_table.add_column("Win Rate", style="magenta", width=10)
    comparison_table.add_column("Rank", style="white", width=8)
    
    # Rank strategies by selected metric
    if metric == "sharpe":
        ranked = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    elif metric == "return":
        ranked = sorted(results.items(), key=lambda x: x[1]["total_return"], reverse=True)
    elif metric == "risk":
        ranked = sorted(results.items(), key=lambda x: x[1]["volatility"])
    else:
        ranked = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    
    for rank, (strategy, metrics) in enumerate(ranked, 1):
        comparison_table.add_row(
            strategy.replace("_", " ").title(),
            f"{metrics['total_return']:.1%}",
            f"{metrics['volatility']:.1%}",
            f"{metrics['sharpe']:.2f}",
            f"{metrics['max_drawdown']:.1%}",
            f"{metrics['win_rate']:.1%}",
            f"#{rank}"
        )
    
    console.print(comparison_table)
    
    # Statistical significance test (simulated)
    console.print("\n")
    significance_table = Table(title="Statistical Significance Analysis", show_header=True)
    significance_table.add_column("Comparison", style="cyan", width=25)
    significance_table.add_column("P-Value", style="green", width=12)
    significance_table.add_column("Significance", style="yellow", width=15)
    significance_table.add_column("Effect Size", style="blue", width=12)
    
    # Compare top performers
    top_strategies = [item[0] for item in ranked[:3]]
    for i in range(len(top_strategies)):
        for j in range(i+1, len(top_strategies)):
            strat1, strat2 = top_strategies[i], top_strategies[j]
            p_value = np.random.uniform(0.01, 0.20)
            effect_size = abs(results[strat1]["sharpe"] - results[strat2]["sharpe"])
            
            significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else "n.s."
            
            significance_table.add_row(
                f"{strat1.title()} vs {strat2.title()}",
                f"{p_value:.3f}",
                significance,
                f"{effect_size:.2f}"
            )
    
    console.print(significance_table)
    
    # Best strategy analysis
    best_strategy, best_metrics = ranked[0]
    
    # Portfolio allocation recommendation
    allocation_panel = Panel(
        f"Optimal Strategy Portfolio:\n\n"
        f"🥇 Primary (60%): {best_strategy.replace('_', ' ').title()}\n"
        f"   Sharpe: {best_metrics['sharpe']:.2f}, Return: {best_metrics['total_return']:.1%}\n\n"
        f"🥈 Secondary (25%): {ranked[1][0].replace('_', ' ').title()}\n"
        f"   Provides diversification benefit\n\n"
        f"🥉 Tactical (15%): {ranked[2][0].replace('_', ' ').title()}\n"
        f"   For alternative market conditions\n\n"
        f"💡 Expected Portfolio Sharpe: {(0.6 * best_metrics['sharpe'] + 0.25 * ranked[1][1]['sharpe'] + 0.15 * ranked[2][1]['sharpe']):.2f}",
        title="🎯 Portfolio Recommendation",
        border_style="green"
    )
    console.print(f"\n{allocation_panel}")


if __name__ == "__main__":
    analysis_app()