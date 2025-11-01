"""
Portfolio Management CLI Commands.

Multi-strategy coordination, allocation management, and performance analysis.
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..portfolio.portfolio_manager import (
    PortfolioManager, PortfolioConfig, StrategyAllocation, StrategyStatus
)

console = Console()
portfolio_app = typer.Typer(name="portfolio", help="Portfolio management and coordination")


@portfolio_app.command("create")
def create_portfolio(
    capital: float = typer.Option(1000000, "--capital", "-c", help="Total portfolio capital"),
    max_strategies: int = typer.Option(10, "--max-strategies", help="Maximum number of strategies"),
    rebalance_freq: str = typer.Option("weekly", "--rebalance-freq", help="Rebalancing frequency"),
    rebalance_threshold: float = typer.Option(0.05, "--threshold", help="Rebalancing threshold"),
    cash_reserve: float = typer.Option(0.05, "--cash-reserve", help="Minimum cash reserve"),
    name: str = typer.Option("Portfolio_1", "--name", "-n", help="Portfolio name"),
    save_config: bool = typer.Option(False, "--save", help="Save configuration")
):
    """Create a new portfolio configuration."""
    
    console.print(f"\n[bold blue]🏦 Creating Portfolio: {name}[/]\n")
    
    # Create portfolio configuration
    config = PortfolioConfig(
        total_capital=capital,
        max_strategies=max_strategies,
        rebalance_frequency=rebalance_freq,
        rebalance_threshold=rebalance_threshold,
        cash_reserve_minimum=cash_reserve,
        enable_auto_rebalancing=True,
        enable_risk_coordination=True
    )
    
    # Initialize portfolio manager
    with console.status("[bold green]Initializing portfolio manager..."):
        portfolio = PortfolioManager(config)
    
    # Display configuration
    table = Table(title=f"Portfolio Configuration - {name}", show_header=True)
    table.add_column("Parameter", style="cyan", width=25)
    table.add_column("Value", style="green", width=20)
    table.add_column("Description", style="dim", width=35)
    
    table.add_row("Total Capital", f"${capital:,.0f}", "Available investment capital")
    table.add_row("Max Strategies", str(max_strategies), "Maximum concurrent strategies")
    table.add_row("Rebalance Frequency", rebalance_freq, "Automatic rebalancing schedule")
    table.add_row("Rebalance Threshold", f"{rebalance_threshold:.1%}", "Deviation trigger for rebalancing")
    table.add_row("Cash Reserve", f"{cash_reserve:.1%}", "Minimum cash percentage")
    table.add_row("Auto Rebalancing", "✅ Enabled", "Automatic portfolio rebalancing")
    table.add_row("Risk Coordination", "✅ Enabled", "Integrated risk management")
    
    console.print(table)
    
    if save_config:
        config_data = {
            "name": name,
            "config": config.dict(),
            "created": datetime.now().isoformat(),
            "status": "initialized"
        }
        
        import json
        filename = f"{name.lower().replace(' ', '_')}_config.json"
        with open(filename, "w") as f:
            json.dump(config_data, f, indent=2)
        
        console.print(f"\n[green]✅ Portfolio configuration saved to {filename}[/]")
    
    success_panel = Panel(
        f"Portfolio '{name}' successfully created!\n\n"
        f"Next steps:\n"
        f"• Add strategies with [bold]rtradez portfolio add-strategy[/bold]\n"
        f"• Monitor performance with [bold]rtradez portfolio status[/bold]\n"
        f"• Manage allocations with [bold]rtradez portfolio rebalance[/bold]",
        title="✅ Success",
        border_style="green"
    )
    console.print(f"\n{success_panel}")


@portfolio_app.command("add-strategy")
def add_strategy(
    name: str = typer.Option(..., "--name", "-n", help="Strategy name"),
    allocation: float = typer.Option(..., "--allocation", "-a", help="Target allocation (0-1)"),
    min_allocation: float = typer.Option(0.0, "--min", help="Minimum allocation"),
    max_allocation: float = typer.Option(0.5, "--max", help="Maximum allocation"),
    expected_return: float = typer.Option(0.10, "--return", "-r", help="Expected annual return"),
    volatility: float = typer.Option(0.15, "--volatility", "-v", help="Expected volatility"),
    portfolio_file: Optional[str] = typer.Option(None, "--portfolio", "-p", help="Portfolio config file")
):
    """Add a strategy to the portfolio."""
    
    console.print(f"\n[bold blue]➕ Adding Strategy: {name}[/]\n")
    
    # Load portfolio configuration (simplified for demo)
    if portfolio_file:
        console.print(f"[dim]Loading portfolio from {portfolio_file}[/]")
    
    # Create mock portfolio for demonstration
    config = PortfolioConfig(total_capital=1000000)
    portfolio = PortfolioManager(config)
    
    # Create mock strategy
    class MockStrategy:
        def __init__(self, name, expected_return, volatility):
            self.name = name
            self.expected_return = expected_return
            self.volatility = volatility
            self.is_fitted = False
        
        def fit(self, X, y):
            self.is_fitted = True
            return self
        
        def predict(self, X):
            return np.random.normal(self.expected_return/252, self.volatility/np.sqrt(252), len(X))
    
    strategy = MockStrategy(name, expected_return, volatility)
    
    # Add strategy to portfolio
    with console.status("[bold green]Adding strategy to portfolio..."):
        success = portfolio.add_strategy(
            strategy_name=name,
            strategy_instance=strategy,
            target_allocation=allocation,
            min_allocation=min_allocation,
            max_allocation=max_allocation
        )
    
    if success:
        # Display strategy details
        table = Table(title=f"Strategy Added - {name}", show_header=True)
        table.add_column("Parameter", style="cyan", width=20)
        table.add_column("Value", style="green", width=20)
        
        table.add_row("Strategy Name", name)
        table.add_row("Target Allocation", f"{allocation:.1%}")
        table.add_row("Min Allocation", f"{min_allocation:.1%}")
        table.add_row("Max Allocation", f"{max_allocation:.1%}")
        table.add_row("Expected Return", f"{expected_return:.1%}")
        table.add_row("Volatility", f"{volatility:.1%}")
        table.add_row("Status", "Active")
        
        console.print(table)
        
        # Show portfolio impact
        portfolio_allocation = sum(s.target_allocation for s in portfolio.strategies.values())
        remaining = 1.0 - portfolio_allocation
        
        impact_panel = Panel(
            f"Portfolio Impact:\n\n"
            f"• Total Allocated: {portfolio_allocation:.1%}\n"
            f"• Remaining Capacity: {remaining:.1%}\n"
            f"• Strategy Count: {len(portfolio.strategies)}\n"
            f"• Capital Allocation: ${allocation * config.total_capital:,.0f}",
            title="📊 Portfolio Impact",
            border_style="blue"
        )
        console.print(f"\n{impact_panel}")
        
    else:
        console.print("[red]❌ Failed to add strategy. Check allocation limits.[/]")


@portfolio_app.command("status")
def portfolio_status(
    portfolio_file: Optional[str] = typer.Option(None, "--portfolio", "-p", help="Portfolio config file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed metrics")
):
    """Show current portfolio status and performance."""
    
    console.print(f"\n[bold blue]📊 Portfolio Status Dashboard[/]\n")
    
    # Create demo portfolio with some strategies
    config = PortfolioConfig(total_capital=2000000)
    portfolio = PortfolioManager(config)
    
    # Add demo strategies
    demo_strategies = [
        {"name": "IronCondor_SPY", "allocation": 0.30, "return": 0.08, "vol": 0.12},
        {"name": "Strangle_QQQ", "allocation": 0.25, "return": 0.15, "vol": 0.25},
        {"name": "CalendarSpread_IWM", "allocation": 0.20, "return": 0.06, "vol": 0.08},
        {"name": "CoveredCall_MSFT", "allocation": 0.15, "return": 0.04, "vol": 0.06},
    ]
    
    class MockStrategy:
        def __init__(self, name): self.name = name
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading portfolio data...", total=len(demo_strategies))
        
        for strategy in demo_strategies:
            portfolio.add_strategy(
                strategy["name"], 
                MockStrategy(strategy["name"]), 
                strategy["allocation"]
            )
            # Simulate current allocation close to target
            portfolio.strategies[strategy["name"]].current_allocation = strategy["allocation"] * (1 + np.random.uniform(-0.1, 0.1))
            progress.advance(task)
    
    # Calculate metrics
    metrics = portfolio.calculate_portfolio_metrics()
    
    # Portfolio overview table
    overview_table = Table(title="Portfolio Overview", show_header=True)
    overview_table.add_column("Metric", style="cyan", width=25)
    overview_table.add_column("Value", style="green", width=20)
    overview_table.add_column("Status", style="magenta", width=15)
    
    overview_table.add_row("Total Capital", f"${metrics['total_capital']:,.0f}", "✅")
    overview_table.add_row("Invested Capital", f"${metrics['invested_capital']:,.0f}", "✅")
    overview_table.add_row("Cash Balance", f"${metrics['cash_balance']:,.0f}", "✅")
    overview_table.add_row("Cash Percentage", f"{metrics['cash_percentage']:.1%}", "✅" if metrics['cash_percentage'] >= 0.05 else "⚠️")
    overview_table.add_row("Active Strategies", str(metrics['num_active_strategies']), "✅")
    overview_table.add_row("Portfolio Status", metrics['status'].title(), "✅")
    
    console.print(overview_table)
    
    # Strategy allocation table
    if detailed:
        console.print("\n")
        allocation_table = Table(title="Strategy Allocations", show_header=True)
        allocation_table.add_column("Strategy", style="cyan", width=20)
        allocation_table.add_column("Target", style="blue", width=12)
        allocation_table.add_column("Current", style="green", width=12)
        allocation_table.add_column("Deviation", style="yellow", width=12)
        allocation_table.add_column("Status", style="magenta", width=10)
        
        for name, allocation in portfolio.strategies.items():
            deviation = allocation.current_allocation - allocation.target_allocation
            status = "✅" if abs(deviation) < 0.05 else "⚠️"
            
            allocation_table.add_row(
                name,
                f"{allocation.target_allocation:.1%}",
                f"{allocation.current_allocation:.1%}",
                f"{deviation:+.1%}",
                status
            )
        
        console.print(allocation_table)
        
        # Rebalancing analysis
        rebalancing_needs = portfolio.calculate_rebalancing_needs()
        if rebalancing_needs:
            console.print("\n[yellow]⚠️ Rebalancing recommended for:[/]")
            for strategy, change in rebalancing_needs.items():
                console.print(f"  • {strategy}: {change:+.1%}")
        else:
            console.print("\n[green]✅ Portfolio is well-balanced[/]")


@portfolio_app.command("rebalance")
def rebalance_portfolio(
    portfolio_file: Optional[str] = typer.Option(None, "--portfolio", "-p", help="Portfolio config file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show rebalancing plan without executing"),
    force: bool = typer.Option(False, "--force", help="Force rebalancing even if not needed")
):
    """Execute portfolio rebalancing."""
    
    console.print(f"\n[bold blue]⚖️ Portfolio Rebalancing[/]\n")
    
    # Create demo portfolio (same as status command)
    config = PortfolioConfig(total_capital=2000000, rebalance_threshold=0.05)
    portfolio = PortfolioManager(config)
    
    # Add strategies with some drift
    demo_strategies = [
        {"name": "IronCondor_SPY", "target": 0.30, "current": 0.27},
        {"name": "Strangle_QQQ", "target": 0.25, "current": 0.31},
        {"name": "CalendarSpread_IWM", "target": 0.20, "current": 0.22},
        {"name": "CoveredCall_MSFT", "target": 0.15, "current": 0.14},
    ]
    
    class MockStrategy:
        def __init__(self, name): self.name = name
    
    for strategy in demo_strategies:
        portfolio.add_strategy(strategy["name"], MockStrategy(strategy["name"]), strategy["target"])
        portfolio.strategies[strategy["name"]].current_allocation = strategy["current"]
    
    # Check rebalancing needs
    rebalancing_needs = portfolio.calculate_rebalancing_needs()
    
    if not rebalancing_needs and not force:
        console.print("[green]✅ Portfolio is already well-balanced. No rebalancing needed.[/]")
        return
    
    # Display rebalancing plan
    plan_table = Table(title="Rebalancing Plan", show_header=True)
    plan_table.add_column("Strategy", style="cyan", width=20)
    plan_table.add_column("Current", style="blue", width=12)
    plan_table.add_column("Target", style="green", width=12)
    plan_table.add_column("Change", style="yellow", width=12)
    plan_table.add_column("Capital Move", style="magenta", width=15)
    
    total_capital_moved = 0
    for name, allocation in portfolio.strategies.items():
        change = allocation.target_allocation - allocation.current_allocation
        capital_change = change * config.total_capital
        total_capital_moved += abs(capital_change)
        
        plan_table.add_row(
            name,
            f"{allocation.current_allocation:.1%}",
            f"{allocation.target_allocation:.1%}",
            f"{change:+.1%}",
            f"${capital_change:+,.0f}"
        )
    
    console.print(plan_table)
    
    console.print(f"\n[bold]Total Capital Movement:[/] ${total_capital_moved:,.0f}")
    
    if dry_run:
        console.print("\n[yellow]🔍 Dry run mode - no changes executed[/]")
        return
    
    # Execute rebalancing
    confirm = typer.confirm("Execute rebalancing?")
    if confirm:
        with console.status("[bold green]Executing rebalancing..."):
            success = portfolio.execute_rebalancing(rebalancing_needs)
        
        if success:
            console.print("\n[green]✅ Rebalancing completed successfully![/]")
            
            # Show updated allocations
            after_table = Table(title="Post-Rebalancing Allocations", show_header=True)
            after_table.add_column("Strategy", style="cyan")
            after_table.add_column("Allocation", style="green")
            after_table.add_column("Status", style="magenta")
            
            for name, allocation in portfolio.strategies.items():
                after_table.add_row(
                    name,
                    f"{allocation.current_allocation:.1%}",
                    "✅ Balanced"
                )
            
            console.print(f"\n{after_table}")
        else:
            console.print("\n[red]❌ Rebalancing failed![/]")
    else:
        console.print("\n[yellow]Rebalancing cancelled.[/]")


@portfolio_app.command("performance")
def portfolio_performance(
    portfolio_file: Optional[str] = typer.Option(None, "--portfolio", "-p", help="Portfolio config file"),
    period: str = typer.Option("1M", "--period", help="Performance period (1D, 1W, 1M, 3M, 1Y)"),
    benchmark: Optional[str] = typer.Option(None, "--benchmark", help="Benchmark comparison")
):
    """Analyze portfolio performance and attribution."""
    
    console.print(f"\n[bold blue]📈 Portfolio Performance Analysis[/]\n")
    
    # Simulate performance data
    np.random.seed(42)
    periods = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 252}
    days = periods.get(period, 30)
    
    # Generate synthetic performance data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    strategies = {
        "IronCondor_SPY": {"weight": 0.30, "return": 0.08, "vol": 0.12},
        "Strangle_QQQ": {"weight": 0.25, "return": 0.15, "vol": 0.25},
        "CalendarSpread_IWM": {"weight": 0.20, "return": 0.06, "vol": 0.08},
        "CoveredCall_MSFT": {"weight": 0.15, "return": 0.04, "vol": 0.06},
    }
    
    portfolio_returns = []
    strategy_returns = {}
    
    for strategy, params in strategies.items():
        daily_returns = np.random.normal(
            params["return"]/252, 
            params["vol"]/np.sqrt(252), 
            days
        )
        strategy_returns[strategy] = daily_returns
        
        if not portfolio_returns:
            portfolio_returns = daily_returns * params["weight"]
        else:
            portfolio_returns += daily_returns * params["weight"]
    
    # Calculate performance metrics
    total_return = np.prod(1 + portfolio_returns) - 1
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = np.mean(portfolio_returns) * 252 / volatility if volatility > 0 else 0
    max_drawdown = np.min(np.cumsum(portfolio_returns))
    
    # Performance summary
    perf_table = Table(title=f"Performance Summary - {period}", show_header=True)
    perf_table.add_column("Metric", style="cyan", width=20)
    perf_table.add_column("Value", style="green", width=15)
    perf_table.add_column("Benchmark", style="blue", width=15)
    
    perf_table.add_row("Total Return", f"{total_return:.2%}", "N/A")
    perf_table.add_row("Annualized Return", f"{total_return * 252/days:.2%}", "N/A")
    perf_table.add_row("Volatility", f"{volatility:.2%}", "N/A")
    perf_table.add_row("Sharpe Ratio", f"{sharpe_ratio:.3f}", "N/A")
    perf_table.add_row("Max Drawdown", f"{max_drawdown:.2%}", "N/A")
    
    console.print(perf_table)
    
    # Strategy attribution
    attribution_table = Table(title="Strategy Attribution", show_header=True)
    attribution_table.add_column("Strategy", style="cyan", width=20)
    attribution_table.add_column("Weight", style="blue", width=10)
    attribution_table.add_column("Return", style="green", width=12)
    attribution_table.add_column("Contribution", style="yellow", width=15)
    attribution_table.add_column("Sharpe", style="magenta", width=10)
    
    for strategy, params in strategies.items():
        strat_returns = strategy_returns[strategy]
        strat_total_return = np.prod(1 + strat_returns) - 1
        contribution = strat_total_return * params["weight"]
        strat_vol = np.std(strat_returns) * np.sqrt(252)
        strat_sharpe = np.mean(strat_returns) * 252 / strat_vol if strat_vol > 0 else 0
        
        attribution_table.add_row(
            strategy,
            f"{params['weight']:.0%}",
            f"{strat_total_return:.2%}",
            f"{contribution:.2%}",
            f"{strat_sharpe:.3f}"
        )
    
    console.print(f"\n{attribution_table}")
    
    # Performance insights
    best_performer = max(strategies.keys(), key=lambda s: np.prod(1 + strategy_returns[s]) - 1)
    worst_performer = min(strategies.keys(), key=lambda s: np.prod(1 + strategy_returns[s]) - 1)
    
    insights_panel = Panel(
        f"Performance Insights:\n\n"
        f"🏆 Best Performer: {best_performer}\n"
        f"📉 Worst Performer: {worst_performer}\n"
        f"🎯 Risk-Adjusted Performance: {sharpe_ratio:.3f} Sharpe\n"
        f"💼 Portfolio Diversification: {'Good' if len(strategies) >= 4 else 'Limited'}\n"
        f"📊 Volatility Level: {'Low' if volatility < 0.15 else 'Moderate' if volatility < 0.25 else 'High'}",
        title="💡 Key Insights",
        border_style="green"
    )
    console.print(f"\n{insights_panel}")


if __name__ == "__main__":
    portfolio_app()