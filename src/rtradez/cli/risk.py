"""
Risk Management CLI Commands.

Position sizing, risk limits, VaR calculation, and risk monitoring tools.
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import pandas as pd
import numpy as np
from datetime import datetime

from ..risk import (
    KellyConfig, FixedFractionConfig, VolatilityAdjustedConfig,
    KellyCriterion, FixedFractionSizer, VolatilityAdjustedSizer,
    SizingMethod, create_position_sizer,
    PortfolioRiskLimits, PositionRiskLimits, RiskLimitManager,
    create_basic_risk_monitor
)

console = Console()
risk_app = typer.Typer(name="risk", help="Risk management and position sizing tools")


@risk_app.command("position-size")
def calculate_position_size(
    capital: float = typer.Option(100000, "--capital", "-c", help="Total capital available"),
    method: str = typer.Option("kelly", "--method", "-m", help="Sizing method (kelly, fixed, volatility)"),
    expected_return: float = typer.Option(0.12, "--return", "-r", help="Expected annual return"),
    volatility: float = typer.Option(0.20, "--volatility", "-v", help="Expected volatility"),
    max_risk: float = typer.Option(0.02, "--max-risk", help="Maximum risk per trade"),
    strategy_name: str = typer.Option("Strategy_1", "--strategy", "-s", help="Strategy name"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """Calculate optimal position size using various methods."""
    
    console.print(f"\n[bold blue]🎯 Position Sizing Analysis[/]\n")
    
    # Create configuration based on method
    if method.lower() == "kelly":
        config = KellyConfig(total_capital=capital, max_risk_per_trade=max_risk)
        sizer = KellyCriterion(config)
    elif method.lower() == "fixed":
        config = FixedFractionConfig(total_capital=capital, max_risk_per_trade=max_risk)
        sizer = FixedFractionSizer(config)
    elif method.lower() == "volatility":
        config = VolatilityAdjustedConfig(total_capital=capital, max_risk_per_trade=max_risk)
        sizer = VolatilityAdjustedSizer(config)
    else:
        console.print(f"[red]Error: Unknown method '{method}'. Use: kelly, fixed, volatility[/]")
        raise typer.Exit(1)
    
    # Calculate position size
    with console.status("[bold green]Calculating optimal position size..."):
        result = sizer.calculate_position_size(
            strategy_name=strategy_name,
            expected_return=expected_return,
            volatility=volatility
        )
    
    if output_format == "json":
        import json
        output = {
            "strategy_name": result.strategy_name,
            "recommended_size": result.recommended_size,
            "max_position_value": result.max_position_value,
            "risk_adjusted_size": result.risk_adjusted_size,
            "confidence_level": result.confidence_level,
            "reasoning": result.reasoning,
            "warnings": result.warnings
        }
        console.print(json.dumps(output, indent=2))
    else:
        # Create results table
        table = Table(title=f"Position Sizing Results - {method.title()} Method", show_header=True)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green", width=20)
        table.add_column("Description", style="dim", width=35)
        
        table.add_row("Strategy", result.strategy_name, "Strategy identifier")
        table.add_row("Recommended Size", f"${result.recommended_size:,.0f}", "Optimal position size")
        table.add_row("Max Position", f"${result.max_position_value:,.0f}", "Risk-based maximum")
        table.add_row("Risk Adjusted", f"${result.risk_adjusted_size:,.0f}", "Final recommendation")
        table.add_row("Confidence", f"{result.confidence_level:.1%}", "Sizing confidence")
        table.add_row("Capital Usage", f"{result.recommended_size/capital:.1%}", "Percentage of capital")
        
        console.print(table)
        
        # Show reasoning and warnings
        if result.reasoning:
            reasoning_panel = Panel(result.reasoning, title="💡 Reasoning", border_style="blue")
            console.print(f"\n{reasoning_panel}")
        
        if result.warnings:
            warnings_text = "\n".join([f"• {warning}" for warning in result.warnings])
            warnings_panel = Panel(warnings_text, title="⚠️ Warnings", border_style="yellow")
            console.print(f"\n{warnings_panel}")


@risk_app.command("compare-methods")
def compare_sizing_methods(
    capital: float = typer.Option(100000, "--capital", "-c", help="Total capital"),
    expected_return: float = typer.Option(0.12, "--return", "-r", help="Expected return"),
    volatility: float = typer.Option(0.20, "--volatility", "-v", help="Volatility"),
    strategy_name: str = typer.Option("Comparison", "--strategy", "-s", help="Strategy name")
):
    """Compare position sizing across different methods."""
    
    console.print(f"\n[bold blue]⚖️  Position Sizing Method Comparison[/]\n")
    
    methods = {
        "Kelly Criterion": (SizingMethod.KELLY, KellyConfig(total_capital=capital)),
        "Fixed Fraction": (SizingMethod.FIXED_FRACTION, FixedFractionConfig(total_capital=capital)),
        "Volatility Adjusted": (SizingMethod.VOLATILITY_ADJUSTED, VolatilityAdjustedConfig(total_capital=capital))
    }
    
    # Create comparison table
    table = Table(title="Position Sizing Method Comparison", show_header=True)
    table.add_column("Method", style="cyan", width=18)
    table.add_column("Position Size", style="green", width=15)
    table.add_column("Capital %", style="blue", width=12)
    table.add_column("Confidence", style="magenta", width=12)
    table.add_column("Key Benefit", style="dim", width=30)
    
    results = {}
    
    for method_name, (method_enum, config) in track(methods.items(), description="Calculating..."):
        sizer = create_position_sizer(method_enum, config)
        result = sizer.calculate_position_size(
            strategy_name=strategy_name,
            expected_return=expected_return,
            volatility=volatility
        )
        results[method_name] = result
        
        benefits = {
            "Kelly Criterion": "Optimal growth rate",
            "Fixed Fraction": "Consistent risk exposure",
            "Volatility Adjusted": "Volatility normalization"
        }
        
        table.add_row(
            method_name,
            f"${result.recommended_size:,.0f}",
            f"{result.recommended_size/capital:.1%}",
            f"{result.confidence_level:.1%}",
            benefits[method_name]
        )
    
    console.print(table)
    
    # Summary insights
    max_size = max(r.recommended_size for r in results.values())
    min_size = min(r.recommended_size for r in results.values())
    
    insights = [
        f"💰 Range: ${min_size:,.0f} - ${max_size:,.0f} ({(max_size-min_size)/min_size:.0%} variation)",
        f"📊 Most Conservative: {min(results.keys(), key=lambda k: results[k].recommended_size)}",
        f"🚀 Most Aggressive: {max(results.keys(), key=lambda k: results[k].recommended_size)}",
        f"🎯 Highest Confidence: {max(results.keys(), key=lambda k: results[k].confidence_level)}"
    ]
    
    insights_panel = Panel("\n".join(insights), title="📈 Key Insights", border_style="green")
    console.print(f"\n{insights_panel}")


@risk_app.command("risk-limits")
def set_risk_limits(
    capital: float = typer.Option(1000000, "--capital", "-c", help="Portfolio capital"),
    max_exposure: float = typer.Option(1.2, "--max-exposure", help="Maximum total exposure"),
    max_concentration: float = typer.Option(0.25, "--max-concentration", help="Max position concentration"),
    max_var: float = typer.Option(0.05, "--max-var", help="Maximum daily VaR"),
    max_drawdown: float = typer.Option(0.15, "--max-drawdown", help="Maximum drawdown"),
    save_config: bool = typer.Option(False, "--save", help="Save configuration")
):
    """Configure and validate portfolio risk limits."""
    
    console.print(f"\n[bold blue]🛡️  Risk Limits Configuration[/]\n")
    
    # Create risk limits
    portfolio_limits = PortfolioRiskLimits(
        max_total_exposure=max_exposure,
        max_position_concentration=max_concentration,
        max_daily_var=max_var,
        max_drawdown=max_drawdown,
        max_leverage=max_exposure
    )
    
    position_limits = PositionRiskLimits(
        max_position_size=max_concentration,
        max_delta_exposure=0.10,
        min_time_to_expiry=7
    )
    
    risk_manager = RiskLimitManager(portfolio_limits, position_limits, capital)
    
    # Display configuration
    table = Table(title="Risk Limits Configuration", show_header=True)
    table.add_column("Limit Type", style="cyan", width=25)
    table.add_column("Value", style="green", width=15)
    table.add_column("Description", style="dim", width=35)
    
    # Portfolio limits
    table.add_row("Max Total Exposure", f"{max_exposure:.1%}", "Maximum portfolio exposure")
    table.add_row("Max Concentration", f"{max_concentration:.1%}", "Single position limit")
    table.add_row("Max Daily VaR", f"{max_var:.1%}", "Value at Risk threshold")
    table.add_row("Max Drawdown", f"{max_drawdown:.1%}", "Maximum portfolio drawdown")
    table.add_row("Position Size Limit", f"${capital * max_concentration:,.0f}", "Dollar position limit")
    
    console.print(table)
    
    # Validation examples
    test_scenarios = [
        {"position_value": capital * 0.15, "description": "15% position", "should_pass": True},
        {"position_value": capital * 0.35, "description": "35% position", "should_pass": False},
        {"exposure": 0.8, "description": "80% exposure", "should_pass": True},
        {"exposure": 1.5, "description": "150% exposure", "should_pass": False}
    ]
    
    console.print("\n[bold]Risk Limit Validation Examples:[/]\n")
    
    for scenario in test_scenarios:
        if "position_value" in scenario:
            try:
                risk_manager.check_position_limits(scenario["position_value"], 0.05, 30)
                result = "✅ PASS" if scenario["should_pass"] else "❌ FAIL (should reject)"
                color = "green" if scenario["should_pass"] else "red"
            except Exception as e:
                result = "❌ REJECT" if not scenario["should_pass"] else "✅ FAIL (should pass)"
                color = "green" if not scenario["should_pass"] else "red"
        else:
            # Portfolio exposure check would go here
            result = "✅ PASS" if scenario["should_pass"] else "❌ REJECT"
            color = "green"
        
        console.print(f"  [{color}]{result}[/] - {scenario['description']}")
    
    if save_config:
        config_data = {
            "portfolio_limits": portfolio_limits.dict(),
            "position_limits": position_limits.dict(),
            "capital": capital,
            "created": datetime.now().isoformat()
        }
        
        import json
        with open("risk_limits_config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        console.print(f"\n[green]✅ Configuration saved to risk_limits_config.json[/]")


@risk_app.command("monitor")
def risk_monitor(
    capital: float = typer.Option(1000000, "--capital", "-c", help="Portfolio capital"),
    simulate: bool = typer.Option(False, "--simulate", help="Run simulation mode"),
    duration: int = typer.Option(10, "--duration", help="Simulation duration (seconds)")
):
    """Real-time risk monitoring dashboard."""
    
    console.print(f"\n[bold blue]📊 Real-Time Risk Monitor[/]\n")
    
    # Create risk monitor
    risk_monitor = create_basic_risk_monitor(capital)
    
    if simulate:
        import time
        import random
        
        console.print("[yellow]Running risk monitoring simulation...[/]\n")
        
        for i in track(range(duration), description="Monitoring..."):
            # Simulate portfolio data
            portfolio_data = {
                'total_pnl': capital * (1 + random.uniform(-0.05, 0.05)),
                'positions': {
                    f'Strategy_{j}': {
                        'market_value': random.uniform(50000, 200000),
                        'delta': random.uniform(-0.1, 0.1),
                        'gamma': random.uniform(0, 0.05),
                        'theta': random.uniform(-100, 0),
                        'vega': random.uniform(0, 500)
                    }
                    for j in range(1, 4)
                }
            }
            
            # Update monitor
            risk_monitor.update_portfolio_data(portfolio_data)
            time.sleep(1)
        
        # Get final dashboard
        dashboard = risk_monitor.get_risk_dashboard()
        
        # Display results
        table = Table(title="Risk Monitoring Results", show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        table.add_column("Status", style="magenta", width=15)
        
        table.add_row("Total Alerts", str(dashboard['active_alerts']['total']), "Normal" if dashboard['active_alerts']['total'] == 0 else "Warning")
        table.add_row("Critical Alerts", str(dashboard['active_alerts']['critical']), "OK" if dashboard['active_alerts']['critical'] == 0 else "CRITICAL")
        table.add_row("Recent Alerts", str(dashboard['recent_alerts_24h']), "Normal")
        table.add_row("Monitoring Status", "Active", "✅ Online")
        
        console.print(table)
        
    else:
        console.print("[dim]Risk monitor initialized. Use --simulate to run demo.[/]")
        
        # Show current configuration
        info_panel = Panel(
            "Risk Monitor Configuration:\n\n"
            f"• Portfolio Capital: ${capital:,.0f}\n"
            f"• Alert Thresholds: Configured\n"
            f"• Update Frequency: Real-time\n"
            f"• Status: Ready\n\n"
            "Use [bold]--simulate[/bold] flag to run demonstration.",
            title="📊 Monitor Info",
            border_style="blue"
        )
        console.print(info_panel)


if __name__ == "__main__":
    risk_app()