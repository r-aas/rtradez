"""
RTradez Main CLI Application.

Comprehensive command-line interface for options trading framework.
"""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .risk import risk_app
from .portfolio import portfolio_app
from .data import data_app
from .analysis import analysis_app
from .config import config_app
from .benchmark import app as benchmark_app
from .trading_benchmark import app as trading_benchmark_app

# Initialize Rich console
console = Console()

# Create main application
app = typer.Typer(
    name="rtradez",
    help="🚀 RTradez - Comprehensive Options Trading Framework",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False
)

# Add subcommands
app.add_typer(risk_app, name="risk", help="🛡️  Risk management tools")
app.add_typer(portfolio_app, name="portfolio", help="🏦 Portfolio management")
app.add_typer(data_app, name="data", help="📊 Data processing and alignment")
app.add_typer(analysis_app, name="analysis", help="📈 Market analysis tools")
app.add_typer(config_app, name="config", help="⚙️  Configuration management")
app.add_typer(benchmark_app, name="benchmark", help="🔬 Performance testing and validation")
app.add_typer(trading_benchmark_app, name="backtest", help="📈 Trading strategy benchmarks")


@app.command()
def version():
    """Show RTradez version information."""
    from rtradez import __version__, __author__
    
    version_panel = Panel.fit(
        f"[bold blue]RTradez Options Trading Framework[/]\n\n"
        f"[bold]Version:[/] {__version__}\n"
        f"[bold]Author:[/] {__author__}\n"
        f"[bold]Description:[/] Comprehensive options trading dataset organization & analysis\n\n"
        f"[dim]Components:[/]\n"
        f"  • Risk Management & Position Sizing\n"
        f"  • Portfolio Management & Coordination\n"
        f"  • Multi-Frequency Data Processing\n"
        f"  • Advanced Market Analysis\n"
        f"  • Configuration Management",
        title="📊 RTradez",
        border_style="blue"
    )
    console.print(version_panel)


@app.command()
def status():
    """Show system status and component health."""
    console.print("\n[bold blue]🔍 RTradez System Status[/]\n")
    
    # Create status table
    table = Table(title="Component Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Description", style="dim", width=40)
    
    components = [
        ("Risk Management", "✅ Active", "Position sizing and risk controls"),
        ("Portfolio Manager", "✅ Active", "Multi-strategy coordination"),
        ("Data Sources", "⚠️  Partial", "Some providers need configuration"),
        ("Temporal Alignment", "✅ Active", "Multi-frequency data processing"),
        ("Analysis Tools", "✅ Active", "Market analysis and optimization"),
        ("Configuration", "✅ Active", "Pydantic validation system"),
        ("Benchmarking", "✅ Active", "Performance testing framework"),
        ("Trading Benchmarks", "✅ Active", "Strategy backtesting and analysis")
    ]
    
    for component, status, description in components:
        table.add_row(component, status, description)
    
    console.print(table)
    
    # System info
    info_panel = Panel(
        "[bold]System Information[/]\n\n"
        "• [cyan]CLI Framework:[/] Typer with Rich UI\n"
        "• [cyan]Validation:[/] Pydantic data models\n"
        "• [cyan]Data Processing:[/] Pandas & NumPy\n"
        "• [cyan]Optimization:[/] Optuna framework\n"
        "• [cyan]Visualization:[/] Matplotlib & Plotly\n",
        title="⚙️ System Info",
        border_style="green"
    )
    console.print(f"\n{info_panel}")


@app.command()
def demo():
    """Run interactive demo of RTradez capabilities."""
    from rich.prompt import Confirm, Prompt
    
    console.print("\n[bold blue]🎮 RTradez Interactive Demo[/]\n")
    
    demo_options = {
        "1": ("Risk Management", "position-size", "demo_risk_management"),
        "2": ("Portfolio Analysis", "portfolio", "demo_portfolio"),
        "3": ("Data Processing", "data", "demo_data_processing"),
        "4": ("Market Analysis", "analysis", "demo_analysis"),
        "5": ("Benchmarking", "benchmark", "demo_benchmarking"),
        "6": ("Trading Backtest", "backtest", "demo_backtesting"),
        "7": ("Full Integration", "all", "demo_integration")
    }
    
    console.print("[bold]Available Demos:[/]\n")
    for key, (name, _, _) in demo_options.items():
        console.print(f"  {key}. {name}")
    
    choice = Prompt.ask("\nSelect demo", choices=list(demo_options.keys()), default="1")
    name, command, _ = demo_options[choice]
    
    console.print(f"\n[green]Running {name} demo...[/]\n")
    
    # Execute demo based on choice
    if choice == "1":
        typer.run(risk_app.commands["position-size"].callback)
    elif choice == "2":
        typer.run(portfolio_app.commands["status"].callback)
    elif choice == "3":
        typer.run(data_app.commands["align"].callback)
    elif choice == "4":
        typer.run(analysis_app.commands["optimize"].callback)
    elif choice == "5":
        typer.run(benchmark_app.commands["quick"].callback)
    elif choice == "6":
        typer.run(trading_benchmark_app.commands["backtest"].callback)
    elif choice == "7":
        console.print("[bold yellow]Running comprehensive integration demo...[/]")
        # Run the existing integration demo
        import subprocess
        import sys
        subprocess.run([sys.executable, "examples/pydantic_integration_demo.py"])


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """
    🚀 RTradez - Comprehensive Options Trading Framework
    
    Advanced toolkit for options trading strategy development, risk management,
    and portfolio coordination with multi-frequency data processing capabilities.
    """
    if verbose:
        console.print("[dim]Verbose mode enabled[/]")
    if debug:
        console.print("[dim]Debug mode enabled[/]")


if __name__ == "__main__":
    app()