"""
Configuration Management CLI Commands.

System configuration, API keys, and environment setup tools.
"""

import typer
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import json
import os
from pathlib import Path

console = Console()
config_app = typer.Typer(name="config", help="Configuration management and system setup")

# Default configuration structure
DEFAULT_CONFIG = {
    "api_keys": {
        "alpha_vantage": "",
        "polygon": "",
        "fred": "",
        "openai": ""
    },
    "data_sources": {
        "default_provider": "yahoo",
        "cache_enabled": True,
        "cache_duration_hours": 24,
        "max_concurrent_requests": 5
    },
    "risk_management": {
        "default_max_risk_per_trade": 0.02,
        "default_portfolio_max_risk": 0.15,
        "emergency_stop_enabled": True,
        "emergency_stop_drawdown": 0.20
    },
    "portfolio": {
        "default_capital": 100000,
        "default_rebalance_frequency": "weekly",
        "default_rebalance_threshold": 0.05,
        "default_cash_reserve": 0.05
    },
    "analysis": {
        "default_optimization_trials": 100,
        "default_backtest_period": "1Y",
        "parallel_processing": True,
        "max_workers": 4
    },
    "logging": {
        "level": "INFO",
        "file_logging": True,
        "log_directory": "logs"
    }
}


def get_config_path() -> Path:
    """Get the configuration file path."""
    config_dir = Path.home() / ".rtradez"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "config.json"


def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config_path = get_config_path()
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    config_path = get_config_path()
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


@config_app.command("init")
def init_config():
    """Initialize RTradez configuration."""
    
    console.print(f"\n[bold blue]🔧 RTradez Configuration Setup[/]\n")
    
    config_path = get_config_path()
    
    if config_path.exists():
        overwrite = Confirm.ask(f"Configuration already exists at {config_path}. Overwrite?")
        if not overwrite:
            console.print("[yellow]Configuration initialization cancelled.[/]")
            return
    
    console.print("[bold]Setting up RTradez configuration...[/]\n")
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Interactive setup for key settings
    console.print("[bold cyan]💰 Portfolio Settings[/]")
    default_capital = Prompt.ask("Default portfolio capital", default="100000")
    config["portfolio"]["default_capital"] = float(default_capital)
    
    console.print("\n[bold cyan]🛡️ Risk Management[/]")
    max_risk = Prompt.ask("Maximum risk per trade (%)", default="2.0")
    config["risk_management"]["default_max_risk_per_trade"] = float(max_risk) / 100
    
    console.print("\n[bold cyan]📊 Data Sources[/]")
    provider = Prompt.ask("Default data provider", choices=["yahoo", "alpha_vantage", "polygon"], default="yahoo")
    config["data_sources"]["default_provider"] = provider
    
    console.print("\n[bold cyan]🔑 API Keys (Optional - can be set later)[/]")
    setup_apis = Confirm.ask("Configure API keys now?", default=False)
    
    if setup_apis:
        for api_name in config["api_keys"].keys():
            key = Prompt.ask(f"{api_name.replace('_', ' ').title()} API key", default="", password=True)
            if key:
                config["api_keys"][api_name] = key
    
    # Save configuration
    save_config(config)
    
    console.print(f"\n[green]✅ Configuration saved to {config_path}[/]")
    
    # Display summary
    summary_table = Table(title="Configuration Summary", show_header=True)
    summary_table.add_column("Setting", style="cyan", width=25)
    summary_table.add_column("Value", style="green", width=30)
    
    summary_table.add_row("Config Location", str(config_path))
    summary_table.add_row("Default Capital", f"${config['portfolio']['default_capital']:,.0f}")
    summary_table.add_row("Max Risk per Trade", f"{config['risk_management']['default_max_risk_per_trade']:.1%}")
    summary_table.add_row("Data Provider", config['data_sources']['default_provider'])
    summary_table.add_row("API Keys Configured", str(sum(1 for k in config['api_keys'].values() if k)))
    
    console.print(f"\n{summary_table}")
    
    next_steps = Panel(
        "Next Steps:\n\n"
        "• Set API keys: [bold]rtradez config set-api <provider> <key>[/]\n"
        "• View current config: [bold]rtradez config show[/]\n"
        "• Test data sources: [bold]rtradez data sources[/]\n"
        "• Create portfolio: [bold]rtradez portfolio create[/]\n"
        "• Run analysis: [bold]rtradez analysis optimize[/]",
        title="🚀 Next Steps",
        border_style="green"
    )
    console.print(f"\n{next_steps}")


@config_app.command("show")
def show_config(
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Show specific section"),
    show_secrets: bool = typer.Option(False, "--show-secrets", help="Show API keys and secrets")
):
    """Display current configuration."""
    
    console.print(f"\n[bold blue]📋 RTradez Configuration[/]\n")
    
    config = load_config()
    
    if section:
        if section in config:
            display_config = {section: config[section]}
        else:
            console.print(f"[red]Error: Section '{section}' not found[/]")
            return
    else:
        display_config = config
    
    for section_name, section_data in display_config.items():
        # Create table for each section
        table = Table(title=f"{section_name.replace('_', ' ').title()} Configuration", show_header=True)
        table.add_column("Setting", style="cyan", width=30)
        table.add_column("Value", style="green", width=40)
        table.add_column("Type", style="blue", width=15)
        
        for key, value in section_data.items():
            # Handle API keys and secrets
            if not show_secrets and ("key" in key.lower() or "secret" in key.lower() or "password" in key.lower()):
                if value:
                    display_value = "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
                else:
                    display_value = "[dim]Not set[/]"
            else:
                if isinstance(value, bool):
                    display_value = "✅ Enabled" if value else "❌ Disabled"
                elif isinstance(value, (int, float)):
                    if key.endswith("_per_trade") or key.endswith("_risk") or key.endswith("_drawdown"):
                        display_value = f"{value:.1%}"
                    elif "capital" in key.lower():
                        display_value = f"${value:,.0f}"
                    else:
                        display_value = str(value)
                else:
                    display_value = str(value) if value else "[dim]Not set[/]"
            
            table.add_row(
                key.replace("_", " ").title(),
                display_value,
                type(value).__name__
            )
        
        console.print(table)
        console.print()  # Add spacing between sections


@config_app.command("set")
def set_config_value(
    key: str = typer.Argument(..., help="Configuration key (e.g., 'portfolio.default_capital')"),
    value: str = typer.Argument(..., help="New value"),
    value_type: str = typer.Option("auto", "--type", "-t", help="Value type (str, int, float, bool)")
):
    """Set a configuration value."""
    
    console.print(f"\n[bold blue]⚙️ Setting Configuration Value[/]\n")
    
    config = load_config()
    
    # Parse nested key
    key_parts = key.split('.')
    if len(key_parts) != 2:
        console.print("[red]Error: Key must be in format 'section.setting'[/]")
        return
    
    section, setting = key_parts
    
    if section not in config:
        console.print(f"[red]Error: Section '{section}' not found[/]")
        return
    
    # Convert value to appropriate type
    if value_type == "auto":
        # Try to infer type from existing value
        if setting in config[section]:
            existing_type = type(config[section][setting])
            if existing_type == bool:
                parsed_value = value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            elif existing_type == int:
                parsed_value = int(value)
            elif existing_type == float:
                parsed_value = float(value)
            else:
                parsed_value = value
        else:
            # Try to parse as number, then bool, then string
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        parsed_value = value.lower() == 'true'
                    else:
                        parsed_value = value
    else:
        # Use specified type
        if value_type == "bool":
            parsed_value = value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        elif value_type == "int":
            parsed_value = int(value)
        elif value_type == "float":
            parsed_value = float(value)
        else:
            parsed_value = value
    
    # Store old value for display
    old_value = config[section].get(setting, "[not set]")
    
    # Update configuration
    config[section][setting] = parsed_value
    save_config(config)
    
    console.print(f"[green]✅ Updated {section}.{setting}[/]")
    console.print(f"  Old value: {old_value}")
    console.print(f"  New value: {parsed_value}")


@config_app.command("set-api")
def set_api_key(
    provider: str = typer.Argument(..., help="API provider name"),
    api_key: str = typer.Argument(..., help="API key"),
    verify: bool = typer.Option(True, "--verify", help="Verify API key works")
):
    """Set API key for a data provider."""
    
    console.print(f"\n[bold blue]🔑 Setting API Key for {provider.title()}[/]\n")
    
    config = load_config()
    
    # Normalize provider name
    provider_key = provider.lower().replace("-", "_")
    
    if provider_key not in config["api_keys"]:
        console.print(f"[red]Error: Unknown provider '{provider}'. Available: {', '.join(config['api_keys'].keys())}[/]")
        return
    
    # Store the API key
    config["api_keys"][provider_key] = api_key
    save_config(config)
    
    console.print(f"[green]✅ API key for {provider} has been saved[/]")
    
    if verify:
        console.print(f"[yellow]⚠️ API verification not implemented yet[/]")
        console.print(f"Use [bold]rtradez data sources[/bold] to test data source connections")


@config_app.command("reset")
def reset_config(
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Reset specific section only"),
    confirm_reset: bool = typer.Option(False, "--yes", help="Skip confirmation prompt")
):
    """Reset configuration to defaults."""
    
    console.print(f"\n[bold blue]🔄 Reset Configuration[/]\n")
    
    if not confirm_reset:
        if section:
            confirm = Confirm.ask(f"Reset {section} section to defaults?")
        else:
            confirm = Confirm.ask("Reset ALL configuration to defaults? This cannot be undone.")
        
        if not confirm:
            console.print("[yellow]Reset cancelled.[/]")
            return
    
    if section:
        if section in DEFAULT_CONFIG:
            config = load_config()
            config[section] = DEFAULT_CONFIG[section].copy()
            save_config(config)
            console.print(f"[green]✅ Reset {section} section to defaults[/]")
        else:
            console.print(f"[red]Error: Section '{section}' not found[/]")
    else:
        save_config(DEFAULT_CONFIG.copy())
        console.print("[green]✅ Reset all configuration to defaults[/]")


@config_app.command("export")
def export_config(
    output_file: str = typer.Option("rtradez_config.json", "--output", "-o", help="Output file path"),
    include_secrets: bool = typer.Option(False, "--include-secrets", help="Include API keys in export")
):
    """Export configuration to a file."""
    
    console.print(f"\n[bold blue]📤 Export Configuration[/]\n")
    
    config = load_config()
    
    if not include_secrets:
        # Remove sensitive information
        export_config = config.copy()
        export_config["api_keys"] = {k: "" for k in export_config["api_keys"]}
    else:
        export_config = config
    
    with open(output_file, 'w') as f:
        json.dump(export_config, f, indent=2)
    
    console.print(f"[green]✅ Configuration exported to {output_file}[/]")
    
    if not include_secrets:
        console.print("[yellow]⚠️ API keys were excluded from export[/]")
        console.print("Use --include-secrets to include sensitive data")


@config_app.command("import")
def import_config(
    input_file: str = typer.Argument(..., help="Input configuration file"),
    merge: bool = typer.Option(False, "--merge", help="Merge with existing config instead of replacing")
):
    """Import configuration from a file."""
    
    console.print(f"\n[bold blue]📥 Import Configuration[/]\n")
    
    if not os.path.exists(input_file):
        console.print(f"[red]Error: File '{input_file}' not found[/]")
        return
    
    try:
        with open(input_file, 'r') as f:
            import_data = json.load(f)
    except json.JSONDecodeError:
        console.print(f"[red]Error: Invalid JSON in '{input_file}'[/]")
        return
    
    if merge:
        # Merge with existing configuration
        existing_config = load_config()
        
        # Deep merge function
        def deep_merge(base, overlay):
            for key, value in overlay.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(existing_config, import_data)
        final_config = existing_config
        operation = "merged"
    else:
        final_config = import_data
        operation = "imported"
    
    save_config(final_config)
    
    console.print(f"[green]✅ Configuration {operation} from {input_file}[/]")
    
    # Show summary of imported settings
    summary_table = Table(title="Import Summary", show_header=True)
    summary_table.add_column("Section", style="cyan")
    summary_table.add_column("Settings", style="green")
    
    for section, settings in import_data.items():
        if isinstance(settings, dict):
            setting_count = len(settings)
            summary_table.add_row(section, f"{setting_count} settings")
        else:
            summary_table.add_row(section, "1 setting")
    
    console.print(f"\n{summary_table}")


@config_app.command("validate")
def validate_config():
    """Validate current configuration."""
    
    console.print(f"\n[bold blue]✅ Configuration Validation[/]\n")
    
    config = load_config()
    issues = []
    warnings = []
    
    # Validate API keys
    api_keys_set = sum(1 for k in config["api_keys"].values() if k)
    if api_keys_set == 0:
        warnings.append("No API keys configured - some data sources may not work")
    
    # Validate risk management settings
    max_risk = config["risk_management"]["default_max_risk_per_trade"]
    if max_risk > 0.10:
        warnings.append(f"High maximum risk per trade: {max_risk:.1%}")
    
    emergency_drawdown = config["risk_management"]["emergency_stop_drawdown"]
    if emergency_drawdown > 0.30:
        warnings.append(f"High emergency stop drawdown: {emergency_drawdown:.1%}")
    
    # Validate portfolio settings
    default_capital = config["portfolio"]["default_capital"]
    if default_capital < 10000:
        warnings.append(f"Low default capital: ${default_capital:,.0f}")
    
    # Validate analysis settings
    max_workers = config["analysis"]["max_workers"]
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if max_workers > cpu_count:
        warnings.append(f"Max workers ({max_workers}) exceeds CPU count ({cpu_count})")
    
    # Create validation table
    validation_table = Table(title="Configuration Validation Results", show_header=True)
    validation_table.add_column("Check", style="cyan", width=30)
    validation_table.add_column("Status", style="green", width=15)
    validation_table.add_column("Details", style="yellow", width=40)
    
    checks = [
        ("Configuration File", "✅ Found", f"Located at {get_config_path()}"),
        ("API Keys", "✅ Valid" if api_keys_set > 0 else "⚠️ None Set", f"{api_keys_set}/4 configured"),
        ("Risk Settings", "✅ Valid" if max_risk <= 0.05 else "⚠️ High Risk", f"Max risk: {max_risk:.1%}"),
        ("Portfolio Settings", "✅ Valid", f"Capital: ${default_capital:,.0f}"),
        ("Analysis Settings", "✅ Valid", f"Workers: {max_workers}/{cpu_count}"),
    ]
    
    for check, status, details in checks:
        validation_table.add_row(check, status, details)
    
    console.print(validation_table)
    
    if issues:
        issues_panel = Panel(
            "\n".join([f"❌ {issue}" for issue in issues]),
            title="🚨 Issues Found",
            border_style="red"
        )
        console.print(f"\n{issues_panel}")
    
    if warnings:
        warnings_panel = Panel(
            "\n".join([f"⚠️ {warning}" for warning in warnings]),
            title="⚠️ Warnings",
            border_style="yellow"
        )
        console.print(f"\n{warnings_panel}")
    
    if not issues and not warnings:
        success_panel = Panel(
            "All configuration checks passed! ✨\n\n"
            "Your RTradez installation is properly configured and ready to use.",
            title="✅ All Good",
            border_style="green"
        )
        console.print(f"\n{success_panel}")


if __name__ == "__main__":
    config_app()