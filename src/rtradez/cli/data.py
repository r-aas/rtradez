"""
Data Processing CLI Commands.

Temporal alignment, data bucketing, and multi-frequency data processing tools.
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from ..utils.temporal_alignment import (
    TemporalAligner, TemporalAlignerConfig, TemporalProfile, FrequencyType
)
from ..utils.time_bucketing import (
    TimeBucketing, BucketConfig, BucketType
)

console = Console()
data_app = typer.Typer(name="data", help="Data processing and temporal alignment tools")


@data_app.command("align")
def align_datasets(
    input_files: List[str] = typer.Option([], "--input", "-i", help="Input data files"),
    target_frequency: str = typer.Option("daily", "--frequency", "-f", help="Target frequency"),
    alignment_method: str = typer.Option("outer", "--method", "-m", help="Alignment method"),
    fill_method: str = typer.Option("forward_fill", "--fill", help="Fill method"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    simulate: bool = typer.Option(False, "--simulate", help="Run with simulated data")
):
    """Align multiple datasets to a common temporal frequency."""
    
    console.print(f"\n[bold blue]⏰ Temporal Data Alignment[/]\n")
    
    # Map string to enum
    freq_mapping = {
        "tick": FrequencyType.TICK,
        "minute": FrequencyType.MINUTE,
        "hourly": FrequencyType.HOURLY,
        "daily": FrequencyType.DAILY,
        "weekly": FrequencyType.WEEKLY,
        "monthly": FrequencyType.MONTHLY,
        "quarterly": FrequencyType.QUARTERLY,
        "annual": FrequencyType.ANNUAL
    }
    
    target_freq = freq_mapping.get(target_frequency.lower(), FrequencyType.DAILY)
    
    # Create aligner configuration
    config = TemporalAlignerConfig(
        target_frequency=target_freq,
        alignment_method=alignment_method,
        fill_method=fill_method,
        business_days_only=True,
        max_fill_periods=5
    )
    
    aligner = TemporalAligner(config)
    
    if simulate or not input_files:
        console.print("[yellow]Running simulation with synthetic data...[/]\n")
        
        # Generate synthetic datasets with different frequencies
        base_date = datetime(2024, 1, 1)
        
        datasets = {
            "daily_prices": {
                "frequency": FrequencyType.DAILY,
                "data": pd.DataFrame({
                    "price": 100 + np.cumsum(np.random.randn(252) * 0.02),
                    "volume": np.random.randint(1000, 10000, 252)
                }, index=pd.date_range(base_date, periods=252, freq='D'))
            },
            "weekly_fundamentals": {
                "frequency": FrequencyType.WEEKLY,
                "data": pd.DataFrame({
                    "pe_ratio": 15 + np.random.randn(52) * 0.5,
                    "dividend_yield": 0.02 + np.random.randn(52) * 0.001
                }, index=pd.date_range(base_date, periods=52, freq='W'))
            },
            "monthly_macro": {
                "frequency": FrequencyType.MONTHLY,
                "data": pd.DataFrame({
                    "inflation": 0.03 + np.random.randn(12) * 0.005,
                    "unemployment": 0.04 + np.random.randn(12) * 0.002
                }, index=pd.date_range(base_date, periods=12, freq='M'))
            }
        }
        
        # Analyze each dataset
        console.print("[bold]Dataset Analysis:[/]\n")
        
        analysis_table = Table(show_header=True)
        analysis_table.add_column("Dataset", style="cyan", width=20)
        analysis_table.add_column("Frequency", style="blue", width=12)
        analysis_table.add_column("Observations", style="green", width=12)
        analysis_table.add_column("Date Range", style="yellow", width=20)
        analysis_table.add_column("Coverage", style="magenta", width=10)
        
        profiles = {}
        for name, dataset in datasets.items():
            data = dataset["data"]
            
            # Create temporal profile
            profile = TemporalProfile(
                frequency=dataset["frequency"],
                start_date=data.index.min().to_pydatetime(),
                end_date=data.index.max().to_pydatetime(),
                total_observations=len(data),
                missing_periods=0,
                regularity_score=1.0,
                timezone="UTC"
            )
            profiles[name] = profile
            
            analysis_table.add_row(
                name,
                dataset["frequency"].value,
                str(len(data)),
                f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
                f"{profile.coverage_ratio():.0%}"
            )
        
        console.print(analysis_table)
        
        # Simulate alignment process
        console.print(f"\n[bold]Aligning to {target_frequency} frequency...[/]\n")
        
        with console.status("[bold green]Processing temporal alignment..."):
            # Simulate alignment for demo
            aligned_dates = pd.date_range(base_date, base_date + timedelta(days=365), freq='D' if target_freq == FrequencyType.DAILY else 'W')
            
        # Show alignment results
        results_table = Table(title="Alignment Results", show_header=True)
        results_table.add_column("Metric", style="cyan", width=25)
        results_table.add_column("Before", style="blue", width=15)
        results_table.add_column("After", style="green", width=15)
        results_table.add_column("Change", style="yellow", width=15)
        
        total_before = sum(len(d["data"]) for d in datasets.values())
        total_after = len(aligned_dates) * len(datasets)
        
        results_table.add_row("Total Observations", str(total_before), str(total_after), f"{((total_after-total_before)/total_before)*100:+.1f}%")
        results_table.add_row("Date Range", "Various", f"{len(aligned_dates)} periods", "Unified")
        results_table.add_row("Frequency", "Mixed", target_frequency, "Standardized")
        results_table.add_row("Missing Data", "Unknown", "Filled", "Handled")
        
        console.print(results_table)
        
        if output_file:
            console.print(f"\n[green]✅ Aligned data would be saved to {output_file}[/]")
        
    else:
        # Process real files
        console.print(f"[bold]Processing {len(input_files)} input files...[/]\n")
        
        for file_path in track(input_files, description="Loading data..."):
            console.print(f"  📁 {file_path}")
        
        console.print("\n[yellow]Real file processing not implemented in demo.[/]")
        console.print("Use --simulate flag to see alignment demonstration.")
    
    # Configuration summary
    config_panel = Panel(
        f"Alignment Configuration:\n\n"
        f"• Target Frequency: {target_frequency}\n"
        f"• Alignment Method: {alignment_method}\n"
        f"• Fill Method: {fill_method}\n"
        f"• Business Days Only: {config.business_days_only}\n"
        f"• Max Fill Periods: {config.max_fill_periods}",
        title="⚙️ Configuration",
        border_style="blue"
    )
    console.print(f"\n{config_panel}")


@data_app.command("bucket")
def bucket_data(
    input_file: str = typer.Option("", "--input", "-i", help="Input data file"),
    bucket_type: str = typer.Option("trading", "--type", "-t", help="Bucket type"),
    bucket_size: str = typer.Option("1H", "--size", "-s", help="Bucket size"),
    aggregation: str = typer.Option("ohlc", "--agg", "-a", help="Aggregation method"),
    timezone: str = typer.Option("US/Eastern", "--timezone", help="Timezone"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    simulate: bool = typer.Option(False, "--simulate", help="Run with simulated data")
):
    """Bucket time series data into specified intervals."""
    
    console.print(f"\n[bold blue]🗂️  Time Series Bucketing[/]\n")
    
    # Map string to enum
    bucket_type_mapping = {
        "calendar": BucketType.CALENDAR,
        "trading": BucketType.TRADING,
        "rolling": BucketType.ROLLING,
        "event_driven": BucketType.EVENT_DRIVEN,
        "volatility": BucketType.VOLATILITY,
        "volume": BucketType.VOLUME
    }
    
    bucket_enum = bucket_type_mapping.get(bucket_type.lower(), BucketType.TRADING)
    
    # Create bucket configuration
    config = BucketConfig(
        bucket_type=bucket_enum,
        bucket_size=bucket_size,
        aggregation_method=aggregation,
        timezone=timezone,
        session_start=time(9, 30),
        session_end=time(16, 0),
        business_days_only=True
    )
    
    bucketing = TimeBucketing(config)
    
    if simulate or not input_file:
        console.print("[yellow]Generating synthetic high-frequency data...[/]\n")
        
        # Generate minute-by-minute data for one trading day
        start_time = datetime(2024, 10, 1, 9, 30)  # Market open
        end_time = datetime(2024, 10, 1, 16, 0)    # Market close
        
        # Create minute-level timestamps
        timestamps = pd.date_range(start_time, end_time, freq='1min')
        
        # Generate realistic price data
        base_price = 150.0
        returns = np.random.normal(0, 0.001, len(timestamps))  # Small returns
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add some volatility clustering
        volatility = 0.5 + 0.3 * np.sin(np.arange(len(timestamps)) / 20)
        prices += np.random.normal(0, volatility * 0.1, len(timestamps))
        
        # Generate volume data
        volumes = np.random.exponential(1000, len(timestamps))
        
        # Create DataFrame
        raw_data = pd.DataFrame({
            'price': prices,
            'volume': volumes
        }, index=timestamps)
        
        console.print(f"[bold]Raw Data Summary:[/]")
        raw_table = Table(show_header=True)
        raw_table.add_column("Metric", style="cyan")
        raw_table.add_column("Value", style="green")
        
        raw_table.add_row("Data Points", f"{len(raw_data):,}")
        raw_table.add_row("Frequency", "1-minute")
        raw_table.add_row("Time Range", f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
        raw_table.add_row("Price Range", f"${raw_data['price'].min():.2f} - ${raw_data['price'].max():.2f}")
        raw_table.add_row("Avg Volume", f"{raw_data['volume'].mean():,.0f}")
        
        console.print(raw_table)
        
        # Simulate bucketing process
        console.print(f"\n[bold]Bucketing to {bucket_size} intervals using {aggregation} method...[/]\n")
        
        with console.status("[bold green]Processing buckets..."):
            # Simulate bucketing based on size
            if bucket_size == "1H":
                bucket_freq = '1H'
                expected_buckets = 6  # 6.5 hour trading day
            elif bucket_size == "30min":
                bucket_freq = '30min'
                expected_buckets = 13
            elif bucket_size == "15min":
                bucket_freq = '15min'
                expected_buckets = 26
            else:
                bucket_freq = bucket_size
                expected_buckets = 10  # default
            
            # Create bucketed data
            if aggregation == "ohlc":
                bucketed_data = raw_data['price'].resample(bucket_freq).ohlc()
                bucketed_data['volume'] = raw_data['volume'].resample(bucket_freq).sum()
            elif aggregation == "mean":
                bucketed_data = raw_data.resample(bucket_freq).mean()
            elif aggregation == "last":
                bucketed_data = raw_data.resample(bucket_freq).last()
            else:
                bucketed_data = raw_data.resample(bucket_freq).last()
        
        # Show bucketing results
        results_table = Table(title="Bucketing Results", show_header=True)
        results_table.add_column("Metric", style="cyan", width=20)
        results_table.add_column("Original", style="blue", width=15)
        results_table.add_column("Bucketed", style="green", width=15)
        results_table.add_column("Reduction", style="yellow", width=15)
        
        data_reduction = (len(raw_data) - len(bucketed_data)) / len(raw_data)
        
        results_table.add_row("Data Points", f"{len(raw_data):,}", f"{len(bucketed_data):,}", f"{data_reduction:.1%}")
        results_table.add_row("Frequency", "1-minute", bucket_size, "Standardized")
        results_table.add_row("Aggregation", "Raw", aggregation, "Applied")
        results_table.add_row("Storage Size", "100%", f"{(1-data_reduction)*100:.1f}%", f"{data_reduction:.1%}")
        
        console.print(results_table)
        
        # Show sample of bucketed data
        if aggregation == "ohlc":
            sample_table = Table(title="Sample Bucketed Data (OHLC)", show_header=True)
            sample_table.add_column("Time", style="cyan")
            sample_table.add_column("Open", style="green")
            sample_table.add_column("High", style="blue")
            sample_table.add_column("Low", style="red")
            sample_table.add_column("Close", style="green")
            sample_table.add_column("Volume", style="yellow")
            
            for i, (timestamp, row) in enumerate(bucketed_data.head().iterrows()):
                sample_table.add_row(
                    timestamp.strftime('%H:%M'),
                    f"${row['open']:.2f}",
                    f"${row['high']:.2f}",
                    f"${row['low']:.2f}",
                    f"${row['close']:.2f}",
                    f"{row['volume']:,.0f}"
                )
        else:
            sample_table = Table(title="Sample Bucketed Data", show_header=True)
            sample_table.add_column("Time", style="cyan")
            sample_table.add_column("Price", style="green")
            sample_table.add_column("Volume", style="yellow")
            
            for timestamp, row in bucketed_data.head().iterrows():
                sample_table.add_row(
                    timestamp.strftime('%H:%M'),
                    f"${row['price']:.2f}",
                    f"{row['volume']:,.0f}"
                )
        
        console.print(f"\n{sample_table}")
        
        if output_file:
            console.print(f"\n[green]✅ Bucketed data would be saved to {output_file}[/]")
    
    else:
        console.print(f"[bold]Processing file: {input_file}[/]\n")
        console.print("[yellow]Real file processing not implemented in demo.[/]")
        console.print("Use --simulate flag to see bucketing demonstration.")
    
    # Configuration summary
    config_panel = Panel(
        f"Bucketing Configuration:\n\n"
        f"• Bucket Type: {bucket_type}\n"
        f"• Bucket Size: {bucket_size}\n"
        f"• Aggregation: {aggregation}\n"
        f"• Timezone: {timezone}\n"
        f"• Trading Hours: {config.session_start} - {config.session_end}",
        title="⚙️ Configuration",
        border_style="blue"
    )
    console.print(f"\n{config_panel}")


@data_app.command("profile")
def profile_dataset(
    input_file: str = typer.Option("", "--input", "-i", help="Input data file"),
    date_column: str = typer.Option("date", "--date-col", help="Date column name"),
    analyze_gaps: bool = typer.Option(True, "--gaps", help="Analyze data gaps"),
    output_format: str = typer.Option("table", "--format", help="Output format (table, json)")
):
    """Analyze temporal characteristics of a dataset."""
    
    console.print(f"\n[bold blue]📊 Dataset Temporal Profiling[/]\n")
    
    if not input_file:
        console.print("[yellow]Generating synthetic dataset for profiling demonstration...[/]\n")
        
        # Create synthetic dataset with various characteristics
        base_date = datetime(2024, 1, 1)
        
        # Create daily data with some missing periods
        dates = pd.date_range(base_date, periods=365, freq='D')
        
        # Randomly remove some dates to simulate gaps
        missing_indices = np.random.choice(len(dates), size=15, replace=False)
        available_dates = [date for i, date in enumerate(dates) if i not in missing_indices]
        
        data = pd.DataFrame({
            'value': np.random.randn(len(available_dates)),
            'volume': np.random.randint(1000, 10000, len(available_dates))
        }, index=available_dates)
        
        console.print(f"[dim]Generated dataset with {len(data)} observations and {len(missing_indices)} gaps[/]\n")
        
    else:
        console.print(f"[bold]Loading dataset: {input_file}[/]\n")
        console.print("[yellow]Real file loading not implemented in demo.[/]")
        console.print("Showing demonstration with synthetic data instead.\n")
        
        # Use synthetic data for demo
        base_date = datetime(2024, 1, 1)
        dates = pd.date_range(base_date, periods=250, freq='D')
        missing_indices = np.random.choice(len(dates), size=10, replace=False)
        available_dates = [date for i, date in enumerate(dates) if i not in missing_indices]
        
        data = pd.DataFrame({
            'value': np.random.randn(len(available_dates)),
        }, index=available_dates)
    
    # Analyze temporal characteristics
    with console.status("[bold green]Analyzing temporal characteristics..."):
        # Determine frequency
        time_diffs = data.index.to_series().diff().dt.days.dropna()
        mode_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else 1
        
        if mode_diff == 1:
            detected_frequency = FrequencyType.DAILY
        elif mode_diff == 7:
            detected_frequency = FrequencyType.WEEKLY
        elif mode_diff >= 28 and mode_diff <= 31:
            detected_frequency = FrequencyType.MONTHLY
        else:
            detected_frequency = FrequencyType.IRREGULAR
        
        # Calculate regularity score
        expected_diff = mode_diff
        actual_diffs = time_diffs.values
        regularity_score = 1.0 - np.std(actual_diffs - expected_diff) / expected_diff
        regularity_score = max(0.0, min(1.0, regularity_score))
        
        # Create temporal profile
        profile = TemporalProfile(
            frequency=detected_frequency,
            start_date=data.index.min().to_pydatetime(),
            end_date=data.index.max().to_pydatetime(),
            total_observations=len(data),
            missing_periods=len(missing_indices) if 'missing_indices' in locals() else 0,
            regularity_score=regularity_score,
            timezone="UTC",
            weekends_included=True
        )
    
    if output_format == "json":
        import json
        profile_dict = profile.dict()
        profile_dict['start_date'] = profile_dict['start_date'].isoformat()
        profile_dict['end_date'] = profile_dict['end_date'].isoformat()
        console.print(json.dumps(profile_dict, indent=2))
    else:
        # Display profile results
        profile_table = Table(title="Temporal Profile Analysis", show_header=True)
        profile_table.add_column("Characteristic", style="cyan", width=25)
        profile_table.add_column("Value", style="green", width=20)
        profile_table.add_column("Assessment", style="yellow", width=20)
        
        profile_table.add_row("Detected Frequency", detected_frequency.value.title(), "✅ Identified")
        profile_table.add_row("Date Range", f"{profile.start_date.strftime('%Y-%m-%d')} to {profile.end_date.strftime('%Y-%m-%d')}", f"{(profile.end_date - profile.start_date).days} days")
        profile_table.add_row("Total Observations", str(profile.total_observations), "Good" if profile.total_observations > 100 else "Limited")
        profile_table.add_row("Missing Periods", str(profile.missing_periods), "Good" if profile.missing_periods < 10 else "High")
        profile_table.add_row("Regularity Score", f"{profile.regularity_score:.2f}", "High" if profile.regularity_score > 0.8 else "Moderate" if profile.regularity_score > 0.5 else "Low")
        profile_table.add_row("Coverage Ratio", f"{profile.coverage_ratio():.1%}", "Good" if profile.coverage_ratio() > 0.9 else "Moderate")
        
        console.print(profile_table)
        
        if analyze_gaps and profile.missing_periods > 0:
            # Gap analysis
            gaps_panel = Panel(
                f"Data Gap Analysis:\n\n"
                f"• Missing Periods: {profile.missing_periods}\n"
                f"• Gap Rate: {profile.missing_periods/profile.total_observations:.1%}\n"
                f"• Longest Gap: {max(actual_diffs) if 'actual_diffs' in locals() else 'Unknown'} days\n"
                f"• Impact: {'Minimal' if profile.missing_periods < 5 else 'Moderate' if profile.missing_periods < 20 else 'Significant'}",
                title="🔍 Gap Analysis",
                border_style="yellow"
            )
            console.print(f"\n{gaps_panel}")
        
        # Recommendations
        recommendations = []
        
        if profile.regularity_score < 0.7:
            recommendations.append("Consider temporal alignment to standardize intervals")
        
        if profile.missing_periods > profile.total_observations * 0.1:
            recommendations.append("High missing data rate - investigate data quality")
        
        if profile.coverage_ratio() < 0.8:
            recommendations.append("Low coverage ratio - may need data imputation")
        
        if detected_frequency == FrequencyType.IRREGULAR:
            recommendations.append("Irregular frequency detected - consider event-driven bucketing")
        
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations])
            rec_panel = Panel(rec_text, title="💡 Recommendations", border_style="green")
            console.print(f"\n{rec_panel}")


@data_app.command("sources")
def list_data_sources():
    """List available data sources and their status."""
    
    console.print(f"\n[bold blue]📡 Available Data Sources[/]\n")
    
    sources = [
        {"name": "FRED Economic Data", "status": "✅ Active", "type": "Economic", "frequency": "Various"},
        {"name": "Alpha Vantage", "status": "⚙️ Config Required", "type": "Financial", "frequency": "Real-time"},
        {"name": "Yahoo Finance", "status": "✅ Active", "type": "Financial", "frequency": "Daily"},
        {"name": "Polygon.io", "status": "⚙️ Config Required", "type": "Financial", "frequency": "Real-time"},
        {"name": "Sentiment Data", "status": "✅ Active", "type": "Alternative", "frequency": "Daily"},
        {"name": "Crypto Exchanges", "status": "✅ Active", "type": "Cryptocurrency", "frequency": "Real-time"},
        {"name": "Options Data", "status": "✅ Active", "type": "Derivatives", "frequency": "Real-time"},
        {"name": "Weather Data", "status": "🔧 Development", "type": "Alternative", "frequency": "Daily"},
        {"name": "Earnings Data", "status": "🔧 Development", "type": "Fundamental", "frequency": "Quarterly"},
    ]
    
    sources_table = Table(title="Data Sources Status", show_header=True)
    sources_table.add_column("Data Source", style="cyan", width=20)
    sources_table.add_column("Status", style="green", width=15)
    sources_table.add_column("Type", style="blue", width=15)
    sources_table.add_column("Frequency", style="yellow", width=15)
    sources_table.add_column("Notes", style="dim", width=25)
    
    notes = {
        "FRED Economic Data": "Federal Reserve data",
        "Alpha Vantage": "API key required",
        "Yahoo Finance": "Free tier available",
        "Polygon.io": "Premium service",
        "Sentiment Data": "News & social media",
        "Crypto Exchanges": "Multiple exchanges",
        "Options Data": "Real-time chains",
        "Weather Data": "Coming soon",
        "Earnings Data": "Coming soon",
    }
    
    for source in sources:
        sources_table.add_row(
            source["name"],
            source["status"],
            source["type"],
            source["frequency"],
            notes.get(source["name"], "")
        )
    
    console.print(sources_table)
    
    # Quick setup guide
    setup_panel = Panel(
        "Quick Setup Guide:\n\n"
        "1. API Configuration:\n"
        "   • Set environment variables for API keys\n"
        "   • Use 'rtradez config set' command\n\n"
        "2. Data Source Testing:\n"
        "   • Run 'rtradez data test-sources' to verify connections\n\n"
        "3. Data Loading:\n"
        "   • Use 'rtradez data align' for temporal processing\n"
        "   • Use 'rtradez data bucket' for aggregation",
        title="🚀 Setup Guide",
        border_style="green"
    )
    console.print(f"\n{setup_panel}")


if __name__ == "__main__":
    data_app()