"""
Benchmarking CLI commands for RTradez.

Provides comprehensive benchmarking capabilities for performance testing,
stress testing, and validation before live trading deployment.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.text import Text
import json
from pathlib import Path
import time
from typing import Optional, List

from ..benchmarks import (
    BenchmarkConfig, BenchmarkSuite, PerformanceBenchmark, 
    StressTester, ValidationBenchmark, LatencyBenchmark, MemoryProfiler
)

console = Console()
app = typer.Typer(name="benchmark", help="RTradez benchmarking and testing framework")


@app.command()
def run(
    suite: str = typer.Option("all", "--suite", "-s", help="Benchmark suite to run (all, performance, stress, validation, latency, memory)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom benchmark configuration file"),
    iterations: int = typer.Option(1000, "--iterations", "-i", help="Number of stress test iterations"),
    memory_limit: int = typer.Option(4096, "--memory-limit", "-m", help="Memory limit in MB"),
    timeout: float = typer.Option(300.0, "--timeout", "-t", help="Maximum execution time in seconds"),
    save_detailed: bool = typer.Option(True, "--detailed/--no-detailed", help="Save detailed benchmark logs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run comprehensive benchmarking suite before live trading."""
    
    console.print("\n🔬 [bold blue]RTradez Comprehensive Benchmarking Suite[/bold blue]")
    console.print("=" * 60)
    
    # Setup configuration
    if config_file and config_file.exists():
        with open(config_file) as f:
            config_data = json.load(f)
        config = BenchmarkConfig(**config_data)
    else:
        config = BenchmarkConfig(
            stress_iterations=iterations,
            memory_limit_mb=memory_limit,
            max_execution_time=timeout,
            save_detailed_logs=save_detailed,
            output_directory=output_dir or Path("benchmark_results")
        )
    
    console.print(f"📁 Output Directory: {config.output_directory}")
    console.print(f"⚙️  Iterations: {config.stress_iterations}")
    console.print(f"💾 Memory Limit: {config.memory_limit_mb}MB")
    console.print(f"⏱️  Timeout: {config.max_execution_time}s")
    
    # Select benchmark suites
    suites_to_run = []
    if suite.lower() == "all":
        suites_to_run = ["performance", "stress", "validation", "latency", "memory"]
    else:
        suites_to_run = [suite.lower()]
    
    console.print(f"🎯 Running Suites: {', '.join(suites_to_run)}")
    console.print()
    
    # Run benchmarks
    all_results = {}
    total_start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("[cyan]Running Benchmarks...", total=len(suites_to_run))
        
        for suite_name in suites_to_run:
            suite_task = progress.add_task(f"[green]Running {suite_name.title()} Suite...", total=1)
            
            try:
                if suite_name == "performance":
                    benchmark = PerformanceBenchmark(config)
                elif suite_name == "stress":
                    benchmark = StressTester(config)
                elif suite_name == "validation":
                    benchmark = ValidationBenchmark(config)
                elif suite_name == "latency":
                    benchmark = LatencyBenchmark(config)
                elif suite_name == "memory":
                    benchmark = MemoryProfiler(config)
                else:
                    console.print(f"❌ Unknown benchmark suite: {suite_name}")
                    continue
                
                # Run the benchmark suite
                results = benchmark.run_benchmarks()
                all_results[suite_name] = results
                
                progress.update(suite_task, completed=1)
                progress.update(main_task, advance=1)
                
                if verbose:
                    _display_suite_results(suite_name, results)
                
            except Exception as e:
                console.print(f"❌ Error running {suite_name} suite: {str(e)}")
                continue
    
    total_time = time.time() - total_start_time
    
    # Display summary
    _display_benchmark_summary(all_results, total_time)
    
    # Save combined results
    if save_detailed:
        combined_results_file = config.output_directory / "combined_benchmark_results.json"
        with open(combined_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        console.print(f"💾 Combined results saved to: {combined_results_file}")


@app.command()
def quick(
    component: str = typer.Option("risk", "--component", "-c", help="Component to test (risk, portfolio, data)"),
    iterations: int = typer.Option(100, "--iterations", "-i", help="Number of iterations for quick test")
):
    """Run quick performance test for specific component."""
    
    console.print(f"\n⚡ [bold blue]Quick Performance Test - {component.title()}[/bold blue]")
    console.print("=" * 50)
    
    config = BenchmarkConfig(
        stress_iterations=iterations,
        max_execution_time=60.0,
        save_detailed_logs=False
    )
    
    if component.lower() == "risk":
        benchmark = PerformanceBenchmark(config)
        console.print("🎯 Testing risk management performance...")
        
        with console.status("[bold green]Running risk calculations..."):
            results = benchmark.suite.benchmarks[0]()  # First benchmark (position sizing)
        
        _display_quick_results("Risk Management", results)
        
    elif component.lower() == "portfolio":
        benchmark = PerformanceBenchmark(config)
        console.print("🎯 Testing portfolio management performance...")
        
        with console.status("[bold green]Running portfolio operations..."):
            results = benchmark.suite.benchmarks[2]()  # Portfolio rebalancing benchmark
        
        _display_quick_results("Portfolio Management", results)
        
    elif component.lower() == "data":
        benchmark = PerformanceBenchmark(config)
        console.print("🎯 Testing data processing performance...")
        
        with console.status("[bold green]Running data alignment..."):
            results = benchmark.suite.benchmarks[4]()  # Temporal alignment benchmark
        
        _display_quick_results("Data Processing", results)
        
    else:
        console.print(f"❌ Unknown component: {component}")
        console.print("Available components: risk, portfolio, data")


@app.command()
def stress(
    target: str = typer.Option("system", "--target", "-t", help="Stress test target (system, memory, concurrent)"),
    duration: int = typer.Option(30, "--duration", "-d", help="Test duration in seconds"),
    intensity: str = typer.Option("medium", "--intensity", "-i", help="Test intensity (low, medium, high)")
):
    """Run focused stress tests."""
    
    console.print(f"\n💥 [bold red]Stress Testing - {target.title()}[/bold red]")
    console.print("=" * 50)
    
    # Set iterations based on intensity
    intensity_map = {"low": 500, "medium": 1000, "high": 5000}
    iterations = intensity_map.get(intensity.lower(), 1000)
    
    config = BenchmarkConfig(
        stress_iterations=iterations,
        max_execution_time=duration * 2,  # Give extra time
        save_detailed_logs=True
    )
    
    stress_tester = StressTester(config)
    
    console.print(f"🎯 Target: {target}")
    console.print(f"⏱️  Duration: {duration}s")
    console.print(f"🔥 Intensity: {intensity} ({iterations} iterations)")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[red]Running {target} stress test...")
        
        try:
            results = stress_tester.run_benchmarks()
            _display_stress_results(target, results)
            
        except Exception as e:
            console.print(f"❌ Stress test failed: {str(e)}")


@app.command()
def validate(
    critical_only: bool = typer.Option(False, "--critical", "-c", help="Run only critical validation tests"),
    save_report: bool = typer.Option(True, "--save/--no-save", help="Save validation report")
):
    """Run validation tests for trading readiness."""
    
    console.print("\n✅ [bold green]Trading Readiness Validation[/bold green]")
    console.print("=" * 50)
    
    config = BenchmarkConfig(
        stress_iterations=100 if critical_only else 1000,
        save_detailed_logs=save_report
    )
    
    validator = ValidationBenchmark(config)
    
    console.print("🔍 Running validation tests...")
    if critical_only:
        console.print("⚠️  Critical tests only")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[green]Validating system...", total=1)
        
        try:
            results = validator.run_benchmarks()
            progress.update(task, completed=1)
            
            _display_validation_results(results, critical_only)
            
        except Exception as e:
            console.print(f"❌ Validation failed: {str(e)}")


@app.command()
def latency(
    test_type: str = typer.Option("realtime", "--type", "-t", help="Latency test type (realtime, burst, concurrent)"),
    frequency: int = typer.Option(100, "--frequency", "-f", help="Target frequency in Hz for realtime tests")
):
    """Run latency benchmarks for high-frequency trading."""
    
    console.print(f"\n⚡ [bold yellow]Latency Testing - {test_type.title()}[/bold yellow]")
    console.print("=" * 50)
    
    config = BenchmarkConfig(
        stress_iterations=frequency * 10,  # 10 seconds worth
        max_execution_time=60.0
    )
    
    latency_tester = LatencyBenchmark(config)
    
    console.print(f"🎯 Test Type: {test_type}")
    if test_type == "realtime":
        console.print(f"📡 Target Frequency: {frequency}Hz")
    console.print()
    
    with console.status(f"[bold yellow]Running {test_type} latency test..."):
        try:
            results = latency_tester.run_benchmarks()
            _display_latency_results(test_type, results)
            
        except Exception as e:
            console.print(f"❌ Latency test failed: {str(e)}")


@app.command()
def memory(
    test_type: str = typer.Option("efficiency", "--type", "-t", help="Memory test type (efficiency, leaks, gc)"),
    profile_detailed: bool = typer.Option(False, "--detailed", "-d", help="Run detailed memory profiling")
):
    """Run memory profiling and optimization tests."""
    
    console.print(f"\n💾 [bold cyan]Memory Profiling - {test_type.title()}[/bold cyan]")
    console.print("=" * 50)
    
    config = BenchmarkConfig(
        stress_iterations=500 if profile_detailed else 100,
        save_detailed_logs=profile_detailed
    )
    
    memory_profiler = MemoryProfiler(config)
    
    console.print(f"🎯 Test Type: {test_type}")
    if profile_detailed:
        console.print("🔍 Detailed profiling enabled")
    console.print()
    
    with console.status(f"[bold cyan]Running {test_type} memory test..."):
        try:
            results = memory_profiler.run_benchmarks()
            _display_memory_results(test_type, results)
            
        except Exception as e:
            console.print(f"❌ Memory test failed: {str(e)}")


@app.command()
def report(
    results_dir: Path = typer.Option(Path("benchmark_results"), "--dir", "-d", help="Results directory"),
    format: str = typer.Option("table", "--format", "-f", help="Report format (table, json, summary)")
):
    """Generate benchmark report from previous results."""
    
    console.print("\n📊 [bold blue]Benchmark Report Generator[/bold blue]")
    console.print("=" * 50)
    
    if not results_dir.exists():
        console.print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Look for result files
    result_files = list(results_dir.glob("*results*.json"))
    
    if not result_files:
        console.print(f"❌ No result files found in: {results_dir}")
        return
    
    console.print(f"📁 Found {len(result_files)} result files")
    
    for result_file in result_files:
        console.print(f"📄 Loading: {result_file.name}")
        
        try:
            with open(result_file) as f:
                data = json.load(f)
            
            if format == "table":
                _display_results_table(result_file.stem, data)
            elif format == "json":
                console.print(Syntax(json.dumps(data, indent=2), "json"))
            else:  # summary
                _display_results_summary(result_file.stem, data)
                
        except Exception as e:
            console.print(f"❌ Error loading {result_file}: {str(e)}")


def _display_suite_results(suite_name: str, results: dict):
    """Display detailed results for a benchmark suite."""
    table = Table(title=f"{suite_name.title()} Suite Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Details", style="white")
    
    if 'results' in results:
        for result in results['results']:
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            duration = f"{result.get('duration', 0):.2f}s"
            warnings = len(result.get('warnings', []))
            details = f"{warnings} warnings" if warnings > 0 else "OK"
            
            table.add_row(
                result.get('name', 'Unknown'),
                status,
                duration,
                details
            )
    
    console.print(table)
    console.print()


def _display_benchmark_summary(all_results: dict, total_time: float):
    """Display comprehensive benchmark summary."""
    console.print("\n🎯 [bold blue]Benchmark Summary[/bold blue]")
    console.print("=" * 60)
    
    summary_table = Table()
    summary_table.add_column("Suite", style="cyan")
    summary_table.add_column("Tests", style="white")
    summary_table.add_column("Passed", style="green")
    summary_table.add_column("Failed", style="red")
    summary_table.add_column("Warnings", style="yellow")
    summary_table.add_column("Success Rate", style="bright_blue")
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_warnings = 0
    
    for suite_name, results in all_results.items():
        if 'summary' in results:
            summary = results['summary']
            tests = summary.get('total_benchmarks', 0)
            passed = summary.get('passed', 0)
            failed = summary.get('failed', 0)
            warnings = summary.get('warnings', 0)
            success_rate = summary.get('success_rate', 0) * 100
            
            total_tests += tests
            total_passed += passed
            total_failed += failed
            total_warnings += warnings
            
            summary_table.add_row(
                suite_name.title(),
                str(tests),
                str(passed),
                str(failed),
                str(warnings),
                f"{success_rate:.1f}%"
            )
    
    console.print(summary_table)
    
    # Overall summary
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    console.print(f"\n📈 [bold]Overall Results:[/bold]")
    console.print(f"   Total Tests: {total_tests}")
    console.print(f"   Passed: {total_passed}")
    console.print(f"   Failed: {total_failed}")
    console.print(f"   Warnings: {total_warnings}")
    console.print(f"   Success Rate: {overall_success_rate:.1f}%")
    console.print(f"   Total Time: {total_time:.1f}s")
    
    # Trading readiness assessment
    if overall_success_rate >= 95 and total_failed == 0:
        status = "[bold green]✅ READY FOR LIVE TRADING[/bold green]"
    elif overall_success_rate >= 90 and total_failed <= 2:
        status = "[bold yellow]⚠️  READY FOR PAPER TRADING[/bold yellow]"
    else:
        status = "[bold red]❌ NOT READY - REQUIRES FIXES[/bold red]"
    
    console.print(f"\n🚨 [bold]Trading Readiness:[/bold] {status}")


def _display_quick_results(component: str, result):
    """Display quick test results."""
    console.print(f"\n📊 [bold]{component} Results:[/bold]")
    
    if hasattr(result, 'passed') and result.passed:
        console.print("✅ [green]PASSED[/green]")
        if hasattr(result, 'duration'):
            console.print(f"⏱️  Duration: {result.duration:.3f}s")
        if hasattr(result, 'warnings') and result.warnings:
            console.print(f"⚠️  Warnings: {len(result.warnings)}")
    else:
        console.print("❌ [red]FAILED[/red]")
        if hasattr(result, 'error_message'):
            console.print(f"Error: {result.error_message}")


def _display_stress_results(target: str, results: dict):
    """Display stress test results."""
    console.print(f"\n💥 [bold]Stress Test Results - {target.title()}:[/bold]")
    
    if 'summary' in results:
        summary = results['summary']
        console.print(f"✅ Passed: {summary.get('passed', 0)}")
        console.print(f"❌ Failed: {summary.get('failed', 0)}")
        console.print(f"⚠️  Warnings: {summary.get('warnings', 0)}")
        console.print(f"📈 Success Rate: {summary.get('success_rate', 0) * 100:.1f}%")


def _display_validation_results(results: dict, critical_only: bool):
    """Display validation test results."""
    console.print(f"\n✅ [bold]Validation Results:{' (Critical Only)' if critical_only else ''}[/bold]")
    
    if 'summary' in results:
        summary = results['summary']
        total = summary.get('total_benchmarks', 0)
        passed = summary.get('passed', 0)
        failed = summary.get('failed', 0)
        
        console.print(f"📊 Tests: {passed}/{total} passed")
        
        if failed == 0:
            console.print("🎉 [bold green]ALL VALIDATION TESTS PASSED![/bold green]")
            console.print("✅ System is ready for trading")
        else:
            console.print(f"❌ [bold red]{failed} tests failed[/bold red]")
            console.print("⚠️  System requires fixes before trading")


def _display_latency_results(test_type: str, results: dict):
    """Display latency test results."""
    console.print(f"\n⚡ [bold]Latency Results - {test_type.title()}:[/bold]")
    
    if 'summary' in results:
        summary = results['summary']
        console.print(f"📊 Tests completed: {summary.get('total_benchmarks', 0)}")
        console.print(f"✅ Success rate: {summary.get('success_rate', 0) * 100:.1f}%")


def _display_memory_results(test_type: str, results: dict):
    """Display memory test results."""
    console.print(f"\n💾 [bold]Memory Results - {test_type.title()}:[/bold]")
    
    if 'summary' in results:
        summary = results['summary']
        console.print(f"📊 Tests completed: {summary.get('total_benchmarks', 0)}")
        console.print(f"✅ Success rate: {summary.get('success_rate', 0) * 100:.1f}%")


def _display_results_table(name: str, data: dict):
    """Display results in table format."""
    console.print(f"\n📊 [bold]{name.title()}[/bold]")
    
    if 'summary' in data:
        summary = data['summary']
        
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in summary.items():
            if isinstance(value, float):
                value = f"{value:.2f}"
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)


def _display_results_summary(name: str, data: dict):
    """Display results summary."""
    console.print(f"\n📄 [bold]{name.title()} Summary[/bold]")
    
    if 'summary' in data:
        summary = data['summary']
        console.print(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        console.print(f"Success Rate: {summary.get('success_rate', 0) * 100:.1f}%")
        console.print(f"Duration: {summary.get('total_duration', 0):.1f}s")


if __name__ == "__main__":
    app()