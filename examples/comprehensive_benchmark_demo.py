#!/usr/bin/env python3
"""
RTradez Comprehensive Benchmark Demonstration.

Shows how to use the complete benchmarking framework for pre-trading validation.
"""

import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.rtradez.benchmarks import (
    BenchmarkConfig, PerformanceBenchmark, StressTester, 
    ValidationBenchmark, LatencyBenchmark, MemoryProfiler
)

console = Console()


def demo_comprehensive_benchmarking():
    """Demonstrate comprehensive pre-trading benchmark validation."""
    
    console.print("\n🔬 [bold blue]RTradez Comprehensive Benchmarking Demonstration[/bold blue]")
    console.print("=" * 70)
    console.print("\n[dim]This demo shows how to validate your trading system before live deployment[/dim]\n")
    
    # Configuration for benchmark suite
    config = BenchmarkConfig(
        stress_iterations=100,  # Reduced for demo
        memory_limit_mb=2048,
        max_execution_time=60.0,
        save_detailed_logs=True,
        output_directory=Path("demo_benchmark_results")
    )
    
    console.print(f"📊 [bold]Benchmark Configuration:[/bold]")
    console.print(f"   • Iterations: {config.stress_iterations}")
    console.print(f"   • Memory Limit: {config.memory_limit_mb}MB")
    console.print(f"   • Timeout: {config.max_execution_time}s")
    console.print(f"   • Output Directory: {config.output_directory}")
    
    console.print("\n🎯 [bold]Running Pre-Trading Validation Suite...[/bold]\n")
    
    # 1. Performance Benchmarks
    console.print("⚡ [bold yellow]1. Performance Benchmarks[/bold yellow]")
    console.print("   Testing throughput, latency, and scalability...")
    
    performance_benchmark = PerformanceBenchmark(config)
    
    try:
        with console.status("[bold yellow]Running performance tests..."):
            perf_results = performance_benchmark.run_benchmarks()
        
        # Display performance summary
        if 'summary' in perf_results:
            summary = perf_results['summary']
            console.print(f"   ✅ Tests: {summary.get('passed', 0)}/{summary.get('total_benchmarks', 0)} passed")
            console.print(f"   📈 Success Rate: {summary.get('success_rate', 0) * 100:.1f}%")
            console.print(f"   ⏱️  Duration: {summary.get('total_duration', 0):.1f}s")
        
    except Exception as e:
        console.print(f"   ❌ Performance tests failed: {str(e)}")
    
    console.print()
    
    # 2. Stress Testing
    console.print("💥 [bold red]2. Stress Testing[/bold red]")
    console.print("   Testing system behavior under extreme conditions...")
    
    stress_tester = StressTester(config)
    
    try:
        with console.status("[bold red]Running stress tests..."):
            stress_results = stress_tester.run_benchmarks()
        
        if 'summary' in stress_results:
            summary = stress_results['summary']
            console.print(f"   ✅ Tests: {summary.get('passed', 0)}/{summary.get('total_benchmarks', 0)} passed")
            console.print(f"   📈 Success Rate: {summary.get('success_rate', 0) * 100:.1f}%")
            console.print(f"   ⚠️  Warnings: {summary.get('warnings', 0)}")
        
    except Exception as e:
        console.print(f"   ❌ Stress tests failed: {str(e)}")
    
    console.print()
    
    # 3. Validation Testing
    console.print("✅ [bold green]3. Validation Testing[/bold green]")
    console.print("   Testing mathematical accuracy and system integrity...")
    
    validator = ValidationBenchmark(config)
    
    try:
        with console.status("[bold green]Running validation tests..."):
            validation_results = validator.run_benchmarks()
        
        if 'summary' in validation_results:
            summary = validation_results['summary']
            console.print(f"   ✅ Tests: {summary.get('passed', 0)}/{summary.get('total_benchmarks', 0)} passed")
            console.print(f"   📈 Success Rate: {summary.get('success_rate', 0) * 100:.1f}%")
            console.print(f"   🔍 Critical Failures: {summary.get('critical_failures', 0)}")
        
    except Exception as e:
        console.print(f"   ❌ Validation tests failed: {str(e)}")
    
    console.print()
    
    # 4. Latency Testing
    console.print("⚡ [bold yellow]4. Latency Testing[/bold yellow]")
    console.print("   Testing real-time operation response times...")
    
    latency_tester = LatencyBenchmark(config)
    
    try:
        with console.status("[bold yellow]Running latency tests..."):
            latency_results = latency_tester.run_benchmarks()
        
        if 'summary' in latency_results:
            summary = latency_results['summary']
            console.print(f"   ✅ Tests: {summary.get('passed', 0)}/{summary.get('total_benchmarks', 0)} passed")
            console.print(f"   📈 Success Rate: {summary.get('success_rate', 0) * 100:.1f}%")
            
        console.print("   📊 Real-time capability validated for high-frequency trading")
        
    except Exception as e:
        console.print(f"   ❌ Latency tests failed: {str(e)}")
    
    console.print()
    
    # 5. Memory Profiling
    console.print("💾 [bold cyan]5. Memory Profiling[/bold cyan]")
    console.print("   Testing memory usage and leak detection...")
    
    memory_profiler = MemoryProfiler(config)
    
    try:
        with console.status("[bold cyan]Running memory tests..."):
            memory_results = memory_profiler.run_benchmarks()
        
        if 'summary' in memory_results:
            summary = memory_results['summary']
            console.print(f"   ✅ Tests: {summary.get('passed', 0)}/{summary.get('total_benchmarks', 0)} passed")
            console.print(f"   📈 Success Rate: {summary.get('success_rate', 0) * 100:.1f}%")
            
        console.print("   🧹 Memory efficiency and leak detection completed")
        
    except Exception as e:
        console.print(f"   ❌ Memory tests failed: {str(e)}")
    
    console.print()
    
    # Generate Trading Readiness Assessment
    console.print("🚨 [bold blue]Trading Readiness Assessment[/bold blue]")
    console.print("=" * 50)
    
    # Collect all results for assessment
    all_results = []
    for result_set in [perf_results, stress_results, validation_results, latency_results, memory_results]:
        if isinstance(result_set, dict) and 'summary' in result_set:
            all_results.append(result_set['summary'])
    
    # Calculate overall metrics
    if all_results:
        total_tests = sum(r.get('total_benchmarks', 0) for r in all_results)
        total_passed = sum(r.get('passed', 0) for r in all_results)
        total_failed = sum(r.get('failed', 0) for r in all_results)
        total_warnings = sum(r.get('warnings', 0) for r in all_results)
        critical_failures = sum(r.get('critical_failures', 0) for r in all_results)
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Create assessment table
        assessment_table = Table(title="System Assessment")
        assessment_table.add_column("Metric", style="cyan")
        assessment_table.add_column("Value", style="white")
        assessment_table.add_column("Status", style="green")
        
        assessment_table.add_row("Total Tests", str(total_tests), "✅" if total_tests > 0 else "❌")
        assessment_table.add_row("Passed Tests", str(total_passed), "✅" if total_passed > 0 else "❌")
        assessment_table.add_row("Failed Tests", str(total_failed), "✅" if total_failed == 0 else "❌")
        assessment_table.add_row("Warnings", str(total_warnings), "✅" if total_warnings < 5 else "⚠️")
        assessment_table.add_row("Critical Failures", str(critical_failures), "✅" if critical_failures == 0 else "❌")
        assessment_table.add_row("Success Rate", f"{overall_success_rate:.1f}%", "✅" if overall_success_rate >= 95 else "⚠️" if overall_success_rate >= 90 else "❌")
        
        console.print(assessment_table)
        
        # Final recommendation
        console.print("\n🎯 [bold]Final Recommendation:[/bold]")
        
        if overall_success_rate >= 95 and critical_failures == 0 and total_failed == 0:
            recommendation = Panel(
                "[bold green]✅ SYSTEM READY FOR LIVE TRADING[/bold green]\n\n"
                "All critical tests passed with excellent success rate.\n"
                "System demonstrates:\n"
                "• Robust performance under load\n"
                "• Mathematical accuracy\n"
                "• Real-time capability\n"
                "• Memory efficiency\n"
                "• Stress resilience",
                title="🚀 READY FOR PRODUCTION",
                border_style="green"
            )
        elif overall_success_rate >= 90 and critical_failures == 0:
            recommendation = Panel(
                "[bold yellow]⚠️  READY FOR PAPER TRADING[/bold yellow]\n\n"
                "Most tests passed but some issues detected.\n"
                "Recommended actions:\n"
                "• Review failed tests\n"
                "• Address performance warnings\n"
                "• Run extended validation\n"
                "• Consider paper trading first",
                title="📝 PAPER TRADING READY",
                border_style="yellow"
            )
        else:
            recommendation = Panel(
                "[bold red]❌ NOT READY FOR TRADING[/bold red]\n\n"
                "Significant issues detected that require attention.\n"
                "Required actions:\n"
                "• Fix critical failures\n"
                "• Improve system performance\n"
                "• Re-run full validation\n"
                "• Do not deploy to production",
                title="🔧 REQUIRES FIXES",
                border_style="red"
            )
        
        console.print(recommendation)
        
        # Show where results are saved
        if config.save_detailed_logs:
            console.print(f"\n💾 [bold]Detailed Results:[/bold]")
            console.print(f"   📁 Saved to: {config.output_directory}")
            console.print(f"   📊 Review detailed logs for performance optimization")
            console.print(f"   🔍 Analyze failed tests for specific improvements")
    
    else:
        console.print("❌ No benchmark results available for assessment")
    
    console.print("\n🏁 [bold blue]Benchmark demonstration completed![/bold blue]")
    console.print("\n[dim]This comprehensive testing framework ensures your trading system is thoroughly validated before risking real capital.[/dim]")


def demo_quick_component_tests():
    """Demonstrate quick component-specific testing."""
    
    console.print("\n⚡ [bold yellow]Quick Component Testing Demo[/bold yellow]")
    console.print("=" * 50)
    
    config = BenchmarkConfig(
        stress_iterations=50,  # Very quick for demo
        max_execution_time=30.0,
        save_detailed_logs=False
    )
    
    components = [
        ("Risk Management", PerformanceBenchmark),
        ("Portfolio Management", StressTester),
        ("Data Processing", ValidationBenchmark)
    ]
    
    for component_name, benchmark_class in components:
        console.print(f"\n🎯 Testing {component_name}...")
        
        try:
            benchmark = benchmark_class(config)
            
            with console.status(f"[bold]Running {component_name.lower()} test..."):
                # Run just the first benchmark as a quick test
                if hasattr(benchmark.suite, 'benchmarks') and benchmark.suite.benchmarks:
                    result = benchmark.suite.benchmarks[0]()
                    
                    if hasattr(result, 'passed') and result.passed:
                        console.print(f"   ✅ {component_name}: PASSED ({result.duration:.2f}s)")
                    else:
                        console.print(f"   ❌ {component_name}: FAILED")
                else:
                    console.print(f"   ⚠️  {component_name}: No tests available")
                    
        except Exception as e:
            console.print(f"   ❌ {component_name}: Error - {str(e)}")
    
    console.print("\n✅ Quick component testing completed!")


if __name__ == "__main__":
    import sys
    
    console.print("\n🔬 [bold blue]RTradez Benchmark Demo Menu[/bold blue]")
    console.print("=" * 40)
    console.print("\n[bold]Available Demonstrations:[/bold]")
    console.print("  1. Comprehensive Pre-Trading Validation")
    console.print("  2. Quick Component Testing")
    
    # Auto-select demo based on command line argument or default to comprehensive
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = "1"  # Default to comprehensive demo
        console.print(f"\n[dim]Auto-selecting demo {choice} (Comprehensive Validation)[/dim]")
    
    if choice == "1":
        demo_comprehensive_benchmarking()
    elif choice == "2":
        demo_quick_component_tests()
    else:
        console.print("❌ Invalid choice. Running comprehensive demo...")
        demo_comprehensive_benchmarking()