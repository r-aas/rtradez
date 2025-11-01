"""
Core benchmarking framework for RTradez.

Provides base classes and infrastructure for comprehensive performance testing,
stress testing, and validation before live trading deployment.
"""

import time
import sys
import psutil
import gc
import tracemalloc
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BenchmarkStatus(Enum):
    """Benchmark execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BenchmarkSeverity(Enum):
    """Benchmark failure severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    cpu_count: int
    cpu_freq: float
    memory_total: int
    memory_available: int
    disk_usage: Dict[str, Any]
    python_version: str
    platform: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during benchmarks."""
    execution_time: float
    memory_peak: int
    memory_average: int
    cpu_usage_peak: float
    cpu_usage_average: float
    throughput: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    error_rate: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Result of a single benchmark execution."""
    name: str
    description: str
    status: BenchmarkStatus
    severity: BenchmarkSeverity = BenchmarkSeverity.INFO
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    metrics: Optional[PerformanceMetrics] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    system_info: Optional[SystemInfo] = None
    passed: bool = False
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""
    
    # Execution settings
    max_execution_time: float = Field(300.0, description="Maximum execution time in seconds")
    memory_limit_mb: int = Field(4096, description="Memory limit in MB")
    cpu_timeout: float = Field(60.0, description="CPU timeout for operations")
    
    # Stress testing settings
    stress_iterations: int = Field(1000, description="Number of stress test iterations")
    concurrent_operations: int = Field(10, description="Concurrent operations for load testing")
    data_scale_factors: List[float] = Field([1.0, 10.0, 100.0], description="Data scaling factors")
    
    # Performance thresholds
    max_latency_ms: float = Field(100.0, description="Maximum acceptable latency in ms")
    min_throughput: float = Field(100.0, description="Minimum operations per second")
    max_memory_growth_mb: float = Field(100.0, description="Maximum memory growth in MB")
    max_error_rate: float = Field(0.01, description="Maximum error rate (1%)")
    
    # Output settings
    save_detailed_logs: bool = Field(True, description="Save detailed benchmark logs")
    output_directory: Path = Field(Path("benchmark_results"), description="Output directory")
    generate_plots: bool = Field(True, description="Generate performance plots")


class BenchmarkSuite:
    """Main benchmark suite orchestrator."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.benchmarks: List[Callable] = []
        self.setup_functions: List[Callable] = []
        self.teardown_functions: List[Callable] = []
        
        # Ensure output directory exists
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup detailed logging for benchmarks."""
        log_file = self.config.output_directory / "benchmark.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
    
    def register_benchmark(self, 
                          name: str, 
                          description: str, 
                          severity: BenchmarkSeverity = BenchmarkSeverity.INFO):
        """Decorator to register a benchmark function."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                return self._execute_benchmark(func, name, description, severity, *args, **kwargs)
            wrapper.benchmark_name = name
            wrapper.benchmark_description = description
            wrapper.benchmark_severity = severity
            self.benchmarks.append(wrapper)
            return wrapper
        return decorator
    
    def add_setup(self, func: Callable):
        """Add a setup function to run before benchmarks."""
        self.setup_functions.append(func)
    
    def add_teardown(self, func: Callable):
        """Add a teardown function to run after benchmarks."""
        self.teardown_functions.append(func)
    
    def _get_system_info(self) -> SystemInfo:
        """Collect current system information."""
        return SystemInfo(
            cpu_count=psutil.cpu_count(),
            cpu_freq=psutil.cpu_freq().current if psutil.cpu_freq() else 0.0,
            memory_total=psutil.virtual_memory().total,
            memory_available=psutil.virtual_memory().available,
            disk_usage=dict(psutil.disk_usage('/')._asdict()),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            platform=sys.platform
        )
    
    def _monitor_performance(self, func: Callable, *args, **kwargs) -> tuple:
        """Monitor performance metrics during function execution."""
        # Start monitoring
        tracemalloc.start()
        process = psutil.Process()
        
        start_memory = process.memory_info().rss
        start_time = time.perf_counter()
        start_cpu_time = process.cpu_times()
        
        cpu_samples = []
        memory_samples = []
        latency_samples = []
        
        # Execute function with monitoring
        try:
            operation_start = time.perf_counter()
            result = func(*args, **kwargs)
            operation_end = time.perf_counter()
            latency_samples.append((operation_end - operation_start) * 1000)  # ms
            
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Collect final metrics
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss
        end_cpu_time = process.cpu_times()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_growth = end_memory - start_memory
        cpu_usage = ((end_cpu_time.user + end_cpu_time.system) - 
                    (start_cpu_time.user + start_cpu_time.system)) / execution_time * 100
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_peak=peak,
            memory_average=current,
            cpu_usage_peak=cpu_usage,
            cpu_usage_average=cpu_usage,
            latency_p50=np.percentile(latency_samples, 50) if latency_samples else None,
            latency_p95=np.percentile(latency_samples, 95) if latency_samples else None,
            latency_p99=np.percentile(latency_samples, 99) if latency_samples else None,
            error_rate=0.0 if success else 1.0
        )
        
        return result, metrics, success, error
    
    def _execute_benchmark(self, 
                          func: Callable, 
                          name: str, 
                          description: str, 
                          severity: BenchmarkSeverity,
                          *args, **kwargs) -> BenchmarkResult:
        """Execute a single benchmark with full monitoring."""
        
        logger.info(f"Starting benchmark: {name}")
        
        start_time = datetime.now()
        result = BenchmarkResult(
            name=name,
            description=description,
            status=BenchmarkStatus.RUNNING,
            severity=severity,
            start_time=start_time,
            system_info=self._get_system_info()
        )
        
        try:
            # Execute with performance monitoring
            func_result, metrics, success, error = self._monitor_performance(func, *args, **kwargs)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update result
            result.end_time = end_time
            result.duration = duration
            result.metrics = metrics
            result.passed = success
            result.status = BenchmarkStatus.COMPLETED if success else BenchmarkStatus.FAILED
            
            if error:
                result.error_message = error
            
            # Validate against thresholds
            self._validate_thresholds(result)
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            result.end_time = datetime.now()
            result.duration = (result.end_time - start_time).total_seconds()
            
            logger.error(f"Benchmark {name} failed: {e}")
        
        logger.info(f"Completed benchmark: {name} - Status: {result.status.value}")
        return result
    
    def _validate_thresholds(self, result: BenchmarkResult):
        """Validate benchmark results against configured thresholds."""
        if not result.metrics:
            return
        
        warnings = []
        
        # Check execution time
        if result.duration and result.duration > self.config.max_execution_time:
            warnings.append(f"Execution time {result.duration:.2f}s exceeds limit {self.config.max_execution_time}s")
        
        # Check memory usage
        if result.metrics.memory_peak > self.config.memory_limit_mb * 1024 * 1024:
            warnings.append(f"Memory usage {result.metrics.memory_peak / 1024 / 1024:.1f}MB exceeds limit {self.config.memory_limit_mb}MB")
        
        # Check latency
        if result.metrics.latency_p95 and result.metrics.latency_p95 > self.config.max_latency_ms:
            warnings.append(f"P95 latency {result.metrics.latency_p95:.1f}ms exceeds limit {self.config.max_latency_ms}ms")
        
        # Check error rate
        if result.metrics.error_rate > self.config.max_error_rate:
            warnings.append(f"Error rate {result.metrics.error_rate:.1%} exceeds limit {self.config.max_error_rate:.1%}")
        
        result.warnings = warnings
        
        # Upgrade severity if critical thresholds exceeded
        if warnings and result.severity == BenchmarkSeverity.INFO:
            result.severity = BenchmarkSeverity.WARNING
    
    def run_all(self) -> Dict[str, Any]:
        """Run all registered benchmarks."""
        logger.info(f"Starting benchmark suite with {len(self.benchmarks)} benchmarks")
        
        start_time = datetime.now()
        
        # Run setup functions
        for setup_func in self.setup_functions:
            try:
                setup_func()
            except Exception as e:
                logger.error(f"Setup function failed: {e}")
        
        # Execute benchmarks
        for benchmark in self.benchmarks:
            result = benchmark()
            self.results.append(result)
        
        # Run teardown functions
        for teardown_func in self.teardown_functions:
            try:
                teardown_func()
            except Exception as e:
                logger.error(f"Teardown function failed: {e}")
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = self._generate_summary(total_duration)
        
        # Save results
        if self.config.save_detailed_logs:
            self._save_results(summary)
        
        logger.info(f"Benchmark suite completed in {total_duration:.2f}s")
        
        return summary
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate benchmark summary report."""
        total_benchmarks = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        warnings = sum(len(r.warnings) for r in self.results)
        
        critical_failures = [r for r in self.results 
                           if r.severity == BenchmarkSeverity.CRITICAL and not r.passed]
        
        return {
            'summary': {
                'total_benchmarks': total_benchmarks,
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'success_rate': passed / total_benchmarks if total_benchmarks > 0 else 0,
                'total_duration': total_duration,
                'critical_failures': len(critical_failures)
            },
            'results': [r.dict() for r in self.results],
            'critical_failures': [r.dict() for r in critical_failures],
            'timestamp': datetime.now().isoformat(),
            'config': self.config.dict()
        }
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save benchmark results to files."""
        # Save JSON summary
        json_file = self.config.output_directory / "benchmark_results.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save CSV for analysis
        if self.results:
            df = pd.DataFrame([{
                'name': r.name,
                'status': r.status.value,
                'duration': r.duration,
                'passed': r.passed,
                'warnings': len(r.warnings),
                'memory_peak_mb': r.metrics.memory_peak / 1024 / 1024 if r.metrics else None,
                'cpu_usage': r.metrics.cpu_usage_average if r.metrics else None,
                'latency_p95': r.metrics.latency_p95 if r.metrics else None
            } for r in self.results])
            
            csv_file = self.config.output_directory / "benchmark_results.csv"
            df.to_csv(csv_file, index=False)
        
        logger.info(f"Benchmark results saved to {self.config.output_directory}")


class ComponentBenchmark:
    """Base class for component-specific benchmarks."""
    
    def __init__(self, name: str, config: BenchmarkConfig):
        self.name = name
        self.config = config
        self.suite = BenchmarkSuite(config)
    
    def setup(self):
        """Override for component-specific setup."""
        pass
    
    def teardown(self):
        """Override for component-specific teardown."""
        pass
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks for this component."""
        self.suite.add_setup(self.setup)
        self.suite.add_teardown(self.teardown)
        return self.suite.run_all()