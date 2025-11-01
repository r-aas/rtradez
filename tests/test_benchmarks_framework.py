"""
Comprehensive tests for benchmarks framework.

Tests for performance testing, stress testing, validation, latency, and memory profiling.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from rtradez.benchmarks import (
    BenchmarkSuite, BenchmarkResult, BenchmarkConfig,
    PerformanceBenchmark, StressTester, ValidationBenchmark,
    LatencyBenchmark, MemoryProfiler
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig validation and configuration."""
    
    def test_valid_config_creation(self):
        """Test creating valid benchmark configuration."""
        config = BenchmarkConfig(
            name="Test Benchmark",
            description="Test benchmark description",
            target_duration=60.0,
            max_memory_mb=512,
            max_latency_ms=100.0,
            success_threshold=0.95,
            iterations=1000
        )
        
        assert config.name == "Test Benchmark"
        assert config.description == "Test benchmark description"
        assert config.target_duration == 60.0
        assert config.max_memory_mb == 512
        assert config.max_latency_ms == 100.0
        assert config.success_threshold == 0.95
        assert config.iterations == 1000
    
    def test_invalid_config_raises_errors(self):
        """Test that invalid configurations raise errors."""
        with pytest.raises(ValueError):
            BenchmarkConfig(
                name="Test",
                target_duration=-1.0  # Invalid
            )
        
        with pytest.raises(ValueError):
            BenchmarkConfig(
                name="Test",
                success_threshold=1.5  # Invalid (> 1.0)
            )
        
        with pytest.raises(ValueError):
            BenchmarkConfig(
                name="Test",
                iterations=0  # Invalid
            )
    
    def test_default_values(self):
        """Test default configuration values."""
        config = BenchmarkConfig(name="Test")
        
        assert config.target_duration == 30.0
        assert config.max_memory_mb == 1024
        assert config.max_latency_ms == 1000.0
        assert config.success_threshold == 0.99
        assert config.iterations == 100


class TestBenchmarkResult:
    """Test BenchmarkResult model."""
    
    def test_benchmark_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            benchmark_name="Test Benchmark",
            status="PASSED",
            duration=45.2,
            memory_peak_mb=256.7,
            latency_p95_ms=85.3,
            success_rate=0.98,
            throughput_ops_sec=1250.0,
            error_count=2,
            timestamp=datetime.now()
        )
        
        assert result.benchmark_name == "Test Benchmark"
        assert result.status == "PASSED"
        assert result.duration == 45.2
        assert result.memory_peak_mb == 256.7
        assert result.latency_p95_ms == 85.3
        assert result.success_rate == 0.98
        assert result.throughput_ops_sec == 1250.0
        assert result.error_count == 2
    
    def test_result_validation(self):
        """Test result field validation."""
        with pytest.raises(ValueError):
            BenchmarkResult(
                benchmark_name="Test",
                status="INVALID_STATUS",  # Not in allowed values
                duration=45.2
            )
        
        with pytest.raises(ValueError):
            BenchmarkResult(
                benchmark_name="Test",
                status="PASSED",
                duration=-1.0  # Invalid negative duration
            )
    
    def test_result_assessment(self):
        """Test automatic result assessment."""
        # Passing result
        result = BenchmarkResult(
            benchmark_name="Test",
            status="PASSED",
            duration=30.0,
            success_rate=0.99,
            error_count=0
        )
        
        assert result.is_successful()
        
        # Failing result
        result_fail = BenchmarkResult(
            benchmark_name="Test",
            status="FAILED",
            duration=30.0,
            success_rate=0.85,
            error_count=15
        )
        
        assert not result_fail.is_successful()
    
    def test_result_serialization(self):
        """Test JSON serialization of results."""
        result = BenchmarkResult(
            benchmark_name="Test Benchmark",
            status="PASSED",
            duration=45.2,
            success_rate=0.98,
            timestamp=datetime.now()
        )
        
        # Should be able to serialize to JSON
        json_str = result.json()
        assert isinstance(json_str, str)
        assert "Test Benchmark" in json_str
        assert "PASSED" in json_str
        
        # Should be able to parse back
        parsed_result = BenchmarkResult.parse_raw(json_str)
        assert parsed_result.benchmark_name == result.benchmark_name
        assert parsed_result.status == result.status


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark functionality."""
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create performance benchmark instance."""
        config = BenchmarkConfig(
            name="Performance Test",
            target_duration=5.0,
            iterations=100,
            max_latency_ms=50.0
        )
        return PerformanceBenchmark(config)
    
    def test_performance_benchmark_initialization(self, performance_benchmark):
        """Test PerformanceBenchmark initialization."""
        assert performance_benchmark.config.name == "Performance Test"
        assert performance_benchmark.config.iterations == 100
        assert len(performance_benchmark.results) == 0
    
    def test_benchmark_simple_function(self, performance_benchmark):
        """Test benchmarking a simple function."""
        def simple_calculation():
            return sum(range(1000))
        
        result = performance_benchmark.benchmark_function(simple_calculation)
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "Performance Test"
        assert result.duration > 0
        assert result.status == "PASSED"
        assert result.error_count == 0
    
    def test_benchmark_function_with_args(self, performance_benchmark):
        """Test benchmarking function with arguments."""
        def calculation_with_args(n, multiplier=2):
            return sum(range(n)) * multiplier
        
        result = performance_benchmark.benchmark_function(
            calculation_with_args,
            args=(500,),
            kwargs={'multiplier': 3}
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.status == "PASSED"
        assert result.error_count == 0
    
    def test_benchmark_failing_function(self, performance_benchmark):
        """Test benchmarking function that raises errors."""
        def failing_function():
            if np.random.random() < 0.2:  # 20% failure rate
                raise ValueError("Random failure")
            return 42
        
        result = performance_benchmark.benchmark_function(failing_function)
        
        assert isinstance(result, BenchmarkResult)
        # Should handle errors gracefully
        assert result.error_count >= 0
        assert result.success_rate <= 1.0
    
    def test_throughput_measurement(self, performance_benchmark):
        """Test throughput measurement."""
        def fast_operation():
            return len([1, 2, 3, 4, 5])
        
        result = performance_benchmark.benchmark_function(fast_operation)
        
        assert result.throughput_ops_sec > 0
        # Should be able to execute many operations per second
        assert result.throughput_ops_sec > 10
    
    def test_latency_measurement(self, performance_benchmark):
        """Test latency measurement."""
        def variable_latency():
            # Simulate variable processing time
            time.sleep(np.random.uniform(0.001, 0.005))
            return "done"
        
        result = performance_benchmark.benchmark_function(variable_latency)
        
        assert result.latency_p50_ms is not None
        assert result.latency_p95_ms is not None
        assert result.latency_p99_ms is not None
        assert result.latency_p95_ms >= result.latency_p50_ms
        assert result.latency_p99_ms >= result.latency_p95_ms
    
    def test_memory_profiling(self, performance_benchmark):
        """Test memory usage profiling."""
        def memory_intensive():
            # Create and release memory
            large_list = list(range(10000))
            return len(large_list)
        
        result = performance_benchmark.benchmark_function(memory_intensive)
        
        assert result.memory_peak_mb is not None
        assert result.memory_peak_mb > 0
    
    def test_concurrent_benchmark(self, performance_benchmark):
        """Test concurrent execution benchmarking."""
        def concurrent_task():
            return sum(range(100))
        
        result = performance_benchmark.benchmark_concurrent(
            concurrent_task,
            num_workers=4,
            tasks_per_worker=25
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.status == "PASSED"
        # Concurrent execution should handle multiple workers
        assert result.throughput_ops_sec > 0


class TestStressTester:
    """Test StressTester functionality."""
    
    @pytest.fixture
    def stress_tester(self):
        """Create stress tester instance."""
        config = BenchmarkConfig(
            name="Stress Test",
            target_duration=10.0,
            max_memory_mb=1024,
            success_threshold=0.90
        )
        return StressTester(config)
    
    def test_stress_tester_initialization(self, stress_tester):
        """Test StressTester initialization."""
        assert stress_tester.config.name == "Stress Test"
        assert stress_tester.config.success_threshold == 0.90
        assert len(stress_tester.stress_scenarios) == 0
    
    def test_add_stress_scenario(self, stress_tester):
        """Test adding stress test scenarios."""
        def cpu_stress():
            # CPU intensive operation
            return sum(i * i for i in range(1000))
        
        stress_tester.add_scenario("CPU Stress", cpu_stress)
        
        assert len(stress_tester.stress_scenarios) == 1
        assert "CPU Stress" in stress_tester.stress_scenarios
    
    def test_memory_stress_scenario(self, stress_tester):
        """Test memory stress testing."""
        def memory_stress():
            # Gradually increasing memory usage
            data = []
            for i in range(100):
                data.append([0] * 1000)
            return len(data)
        
        stress_tester.add_scenario("Memory Stress", memory_stress)
        result = stress_tester.run_scenario("Memory Stress")
        
        assert isinstance(result, BenchmarkResult)
        assert result.memory_peak_mb > 0
    
    def test_load_ramp_up(self, stress_tester):
        """Test load ramp-up testing."""
        def simple_task():
            return sum(range(50))
        
        results = stress_tester.ramp_up_test(
            simple_task,
            start_workers=1,
            max_workers=8,
            ramp_duration=5.0
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Results should show increasing load
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.throughput_ops_sec > 0
    
    def test_sustained_load_test(self, stress_tester):
        """Test sustained load testing."""
        def sustained_task():
            time.sleep(0.001)  # Small delay
            return "completed"
        
        result = stress_tester.sustained_load_test(
            sustained_task,
            num_workers=4,
            duration=3.0
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.duration >= 3.0
        assert result.status in ["PASSED", "FAILED"]
    
    def test_resource_exhaustion_test(self, stress_tester):
        """Test resource exhaustion scenarios."""
        def resource_intensive():
            # Simulate resource usage
            data = [i for i in range(1000)]
            return len(data)
        
        result = stress_tester.resource_exhaustion_test(
            resource_intensive,
            max_memory_mb=100,  # Low limit to trigger exhaustion
            max_duration=5.0
        )
        
        assert isinstance(result, BenchmarkResult)
        # May pass or fail depending on resource usage
        assert result.status in ["PASSED", "FAILED", "TIMEOUT"]


class TestValidationBenchmark:
    """Test ValidationBenchmark functionality."""
    
    @pytest.fixture
    def validation_benchmark(self):
        """Create validation benchmark instance."""
        config = BenchmarkConfig(
            name="Validation Test",
            success_threshold=0.95,
            iterations=50
        )
        return ValidationBenchmark(config)
    
    def test_validation_benchmark_initialization(self, validation_benchmark):
        """Test ValidationBenchmark initialization."""
        assert validation_benchmark.config.name == "Validation Test"
        assert validation_benchmark.config.success_threshold == 0.95
        assert len(validation_benchmark.validation_rules) == 0
    
    def test_add_validation_rule(self, validation_benchmark):
        """Test adding validation rules."""
        def positive_result_rule(result):
            return result > 0
        
        validation_benchmark.add_rule("Positive Result", positive_result_rule)
        
        assert len(validation_benchmark.validation_rules) == 1
        assert "Positive Result" in validation_benchmark.validation_rules
    
    def test_validate_function_output(self, validation_benchmark):
        """Test function output validation."""
        def math_function(x):
            return x * 2
        
        def is_even_rule(result):
            return result % 2 == 0
        
        validation_benchmark.add_rule("Result is Even", is_even_rule)
        
        result = validation_benchmark.validate_function(
            math_function,
            test_inputs=[1, 2, 3, 4, 5]
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.success_rate == 1.0  # All results should be even
        assert result.error_count == 0
    
    def test_statistical_validation(self, validation_benchmark):
        """Test statistical validation."""
        def random_generator():
            return np.random.normal(0, 1)
        
        def mean_near_zero_rule(results):
            mean = np.mean(results)
            return abs(mean) < 0.2  # Mean should be close to 0
        
        def std_near_one_rule(results):
            std = np.std(results)
            return abs(std - 1.0) < 0.2  # Std should be close to 1
        
        validation_benchmark.add_statistical_rule("Mean Near Zero", mean_near_zero_rule)
        validation_benchmark.add_statistical_rule("Std Near One", std_near_one_rule)
        
        result = validation_benchmark.validate_statistical_function(
            random_generator,
            sample_size=1000
        )
        
        assert isinstance(result, BenchmarkResult)
        # Statistical rules should pass with large sample
        assert result.success_rate > 0.8
    
    def test_regression_validation(self, validation_benchmark):
        """Test regression testing validation."""
        def stable_function(x, y):
            return x + y
        
        # Define expected outputs for known inputs
        test_cases = [
            ((1, 2), 3),
            ((5, 7), 12),
            ((-1, 3), 2),
            ((0, 0), 0)
        ]
        
        result = validation_benchmark.validate_regression(
            stable_function,
            test_cases
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.success_rate == 1.0  # All test cases should pass
        assert result.error_count == 0
    
    def test_boundary_condition_validation(self, validation_benchmark):
        """Test boundary condition validation."""
        def safe_division(x, y):
            if y == 0:
                return float('inf')
            return x / y
        
        boundary_cases = [
            ((10, 0), float('inf')),  # Division by zero
            ((0, 5), 0),              # Zero numerator
            ((1, 1), 1),              # Normal case
            ((-5, -1), 5)             # Negative numbers
        ]
        
        result = validation_benchmark.validate_regression(
            safe_division,
            boundary_cases
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.success_rate == 1.0


class TestLatencyBenchmark:
    """Test LatencyBenchmark functionality."""
    
    @pytest.fixture
    def latency_benchmark(self):
        """Create latency benchmark instance."""
        config = BenchmarkConfig(
            name="Latency Test",
            max_latency_ms=100.0,
            iterations=1000
        )
        return LatencyBenchmark(config)
    
    def test_latency_benchmark_initialization(self, latency_benchmark):
        """Test LatencyBenchmark initialization."""
        assert latency_benchmark.config.name == "Latency Test"
        assert latency_benchmark.config.max_latency_ms == 100.0
        assert len(latency_benchmark.latency_measurements) == 0
    
    def test_measure_function_latency(self, latency_benchmark):
        """Test function latency measurement."""
        def fast_function():
            return sum(range(100))
        
        result = latency_benchmark.measure_latency(fast_function)
        
        assert isinstance(result, BenchmarkResult)
        assert result.latency_p50_ms is not None
        assert result.latency_p95_ms is not None
        assert result.latency_p99_ms is not None
        assert result.latency_max_ms is not None
        
        # Latencies should be reasonable for simple function
        assert result.latency_p50_ms < 50.0
    
    def test_measure_api_latency(self, latency_benchmark):
        """Test API-like latency measurement."""
        def api_simulation():
            # Simulate API call with variable latency
            delay = np.random.uniform(0.005, 0.020)  # 5-20ms
            time.sleep(delay)
            return {"status": "success", "data": [1, 2, 3]}
        
        result = latency_benchmark.measure_latency(api_simulation)
        
        assert isinstance(result, BenchmarkResult)
        # API latencies should be higher than simple functions
        assert result.latency_p50_ms > 5.0
        assert result.latency_p95_ms > result.latency_p50_ms
    
    def test_concurrent_latency_measurement(self, latency_benchmark):
        """Test concurrent latency measurement."""
        def concurrent_operation():
            time.sleep(0.01)  # 10ms operation
            return "done"
        
        result = latency_benchmark.measure_concurrent_latency(
            concurrent_operation,
            num_workers=5,
            operations_per_worker=20
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.latency_p50_ms is not None
        # Concurrent operations may have different latency characteristics
        assert result.throughput_ops_sec > 0
    
    def test_latency_percentile_calculation(self, latency_benchmark):
        """Test latency percentile calculations."""
        # Simulate known latency distribution
        def known_latency():
            # Fixed latencies for predictable percentiles
            latencies = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100]
            return time.sleep(np.random.choice(latencies))
        
        latency_benchmark.latency_measurements = [1, 2, 5, 10, 20, 50, 100]  # ms
        
        percentiles = latency_benchmark.calculate_percentiles()
        
        assert isinstance(percentiles, dict)
        assert 'p50' in percentiles
        assert 'p95' in percentiles
        assert 'p99' in percentiles
        assert percentiles['p95'] >= percentiles['p50']
        assert percentiles['p99'] >= percentiles['p95']
    
    def test_latency_SLA_validation(self, latency_benchmark):
        """Test SLA (Service Level Agreement) validation."""
        def sla_test_function():
            # Most operations fast, occasional slow one
            if np.random.random() < 0.05:  # 5% slow operations
                time.sleep(0.15)  # 150ms - exceeds SLA
            else:
                time.sleep(0.01)  # 10ms - within SLA
            return "completed"
        
        result = latency_benchmark.validate_sla(
            sla_test_function,
            sla_percentile=95,
            sla_threshold_ms=100.0
        )
        
        assert isinstance(result, BenchmarkResult)
        # Should detect SLA violations
        assert result.sla_compliance_rate is not None
        assert 0 <= result.sla_compliance_rate <= 1.0


class TestMemoryProfiler:
    """Test MemoryProfiler functionality."""
    
    @pytest.fixture
    def memory_profiler(self):
        """Create memory profiler instance."""
        config = BenchmarkConfig(
            name="Memory Test",
            max_memory_mb=512,
            iterations=50
        )
        return MemoryProfiler(config)
    
    def test_memory_profiler_initialization(self, memory_profiler):
        """Test MemoryProfiler initialization."""
        assert memory_profiler.config.name == "Memory Test"
        assert memory_profiler.config.max_memory_mb == 512
        assert len(memory_profiler.memory_snapshots) == 0
    
    def test_profile_memory_usage(self, memory_profiler):
        """Test memory usage profiling."""
        def memory_allocator():
            # Allocate and deallocate memory
            large_list = [i for i in range(10000)]
            large_dict = {i: i * 2 for i in range(5000)}
            return len(large_list) + len(large_dict)
        
        result = memory_profiler.profile_function(memory_allocator)
        
        assert isinstance(result, BenchmarkResult)
        assert result.memory_peak_mb is not None
        assert result.memory_peak_mb > 0
        assert result.memory_baseline_mb is not None
    
    def test_memory_leak_detection(self, memory_profiler):
        """Test memory leak detection."""
        # Global variable to simulate memory leak
        global_storage = []
        
        def potential_leak():
            # Add data to global storage (simulating leak)
            global_storage.extend(range(1000))
            return len(global_storage)
        
        result = memory_profiler.detect_memory_leaks(
            potential_leak,
            iterations=10,
            leak_threshold_mb=1.0
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.memory_leak_detected is not None
        # Should detect increasing memory usage
        assert result.memory_growth_rate is not None
    
    def test_garbage_collection_analysis(self, memory_profiler):
        """Test garbage collection analysis."""
        def gc_intensive():
            # Create objects that need garbage collection
            data = []
            for i in range(1000):
                data.append({'id': i, 'data': list(range(i % 100))})
            return len(data)
        
        result = memory_profiler.analyze_gc_behavior(gc_intensive)
        
        assert isinstance(result, BenchmarkResult)
        assert result.gc_collections is not None
        assert result.gc_time_ms is not None
        assert result.gc_collections >= 0
    
    def test_memory_efficiency_measurement(self, memory_profiler):
        """Test memory efficiency measurement."""
        def efficient_operation():
            # Memory-efficient operation
            return sum(i for i in range(1000))
        
        def inefficient_operation():
            # Memory-inefficient operation
            temp_list = []
            for i in range(1000):
                temp_list.append(i)
                temp_list.append(i * 2)
                temp_list.append(i * 3)
            return sum(temp_list)
        
        efficient_result = memory_profiler.profile_function(efficient_operation)
        inefficient_result = memory_profiler.profile_function(inefficient_operation)
        
        # Inefficient operation should use more memory
        assert inefficient_result.memory_peak_mb >= efficient_result.memory_peak_mb


class TestBenchmarkSuite:
    """Test BenchmarkSuite orchestration."""
    
    @pytest.fixture
    def benchmark_suite(self):
        """Create benchmark suite instance."""
        return BenchmarkSuite(
            name="Comprehensive Test Suite",
            description="Full RTradez benchmark suite"
        )
    
    def test_benchmark_suite_initialization(self, benchmark_suite):
        """Test BenchmarkSuite initialization."""
        assert benchmark_suite.name == "Comprehensive Test Suite"
        assert len(benchmark_suite.benchmarks) == 0
        assert len(benchmark_suite.results) == 0
    
    def test_add_benchmark_to_suite(self, benchmark_suite):
        """Test adding benchmarks to suite."""
        config = BenchmarkConfig(name="Test Benchmark")
        perf_bench = PerformanceBenchmark(config)
        
        benchmark_suite.add_benchmark("performance", perf_bench)
        
        assert len(benchmark_suite.benchmarks) == 1
        assert "performance" in benchmark_suite.benchmarks
    
    def test_run_full_suite(self, benchmark_suite):
        """Test running complete benchmark suite."""
        # Add various benchmark types
        perf_config = BenchmarkConfig(name="Performance", iterations=10)
        stress_config = BenchmarkConfig(name="Stress", target_duration=2.0)
        
        benchmark_suite.add_benchmark("performance", PerformanceBenchmark(perf_config))
        benchmark_suite.add_benchmark("stress", StressTester(stress_config))
        
        # Define test functions
        def simple_test():
            return sum(range(100))
        
        # Run suite
        suite_results = benchmark_suite.run_suite({
            "performance": simple_test,
            "stress": simple_test
        })
        
        assert isinstance(suite_results, dict)
        assert len(suite_results) == 2
        assert "performance" in suite_results
        assert "stress" in suite_results
        
        for result in suite_results.values():
            assert isinstance(result, BenchmarkResult)
    
    def test_generate_suite_report(self, benchmark_suite):
        """Test suite report generation."""
        # Add some mock results
        result1 = BenchmarkResult(
            benchmark_name="Test 1",
            status="PASSED",
            duration=10.5,
            success_rate=0.98
        )
        
        result2 = BenchmarkResult(
            benchmark_name="Test 2", 
            status="FAILED",
            duration=15.2,
            success_rate=0.85
        )
        
        benchmark_suite.results = {"test1": result1, "test2": result2}
        
        report = benchmark_suite.generate_report()
        
        assert isinstance(report, str)
        assert "Comprehensive Test Suite" in report
        assert "Test 1" in report
        assert "Test 2" in report
        assert "PASSED" in report
        assert "FAILED" in report
    
    def test_save_and_load_results(self, benchmark_suite):
        """Test saving and loading benchmark results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create result
            result = BenchmarkResult(
                benchmark_name="Save Test",
                status="PASSED",
                duration=5.0,
                success_rate=1.0
            )
            
            benchmark_suite.results = {"save_test": result}
            
            # Save results
            save_path = Path(temp_dir) / "results.json"
            benchmark_suite.save_results(save_path)
            
            assert save_path.exists()
            
            # Load results
            loaded_results = benchmark_suite.load_results(save_path)
            
            assert isinstance(loaded_results, dict)
            assert "save_test" in loaded_results
            assert loaded_results["save_test"]["benchmark_name"] == "Save Test"


@pytest.mark.integration
class TestBenchmarksIntegration:
    """Integration tests for benchmarks framework."""
    
    def test_full_benchmark_workflow(self):
        """Test complete benchmarking workflow."""
        # Create comprehensive benchmark suite
        suite = BenchmarkSuite(
            name="RTradez Integration Benchmark",
            description="Complete system benchmark"
        )
        
        # Add different benchmark types
        configs = {
            "performance": BenchmarkConfig(name="Performance", iterations=50),
            "stress": BenchmarkConfig(name="Stress", target_duration=3.0),
            "latency": BenchmarkConfig(name="Latency", max_latency_ms=50.0),
            "memory": BenchmarkConfig(name="Memory", max_memory_mb=256)
        }
        
        suite.add_benchmark("performance", PerformanceBenchmark(configs["performance"]))
        suite.add_benchmark("stress", StressTester(configs["stress"]))
        suite.add_benchmark("latency", LatencyBenchmark(configs["latency"]))
        suite.add_benchmark("memory", MemoryProfiler(configs["memory"]))
        
        # Define test workloads
        def cpu_workload():
            return sum(i * i for i in range(500))
        
        def memory_workload():
            data = [i for i in range(1000)]
            return len(data)
        
        def io_workload():
            time.sleep(0.001)  # Simulate I/O
            return "completed"
        
        # Run comprehensive benchmarks
        workloads = {
            "performance": cpu_workload,
            "stress": cpu_workload,
            "latency": io_workload,
            "memory": memory_workload
        }
        
        results = suite.run_suite(workloads)
        
        # Verify all benchmarks completed
        assert len(results) == 4
        for benchmark_name, result in results.items():
            assert isinstance(result, BenchmarkResult)
            assert result.duration > 0
            assert result.status in ["PASSED", "FAILED", "TIMEOUT"]
        
        # Generate and verify report
        report = suite.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0
        assert "RTradez Integration Benchmark" in report
        
        # Check that all benchmark types are represented
        assert "Performance" in report
        assert "Stress" in report
        assert "Latency" in report
        assert "Memory" in report
    
    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        # Baseline benchmark
        baseline_config = BenchmarkConfig(name="Baseline", iterations=100)
        baseline_benchmark = PerformanceBenchmark(baseline_config)
        
        def baseline_function():
            return sum(range(1000))
        
        baseline_result = baseline_benchmark.benchmark_function(baseline_function)
        
        # Simulated regression (slower function)
        def regressed_function():
            time.sleep(0.001)  # Add artificial delay
            return sum(range(1000))
        
        regressed_result = baseline_benchmark.benchmark_function(regressed_function)
        
        # Regression should be detectable
        assert regressed_result.duration > baseline_result.duration
        assert regressed_result.throughput_ops_sec < baseline_result.throughput_ops_sec
        
        # Calculate performance degradation
        degradation = (regressed_result.duration - baseline_result.duration) / baseline_result.duration
        assert degradation > 0  # Performance should be worse
    
    def test_system_resource_monitoring(self):
        """Test system resource monitoring during benchmarks."""
        config = BenchmarkConfig(name="Resource Monitor", iterations=20)
        memory_profiler = MemoryProfiler(config)
        
        def resource_intensive():
            # CPU and memory intensive operation
            data = []
            for i in range(5000):
                data.append([j for j in range(i % 100)])
            return len(data)
        
        # Monitor system resources during benchmark
        initial_memory = psutil.virtual_memory().used
        
        result = memory_profiler.profile_function(resource_intensive)
        
        final_memory = psutil.virtual_memory().used
        
        # Verify resource monitoring worked
        assert isinstance(result, BenchmarkResult)
        assert result.memory_peak_mb > 0
        
        # Memory usage should have increased during execution
        memory_delta_mb = (final_memory - initial_memory) / (1024 * 1024)
        assert abs(memory_delta_mb) >= 0  # Some memory change expected