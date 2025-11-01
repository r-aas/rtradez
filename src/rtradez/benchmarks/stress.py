"""
Stress testing framework for RTradez components.

Tests system behavior under extreme conditions, edge cases, and failure scenarios
to ensure robustness before live trading deployment.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
from datetime import datetime, timedelta
import random

from .core import ComponentBenchmark, BenchmarkConfig, BenchmarkSeverity
from ..risk import (
    KellyConfig, KellyCriterion, FixedFractionConfig, FixedFractionSizer,
    MultiStrategyConfig, MultiStrategyPositionSizer
)
from ..portfolio.portfolio_manager import PortfolioManager, PortfolioConfig
from ..utils.temporal_alignment import TemporalAligner, TemporalAlignerConfig, FrequencyType


class StressTester(ComponentBenchmark):
    """Comprehensive stress testing framework."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("StressTest", config)
        self._register_stress_tests()
    
    def _register_stress_tests(self):
        """Register all stress tests."""
        
        # Risk Management Stress Tests
        @self.suite.register_benchmark(
            "risk_extreme_parameters_stress",
            "Test risk calculations with extreme parameters",
            BenchmarkSeverity.CRITICAL
        )
        def test_extreme_parameters():
            return self._test_extreme_parameters()
        
        @self.suite.register_benchmark(
            "risk_memory_pressure_test",
            "Test risk calculations under memory pressure",
            BenchmarkSeverity.ERROR
        )
        def test_memory_pressure():
            return self._test_memory_pressure()
        
        @self.suite.register_benchmark(
            "risk_high_frequency_operations",
            "Test sustained high-frequency risk calculations",
            BenchmarkSeverity.ERROR
        )
        def test_high_frequency_operations():
            return self._test_high_frequency_operations()
        
        # Portfolio Management Stress Tests
        @self.suite.register_benchmark(
            "portfolio_maximum_strategies_stress",
            "Test portfolio with maximum number of strategies",
            BenchmarkSeverity.ERROR
        )
        def test_maximum_strategies():
            return self._test_maximum_strategies()
        
        @self.suite.register_benchmark(
            "portfolio_rapid_rebalancing_stress",
            "Test rapid consecutive rebalancing operations",
            BenchmarkSeverity.WARNING
        )
        def test_rapid_rebalancing():
            return self._test_rapid_rebalancing()
        
        @self.suite.register_benchmark(
            "portfolio_extreme_allocations_stress",
            "Test portfolio with extreme allocation scenarios",
            BenchmarkSeverity.ERROR
        )
        def test_extreme_allocations():
            return self._test_extreme_allocations()
        
        # Data Processing Stress Tests
        @self.suite.register_benchmark(
            "data_massive_dataset_stress",
            "Test data processing with massive datasets",
            BenchmarkSeverity.WARNING
        )
        def test_massive_datasets():
            return self._test_massive_datasets()
        
        @self.suite.register_benchmark(
            "data_corrupted_input_stress",
            "Test data processing with corrupted inputs",
            BenchmarkSeverity.CRITICAL
        )
        def test_corrupted_inputs():
            return self._test_corrupted_inputs()
        
        # System-wide Stress Tests
        @self.suite.register_benchmark(
            "system_concurrent_load_stress",
            "Test system under maximum concurrent load",
            BenchmarkSeverity.CRITICAL
        )
        def test_concurrent_load():
            return self._test_concurrent_load()
        
        @self.suite.register_benchmark(
            "system_resource_exhaustion_stress",
            "Test system behavior during resource exhaustion",
            BenchmarkSeverity.CRITICAL
        )
        def test_resource_exhaustion():
            return self._test_resource_exhaustion()
    
    def _test_extreme_parameters(self) -> Dict[str, Any]:
        """Test risk calculations with extreme parameters."""
        results = {}
        
        config = KellyConfig(total_capital=100000)
        sizer = KellyCriterion(config)
        
        # Test extreme scenarios
        extreme_cases = [
            # (name, expected_return, volatility, should_succeed)
            ("zero_volatility", 0.10, 0.0, False),
            ("negative_return", -0.50, 0.20, True),
            ("extreme_positive_return", 5.0, 0.20, True),
            ("extreme_volatility", 0.10, 10.0, True),
            ("both_extreme", 2.0, 5.0, True),
            ("tiny_values", 0.0001, 0.0001, True),
            ("huge_values", 1000.0, 100.0, True),
        ]
        
        successes = 0
        failures = 0
        errors = []
        
        for case_name, ret, vol, should_succeed in extreme_cases:
            try:
                result = sizer.calculate_position_size(
                    strategy_name=f"ExtremeSizing_{case_name}",
                    expected_return=ret,
                    volatility=vol
                )
                
                # Validate result makes sense
                if result.recommended_size < 0:
                    errors.append(f"{case_name}: Negative position size")
                elif result.recommended_size > config.total_capital * 2:
                    errors.append(f"{case_name}: Position size exceeds reasonable limits")
                else:
                    successes += 1
                    
            except Exception as e:
                if should_succeed:
                    errors.append(f"{case_name}: Unexpected failure - {str(e)}")
                    failures += 1
                else:
                    successes += 1  # Expected failure
        
        results['extreme_cases_tested'] = len(extreme_cases)
        results['successes'] = successes
        results['failures'] = failures
        results['errors'] = errors
        results['success_rate'] = successes / len(extreme_cases)
        
        # Should handle most extreme cases gracefully
        if results['success_rate'] < 0.8:
            raise AssertionError(f"Too many failures in extreme parameter testing: {results['success_rate']:.1%}")
        
        return results
    
    def _test_memory_pressure(self) -> Dict[str, Any]:
        """Test risk calculations under memory pressure."""
        results = {}
        
        # Create memory pressure by allocating large arrays
        memory_hogs = []
        initial_memory = psutil.Process().memory_info().rss
        
        try:
            # Allocate memory to create pressure (but not exhaust)
            available_memory = psutil.virtual_memory().available
            target_allocation = min(available_memory // 4, 1024 * 1024 * 1024)  # 1GB max
            
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            num_chunks = target_allocation // chunk_size
            
            for i in range(num_chunks):
                memory_hogs.append(np.random.randn(chunk_size // 8))  # 8 bytes per float64
                
                # Test risk calculations every 10 chunks
                if i % 10 == 0:
                    config = KellyConfig(total_capital=100000)
                    sizer = KellyCriterion(config)
                    
                    # Perform calculations
                    for j in range(10):
                        result = sizer.calculate_position_size(
                            strategy_name=f"MemoryPressure_{i}_{j}",
                            expected_return=0.10,
                            volatility=0.20
                        )
                        
                        if result.recommended_size <= 0:
                            raise AssertionError("Invalid position size under memory pressure")
            
            final_memory = psutil.Process().memory_info().rss
            memory_growth = final_memory - initial_memory
            
            results['initial_memory_mb'] = initial_memory / 1024 / 1024
            results['final_memory_mb'] = final_memory / 1024 / 1024
            results['memory_growth_mb'] = memory_growth / 1024 / 1024
            results['memory_chunks_allocated'] = len(memory_hogs)
            results['calculations_completed'] = True
            
        finally:
            # Clean up memory
            del memory_hogs
            gc.collect()
        
        return results
    
    def _test_high_frequency_operations(self) -> Dict[str, Any]:
        """Test sustained high-frequency risk calculations."""
        results = {}
        
        config = KellyConfig(total_capital=1000000)
        sizer = KellyCriterion(config)
        
        # Test sustained high-frequency operations
        duration_seconds = 30  # 30 second stress test
        operations = 0
        errors = 0
        latencies = []
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        while time.perf_counter() < end_time:
            operation_start = time.perf_counter()
            
            try:
                result = sizer.calculate_position_size(
                    strategy_name=f"HighFreq_{operations}",
                    expected_return=random.uniform(0.05, 0.25),
                    volatility=random.uniform(0.10, 0.40)
                )
                
                if result.recommended_size <= 0:
                    errors += 1
                    
            except Exception:
                errors += 1
            
            operation_end = time.perf_counter()
            latencies.append((operation_end - operation_start) * 1000)  # ms
            operations += 1
        
        actual_duration = time.perf_counter() - start_time
        
        results['duration_seconds'] = actual_duration
        results['total_operations'] = operations
        results['operations_per_second'] = operations / actual_duration
        results['error_count'] = errors
        results['error_rate'] = errors / operations if operations > 0 else 0
        results['avg_latency_ms'] = np.mean(latencies) if latencies else 0
        results['p95_latency_ms'] = np.percentile(latencies, 95) if latencies else 0
        results['p99_latency_ms'] = np.percentile(latencies, 99) if latencies else 0
        
        # Validate high-frequency performance
        min_ops_per_sec = 1000  # Should handle at least 1000 ops/sec
        if results['operations_per_second'] < min_ops_per_sec:
            raise AssertionError(f"High-frequency performance {results['operations_per_second']:.0f} ops/sec below {min_ops_per_sec}")
        
        max_error_rate = 0.001  # 0.1% max error rate
        if results['error_rate'] > max_error_rate:
            raise AssertionError(f"High-frequency error rate {results['error_rate']:.3f} exceeds {max_error_rate}")
        
        return results
    
    def _test_maximum_strategies(self) -> Dict[str, Any]:
        """Test portfolio with maximum number of strategies."""
        results = {}
        
        # Test with absolute maximum strategies
        max_strategies = 100  # Push beyond normal limits
        config = PortfolioConfig(
            total_capital=100000000,  # $100M
            max_strategies=max_strategies,
            rebalance_threshold=0.01  # Sensitive rebalancing
        )
        
        portfolio = PortfolioManager(config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        # Add maximum strategies
        start_time = time.perf_counter()
        strategies_added = 0
        
        for i in range(max_strategies):
            try:
                success = portfolio.add_strategy(
                    f"MaxStress_{i}",
                    MockStrategy(f"MaxStress_{i}"),
                    target_allocation=1.0 / max_strategies
                )
                
                if success:
                    strategies_added += 1
                else:
                    break
                    
            except Exception as e:
                results['add_strategy_error'] = str(e)
                break
        
        add_time = time.perf_counter() - start_time
        
        # Test operations with maximum strategies
        operations_start = time.perf_counter()
        
        try:
            # Test portfolio operations
            for _ in range(10):
                metrics = portfolio.calculate_portfolio_metrics()
                summary = portfolio.get_portfolio_summary()
                rebalancing_needs = portfolio.calculate_rebalancing_needs()
                
        except Exception as e:
            results['operations_error'] = str(e)
        
        operations_time = time.perf_counter() - operations_start
        
        results['max_strategies_target'] = max_strategies
        results['strategies_added'] = strategies_added
        results['add_strategies_time'] = add_time
        results['operations_time'] = operations_time
        results['avg_operation_time_ms'] = (operations_time / 10) * 1000
        
        # Validate maximum strategy handling
        if strategies_added < max_strategies * 0.9:  # Should add at least 90%
            raise AssertionError(f"Could only add {strategies_added}/{max_strategies} strategies")
        
        max_operation_time_ms = 100.0  # 100ms max for operations
        if results['avg_operation_time_ms'] > max_operation_time_ms:
            raise AssertionError(f"Operations too slow with max strategies: {results['avg_operation_time_ms']:.1f}ms")
        
        return results
    
    def _test_rapid_rebalancing(self) -> Dict[str, Any]:
        """Test rapid consecutive rebalancing operations."""
        results = {}
        
        config = PortfolioConfig(total_capital=5000000, max_strategies=20)
        portfolio = PortfolioManager(config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        # Add strategies
        strategy_count = 15
        for i in range(strategy_count):
            portfolio.add_strategy(
                f"RapidRebalance_{i}",
                MockStrategy(f"RapidRebalance_{i}"),
                target_allocation=1.0 / strategy_count
            )
        
        # Perform rapid rebalancing
        rebalance_count = 1000
        successful_rebalances = 0
        errors = []
        rebalance_times = []
        
        start_time = time.perf_counter()
        
        for i in range(rebalance_count):
            # Create random drift
            for name, allocation in portfolio.strategies.items():
                drift = random.uniform(-0.05, 0.05)  # ±5% drift
                allocation.current_allocation = max(0, allocation.target_allocation + drift)
            
            rebalance_start = time.perf_counter()
            
            try:
                needs = portfolio.calculate_rebalancing_needs()
                if needs:
                    success = portfolio.execute_rebalancing(needs)
                    if success:
                        successful_rebalances += 1
                else:
                    successful_rebalances += 1  # No rebalancing needed is success
                    
            except Exception as e:
                errors.append(f"Rebalance {i}: {str(e)}")
            
            rebalance_end = time.perf_counter()
            rebalance_times.append((rebalance_end - rebalance_start) * 1000)  # ms
        
        total_time = time.perf_counter() - start_time
        
        results['rapid_rebalances_attempted'] = rebalance_count
        results['successful_rebalances'] = successful_rebalances
        results['rebalancing_errors'] = len(errors)
        results['success_rate'] = successful_rebalances / rebalance_count
        results['total_time_seconds'] = total_time
        results['rebalances_per_second'] = rebalance_count / total_time
        results['avg_rebalance_time_ms'] = np.mean(rebalance_times)
        results['p95_rebalance_time_ms'] = np.percentile(rebalance_times, 95)
        
        # Validate rapid rebalancing
        min_success_rate = 0.98  # 98% success rate
        if results['success_rate'] < min_success_rate:
            raise AssertionError(f"Rapid rebalancing success rate {results['success_rate']:.1%} below {min_success_rate:.1%}")
        
        return results
    
    def _test_extreme_allocations(self) -> Dict[str, Any]:
        """Test portfolio with extreme allocation scenarios."""
        results = {}
        
        config = PortfolioConfig(total_capital=1000000, max_strategies=10)
        portfolio = PortfolioManager(config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        extreme_scenarios = [
            ("single_strategy_100", [(0, 1.0)]),  # One strategy gets 100%
            ("highly_concentrated", [(0, 0.8), (1, 0.2)]),  # 80/20 split
            ("micro_allocations", [(i, 0.01) for i in range(10)]),  # 1% each
            ("zero_allocations", [(i, 0.0) for i in range(5)]),  # All zero
        ]
        
        scenario_results = {}
        
        for scenario_name, allocations in extreme_scenarios:
            # Create fresh portfolio
            portfolio = PortfolioManager(config)
            
            try:
                # Add strategies with extreme allocations
                for i, allocation in allocations:
                    portfolio.add_strategy(
                        f"Extreme_{scenario_name}_{i}",
                        MockStrategy(f"Extreme_{scenario_name}_{i}"),
                        target_allocation=allocation
                    )
                
                # Test operations
                metrics = portfolio.calculate_portfolio_metrics()
                summary = portfolio.get_portfolio_summary()
                
                # Test rebalancing
                for name, alloc in portfolio.strategies.items():
                    alloc.current_allocation = alloc.target_allocation * random.uniform(0.8, 1.2)
                
                needs = portfolio.calculate_rebalancing_needs()
                if needs:
                    portfolio.execute_rebalancing(needs)
                
                scenario_results[scenario_name] = {
                    'success': True,
                    'strategies_added': len(portfolio.strategies),
                    'total_allocation': sum(s.target_allocation for s in portfolio.strategies.values()),
                    'metrics_calculated': True
                }
                
            except Exception as e:
                scenario_results[scenario_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        results['extreme_scenarios'] = scenario_results
        results['scenarios_tested'] = len(extreme_scenarios)
        results['successful_scenarios'] = sum(1 for r in scenario_results.values() if r.get('success', False))
        
        # Should handle most extreme scenarios
        min_success_rate = 0.75  # 75% of extreme scenarios should work
        actual_success_rate = results['successful_scenarios'] / results['scenarios_tested']
        if actual_success_rate < min_success_rate:
            raise AssertionError(f"Extreme allocation success rate {actual_success_rate:.1%} below {min_success_rate:.1%}")
        
        return results
    
    def _test_massive_datasets(self) -> Dict[str, Any]:
        """Test data processing with massive datasets."""
        results = {}
        
        # Test with progressively larger datasets
        dataset_sizes = [100000, 500000, 1000000]  # Up to 1M rows
        
        for size in dataset_sizes:
            # Generate large dataset
            start_time = time.perf_counter()
            
            dates = pd.date_range(start='2020-01-01', periods=size, freq='1min')
            large_data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(size) * 0.01),
                'volume': np.random.randint(100, 10000, size),
                'bid': np.random.uniform(99, 101, size),
                'ask': np.random.uniform(99, 101, size)
            }, index=dates)
            
            generation_time = time.perf_counter() - start_time
            
            # Test processing operations
            processing_start = time.perf_counter()
            
            try:
                # Various processing operations
                daily_aggregated = large_data.resample('D').agg({
                    'price': 'last',
                    'volume': 'sum',
                    'bid': 'mean',
                    'ask': 'mean'
                })
                
                # Calculate rolling statistics
                rolling_stats = large_data['price'].rolling(window=100).agg(['mean', 'std'])
                
                # Memory usage check
                memory_usage = large_data.memory_usage(deep=True).sum()
                
                processing_time = time.perf_counter() - processing_start
                
                results[f'dataset_{size}_generation_time'] = generation_time
                results[f'dataset_{size}_processing_time'] = processing_time
                results[f'dataset_{size}_memory_mb'] = memory_usage / 1024 / 1024
                results[f'dataset_{size}_rows_per_second'] = size / processing_time
                
            except Exception as e:
                results[f'dataset_{size}_error'] = str(e)
            
            # Clean up to prevent memory issues
            del large_data
            gc.collect()
        
        # Validate massive dataset handling
        largest_size = max(dataset_sizes)
        min_throughput = 10000  # 10k rows/sec minimum
        actual_throughput = results.get(f'dataset_{largest_size}_rows_per_second', 0)
        
        if actual_throughput < min_throughput:
            raise AssertionError(f"Massive dataset throughput {actual_throughput:.0f} below {min_throughput} rows/sec")
        
        return results
    
    def _test_corrupted_inputs(self) -> Dict[str, Any]:
        """Test data processing with corrupted inputs."""
        results = {}
        
        def apply_missing_values(df):
            df.iloc[::10] = np.nan
            
        def apply_infinite_values(df):
            df.iloc[::20] = np.inf
            
        def apply_negative_infinite(df):
            df.iloc[::30] = -np.inf
            
        def apply_extreme_outliers(df):
            df.iloc[::50] = df.iloc[::50] * 1000
        
        corruption_tests = [
            ("missing_values", apply_missing_values),
            ("infinite_values", apply_infinite_values),
            ("negative_infinite", apply_negative_infinite),
            ("extreme_outliers", apply_extreme_outliers),
            ("wrong_data_types", None),  # Will handle separately
            ("duplicate_timestamps", None),  # Will handle separately
        ]
        
        base_data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.randint(100, 1000, 1000)
        }, index=pd.date_range('2024-01-01', periods=1000, freq='1min'))
        
        corruption_results = {}
        
        for corruption_name, corruption_func in corruption_tests:
            try:
                # Apply corruption
                corrupted_data = base_data.copy()
                
                if corruption_name == "wrong_data_types":
                    # Mix in string data
                    corrupted_data.loc[corrupted_data.index[::10], 'price'] = 'invalid'
                elif corruption_name == "duplicate_timestamps":
                    # Create duplicate timestamps
                    corrupted_data = pd.concat([corrupted_data, corrupted_data.iloc[:100]])
                elif corruption_func is not None:
                    corruption_func(corrupted_data)
                
                # Test processing with corrupted data
                processing_start = time.perf_counter()
                
                # Attempt various operations
                try:
                    # Basic statistics (should handle NaN/inf gracefully)
                    stats = corrupted_data.describe()
                    
                    # Resampling
                    resampled = corrupted_data.resample('H').mean()
                    
                    # Rolling calculations
                    rolling = corrupted_data['price'].rolling(10).mean()
                    
                    processing_time = time.perf_counter() - processing_start
                    
                    corruption_results[corruption_name] = {
                        'success': True,
                        'processing_time': processing_time,
                        'rows_processed': len(corrupted_data),
                        'nan_count': corrupted_data.isna().sum().sum(),
                        'inf_count': np.isinf(corrupted_data.select_dtypes(include=[np.number])).sum().sum()
                    }
                    
                except Exception as e:
                    corruption_results[corruption_name] = {
                        'success': False,
                        'error': str(e)
                    }
                
            except Exception as e:
                corruption_results[corruption_name] = {
                    'success': False,
                    'setup_error': str(e)
                }
        
        results['corruption_tests'] = corruption_results
        results['tests_run'] = len(corruption_tests)
        results['successful_tests'] = sum(1 for r in corruption_results.values() if r.get('success', False))
        results['robustness_score'] = results['successful_tests'] / results['tests_run']
        
        # Should handle most corruption gracefully
        min_robustness = 0.7  # 70% of corruption tests should succeed
        if results['robustness_score'] < min_robustness:
            raise AssertionError(f"Data corruption robustness {results['robustness_score']:.1%} below {min_robustness:.1%}")
        
        return results
    
    def _test_concurrent_load(self) -> Dict[str, Any]:
        """Test system under maximum concurrent load."""
        results = {}
        
        # Test maximum concurrent operations across all components
        max_workers = min(psutil.cpu_count() * 2, 20)  # Don't overwhelm system
        operations_per_worker = 100
        
        def mixed_workload(worker_id: int):
            """Mixed workload combining all components."""
            operations = 0
            errors = 0
            
            # Risk management operations
            config = KellyConfig(total_capital=100000)
            sizer = KellyCriterion(config)
            
            # Portfolio operations
            portfolio_config = PortfolioConfig(total_capital=500000)
            portfolio = PortfolioManager(portfolio_config)
            
            class MockStrategy:
                def __init__(self, name): self.name = name
            
            for i in range(operations_per_worker):
                try:
                    # Risk calculation
                    sizer.calculate_position_size(
                        strategy_name=f"Concurrent_{worker_id}_{i}",
                        expected_return=random.uniform(0.05, 0.20),
                        volatility=random.uniform(0.10, 0.30)
                    )
                    
                    # Portfolio operation (every 10 iterations)
                    if i % 10 == 0:
                        portfolio.add_strategy(
                            f"ConcurrentStrat_{worker_id}_{i}",
                            MockStrategy(f"ConcurrentStrat_{worker_id}_{i}"),
                            target_allocation=0.1
                        )
                        
                        if len(portfolio.strategies) > 5:
                            portfolio.calculate_portfolio_metrics()
                    
                    operations += 1
                    
                except Exception:
                    errors += 1
            
            return {'operations': operations, 'errors': errors, 'worker_id': worker_id}
        
        # Execute concurrent load test
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(mixed_workload, i) for i in range(max_workers)]
            worker_results = [future.result() for future in futures]
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        # Aggregate results
        total_operations = sum(r['operations'] for r in worker_results)
        total_errors = sum(r['errors'] for r in worker_results)
        
        results['concurrent_workers'] = max_workers
        results['operations_per_worker'] = operations_per_worker
        results['total_operations'] = total_operations
        results['total_errors'] = total_errors
        results['error_rate'] = total_errors / total_operations if total_operations > 0 else 0
        results['duration_seconds'] = total_duration
        results['operations_per_second'] = total_operations / total_duration
        results['worker_results'] = worker_results
        
        # Validate concurrent load handling
        max_error_rate = 0.05  # 5% max error rate under load
        if results['error_rate'] > max_error_rate:
            raise AssertionError(f"Concurrent load error rate {results['error_rate']:.1%} exceeds {max_error_rate:.1%}")
        
        min_ops_per_sec = 500  # Should maintain decent throughput
        if results['operations_per_second'] < min_ops_per_sec:
            raise AssertionError(f"Concurrent load throughput {results['operations_per_second']:.0f} below {min_ops_per_sec} ops/sec")
        
        return results
    
    def _test_resource_exhaustion(self) -> Dict[str, Any]:
        """Test system behavior during resource exhaustion."""
        results = {}
        
        initial_memory = psutil.Process().memory_info().rss
        initial_cpu_percent = psutil.cpu_percent()
        
        # Test graceful degradation under resource pressure
        memory_allocations = []
        operations_completed = 0
        errors_encountered = 0
        
        try:
            # Gradually increase memory pressure
            available_memory = psutil.virtual_memory().available
            target_memory = min(available_memory // 2, 2 * 1024 * 1024 * 1024)  # 2GB max
            
            chunk_size = 50 * 1024 * 1024  # 50MB chunks
            max_chunks = target_memory // chunk_size
            
            for i in range(max_chunks):
                # Allocate memory
                memory_allocations.append(np.random.randn(chunk_size // 8))
                
                # Test operations under increasing memory pressure
                if i % 5 == 0:  # Every 5 chunks
                    try:
                        # Risk calculation
                        config = KellyConfig(total_capital=100000)
                        sizer = KellyCriterion(config)
                        
                        result = sizer.calculate_position_size(
                            strategy_name=f"ResourceExhaustion_{i}",
                            expected_return=0.10,
                            volatility=0.20
                        )
                        
                        if result.recommended_size > 0:
                            operations_completed += 1
                        else:
                            errors_encountered += 1
                            
                    except Exception:
                        errors_encountered += 1
                
                # Check if we're approaching memory limits
                current_memory = psutil.Process().memory_info().rss
                if current_memory > initial_memory + target_memory:
                    break
            
            final_memory = psutil.Process().memory_info().rss
            memory_allocated = final_memory - initial_memory
            
            results['initial_memory_mb'] = initial_memory / 1024 / 1024
            results['final_memory_mb'] = final_memory / 1024 / 1024
            results['memory_allocated_mb'] = memory_allocated / 1024 / 1024
            results['memory_chunks_allocated'] = len(memory_allocations)
            results['operations_completed'] = operations_completed
            results['errors_encountered'] = errors_encountered
            results['operation_success_rate'] = operations_completed / (operations_completed + errors_encountered) if (operations_completed + errors_encountered) > 0 else 0
            
        finally:
            # Clean up
            del memory_allocations
            gc.collect()
        
        # Should maintain some functionality even under resource pressure
        min_success_rate = 0.5  # 50% success rate under pressure
        if results['operation_success_rate'] < min_success_rate:
            raise AssertionError(f"Resource exhaustion success rate {results['operation_success_rate']:.1%} below {min_success_rate:.1%}")
        
        return results