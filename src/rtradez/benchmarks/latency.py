"""
Latency benchmarking framework for RTradez components.

Tests real-time operation latency, response times, and timing-critical
performance for high-frequency trading readiness.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import statistics
import threading
import queue
import random

from .core import ComponentBenchmark, BenchmarkConfig, BenchmarkSeverity
from ..risk import (
    KellyConfig, KellyCriterion, FixedFractionConfig, FixedFractionSizer,
    VolatilityAdjustedConfig, VolatilityAdjustedSizer, MultiStrategyConfig,
    MultiStrategyPositionSizer
)
from ..portfolio.portfolio_manager import PortfolioManager, PortfolioConfig
from ..utils.temporal_alignment import TemporalAligner, TemporalAlignerConfig, FrequencyType


class LatencyBenchmark(ComponentBenchmark):
    """Comprehensive latency testing framework for real-time operations."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("Latency", config)
        self._register_latency_tests()
    
    def _register_latency_tests(self):
        """Register all latency benchmark tests."""
        
        # Single Operation Latency Tests
        @self.suite.register_benchmark(
            "position_sizing_latency",
            "Test single position sizing operation latency",
            BenchmarkSeverity.CRITICAL
        )
        def test_position_sizing_latency():
            return self._test_position_sizing_latency()
        
        @self.suite.register_benchmark(
            "portfolio_operation_latency",
            "Test portfolio management operation latency",
            BenchmarkSeverity.ERROR
        )
        def test_portfolio_latency():
            return self._test_portfolio_operation_latency()
        
        # Burst Operation Latency Tests
        @self.suite.register_benchmark(
            "burst_calculation_latency",
            "Test latency under burst calculation loads",
            BenchmarkSeverity.ERROR
        )
        def test_burst_latency():
            return self._test_burst_calculation_latency()
        
        @self.suite.register_benchmark(
            "concurrent_operation_latency",
            "Test latency under concurrent operations",
            BenchmarkSeverity.WARNING
        )
        def test_concurrent_latency():
            return self._test_concurrent_operation_latency()
        
        # Data Processing Latency Tests
        @self.suite.register_benchmark(
            "data_alignment_latency",
            "Test data temporal alignment latency",
            BenchmarkSeverity.WARNING
        )
        def test_data_alignment_latency():
            return self._test_data_alignment_latency()
        
        @self.suite.register_benchmark(
            "real_time_data_processing_latency",
            "Test real-time data processing latency",
            BenchmarkSeverity.CRITICAL
        )
        def test_realtime_processing_latency():
            return self._test_realtime_processing_latency()
        
        # System Response Latency Tests
        @self.suite.register_benchmark(
            "cold_start_latency",
            "Test system cold start latency",
            BenchmarkSeverity.WARNING
        )
        def test_cold_start_latency():
            return self._test_cold_start_latency()
        
        @self.suite.register_benchmark(
            "warm_operation_latency",
            "Test warm system operation latency",
            BenchmarkSeverity.ERROR
        )
        def test_warm_operation_latency():
            return self._test_warm_operation_latency()
        
        # Market Simulation Latency Tests
        @self.suite.register_benchmark(
            "market_tick_processing_latency",
            "Test market tick processing latency",
            BenchmarkSeverity.CRITICAL
        )
        def test_market_tick_latency():
            return self._test_market_tick_processing_latency()
        
        @self.suite.register_benchmark(
            "order_decision_latency",
            "Test order decision making latency",
            BenchmarkSeverity.CRITICAL
        )
        def test_order_decision_latency():
            return self._test_order_decision_latency()
    
    def _measure_operation_latency(self, operation: Callable, *args, **kwargs) -> Tuple[float, Any]:
        """Measure latency of a single operation with high precision."""
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return latency_ms, result
    
    def _measure_multiple_operations(self, operation: Callable, iterations: int, *args, **kwargs) -> Dict[str, float]:
        """Measure latency statistics for multiple operation iterations."""
        latencies = []
        
        for _ in range(iterations):
            latency_ms, _ = self._measure_operation_latency(operation, *args, **kwargs)
            latencies.append(latency_ms)
        
        return {
            'mean_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'total_operations': iterations,
            'latencies': latencies
        }
    
    def _test_position_sizing_latency(self) -> Dict[str, Any]:
        """Test single position sizing operation latency."""
        results = {}
        
        # Test different sizing methods
        methods = {
            'kelly': KellyCriterion(KellyConfig(total_capital=100000)),
            'fixed_fraction': FixedFractionSizer(FixedFractionConfig(total_capital=100000)),
            'volatility_adjusted': VolatilityAdjustedSizer(VolatilityAdjustedConfig(total_capital=100000))
        }
        
        iterations = 100
        
        for method_name, sizer in methods.items():
            def single_calculation():
                return sizer.calculate_position_size(
                    strategy_name="LatencyTest",
                    expected_return=random.uniform(0.05, 0.20),
                    volatility=random.uniform(0.10, 0.30)
                )
            
            latency_stats = self._measure_multiple_operations(single_calculation, iterations)
            results[f'{method_name}_latency'] = latency_stats
        
        # Validate latency requirements
        max_acceptable_p95_ms = 5.0  # 5ms P95 latency for position sizing
        
        for method_name in methods.keys():
            p95_latency = results[f'{method_name}_latency']['p95_latency_ms']
            if p95_latency > max_acceptable_p95_ms:
                raise AssertionError(f"{method_name} P95 latency {p95_latency:.2f}ms exceeds {max_acceptable_p95_ms}ms")
        
        # Calculate aggregate statistics
        all_latencies = []
        for method_name in methods.keys():
            all_latencies.extend(results[f'{method_name}_latency']['latencies'])
        
        results['aggregate_stats'] = {
            'overall_mean_ms': statistics.mean(all_latencies),
            'overall_p95_ms': np.percentile(all_latencies, 95),
            'overall_p99_ms': np.percentile(all_latencies, 99),
            'total_operations': len(all_latencies)
        }
        
        return results
    
    def _test_portfolio_operation_latency(self) -> Dict[str, Any]:
        """Test portfolio management operation latency."""
        results = {}
        
        config = PortfolioConfig(total_capital=1000000, max_strategies=10)
        portfolio = PortfolioManager(config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        # Setup portfolio with strategies
        for i in range(5):
            portfolio.add_strategy(
                f"LatencyTest_{i}",
                MockStrategy(f"LatencyTest_{i}"),
                target_allocation=0.2
            )
        
        # Test different portfolio operations
        operations = {
            'metrics_calculation': lambda: portfolio.calculate_portfolio_metrics(),
            'portfolio_summary': lambda: portfolio.get_portfolio_summary(),
            'rebalancing_needs': lambda: portfolio.calculate_rebalancing_needs()
        }
        
        iterations = 50  # Fewer iterations for more complex operations
        
        for op_name, operation in operations.items():
            latency_stats = self._measure_multiple_operations(operation, iterations)
            results[f'{op_name}_latency'] = latency_stats
        
        # Test rebalancing execution latency
        def rebalancing_execution():
            # Create some drift
            for name, allocation in portfolio.strategies.items():
                allocation.current_allocation = allocation.target_allocation * random.uniform(0.9, 1.1)
            
            needs = portfolio.calculate_rebalancing_needs()
            if needs:
                return portfolio.execute_rebalancing(needs)
            return True
        
        rebalancing_stats = self._measure_multiple_operations(rebalancing_execution, 20)
        results['rebalancing_execution_latency'] = rebalancing_stats
        
        # Validate portfolio operation latency requirements
        max_acceptable_metrics_p95_ms = 20.0  # 20ms for metrics calculation
        max_acceptable_rebalancing_p95_ms = 50.0  # 50ms for rebalancing
        
        metrics_p95 = results['metrics_calculation_latency']['p95_latency_ms']
        if metrics_p95 > max_acceptable_metrics_p95_ms:
            raise AssertionError(f"Portfolio metrics P95 latency {metrics_p95:.2f}ms exceeds {max_acceptable_metrics_p95_ms}ms")
        
        rebalancing_p95 = results['rebalancing_execution_latency']['p95_latency_ms']
        if rebalancing_p95 > max_acceptable_rebalancing_p95_ms:
            raise AssertionError(f"Rebalancing P95 latency {rebalancing_p95:.2f}ms exceeds {max_acceptable_rebalancing_p95_ms}ms")
        
        return results
    
    def _test_burst_calculation_latency(self) -> Dict[str, Any]:
        """Test latency under burst calculation loads."""
        results = {}
        
        config = KellyConfig(total_capital=500000)
        sizer = KellyCriterion(config)
        
        # Test burst sizes
        burst_sizes = [10, 50, 100, 500]
        
        for burst_size in burst_sizes:
            # Measure time to complete entire burst
            start_time = time.perf_counter()
            
            burst_latencies = []
            for i in range(burst_size):
                operation_start = time.perf_counter()
                
                sizer.calculate_position_size(
                    strategy_name=f"Burst_{i}",
                    expected_return=random.uniform(0.05, 0.25),
                    volatility=random.uniform(0.10, 0.40)
                )
                
                operation_end = time.perf_counter()
                burst_latencies.append((operation_end - operation_start) * 1000)
            
            end_time = time.perf_counter()
            total_burst_time = (end_time - start_time) * 1000  # ms
            
            results[f'burst_{burst_size}'] = {
                'total_burst_time_ms': total_burst_time,
                'avg_operation_latency_ms': statistics.mean(burst_latencies),
                'p95_operation_latency_ms': np.percentile(burst_latencies, 95),
                'p99_operation_latency_ms': np.percentile(burst_latencies, 99),
                'throughput_ops_per_sec': burst_size / (total_burst_time / 1000),
                'latency_degradation': np.percentile(burst_latencies, 95) / burst_latencies[0] if burst_latencies[0] > 0 else 1.0
            }
        
        # Test for latency degradation under burst load
        small_burst_p95 = results['burst_10']['p95_operation_latency_ms']
        large_burst_p95 = results['burst_500']['p95_operation_latency_ms']
        
        max_degradation_factor = 3.0  # Latency shouldn't degrade more than 3x
        actual_degradation = large_burst_p95 / small_burst_p95 if small_burst_p95 > 0 else 1.0
        
        if actual_degradation > max_degradation_factor:
            raise AssertionError(f"Burst latency degradation {actual_degradation:.1f}x exceeds {max_degradation_factor}x")
        
        results['degradation_analysis'] = {
            'small_burst_p95_ms': small_burst_p95,
            'large_burst_p95_ms': large_burst_p95,
            'degradation_factor': actual_degradation
        }
        
        return results
    
    def _test_concurrent_operation_latency(self) -> Dict[str, Any]:
        """Test latency under concurrent operations."""
        results = {}
        
        config = KellyConfig(total_capital=200000)
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        operations_per_thread = 50
        
        for concurrency in concurrency_levels:
            def worker_task(worker_id: int):
                sizer = KellyCriterion(config)  # Each worker gets its own instance
                worker_latencies = []
                
                for i in range(operations_per_thread):
                    start_time = time.perf_counter()
                    
                    sizer.calculate_position_size(
                        strategy_name=f"Concurrent_{worker_id}_{i}",
                        expected_return=random.uniform(0.05, 0.20),
                        volatility=random.uniform(0.10, 0.30)
                    )
                    
                    end_time = time.perf_counter()
                    worker_latencies.append((end_time - start_time) * 1000)
                
                return worker_latencies
            
            # Execute concurrent operations
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(worker_task, i) for i in range(concurrency)]
                
                all_latencies = []
                for future in as_completed(futures):
                    worker_latencies = future.result()
                    all_latencies.extend(worker_latencies)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            results[f'concurrency_{concurrency}'] = {
                'total_operations': len(all_latencies),
                'total_time_sec': total_time,
                'throughput_ops_per_sec': len(all_latencies) / total_time,
                'mean_latency_ms': statistics.mean(all_latencies),
                'p95_latency_ms': np.percentile(all_latencies, 95),
                'p99_latency_ms': np.percentile(all_latencies, 99),
                'max_latency_ms': max(all_latencies),
                'concurrency_level': concurrency
            }
        
        # Analyze concurrency impact on latency
        single_thread_p95 = results['concurrency_1']['p95_latency_ms']
        high_concurrency_p95 = results[f'concurrency_{max(concurrency_levels)}']['p95_latency_ms']
        
        latency_increase_factor = high_concurrency_p95 / single_thread_p95 if single_thread_p95 > 0 else 1.0
        
        max_acceptable_increase = 5.0  # 5x latency increase under high concurrency
        if latency_increase_factor > max_acceptable_increase:
            raise AssertionError(f"Concurrent latency increase {latency_increase_factor:.1f}x exceeds {max_acceptable_increase}x")
        
        results['concurrency_analysis'] = {
            'single_thread_p95_ms': single_thread_p95,
            'high_concurrency_p95_ms': high_concurrency_p95,
            'latency_increase_factor': latency_increase_factor
        }
        
        return results
    
    def _test_data_alignment_latency(self) -> Dict[str, Any]:
        """Test data temporal alignment latency."""
        results = {}
        
        config = TemporalAlignerConfig(
            target_frequency=FrequencyType.DAILY,
            alignment_method='outer',
            fill_method='forward_fill'
        )
        aligner = TemporalAligner(config)
        
        # Test with different data sizes
        data_sizes = [100, 1000, 10000]
        
        for size in data_sizes:
            # Generate test data
            dates = pd.date_range('2024-01-01', periods=size, freq='H')
            test_data = pd.DataFrame({
                'value': np.random.randn(size),
                'volume': np.random.randint(100, 1000, size)
            }, index=dates)
            
            def alignment_operation():
                # Simplified alignment operation (resampling)
                return test_data.resample('D').mean()
            
            latency_stats = self._measure_multiple_operations(alignment_operation, 10)
            
            results[f'data_size_{size}'] = latency_stats
            results[f'data_size_{size}']['throughput_rows_per_sec'] = size / (latency_stats['mean_latency_ms'] / 1000)
        
        # Validate data processing latency
        large_data_p95 = results['data_size_10000']['p95_latency_ms']
        max_acceptable_large_data_p95_ms = 200.0  # 200ms for 10k rows
        
        if large_data_p95 > max_acceptable_large_data_p95_ms:
            raise AssertionError(f"Large data alignment P95 latency {large_data_p95:.2f}ms exceeds {max_acceptable_large_data_p95_ms}ms")
        
        return results
    
    def _test_realtime_processing_latency(self) -> Dict[str, Any]:
        """Test real-time data processing latency."""
        results = {}
        
        # Simulate real-time market data processing
        config = KellyConfig(total_capital=100000)
        sizer = KellyCriterion(config)
        
        # Test continuous processing with timing constraints
        processing_duration = 10.0  # 10 seconds of continuous processing
        target_frequency_hz = 100    # 100 operations per second target
        
        operation_times = []
        processing_latencies = []
        
        start_time = time.perf_counter()
        operation_count = 0
        
        while (time.perf_counter() - start_time) < processing_duration:
            operation_start = time.perf_counter()
            
            # Simulate market data processing + position sizing
            market_return = random.uniform(0.05, 0.20)
            market_vol = random.uniform(0.10, 0.30)
            
            result = sizer.calculate_position_size(
                strategy_name=f"RealTime_{operation_count}",
                expected_return=market_return,
                volatility=market_vol
            )
            
            operation_end = time.perf_counter()
            
            operation_latency = (operation_end - operation_start) * 1000  # ms
            processing_latencies.append(operation_latency)
            operation_times.append(operation_end)
            operation_count += 1
            
            # Simulate target frequency (sleep if we're ahead)
            target_interval = 1.0 / target_frequency_hz
            elapsed = operation_end - operation_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
        
        total_time = time.perf_counter() - start_time
        actual_frequency = operation_count / total_time
        
        results['realtime_processing'] = {
            'target_frequency_hz': target_frequency_hz,
            'actual_frequency_hz': actual_frequency,
            'frequency_achievement_rate': actual_frequency / target_frequency_hz,
            'total_operations': operation_count,
            'total_time_sec': total_time,
            'mean_latency_ms': statistics.mean(processing_latencies),
            'p95_latency_ms': np.percentile(processing_latencies, 95),
            'p99_latency_ms': np.percentile(processing_latencies, 99),
            'max_latency_ms': max(processing_latencies)
        }
        
        # Validate real-time processing requirements
        min_frequency_achievement = 0.95  # Should achieve 95% of target frequency
        max_p95_latency_ms = 15.0  # 15ms P95 latency for real-time processing
        
        if results['realtime_processing']['frequency_achievement_rate'] < min_frequency_achievement:
            raise AssertionError(f"Real-time frequency achievement {results['realtime_processing']['frequency_achievement_rate']:.1%} below {min_frequency_achievement:.1%}")
        
        if results['realtime_processing']['p95_latency_ms'] > max_p95_latency_ms:
            raise AssertionError(f"Real-time P95 latency {results['realtime_processing']['p95_latency_ms']:.2f}ms exceeds {max_p95_latency_ms}ms")
        
        return results
    
    def _test_cold_start_latency(self) -> Dict[str, Any]:
        """Test system cold start latency."""
        results = {}
        
        # Test cold start times for different components
        cold_start_tests = {}
        
        # Test 1: Risk management cold start
        def risk_cold_start():
            config = KellyConfig(total_capital=100000)
            sizer = KellyCriterion(config)
            return sizer.calculate_position_size("ColdStart", 0.12, 0.18)
        
        cold_start_latency, _ = self._measure_operation_latency(risk_cold_start)
        cold_start_tests['risk_management'] = cold_start_latency
        
        # Test 2: Portfolio management cold start
        def portfolio_cold_start():
            config = PortfolioConfig(total_capital=500000, max_strategies=5)
            portfolio = PortfolioManager(config)
            
            class MockStrategy:
                def __init__(self, name): self.name = name
            
            portfolio.add_strategy("ColdStart", MockStrategy("ColdStart"), 1.0)
            return portfolio.get_portfolio_summary()
        
        cold_start_latency, _ = self._measure_operation_latency(portfolio_cold_start)
        cold_start_tests['portfolio_management'] = cold_start_latency
        
        # Test 3: Multi-strategy cold start
        def multi_strategy_cold_start():
            config = MultiStrategyConfig(total_capital=1000000, max_total_risk=0.20)
            multi_sizer = MultiStrategyPositionSizer(config)
            
            from ..risk.position_sizing import PositionSizeResult
            result = PositionSizeResult(
                strategy_name="ColdStart",
                recommended_size=50000,
                max_position_value=100000,
                risk_adjusted_size=45000,
                confidence_level=0.8,
                reasoning="Cold start test"
            )
            multi_sizer.add_strategy_sizing(result)
            return multi_sizer.optimize_portfolio_allocation()
        
        cold_start_latency, _ = self._measure_operation_latency(multi_strategy_cold_start)
        cold_start_tests['multi_strategy'] = cold_start_latency
        
        results['cold_start_latencies'] = cold_start_tests
        results['max_cold_start_ms'] = max(cold_start_tests.values())
        results['avg_cold_start_ms'] = statistics.mean(cold_start_tests.values())
        
        # Validate cold start requirements
        max_acceptable_cold_start_ms = 100.0  # 100ms max cold start
        if results['max_cold_start_ms'] > max_acceptable_cold_start_ms:
            raise AssertionError(f"Cold start latency {results['max_cold_start_ms']:.2f}ms exceeds {max_acceptable_cold_start_ms}ms")
        
        return results
    
    def _test_warm_operation_latency(self) -> Dict[str, Any]:
        """Test warm system operation latency."""
        results = {}
        
        # Pre-warm the systems
        config = KellyConfig(total_capital=100000)
        sizer = KellyCriterion(config)
        
        portfolio_config = PortfolioConfig(total_capital=500000, max_strategies=5)
        portfolio = PortfolioManager(portfolio_config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        # Warm up with some operations
        for i in range(10):
            sizer.calculate_position_size(f"Warmup_{i}", 0.10, 0.20)
            
        portfolio.add_strategy("Warmup", MockStrategy("Warmup"), 1.0)
        portfolio.get_portfolio_summary()
        
        # Now test warm operation latency
        warm_operations = {
            'risk_calculation': lambda: sizer.calculate_position_size("WarmTest", 0.12, 0.18),
            'portfolio_metrics': lambda: portfolio.calculate_portfolio_metrics(),
            'portfolio_summary': lambda: portfolio.get_portfolio_summary()
        }
        
        iterations = 100
        
        for op_name, operation in warm_operations.items():
            latency_stats = self._measure_multiple_operations(operation, iterations)
            results[f'{op_name}_warm'] = latency_stats
        
        # Validate warm operation latency
        max_warm_p95_ms = 3.0  # 3ms P95 for warm operations
        
        for op_name in warm_operations.keys():
            p95_latency = results[f'{op_name}_warm']['p95_latency_ms']
            if p95_latency > max_warm_p95_ms:
                raise AssertionError(f"Warm {op_name} P95 latency {p95_latency:.2f}ms exceeds {max_warm_p95_ms}ms")
        
        return results
    
    def _test_market_tick_processing_latency(self) -> Dict[str, Any]:
        """Test market tick processing latency."""
        results = {}
        
        # Simulate high-frequency market tick processing
        config = KellyConfig(total_capital=200000)
        sizer = KellyCriterion(config)
        
        # Generate market ticks
        num_ticks = 10000
        tick_processing_latencies = []
        
        for tick_id in range(num_ticks):
            # Simulate market tick data
            tick_price = 100 + random.uniform(-5, 5)
            tick_volume = random.randint(100, 1000)
            tick_timestamp = time.perf_counter()
            
            # Measure processing latency
            processing_start = time.perf_counter()
            
            # Process tick (simulate strategy decision)
            expected_return = random.uniform(0.05, 0.15)
            volatility = random.uniform(0.10, 0.25)
            
            position_result = sizer.calculate_position_size(
                strategy_name=f"Tick_{tick_id}",
                expected_return=expected_return,
                volatility=volatility
            )
            
            processing_end = time.perf_counter()
            
            tick_latency = (processing_end - processing_start) * 1000000  # microseconds
            tick_processing_latencies.append(tick_latency)
            
            # Simulate tick frequency (1000 Hz)
            if tick_id % 1000 == 0:
                time.sleep(0.001)  # Brief pause every 1000 ticks
        
        results['tick_processing'] = {
            'total_ticks': num_ticks,
            'mean_latency_us': statistics.mean(tick_processing_latencies),
            'median_latency_us': statistics.median(tick_processing_latencies),
            'p95_latency_us': np.percentile(tick_processing_latencies, 95),
            'p99_latency_us': np.percentile(tick_processing_latencies, 99),
            'max_latency_us': max(tick_processing_latencies),
            'min_latency_us': min(tick_processing_latencies)
        }
        
        # Validate tick processing latency requirements
        max_p95_latency_us = 1000.0  # 1ms (1000 microseconds) P95 latency
        if results['tick_processing']['p95_latency_us'] > max_p95_latency_us:
            raise AssertionError(f"Tick processing P95 latency {results['tick_processing']['p95_latency_us']:.0f}μs exceeds {max_p95_latency_us:.0f}μs")
        
        return results
    
    def _test_order_decision_latency(self) -> Dict[str, Any]:
        """Test order decision making latency."""
        results = {}
        
        # Simulate complete order decision workflow
        kelly_config = KellyConfig(total_capital=500000)
        kelly_sizer = KellyCriterion(kelly_config)
        
        portfolio_config = PortfolioConfig(total_capital=500000, max_strategies=3)
        portfolio = PortfolioManager(portfolio_config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        # Setup portfolio
        for i in range(3):
            portfolio.add_strategy(f"OrderDecision_{i}", MockStrategy(f"OrderDecision_{i}"), 1.0/3)
        
        # Test order decision pipeline latency
        num_decisions = 1000
        decision_latencies = []
        
        for decision_id in range(num_decisions):
            decision_start = time.perf_counter()
            
            # Step 1: Market data analysis (simulated)
            market_signal = random.choice(['buy', 'sell', 'hold'])
            market_strength = random.uniform(0.1, 1.0)
            
            # Step 2: Risk calculation
            expected_return = random.uniform(0.05, 0.20) * market_strength
            volatility = random.uniform(0.10, 0.30)
            
            position_result = kelly_sizer.calculate_position_size(
                strategy_name=f"Decision_{decision_id}",
                expected_return=expected_return,
                volatility=volatility
            )
            
            # Step 3: Portfolio constraint check
            current_portfolio_value = sum(
                s.current_allocation * portfolio_config.total_capital 
                for s in portfolio.strategies.values()
            )
            
            # Step 4: Position size adjustment based on portfolio
            available_capital = portfolio_config.total_capital - current_portfolio_value
            adjusted_position = min(position_result.recommended_size, available_capital * 0.5)
            
            # Step 5: Order decision
            if market_signal == 'buy' and adjusted_position > 1000:
                order_decision = {'action': 'buy', 'size': adjusted_position}
            elif market_signal == 'sell' and adjusted_position > 1000:
                order_decision = {'action': 'sell', 'size': adjusted_position}
            else:
                order_decision = {'action': 'hold', 'size': 0}
            
            decision_end = time.perf_counter()
            
            decision_latency = (decision_end - decision_start) * 1000  # ms
            decision_latencies.append(decision_latency)
        
        results['order_decision'] = {
            'total_decisions': num_decisions,
            'mean_latency_ms': statistics.mean(decision_latencies),
            'median_latency_ms': statistics.median(decision_latencies),
            'p95_latency_ms': np.percentile(decision_latencies, 95),
            'p99_latency_ms': np.percentile(decision_latencies, 99),
            'max_latency_ms': max(decision_latencies),
            'decisions_per_second': num_decisions / (sum(decision_latencies) / 1000)
        }
        
        # Validate order decision latency requirements
        max_p95_decision_latency_ms = 10.0  # 10ms P95 for order decisions
        if results['order_decision']['p95_latency_ms'] > max_p95_decision_latency_ms:
            raise AssertionError(f"Order decision P95 latency {results['order_decision']['p95_latency_ms']:.2f}ms exceeds {max_p95_decision_latency_ms}ms")
        
        return results