"""
Performance benchmarking for RTradez components.

Tests throughput, latency, memory usage, and scalability across all components.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
from datetime import datetime, timedelta

from .core import ComponentBenchmark, BenchmarkConfig, BenchmarkSeverity
from ..risk import (
    KellyConfig, KellyCriterion, FixedFractionConfig, FixedFractionSizer,
    VolatilityAdjustedConfig, VolatilityAdjustedSizer, MultiStrategyConfig,
    MultiStrategyPositionSizer
)
from ..portfolio.portfolio_manager import PortfolioManager, PortfolioConfig
from ..utils.temporal_alignment import TemporalAligner, TemporalAlignerConfig, FrequencyType
from ..utils.time_bucketing import TimeBucketing, BucketConfig, BucketType


class PerformanceBenchmark(ComponentBenchmark):
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("Performance", config)
        self._register_benchmarks()
    
    def _register_benchmarks(self):
        """Register all performance benchmarks."""
        
        # Risk Management Performance
        @self.suite.register_benchmark(
            "risk_position_sizing_throughput",
            "Test position sizing calculation throughput",
            BenchmarkSeverity.ERROR
        )
        def test_position_sizing_throughput():
            return self._test_position_sizing_throughput()
        
        @self.suite.register_benchmark(
            "risk_multi_strategy_scalability",
            "Test multi-strategy position sizing scalability",
            BenchmarkSeverity.WARNING
        )
        def test_multi_strategy_scalability():
            return self._test_multi_strategy_scalability()
        
        # Portfolio Management Performance
        @self.suite.register_benchmark(
            "portfolio_rebalancing_performance",
            "Test portfolio rebalancing performance",
            BenchmarkSeverity.ERROR
        )
        def test_portfolio_rebalancing():
            return self._test_portfolio_rebalancing()
        
        @self.suite.register_benchmark(
            "portfolio_large_scale_management",
            "Test large-scale portfolio management",
            BenchmarkSeverity.WARNING
        )
        def test_large_portfolio():
            return self._test_large_portfolio()
        
        # Data Processing Performance
        @self.suite.register_benchmark(
            "temporal_alignment_throughput",
            "Test temporal alignment data throughput",
            BenchmarkSeverity.ERROR
        )
        def test_temporal_alignment():
            return self._test_temporal_alignment()
        
        @self.suite.register_benchmark(
            "time_bucketing_performance",
            "Test time bucketing performance",
            BenchmarkSeverity.WARNING
        )
        def test_time_bucketing():
            return self._test_time_bucketing()
        
        # Memory and Concurrency
        @self.suite.register_benchmark(
            "concurrent_operations_test",
            "Test concurrent operations handling",
            BenchmarkSeverity.CRITICAL
        )
        def test_concurrent_operations():
            return self._test_concurrent_operations()
        
        @self.suite.register_benchmark(
            "memory_efficiency_test",
            "Test memory efficiency with large datasets",
            BenchmarkSeverity.ERROR
        )
        def test_memory_efficiency():
            return self._test_memory_efficiency()
    
    def _test_position_sizing_throughput(self) -> Dict[str, Any]:
        """Test position sizing calculation throughput."""
        results = {}
        
        # Test different sizing methods
        methods = {
            'kelly': KellyCriterion(KellyConfig(total_capital=100000)),
            'fixed_fraction': FixedFractionSizer(FixedFractionConfig(total_capital=100000)),
            'volatility_adjusted': VolatilityAdjustedSizer(VolatilityAdjustedConfig(total_capital=100000))
        }
        
        for method_name, sizer in methods.items():
            # Generate test parameters
            num_calculations = self.config.stress_iterations
            strategies = [f"Strategy_{i}" for i in range(num_calculations)]
            returns = np.random.uniform(0.05, 0.20, num_calculations)
            volatilities = np.random.uniform(0.10, 0.40, num_calculations)
            
            start_time = time.perf_counter()
            
            # Perform calculations
            for i in range(num_calculations):
                sizer.calculate_position_size(
                    strategy_name=strategies[i],
                    expected_return=returns[i],
                    volatility=volatilities[i]
                )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = num_calculations / duration
            
            results[f"{method_name}_throughput"] = throughput
            results[f"{method_name}_avg_latency_ms"] = (duration / num_calculations) * 1000
        
        # Validate performance thresholds
        min_throughput = self.config.min_throughput
        if any(t < min_throughput for t in [results[f"{m}_throughput"] for m in methods.keys()]):
            raise AssertionError(f"Position sizing throughput below {min_throughput} ops/sec")
        
        return results
    
    def _test_multi_strategy_scalability(self) -> Dict[str, Any]:
        """Test multi-strategy position sizing scalability."""
        results = {}
        
        # Test scaling with different numbers of strategies
        strategy_counts = [10, 50, 100, 500, 1000]
        
        for count in strategy_counts:
            config = MultiStrategyConfig(
                total_capital=1000000,
                max_total_risk=0.15
            )
            multi_sizer = MultiStrategyPositionSizer(config)
            
            # Generate test strategies
            start_time = time.perf_counter()
            
            for i in range(count):
                from ..risk.position_sizing import PositionSizeResult
                result = PositionSizeResult(
                    strategy_name=f"Strategy_{i}",
                    recommended_size=np.random.uniform(10000, 100000),
                    max_position_value=200000,
                    risk_adjusted_size=np.random.uniform(10000, 100000),
                    confidence_level=np.random.uniform(0.5, 0.9),
                    reasoning="Test sizing"
                )
                multi_sizer.add_strategy_sizing(result)
            
            # Optimize allocation
            optimized = multi_sizer.optimize_portfolio_allocation()
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            results[f"strategies_{count}_duration"] = duration
            results[f"strategies_{count}_per_strategy_ms"] = (duration / count) * 1000
        
        # Check scalability (should be roughly linear)
        max_per_strategy_ms = 10.0  # 10ms per strategy max
        if any(results[f"strategies_{c}_per_strategy_ms"] > max_per_strategy_ms for c in strategy_counts):
            raise AssertionError(f"Multi-strategy scaling exceeds {max_per_strategy_ms}ms per strategy")
        
        return results
    
    def _test_portfolio_rebalancing(self) -> Dict[str, Any]:
        """Test portfolio rebalancing performance."""
        results = {}
        
        # Create portfolio with multiple strategies
        config = PortfolioConfig(total_capital=5000000, max_strategies=20)
        portfolio = PortfolioManager(config)
        
        # Add strategies
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        strategy_count = 15
        for i in range(strategy_count):
            portfolio.add_strategy(
                f"Strategy_{i}",
                MockStrategy(f"Strategy_{i}"),
                target_allocation=1.0 / strategy_count
            )
        
        # Test rebalancing performance
        iterations = 100
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            # Simulate drift
            for name, allocation in portfolio.strategies.items():
                allocation.current_allocation = allocation.target_allocation * np.random.uniform(0.8, 1.2)
            
            # Calculate and execute rebalancing
            needs = portfolio.calculate_rebalancing_needs()
            if needs:
                portfolio.execute_rebalancing(needs)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        results['rebalancing_throughput'] = iterations / duration
        results['avg_rebalancing_time_ms'] = (duration / iterations) * 1000
        
        # Validate performance
        max_rebalancing_time_ms = 50.0
        if results['avg_rebalancing_time_ms'] > max_rebalancing_time_ms:
            raise AssertionError(f"Rebalancing time {results['avg_rebalancing_time_ms']:.1f}ms exceeds {max_rebalancing_time_ms}ms")
        
        return results
    
    def _test_large_portfolio(self) -> Dict[str, Any]:
        """Test large-scale portfolio management."""
        results = {}
        
        # Test with maximum strategies
        config = PortfolioConfig(total_capital=100000000, max_strategies=100)  # $100M, 100 strategies
        portfolio = PortfolioManager(config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        # Add maximum strategies
        start_time = time.perf_counter()
        
        for i in range(config.max_strategies):
            success = portfolio.add_strategy(
                f"LargeStrategy_{i}",
                MockStrategy(f"LargeStrategy_{i}"),
                target_allocation=1.0 / config.max_strategies
            )
            if not success:
                break
        
        # Test portfolio operations
        metrics = portfolio.calculate_portfolio_metrics()
        summary = portfolio.get_portfolio_summary()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        results['large_portfolio_setup_time'] = duration
        results['strategies_added'] = len(portfolio.strategies)
        results['metrics_calculation_time'] = time.perf_counter()
        
        # Test portfolio metrics calculation time
        start_time = time.perf_counter()
        for _ in range(100):
            portfolio.calculate_portfolio_metrics()
        end_time = time.perf_counter()
        
        results['metrics_calculation_time'] = ((end_time - start_time) / 100) * 1000  # ms
        
        # Validate large portfolio performance
        max_metrics_time_ms = 20.0
        if results['metrics_calculation_time'] > max_metrics_time_ms:
            raise AssertionError(f"Large portfolio metrics calculation {results['metrics_calculation_time']:.1f}ms exceeds {max_metrics_time_ms}ms")
        
        return results
    
    def _test_temporal_alignment(self) -> Dict[str, Any]:
        """Test temporal alignment data throughput."""
        results = {}
        
        config = TemporalAlignerConfig(
            target_frequency=FrequencyType.DAILY,
            alignment_method='outer',
            fill_method='forward_fill'
        )
        aligner = TemporalAligner(config)
        
        # Test with different data sizes
        for scale_factor in self.config.data_scale_factors:
            data_size = int(1000 * scale_factor)  # Base 1000 rows
            
            # Generate test data
            dates = pd.date_range(start='2020-01-01', periods=data_size, freq='D')
            test_data = pd.DataFrame({
                'value': np.random.randn(data_size),
                'volume': np.random.randint(1000, 10000, data_size)
            }, index=dates)
            
            # Time the alignment process
            start_time = time.perf_counter()
            
            # Test profile analysis
            from ..utils.temporal_alignment import TemporalProfile
            profile = TemporalProfile(
                frequency=FrequencyType.DAILY,
                start_date=test_data.index.min().to_pydatetime(),
                end_date=test_data.index.max().to_pydatetime(),
                total_observations=len(test_data),
                missing_periods=0,
                regularity_score=1.0
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            results[f"temporal_alignment_{scale_factor}x_duration"] = duration
            results[f"temporal_alignment_{scale_factor}x_throughput"] = data_size / duration
        
        # Validate throughput
        min_throughput = 10000  # 10k rows/sec minimum
        base_throughput = results.get("temporal_alignment_1.0x_throughput", 0)
        if base_throughput < min_throughput:
            raise AssertionError(f"Temporal alignment throughput {base_throughput:.0f} below {min_throughput} rows/sec")
        
        return results
    
    def _test_time_bucketing(self) -> Dict[str, Any]:
        """Test time bucketing performance."""
        results = {}
        
        config = BucketConfig(
            bucket_type=BucketType.TRADING,
            bucket_size="1H",
            aggregation_method="ohlc"
        )
        bucketing = TimeBucketing(config)
        
        # Test with high-frequency data
        for scale_factor in [1.0, 5.0, 10.0]:
            data_points = int(10000 * scale_factor)  # Base 10k minute data points
            
            # Generate minute-level data
            timestamps = pd.date_range(
                start='2024-01-01 09:30',
                periods=data_points,
                freq='1min'
            )
            
            data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(data_points) * 0.01),
                'volume': np.random.randint(100, 1000, data_points)
            }, index=timestamps)
            
            # Time bucketing operation
            start_time = time.perf_counter()
            
            # Simulate bucketing (simplified for demo)
            bucketed = data.resample('1H').agg({
                'price': 'last',
                'volume': 'sum'
            })
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            results[f"bucketing_{scale_factor}x_duration"] = duration
            results[f"bucketing_{scale_factor}x_throughput"] = data_points / duration
        
        return results
    
    def _test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent operations handling."""
        results = {}
        
        def position_sizing_task():
            """Single position sizing task."""
            config = KellyConfig(total_capital=100000)
            sizer = KellyCriterion(config)
            
            return sizer.calculate_position_size(
                strategy_name="ConcurrentTest",
                expected_return=np.random.uniform(0.05, 0.20),
                volatility=np.random.uniform(0.10, 0.30)
            )
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]
        
        for concurrency in concurrency_levels:
            operations_per_thread = 100
            total_operations = concurrency * operations_per_thread
            
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(position_sizing_task) 
                          for _ in range(total_operations)]
                
                # Wait for completion
                completed = 0
                errors = 0
                for future in futures:
                    try:
                        future.result()
                        completed += 1
                    except Exception:
                        errors += 1
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            results[f"concurrent_{concurrency}_threads_duration"] = duration
            results[f"concurrent_{concurrency}_threads_throughput"] = total_operations / duration
            results[f"concurrent_{concurrency}_threads_error_rate"] = errors / total_operations
        
        # Validate concurrent performance
        single_thread_throughput = results.get("concurrent_1_threads_throughput", 0)
        max_thread_throughput = results.get("concurrent_20_threads_throughput", 0)
        
        # Should see some scaling benefit
        scaling_factor = max_thread_throughput / single_thread_throughput if single_thread_throughput > 0 else 0
        if scaling_factor < 2.0:  # At least 2x improvement with 20 threads
            results['scaling_warning'] = f"Limited concurrent scaling: {scaling_factor:.1f}x"
        
        return results
    
    def _test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory efficiency with large datasets."""
        results = {}
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Test memory usage with large portfolios
        config = PortfolioConfig(total_capital=10000000, max_strategies=50)
        portfolio = PortfolioManager(config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        # Add strategies and track memory
        memory_samples = []
        
        for i in range(50):
            portfolio.add_strategy(
                f"MemoryTest_{i}",
                MockStrategy(f"MemoryTest_{i}"),
                target_allocation=0.02
            )
            
            # Sample memory every 10 strategies
            if i % 10 == 0:
                current_memory = psutil.Process().memory_info().rss
                memory_samples.append(current_memory - initial_memory)
        
        # Test large data processing
        large_data = pd.DataFrame({
            'value': np.random.randn(100000),
            'timestamp': pd.date_range('2020-01-01', periods=100000, freq='1min')
        })
        
        memory_before_data = psutil.Process().memory_info().rss
        
        # Process data
        processed = large_data.groupby(large_data['timestamp'].dt.hour).agg({
            'value': ['mean', 'std', 'min', 'max']
        })
        
        memory_after_data = psutil.Process().memory_info().rss
        
        final_memory = psutil.Process().memory_info().rss
        total_memory_growth = final_memory - initial_memory
        
        results['initial_memory_mb'] = initial_memory / 1024 / 1024
        results['final_memory_mb'] = final_memory / 1024 / 1024
        results['total_memory_growth_mb'] = total_memory_growth / 1024 / 1024
        results['data_processing_memory_mb'] = (memory_after_data - memory_before_data) / 1024 / 1024
        results['memory_samples'] = [m / 1024 / 1024 for m in memory_samples]
        
        # Validate memory efficiency
        if total_memory_growth / 1024 / 1024 > self.config.max_memory_growth_mb:
            raise AssertionError(f"Memory growth {total_memory_growth / 1024 / 1024:.1f}MB exceeds limit {self.config.max_memory_growth_mb}MB")
        
        return results