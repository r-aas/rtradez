"""
Memory profiling and benchmarking framework for RTradez components.

Tests memory usage patterns, leak detection, garbage collection behavior,
and memory efficiency under various load conditions.
"""

import time
import gc
import tracemalloc
import sys
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import threading
import weakref
from pathlib import Path
import pickle
import json

from .core import ComponentBenchmark, BenchmarkConfig, BenchmarkSeverity
from ..risk import (
    KellyConfig, KellyCriterion, FixedFractionConfig, FixedFractionSizer,
    VolatilityAdjustedConfig, VolatilityAdjustedSizer, MultiStrategyConfig,
    MultiStrategyPositionSizer
)
from ..portfolio.portfolio_manager import PortfolioManager, PortfolioConfig
from ..utils.temporal_alignment import TemporalAligner, TemporalAlignerConfig, FrequencyType


class MemoryProfiler(ComponentBenchmark):
    """Comprehensive memory profiling and analysis framework."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("Memory", config)
        self.process = psutil.Process()
        self._register_memory_tests()
    
    def _register_memory_tests(self):
        """Register all memory profiling tests."""
        
        # Basic Memory Usage Tests
        @self.suite.register_benchmark(
            "memory_baseline_usage",
            "Test baseline memory usage of components",
            BenchmarkSeverity.INFO
        )
        def test_memory_baseline():
            return self._test_memory_baseline()
        
        @self.suite.register_benchmark(
            "memory_growth_patterns",
            "Test memory growth patterns under load",
            BenchmarkSeverity.ERROR
        )
        def test_memory_growth():
            return self._test_memory_growth_patterns()
        
        # Memory Leak Detection Tests
        @self.suite.register_benchmark(
            "memory_leak_detection",
            "Test for memory leaks in repeated operations",
            BenchmarkSeverity.CRITICAL
        )
        def test_memory_leaks():
            return self._test_memory_leak_detection()
        
        @self.suite.register_benchmark(
            "object_lifecycle_memory",
            "Test object lifecycle memory management",
            BenchmarkSeverity.ERROR
        )
        def test_object_lifecycle():
            return self._test_object_lifecycle_memory()
        
        # Garbage Collection Tests
        @self.suite.register_benchmark(
            "garbage_collection_efficiency",
            "Test garbage collection efficiency",
            BenchmarkSeverity.WARNING
        )
        def test_gc_efficiency():
            return self._test_garbage_collection_efficiency()
        
        @self.suite.register_benchmark(
            "gc_pressure_response",
            "Test system response to GC pressure",
            BenchmarkSeverity.WARNING
        )
        def test_gc_pressure():
            return self._test_gc_pressure_response()
        
        # Large Data Handling Tests
        @self.suite.register_benchmark(
            "large_dataset_memory",
            "Test memory usage with large datasets",
            BenchmarkSeverity.ERROR
        )
        def test_large_datasets():
            return self._test_large_dataset_memory()
        
        @self.suite.register_benchmark(
            "streaming_data_memory",
            "Test memory usage in streaming data scenarios",
            BenchmarkSeverity.ERROR
        )
        def test_streaming_memory():
            return self._test_streaming_data_memory()
        
        # Memory Efficiency Tests
        @self.suite.register_benchmark(
            "memory_allocation_efficiency",
            "Test memory allocation and deallocation efficiency",
            BenchmarkSeverity.WARNING
        )
        def test_allocation_efficiency():
            return self._test_allocation_efficiency()
        
        @self.suite.register_benchmark(
            "peak_memory_optimization",
            "Test peak memory usage optimization",
            BenchmarkSeverity.ERROR
        )
        def test_peak_memory():
            return self._test_peak_memory_optimization()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current detailed memory information."""
        mem = self.process.memory_info()
        vm = psutil.virtual_memory()
        
        return {
            'rss_mb': mem.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available_mb': vm.available / 1024 / 1024,
            'total_mb': vm.total / 1024 / 1024,
            'timestamp': time.perf_counter()
        }
    
    def _monitor_memory_during_operation(self, operation: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Monitor memory usage during operation execution."""
        tracemalloc.start()
        gc.collect()  # Clean start
        
        initial_memory = self._get_memory_info()
        memory_samples = [initial_memory]
        
        # Start background memory monitoring
        monitoring = True
        def memory_monitor():
            while monitoring:
                memory_samples.append(self._get_memory_info())
                time.sleep(0.01)  # 10ms sampling
        
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        
        try:
            # Execute operation
            start_time = time.perf_counter()
            result = operation(*args, **kwargs)
            end_time = time.perf_counter()
            
        finally:
            monitoring = False
            monitor_thread.join(timeout=1.0)
            
            # Get peak memory from tracemalloc
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            final_memory = self._get_memory_info()
            gc.collect()
            post_gc_memory = self._get_memory_info()
        
        # Analyze memory usage
        memory_analysis = {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'post_gc_memory': post_gc_memory,
            'peak_tracemalloc_mb': peak_mem / 1024 / 1024,
            'current_tracemalloc_mb': current_mem / 1024 / 1024,
            'memory_growth_mb': final_memory['rss_mb'] - initial_memory['rss_mb'],
            'memory_growth_after_gc_mb': post_gc_memory['rss_mb'] - initial_memory['rss_mb'],
            'peak_rss_mb': max(sample['rss_mb'] for sample in memory_samples),
            'execution_time': end_time - start_time,
            'memory_samples': memory_samples,
            'gc_collections': gc.get_stats()
        }
        
        return result, memory_analysis
    
    def _test_memory_baseline(self) -> Dict[str, Any]:
        """Test baseline memory usage of components."""
        results = {}
        
        # Test baseline memory for different components
        components = {}
        
        # Kelly Criterion baseline
        def kelly_baseline():
            config = KellyConfig(total_capital=100000)
            sizer = KellyCriterion(config)
            return sizer.calculate_position_size("Baseline", 0.10, 0.20)
        
        _, kelly_memory = self._monitor_memory_during_operation(kelly_baseline)
        components['kelly_criterion'] = kelly_memory
        
        # Portfolio Manager baseline
        def portfolio_baseline():
            config = PortfolioConfig(total_capital=500000, max_strategies=5)
            portfolio = PortfolioManager(config)
            
            class MockStrategy:
                def __init__(self, name): self.name = name
            
            portfolio.add_strategy("Baseline", MockStrategy("Baseline"), 1.0)
            return portfolio.get_portfolio_summary()
        
        _, portfolio_memory = self._monitor_memory_during_operation(portfolio_baseline)
        components['portfolio_manager'] = portfolio_memory
        
        # Multi-strategy baseline
        def multi_strategy_baseline():
            config = MultiStrategyConfig(total_capital=1000000, max_total_risk=0.20)
            multi_sizer = MultiStrategyPositionSizer(config)
            
            from ..risk.position_sizing import PositionSizeResult
            result = PositionSizeResult(
                strategy_name="Baseline",
                recommended_size=50000,
                max_position_value=100000,
                risk_adjusted_size=45000,
                confidence_level=0.8,
                reasoning="Baseline test"
            )
            multi_sizer.add_strategy_sizing(result)
            return multi_sizer.optimize_portfolio_allocation()
        
        _, multi_memory = self._monitor_memory_during_operation(multi_strategy_baseline)
        components['multi_strategy'] = multi_memory
        
        # Analyze baseline requirements
        for component_name, memory_info in components.items():
            results[f'{component_name}_baseline'] = {
                'memory_growth_mb': memory_info['memory_growth_mb'],
                'peak_memory_mb': memory_info['peak_rss_mb'],
                'tracemalloc_peak_mb': memory_info['peak_tracemalloc_mb'],
                'execution_time': memory_info['execution_time']
            }
        
        # Calculate aggregate baseline
        total_baseline_memory = sum(comp['memory_growth_mb'] for comp in components.values())
        max_component_memory = max(comp['peak_rss_mb'] for comp in components.values())
        
        results['aggregate_baseline'] = {
            'total_memory_growth_mb': total_baseline_memory,
            'max_component_peak_mb': max_component_memory,
            'components_tested': len(components)
        }
        
        # Validate baseline memory usage
        max_acceptable_component_memory_mb = 50.0  # 50MB per component max
        if max_component_memory > max_acceptable_component_memory_mb:
            raise AssertionError(f"Component baseline memory {max_component_memory:.1f}MB exceeds {max_acceptable_component_memory_mb}MB")
        
        return results
    
    def _test_memory_growth_patterns(self) -> Dict[str, Any]:
        """Test memory growth patterns under load."""
        results = {}
        
        config = KellyConfig(total_capital=200000)
        sizer = KellyCriterion(config)
        
        # Test memory growth with increasing operations
        operation_counts = [100, 500, 1000, 5000]
        growth_patterns = {}
        
        for count in operation_counts:
            def bulk_operations():
                results_list = []
                for i in range(count):
                    result = sizer.calculate_position_size(
                        strategy_name=f"Growth_{i}",
                        expected_return=np.random.uniform(0.05, 0.20),
                        volatility=np.random.uniform(0.10, 0.30)
                    )
                    results_list.append(result)
                return results_list
            
            _, memory_info = self._monitor_memory_during_operation(bulk_operations)
            
            growth_patterns[f'operations_{count}'] = {
                'memory_growth_mb': memory_info['memory_growth_mb'],
                'memory_growth_after_gc_mb': memory_info['memory_growth_after_gc_mb'],
                'peak_memory_mb': memory_info['peak_rss_mb'],
                'memory_per_operation_kb': (memory_info['memory_growth_mb'] * 1024) / count,
                'operations': count
            }
        
        # Analyze growth linearity
        operations = [100, 500, 1000, 5000]
        memory_growths = [growth_patterns[f'operations_{op}']['memory_growth_mb'] for op in operations]
        
        # Check if growth is roughly linear (coefficient should be close to 1)
        if len(operations) > 1:
            correlation = np.corrcoef(operations, memory_growths)[0, 1]
        else:
            correlation = 1.0
        
        results['growth_patterns'] = growth_patterns
        results['linearity_correlation'] = correlation
        results['memory_efficiency_score'] = correlation  # Higher correlation = more predictable growth
        
        # Test memory growth stability
        max_growth_5k = growth_patterns['operations_5000']['memory_growth_mb']
        max_acceptable_growth_5k_mb = 100.0  # 100MB for 5000 operations
        
        if max_growth_5k > max_acceptable_growth_5k_mb:
            raise AssertionError(f"Memory growth for 5000 operations {max_growth_5k:.1f}MB exceeds {max_acceptable_growth_5k_mb}MB")
        
        return results
    
    def _test_memory_leak_detection(self) -> Dict[str, Any]:
        """Test for memory leaks in repeated operations."""
        results = {}
        
        # Test for leaks in repeated component creation/destruction
        leak_tests = {}
        
        # Test 1: Kelly Criterion leak test
        def kelly_leak_test():
            memory_samples = []
            initial_memory = self._get_memory_info()
            
            for cycle in range(10):  # 10 cycles
                # Create and use multiple instances
                for i in range(100):
                    config = KellyConfig(total_capital=100000 + i)
                    sizer = KellyCriterion(config)
                    result = sizer.calculate_position_size(f"Leak_{cycle}_{i}", 0.10, 0.20)
                    del sizer, config  # Explicit cleanup
                
                # Force garbage collection
                gc.collect()
                current_memory = self._get_memory_info()
                memory_samples.append(current_memory['rss_mb'] - initial_memory['rss_mb'])
            
            return memory_samples
        
        kelly_samples = kelly_leak_test()
        leak_tests['kelly_criterion'] = kelly_samples
        
        # Test 2: Portfolio Manager leak test
        def portfolio_leak_test():
            memory_samples = []
            initial_memory = self._get_memory_info()
            
            class MockStrategy:
                def __init__(self, name): self.name = name
            
            for cycle in range(10):
                for i in range(50):  # Fewer iterations for heavier objects
                    config = PortfolioConfig(total_capital=500000 + i * 1000, max_strategies=5)
                    portfolio = PortfolioManager(config)
                    
                    portfolio.add_strategy(f"Leak_{cycle}_{i}", MockStrategy(f"Leak_{cycle}_{i}"), 1.0)
                    summary = portfolio.get_portfolio_summary()
                    
                    del portfolio, config  # Explicit cleanup
                
                gc.collect()
                current_memory = self._get_memory_info()
                memory_samples.append(current_memory['rss_mb'] - initial_memory['rss_mb'])
            
            return memory_samples
        
        portfolio_samples = portfolio_leak_test()
        leak_tests['portfolio_manager'] = portfolio_samples
        
        # Analyze leak patterns
        for test_name, samples in leak_tests.items():
            # Calculate memory trend (should be flat if no leaks)
            if len(samples) > 1:
                x = np.arange(len(samples))
                slope, _ = np.polyfit(x, samples, 1)
                trend_mb_per_cycle = slope
            else:
                trend_mb_per_cycle = 0
            
            # Calculate memory stability
            memory_variance = np.var(samples) if len(samples) > 1 else 0
            
            leak_tests[f'{test_name}_analysis'] = {
                'memory_trend_mb_per_cycle': trend_mb_per_cycle,
                'memory_variance': memory_variance,
                'final_memory_growth_mb': samples[-1] if samples else 0,
                'max_memory_growth_mb': max(samples) if samples else 0,
                'samples': samples
            }
        
        results['leak_tests'] = leak_tests
        
        # Detect potential leaks
        leak_detected = False
        leak_threshold_mb_per_cycle = 1.0  # 1MB growth per cycle indicates potential leak
        
        for test_name in ['kelly_criterion', 'portfolio_manager']:
            trend = leak_tests[f'{test_name}_analysis']['memory_trend_mb_per_cycle']
            if abs(trend) > leak_threshold_mb_per_cycle:
                leak_detected = True
                results[f'{test_name}_leak_detected'] = True
        
        results['leak_detected'] = leak_detected
        
        if leak_detected:
            raise AssertionError("Potential memory leak detected in repeated operations")
        
        return results
    
    def _test_object_lifecycle_memory(self) -> Dict[str, Any]:
        """Test object lifecycle memory management."""
        results = {}
        
        # Test weak reference cleanup
        def test_weak_references():
            objects_created = []
            weak_refs = []
            
            # Create objects with weak references
            for i in range(1000):
                config = KellyConfig(total_capital=100000 + i)
                sizer = KellyCriterion(config)
                
                objects_created.append(sizer)
                weak_refs.append(weakref.ref(sizer))
            
            # Check all objects are alive
            alive_before = sum(1 for ref in weak_refs if ref() is not None)
            
            # Delete strong references
            del objects_created
            gc.collect()
            
            # Check weak references after cleanup
            alive_after = sum(1 for ref in weak_refs if ref() is not None)
            
            return {
                'objects_created': 1000,
                'alive_before_cleanup': alive_before,
                'alive_after_cleanup': alive_after,
                'cleanup_efficiency': (alive_before - alive_after) / alive_before if alive_before > 0 else 1.0
            }
        
        weak_ref_results = test_weak_references()
        results['weak_reference_cleanup'] = weak_ref_results
        
        # Test memory release after object destruction
        def test_memory_release():
            memory_before = self._get_memory_info()
            
            # Create large object structure
            large_objects = []
            for i in range(500):
                config = PortfolioConfig(total_capital=1000000 + i, max_strategies=10)
                portfolio = PortfolioManager(config)
                
                class MockStrategy:
                    def __init__(self, name): 
                        self.name = name
                        self.data = np.random.randn(1000)  # Add some data
                
                for j in range(5):
                    portfolio.add_strategy(f"Memory_{i}_{j}", MockStrategy(f"Memory_{i}_{j}"), 0.2)
                
                large_objects.append(portfolio)
            
            memory_peak = self._get_memory_info()
            
            # Delete objects
            del large_objects
            gc.collect()
            
            memory_after = self._get_memory_info()
            
            return {
                'memory_before_mb': memory_before['rss_mb'],
                'memory_peak_mb': memory_peak['rss_mb'],
                'memory_after_mb': memory_after['rss_mb'],
                'memory_allocated_mb': memory_peak['rss_mb'] - memory_before['rss_mb'],
                'memory_released_mb': memory_peak['rss_mb'] - memory_after['rss_mb'],
                'release_efficiency': (memory_peak['rss_mb'] - memory_after['rss_mb']) / (memory_peak['rss_mb'] - memory_before['rss_mb']) if memory_peak['rss_mb'] > memory_before['rss_mb'] else 1.0
            }
        
        memory_release_results = test_memory_release()
        results['memory_release'] = memory_release_results
        
        # Validate object lifecycle management
        min_cleanup_efficiency = 0.95  # 95% of objects should be cleaned up
        min_release_efficiency = 0.7   # 70% of memory should be released
        
        if weak_ref_results['cleanup_efficiency'] < min_cleanup_efficiency:
            raise AssertionError(f"Object cleanup efficiency {weak_ref_results['cleanup_efficiency']:.1%} below {min_cleanup_efficiency:.1%}")
        
        if memory_release_results['release_efficiency'] < min_release_efficiency:
            raise AssertionError(f"Memory release efficiency {memory_release_results['release_efficiency']:.1%} below {min_release_efficiency:.1%}")
        
        return results
    
    def _test_garbage_collection_efficiency(self) -> Dict[str, Any]:
        """Test garbage collection efficiency."""
        results = {}
        
        # Monitor GC behavior during operations
        # gc.set_debug(gc.DEBUG_STATS)  # Disabled for cleaner output
        initial_stats = gc.get_stats()
        
        def gc_stress_operations():
            # Create objects that will need garbage collection
            objects = []
            
            for i in range(1000):
                # Create circular references
                config = KellyConfig(total_capital=100000)
                sizer = KellyCriterion(config)
                
                # Create some complex object structures
                data = {
                    'sizer': sizer,
                    'results': [],
                    'metadata': {
                        'created_at': time.time(),
                        'iteration': i
                    }
                }
                
                # Add results that reference back to data
                for j in range(10):
                    result = sizer.calculate_position_size(f"GC_Test_{i}_{j}", 0.10, 0.20)
                    data['results'].append(result)
                
                objects.append(data)
                
                # Periodically trigger GC
                if i % 100 == 0:
                    gc.collect()
            
            return objects
        
        # Monitor memory and GC during operations
        memory_before = self._get_memory_info()
        start_time = time.perf_counter()
        
        objects = gc_stress_operations()
        
        memory_peak = self._get_memory_info()
        
        # Force full GC
        collected = gc.collect()
        
        memory_after_gc = self._get_memory_info()
        end_time = time.perf_counter()
        
        final_stats = gc.get_stats()
        # gc.set_debug(0)  # Turn off debug (already disabled)
        
        # Analyze GC efficiency
        results['gc_performance'] = {
            'execution_time': end_time - start_time,
            'objects_created': len(objects),
            'memory_before_mb': memory_before['rss_mb'],
            'memory_peak_mb': memory_peak['rss_mb'],
            'memory_after_gc_mb': memory_after_gc['rss_mb'],
            'memory_reclaimed_mb': memory_peak['rss_mb'] - memory_after_gc['rss_mb'],
            'objects_collected': collected,
            'initial_gc_stats': initial_stats,
            'final_gc_stats': final_stats
        }
        
        # Calculate GC efficiency metrics
        memory_allocated = memory_peak['rss_mb'] - memory_before['rss_mb']
        memory_reclaimed = memory_peak['rss_mb'] - memory_after_gc['rss_mb']
        
        gc_efficiency = memory_reclaimed / memory_allocated if memory_allocated > 0 else 1.0
        
        results['gc_efficiency'] = gc_efficiency
        results['gc_effectiveness'] = memory_reclaimed / len(objects) if len(objects) > 0 else 0  # MB per object
        
        # Clean up
        del objects
        gc.collect()
        
        # Validate GC efficiency
        min_gc_efficiency = 0.5  # Should reclaim at least 50% of allocated memory
        if gc_efficiency < min_gc_efficiency:
            raise AssertionError(f"GC efficiency {gc_efficiency:.1%} below {min_gc_efficiency:.1%}")
        
        return results
    
    def _test_gc_pressure_response(self) -> Dict[str, Any]:
        """Test system response to GC pressure."""
        results = {}
        
        # Test performance under different GC pressure levels
        gc_pressure_tests = {}
        
        # Low pressure test
        def low_pressure_test():
            config = KellyConfig(total_capital=100000)
            sizer = KellyCriterion(config)
            
            start_time = time.perf_counter()
            for i in range(100):
                result = sizer.calculate_position_size(f"LowPressure_{i}", 0.10, 0.20)
            end_time = time.perf_counter()
            
            return end_time - start_time
        
        low_pressure_time = low_pressure_test()
        gc_pressure_tests['low_pressure'] = low_pressure_time
        
        # High pressure test (frequent GC)
        def high_pressure_test():
            config = KellyConfig(total_capital=100000)
            sizer = KellyCriterion(config)
            
            start_time = time.perf_counter()
            for i in range(100):
                # Create temporary objects that will need collection
                temp_data = [np.random.randn(1000) for _ in range(10)]
                result = sizer.calculate_position_size(f"HighPressure_{i}", 0.10, 0.20)
                
                # Force frequent GC
                if i % 10 == 0:
                    gc.collect()
                
                del temp_data
            
            end_time = time.perf_counter()
            return end_time - start_time
        
        high_pressure_time = high_pressure_test()
        gc_pressure_tests['high_pressure'] = high_pressure_time
        
        # Calculate pressure impact
        pressure_impact = high_pressure_time / low_pressure_time if low_pressure_time > 0 else 1.0
        
        results['gc_pressure_tests'] = gc_pressure_tests
        results['pressure_impact_factor'] = pressure_impact
        
        # Validate GC pressure response
        max_pressure_impact = 3.0  # Performance shouldn't degrade more than 3x under GC pressure
        if pressure_impact > max_pressure_impact:
            raise AssertionError(f"GC pressure impact {pressure_impact:.1f}x exceeds {max_pressure_impact}x")
        
        return results
    
    def _test_large_dataset_memory(self) -> Dict[str, Any]:
        """Test memory usage with large datasets."""
        results = {}
        
        # Test memory usage with progressively larger datasets
        dataset_sizes = [10000, 50000, 100000]  # Number of rows
        
        for size in dataset_sizes:
            def large_dataset_processing():
                # Generate large dataset
                dates = pd.date_range('2020-01-01', periods=size, freq='1min')
                large_data = pd.DataFrame({
                    'price': 100 + np.cumsum(np.random.randn(size) * 0.01),
                    'volume': np.random.randint(100, 10000, size),
                    'returns': np.random.randn(size) * 0.02,
                    'volatility': np.random.uniform(0.1, 0.3, size)
                }, index=dates)
                
                # Process data
                processed_data = large_data.copy()
                processed_data['ma_short'] = processed_data['price'].rolling(10).mean()
                processed_data['ma_long'] = processed_data['price'].rolling(50).mean()
                processed_data['signals'] = (processed_data['ma_short'] > processed_data['ma_long']).astype(int)
                
                # Calculate some statistics
                daily_stats = processed_data.resample('D').agg({
                    'price': ['first', 'last', 'min', 'max'],
                    'volume': 'sum',
                    'returns': ['mean', 'std']
                })
                
                return processed_data, daily_stats
            
            _, memory_info = self._monitor_memory_during_operation(large_dataset_processing)
            
            results[f'dataset_size_{size}'] = {
                'size_rows': size,
                'memory_growth_mb': memory_info['memory_growth_mb'],
                'peak_memory_mb': memory_info['peak_rss_mb'],
                'memory_per_row_kb': (memory_info['memory_growth_mb'] * 1024) / size,
                'execution_time': memory_info['execution_time'],
                'memory_efficiency': size / memory_info['memory_growth_mb'] if memory_info['memory_growth_mb'] > 0 else float('inf')
            }
        
        # Analyze memory scaling
        sizes = [10000, 50000, 100000]
        memory_growths = [results[f'dataset_size_{size}']['memory_growth_mb'] for size in sizes]
        
        # Check memory scaling linearity
        if len(sizes) > 1:
            correlation = np.corrcoef(sizes, memory_growths)[0, 1]
        else:
            correlation = 1.0
        
        results['scaling_analysis'] = {
            'memory_scaling_correlation': correlation,
            'largest_dataset_memory_mb': memory_growths[-1],
            'memory_efficiency_trend': correlation
        }
        
        # Validate large dataset memory usage
        max_memory_100k_mb = 500.0  # 500MB for 100k rows max
        actual_memory_100k = results['dataset_size_100000']['memory_growth_mb']
        
        if actual_memory_100k > max_memory_100k_mb:
            raise AssertionError(f"Large dataset memory usage {actual_memory_100k:.1f}MB exceeds {max_memory_100k_mb}MB")
        
        return results
    
    def _test_streaming_data_memory(self) -> Dict[str, Any]:
        """Test memory usage in streaming data scenarios."""
        results = {}
        
        # Simulate streaming data processing
        config = KellyConfig(total_capital=200000)
        sizer = KellyCriterion(config)
        
        # Test streaming with memory management
        def streaming_simulation():
            memory_samples = []
            processed_items = 0
            
            # Simulate 1 hour of 1-second ticks
            for tick in range(3600):
                # Process incoming tick
                tick_data = {
                    'timestamp': time.time(),
                    'price': 100 + np.random.randn() * 0.1,
                    'volume': np.random.randint(100, 1000)
                }
                
                # Calculate position sizing based on recent data
                if tick % 60 == 0:  # Every minute
                    result = sizer.calculate_position_size(
                        strategy_name=f"Stream_{tick}",
                        expected_return=np.random.uniform(0.05, 0.15),
                        volatility=np.random.uniform(0.10, 0.25)
                    )
                    processed_items += 1
                
                # Sample memory every 300 ticks (5 minutes)
                if tick % 300 == 0:
                    current_memory = self._get_memory_info()
                    memory_samples.append(current_memory['rss_mb'])
                
                # Simulate memory management - cleanup old data
                if tick % 1800 == 0:  # Every 30 minutes
                    gc.collect()
            
            return memory_samples, processed_items
        
        memory_samples, processed_items = streaming_simulation()
        
        # Analyze streaming memory pattern
        if len(memory_samples) > 1:
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
            memory_variance = np.var(memory_samples)
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
        else:
            memory_trend = 0
            memory_variance = 0
            max_memory = memory_samples[0] if memory_samples else 0
            min_memory = memory_samples[0] if memory_samples else 0
        
        results['streaming_analysis'] = {
            'processed_items': processed_items,
            'memory_samples': len(memory_samples),
            'memory_trend_mb_per_sample': memory_trend,
            'memory_variance': memory_variance,
            'max_memory_mb': max_memory,
            'min_memory_mb': min_memory,
            'memory_range_mb': max_memory - min_memory,
            'memory_stability': 1.0 / (1.0 + memory_variance)  # Higher is more stable
        }
        
        # Test bounded memory growth in continuous operation
        def bounded_memory_test():
            initial_memory = self._get_memory_info()
            
            # Run for extended period with periodic cleanup
            for cycle in range(100):
                # Create batch of operations
                batch_data = []
                for i in range(100):
                    result = sizer.calculate_position_size(
                        strategy_name=f"Bounded_{cycle}_{i}",
                        expected_return=np.random.uniform(0.05, 0.15),
                        volatility=np.random.uniform(0.10, 0.25)
                    )
                    batch_data.append(result)
                
                # Cleanup batch (simulate sliding window)
                del batch_data
                
                if cycle % 20 == 0:
                    gc.collect()
            
            final_memory = self._get_memory_info()
            return final_memory['rss_mb'] - initial_memory['rss_mb']
        
        bounded_memory_growth = bounded_memory_test()
        results['bounded_memory_growth_mb'] = bounded_memory_growth
        
        # Validate streaming memory behavior
        max_memory_trend = 0.5  # 0.5MB growth per sample max
        max_bounded_growth = 20.0  # 20MB total growth for bounded test
        
        if abs(memory_trend) > max_memory_trend:
            raise AssertionError(f"Streaming memory trend {memory_trend:.2f}MB/sample exceeds {max_memory_trend}MB/sample")
        
        if bounded_memory_growth > max_bounded_growth:
            raise AssertionError(f"Bounded memory growth {bounded_memory_growth:.1f}MB exceeds {max_bounded_growth}MB")
        
        return results
    
    def _test_allocation_efficiency(self) -> Dict[str, Any]:
        """Test memory allocation and deallocation efficiency."""
        results = {}
        
        # Test allocation patterns
        allocation_tests = {}
        
        # Test frequent small allocations
        def small_allocation_test():
            start_time = time.perf_counter()
            small_objects = []
            
            for i in range(10000):
                obj = {
                    'id': i,
                    'data': np.random.randn(10),  # Small array
                    'metadata': {'created': time.time()}
                }
                small_objects.append(obj)
            
            allocation_time = time.perf_counter() - start_time
            
            # Clean up
            del small_objects
            gc.collect()
            cleanup_time = time.perf_counter() - start_time - allocation_time
            
            return {
                'allocation_time': allocation_time,
                'cleanup_time': cleanup_time,
                'objects_created': 10000,
                'allocation_rate': 10000 / allocation_time
            }
        
        allocation_tests['small_allocations'] = small_allocation_test()
        
        # Test large object allocations
        def large_allocation_test():
            start_time = time.perf_counter()
            large_objects = []
            
            for i in range(100):
                obj = {
                    'id': i,
                    'data': np.random.randn(10000),  # Large array
                    'metadata': {'created': time.time(), 'size': 'large'}
                }
                large_objects.append(obj)
            
            allocation_time = time.perf_counter() - start_time
            
            # Clean up
            del large_objects
            gc.collect()
            cleanup_time = time.perf_counter() - start_time - allocation_time
            
            return {
                'allocation_time': allocation_time,
                'cleanup_time': cleanup_time,
                'objects_created': 100,
                'allocation_rate': 100 / allocation_time
            }
        
        allocation_tests['large_allocations'] = large_allocation_test()
        
        # Test mixed allocation patterns
        def mixed_allocation_test():
            start_time = time.perf_counter()
            mixed_objects = []
            
            for i in range(1000):
                if i % 10 == 0:
                    # Large object
                    obj = np.random.randn(5000)
                else:
                    # Small object
                    obj = np.random.randn(50)
                
                mixed_objects.append(obj)
            
            allocation_time = time.perf_counter() - start_time
            
            # Clean up
            del mixed_objects
            gc.collect()
            cleanup_time = time.perf_counter() - start_time - allocation_time
            
            return {
                'allocation_time': allocation_time,
                'cleanup_time': cleanup_time,
                'objects_created': 1000,
                'allocation_rate': 1000 / allocation_time
            }
        
        allocation_tests['mixed_allocations'] = mixed_allocation_test()
        
        results['allocation_efficiency'] = allocation_tests
        
        # Calculate efficiency metrics
        total_allocation_time = sum(test['allocation_time'] for test in allocation_tests.values())
        total_objects = sum(test['objects_created'] for test in allocation_tests.values())
        
        results['aggregate_efficiency'] = {
            'total_allocation_time': total_allocation_time,
            'total_objects_created': total_objects,
            'overall_allocation_rate': total_objects / total_allocation_time
        }
        
        # Validate allocation efficiency
        min_allocation_rate = 1000.0  # 1000 objects per second minimum
        actual_rate = results['aggregate_efficiency']['overall_allocation_rate']
        
        if actual_rate < min_allocation_rate:
            raise AssertionError(f"Allocation rate {actual_rate:.0f} obj/sec below {min_allocation_rate} obj/sec")
        
        return results
    
    def _test_peak_memory_optimization(self) -> Dict[str, Any]:
        """Test peak memory usage optimization."""
        results = {}
        
        # Test memory optimization strategies
        optimization_tests = {}
        
        # Test without optimization (baseline)
        def unoptimized_processing():
            # Load all data at once
            all_data = []
            for i in range(1000):
                data = {
                    'strategy': f"Unoptimized_{i}",
                    'prices': np.random.randn(1000),
                    'volumes': np.random.randint(100, 1000, 1000),
                    'calculations': []
                }
                
                # Perform calculations
                for j in range(100):
                    config = KellyConfig(total_capital=100000 + j)
                    sizer = KellyCriterion(config)
                    result = sizer.calculate_position_size(f"Calc_{i}_{j}", 0.10, 0.20)
                    data['calculations'].append(result)
                
                all_data.append(data)
            
            return all_data
        
        _, unoptimized_memory = self._monitor_memory_during_operation(unoptimized_processing)
        optimization_tests['unoptimized'] = unoptimized_memory
        
        # Test with batch processing optimization
        def batch_optimized_processing():
            batch_size = 100
            total_processed = 0
            
            for batch in range(10):  # 10 batches of 100
                batch_data = []
                
                for i in range(batch_size):
                    data = {
                        'strategy': f"Optimized_{batch}_{i}",
                        'prices': np.random.randn(1000),
                        'volumes': np.random.randint(100, 1000, 1000),
                        'calculations': []
                    }
                    
                    # Perform calculations
                    for j in range(100):
                        config = KellyConfig(total_capital=100000 + j)
                        sizer = KellyCriterion(config)
                        result = sizer.calculate_position_size(f"Calc_{batch}_{i}_{j}", 0.10, 0.20)
                        data['calculations'].append(result)
                    
                    batch_data.append(data)
                
                total_processed += len(batch_data)
                
                # Process batch and cleanup
                del batch_data
                gc.collect()
            
            return total_processed
        
        _, optimized_memory = self._monitor_memory_during_operation(batch_optimized_processing)
        optimization_tests['batch_optimized'] = optimized_memory
        
        # Calculate optimization effectiveness
        peak_reduction = (unoptimized_memory['peak_rss_mb'] - optimized_memory['peak_rss_mb']) / unoptimized_memory['peak_rss_mb']
        memory_efficiency_improvement = peak_reduction
        
        results['optimization_comparison'] = {
            'unoptimized_peak_mb': unoptimized_memory['peak_rss_mb'],
            'optimized_peak_mb': optimized_memory['peak_rss_mb'],
            'peak_reduction_mb': unoptimized_memory['peak_rss_mb'] - optimized_memory['peak_rss_mb'],
            'peak_reduction_percent': peak_reduction * 100,
            'memory_efficiency_improvement': memory_efficiency_improvement
        }
        
        results['optimization_tests'] = optimization_tests
        
        # Validate optimization effectiveness
        min_optimization_improvement = 0.2  # 20% peak memory reduction minimum
        if memory_efficiency_improvement < min_optimization_improvement:
            raise AssertionError(f"Memory optimization improvement {memory_efficiency_improvement:.1%} below {min_optimization_improvement:.1%}")
        
        return results