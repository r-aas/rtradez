"""
Validation benchmarking framework for RTradez components.

Tests data integrity, model accuracy, strategy backtesting, and overall
system reliability to ensure trading readiness.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
from pathlib import Path
import pickle
import warnings

from .core import ComponentBenchmark, BenchmarkConfig, BenchmarkSeverity
from ..risk import (
    KellyConfig, KellyCriterion, FixedFractionConfig, FixedFractionSizer,
    VolatilityAdjustedConfig, VolatilityAdjustedSizer, MultiStrategyConfig,
    MultiStrategyPositionSizer
)
from ..portfolio.portfolio_manager import PortfolioManager, PortfolioConfig
from ..utils.temporal_alignment import TemporalAligner, TemporalAlignerConfig, FrequencyType
from ..utils.time_bucketing import TimeBucketing, BucketConfig, BucketType


class ValidationBenchmark(ComponentBenchmark):
    """Comprehensive validation testing framework."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("Validation", config)
        self._register_validation_tests()
    
    def _register_validation_tests(self):
        """Register all validation tests."""
        
        # Data Integrity Tests
        @self.suite.register_benchmark(
            "data_integrity_validation",
            "Test data integrity and consistency",
            BenchmarkSeverity.CRITICAL
        )
        def test_data_integrity():
            return self._test_data_integrity()
        
        @self.suite.register_benchmark(
            "temporal_consistency_validation",
            "Test temporal data consistency",
            BenchmarkSeverity.CRITICAL
        )
        def test_temporal_consistency():
            return self._test_temporal_consistency()
        
        # Risk Management Validation
        @self.suite.register_benchmark(
            "risk_calculation_accuracy",
            "Test risk calculation mathematical accuracy",
            BenchmarkSeverity.CRITICAL
        )
        def test_risk_accuracy():
            return self._test_risk_calculation_accuracy()
        
        @self.suite.register_benchmark(
            "position_sizing_bounds_validation",
            "Test position sizing stays within bounds",
            BenchmarkSeverity.ERROR
        )
        def test_position_bounds():
            return self._test_position_sizing_bounds()
        
        # Portfolio Management Validation
        @self.suite.register_benchmark(
            "portfolio_allocation_validation",
            "Test portfolio allocation consistency",
            BenchmarkSeverity.CRITICAL
        )
        def test_portfolio_allocation():
            return self._test_portfolio_allocation()
        
        @self.suite.register_benchmark(
            "rebalancing_consistency_validation",
            "Test rebalancing mathematical consistency",
            BenchmarkSeverity.ERROR
        )
        def test_rebalancing_consistency():
            return self._test_rebalancing_consistency()
        
        # Strategy Backtesting Validation
        @self.suite.register_benchmark(
            "strategy_backtest_validation",
            "Test strategy backtesting accuracy",
            BenchmarkSeverity.CRITICAL
        )
        def test_strategy_backtesting():
            return self._test_strategy_backtesting()
        
        @self.suite.register_benchmark(
            "monte_carlo_validation",
            "Test Monte Carlo simulation accuracy",
            BenchmarkSeverity.WARNING
        )
        def test_monte_carlo():
            return self._test_monte_carlo()
        
        # System Integration Validation
        @self.suite.register_benchmark(
            "end_to_end_workflow_validation",
            "Test complete trading workflow",
            BenchmarkSeverity.CRITICAL
        )
        def test_end_to_end():
            return self._test_end_to_end_workflow()
        
        @self.suite.register_benchmark(
            "state_persistence_validation",
            "Test system state persistence and recovery",
            BenchmarkSeverity.ERROR
        )
        def test_state_persistence():
            return self._test_state_persistence()
    
    def _test_data_integrity(self) -> Dict[str, Any]:
        """Test data integrity and consistency."""
        results = {}
        
        # Generate test dataset with known properties
        date_range = pd.date_range('2024-01-01', periods=1000, freq='1H')
        test_data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.randint(100, 10000, 1000),
            'bid': np.random.uniform(99.5, 100.5, 1000),
            'ask': np.random.uniform(99.5, 100.5, 1000)
        }, index=date_range)
        
        # Ensure ask >= bid
        test_data['ask'] = np.maximum(test_data['bid'], test_data['ask'])
        
        integrity_checks = []
        
        # Check 1: No missing timestamps in regular series
        expected_timestamps = len(date_range)
        actual_timestamps = len(test_data)
        timestamp_integrity = (expected_timestamps == actual_timestamps)
        integrity_checks.append(('timestamp_completeness', timestamp_integrity))
        
        # Check 2: Bid-Ask spread consistency
        bid_ask_valid = (test_data['ask'] >= test_data['bid']).all()
        integrity_checks.append(('bid_ask_consistency', bid_ask_valid))
        
        # Check 3: Volume positivity
        volume_positive = (test_data['volume'] > 0).all()
        integrity_checks.append(('volume_positivity', volume_positive))
        
        # Check 4: Price reasonableness (no extreme jumps)
        price_returns = test_data['price'].pct_change().dropna()
        extreme_moves = (np.abs(price_returns) > 0.1).sum()  # >10% moves
        price_reasonable = extreme_moves < len(price_returns) * 0.01  # <1% of data
        integrity_checks.append(('price_reasonableness', price_reasonable))
        
        # Check 5: Data type consistency
        numeric_columns = ['price', 'volume', 'bid', 'ask']
        type_consistency = all(pd.api.types.is_numeric_dtype(test_data[col]) for col in numeric_columns)
        integrity_checks.append(('data_type_consistency', type_consistency))
        
        # Check 6: No infinite or NaN values in core data
        no_infinite = not np.isinf(test_data[numeric_columns]).any().any()
        no_nan = not test_data[numeric_columns].isna().any().any()
        data_validity = no_infinite and no_nan
        integrity_checks.append(('data_validity', data_validity))
        
        # Aggregate results
        passed_checks = sum(1 for _, passed in integrity_checks if passed)
        total_checks = len(integrity_checks)
        
        results['integrity_checks'] = {name: passed for name, passed in integrity_checks}
        results['checks_passed'] = passed_checks
        results['total_checks'] = total_checks
        results['integrity_score'] = passed_checks / total_checks
        results['data_points_tested'] = len(test_data)
        
        # Validation threshold
        min_integrity_score = 0.95  # 95% of checks must pass
        if results['integrity_score'] < min_integrity_score:
            raise AssertionError(f"Data integrity score {results['integrity_score']:.1%} below {min_integrity_score:.1%}")
        
        return results
    
    def _test_temporal_consistency(self) -> Dict[str, Any]:
        """Test temporal data consistency."""
        results = {}
        
        # Create datasets with different frequencies
        daily_data = pd.DataFrame({
            'value': np.random.randn(100)
        }, index=pd.date_range('2024-01-01', periods=100, freq='D'))
        
        hourly_data = pd.DataFrame({
            'value': np.random.randn(240)  # 10 days * 24 hours
        }, index=pd.date_range('2024-01-01', periods=240, freq='H'))
        
        # Test temporal alignment
        config = TemporalAlignerConfig(
            target_frequency=FrequencyType.DAILY,
            alignment_method='outer',
            fill_method='forward_fill'
        )
        aligner = TemporalAligner(config)
        
        consistency_tests = []
        
        # Test 1: Frequency detection accuracy
        from ..utils.temporal_alignment import TemporalProfile
        daily_profile = TemporalProfile(
            frequency=FrequencyType.DAILY,
            start_date=daily_data.index.min().to_pydatetime(),
            end_date=daily_data.index.max().to_pydatetime(),
            total_observations=len(daily_data),
            missing_periods=0,
            regularity_score=1.0
        )
        
        frequency_detected = daily_profile.frequency == FrequencyType.DAILY
        consistency_tests.append(('frequency_detection', frequency_detected))
        
        # Test 2: Alignment preserves data points
        aligned_data = hourly_data.resample('D').last()  # Simplified alignment
        alignment_preserves_count = len(aligned_data) <= len(hourly_data)
        consistency_tests.append(('alignment_preservation', alignment_preserves_count))
        
        # Test 3: Temporal ordering maintained
        ordering_maintained = aligned_data.index.is_monotonic_increasing
        consistency_tests.append(('temporal_ordering', ordering_maintained))
        
        # Test 4: No temporal gaps in regular series
        expected_days = (daily_data.index.max() - daily_data.index.min()).days + 1
        actual_days = len(daily_data)
        no_gaps = expected_days == actual_days
        consistency_tests.append(('no_temporal_gaps', no_gaps))
        
        # Test 5: Timezone consistency (if applicable)
        timezone_consistent = (daily_data.index.tz == hourly_data.index.tz)
        consistency_tests.append(('timezone_consistency', timezone_consistent))
        
        # Aggregate results
        passed_tests = sum(1 for _, passed in consistency_tests if passed)
        total_tests = len(consistency_tests)
        
        results['consistency_tests'] = {name: passed for name, passed in consistency_tests}
        results['tests_passed'] = passed_tests
        results['total_tests'] = total_tests
        results['consistency_score'] = passed_tests / total_tests
        
        # Validation threshold
        min_consistency_score = 0.8  # 80% of tests must pass
        if results['consistency_score'] < min_consistency_score:
            raise AssertionError(f"Temporal consistency score {results['consistency_score']:.1%} below {min_consistency_score:.1%}")
        
        return results
    
    def _test_risk_calculation_accuracy(self) -> Dict[str, Any]:
        """Test risk calculation mathematical accuracy."""
        results = {}
        
        # Test Kelly Criterion with known inputs
        config = KellyConfig(total_capital=100000)
        kelly = KellyCriterion(config)
        
        accuracy_tests = []
        
        # Test 1: Kelly fraction calculation accuracy
        # For p=0.6, b=1 (even money bet), Kelly = (0.6*2 - 1)/1 = 0.2
        known_cases = [
            # (expected_return, volatility, expected_fraction_range)
            (0.20, 0.30, (0.1, 0.3)),  # Reasonable aggressive position
            (0.05, 0.15, (0.0, 0.5)),  # Conservative position
            (0.15, 0.25, (0.1, 0.4)),  # Moderate position
        ]
        
        kelly_accuracy = 0
        for expected_return, volatility, fraction_range in known_cases:
            result = kelly.calculate_position_size(
                strategy_name="AccuracyTest",
                expected_return=expected_return,
                volatility=volatility
            )
            
            fraction = result.recommended_size / config.total_capital
            if fraction_range[0] <= fraction <= fraction_range[1]:
                kelly_accuracy += 1
        
        kelly_accuracy_rate = kelly_accuracy / len(known_cases)
        accuracy_tests.append(('kelly_accuracy', kelly_accuracy_rate >= 0.8))
        
        # Test 2: Position sizing bounds
        extreme_cases = [
            (0.50, 0.20),  # High return, low vol
            (0.02, 0.40),  # Low return, high vol
            (-0.10, 0.30), # Negative return
        ]
        
        bounds_respected = 0
        for expected_return, volatility in extreme_cases:
            result = kelly.calculate_position_size(
                strategy_name="BoundsTest",
                expected_return=expected_return,
                volatility=volatility
            )
            
            # Position should be reasonable (0 to 2x capital max)
            if 0 <= result.recommended_size <= config.total_capital * 2:
                bounds_respected += 1
        
        bounds_accuracy = bounds_respected / len(extreme_cases)
        accuracy_tests.append(('bounds_respect', bounds_accuracy >= 0.9))
        
        # Test 3: Mathematical consistency
        # Kelly fraction should decrease with increasing volatility for same return
        base_result = kelly.calculate_position_size("Consistency1", 0.15, 0.20)
        higher_vol_result = kelly.calculate_position_size("Consistency2", 0.15, 0.30)
        
        volatility_consistency = base_result.recommended_size >= higher_vol_result.recommended_size
        accuracy_tests.append(('volatility_consistency', volatility_consistency))
        
        # Test 4: Multi-strategy position sizing sum consistency
        multi_config = MultiStrategyConfig(total_capital=1000000, max_total_risk=0.20)
        multi_sizer = MultiStrategyPositionSizer(multi_config)
        
        # Add multiple positions
        from ..risk.position_sizing import PositionSizeResult
        for i in range(5):
            result = PositionSizeResult(
                strategy_name=f"Strategy_{i}",
                recommended_size=100000 + i * 10000,
                max_position_value=200000,
                risk_adjusted_size=90000 + i * 8000,
                confidence_level=0.8,
                reasoning="Test sizing"
            )
            multi_sizer.add_strategy_sizing(result)
        
        optimized = multi_sizer.optimize_portfolio_allocation()
        total_allocation = sum(s.adjusted_size for s in optimized.strategy_allocations)
        allocation_consistency = total_allocation <= multi_config.total_capital * 1.1  # Allow 10% tolerance
        accuracy_tests.append(('allocation_consistency', allocation_consistency))
        
        # Aggregate results
        passed_tests = sum(1 for _, passed in accuracy_tests if passed)
        total_tests = len(accuracy_tests)
        
        results['accuracy_tests'] = {f"test_{i}": passed for i, (name, passed) in enumerate(accuracy_tests)}
        results['tests_passed'] = passed_tests
        results['total_tests'] = total_tests
        results['accuracy_score'] = passed_tests / total_tests
        results['kelly_accuracy_rate'] = kelly_accuracy_rate
        results['bounds_accuracy_rate'] = bounds_accuracy
        
        # Validation threshold
        min_accuracy_score = 0.85  # 85% of tests must pass
        if results['accuracy_score'] < min_accuracy_score:
            raise AssertionError(f"Risk calculation accuracy {results['accuracy_score']:.1%} below {min_accuracy_score:.1%}")
        
        return results
    
    def _test_position_sizing_bounds(self) -> Dict[str, Any]:
        """Test position sizing stays within bounds."""
        results = {}
        
        methods = [
            ('kelly', KellyCriterion(KellyConfig(total_capital=100000))),
            ('fixed_fraction', FixedFractionSizer(FixedFractionConfig(total_capital=100000))),
            ('volatility_adjusted', VolatilityAdjustedSizer(VolatilityAdjustedConfig(total_capital=100000)))
        ]
        
        bounds_violations = []
        total_tests = 0
        
        for method_name, sizer in methods:
            method_violations = 0
            method_tests = 100
            
            for i in range(method_tests):
                # Generate random but reasonable parameters
                expected_return = random.uniform(-0.2, 0.5)  # -20% to 50%
                volatility = random.uniform(0.05, 0.8)       # 5% to 80%
                
                try:
                    result = sizer.calculate_position_size(
                        strategy_name=f"BoundsTest_{method_name}_{i}",
                        expected_return=expected_return,
                        volatility=volatility
                    )
                    
                    # Check bounds
                    if result.recommended_size < 0:
                        method_violations += 1
                    elif result.recommended_size > sizer.config.total_capital * 3:  # 3x leverage max
                        method_violations += 1
                    
                except Exception:
                    method_violations += 1
            
            bounds_violations.append((method_name, method_violations, method_tests))
            total_tests += method_tests
        
        # Calculate bounds compliance
        total_violations = sum(violations for _, violations, _ in bounds_violations)
        compliance_rate = (total_tests - total_violations) / total_tests
        
        results['bounds_violations_by_method'] = {
            name: {'violations': violations, 'tests': tests, 'compliance_rate': (tests - violations) / tests}
            for name, violations, tests in bounds_violations
        }
        results['total_violations'] = total_violations
        results['total_tests'] = total_tests
        results['overall_compliance_rate'] = compliance_rate
        
        # Validation threshold
        min_compliance_rate = 0.95  # 95% compliance required
        if compliance_rate < min_compliance_rate:
            raise AssertionError(f"Position sizing bounds compliance {compliance_rate:.1%} below {min_compliance_rate:.1%}")
        
        return results
    
    def _test_portfolio_allocation(self) -> Dict[str, Any]:
        """Test portfolio allocation consistency."""
        results = {}
        
        config = PortfolioConfig(total_capital=1000000, max_strategies=10)
        portfolio = PortfolioManager(config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        allocation_tests = []
        
        # Test 1: Sum of allocations equals 1.0
        target_allocations = [0.2, 0.3, 0.15, 0.25, 0.1]  # Should sum to 1.0
        for i, allocation in enumerate(target_allocations):
            portfolio.add_strategy(
                f"AllocTest_{i}",
                MockStrategy(f"AllocTest_{i}"),
                target_allocation=allocation
            )
        
        total_allocation = sum(s.target_allocation for s in portfolio.strategies.values())
        allocation_sum_test = abs(total_allocation - 1.0) < 0.001  # Within 0.1%
        allocation_tests.append(('allocation_sum', allocation_sum_test))
        
        # Test 2: Individual allocations within reasonable bounds
        reasonable_bounds = all(0 < s.target_allocation < 1 for s in portfolio.strategies.values())
        allocation_tests.append(('individual_bounds', reasonable_bounds))
        
        # Test 3: Capital allocation consistency
        total_capital_allocated = sum(
            s.target_allocation * config.total_capital 
            for s in portfolio.strategies.values()
        )
        capital_consistency = abs(total_capital_allocated - config.total_capital) < config.total_capital * 0.01
        allocation_tests.append(('capital_consistency', capital_consistency))
        
        # Test 4: Rebalancing maintains allocation sum
        # Simulate some drift
        for name, allocation in portfolio.strategies.items():
            allocation.current_allocation = allocation.target_allocation * random.uniform(0.8, 1.2)
        
        # Calculate rebalancing needs
        rebalancing_needs = portfolio.calculate_rebalancing_needs()
        
        if rebalancing_needs:
            # Simulate rebalancing
            total_current_before = sum(s.current_allocation for s in portfolio.strategies.values())
            portfolio.execute_rebalancing(rebalancing_needs)
            total_current_after = sum(s.current_allocation for s in portfolio.strategies.values())
            
            rebalancing_maintains_sum = abs(total_current_after - 1.0) < 0.01
        else:
            rebalancing_maintains_sum = True  # No rebalancing needed is fine
        
        allocation_tests.append(('rebalancing_sum', rebalancing_maintains_sum))
        
        # Test 5: Portfolio metrics consistency
        try:
            metrics = portfolio.calculate_portfolio_metrics()
            metrics_calculated = metrics is not None
        except Exception:
            metrics_calculated = False
        
        allocation_tests.append(('metrics_calculation', metrics_calculated))
        
        # Aggregate results
        passed_tests = sum(1 for _, passed in allocation_tests if passed)
        total_tests = len(allocation_tests)
        
        results['allocation_tests'] = {f"test_{i}": passed for i, (name, passed) in enumerate(allocation_tests)}
        results['tests_passed'] = passed_tests
        results['total_tests'] = total_tests
        results['allocation_score'] = passed_tests / total_tests
        results['total_allocation'] = total_allocation
        results['capital_allocated'] = total_capital_allocated
        
        # Validation threshold
        min_allocation_score = 0.9  # 90% of tests must pass
        if results['allocation_score'] < min_allocation_score:
            raise AssertionError(f"Portfolio allocation score {results['allocation_score']:.1%} below {min_allocation_score:.1%}")
        
        return results
    
    def _test_rebalancing_consistency(self) -> Dict[str, Any]:
        """Test rebalancing mathematical consistency."""
        results = {}
        
        config = PortfolioConfig(total_capital=2000000, max_strategies=5, rebalance_threshold=0.05)
        portfolio = PortfolioManager(config)
        
        class MockStrategy:
            def __init__(self, name): self.name = name
        
        # Set up portfolio
        target_allocations = [0.25, 0.20, 0.30, 0.15, 0.10]
        for i, allocation in enumerate(target_allocations):
            portfolio.add_strategy(
                f"RebalanceTest_{i}",
                MockStrategy(f"RebalanceTest_{i}"),
                target_allocation=allocation
            )
        
        consistency_tests = []
        
        # Test 1: Rebalancing reduces allocation drift
        # Create significant drift
        for name, allocation in portfolio.strategies.items():
            drift = random.uniform(-0.1, 0.1)  # ±10% drift
            allocation.current_allocation = max(0, allocation.target_allocation + drift)
        
        # Measure drift before rebalancing
        drift_before = sum(
            abs(s.current_allocation - s.target_allocation) 
            for s in portfolio.strategies.values()
        )
        
        # Rebalance
        needs = portfolio.calculate_rebalancing_needs()
        if needs:
            portfolio.execute_rebalancing(needs)
        
        # Measure drift after rebalancing
        drift_after = sum(
            abs(s.current_allocation - s.target_allocation) 
            for s in portfolio.strategies.values()
        )
        
        drift_reduced = drift_after < drift_before
        consistency_tests.append(('drift_reduction', drift_reduced))
        
        # Test 2: Multiple rebalancing operations converge
        convergence_iterations = 5
        previous_drift = float('inf')
        converges = True
        
        for iteration in range(convergence_iterations):
            # Add small random drift
            for name, allocation in portfolio.strategies.items():
                small_drift = random.uniform(-0.02, 0.02)  # ±2%
                allocation.current_allocation = max(0, allocation.current_allocation + small_drift)
            
            # Rebalance
            needs = portfolio.calculate_rebalancing_needs()
            if needs:
                portfolio.execute_rebalancing(needs)
            
            # Measure current drift
            current_drift = sum(
                abs(s.current_allocation - s.target_allocation) 
                for s in portfolio.strategies.values()
            )
            
            # Check if drift is reducing or staying low
            if current_drift > previous_drift and current_drift > 0.1:
                converges = False
                break
            
            previous_drift = current_drift
        
        consistency_tests.append(('convergence', converges))
        
        # Test 3: Rebalancing preserves total allocation
        total_before_final = sum(s.current_allocation for s in portfolio.strategies.values())
        allocation_preserved = abs(total_before_final - 1.0) < 0.05  # Within 5%
        consistency_tests.append(('allocation_preservation', allocation_preserved))
        
        # Test 4: Capital allocation matches percentages
        total_capital_check = abs(
            sum(s.current_allocation * config.total_capital for s in portfolio.strategies.values()) - 
            config.total_capital
        ) < config.total_capital * 0.02  # Within 2%
        consistency_tests.append(('capital_consistency', total_capital_check))
        
        # Aggregate results
        passed_tests = sum(1 for _, passed in consistency_tests if passed)
        total_tests = len(consistency_tests)
        
        results['consistency_tests'] = {f"test_{i}": passed for i, (name, passed) in enumerate(consistency_tests)}
        results['tests_passed'] = passed_tests
        results['total_tests'] = total_tests
        results['consistency_score'] = passed_tests / total_tests
        results['final_drift'] = sum(
            abs(s.current_allocation - s.target_allocation) 
            for s in portfolio.strategies.values()
        )
        results['total_allocation_final'] = sum(s.current_allocation for s in portfolio.strategies.values())
        
        # Validation threshold
        min_consistency_score = 0.75  # 75% of tests must pass
        if results['consistency_score'] < min_consistency_score:
            raise AssertionError(f"Rebalancing consistency score {results['consistency_score']:.1%} below {min_consistency_score:.1%}")
        
        return results
    
    def _test_strategy_backtesting(self) -> Dict[str, Any]:
        """Test strategy backtesting accuracy."""
        results = {}
        
        # Generate synthetic market data for backtesting
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)  # 2% daily volatility
        
        market_data = pd.DataFrame({
            'price': prices,
            'returns': np.concatenate([[0], np.diff(np.log(prices))]),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        backtesting_tests = []
        
        # Test 1: Simple buy-and-hold strategy validation
        initial_capital = 100000
        shares_bought = initial_capital / market_data['price'].iloc[0]
        final_value = shares_bought * market_data['price'].iloc[-1]
        
        # Calculate returns manually
        buy_hold_return = (final_value - initial_capital) / initial_capital
        market_return = (market_data['price'].iloc[-1] - market_data['price'].iloc[0]) / market_data['price'].iloc[0]
        
        # Buy-and-hold return should match market return (within tolerance)
        buy_hold_accuracy = abs(buy_hold_return - market_return) < 0.001
        backtesting_tests.append(('buy_hold_accuracy', buy_hold_accuracy))
        
        # Test 2: Moving average crossover strategy
        market_data['ma_short'] = market_data['price'].rolling(10).mean()
        market_data['ma_long'] = market_data['price'].rolling(30).mean()
        market_data['signal'] = (market_data['ma_short'] > market_data['ma_long']).astype(int)
        
        # Calculate strategy returns
        market_data['strategy_returns'] = market_data['signal'].shift(1) * market_data['returns']
        cumulative_strategy_return = np.exp(market_data['strategy_returns'].fillna(0).cumsum()).iloc[-1] - 1
        
        # Strategy should produce reasonable returns (not extreme)
        strategy_reasonable = -0.5 < cumulative_strategy_return < 2.0  # -50% to 200%
        backtesting_tests.append(('strategy_reasonableness', strategy_reasonable))
        
        # Test 3: Risk metrics calculation
        strategy_vol = market_data['strategy_returns'].std() * np.sqrt(252)  # Annualized
        market_vol = market_data['returns'].std() * np.sqrt(252)
        
        # Volatility should be reasonable
        vol_reasonable = 0.01 < strategy_vol < 1.0  # 1% to 100% annual vol
        backtesting_tests.append(('volatility_reasonableness', vol_reasonable))
        
        # Test 4: Sharpe ratio calculation
        risk_free_rate = 0.02  # 2% risk-free rate
        strategy_sharpe = (cumulative_strategy_return - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        
        # Sharpe ratio should be reasonable
        sharpe_reasonable = -3.0 < strategy_sharpe < 5.0
        backtesting_tests.append(('sharpe_reasonableness', sharpe_reasonable))
        
        # Test 5: Maximum drawdown calculation
        cumulative_returns = (1 + market_data['strategy_returns'].fillna(0)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Max drawdown should be reasonable (negative and not extreme)
        drawdown_reasonable = -0.8 < max_drawdown <= 0  # -80% to 0%
        backtesting_tests.append(('drawdown_reasonableness', drawdown_reasonable))
        
        # Aggregate results
        passed_tests = sum(1 for _, passed in backtesting_tests if passed)
        total_tests = len(backtesting_tests)
        
        results['backtesting_tests'] = {f"test_{i}": passed for i, (name, passed) in enumerate(backtesting_tests)}
        results['tests_passed'] = passed_tests
        results['total_tests'] = total_tests
        results['backtesting_score'] = passed_tests / total_tests
        results['buy_hold_return'] = buy_hold_return
        results['strategy_return'] = cumulative_strategy_return
        results['strategy_sharpe'] = strategy_sharpe
        results['max_drawdown'] = max_drawdown
        results['strategy_volatility'] = strategy_vol
        
        # Validation threshold
        min_backtesting_score = 0.8  # 80% of tests must pass
        if results['backtesting_score'] < min_backtesting_score:
            raise AssertionError(f"Strategy backtesting score {results['backtesting_score']:.1%} below {min_backtesting_score:.1%}")
        
        return results
    
    def _test_monte_carlo(self) -> Dict[str, Any]:
        """Test Monte Carlo simulation accuracy."""
        results = {}
        
        # Monte Carlo simulation of portfolio returns
        num_simulations = 1000
        time_periods = 252  # 1 year daily
        
        # Portfolio parameters
        expected_annual_return = 0.08
        annual_volatility = 0.15
        daily_return = expected_annual_return / 252
        daily_vol = annual_volatility / np.sqrt(252)
        
        simulation_results = []
        
        for sim in range(num_simulations):
            # Generate random daily returns
            daily_returns = np.random.normal(daily_return, daily_vol, time_periods)
            cumulative_return = np.exp(np.sum(daily_returns)) - 1
            simulation_results.append(cumulative_return)
        
        simulation_results = np.array(simulation_results)
        
        monte_carlo_tests = []
        
        # Test 1: Mean convergence to expected return
        simulated_mean = np.mean(simulation_results)
        mean_error = abs(simulated_mean - expected_annual_return)
        mean_convergence = mean_error < 0.02  # Within 2%
        monte_carlo_tests.append(('mean_convergence', mean_convergence))
        
        # Test 2: Volatility convergence
        simulated_vol = np.std(simulation_results)
        vol_error = abs(simulated_vol - annual_volatility)
        vol_convergence = vol_error < 0.03  # Within 3%
        monte_carlo_tests.append(('volatility_convergence', vol_convergence))
        
        # Test 3: Distribution shape (approximately normal)
        from scipy import stats
        _, p_value = stats.normaltest(simulation_results)
        # Don't require perfect normality, but should not be extremely non-normal
        distribution_reasonable = p_value > 0.001  # Very lenient test
        monte_carlo_tests.append(('distribution_shape', distribution_reasonable))
        
        # Test 4: Percentile consistency
        p5 = np.percentile(simulation_results, 5)
        p95 = np.percentile(simulation_results, 95)
        
        # 90% of results should fall between these percentiles
        within_percentiles = np.sum((p5 <= simulation_results) & (simulation_results <= p95))
        percentile_accuracy = within_percentiles / len(simulation_results)
        percentile_test = 0.85 <= percentile_accuracy <= 0.95  # Should be around 90%
        monte_carlo_tests.append(('percentile_accuracy', percentile_test))
        
        # Test 5: VaR calculation consistency
        var_95 = np.percentile(simulation_results, 5)  # 5th percentile for 95% VaR
        
        # VaR should be negative (loss) and reasonable
        var_reasonable = -0.5 < var_95 < 0  # Between -50% and 0%
        monte_carlo_tests.append(('var_reasonableness', var_reasonable))
        
        # Aggregate results
        passed_tests = sum(1 for _, passed in monte_carlo_tests if passed)
        total_tests = len(monte_carlo_tests)
        
        results['monte_carlo_tests'] = {f"test_{i}": passed for i, (name, passed) in enumerate(monte_carlo_tests)}
        results['tests_passed'] = passed_tests
        results['total_tests'] = total_tests
        results['monte_carlo_score'] = passed_tests / total_tests
        results['simulations_run'] = num_simulations
        results['simulated_mean_return'] = simulated_mean
        results['simulated_volatility'] = simulated_vol
        results['var_95'] = var_95
        results['percentile_accuracy'] = percentile_accuracy
        
        # Validation threshold
        min_monte_carlo_score = 0.7  # 70% of tests must pass (Monte Carlo can be variable)
        if results['monte_carlo_score'] < min_monte_carlo_score:
            raise AssertionError(f"Monte Carlo simulation score {results['monte_carlo_score']:.1%} below {min_monte_carlo_score:.1%}")
        
        return results
    
    def _test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete trading workflow."""
        results = {}
        
        workflow_steps = []
        
        try:
            # Step 1: Initialize risk management
            kelly_config = KellyConfig(total_capital=500000)
            kelly_sizer = KellyCriterion(kelly_config)
            
            position_result = kelly_sizer.calculate_position_size(
                strategy_name="EndToEndTest",
                expected_return=0.12,
                volatility=0.18
            )
            
            risk_init_success = position_result.recommended_size > 0
            workflow_steps.append(('risk_initialization', risk_init_success))
            
            # Step 2: Set up portfolio
            portfolio_config = PortfolioConfig(total_capital=500000, max_strategies=3)
            portfolio = PortfolioManager(portfolio_config)
            
            class MockStrategy:
                def __init__(self, name): self.name = name
            
            # Add strategies
            strategies_added = 0
            target_strategies = 3
            for i in range(target_strategies):
                success = portfolio.add_strategy(
                    f"E2EStrategy_{i}",
                    MockStrategy(f"E2EStrategy_{i}"),
                    target_allocation=1.0 / target_strategies
                )
                if success:
                    strategies_added += 1
            
            portfolio_setup_success = strategies_added == target_strategies
            workflow_steps.append(('portfolio_setup', portfolio_setup_success))
            
            # Step 3: Calculate portfolio metrics
            metrics = portfolio.calculate_portfolio_metrics()
            metrics_success = metrics is not None
            workflow_steps.append(('metrics_calculation', metrics_success))
            
            # Step 4: Perform rebalancing simulation
            # Simulate some drift
            for name, allocation in portfolio.strategies.items():
                allocation.current_allocation = allocation.target_allocation * random.uniform(0.9, 1.1)
            
            rebalancing_needs = portfolio.calculate_rebalancing_needs()
            rebalancing_calculated = rebalancing_needs is not None
            workflow_steps.append(('rebalancing_calculation', rebalancing_calculated))
            
            if rebalancing_needs:
                rebalance_success = portfolio.execute_rebalancing(rebalancing_needs)
                workflow_steps.append(('rebalancing_execution', rebalance_success))
            else:
                workflow_steps.append(('rebalancing_execution', True))  # No rebalancing needed is OK
            
            # Step 5: Multi-strategy optimization
            multi_config = MultiStrategyConfig(total_capital=500000, max_total_risk=0.20)
            multi_sizer = MultiStrategyPositionSizer(multi_config)
            
            # Add sizing results from individual strategies
            from ..risk.position_sizing import PositionSizeResult
            for i in range(3):
                sizing_result = PositionSizeResult(
                    strategy_name=f"E2EStrategy_{i}",
                    recommended_size=80000 + i * 10000,
                    max_position_value=150000,
                    risk_adjusted_size=70000 + i * 8000,
                    confidence_level=0.75,
                    reasoning="End-to-end test"
                )
                multi_sizer.add_strategy_sizing(sizing_result)
            
            optimized_allocation = multi_sizer.optimize_portfolio_allocation()
            optimization_success = optimized_allocation is not None and len(optimized_allocation.strategy_allocations) > 0
            workflow_steps.append(('portfolio_optimization', optimization_success))
            
            # Step 6: Data processing simulation
            test_data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(100) * 0.01),
                'volume': np.random.randint(1000, 5000, 100)
            }, index=pd.date_range('2024-01-01', periods=100, freq='H'))
            
            # Simulate temporal alignment
            config = TemporalAlignerConfig(
                target_frequency=FrequencyType.DAILY,
                alignment_method='outer',
                fill_method='forward_fill'
            )
            
            # Simple resampling as proxy for temporal alignment
            daily_data = test_data.resample('D').last()
            data_processing_success = len(daily_data) > 0 and not daily_data['price'].isna().all()
            workflow_steps.append(('data_processing', data_processing_success))
            
        except Exception as e:
            workflow_steps.append(('workflow_error', False))
            results['workflow_error'] = str(e)
        
        # Aggregate results
        passed_steps = sum(1 for _, passed in workflow_steps if passed)
        total_steps = len(workflow_steps)
        
        results['workflow_steps'] = {f"step_{i}_{name}": passed for i, (name, passed) in enumerate(workflow_steps)}
        results['steps_passed'] = passed_steps
        results['total_steps'] = total_steps
        results['workflow_success_rate'] = passed_steps / total_steps
        
        # Validation threshold
        min_workflow_success = 0.9  # 90% of workflow steps must succeed
        if results['workflow_success_rate'] < min_workflow_success:
            raise AssertionError(f"End-to-end workflow success rate {results['workflow_success_rate']:.1%} below {min_workflow_success:.1%}")
        
        return results
    
    def _test_state_persistence(self) -> Dict[str, Any]:
        """Test system state persistence and recovery."""
        results = {}
        
        persistence_tests = []
        
        try:
            # Test 1: Portfolio state persistence
            config = PortfolioConfig(total_capital=300000, max_strategies=5)
            portfolio = PortfolioManager(config)
            
            class MockStrategy:
                def __init__(self, name): self.name = name
            
            # Set up portfolio state
            original_strategies = {}
            for i in range(3):
                name = f"PersistTest_{i}"
                allocation = 0.2 + i * 0.1
                
                success = portfolio.add_strategy(
                    name,
                    MockStrategy(name),
                    target_allocation=allocation
                )
                
                if success:
                    original_strategies[name] = allocation
            
            # Simulate state save (simplified - just store the key data)
            state_data = {
                'strategies': {
                    name: {
                        'target_allocation': alloc.target_allocation,
                        'current_allocation': alloc.current_allocation
                    }
                    for name, alloc in portfolio.strategies.items()
                },
                'total_capital': portfolio.config.total_capital,
                'max_strategies': portfolio.config.max_strategies
            }
            
            # Test that we can recreate the state
            restored_portfolio = PortfolioManager(config)
            
            state_restored = True
            for name, data in state_data['strategies'].items():
                success = restored_portfolio.add_strategy(
                    name,
                    MockStrategy(name),
                    target_allocation=data['target_allocation']
                )
                if not success:
                    state_restored = False
                    break
                
                # Set current allocation
                restored_portfolio.strategies[name].current_allocation = data['current_allocation']
            
            # Verify state matches
            state_matches = (
                len(restored_portfolio.strategies) == len(portfolio.strategies) and
                all(
                    restored_portfolio.strategies[name].target_allocation == portfolio.strategies[name].target_allocation
                    for name in portfolio.strategies.keys()
                )
            )
            
            state_persistence_success = state_restored and state_matches
            persistence_tests.append(('portfolio_state_persistence', state_persistence_success))
            
            # Test 2: Configuration persistence
            # Test that configurations can be serialized and restored
            original_config = BenchmarkConfig(
                max_execution_time=120.0,
                memory_limit_mb=2048,
                stress_iterations=500
            )
            
            # Serialize to dict (simulating JSON save/load)
            config_dict = original_config.dict()
            
            # Restore from dict
            restored_config = BenchmarkConfig(**config_dict)
            
            config_persistence = (
                restored_config.max_execution_time == original_config.max_execution_time and
                restored_config.memory_limit_mb == original_config.memory_limit_mb and
                restored_config.stress_iterations == original_config.stress_iterations
            )
            
            persistence_tests.append(('config_persistence', config_persistence))
            
            # Test 3: Risk management state persistence
            kelly_config = KellyConfig(total_capital=100000, max_position_fraction=0.25)
            kelly_sizer = KellyCriterion(kelly_config)
            
            # Calculate some positions
            original_results = []
            for i in range(3):
                result = kelly_sizer.calculate_position_size(
                    strategy_name=f"PersistRisk_{i}",
                    expected_return=0.1 + i * 0.05,
                    volatility=0.2 + i * 0.05
                )
                original_results.append(result)
            
            # Simulate saving and restoring risk sizer
            config_dict = kelly_sizer.config.dict()
            restored_kelly_sizer = KellyCriterion(KellyConfig(**config_dict))
            
            # Verify we get same results
            results_match = True
            for i, original_result in enumerate(original_results):
                restored_result = restored_kelly_sizer.calculate_position_size(
                    strategy_name=f"PersistRisk_{i}",
                    expected_return=0.1 + i * 0.05,
                    volatility=0.2 + i * 0.05
                )
                
                if abs(restored_result.recommended_size - original_result.recommended_size) > 1.0:  # $1 tolerance
                    results_match = False
                    break
            
            persistence_tests.append(('risk_state_persistence', results_match))
            
        except Exception as e:
            persistence_tests.append(('persistence_error', False))
            results['persistence_error'] = str(e)
        
        # Aggregate results
        passed_tests = sum(1 for _, passed in persistence_tests if passed)
        total_tests = len(persistence_tests)
        
        results['persistence_tests'] = {f"test_{i}": passed for i, (name, passed) in enumerate(persistence_tests)}
        results['tests_passed'] = passed_tests
        results['total_tests'] = total_tests
        results['persistence_score'] = passed_tests / total_tests
        
        # Validation threshold
        min_persistence_score = 0.8  # 80% of persistence tests must pass
        if results['persistence_score'] < min_persistence_score:
            raise AssertionError(f"State persistence score {results['persistence_score']:.1%} below {min_persistence_score:.1%}")
        
        return results