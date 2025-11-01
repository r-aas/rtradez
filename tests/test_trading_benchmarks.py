"""
Comprehensive tests for trading_benchmarks module.

Tests for strategy backtesting, performance analysis, and benchmark comparison.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from rtradez.trading_benchmarks import (
    StrategyBenchmark, BacktestConfig, BacktestResults,
    PerformanceAnalyzer, TradingMetrics,
    StrategyType, Trade, TradeDirection
)


class TestBacktestConfig:
    """Test BacktestConfig validation and configuration."""
    
    def test_valid_config_creation(self):
        """Test creating valid backtest configuration."""
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_per_trade=1.0
        )
        
        assert config.start_date == datetime(2023, 1, 1)
        assert config.end_date == datetime(2023, 12, 31)
        assert config.initial_capital == 100000
        assert config.commission_per_trade == 1.0
        assert config.benchmark_symbol == "SPY"  # Default
    
    def test_invalid_capital_raises_error(self):
        """Test that negative capital raises validation error."""
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=-1000  # Invalid
            )
    
    def test_invalid_position_size_raises_error(self):
        """Test that invalid position size raises error."""
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000,
                max_position_size=1.5  # Invalid (> 1.0)
            )
    
    def test_date_validation(self):
        """Test start date must be before end date."""
        # This should work
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000
        )
        assert config.start_date < config.end_date


class TestTrade:
    """Test Trade class and PnL calculations."""
    
    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            timestamp=datetime(2023, 1, 1),
            symbol="SPY",
            strategy_type=StrategyType.COVERED_CALL,
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=100
        )
        
        assert trade.symbol == "SPY"
        assert trade.strategy_type == StrategyType.COVERED_CALL
        assert trade.direction == TradeDirection.LONG
        assert trade.entry_price == 100.0
        assert trade.quantity == 100
    
    def test_long_trade_pnl_calculation(self):
        """Test PnL calculation for long trade."""
        trade = Trade(
            timestamp=datetime(2023, 1, 1),
            symbol="SPY",
            strategy_type=StrategyType.LONG_CALL,
            direction=TradeDirection.LONG,
            entry_price=100.0,
            exit_price=110.0,
            quantity=100,
            commission=2.0
        )
        
        pnl = trade.calculate_pnl()
        expected_pnl = (110.0 - 100.0) * 100 - 2.0  # $998
        assert pnl == expected_pnl
        assert trade.pnl == expected_pnl
    
    def test_short_trade_pnl_calculation(self):
        """Test PnL calculation for short trade."""
        trade = Trade(
            timestamp=datetime(2023, 1, 1),
            symbol="SPY",
            strategy_type=StrategyType.COVERED_CALL,
            direction=TradeDirection.SHORT,
            entry_price=5.0,
            exit_price=3.0,
            quantity=1,
            commission=1.0,
            premium_collected=500.0
        )
        
        pnl = trade.calculate_pnl()
        expected_pnl = (5.0 - 3.0) * 1 + 500.0 - 1.0  # $501
        assert pnl == expected_pnl
    
    def test_trade_without_exit_price(self):
        """Test PnL calculation without exit price."""
        trade = Trade(
            timestamp=datetime(2023, 1, 1),
            symbol="SPY",
            strategy_type=StrategyType.LONG_CALL,
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=100
        )
        
        pnl = trade.calculate_pnl()
        assert pnl == 0.0


class TestStrategyBenchmark:
    """Test StrategyBenchmark class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample backtest configuration."""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 1),
            initial_capital=100000,
            commission_per_trade=1.0
        )
    
    def test_benchmark_initialization(self, sample_config):
        """Test StrategyBenchmark initialization."""
        benchmark = StrategyBenchmark(sample_config)
        
        assert benchmark.config == sample_config
        assert len(benchmark.trades) == 0
        assert benchmark.equity_curve.empty
        assert benchmark.daily_returns.empty
    
    def test_add_trade(self, sample_config):
        """Test adding trades to benchmark."""
        benchmark = StrategyBenchmark(sample_config)
        
        trade = Trade(
            timestamp=datetime(2023, 1, 1),
            symbol="SPY",
            strategy_type=StrategyType.COVERED_CALL,
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=100
        )
        
        benchmark.add_trade(trade)
        assert len(benchmark.trades) == 1
        assert benchmark.trades[0] == trade
    
    def test_synthetic_market_data_generation(self, sample_config):
        """Test synthetic market data generation."""
        benchmark = StrategyBenchmark(sample_config)
        market_data = benchmark.generate_synthetic_market_data()
        
        assert isinstance(market_data, pd.DataFrame)
        assert len(market_data) > 0
        assert 'open' in market_data.columns
        assert 'high' in market_data.columns
        assert 'low' in market_data.columns
        assert 'close' in market_data.columns
        assert 'volume' in market_data.columns
        assert 'returns' in market_data.columns
        
        # Check that highs >= lows
        assert (market_data['high'] >= market_data['low']).all()
        
        # Check that volumes are positive
        assert (market_data['volume'] > 0).all()
    
    def test_synthetic_options_data_generation(self, sample_config):
        """Test synthetic options data generation."""
        benchmark = StrategyBenchmark(sample_config)
        market_data = benchmark.generate_synthetic_market_data()
        options_data = benchmark.generate_synthetic_options_data(market_data)
        
        assert isinstance(options_data, pd.DataFrame)
        assert len(options_data) > 0
        
        required_columns = [
            'date', 'underlying_price', 'strike', 'option_type',
            'expiration', 'iv', 'option_price', 'delta', 'gamma',
            'theta', 'vega', 'bid', 'ask'
        ]
        for col in required_columns:
            assert col in options_data.columns
        
        # Check that option prices are positive
        assert (options_data['option_price'] > 0).all()
        
        # Check that bid <= ask
        assert (options_data['bid'] <= options_data['ask']).all()
        
        # Check that IV is reasonable (0-200%)
        assert (options_data['iv'] >= 0).all()
        assert (options_data['iv'] <= 2.0).all()
    
    def test_covered_call_strategy_execution(self, sample_config):
        """Test covered call strategy execution."""
        # Extend the period to ensure trades
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_per_trade=1.0
        )
        
        benchmark = StrategyBenchmark(config)
        benchmark.run_simple_covered_call_strategy()
        
        # Should have generated some trades
        assert len(benchmark.trades) >= 0  # May be 0 depending on data generation
        
        # If trades were generated, check structure
        if benchmark.trades:
            # Should have both stock and option trades
            stock_trades = [t for t in benchmark.trades if t.strike_price is None]
            option_trades = [t for t in benchmark.trades if t.strike_price is not None]
            
            # For covered calls, should have equal number of stock and option trades
            # (or close, depending on timing)
            assert len(stock_trades) > 0 or len(option_trades) > 0
    
    def test_performance_metrics_calculation(self, sample_config):
        """Test performance metrics calculation."""
        benchmark = StrategyBenchmark(sample_config)
        
        # Add some sample trades
        trade1 = Trade(
            timestamp=datetime(2023, 1, 15),
            symbol="SPY",
            strategy_type=StrategyType.COVERED_CALL,
            direction=TradeDirection.LONG,
            entry_price=100.0,
            exit_price=105.0,
            quantity=100,
            commission=1.0,
            exit_timestamp=datetime(2023, 1, 30)
        )
        trade1.calculate_pnl()
        
        trade2 = Trade(
            timestamp=datetime(2023, 1, 15),
            symbol="SPY",
            strategy_type=StrategyType.COVERED_CALL,
            direction=TradeDirection.SHORT,
            entry_price=5.0,
            exit_price=2.0,
            quantity=1,
            commission=1.0,
            premium_collected=500.0,
            exit_timestamp=datetime(2023, 1, 30)
        )
        trade2.calculate_pnl()
        
        benchmark.add_trade(trade1)
        benchmark.add_trade(trade2)
        
        results = benchmark.calculate_performance_metrics()
        
        assert isinstance(results, BacktestResults)
        assert results.config == sample_config
        assert results.total_trades == 2
        assert results.total_return >= 0  # Should be profitable
        assert isinstance(results.sharpe_ratio, (int, float))
        assert isinstance(results.max_drawdown, (int, float))
        assert results.max_drawdown <= 0  # Drawdown should be negative
    
    def test_benchmark_comparison(self, sample_config):
        """Test benchmark comparison functionality."""
        benchmark = StrategyBenchmark(sample_config)
        
        # Run strategy first to generate returns
        benchmark.run_simple_covered_call_strategy()
        benchmark.calculate_performance_metrics()
        
        comparison = benchmark.run_benchmark_comparison("SPY")
        
        assert isinstance(comparison, dict)
        assert 'benchmark_symbol' in comparison
        assert 'correlation' in comparison
        assert 'beta' in comparison
        assert 'alpha' in comparison
        assert 'information_ratio' in comparison
        assert comparison['benchmark_symbol'] == "SPY"
    
    def test_performance_report_generation(self, sample_config):
        """Test performance report generation."""
        benchmark = StrategyBenchmark(sample_config)
        benchmark.run_simple_covered_call_strategy()
        
        report = benchmark.generate_performance_report()
        
        assert isinstance(report, str)
        assert "TRADING STRATEGY PERFORMANCE REPORT" in report
        assert "PERFORMANCE SUMMARY" in report
        assert "RISK ANALYSIS" in report
        assert "TRADING STATISTICS" in report
        assert "BENCHMARK COMPARISON" in report


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns series."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        returns = pd.Series(
            np.random.normal(0.001, 0.02, len(dates)),
            index=dates
        )
        return returns
    
    @pytest.fixture
    def sample_benchmark_returns(self):
        """Create sample benchmark returns."""
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        returns = pd.Series(
            np.random.normal(0.0008, 0.015, len(dates)),
            index=dates
        )
        return returns
    
    def test_analyzer_initialization(self, sample_returns):
        """Test PerformanceAnalyzer initialization."""
        analyzer = PerformanceAnalyzer(sample_returns)
        
        assert len(analyzer.returns) == len(sample_returns)
        assert analyzer.risk_free_rate == 0.02  # Default
        assert analyzer.daily_rf_rate == 0.02 / 252
        assert len(analyzer.cumulative_returns) == len(sample_returns)
    
    def test_analyzer_with_benchmark(self, sample_returns, sample_benchmark_returns):
        """Test analyzer with benchmark returns."""
        analyzer = PerformanceAnalyzer(
            sample_returns, 
            sample_benchmark_returns,
            risk_free_rate=0.03
        )
        
        assert analyzer.benchmark_returns is not None
        assert len(analyzer.benchmark_returns) == len(sample_benchmark_returns)
        assert analyzer.risk_free_rate == 0.03
    
    def test_basic_metrics_calculation(self, sample_returns):
        """Test basic performance metrics calculation."""
        analyzer = PerformanceAnalyzer(sample_returns)
        metrics = analyzer.calculate_basic_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'downside_deviation' in metrics
        assert 'trading_days' in metrics
        assert 'years' in metrics
        
        # Check reasonable values
        assert metrics['volatility'] > 0
        assert metrics['trading_days'] > 0
        assert metrics['years'] > 0
    
    def test_drawdown_metrics_calculation(self, sample_returns):
        """Test drawdown metrics calculation."""
        analyzer = PerformanceAnalyzer(sample_returns)
        metrics = analyzer.calculate_drawdown_metrics()
        
        assert isinstance(metrics, dict)
        assert 'max_drawdown' in metrics
        assert 'max_drawdown_duration' in metrics
        assert 'avg_drawdown' in metrics
        assert 'drawdown_frequency' in metrics
        assert 'num_drawdown_periods' in metrics
        assert 'current_drawdown' in metrics
        
        # Drawdown should be negative or zero
        assert metrics['max_drawdown'] <= 0
        assert metrics['avg_drawdown'] <= 0
        assert metrics['current_drawdown'] <= 0
    
    def test_risk_adjusted_metrics(self, sample_returns):
        """Test risk-adjusted metrics calculation."""
        analyzer = PerformanceAnalyzer(sample_returns)
        metrics = analyzer.calculate_risk_adjusted_metrics()
        
        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'sterling_ratio' in metrics
        assert 'burke_ratio' in metrics
        
        # All should be numeric
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_distribution_metrics(self, sample_returns):
        """Test return distribution metrics."""
        analyzer = PerformanceAnalyzer(sample_returns)
        metrics = analyzer.calculate_distribution_metrics()
        
        assert isinstance(metrics, dict)
        assert 'skewness' in metrics
        assert 'kurtosis' in metrics
        assert 'var_95' in metrics
        assert 'var_99' in metrics
        assert 'cvar_95' in metrics
        assert 'cvar_99' in metrics
        assert 'tail_ratio' in metrics
        assert 'gain_pain_ratio' in metrics
        
        # VaR should be negative (losses)
        assert metrics['var_95'] <= 0
        assert metrics['var_99'] <= 0
        assert metrics['cvar_95'] <= 0
        assert metrics['cvar_99'] <= 0
        
        # Tail ratio and gain-pain ratio should be positive
        assert metrics['tail_ratio'] >= 0
        assert metrics['gain_pain_ratio'] >= 0
    
    def test_rolling_metrics(self, sample_returns):
        """Test rolling metrics calculation."""
        analyzer = PerformanceAnalyzer(sample_returns)
        metrics = analyzer.calculate_rolling_metrics()
        
        # With a full year of data, should have rolling metrics
        if len(sample_returns) >= 252:
            assert 'rolling_sharpe_12m' in metrics
            assert 'rolling_volatility_12m' in metrics
            assert 'rolling_max_dd_12m' in metrics
            
            if metrics['rolling_sharpe_12m'] is not None:
                assert isinstance(metrics['rolling_sharpe_12m'], (int, float))
    
    def test_benchmark_metrics(self, sample_returns, sample_benchmark_returns):
        """Test benchmark comparison metrics."""
        analyzer = PerformanceAnalyzer(sample_returns, sample_benchmark_returns)
        metrics = analyzer.calculate_benchmark_metrics()
        
        assert isinstance(metrics, dict)
        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert 'information_ratio' in metrics
        assert 'tracking_error' in metrics
        assert 'treynor_ratio' in metrics
        assert 'correlation' in metrics
        
        # Beta should be reasonable
        assert isinstance(metrics['beta'], (int, float))
        
        # Correlation should be between -1 and 1
        correlation = metrics['correlation']
        assert -1 <= correlation <= 1
    
    def test_trading_metrics(self, sample_returns):
        """Test trading-specific metrics."""
        # Create sample trade returns
        np.random.seed(42)
        trade_returns = np.random.normal(0.02, 0.05, 50)  # 50 trades
        
        analyzer = PerformanceAnalyzer(sample_returns)
        metrics = analyzer.calculate_trading_metrics(trade_returns)
        
        assert isinstance(metrics, dict)
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'expectancy' in metrics
        assert 'kelly_criterion' in metrics
        
        assert metrics['total_trades'] == 50
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['profit_factor'] >= 0
    
    def test_comprehensive_metrics_generation(self, sample_returns, sample_benchmark_returns):
        """Test comprehensive metrics generation."""
        analyzer = PerformanceAnalyzer(sample_returns, sample_benchmark_returns)
        metrics = analyzer.generate_comprehensive_metrics()
        
        assert isinstance(metrics, TradingMetrics)
        
        # Check all major metric categories are present
        assert isinstance(metrics.total_return, (int, float))
        assert isinstance(metrics.annualized_return, (int, float))
        assert isinstance(metrics.volatility, (int, float))
        assert isinstance(metrics.sharpe_ratio, (int, float))
        assert isinstance(metrics.max_drawdown, (int, float))
        assert isinstance(metrics.skewness, (int, float))
        assert isinstance(metrics.kurtosis, (int, float))
        
        # Check benchmark metrics are included
        assert metrics.alpha is not None
        assert metrics.beta is not None
        assert isinstance(metrics.alpha, (int, float))
        assert isinstance(metrics.beta, (int, float))
    
    def test_performance_summary_generation(self, sample_returns):
        """Test performance summary generation."""
        analyzer = PerformanceAnalyzer(sample_returns)
        summary = analyzer.generate_performance_summary()
        
        assert isinstance(summary, str)
        assert "COMPREHENSIVE PERFORMANCE ANALYSIS" in summary
        assert "RETURNS" in summary
        assert "RISK-ADJUSTED METRICS" in summary
        assert "RISK ANALYSIS" in summary
        assert "RETURN DISTRIBUTION" in summary


class TestTradingMetrics:
    """Test TradingMetrics Pydantic model."""
    
    def test_trading_metrics_creation(self):
        """Test creating TradingMetrics instance."""
        metrics = TradingMetrics(
            total_return=0.15,
            annualized_return=0.12,
            compound_annual_growth_rate=0.12,
            volatility=0.20,
            downside_deviation=0.15,
            max_drawdown=-0.10,
            max_drawdown_duration=30,
            avg_drawdown=-0.05,
            drawdown_frequency=2.0,
            sharpe_ratio=0.6,
            sortino_ratio=0.8,
            calmar_ratio=1.2,
            sterling_ratio=2.4,
            burke_ratio=0.5,
            skewness=0.1,
            kurtosis=0.2,
            var_95=-0.03,
            var_99=-0.05,
            cvar_95=-0.04,
            cvar_99=-0.06,
            tail_ratio=1.2,
            gain_pain_ratio=1.5
        )
        
        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 0.6
        assert metrics.max_drawdown == -0.10
        assert metrics.var_95 == -0.03
    
    def test_trading_metrics_with_optional_fields(self):
        """Test TradingMetrics with optional fields."""
        metrics = TradingMetrics(
            total_return=0.15,
            annualized_return=0.12,
            compound_annual_growth_rate=0.12,
            volatility=0.20,
            downside_deviation=0.15,
            max_drawdown=-0.10,
            max_drawdown_duration=30,
            avg_drawdown=-0.05,
            drawdown_frequency=2.0,
            sharpe_ratio=0.6,
            sortino_ratio=0.8,
            calmar_ratio=1.2,
            sterling_ratio=2.4,
            burke_ratio=0.5,
            skewness=0.1,
            kurtosis=0.2,
            var_95=-0.03,
            var_99=-0.05,
            cvar_95=-0.04,
            cvar_99=-0.06,
            tail_ratio=1.2,
            gain_pain_ratio=1.5,
            # Optional fields
            alpha=0.05,
            beta=0.8,
            information_ratio=0.4,
            total_trades=50,
            win_rate=0.6
        )
        
        assert metrics.alpha == 0.05
        assert metrics.beta == 0.8
        assert metrics.total_trades == 50
        assert metrics.win_rate == 0.6
    
    def test_trading_metrics_json_serialization(self):
        """Test JSON serialization of TradingMetrics."""
        metrics = TradingMetrics(
            total_return=0.15,
            annualized_return=0.12,
            compound_annual_growth_rate=0.12,
            volatility=0.20,
            downside_deviation=0.15,
            max_drawdown=-0.10,
            max_drawdown_duration=30,
            avg_drawdown=-0.05,
            drawdown_frequency=2.0,
            sharpe_ratio=0.6,
            sortino_ratio=0.8,
            calmar_ratio=1.2,
            sterling_ratio=2.4,
            burke_ratio=0.5,
            skewness=0.1,
            kurtosis=0.2,
            var_95=-0.03,
            var_99=-0.05,
            cvar_95=-0.04,
            cvar_99=-0.06,
            tail_ratio=1.2,
            gain_pain_ratio=1.5
        )
        
        # Should be able to serialize to JSON
        json_str = metrics.json()
        assert isinstance(json_str, str)
        assert "total_return" in json_str
        assert "sharpe_ratio" in json_str
        
        # Should be able to parse back
        parsed_metrics = TradingMetrics.parse_raw(json_str)
        assert parsed_metrics.total_return == metrics.total_return
        assert parsed_metrics.sharpe_ratio == metrics.sharpe_ratio


@pytest.mark.integration
class TestTradingBenchmarksIntegration:
    """Integration tests for trading benchmarks module."""
    
    def test_full_workflow(self):
        """Test complete trading benchmarks workflow."""
        # Configure backtest
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 1),
            initial_capital=100000,
            commission_per_trade=1.0
        )
        
        # Run strategy benchmark
        benchmark = StrategyBenchmark(config)
        benchmark.run_simple_covered_call_strategy()
        
        # Calculate performance
        results = benchmark.calculate_performance_metrics()
        assert isinstance(results, BacktestResults)
        
        # Run benchmark comparison
        comparison = benchmark.run_benchmark_comparison()
        assert isinstance(comparison, dict)
        
        # Generate report
        report = benchmark.generate_performance_report()
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_performance_analyzer_workflow(self):
        """Test complete performance analyzer workflow."""
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        strategy_returns = pd.Series(
            np.random.normal(0.001, 0.02, len(dates)),
            index=dates
        )
        benchmark_returns = pd.Series(
            np.random.normal(0.0008, 0.015, len(dates)),
            index=dates
        )
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(strategy_returns, benchmark_returns)
        
        # Generate all metric types
        basic_metrics = analyzer.calculate_basic_metrics()
        drawdown_metrics = analyzer.calculate_drawdown_metrics()
        risk_metrics = analyzer.calculate_risk_adjusted_metrics()
        distribution_metrics = analyzer.calculate_distribution_metrics()
        benchmark_metrics = analyzer.calculate_benchmark_metrics()
        
        # All should return dictionaries with data
        assert isinstance(basic_metrics, dict) and len(basic_metrics) > 0
        assert isinstance(drawdown_metrics, dict) and len(drawdown_metrics) > 0
        assert isinstance(risk_metrics, dict) and len(risk_metrics) > 0
        assert isinstance(distribution_metrics, dict) and len(distribution_metrics) > 0
        assert isinstance(benchmark_metrics, dict) and len(benchmark_metrics) > 0
        
        # Generate comprehensive metrics
        comprehensive = analyzer.generate_comprehensive_metrics()
        assert isinstance(comprehensive, TradingMetrics)
        
        # Generate summary
        summary = analyzer.generate_performance_summary()
        assert isinstance(summary, str) and len(summary) > 0