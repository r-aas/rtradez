"""
Comprehensive tests for risk management module.

Tests for position sizing, risk limits, portfolio risk, and risk monitoring.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal

from rtradez.risk import (
    KellyCriterion, KellyConfig, PositionSizeResult,
    RiskLimits, PortfolioRisk, RiskMonitor,
    MarginCalculator, PositionSizer
)


class TestKellyConfig:
    """Test KellyConfig validation and configuration."""
    
    def test_valid_config_creation(self):
        """Test creating valid Kelly configuration."""
        config = KellyConfig(
            lookback_period=252,
            min_trades=30,
            max_fraction=0.25,
            confidence_level=0.95,
            adjustment_factor=0.75
        )
        
        assert config.lookback_period == 252
        assert config.min_trades == 30
        assert config.max_fraction == 0.25
        assert config.confidence_level == 0.95
        assert config.adjustment_factor == 0.75
    
    def test_invalid_config_raises_errors(self):
        """Test that invalid configurations raise errors."""
        with pytest.raises(ValueError):
            KellyConfig(
                lookback_period=0,  # Invalid
                min_trades=30
            )
        
        with pytest.raises(ValueError):
            KellyConfig(
                lookback_period=252,
                max_fraction=1.5  # Invalid (> 1.0)
            )
        
        with pytest.raises(ValueError):
            KellyConfig(
                lookback_period=252,
                confidence_level=1.5  # Invalid (> 1.0)
            )
    
    def test_default_values(self):
        """Test default configuration values."""
        config = KellyConfig()
        
        assert config.lookback_period == 252  # 1 year default
        assert config.min_trades == 20
        assert config.max_fraction == 0.20  # 20% max
        assert config.confidence_level == 0.95
        assert config.adjustment_factor == 0.50  # Conservative default


class TestKellyCriterion:
    """Test Kelly Criterion calculations."""
    
    @pytest.fixture
    def kelly_calculator(self):
        """Create Kelly criterion calculator."""
        config = KellyConfig(
            lookback_period=100,
            min_trades=10,
            max_fraction=0.30
        )
        return KellyCriterion(config)
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample trade returns."""
        np.random.seed(42)
        # Create returns with positive expectancy
        returns = []
        for _ in range(50):
            if np.random.random() < 0.6:  # 60% win rate
                returns.append(np.random.uniform(0.02, 0.08))  # 2-8% wins
            else:
                returns.append(np.random.uniform(-0.04, -0.01))  # 1-4% losses
        return returns
    
    def test_kelly_initialization(self, kelly_calculator):
        """Test Kelly criterion initialization."""
        assert kelly_calculator.config.lookback_period == 100
        assert kelly_calculator.config.min_trades == 10
        assert kelly_calculator.config.max_fraction == 0.30
    
    def test_calculate_optimal_fraction(self, kelly_calculator, sample_returns):
        """Test optimal fraction calculation."""
        fraction = kelly_calculator.calculate_optimal_fraction(sample_returns)
        
        assert isinstance(fraction, float)
        assert 0 <= fraction <= kelly_calculator.config.max_fraction
    
    def test_insufficient_trades(self, kelly_calculator):
        """Test behavior with insufficient trade history."""
        few_returns = [0.02, -0.01, 0.03]  # Only 3 trades, need 10 minimum
        
        fraction = kelly_calculator.calculate_optimal_fraction(few_returns)
        assert fraction == 0.0  # Should return 0 for insufficient data
    
    def test_negative_expectancy(self, kelly_calculator):
        """Test behavior with negative expectancy."""
        losing_returns = [-0.02, -0.01, -0.03, -0.01] * 10  # All losses
        
        fraction = kelly_calculator.calculate_optimal_fraction(losing_returns)
        assert fraction == 0.0  # Should return 0 for negative expectancy
    
    def test_calculate_with_confidence_adjustment(self, kelly_calculator, sample_returns):
        """Test Kelly calculation with confidence adjustment."""
        result = kelly_calculator.calculate_with_confidence(sample_returns)
        
        assert isinstance(result, PositionSizeResult)
        assert result.recommended_size >= 0
        assert result.recommended_size <= kelly_calculator.config.max_fraction
        assert result.confidence_level == kelly_calculator.config.confidence_level
        assert isinstance(result.expected_return, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.avg_win, float)
        assert isinstance(result.avg_loss, float)
    
    def test_position_size_result_validation(self, kelly_calculator, sample_returns):
        """Test PositionSizeResult validation."""
        result = kelly_calculator.calculate_with_confidence(sample_returns)
        
        # Check that all fields are properly calculated
        assert 0 <= result.win_rate <= 1
        assert result.avg_win >= 0
        assert result.avg_loss <= 0
        assert result.total_trades > 0
        assert result.raw_kelly_fraction >= 0
    
    def test_stress_test_scenarios(self, kelly_calculator):
        """Test Kelly calculation under stress scenarios."""
        # High volatility scenario
        high_vol_returns = np.random.normal(0.01, 0.10, 50)  # High volatility
        fraction_high_vol = kelly_calculator.calculate_optimal_fraction(high_vol_returns)
        
        # Low volatility scenario
        low_vol_returns = np.random.normal(0.01, 0.02, 50)  # Low volatility
        fraction_low_vol = kelly_calculator.calculate_optimal_fraction(low_vol_returns)
        
        # High volatility should generally result in smaller position sizes
        assert isinstance(fraction_high_vol, float)
        assert isinstance(fraction_low_vol, float)


class TestRiskLimits:
    """Test RiskLimits functionality."""
    
    @pytest.fixture
    def risk_limits(self):
        """Create risk limits instance."""
        return RiskLimits(
            max_portfolio_risk=0.02,
            max_single_position=0.10,
            max_sector_allocation=0.30,
            max_strategy_allocation=0.40,
            max_correlation=0.80,
            max_drawdown=0.15,
            var_limit=0.05,
            leverage_limit=2.0
        )
    
    def test_risk_limits_initialization(self, risk_limits):
        """Test RiskLimits initialization."""
        assert risk_limits.max_portfolio_risk == 0.02
        assert risk_limits.max_single_position == 0.10
        assert risk_limits.max_sector_allocation == 0.30
        assert risk_limits.max_strategy_allocation == 0.40
        assert risk_limits.max_correlation == 0.80
        assert risk_limits.max_drawdown == 0.15
        assert risk_limits.var_limit == 0.05
        assert risk_limits.leverage_limit == 2.0
    
    def test_validate_position_size(self, risk_limits):
        """Test position size validation."""
        # Valid position
        assert risk_limits.validate_position_size(0.08, 100000) == True
        
        # Invalid position (exceeds limit)
        assert risk_limits.validate_position_size(0.12, 100000) == False
    
    def test_validate_portfolio_risk(self, risk_limits):
        """Test portfolio risk validation."""
        # Valid risk level
        assert risk_limits.validate_portfolio_risk(0.015) == True
        
        # Invalid risk level (exceeds limit)
        assert risk_limits.validate_portfolio_risk(0.025) == False
    
    def test_validate_sector_allocation(self, risk_limits):
        """Test sector allocation validation."""
        sector_allocations = {
            'Technology': 0.25,
            'Healthcare': 0.20,
            'Finance': 0.15,
            'Energy': 0.10
        }
        
        # All sectors within limits
        assert risk_limits.validate_sector_allocations(sector_allocations) == True
        
        # One sector exceeds limit
        sector_allocations['Technology'] = 0.35
        assert risk_limits.validate_sector_allocations(sector_allocations) == False
    
    def test_validate_correlation_limit(self, risk_limits):
        """Test correlation limit validation."""
        # Valid correlation
        assert risk_limits.validate_correlation(0.75) == True
        
        # Invalid correlation (too high)
        assert risk_limits.validate_correlation(0.85) == False
    
    def test_validate_drawdown_limit(self, risk_limits):
        """Test drawdown limit validation."""
        # Valid drawdown
        assert risk_limits.validate_drawdown(-0.10) == True
        
        # Invalid drawdown (exceeds limit)
        assert risk_limits.validate_drawdown(-0.20) == False
    
    def test_check_all_limits(self, risk_limits):
        """Test comprehensive limit checking."""
        portfolio_data = {
            'portfolio_risk': 0.018,
            'max_position_size': 0.09,
            'max_sector_allocation': 0.28,
            'max_correlation': 0.75,
            'current_drawdown': -0.12,
            'var_95': 0.04,
            'leverage': 1.5
        }
        
        violations = risk_limits.check_all_limits(portfolio_data)
        assert isinstance(violations, list)
        assert len(violations) == 0  # All limits satisfied
        
        # Add violation
        portfolio_data['portfolio_risk'] = 0.025
        violations = risk_limits.check_all_limits(portfolio_data)
        assert len(violations) > 0
        assert any("portfolio risk" in v.lower() for v in violations)


class TestPortfolioRisk:
    """Test PortfolioRisk calculations."""
    
    @pytest.fixture
    def portfolio_risk(self):
        """Create portfolio risk calculator."""
        return PortfolioRisk(
            confidence_level=0.95,
            time_horizon=1,  # 1 day
            simulation_runs=1000
        )
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio data."""
        positions = pd.DataFrame({
            'symbol': ['SPY', 'QQQ', 'IWM'],
            'weight': [0.50, 0.30, 0.20],
            'volatility': [0.15, 0.20, 0.25],
            'expected_return': [0.08, 0.10, 0.12]
        })
        
        correlation_matrix = pd.DataFrame({
            'SPY': [1.00, 0.80, 0.75],
            'QQQ': [0.80, 1.00, 0.70],
            'IWM': [0.75, 0.70, 1.00]
        }, index=['SPY', 'QQQ', 'IWM'])
        
        return positions, correlation_matrix
    
    def test_portfolio_risk_initialization(self, portfolio_risk):
        """Test PortfolioRisk initialization."""
        assert portfolio_risk.confidence_level == 0.95
        assert portfolio_risk.time_horizon == 1
        assert portfolio_risk.simulation_runs == 1000
    
    def test_calculate_portfolio_var(self, portfolio_risk, sample_portfolio_data):
        """Test portfolio VaR calculation."""
        positions, correlation_matrix = sample_portfolio_data
        
        var = portfolio_risk.calculate_portfolio_var(positions, correlation_matrix)
        
        assert isinstance(var, float)
        assert var < 0  # VaR should be negative (loss)
        assert var > -1  # Reasonable bounds
    
    def test_calculate_component_var(self, portfolio_risk, sample_portfolio_data):
        """Test component VaR calculation."""
        positions, correlation_matrix = sample_portfolio_data
        
        component_vars = portfolio_risk.calculate_component_var(positions, correlation_matrix)
        
        assert isinstance(component_vars, pd.Series)
        assert len(component_vars) == len(positions)
        assert all(component_vars.index == positions['symbol'])
        
        # Component VaRs should sum approximately to portfolio VaR
        portfolio_var = portfolio_risk.calculate_portfolio_var(positions, correlation_matrix)
        assert abs(component_vars.sum() - portfolio_var) < 0.001
    
    def test_calculate_marginal_var(self, portfolio_risk, sample_portfolio_data):
        """Test marginal VaR calculation."""
        positions, correlation_matrix = sample_portfolio_data
        
        marginal_vars = portfolio_risk.calculate_marginal_var(positions, correlation_matrix)
        
        assert isinstance(marginal_vars, pd.Series)
        assert len(marginal_vars) == len(positions)
        assert all(marginal_vars.index == positions['symbol'])
    
    @patch('numpy.random.multivariate_normal')
    def test_monte_carlo_simulation(self, mock_random, portfolio_risk, sample_portfolio_data):
        """Test Monte Carlo portfolio simulation."""
        positions, correlation_matrix = sample_portfolio_data
        
        # Mock random returns
        np.random.seed(42)
        mock_returns = np.random.multivariate_normal(
            mean=[0.0003, 0.0004, 0.0005],  # Daily returns
            cov=correlation_matrix.values * 0.01,  # Scaled covariance
            size=1000
        )
        mock_random.return_value = mock_returns
        
        portfolio_returns = portfolio_risk.monte_carlo_simulation(positions, correlation_matrix)
        
        assert isinstance(portfolio_returns, np.ndarray)
        assert len(portfolio_returns) == portfolio_risk.simulation_runs
        
        # Calculate VaR from simulation
        var_simulated = np.percentile(portfolio_returns, (1 - portfolio_risk.confidence_level) * 100)
        assert isinstance(var_simulated, float)
    
    def test_calculate_expected_shortfall(self, portfolio_risk, sample_portfolio_data):
        """Test Expected Shortfall (CVaR) calculation."""
        positions, correlation_matrix = sample_portfolio_data
        
        var = portfolio_risk.calculate_portfolio_var(positions, correlation_matrix)
        es = portfolio_risk.calculate_expected_shortfall(positions, correlation_matrix)
        
        assert isinstance(es, float)
        assert es < var  # ES should be more negative than VaR
    
    def test_stress_test_scenarios(self, portfolio_risk, sample_portfolio_data):
        """Test stress testing scenarios."""
        positions, correlation_matrix = sample_portfolio_data
        
        # Market crash scenario: high correlation, high volatility
        stress_correlation = correlation_matrix * 1.2  # Increase correlations
        np.fill_diagonal(stress_correlation.values, 1.0)  # Keep diagonal as 1.0
        
        stress_positions = positions.copy()
        stress_positions['volatility'] *= 2.0  # Double volatility
        
        normal_var = portfolio_risk.calculate_portfolio_var(positions, correlation_matrix)
        stress_var = portfolio_risk.calculate_portfolio_var(stress_positions, stress_correlation)
        
        # Stress VaR should be more negative (worse)
        assert stress_var < normal_var


class TestRiskMonitor:
    """Test RiskMonitor functionality."""
    
    @pytest.fixture
    def risk_limits(self):
        """Create risk limits for monitoring."""
        return RiskLimits(
            max_portfolio_risk=0.02,
            max_single_position=0.10,
            max_drawdown=0.15,
            var_limit=0.05
        )
    
    @pytest.fixture
    def risk_monitor(self, risk_limits):
        """Create risk monitor instance."""
        return RiskMonitor(
            risk_limits=risk_limits,
            monitoring_frequency=timedelta(minutes=5),
            alert_threshold=0.8  # Alert at 80% of limit
        )
    
    def test_risk_monitor_initialization(self, risk_monitor, risk_limits):
        """Test RiskMonitor initialization."""
        assert risk_monitor.risk_limits == risk_limits
        assert risk_monitor.monitoring_frequency == timedelta(minutes=5)
        assert risk_monitor.alert_threshold == 0.8
        assert len(risk_monitor.alerts) == 0
    
    def test_monitor_portfolio_risk(self, risk_monitor):
        """Test portfolio risk monitoring."""
        portfolio_data = {
            'portfolio_id': 'TEST_PORTFOLIO',
            'total_value': 100000,
            'portfolio_risk': 0.018,  # 90% of 0.02 limit
            'largest_position': 0.09,  # 90% of 0.10 limit
            'current_drawdown': -0.12,  # 80% of 0.15 limit
            'var_95': 0.04,  # 80% of 0.05 limit
            'timestamp': datetime.now()
        }
        
        alerts = risk_monitor.monitor_portfolio(portfolio_data)
        
        # Should generate alerts for metrics near limits
        assert isinstance(alerts, list)
        # Since all metrics are at 80-90% of limits, should trigger alerts
        assert len(alerts) > 0
    
    def test_monitor_position_limits(self, risk_monitor):
        """Test individual position monitoring."""
        position_data = {
            'position_id': 'POS_001',
            'symbol': 'SPY',
            'size_pct': 0.085,  # 85% of 10% limit
            'portfolio_value': 100000,
            'timestamp': datetime.now()
        }
        
        alerts = risk_monitor.monitor_position(position_data)
        
        # Should generate alert for position size near limit
        assert isinstance(alerts, list)
        assert len(alerts) > 0
        assert any("position size" in alert.message.lower() for alert in alerts)
    
    def test_breach_detection(self, risk_monitor):
        """Test limit breach detection."""
        portfolio_data = {
            'portfolio_id': 'BREACH_TEST',
            'portfolio_risk': 0.025,  # Exceeds 0.02 limit
            'largest_position': 0.12,  # Exceeds 0.10 limit
            'timestamp': datetime.now()
        }
        
        alerts = risk_monitor.monitor_portfolio(portfolio_data)
        
        # Should generate breach alerts
        breach_alerts = [a for a in alerts if a.severity == 'CRITICAL']
        assert len(breach_alerts) > 0
    
    def test_alert_history(self, risk_monitor):
        """Test alert history tracking."""
        portfolio_data = {
            'portfolio_id': 'HISTORY_TEST',
            'portfolio_risk': 0.025,  # Exceeds limit
            'timestamp': datetime.now()
        }
        
        initial_alert_count = len(risk_monitor.alerts)
        
        risk_monitor.monitor_portfolio(portfolio_data)
        
        # Should have added alerts to history
        assert len(risk_monitor.alerts) > initial_alert_count
        
        # Test alert retrieval
        recent_alerts = risk_monitor.get_recent_alerts(hours=1)
        assert len(recent_alerts) > 0
    
    def test_risk_trend_analysis(self, risk_monitor):
        """Test risk trend analysis."""
        # Simulate increasing risk over time
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(5, 0, -1)]
        risk_levels = [0.012, 0.014, 0.016, 0.018, 0.020]  # Increasing trend
        
        portfolio_data_series = []
        for timestamp, risk_level in zip(timestamps, risk_levels):
            portfolio_data_series.append({
                'portfolio_id': 'TREND_TEST',
                'portfolio_risk': risk_level,
                'timestamp': timestamp
            })
        
        # Monitor each data point
        for data in portfolio_data_series:
            risk_monitor.monitor_portfolio(data)
        
        # Analyze trend
        trend_analysis = risk_monitor.analyze_risk_trend('TREND_TEST', hours=6)
        
        assert isinstance(trend_analysis, dict)
        assert 'trend_direction' in trend_analysis
        assert trend_analysis['trend_direction'] == 'increasing'
    
    def test_generate_risk_report(self, risk_monitor):
        """Test risk report generation."""
        # Add some historical data
        portfolio_data = {
            'portfolio_id': 'REPORT_TEST',
            'portfolio_risk': 0.018,
            'largest_position': 0.09,
            'current_drawdown': -0.08,
            'timestamp': datetime.now()
        }
        
        risk_monitor.monitor_portfolio(portfolio_data)
        
        report = risk_monitor.generate_risk_report('REPORT_TEST')
        
        assert isinstance(report, str)
        assert "Risk Monitoring Report" in report
        assert "REPORT_TEST" in report
        assert len(report) > 0


class TestMarginCalculator:
    """Test MarginCalculator functionality."""
    
    @pytest.fixture
    def margin_calculator(self):
        """Create margin calculator instance."""
        return MarginCalculator(
            initial_margin_rate=0.50,
            maintenance_margin_rate=0.30,
            options_margin_rate=0.20,
            portfolio_margin=True
        )
    
    def test_margin_calculator_initialization(self, margin_calculator):
        """Test MarginCalculator initialization."""
        assert margin_calculator.initial_margin_rate == 0.50
        assert margin_calculator.maintenance_margin_rate == 0.30
        assert margin_calculator.options_margin_rate == 0.20
        assert margin_calculator.portfolio_margin == True
    
    def test_calculate_stock_margin(self, margin_calculator):
        """Test stock position margin calculation."""
        position = {
            'symbol': 'SPY',
            'quantity': 100,
            'price': 400.0,
            'position_type': 'long'
        }
        
        margin_required = margin_calculator.calculate_stock_margin(position)
        
        # Long stock: 50% of position value
        expected_margin = 100 * 400.0 * 0.50
        assert margin_required == expected_margin
    
    def test_calculate_options_margin(self, margin_calculator):
        """Test options position margin calculation."""
        position = {
            'symbol': 'SPY',
            'option_type': 'call',
            'quantity': 10,
            'strike': 400.0,
            'premium': 5.0,
            'underlying_price': 405.0,
            'position_type': 'short'
        }
        
        margin_required = margin_calculator.calculate_options_margin(position)
        
        assert isinstance(margin_required, float)
        assert margin_required > 0
    
    def test_calculate_portfolio_margin(self, margin_calculator):
        """Test portfolio-level margin calculation."""
        positions = [
            {
                'symbol': 'SPY',
                'quantity': 100,
                'price': 400.0,
                'position_type': 'long',
                'asset_type': 'stock'
            },
            {
                'symbol': 'QQQ',
                'quantity': 200,
                'price': 250.0,
                'position_type': 'short',
                'asset_type': 'stock'
            }
        ]
        
        total_margin = margin_calculator.calculate_portfolio_margin(positions)
        
        assert isinstance(total_margin, float)
        assert total_margin > 0
    
    def test_margin_call_detection(self, margin_calculator):
        """Test margin call detection."""
        account_data = {
            'account_value': 50000,
            'margin_used': 35000,
            'maintenance_margin_required': 20000
        }
        
        is_margin_call = margin_calculator.check_margin_call(account_data)
        assert is_margin_call == False  # Sufficient margin
        
        # Reduce account value
        account_data['account_value'] = 18000
        is_margin_call = margin_calculator.check_margin_call(account_data)
        assert is_margin_call == True  # Margin call triggered


class TestPositionSizer:
    """Test PositionSizer integration."""
    
    @pytest.fixture
    def position_sizer(self):
        """Create position sizer instance."""
        kelly_config = KellyConfig(max_fraction=0.25)
        risk_limits = RiskLimits(max_single_position=0.10)
        
        return PositionSizer(
            kelly_config=kelly_config,
            risk_limits=risk_limits,
            default_method='kelly_criterion'
        )
    
    def test_position_sizer_initialization(self, position_sizer):
        """Test PositionSizer initialization."""
        assert position_sizer.kelly_config.max_fraction == 0.25
        assert position_sizer.risk_limits.max_single_position == 0.10
        assert position_sizer.default_method == 'kelly_criterion'
    
    @patch('rtradez.risk.position_sizing.KellyCriterion')
    def test_calculate_position_size(self, mock_kelly, position_sizer):
        """Test position size calculation."""
        # Mock Kelly criterion calculation
        mock_kelly_instance = Mock()
        mock_kelly.return_value = mock_kelly_instance
        mock_kelly_instance.calculate_with_confidence.return_value = PositionSizeResult(
            recommended_size=0.08,
            confidence_level=0.95,
            expected_return=0.12,
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.02,
            total_trades=50,
            raw_kelly_fraction=0.12,
            adjusted_fraction=0.08
        )
        
        trade_history = [0.02, -0.01, 0.03, -0.01, 0.04] * 10
        portfolio_value = 100000
        
        result = position_sizer.calculate_position_size(
            trade_history=trade_history,
            portfolio_value=portfolio_value,
            method='kelly_criterion'
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.recommended_size <= position_sizer.risk_limits.max_single_position
        assert 0 <= result.recommended_size <= 1


@pytest.mark.integration
class TestRiskManagementIntegration:
    """Integration tests for risk management components."""
    
    def test_full_risk_management_workflow(self):
        """Test complete risk management workflow."""
        # Setup components
        kelly_config = KellyConfig(max_fraction=0.20)
        risk_limits = RiskLimits(
            max_portfolio_risk=0.02,
            max_single_position=0.08,
            max_drawdown=0.15
        )
        portfolio_risk = PortfolioRisk(confidence_level=0.95)
        risk_monitor = RiskMonitor(risk_limits)
        
        # Simulate trading history
        trade_history = []
        np.random.seed(42)
        for _ in range(100):
            if np.random.random() < 0.65:  # 65% win rate
                trade_history.append(np.random.uniform(0.01, 0.06))
            else:
                trade_history.append(np.random.uniform(-0.03, -0.01))
        
        # Calculate Kelly optimal fraction
        kelly_calculator = KellyCriterion(kelly_config)
        kelly_result = kelly_calculator.calculate_with_confidence(trade_history)
        
        assert isinstance(kelly_result, PositionSizeResult)
        assert kelly_result.recommended_size > 0
        
        # Portfolio risk assessment
        positions = pd.DataFrame({
            'symbol': ['SPY', 'QQQ'],
            'weight': [0.60, 0.40],
            'volatility': [0.15, 0.20],
            'expected_return': [0.08, 0.10]
        })
        
        correlation_matrix = pd.DataFrame({
            'SPY': [1.0, 0.8],
            'QQQ': [0.8, 1.0]
        }, index=['SPY', 'QQQ'])
        
        portfolio_var = portfolio_risk.calculate_portfolio_var(positions, correlation_matrix)
        assert portfolio_var < 0  # Should be negative (loss)
        
        # Risk monitoring
        portfolio_data = {
            'portfolio_id': 'INTEGRATION_TEST',
            'portfolio_risk': abs(portfolio_var),
            'largest_position': positions['weight'].max(),
            'timestamp': datetime.now()
        }
        
        alerts = risk_monitor.monitor_portfolio(portfolio_data)
        assert isinstance(alerts, list)
        
        # Generate comprehensive risk report
        risk_report = risk_monitor.generate_risk_report('INTEGRATION_TEST')
        assert isinstance(risk_report, str)
        assert len(risk_report) > 0
    
    def test_stress_testing_integration(self):
        """Test integrated stress testing."""
        # Create stressed market conditions
        kelly_config = KellyConfig(max_fraction=0.15)
        risk_limits = RiskLimits(max_portfolio_risk=0.015)  # Tighter limits
        
        # Simulate stressed trade returns (higher volatility, lower wins)
        stressed_returns = []
        np.random.seed(123)
        for _ in range(50):
            if np.random.random() < 0.45:  # Lower win rate
                stressed_returns.append(np.random.uniform(0.005, 0.04))
            else:
                stressed_returns.append(np.random.uniform(-0.06, -0.01))  # Larger losses
        
        kelly_calculator = KellyCriterion(kelly_config)
        stressed_kelly = kelly_calculator.calculate_with_confidence(stressed_returns)
        
        # Under stress, Kelly should recommend smaller positions
        assert stressed_kelly.recommended_size < 0.15
        
        # Verify risk limits still enforced
        assert stressed_kelly.recommended_size <= risk_limits.max_single_position