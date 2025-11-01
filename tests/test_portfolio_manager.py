"""
Comprehensive tests for portfolio management module.

Tests for portfolio coordination, multi-strategy management, and allocation optimization.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal

from rtradez.portfolio.portfolio_manager import (
    PortfolioManager, PortfolioConfig, StrategyAllocation,
    PositionManager, Portfolio, PortfolioMetrics
)
from rtradez.risk.position_sizing import PositionSizeResult, KellyCriterion


class TestPortfolioConfig:
    """Test PortfolioConfig validation and configuration."""
    
    def test_valid_config_creation(self):
        """Test creating valid portfolio configuration."""
        config = PortfolioConfig(
            total_capital=100000,
            max_portfolio_risk=0.02,
            max_single_position=0.10,
            max_correlation=0.70,
            rebalance_frequency="weekly"
        )
        
        assert config.total_capital == 100000
        assert config.max_portfolio_risk == 0.02
        assert config.max_single_position == 0.10
        assert config.max_correlation == 0.70
        assert config.rebalance_frequency == "weekly"
    
    def test_invalid_capital_raises_error(self):
        """Test that negative capital raises validation error."""
        with pytest.raises(ValueError):
            PortfolioConfig(
                total_capital=-1000,  # Invalid
                max_portfolio_risk=0.02
            )
    
    def test_invalid_risk_limit_raises_error(self):
        """Test that invalid risk limits raise errors."""
        with pytest.raises(ValueError):
            PortfolioConfig(
                total_capital=100000,
                max_portfolio_risk=1.5  # Invalid (> 1.0)
            )
        
        with pytest.raises(ValueError):
            PortfolioConfig(
                total_capital=100000,
                max_portfolio_risk=0.02,
                max_single_position=1.2  # Invalid (> 1.0)
            )
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PortfolioConfig(total_capital=100000)
        
        assert config.max_portfolio_risk == 0.02  # 2% default
        assert config.max_single_position == 0.10  # 10% default
        assert config.max_correlation == 0.80  # 80% default
        assert config.rebalance_frequency == "daily"


class TestStrategyAllocation:
    """Test StrategyAllocation model."""
    
    def test_strategy_allocation_creation(self):
        """Test creating strategy allocation."""
        allocation = StrategyAllocation(
            strategy_id="covered_calls",
            strategy_name="Covered Call Strategy",
            target_allocation=0.30,
            current_allocation=0.25,
            max_allocation=0.40,
            min_allocation=0.10,
            risk_budget=0.008,
            expected_return=0.12,
            volatility=0.15
        )
        
        assert allocation.strategy_id == "covered_calls"
        assert allocation.target_allocation == 0.30
        assert allocation.current_allocation == 0.25
        assert allocation.max_allocation == 0.40
        assert allocation.risk_budget == 0.008
    
    def test_allocation_validation(self):
        """Test allocation validation."""
        with pytest.raises(ValueError):
            StrategyAllocation(
                strategy_id="test",
                strategy_name="Test",
                target_allocation=1.5,  # Invalid (> 1.0)
                risk_budget=0.01,
                expected_return=0.10,
                volatility=0.15
            )
    
    def test_allocation_drift_calculation(self):
        """Test allocation drift calculation."""
        allocation = StrategyAllocation(
            strategy_id="test",
            strategy_name="Test Strategy",
            target_allocation=0.30,
            current_allocation=0.35,
            risk_budget=0.01,
            expected_return=0.10,
            volatility=0.15
        )
        
        drift = allocation.calculate_drift()
        assert drift == 0.05  # 35% - 30% = 5%
    
    def test_rebalance_needed(self):
        """Test rebalance threshold detection."""
        allocation = StrategyAllocation(
            strategy_id="test",
            strategy_name="Test Strategy",
            target_allocation=0.30,
            current_allocation=0.35,
            drift_threshold=0.03,
            risk_budget=0.01,
            expected_return=0.10,
            volatility=0.15
        )
        
        assert allocation.needs_rebalance()  # 5% drift > 3% threshold
        
        allocation.current_allocation = 0.32
        assert not allocation.needs_rebalance()  # 2% drift < 3% threshold


class TestPositionManager:
    """Test PositionManager functionality."""
    
    @pytest.fixture
    def position_manager(self):
        """Create position manager instance."""
        return PositionManager(
            max_positions=10,
            position_size_method="fixed_fraction",
            default_position_size=0.02
        )
    
    def test_position_manager_initialization(self, position_manager):
        """Test PositionManager initialization."""
        assert position_manager.max_positions == 10
        assert position_manager.position_size_method == "fixed_fraction"
        assert position_manager.default_position_size == 0.02
        assert len(position_manager.positions) == 0
    
    def test_add_position(self, position_manager):
        """Test adding new position."""
        position = {
            'symbol': 'SPY',
            'strategy': 'covered_call',
            'quantity': 100,
            'entry_price': 400.0,
            'entry_date': datetime.now(),
            'position_value': 40000.0
        }
        
        position_manager.add_position("POS_001", position)
        
        assert len(position_manager.positions) == 1
        assert "POS_001" in position_manager.positions
        assert position_manager.positions["POS_001"]['symbol'] == 'SPY'
    
    def test_remove_position(self, position_manager):
        """Test removing position."""
        position = {
            'symbol': 'SPY',
            'strategy': 'covered_call',
            'quantity': 100,
            'entry_price': 400.0,
            'entry_date': datetime.now(),
            'position_value': 40000.0
        }
        
        position_manager.add_position("POS_001", position)
        assert len(position_manager.positions) == 1
        
        removed = position_manager.remove_position("POS_001")
        assert len(position_manager.positions) == 0
        assert removed == position
    
    def test_update_position(self, position_manager):
        """Test updating existing position."""
        position = {
            'symbol': 'SPY',
            'strategy': 'covered_call',
            'quantity': 100,
            'entry_price': 400.0,
            'entry_date': datetime.now(),
            'position_value': 40000.0
        }
        
        position_manager.add_position("POS_001", position)
        
        updates = {'current_price': 410.0, 'position_value': 41000.0}
        position_manager.update_position("POS_001", updates)
        
        updated_position = position_manager.get_position("POS_001")
        assert updated_position['current_price'] == 410.0
        assert updated_position['position_value'] == 41000.0
    
    def test_max_positions_limit(self, position_manager):
        """Test maximum positions limit."""
        # Add positions up to limit
        for i in range(10):
            position = {
                'symbol': f'STOCK_{i}',
                'strategy': 'test',
                'quantity': 100,
                'entry_price': 100.0,
                'position_value': 10000.0
            }
            position_manager.add_position(f"POS_{i:03d}", position)
        
        assert len(position_manager.positions) == 10
        
        # Try to add one more - should raise error
        with pytest.raises(ValueError, match="Maximum positions limit"):
            position = {
                'symbol': 'OVERFLOW',
                'strategy': 'test',
                'quantity': 100,
                'entry_price': 100.0,
                'position_value': 10000.0
            }
            position_manager.add_position("POS_OVERFLOW", position)
    
    def test_get_positions_by_strategy(self, position_manager):
        """Test filtering positions by strategy."""
        # Add positions with different strategies
        strategies = ['covered_call', 'iron_condor', 'covered_call']
        for i, strategy in enumerate(strategies):
            position = {
                'symbol': f'STOCK_{i}',
                'strategy': strategy,
                'quantity': 100,
                'entry_price': 100.0,
                'position_value': 10000.0
            }
            position_manager.add_position(f"POS_{i:03d}", position)
        
        covered_call_positions = position_manager.get_positions_by_strategy('covered_call')
        assert len(covered_call_positions) == 2
        
        iron_condor_positions = position_manager.get_positions_by_strategy('iron_condor')
        assert len(iron_condor_positions) == 1
    
    def test_calculate_total_value(self, position_manager):
        """Test total portfolio value calculation."""
        positions = [
            {'symbol': 'SPY', 'position_value': 10000.0},
            {'symbol': 'QQQ', 'position_value': 15000.0},
            {'symbol': 'IWM', 'position_value': 8000.0}
        ]
        
        for i, pos in enumerate(positions):
            position_manager.add_position(f"POS_{i:03d}", pos)
        
        total_value = position_manager.calculate_total_value()
        assert total_value == 33000.0


class TestPortfolio:
    """Test Portfolio class."""
    
    @pytest.fixture
    def portfolio_config(self):
        """Create portfolio configuration."""
        return PortfolioConfig(
            total_capital=100000,
            max_portfolio_risk=0.02,
            max_single_position=0.10
        )
    
    @pytest.fixture
    def portfolio(self, portfolio_config):
        """Create portfolio instance."""
        return Portfolio(
            portfolio_id="TEST_PORTFOLIO",
            config=portfolio_config
        )
    
    def test_portfolio_initialization(self, portfolio, portfolio_config):
        """Test Portfolio initialization."""
        assert portfolio.portfolio_id == "TEST_PORTFOLIO"
        assert portfolio.config == portfolio_config
        assert portfolio.cash == 100000  # Initial capital
        assert len(portfolio.positions) == 0
        assert len(portfolio.strategy_allocations) == 0
    
    def test_add_strategy_allocation(self, portfolio):
        """Test adding strategy allocation."""
        allocation = StrategyAllocation(
            strategy_id="covered_calls",
            strategy_name="Covered Call Strategy",
            target_allocation=0.30,
            risk_budget=0.008,
            expected_return=0.12,
            volatility=0.15
        )
        
        portfolio.add_strategy_allocation(allocation)
        
        assert len(portfolio.strategy_allocations) == 1
        assert "covered_calls" in portfolio.strategy_allocations
        assert portfolio.strategy_allocations["covered_calls"] == allocation
    
    def test_calculate_portfolio_metrics(self, portfolio):
        """Test portfolio metrics calculation."""
        # Add some positions
        positions = [
            {
                'symbol': 'SPY',
                'quantity': 100,
                'current_price': 400.0,
                'entry_price': 390.0,
                'position_value': 40000.0,
                'unrealized_pnl': 1000.0
            },
            {
                'symbol': 'QQQ',
                'quantity': 50,
                'current_price': 300.0,
                'entry_price': 295.0,
                'position_value': 15000.0,
                'unrealized_pnl': 250.0
            }
        ]
        
        for i, pos in enumerate(positions):
            portfolio.position_manager.add_position(f"POS_{i:03d}", pos)
        
        metrics = portfolio.calculate_portfolio_metrics()
        
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.total_value == 55000.0  # 40000 + 15000
        assert metrics.total_unrealized_pnl == 1250.0  # 1000 + 250
        assert metrics.cash_balance == 45000.0  # 100000 - 55000
    
    def test_rebalance_needed(self, portfolio):
        """Test rebalance detection."""
        # Add strategy allocation with drift
        allocation = StrategyAllocation(
            strategy_id="test_strategy",
            strategy_name="Test Strategy",
            target_allocation=0.30,
            current_allocation=0.35,  # 5% drift
            drift_threshold=0.03,  # 3% threshold
            risk_budget=0.01,
            expected_return=0.10,
            volatility=0.15
        )
        
        portfolio.add_strategy_allocation(allocation)
        
        assert portfolio.needs_rebalance()
    
    def test_risk_limit_validation(self, portfolio):
        """Test portfolio risk limit validation."""
        # Add position that would exceed single position limit
        large_position = {
            'symbol': 'SPY',
            'quantity': 300,
            'current_price': 400.0,
            'position_value': 120000.0  # 120% of portfolio - exceeds 10% limit
        }
        
        with pytest.raises(ValueError, match="exceeds maximum single position"):
            portfolio.validate_position_limits(large_position)


class TestPortfolioManager:
    """Test PortfolioManager main class."""
    
    @pytest.fixture
    def portfolio_config(self):
        """Create portfolio configuration."""
        return PortfolioConfig(
            total_capital=1000000,
            max_portfolio_risk=0.02,
            max_single_position=0.05,
            max_correlation=0.70
        )
    
    @pytest.fixture
    def portfolio_manager(self, portfolio_config):
        """Create portfolio manager instance."""
        return PortfolioManager(portfolio_config)
    
    def test_portfolio_manager_initialization(self, portfolio_manager, portfolio_config):
        """Test PortfolioManager initialization."""
        assert portfolio_manager.config == portfolio_config
        assert len(portfolio_manager.portfolios) == 0
        assert portfolio_manager.risk_manager is not None
    
    def test_create_portfolio(self, portfolio_manager):
        """Test creating new portfolio."""
        portfolio = portfolio_manager.create_portfolio("MAIN_PORTFOLIO")
        
        assert isinstance(portfolio, Portfolio)
        assert portfolio.portfolio_id == "MAIN_PORTFOLIO"
        assert len(portfolio_manager.portfolios) == 1
        assert "MAIN_PORTFOLIO" in portfolio_manager.portfolios
    
    def test_add_strategy_to_portfolio(self, portfolio_manager):
        """Test adding strategy to portfolio."""
        portfolio = portfolio_manager.create_portfolio("MAIN_PORTFOLIO")
        
        allocation = StrategyAllocation(
            strategy_id="covered_calls",
            strategy_name="Covered Call Strategy",
            target_allocation=0.25,
            risk_budget=0.005,
            expected_return=0.10,
            volatility=0.12
        )
        
        portfolio_manager.add_strategy_to_portfolio("MAIN_PORTFOLIO", allocation)
        
        assert len(portfolio.strategy_allocations) == 1
        assert "covered_calls" in portfolio.strategy_allocations
    
    @patch('rtradez.portfolio.portfolio_manager.KellyCriterion')
    def test_calculate_optimal_allocation(self, mock_kelly, portfolio_manager):
        """Test optimal allocation calculation."""
        portfolio = portfolio_manager.create_portfolio("MAIN_PORTFOLIO")
        
        # Add multiple strategies
        strategies = [
            StrategyAllocation(
                strategy_id="covered_calls",
                strategy_name="Covered Calls",
                target_allocation=0.30,
                risk_budget=0.006,
                expected_return=0.10,
                volatility=0.12
            ),
            StrategyAllocation(
                strategy_id="iron_condors",
                strategy_name="Iron Condors",
                target_allocation=0.25,
                risk_budget=0.005,
                expected_return=0.08,
                volatility=0.10
            )
        ]
        
        for allocation in strategies:
            portfolio_manager.add_strategy_to_portfolio("MAIN_PORTFOLIO", allocation)
        
        # Mock Kelly Criterion calculations
        mock_kelly_instance = Mock()
        mock_kelly.return_value = mock_kelly_instance
        mock_kelly_instance.calculate_optimal_fraction.return_value = 0.15
        
        optimal_allocations = portfolio_manager.calculate_optimal_allocation("MAIN_PORTFOLIO")
        
        assert isinstance(optimal_allocations, dict)
        assert len(optimal_allocations) == 2
        assert "covered_calls" in optimal_allocations
        assert "iron_condors" in optimal_allocations
    
    def test_rebalance_portfolio(self, portfolio_manager):
        """Test portfolio rebalancing."""
        portfolio = portfolio_manager.create_portfolio("MAIN_PORTFOLIO")
        
        # Add strategy with drift requiring rebalance
        allocation = StrategyAllocation(
            strategy_id="test_strategy",
            strategy_name="Test Strategy",
            target_allocation=0.30,
            current_allocation=0.40,  # 10% drift
            drift_threshold=0.05,  # 5% threshold
            risk_budget=0.01,
            expected_return=0.10,
            volatility=0.15
        )
        
        portfolio_manager.add_strategy_to_portfolio("MAIN_PORTFOLIO", allocation)
        
        # Mock rebalancing logic
        with patch.object(portfolio_manager, '_execute_rebalance') as mock_rebalance:
            rebalance_actions = portfolio_manager.rebalance_portfolio("MAIN_PORTFOLIO")
            
            assert isinstance(rebalance_actions, list)
            # Should detect need for rebalancing
            assert len(rebalance_actions) > 0 or mock_rebalance.called
    
    def test_calculate_portfolio_risk(self, portfolio_manager):
        """Test portfolio risk calculation."""
        portfolio = portfolio_manager.create_portfolio("MAIN_PORTFOLIO")
        
        # Add positions with known correlations
        positions_data = [
            {
                'symbol': 'SPY',
                'strategy': 'covered_calls',
                'position_value': 50000.0,
                'volatility': 0.20,
                'expected_return': 0.10
            },
            {
                'symbol': 'QQQ',
                'strategy': 'iron_condors',
                'position_value': 30000.0,
                'volatility': 0.25,
                'expected_return': 0.12
            }
        ]
        
        for i, pos in enumerate(positions_data):
            portfolio.position_manager.add_position(f"POS_{i:03d}", pos)
        
        # Mock correlation matrix
        correlation_matrix = pd.DataFrame({
            'SPY': [1.0, 0.7],
            'QQQ': [0.7, 1.0]
        }, index=['SPY', 'QQQ'])
        
        with patch.object(portfolio_manager, '_get_correlation_matrix', return_value=correlation_matrix):
            portfolio_risk = portfolio_manager.calculate_portfolio_risk("MAIN_PORTFOLIO")
            
            assert isinstance(portfolio_risk, float)
            assert portfolio_risk > 0
    
    def test_risk_limit_enforcement(self, portfolio_manager):
        """Test risk limit enforcement."""
        portfolio = portfolio_manager.create_portfolio("MAIN_PORTFOLIO")
        
        # Try to add position that exceeds risk limits
        high_risk_position = {
            'symbol': 'RISKY_STOCK',
            'strategy': 'high_risk',
            'position_value': 60000.0,  # 6% of portfolio, exceeds 5% limit
            'volatility': 0.50,
            'expected_return': 0.20
        }
        
        with pytest.raises(ValueError, match="risk limit"):
            portfolio_manager.validate_new_position("MAIN_PORTFOLIO", high_risk_position)
    
    def test_generate_allocation_report(self, portfolio_manager):
        """Test allocation report generation."""
        portfolio = portfolio_manager.create_portfolio("MAIN_PORTFOLIO")
        
        # Add strategies and positions
        allocation = StrategyAllocation(
            strategy_id="covered_calls",
            strategy_name="Covered Call Strategy",
            target_allocation=0.30,
            current_allocation=0.28,
            risk_budget=0.006,
            expected_return=0.10,
            volatility=0.12
        )
        
        portfolio_manager.add_strategy_to_portfolio("MAIN_PORTFOLIO", allocation)
        
        report = portfolio_manager.generate_allocation_report("MAIN_PORTFOLIO")
        
        assert isinstance(report, str)
        assert "Portfolio Allocation Report" in report
        assert "covered_calls" in report
        assert "28.0%" in report  # Current allocation


@pytest.mark.integration
class TestPortfolioIntegration:
    """Integration tests for portfolio management."""
    
    def test_full_portfolio_lifecycle(self):
        """Test complete portfolio management lifecycle."""
        # Create portfolio manager
        config = PortfolioConfig(
            total_capital=500000,
            max_portfolio_risk=0.015,
            max_single_position=0.08
        )
        manager = PortfolioManager(config)
        
        # Create portfolio
        portfolio = manager.create_portfolio("INTEGRATION_TEST")
        
        # Add strategies
        strategies = [
            StrategyAllocation(
                strategy_id="covered_calls",
                strategy_name="Covered Call Strategy",
                target_allocation=0.40,
                risk_budget=0.008,
                expected_return=0.10,
                volatility=0.15
            ),
            StrategyAllocation(
                strategy_id="cash_secured_puts",
                strategy_name="Cash Secured Put Strategy",
                target_allocation=0.35,
                risk_budget=0.007,
                expected_return=0.09,
                volatility=0.14
            ),
            StrategyAllocation(
                strategy_id="iron_condors",
                strategy_name="Iron Condor Strategy",
                target_allocation=0.25,
                risk_budget=0.005,
                expected_return=0.08,
                volatility=0.12
            )
        ]
        
        for strategy in strategies:
            manager.add_strategy_to_portfolio("INTEGRATION_TEST", strategy)
        
        # Verify total allocation
        total_allocation = sum(s.target_allocation for s in strategies)
        assert abs(total_allocation - 1.0) < 0.01  # Should sum to ~100%
        
        # Add positions
        positions = [
            {
                'symbol': 'SPY',
                'strategy': 'covered_calls',
                'quantity': 500,
                'current_price': 400.0,
                'entry_price': 395.0,
                'position_value': 200000.0
            },
            {
                'symbol': 'QQQ',
                'strategy': 'cash_secured_puts',
                'quantity': 600,
                'current_price': 250.0,
                'entry_price': 245.0,
                'position_value': 150000.0
            }
        ]
        
        for i, pos in enumerate(positions):
            portfolio.position_manager.add_position(f"POS_{i:03d}", pos)
        
        # Calculate metrics
        metrics = portfolio.calculate_portfolio_metrics()
        assert metrics.total_value == 350000.0
        assert metrics.cash_balance == 150000.0  # 500000 - 350000
        
        # Test risk calculation
        with patch.object(manager, '_get_correlation_matrix') as mock_corr:
            mock_corr.return_value = pd.DataFrame({
                'SPY': [1.0, 0.6],
                'QQQ': [0.6, 1.0]
            }, index=['SPY', 'QQQ'])
            
            portfolio_risk = manager.calculate_portfolio_risk("INTEGRATION_TEST")
            assert isinstance(portfolio_risk, float)
            assert portfolio_risk > 0
        
        # Generate report
        report = manager.generate_allocation_report("INTEGRATION_TEST")
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Test rebalancing detection
        # Simulate drift by updating allocations
        portfolio.strategy_allocations["covered_calls"].current_allocation = 0.45  # 5% drift
        
        assert portfolio.needs_rebalance()
    
    def test_risk_management_integration(self):
        """Test integration with risk management systems."""
        config = PortfolioConfig(
            total_capital=100000,
            max_portfolio_risk=0.01,  # Very conservative
            max_single_position=0.05
        )
        manager = PortfolioManager(config)
        portfolio = manager.create_portfolio("RISK_TEST")
        
        # Try to add position that exceeds risk limits
        risky_position = {
            'symbol': 'VOLATILE_STOCK',
            'strategy': 'high_vol',
            'quantity': 200,
            'current_price': 300.0,
            'position_value': 60000.0,  # 60% of portfolio
            'volatility': 0.80,
            'expected_return': 0.25
        }
        
        # Should be rejected due to risk limits
        with pytest.raises(ValueError):
            manager.validate_new_position("RISK_TEST", risky_position)
        
        # Add acceptable position
        safe_position = {
            'symbol': 'SAFE_STOCK',
            'strategy': 'low_vol',
            'quantity': 100,
            'current_price': 40.0,
            'position_value': 4000.0,  # 4% of portfolio
            'volatility': 0.10,
            'expected_return': 0.08
        }
        
        # Should be accepted
        try:
            manager.validate_new_position("RISK_TEST", safe_position)
            portfolio.position_manager.add_position("SAFE_POS", safe_position)
            assert len(portfolio.positions) == 1
        except ValueError:
            pytest.fail("Safe position should have been accepted")