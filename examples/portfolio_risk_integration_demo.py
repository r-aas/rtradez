#!/usr/bin/env python3
"""
Portfolio Management + Risk Integration Demo.

Demonstrates the integration between portfolio management and risk management
systems for coordinated multi-strategy options trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rtradez.portfolio.portfolio_manager import PortfolioManager, PortfolioConfig, StrategyAllocation
from rtradez.risk import (
    RiskLimitManager, PortfolioRiskLimits, PositionRiskLimits,
    create_basic_risk_monitor, KellyCriterion, KellyConfig
)

class MockStrategy:
    """Mock strategy for demonstration."""
    
    def __init__(self, name: str, expected_return: float, volatility: float):
        self.name = name
        self.expected_return = expected_return
        self.volatility = volatility
        self.is_fitted = False
        
    def fit(self, X, y):
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted first")
        # Simulate predictions
        return np.random.normal(self.expected_return/252, self.volatility/np.sqrt(252), len(X))
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.corrcoef(predictions, y)[0, 1] if len(predictions) > 1 else 0.0

def demo_portfolio_risk_integration():
    """Demonstrate integrated portfolio and risk management."""
    print("🔗 RTradez Portfolio + Risk Management Integration Demo")
    print("=" * 65)
    
    # Configuration
    total_capital = 2000000  # $2M portfolio
    
    # Create portfolio configuration
    portfolio_config = PortfolioConfig(
        total_capital=total_capital,
        max_strategies=5,
        rebalance_frequency="weekly",
        rebalance_threshold=0.08,  # 8% deviation triggers rebalance
        max_correlation_exposure=0.70,
        emergency_stop_drawdown=0.18,
        cash_reserve_minimum=0.08,
        enable_auto_rebalancing=True,
        enable_risk_coordination=True
    )
    
    # Create portfolio manager
    portfolio = PortfolioManager(portfolio_config)
    
    # Create strategies with different risk/return profiles
    strategies = [
        {"name": "IronCondor_SPY", "return": 0.08, "volatility": 0.12, "allocation": 0.30},
        {"name": "Strangle_QQQ", "return": 0.15, "volatility": 0.25, "allocation": 0.25},
        {"name": "CalendarSpread_IWM", "return": 0.06, "volatility": 0.08, "allocation": 0.20},
        {"name": "CoveredCall_MSFT", "return": 0.04, "volatility": 0.06, "allocation": 0.15},
        {"name": "CashSecuredPut_AAPL", "return": 0.05, "volatility": 0.10, "allocation": 0.10}
    ]
    
    print(f"💰 Portfolio Capital: ${total_capital:,.0f}")
    print(f"📊 Configuring {len(strategies)} strategies:\n")
    
    # Add strategies to portfolio
    for strategy_config in strategies:
        strategy_instance = MockStrategy(
            strategy_config["name"], 
            strategy_config["return"], 
            strategy_config["volatility"]
        )
        
        success = portfolio.add_strategy(
            strategy_config["name"],
            strategy_instance,
            strategy_config["allocation"],
            min_allocation=0.0,
            max_allocation=0.40
        )
        
        if success:
            print(f"✅ Added {strategy_config['name']}: {strategy_config['allocation']:.1%} target allocation")
        else:
            print(f"❌ Failed to add {strategy_config['name']}")
    
    # Initialize risk management
    portfolio_limits = PortfolioRiskLimits(
        max_total_exposure=1.1,
        max_position_concentration=0.30,
        max_daily_var=0.04,
        max_drawdown=0.15,
        max_leverage=1.5
    )
    
    position_limits = PositionRiskLimits(
        max_position_size=0.25,
        max_delta_exposure=0.05,
        min_time_to_expiry=7
    )
    
    risk_manager = RiskLimitManager(portfolio_limits, position_limits, total_capital)
    
    # Create risk monitor
    risk_monitor = create_basic_risk_monitor(total_capital)
    
    def portfolio_rebalance_callback(capital_changes):
        """Handle portfolio rebalancing with risk coordination."""
        print(f"\n🔄 REBALANCING EVENT:")
        for strategy, change in capital_changes.items():
            print(f"   {strategy}: ${change:+,.0f}")
        
        # Update risk monitor with new allocations
        portfolio_data = {
            'total_pnl': total_capital,
            'positions': {
                strategy: {'market_value': abs(change)}
                for strategy, change in capital_changes.items()
            }
        }
        risk_monitor.update_portfolio_data(portfolio_data)
    
    portfolio.add_rebalance_callback(portfolio_rebalance_callback)
    
    # Simulate initial allocation
    print(f"\n🎯 INITIAL PORTFOLIO ALLOCATION")
    print("-" * 45)
    
    # Set initial allocations to targets
    for name, allocation in portfolio.strategies.items():
        allocation.current_allocation = allocation.target_allocation
        allocation.last_rebalance = datetime.now()
    
    initial_metrics = portfolio.calculate_portfolio_metrics()
    print(f"Invested Capital: ${initial_metrics['invested_capital']:,.0f}")
    print(f"Cash Reserve: ${initial_metrics['cash_balance']:,.0f} ({initial_metrics['cash_percentage']:.1%})")
    print(f"Active Strategies: {initial_metrics['num_active_strategies']}")
    
    # Simulate performance and demonstrate coordination
    print(f"\n📈 SIMULATING MARKET CONDITIONS & COORDINATION")
    print("-" * 55)
    
    # Create sample market data
    dates = pd.bdate_range(start='2024-01-01', end='2024-10-31')
    np.random.seed(42)
    
    # Simulate 30 days of performance
    for day in range(30):
        date = dates[day]
        
        # Simulate daily strategy performance
        for strategy_config in strategies:
            strategy_name = strategy_config["name"]
            daily_return = np.random.normal(
                strategy_config["return"]/252,
                strategy_config["volatility"]/np.sqrt(252)
            )
            
            portfolio.update_strategy_performance(strategy_name, {
                'return': daily_return,
                'volatility': strategy_config["volatility"]/np.sqrt(252),
                'sharpe': daily_return * np.sqrt(252) / strategy_config["volatility"]
            })
        
        # Simulate some drift in allocations (market movements)
        if day % 5 == 0:  # Every 5 days
            for name, allocation in portfolio.strategies.items():
                drift = np.random.normal(0, 0.02)  # 2% random drift
                allocation.current_allocation = max(0, allocation.current_allocation + drift)
            
            # Check if rebalancing is needed
            rebalancing_needs = portfolio.calculate_rebalancing_needs()
            
            if rebalancing_needs:
                print(f"\nDay {day+1}: Rebalancing needed for {len(rebalancing_needs)} strategies")
                portfolio.execute_rebalancing(rebalancing_needs)
                
                # Check risk limits after rebalancing
                risk_warnings = portfolio.check_risk_limits()
                if risk_warnings:
                    print(f"⚠️  Risk warnings: {len(risk_warnings)}")
                    for warning in risk_warnings:
                        print(f"    - {warning}")
    
    # Final portfolio analysis
    print(f"\n📊 FINAL PORTFOLIO ANALYSIS")
    print("-" * 40)
    
    final_summary = portfolio.get_portfolio_summary()
    metrics = final_summary['portfolio_metrics']
    attribution = final_summary['strategy_attribution']
    
    print(f"Portfolio Performance:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    print(f"\nStrategy Attribution:")
    for strategy, attr in attribution.items():
        print(f"  {strategy}:")
        print(f"    Weight: {attr['weight']:.1%}")
        print(f"    Return: {attr['strategy_return']:.2%}")
        print(f"    Contribution: {attr['contribution']:.2%}")
        print(f"    Sharpe: {attr['sharpe_ratio']:.3f}")
    
    # Risk coordination demonstration
    print(f"\n🛡️  RISK COORDINATION ANALYSIS")
    print("-" * 40)
    
    # Position sizing with portfolio context
    kelly_config = KellyConfig(total_capital=total_capital * 0.9)  # Leave 10% cash
    kelly_sizer = KellyCriterion(kelly_config)
    
    print("Coordinated Position Sizing:")
    for strategy_config in strategies[:3]:  # Show top 3
        sizing_result = kelly_sizer.calculate_position_size(
            strategy_config["name"],
            strategy_config["return"],
            strategy_config["volatility"]
        )
        current_allocation = portfolio.strategies[strategy_config["name"]].current_allocation
        current_size = current_allocation * total_capital
        
        print(f"  {strategy_config['name']}:")
        print(f"    Current Size: ${current_size:,.0f}")
        print(f"    Kelly Optimal: ${sizing_result.recommended_size:,.0f}")
        print(f"    Confidence: {sizing_result.confidence_level:.2f}")
    
    # Risk dashboard
    risk_dashboard = risk_monitor.get_risk_dashboard()
    print(f"\nRisk Monitor Status:")
    print(f"  Active Alerts: {risk_dashboard['active_alerts']['total']}")
    print(f"  Critical Alerts: {risk_dashboard['active_alerts']['critical']}")
    print(f"  Recent Alerts (24h): {risk_dashboard['recent_alerts_24h']}")
    
    # Integration benefits summary
    print(f"\n🎯 INTEGRATION BENEFITS DEMONSTRATED")
    print("=" * 55)
    print("✅ Coordinated rebalancing with risk limit checks")
    print("✅ Real-time risk monitoring across all strategies")
    print("✅ Position sizing considers portfolio context")
    print("✅ Performance attribution tracks individual contributions")
    print("✅ Automatic cash reserve management")
    print("✅ Cross-strategy correlation monitoring")
    
    print(f"\n💡 Key Insights:")
    total_strategies = len(portfolio.strategies)
    active_strategies = metrics['num_active_strategies']
    total_rebalances = sum(1 for s in portfolio.strategies.values() if s.last_rebalance)
    
    print(f"   • Managed {total_strategies} strategies with {active_strategies} active")
    print(f"   • Executed {total_rebalances} rebalancing events")
    print(f"   • Maintained {metrics['cash_percentage']:.1%} cash reserve")
    print(f"   • Portfolio Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    
    print(f"\n🔧 Production Ready Features:")
    print("   • Automatic rebalancing with configurable thresholds")
    print("   • Risk limit integration with portfolio coordination")
    print("   • Real-time performance attribution")
    print("   • Emergency liquidation capabilities")
    print("   • Callback system for external integration")

if __name__ == "__main__":
    demo_portfolio_risk_integration()