#!/usr/bin/env python3
"""
Comprehensive Risk Management System Demo for RTradez.

Demonstrates the full risk management capabilities including position sizing,
risk limits, portfolio risk calculation, margin requirements, and real-time monitoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rtradez.risk import (
    PositionSizer, KellyCriterion, FixedFractionSizer, SizingMethod,
    RiskLimitManager, PortfolioRiskLimits, PositionRiskLimits,
    PortfolioRiskCalculator, VaRCalculator, VaRMethod,
    OptionsMarginCalculator, OptionPosition, OptionType, StrategyType,
    RealTimeRiskMonitor, AlertLevel, create_basic_risk_monitor
)

def demo_comprehensive_risk_management():
    """Demonstrate comprehensive risk management system."""
    print("🛡️  RTradez Comprehensive Risk Management Demo")
    print("=" * 60)
    
    # Demo parameters
    total_capital = 1000000  # $1M portfolio
    
    print(f"\n💰 Portfolio Capital: ${total_capital:,.0f}")
    print("📊 Demonstrating all risk management components:\n")
    
    # ==========================================
    # 1. POSITION SIZING DEMO
    # ==========================================
    print("1️⃣  POSITION SIZING ANALYSIS")
    print("-" * 40)
    
    # Create different position sizers
    kelly_sizer = KellyCriterion(total_capital, max_kelly_fraction=0.15)
    fixed_sizer = FixedFractionSizer(total_capital, base_fraction=0.08)
    
    # Test strategies with different risk/return profiles
    strategies = [
        {"name": "Iron Condor", "return": 0.08, "volatility": 0.12, "win_rate": 0.75, "avg_win": 0.05, "avg_loss": 0.15},
        {"name": "Strangle", "return": 0.15, "volatility": 0.25, "win_rate": 0.60, "avg_win": 0.12, "avg_loss": 0.20},
        {"name": "Calendar Spread", "return": 0.06, "volatility": 0.08, "win_rate": 0.80, "avg_win": 0.03, "avg_loss": 0.10}
    ]
    
    print("Strategy Position Sizing Results:")
    for strategy in strategies:
        kelly_result = kelly_sizer.calculate_position_size(
            strategy["name"], strategy["return"], strategy["volatility"],
            win_rate=strategy["win_rate"], avg_win=strategy["avg_win"], avg_loss=strategy["avg_loss"]
        )
        
        fixed_result = fixed_sizer.calculate_position_size(
            strategy["name"], strategy["return"], strategy["volatility"]
        )
        
        print(f"\n{strategy['name']}:")
        print(f"  Kelly Size: ${kelly_result.recommended_size:,.0f} (conf: {kelly_result.confidence_level:.2f})")
        print(f"  Fixed Size: ${fixed_result.recommended_size:,.0f} (conf: {fixed_result.confidence_level:.2f})")
        print(f"  Expected Return: {strategy['return']:.1%}, Volatility: {strategy['volatility']:.1%}")
    
    # ==========================================
    # 2. RISK LIMITS MANAGEMENT
    # ==========================================
    print(f"\n\n2️⃣  RISK LIMITS MANAGEMENT")
    print("-" * 40)
    
    # Create risk limits
    portfolio_limits = PortfolioRiskLimits(
        max_total_exposure=1.2,
        max_position_concentration=0.15,
        max_daily_var=0.03,
        max_drawdown=0.12,
        max_leverage=1.8
    )
    
    position_limits = PositionRiskLimits(
        max_position_size=0.08,
        max_delta_exposure=0.04,
        min_time_to_expiry=10
    )
    
    risk_manager = RiskLimitManager(portfolio_limits, position_limits, total_capital)
    
    # Simulate portfolio positions
    portfolio_positions = {
        "IronCondor_SPY": {
            "market_value": 120000,
            "delta": 0.05,
            "gamma": 0.02,
            "theta": -50,
            "days_to_expiry": 25
        },
        "Strangle_QQQ": {
            "market_value": 180000,
            "delta": -0.08,
            "gamma": 0.03,
            "theta": -75,
            "days_to_expiry": 15
        },
        "CalendarSpread_IWM": {
            "market_value": 95000,
            "delta": 0.02,
            "gamma": 0.01,
            "theta": -25,
            "days_to_expiry": 35
        }
    }
    
    # Simulate portfolio P&L history with drawdown
    dates = pd.bdate_range(start='2024-01-01', end='2024-10-31')
    np.random.seed(42)
    daily_returns = np.random.normal(0.0008, 0.015, len(dates))
    # Simulate a drawdown period
    daily_returns[50:70] = np.random.normal(-0.003, 0.025, 20)  # Bad period
    portfolio_pnl = pd.Series(daily_returns, index=dates).cumsum() * total_capital
    
    # Check portfolio limits
    portfolio_breaches = risk_manager.check_portfolio_limits(portfolio_positions, portfolio_pnl)
    
    print("Portfolio Risk Analysis:")
    print(f"  Total Exposure: ${sum(abs(pos['market_value']) for pos in portfolio_positions.values()):,.0f}")
    print(f"  Largest Position: {max(portfolio_positions.keys(), key=lambda x: abs(portfolio_positions[x]['market_value']))}")
    print(f"  Current Drawdown: {abs(portfolio_pnl.min() - portfolio_pnl.max()) / total_capital:.2%}")
    
    if portfolio_breaches:
        print(f"\n⚠️  Found {len(portfolio_breaches)} portfolio risk breaches:")
        for breach in portfolio_breaches:
            print(f"    - {breach.limit_type.value}: {breach.current_value:.3f} > {breach.threshold:.3f}")
            print(f"      Action: {breach.recommended_action}")
    else:
        print("✅ No portfolio risk limit breaches")
    
    # Check individual position limits
    position_breaches = []
    for strategy_name, position_data in portfolio_positions.items():
        breaches = risk_manager.check_position_limits(strategy_name, position_data)
        position_breaches.extend(breaches)
    
    if position_breaches:
        print(f"\n⚠️  Found {len(position_breaches)} position risk breaches:")
        for breach in position_breaches:
            print(f"    - {breach.affected_strategies[0]}: {breach.limit_type.value}")
    else:
        print("✅ No position risk limit breaches")
    
    # ==========================================
    # 3. PORTFOLIO RISK CALCULATION  
    # ==========================================
    print(f"\n\n3️⃣  PORTFOLIO RISK CALCULATION")
    print("-" * 40)
    
    # Create strategy returns data
    np.random.seed(42)
    strategy_returns = pd.DataFrame({
        'IronCondor_SPY': np.random.normal(0.0008, 0.012, len(dates)),
        'Strangle_QQQ': np.random.normal(0.0012, 0.025, len(dates)),
        'CalendarSpread_IWM': np.random.normal(0.0006, 0.008, len(dates))
    }, index=dates)
    
    # Add some correlation
    strategy_returns['Strangle_QQQ'] += 0.3 * strategy_returns['IronCondor_SPY']
    strategy_returns['CalendarSpread_IWM'] += 0.1 * strategy_returns['IronCondor_SPY']
    
    strategy_weights = {
        'IronCondor_SPY': 0.4,
        'Strangle_QQQ': 0.45,
        'CalendarSpread_IWM': 0.15
    }
    
    # Calculate comprehensive portfolio risk
    portfolio_risk_calc = PortfolioRiskCalculator()
    risk_analysis = portfolio_risk_calc.calculate_portfolio_risk(
        strategy_returns, strategy_weights, total_capital
    )
    
    var_result = risk_analysis['var_result']
    correlation_analysis = risk_analysis['correlation_analysis']
    
    print("Portfolio Risk Metrics:")
    print(f"  1-Day VaR (95%): ${var_result.var_1d:,.0f}")
    print(f"  5-Day VaR (95%): ${var_result.var_5d:,.0f}")
    print(f"  Expected Shortfall: ${var_result.expected_shortfall:,.0f}")
    print(f"  Portfolio Volatility: {var_result.portfolio_volatility:.2%}")
    print(f"  Diversification Ratio: {var_result.diversification_ratio:.2f}")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Effective Strategies: {correlation_analysis.effective_strategies:.1f}")
    print(f"  Max Correlation: {correlation_analysis.max_correlation:.3f}")
    print(f"  Average Correlation: {correlation_analysis.avg_correlation:.3f}")
    print(f"  Concentration Ratio: {correlation_analysis.concentration_ratio:.3f}")
    
    print(f"\nRisk Contribution by Strategy:")
    for strategy, contribution in risk_analysis['component_var'].items():
        print(f"  {strategy}: ${contribution:,.0f} ({contribution/var_result.var_1d:.1%})")
    
    # ==========================================
    # 4. MARGIN CALCULATION
    # ==========================================
    print(f"\n\n4️⃣  OPTIONS MARGIN CALCULATION")
    print("-" * 40)
    
    margin_calc = OptionsMarginCalculator(account_type="margin")
    
    # Iron Condor example
    iron_condor_positions = [
        OptionPosition(OptionType.PUT, 480, 25, -1, 2.50, 500, 0.20),    # Short put
        OptionPosition(OptionType.PUT, 470, 25, 1, 1.20, 500, 0.20),     # Long put  
        OptionPosition(OptionType.CALL, 520, 25, -1, 2.80, 500, 0.20),   # Short call
        OptionPosition(OptionType.CALL, 530, 25, 1, 1.50, 500, 0.20),    # Long call
    ]
    
    iron_condor_margin = margin_calc.calculate_strategy_margin(
        iron_condor_positions, StrategyType.IRON_CONDOR, 500
    )
    
    # Strangle example  
    strangle_positions = [
        OptionPosition(OptionType.PUT, 480, 30, -1, 3.20, 500, 0.22),
        OptionPosition(OptionType.CALL, 520, 30, -1, 3.50, 500, 0.22),
    ]
    
    strangle_margin = margin_calc.calculate_strategy_margin(
        strangle_positions, StrategyType.STRANGLE, 500
    )
    
    print("Margin Requirements:")
    print(f"\nIron Condor (SPY 500):")
    print(f"  Initial Margin: ${iron_condor_margin.initial_margin:,.0f}")
    print(f"  Buying Power Reduction: ${iron_condor_margin.buying_power_reduction:,.0f}")
    print(f"  Max Loss: ${iron_condor_margin.max_loss:,.0f}")
    print(f"  Max Profit: ${iron_condor_margin.max_profit:,.0f}")
    print(f"  Breakevens: {[f'${be:.0f}' for be in iron_condor_margin.breakeven_points]}")
    
    print(f"\nShort Strangle (SPY 500):")
    print(f"  Initial Margin: ${strangle_margin.initial_margin:,.0f}")
    print(f"  Buying Power Reduction: ${strangle_margin.buying_power_reduction:,.0f}")
    print(f"  Max Profit: ${strangle_margin.max_profit:,.0f}")
    print(f"  Breakevens: {[f'${be:.0f}' for be in strangle_margin.breakeven_points]}")
    
    # Portfolio margin summary
    portfolio_margin = margin_calc.calculate_portfolio_margin([iron_condor_margin, strangle_margin])
    print(f"\nPortfolio Margin Summary:")
    print(f"  Total Initial Margin: ${portfolio_margin['total_initial_margin']:,.0f}")
    print(f"  Total Buying Power Used: ${portfolio_margin['total_buying_power_reduction']:,.0f}")
    print(f"  Margin Utilization: {portfolio_margin['total_buying_power_reduction']/total_capital:.1%}")
    
    # ==========================================
    # 5. REAL-TIME RISK MONITORING
    # ==========================================
    print(f"\n\n5️⃣  REAL-TIME RISK MONITORING")
    print("-" * 40)
    
    # Create and configure risk monitor
    risk_monitor = create_basic_risk_monitor(total_capital)
    
    # Add custom alert callback
    def alert_handler(alert):
        print(f"🚨 ALERT: {alert.level.value.upper()} - {alert.message}")
        print(f"   Action: {alert.recommended_action}")
    
    risk_monitor.add_alert_callback(alert_handler)
    
    # Simulate real-time portfolio updates
    print("Simulating real-time monitoring (5 seconds)...")
    risk_monitor.start_monitoring()
    
    for i in range(5):
        # Simulate portfolio data with potential issues
        portfolio_data = {
            'total_pnl': total_capital * (0.98 if i > 2 else 1.0),  # Simulate loss
            'positions': {
                'IronCondor_SPY': {
                    'market_value': 120000 * (1.2 if i > 3 else 1.0)  # Simulate concentration issue
                },
                'Strangle_QQQ': {
                    'market_value': 180000
                }
            }
        }
        
        risk_monitor.update_portfolio_data(portfolio_data)
        time.sleep(1)
    
    # Get final risk dashboard
    dashboard = risk_monitor.get_risk_dashboard()
    print(f"\nRisk Dashboard Summary:")
    print(f"  Monitoring Status: {'🟢 Active' if dashboard['monitoring_status'] else '🔴 Inactive'}")
    print(f"  Active Alerts: {dashboard['active_alerts']['total']}")
    print(f"  Critical Alerts: {dashboard['active_alerts']['critical']}")
    print(f"  Recent Alerts (24h): {dashboard['recent_alerts_24h']}")
    
    risk_monitor.stop_monitoring()
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n\n🎯 RISK MANAGEMENT SUMMARY")
    print("=" * 60)
    print("✅ Position Sizing: Kelly Criterion and Fixed Fraction implemented")
    print("✅ Risk Limits: Portfolio and position-level monitoring active")
    print("✅ Portfolio Risk: VaR, correlation analysis, and diversification metrics")
    print("✅ Margin Calculation: Options strategy margin requirements calculated")
    print("✅ Real-time Monitoring: Alert system with configurable thresholds")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Portfolio VaR (1-day): ${var_result.var_1d:,.0f} ({var_result.var_1d/total_capital:.2%} of capital)")
    print(f"   • Diversification Ratio: {var_result.diversification_ratio:.2f} (>1 = good diversification)")
    print(f"   • Effective Strategies: {correlation_analysis.effective_strategies:.1f} (vs {len(strategy_weights)} actual)")
    print(f"   • Margin Utilization: {portfolio_margin['total_buying_power_reduction']/total_capital:.1%}")
    
    print(f"\n🔧 Next Steps for Production:")
    print("   1. Integrate with live data feeds")
    print("   2. Connect to broker APIs for real margin data")
    print("   3. Implement automatic position scaling")
    print("   4. Add email/SMS alert notifications")
    print("   5. Create risk management dashboard")

if __name__ == "__main__":
    demo_comprehensive_risk_management()