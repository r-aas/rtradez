#!/usr/bin/env python3
"""
RTradez Quick Start Example

Demonstrates the core functionality using proven open-source libraries:
- Data loading with yfinance
- Strategy backtesting with VectorBT
- Performance analysis with QuantStats
- Options calculations with Mibian
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# RTradez imports
from rtradez.datasets import OptionsDataset
from rtradez.methods import OptionsStrategy
from rtradez.metrics import PerformanceAnalyzer

def main():
    """Run quick start example."""
    
    print("🚀 RTradez Quick Start Example")
    print("=" * 50)
    
    # 1. Load options dataset using yfinance
    print("\n📊 Loading SPY options data...")
    try:
        dataset = OptionsDataset.from_source('yahoo', 'SPY', period='1y')
        print(f"✅ Loaded: {dataset}")
        print(f"📈 Summary: {dataset.summary()}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # 2. Create and backtest Iron Condor strategy using VectorBT
    print("\n🎯 Creating Iron Condor strategy...")
    strategy = OptionsStrategy.create(
        'iron_condor',
        dte_entry=30,
        dte_exit=7,
        profit_target=0.5,
        stop_loss=2.0
    )
    print(f"✅ Created: {strategy}")
    
    # 3. Run backtest
    print("\n⚡ Running backtest...")
    try:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        results = strategy.backtest(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000
        )
        
        print(f"✅ Backtest completed!")
        
        # 4. Analyze performance using QuantStats
        print("\n📊 Performance Analysis...")
        
        # Get returns from VectorBT results
        returns = results.returns()
        
        analyzer = PerformanceAnalyzer(returns)
        metrics = analyzer.get_metrics()
        
        print(f"📈 Total Return: {metrics['total_return']:.2%}")
        print(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")
        print(f"💰 Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Options-specific metrics
        options_metrics = metrics['options_specific']
        print(f"⏰ Theta Efficiency: {options_metrics['theta_efficiency']:.2%}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        
        # Demo with synthetic data instead
        print("\n🔄 Using synthetic data for demo...")
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        synthetic_returns = pd.Series(
            np.random.normal(0.0005, 0.01, len(dates)),  # Slight positive bias
            index=dates
        )
        
        analyzer = PerformanceAnalyzer(synthetic_returns)
        metrics = analyzer.get_metrics()
        
        print(f"📈 Total Return: {metrics['total_return']:.2%}")
        print(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # 5. Demonstrate available strategies
    print("\n🎲 Available Strategies:")
    strategies = OptionsStrategy.list_strategies()
    for i, strat in enumerate(strategies, 1):
        config = OptionsStrategy.STRATEGY_REGISTRY[strat]
        print(f"{i}. {config['name']} ({config['risk_profile']} risk)")
        print(f"   📝 {config['description']}")
    
    print("\n🎉 RTradez framework ready for options trading analysis!")
    print("📚 Next steps:")
    print("   • Explore different strategies")
    print("   • Add real options pricing data")
    print("   • Implement custom metrics")
    print("   • Build automated trading systems")


if __name__ == "__main__":
    main()