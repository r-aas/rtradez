"""Advanced research capabilities demonstration for RTradez."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# RTradez imports
from rtradez.research import (
    MarketRegimeDetector, RegimeBasedStrategy,
    AdvancedBacktester, TransactionCostModel,
    GreeksAnalyzer, DeltaHedger, GreeksProfile,
    ResearchVisualizer, InteractivePlotter
)
from rtradez.methods.strategies import OptionsStrategy
from rtradez.datasets.benchmark_datasets import BenchmarkDatasets


def generate_comprehensive_market_data(symbol: str = 'SPY', periods: int = 504) -> pd.DataFrame:
    """Generate realistic market data for demonstration."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2022-01-01', periods=periods, freq='D')
    dates = dates[dates.weekday < 5]  # Business days only
    
    # Generate realistic price series with regime changes
    base_vol = 0.20
    regime_changes = [0, 168, 336]  # Regime changes at different periods
    regimes = [0.15, 0.35, 0.25]   # Different volatility regimes
    
    returns = []
    current_regime = 0
    
    for i, date in enumerate(dates):
        # Check for regime change
        if i in regime_changes[1:]:
            current_regime += 1
            current_regime = min(current_regime, len(regimes) - 1)
        
        vol = regimes[current_regime]
        
        # Add market trends
        if current_regime == 0:  # Bull market
            drift = 0.0008
        elif current_regime == 1:  # Bear market
            drift = -0.0005
        else:  # Sideways market
            drift = 0.0001
        
        daily_return = np.random.normal(drift, vol / np.sqrt(252))
        returns.append(daily_return)
    
    # Convert to prices
    prices = 400 * np.cumprod(1 + np.array(returns))
    
    # Generate volume with realistic patterns
    volumes = np.random.lognormal(15, 0.5, len(dates))
    
    # Create OHLC data
    data = pd.DataFrame(index=dates[:len(prices)])
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.002, len(data)))
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, len(data))))
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, len(data))))
    data['Volume'] = volumes[:len(data)]
    
    # Fill NaN values
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    return data


def demonstrate_regime_detection():
    """Demonstrate market regime detection capabilities."""
    print("🔍 MARKET REGIME DETECTION DEMO")
    print("=" * 50)
    
    # Generate market data
    market_data = generate_comprehensive_market_data('SPY', 504)
    
    # Test different regime detection methods
    methods = ['volatility', 'trend', 'combined']
    
    for method in methods:
        print(f"\n📊 Testing {method.upper()} regime detection:")
        
        # Initialize and fit regime detector
        detector = MarketRegimeDetector(method=method, n_regimes=3)
        detector.fit(market_data)
        
        # Transform data to include regimes
        data_with_regimes = detector.transform(market_data)
        
        # Get regime statistics
        regime_stats = detector.get_regime_statistics(market_data)
        print(f"   Regime Statistics:")
        for _, stats in regime_stats.iterrows():
            print(f"   • {stats['regime_name']}: {stats['frequency']:.1%} frequency, "
                  f"{stats['avg_return']:.1%} annual return, "
                  f"{stats['sharpe_ratio']:.2f} Sharpe")
    
    print("\n✅ Regime detection demo completed!")
    return detector, data_with_regimes


def demonstrate_advanced_backtesting():
    """Demonstrate advanced backtesting with realistic costs."""
    print("\n🚀 ADVANCED BACKTESTING DEMO")
    print("=" * 50)
    
    # Generate market data
    market_data = generate_comprehensive_market_data('SPY', 252)
    
    # Create strategy
    strategy = OptionsStrategy('iron_condor', profit_target=0.35, stop_loss=2.0)
    
    # Add technical indicators for strategy
    returns = market_data['Close'].pct_change()
    market_data['returns'] = returns
    market_data['rsi'] = _calculate_rsi(market_data['Close'])
    market_data['volatility'] = returns.rolling(20).std() * np.sqrt(252)
    
    feature_columns = ['rsi', 'volatility']
    features = market_data[feature_columns].fillna(method='ffill')
    
    # Fit strategy
    strategy.fit(features, returns)
    
    # Create transaction cost model
    cost_model = TransactionCostModel(
        commission_per_contract=0.65,
        commission_per_trade=1.0,
        bid_ask_spread_model='linear',
        slippage_model='sqrt'
    )
    
    # Initialize advanced backtester
    backtester = AdvancedBacktester(
        initial_capital=100000,
        cost_model=cost_model
    )
    
    # Run backtest
    print("   Running advanced backtest...")
    results = backtester.backtest_strategy(
        strategy=strategy,
        market_data=market_data,
        position_sizing=0.1,
        rebalance_frequency='monthly'
    )
    
    # Display results
    metrics = results['performance_metrics']
    print(f"\n   📈 Backtest Results:")
    print(f"   • Total Return: {metrics['total_return']:.2%}")
    print(f"   • Annual Return: {metrics['annual_return']:.2%}")
    print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   • Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"   • Total Trades: {metrics['total_trades']}")
    print(f"   • Transaction Costs: ${metrics['total_commissions'] + metrics['total_slippage']:,.0f}")
    print(f"   • Final Capital: ${metrics['final_capital']:,.0f}")
    
    print("\n✅ Advanced backtesting demo completed!")
    return results


def demonstrate_greeks_analysis():
    """Demonstrate Greeks analysis and risk management."""
    print("\n⚡ GREEKS ANALYSIS DEMO")
    print("=" * 50)
    
    # Generate market data
    market_data = generate_comprehensive_market_data('SPY', 126)
    
    # Initialize Greeks analyzer
    greeks_analyzer = GreeksAnalyzer(risk_free_rate=0.05)
    greeks_analyzer.fit(market_data)
    
    # Simulate options positions for analysis
    from rtradez.research.advanced_backtest import OptionsContract, OptionType
    
    current_price = market_data['Close'].iloc[-1]
    
    # Create sample Iron Condor position
    positions = [
        (OptionsContract('SPY', '2024-01-19', current_price * 0.90, OptionType.PUT, 
                        bid=0.50, ask=0.55, volume=100), 1),  # Long put
        (OptionsContract('SPY', '2024-01-19', current_price * 0.95, OptionType.PUT,
                        bid=1.50, ask=1.55, volume=200), -1), # Short put
        (OptionsContract('SPY', '2024-01-19', current_price * 1.05, OptionType.CALL,
                        bid=1.50, ask=1.55, volume=200), -1), # Short call
        (OptionsContract('SPY', '2024-01-19', current_price * 1.10, OptionType.CALL,
                        bid=0.50, ask=0.55, volume=100), 1)   # Long call
    ]
    
    # Analyze portfolio Greeks
    portfolio_greeks = greeks_analyzer.analyze_portfolio_greeks(
        positions, current_price, 0.25
    )
    
    print(f"   📊 Portfolio Greeks Analysis:")
    print(f"   • Delta: {portfolio_greeks.delta:.3f} (${portfolio_greeks.delta_dollars:,.0f})")
    print(f"   • Gamma: {portfolio_greeks.gamma:.3f} (${portfolio_greeks.gamma_dollars:,.0f})")
    print(f"   • Theta: {portfolio_greeks.theta:.3f} (${portfolio_greeks.theta_dollars:,.0f})")
    print(f"   • Vega: {portfolio_greeks.vega:.3f} (${portfolio_greeks.vega_dollars:,.0f})")
    
    # Calculate hedge recommendations
    hedge_contract = OptionsContract('SPY', '2024-01-19', current_price, OptionType.CALL,
                                   bid=5.0, ask=5.10, volume=1000)
    
    hedge_recs = greeks_analyzer.calculate_hedge_recommendations(
        portfolio_greeks, hedge_contract, current_price
    )
    
    print(f"\n   🛡️ Hedge Recommendations:")
    for greek, quantity in hedge_recs.items():
        print(f"   • {greek.replace('_', ' ').title()}: {quantity:.0f} contracts")
    
    # P&L attribution analysis
    price_change = 5.0  # $5 price change
    vol_change = 0.02   # 2% volatility change
    time_decay = 1      # 1 day
    
    pnl_attribution = greeks_analyzer.analyze_greeks_pnl_attribution(
        portfolio_greeks, price_change, vol_change, time_decay
    )
    
    print(f"\n   💰 P&L Attribution (Price +${price_change}, Vol +{vol_change:.0%}, 1 day):")
    for component, pnl in pnl_attribution.items():
        print(f"   • {component.replace('_', ' ').title()}: ${pnl:,.0f}")
    
    print("\n✅ Greeks analysis demo completed!")
    return greeks_analyzer, portfolio_greeks


def demonstrate_research_visualization():
    """Demonstrate advanced research visualization."""
    print("\n📊 RESEARCH VISUALIZATION DEMO")
    print("=" * 50)
    
    # Generate sample data
    market_data = generate_comprehensive_market_data('SPY', 252)
    
    # Run a quick backtest for visualization
    strategy = OptionsStrategy('iron_condor')
    
    # Add features
    returns = market_data['Close'].pct_change()
    market_data['returns'] = returns
    market_data['rsi'] = _calculate_rsi(market_data['Close'])
    market_data['volatility'] = returns.rolling(20).std() * np.sqrt(252)
    
    features = market_data[['rsi', 'volatility']].fillna(method='ffill')
    strategy.fit(features, returns)
    
    # Create mock backtest results
    equity_curve = pd.DataFrame({
        'date': market_data.index,
        'total_value': np.cumsum(np.random.normal(100, 1000, len(market_data))) + 100000,
        'capital': 100000,
        'position_value': np.random.normal(0, 5000, len(market_data))
    })
    
    mock_results = {
        'equity_curve': equity_curve,
        'performance_metrics': {
            'total_return': 0.15,
            'annual_return': 0.12,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'total_trades': 24
        },
        'trades': []
    }
    
    # Initialize visualizer
    visualizer = ResearchVisualizer()
    
    print("   📈 Creating performance attribution chart...")
    perf_fig = visualizer.plot_performance_attribution(mock_results)
    print(f"   ✅ Performance chart created with {len(perf_fig.data)} traces")
    
    # Regime detection visualization
    regime_detector = MarketRegimeDetector(method='combined', n_regimes=3)
    regime_detector.fit(market_data)
    
    print("   🔍 Creating market regime analysis...")
    regime_fig = visualizer.plot_market_regimes(market_data, regime_detector)
    print(f"   ✅ Regime chart created with {len(regime_fig.data)} traces")
    
    # Strategy comparison
    strategy_results = {
        'Iron Condor': mock_results,
        'Strangle': {
            'performance_metrics': {
                'total_return': 0.12,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.10,
                'win_rate': 0.60
            }
        },
        'Straddle': {
            'performance_metrics': {
                'total_return': 0.08,
                'sharpe_ratio': 0.9,
                'max_drawdown': -0.12,
                'win_rate': 0.55
            }
        }
    }
    
    print("   📊 Creating strategy comparison...")
    comparison_fig = visualizer.plot_strategy_comparison(strategy_results)
    print(f"   ✅ Comparison chart created with {len(comparison_fig.data)} traces")
    
    print("\n✅ Research visualization demo completed!")
    return visualizer


def demonstrate_comprehensive_research_workflow():
    """Demonstrate complete research workflow."""
    print("\n🧪 COMPREHENSIVE RESEARCH WORKFLOW")
    print("=" * 80)
    
    # 1. Data preparation with multiple symbols
    print("\n1️⃣ Data Preparation...")
    symbols = ['SPY', 'QQQ', 'IWM']
    datasets = {}
    
    for symbol in symbols:
        datasets[symbol] = generate_comprehensive_market_data(symbol, 252)
        print(f"   ✅ Generated {len(datasets[symbol])} days of data for {symbol}")
    
    # 2. Regime detection across multiple assets
    print("\n2️⃣ Multi-Asset Regime Detection...")
    regime_detectors = {}
    
    for symbol, data in datasets.items():
        detector = MarketRegimeDetector(method='combined', n_regimes=3)
        detector.fit(data)
        regime_detectors[symbol] = detector
        
        regime_stats = detector.get_regime_statistics(data)
        avg_sharpe = regime_stats['sharpe_ratio'].mean()
        print(f"   📊 {symbol}: {len(regime_stats)} regimes detected, avg Sharpe: {avg_sharpe:.2f}")
    
    # 3. Strategy optimization across regimes
    print("\n3️⃣ Regime-Based Strategy Optimization...")
    strategy_configs = {
        0: {'strategy_type': 'iron_condor', 'profit_target': 0.4},    # Low vol
        1: {'strategy_type': 'strangle', 'profit_target': 0.3},      # High vol
        2: {'strategy_type': 'calendar_spread', 'profit_target': 0.5} # Medium vol
    }
    
    regime_strategies = {}
    for symbol in symbols:
        detector = regime_detectors[symbol]
        data = datasets[symbol]
        
        # Add features
        returns = data['Close'].pct_change()
        data['returns'] = returns
        data['rsi'] = _calculate_rsi(data['Close'])
        data['volatility'] = returns.rolling(20).std() * np.sqrt(252)
        
        features = data[['rsi', 'volatility']].fillna(method='ffill')
        
        regime_strategy = RegimeBasedStrategy(
            regime_detector=detector,
            regime_strategies=strategy_configs,
            default_strategy='iron_condor'
        )
        
        regime_strategy.fit(features, returns)
        regime_strategies[symbol] = regime_strategy
        
        score = regime_strategy.score(features, returns)
        print(f"   ⚡ {symbol} regime-based strategy: {score:.3f} Sharpe ratio")
    
    # 4. Advanced backtesting with transaction costs
    print("\n4️⃣ Advanced Backtesting...")
    backtest_results = {}
    
    cost_model = TransactionCostModel(
        commission_per_contract=0.65,
        bid_ask_spread_model='linear'
    )
    
    for symbol in symbols[:2]:  # Test first 2 symbols
        strategy = regime_strategies[symbol]
        data = datasets[symbol]
        
        backtester = AdvancedBacktester(
            initial_capital=100000,
            cost_model=cost_model
        )
        
        results = backtester.backtest_strategy(
            strategy=strategy,
            market_data=data,
            position_sizing=0.1,
            rebalance_frequency='monthly'
        )
        
        backtest_results[symbol] = results
        metrics = results['performance_metrics']
        
        print(f"   📈 {symbol}: {metrics['annual_return']:.1%} return, "
              f"{metrics['sharpe_ratio']:.2f} Sharpe, "
              f"${metrics['total_commissions']:,.0f} costs")
    
    # 5. Research visualization and analysis
    print("\n5️⃣ Research Analysis & Visualization...")
    visualizer = ResearchVisualizer()
    
    # Create comprehensive analysis
    best_symbol = max(backtest_results.keys(), 
                     key=lambda s: backtest_results[s]['performance_metrics']['sharpe_ratio'])
    
    best_results = backtest_results[best_symbol]
    
    print(f"   🏆 Best performing strategy: {best_symbol}")
    print(f"   📊 Creating comprehensive research dashboard...")
    
    # Performance analysis
    perf_metrics = best_results['performance_metrics']
    
    print(f"\n   📈 FINAL RESEARCH RESULTS:")
    print(f"   • Best Symbol: {best_symbol}")
    print(f"   • Total Return: {perf_metrics['total_return']:.2%}")
    print(f"   • Sharpe Ratio: {perf_metrics['sharpe_ratio']:.2f}")
    print(f"   • Max Drawdown: {perf_metrics['max_drawdown']:.2%}")
    print(f"   • Transaction Costs: ${perf_metrics.get('total_commissions', 0):,.0f}")
    
    print("\n✅ Comprehensive research workflow completed!")
    return {
        'datasets': datasets,
        'regime_detectors': regime_detectors,
        'strategies': regime_strategies,
        'backtest_results': backtest_results,
        'best_symbol': best_symbol
    }


def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def main():
    """Run comprehensive advanced research demonstration."""
    print("🚀 RTRADEZ ADVANCED RESEARCH & DEVELOPMENT DEMO")
    print("=" * 80)
    print("Demonstrating cutting-edge quantitative research capabilities...")
    print("=" * 80)
    
    try:
        # Individual component demos
        regime_detector, regime_data = demonstrate_regime_detection()
        backtest_results = demonstrate_advanced_backtesting()
        greeks_analyzer, greeks_profile = demonstrate_greeks_analysis()
        visualizer = demonstrate_research_visualization()
        
        # Comprehensive workflow
        workflow_results = demonstrate_comprehensive_research_workflow()
        
        print(f"\n🎯 DEMO SUMMARY")
        print("=" * 50)
        print("✅ Market regime detection across multiple methodologies")
        print("✅ Advanced backtesting with realistic transaction costs")
        print("✅ Comprehensive Greeks analysis and risk management")
        print("✅ Professional research visualization suite")
        print("✅ End-to-end quantitative research workflow")
        print("\n🔬 RTradez R&D platform is ready for advanced research!")
        
        return {
            'regime_detector': regime_detector,
            'backtest_results': backtest_results,
            'greeks_analyzer': greeks_analyzer,
            'visualizer': visualizer,
            'workflow_results': workflow_results
        }
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()