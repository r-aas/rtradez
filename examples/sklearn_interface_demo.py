#!/usr/bin/env python3
"""
RTradez Sklearn-like Interface Demo

Demonstrates the sklearn-style API across all RTradez components:
- .fit() -> learn from data
- .predict() -> make predictions/generate signals
- .transform() -> transform data
- .score() -> evaluate performance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# RTradez sklearn-like imports
from rtradez.datasets import OptionsDataset
from rtradez.datasets.transformers import (
    TechnicalIndicatorTransformer, 
    ReturnsTransformer,
    OptionsChainStandardizer
)
from rtradez.methods.strategies import OptionsStrategy
from rtradez.metrics.performance import PerformanceAnalyzer


def demonstrate_sklearn_interface():
    """Demonstrate sklearn-like interface patterns."""
    
    print("🔬 RTradez Sklearn-like Interface Demo")
    print("=" * 60)
    
    # 1. Load and prepare data (sklearn-style)
    print("\n📊 Step 1: Data Loading and Preparation")
    try:
        dataset = OptionsDataset.from_source('yahoo', 'SPY', period='6mo')
        price_data = dataset.underlying_data
        print(f"✅ Loaded {len(price_data)} days of SPY data")
        
        # Prepare features with required columns
        if 'High' not in price_data.columns:
            price_data['High'] = price_data['Close']
        if 'Low' not in price_data.columns:
            price_data['Low'] = price_data['Close']
        if 'Volume' not in price_data.columns:
            price_data['Volume'] = 1000000  # Default volume
            
        print(f"📈 Columns available: {list(price_data.columns)}")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("🔄 Using synthetic data for demo...")
        
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        
        price_data = pd.DataFrame({
            'Close': prices,
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Volume': np.random.randint(500000, 2000000, len(dates))
        }, index=dates)
        
        print(f"✅ Created synthetic data: {len(price_data)} days")
    
    # 2. Data Transformation (sklearn-style fit/transform)
    print("\n🔧 Step 2: Data Transformation (sklearn-style)")
    
    # Technical indicators transformer
    tech_transformer = TechnicalIndicatorTransformer(
        indicators=['rsi', 'macd', 'volatility']
    )
    
    print("📊 Fitting technical indicator transformer...")
    tech_transformer.fit(price_data)
    enriched_data = tech_transformer.transform(price_data)
    print(f"✅ Added technical indicators. New shape: {enriched_data.shape}")
    print(f"📈 New columns: {list(enriched_data.columns)}")
    
    # Returns transformer
    returns_transformer = ReturnsTransformer(method='simple', periods=1)
    returns = returns_transformer.fit_transform(enriched_data[['Close']])
    print(f"✅ Calculated returns. Shape: {returns.shape}")
    
    # 3. Strategy Training (sklearn-style fit/predict)
    print("\n🎯 Step 3: Strategy Training (sklearn-style)")
    
    # Create strategy
    strategy = OptionsStrategy('iron_condor', 
                              profit_target=0.5, 
                              stop_loss=2.0)
    
    print("🔬 Training strategy on historical data...")
    strategy.fit(enriched_data, returns['Close'])
    print(f"✅ Strategy fitted! Learned volatility: {strategy.learned_volatility_:.3f}")
    print(f"📊 Optimal parameters: {strategy.optimal_params_}")
    
    # Generate signals on new data
    signals = strategy.predict(enriched_data)
    print(f"✅ Generated {len(signals)} trading signals")
    print(f"📈 Signal distribution: {np.bincount(signals.astype(int) + 1)} (sell/hold/buy)")
    
    # 4. Performance Evaluation (sklearn-style scoring)
    print("\n📊 Step 4: Performance Evaluation (sklearn-style)")
    
    # Calculate strategy performance
    strategy_score = strategy.score(enriched_data, returns['Close'])
    print(f"✅ Strategy Sharpe ratio: {strategy_score:.3f}")
    
    if strategy.historical_performance_:
        perf = strategy.historical_performance_
        print(f"📈 Historical total return: {perf['total_return']:.2%}")
        print(f"📉 Historical max drawdown: {perf['max_drawdown']:.2%}")
    
    # 5. Advanced Metrics (sklearn-style fit/score)
    print("\n📏 Step 5: Advanced Metrics (sklearn-style)")
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    # Fit analyzer to learn baseline parameters
    analyzer.fit(pd.DataFrame({'returns': returns['Close'].dropna()}))
    
    # Calculate strategy returns for scoring with proper alignment
    min_length = min(len(signals) - 1, len(returns['Close']) - 1)
    aligned_signals = signals[:min_length]
    aligned_returns = returns['Close'].values[1:min_length + 1]
    
    strategy_returns = pd.Series(aligned_signals * aligned_returns, 
                                index=returns.index[1:min_length + 1])
    
    # Score performance
    performance_score = analyzer.score(returns['Close'][1:], strategy_returns)
    print(f"✅ Performance analyzer score: {performance_score:.3f}")
    
    # 6. Pipeline Demo (sklearn-style pipeline)
    print("\n🔄 Step 6: Sklearn-style Pipeline Demo")
    
    print("🚀 Complete pipeline execution:")
    print("  1. Raw data → Technical indicators (fit/transform)")
    print("  2. Price data → Returns (fit/transform)")  
    print("  3. Historical data → Strategy training (fit)")
    print("  4. Market conditions → Trading signals (predict)")
    print("  5. Actual vs predicted → Performance score (score)")
    
    # Demonstrate pipeline-like chaining
    pipeline_result = (
        tech_transformer
        .fit(price_data)
        .transform(price_data)
    )
    
    strategy_pipeline = (
        strategy
        .fit(pipeline_result, returns['Close'])
        .predict(pipeline_result)
    )
    
    print(f"✅ Pipeline executed successfully!")
    print(f"📊 Final signals shape: {strategy_pipeline.shape}")
    
    # 7. Summary of sklearn-like patterns
    print("\n📋 Step 7: Sklearn Interface Summary")
    print("🎯 RTradez now follows sklearn conventions:")
    print()
    print("📊 TRANSFORMERS:")
    print("  • .fit(X) -> learn transformation parameters")
    print("  • .transform(X) -> apply transformation")
    print("  • .fit_transform(X) -> combined fit and transform")
    print()
    print("🎯 STRATEGIES:")
    print("  • .fit(market_data, returns) -> learn optimal parameters")
    print("  • .predict(market_conditions) -> generate trading signals")
    print("  • .score(X, y) -> evaluate strategy performance")
    print()
    print("📏 METRICS:")
    print("  • .fit(historical_data) -> learn baseline parameters")
    print("  • .score(y_true, y_pred) -> calculate performance metrics")
    print()
    print("🔧 ESTIMATORS:")
    print("  • .fit(features, target) -> train model")
    print("  • .predict(features) -> make predictions")
    print("  • .score(features, target) -> evaluate predictions")
    
    print("\n🎉 Sklearn-like interface demonstration complete!")
    print("🚀 RTradez now provides familiar, consistent API patterns!")


if __name__ == "__main__":
    demonstrate_sklearn_interface()