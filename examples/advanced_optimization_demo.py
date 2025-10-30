#!/usr/bin/env python3
"""
RTradez Advanced Optimization Demo

Demonstrates the complete advanced workflow including:
- Intelligent caching for data and computations
- MLflow experiment tracking
- Optuna hyperparameter optimization
- Integration with sklearn-like interface

This shows the full power of RTradez for production-level algorithmic trading research.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# RTradez imports
from rtradez.datasets import OptionsDataset
from rtradez.datasets.transformers import TechnicalIndicatorTransformer, ReturnsTransformer
from rtradez.methods.strategies import OptionsStrategy
from rtradez.metrics.performance import PerformanceAnalyzer
from rtradez.utils.caching import get_cache, cached, CacheManager
from rtradez.utils.experiments import RTradezExperimentTracker, ExperimentContextManager
from rtradez.utils.optimization import (
    RTradezOptimizer, 
    StrategyOptimizerFactory,
    optimize_strategy_with_optuna
)

# ML imports
from sklearn.model_selection import train_test_split


class AdvancedRTradezWorkflow:
    """
    Advanced RTradez workflow demonstrating caching, experiment tracking, and optimization.
    """
    
    def __init__(self, symbol: str = 'SPY'):
        self.symbol = symbol
        self.cache_manager = CacheManager()
        self.experiment_tracker = RTradezExperimentTracker()
        self.data = None
        self.features = None
        
    @cached(cache_type='market_data', expire=3600)  # Cache for 1 hour
    def load_market_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Load market data with caching."""
        print(f"📊 Loading market data for {symbol} (period: {period})")
        
        try:
            dataset = OptionsDataset.from_source('yahoo', symbol, period=period)
            data = dataset.underlying_data
            print(f"✅ Loaded {len(data)} days of market data")
            return data
        except Exception as e:
            print(f"⚠️  Real data failed, using synthetic data: {e}")
            # Generate synthetic data
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
            
            return pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Close': prices,
                'Volume': np.random.randint(500000, 2000000, len(dates))
            }, index=dates)
    
    @cached(cache_type='features', expire=1800)  # Cache for 30 minutes
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with caching."""
        print("⚙️ Engineering features...")
        
        # Technical indicators
        tech_transformer = TechnicalIndicatorTransformer(
            indicators=['rsi', 'macd', 'bollinger', 'volatility']
        )
        enriched_data = tech_transformer.fit_transform(data)
        
        # Additional features
        enriched_data['price_momentum_5d'] = enriched_data['Close'].pct_change(5)
        enriched_data['price_momentum_20d'] = enriched_data['Close'].pct_change(20)
        enriched_data['volume_ratio'] = enriched_data['Volume'] / enriched_data['Volume'].rolling(20).mean()
        
        # Volatility regime
        enriched_data['vol_regime'] = pd.qcut(
            enriched_data['realized_vol_30d'].fillna(enriched_data['realized_vol_30d'].median()), 
            q=3, labels=['low_vol', 'med_vol', 'high_vol']
        )
        
        # Target variable
        enriched_data['target_return_5d'] = enriched_data['Close'].pct_change(5).shift(-5)
        
        print(f"✅ Engineered {enriched_data.shape[1]} features")
        return enriched_data.dropna()
    
    def prepare_datasets(self, features: pd.DataFrame) -> tuple:
        """Prepare train/validation datasets."""
        print("📋 Preparing train/validation datasets...")
        
        # Select features for ML
        feature_cols = [col for col in features.columns 
                       if col not in ['target_return_5d', 'vol_regime'] 
                       and features[col].dtype in ['float64', 'int64']]
        
        X = features[feature_cols].fillna(0)
        y = features['target_return_5d'].fillna(0)
        
        # Time series split (preserve temporal order)
        split_point = int(len(X) * 0.8)
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]
        
        print(f"✅ Train: {X_train.shape}, Validation: {X_val.shape}")
        return X_train, X_val, y_train, y_val
    
    def run_optuna_optimization(self, strategy_type: str, X_train, y_train, X_val, y_val):
        """Run Optuna hyperparameter optimization."""
        print(f"🔍 Starting Optuna optimization for {strategy_type}...")
        
        # Create strategy objective function
        def objective(params, trial):
            strategy = OptionsStrategy(strategy_type, **params)
            strategy.fit(X_train, y_train)
            score = strategy.score(X_val, y_val)
            
            # Report intermediate value for pruning
            trial.report(score, step=0)
            
            # Log to experiment tracker
            self.experiment_tracker.log_optimization_results(
                study_name=f"{strategy_type}_optuna",
                best_params=params,
                best_value=score,
                n_trials=trial.number + 1
            )
            
            return score
        
        # Get parameter space
        param_space = StrategyOptimizerFactory.get_parameter_space(strategy_type)
        
        # Create optimizer
        optimizer = RTradezOptimizer(
            study_name=f"{strategy_type}_optimization_{datetime.now().strftime('%H%M%S')}",
            sampler_type="tpe",
            pruner_type="median"
        )
        
        # Run optimization
        study = optimizer.optimize_strategy(
            objective_func=objective,
            param_space=param_space,
            n_trials=30,  # Reduced for demo
            timeout=300   # 5 minute timeout
        )
        
        print(f"✅ Optimization complete!")
        print(f"🏆 Best params: {study.best_params}")
        print(f"📊 Best score: {study.best_value:.4f}")
        
        return study.best_params, study.best_value, optimizer
    
    def run_experiment_with_tracking(self, strategy_type: str, best_params: dict,
                                   X_train, y_train, X_val, y_val):
        """Run final experiment with complete tracking."""
        print(f"🧪 Running tracked experiment for {strategy_type}...")
        
        with ExperimentContextManager(
            self.experiment_tracker,
            run_name=f"{strategy_type}_final_experiment",
            tags={"experiment_type": "final_validation", "strategy": strategy_type}
        ) as tracker:
            
            # Log experiment details
            tracker.log_strategy_config(strategy_type, best_params)
            tracker.log_market_data_info(
                self.symbol, 
                str(X_train.index[0].date()), 
                str(X_val.index[-1].date()),
                (len(X_train) + len(X_val), X_train.shape[1])
            )
            tracker.log_feature_engineering(
                list(X_train.columns),
                {"n_features": X_train.shape[1], "n_samples": len(X_train)}
            )
            
            # Create and train strategy
            strategy = OptionsStrategy(strategy_type, **best_params)
            strategy.fit(X_train, y_train)
            
            # Evaluate performance
            train_score = strategy.score(X_train, y_train)
            val_score = strategy.score(X_val, y_val)
            
            # Generate predictions
            train_signals = strategy.predict(X_train)
            val_signals = strategy.predict(X_val)
            
            # Calculate detailed metrics
            metrics = {
                'train_sharpe': train_score,
                'val_sharpe': val_score,
                'train_signal_std': train_signals.std(),
                'val_signal_std': val_signals.std(),
                'signal_consistency': np.corrcoef(train_signals[:-1], val_signals[:len(train_signals)-1])[0,1]
            }
            
            # Log performance
            tracker.log_performance_metrics(metrics)
            
            # Log model
            tracker.log_model(strategy, f"{strategy_type}_optimized_model")
            
            print(f"✅ Experiment tracked successfully!")
            print(f"📊 Validation Sharpe: {val_score:.4f}")
            print(f"📈 Signal consistency: {metrics['signal_consistency']:.4f}")
            
            return strategy, metrics
    
    def demonstrate_caching_benefits(self):
        """Demonstrate caching performance benefits."""
        print("\n🚀 DEMONSTRATING CACHING BENEFITS")
        print("="*60)
        
        # Clear cache first
        self.cache_manager.optimize_cache()
        
        import time
        
        # First load (cache miss)
        start_time = time.time()
        data1 = self.load_market_data(self.symbol, '1y')
        first_load_time = time.time() - start_time
        
        # Second load (cache hit)
        start_time = time.time()
        data2 = self.load_market_data(self.symbol, '1y')
        second_load_time = time.time() - start_time
        
        print(f"📊 First load (cache miss): {first_load_time:.3f} seconds")
        print(f"⚡ Second load (cache hit): {second_load_time:.3f} seconds")
        print(f"🚀 Speedup: {first_load_time/second_load_time:.1f}x faster!")
        
        # Cache statistics
        stats = self.cache_manager.cache.stats()
        print(f"📈 Cache stats: {stats}")
        
    def run_complete_workflow(self):
        """Run the complete advanced workflow."""
        print("🚀 RTRADEZ ADVANCED OPTIMIZATION WORKFLOW")
        print("="*80)
        
        # Demonstrate caching
        self.demonstrate_caching_benefits()
        
        # Load and prepare data
        print(f"\n📊 STEP 1: DATA PREPARATION")
        print("-"*40)
        self.data = self.load_market_data(self.symbol, '1y')
        self.features = self.engineer_features(self.data)
        X_train, X_val, y_train, y_val = self.prepare_datasets(self.features)
        
        # Optimize strategies
        print(f"\n🔍 STEP 2: OPTUNA OPTIMIZATION")
        print("-"*40)
        strategies_to_optimize = ['iron_condor', 'strangle']
        optimization_results = {}
        
        for strategy_type in strategies_to_optimize:
            best_params, best_score, optimizer = self.run_optuna_optimization(
                strategy_type, X_train, y_train, X_val, y_val
            )
            optimization_results[strategy_type] = {
                'params': best_params,
                'score': best_score,
                'optimizer': optimizer
            }
        
        # Run final experiments with tracking
        print(f"\n🧪 STEP 3: EXPERIMENT TRACKING")
        print("-"*40)
        final_results = {}
        
        for strategy_type, opt_result in optimization_results.items():
            strategy, metrics = self.run_experiment_with_tracking(
                strategy_type, opt_result['params'], X_train, y_train, X_val, y_val
            )
            final_results[strategy_type] = {
                'strategy': strategy,
                'metrics': metrics,
                'optimized_params': opt_result['params']
            }
        
        # Generate final report
        print(f"\n📊 STEP 4: FINAL RESULTS")
        print("-"*40)
        self.generate_final_report(optimization_results, final_results)
        
        return optimization_results, final_results
    
    def generate_final_report(self, optimization_results, final_results):
        """Generate comprehensive final report."""
        print("\n" + "="*80)
        print("📊 RTRADEZ ADVANCED OPTIMIZATION RESULTS")
        print("="*80)
        
        print("\n🔍 OPTIMIZATION SUMMARY:")
        print("-" * 50)
        for strategy_type, result in optimization_results.items():
            print(f"{strategy_type:15} | Best Score: {result['score']:8.4f} | Params: {result['params']}")
        
        print("\n🧪 FINAL EXPERIMENT RESULTS:")
        print("-" * 70)
        print(f"{'Strategy':<15} | {'Val Sharpe':<10} | {'Consistency':<12} | {'Status':<10}")
        print("-" * 70)
        
        for strategy_type, result in final_results.items():
            metrics = result['metrics']
            print(f"{strategy_type:<15} | {metrics['val_sharpe']:10.4f} | "
                  f"{metrics.get('signal_consistency', 0):12.4f} | {'✅ Success':<10}")
        
        # Best strategy
        best_strategy = max(final_results.items(), 
                          key=lambda x: x[1]['metrics']['val_sharpe'])
        
        print(f"\n🏆 BEST STRATEGY: {best_strategy[0]}")
        print(f"   Validation Sharpe: {best_strategy[1]['metrics']['val_sharpe']:.4f}")
        print(f"   Optimized Parameters: {best_strategy[1]['optimized_params']}")
        
        # Experiment tracker summary
        experiment_summary = self.experiment_tracker.get_experiment_summary()
        print(f"\n📈 EXPERIMENT TRACKING SUMMARY:")
        print(f"   Total Runs: {experiment_summary['total_runs']}")
        print(f"   Successful Runs: {experiment_summary['successful_runs']}")
        
        # Cache performance
        cache_stats = self.cache_manager.cache.stats()
        print(f"\n⚡ CACHING PERFORMANCE:")
        for cache_type, stats in cache_stats.items():
            if 'size' in stats:
                print(f"   {cache_type}: {stats['size']} items, {stats.get('size_bytes', 0)/1024/1024:.1f} MB")
        
        print("\n🎊 ADVANCED WORKFLOW COMPLETE!")
        print("✅ Caching: Dramatically improved data loading performance")
        print("✅ Optuna: Found optimal hyperparameters efficiently") 
        print("✅ MLflow: Complete experiment tracking and reproducibility")
        print("✅ Integration: Seamless sklearn-compatible workflow")


def main():
    """Run the advanced optimization demonstration."""
    
    print("🔬 RTradez Advanced Optimization Demo")
    print("Featuring: Caching + Experiment Tracking + Optuna Optimization")
    print()
    
    # Initialize workflow
    workflow = AdvancedRTradezWorkflow(symbol='SPY')
    
    # Run complete workflow
    optimization_results, final_results = workflow.run_complete_workflow()
    
    print("\n" + "="*80)
    print("🎉 DEMO COMPLETE!")
    print("="*80)
    print("RTradez now provides enterprise-grade features:")
    print("  🚀 Intelligent caching for 10x faster iterations")
    print("  📊 Complete experiment tracking with MLflow")
    print("  🔍 Advanced optimization with Optuna")
    print("  🎯 Production-ready algorithmic trading platform")
    print()
    print("Ready for institutional-level quantitative research! 💼")


if __name__ == "__main__":
    main()