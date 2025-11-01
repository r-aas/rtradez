"""Comprehensive benchmark validation across multiple asset classes and market conditions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta

from rtradez.datasets.benchmark_datasets import BenchmarkDatasets, BENCHMARK_SUITES, AssetClass
from rtradez.methods.strategies import OptionsStrategy
from rtradez.utils.caching import cached
from rtradez.utils.experiments import RTradezExperimentTracker
from rtradez.utils.optimization import RTradezOptimizer


class ComprehensiveBenchmarkValidator:
    """Validate strategies across comprehensive benchmark datasets."""
    
    def __init__(self, 
                 benchmark_suite: str = 'comprehensive',
                 validation_periods: Optional[List[str]] = None,
                 strategy_types: Optional[List[str]] = None):
        """
        Initialize comprehensive benchmark validator.
        
        Args:
            benchmark_suite: Predefined suite or 'custom'
            validation_periods: List of start dates for validation periods
            strategy_types: List of strategy types to test
        """
        self.benchmark_suite = benchmark_suite
        self.symbols = BENCHMARK_SUITES.get(benchmark_suite, BenchmarkDatasets.get_high_priority_symbols())
        self.validation_periods = validation_periods or ['2020-01-01', '2022-01-01', '2024-01-01']
        self.strategy_types = strategy_types or ['iron_condor', 'strangle', 'straddle', 'calendar_spread']
        
        # Get dataset metadata
        all_datasets = BenchmarkDatasets.get_all_datasets()
        self.dataset_info = {d.symbol: d for d in all_datasets if d.symbol in self.symbols}
        
        self.tracker = RTradezExperimentTracker(f"comprehensive_benchmark_{benchmark_suite}")
        self.optimizer = RTradezOptimizer(f"benchmark_optimization_{benchmark_suite}")
        
        # Results storage
        self.results = {}
        self.performance_matrix = pd.DataFrame()
        
    @cached(cache_type='market_data', expire=86400)
    def load_benchmark_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load benchmark data with caching."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"Warning: No data available for {symbol}")
                return self._generate_synthetic_data(symbol, start_date, end_date)
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return self._generate_synthetic_data(symbol, start_date, end_date)
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for feature engineering."""
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        data['bb_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volatility
        data['returns'] = data['Close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volume indicators
        data['volume_sma'] = data['Volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        
        return data.fillna(method='bfill').fillna(method='ffill')
    
    def _generate_synthetic_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic data for testing when real data unavailable."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Business days only
        
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        # Get dataset info for realistic parameters
        dataset_info = self.dataset_info.get(symbol)
        base_vol = dataset_info.typical_iv if dataset_info and dataset_info.typical_iv else 0.25
        
        # Generate price series
        returns = np.random.normal(0.0002, base_vol / np.sqrt(252), len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        # Generate volume
        volumes = np.random.lognormal(10, 1, len(dates))
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return self._add_technical_indicators(data)
    
    def run_asset_class_analysis(self) -> Dict:
        """Analyze performance by asset class."""
        asset_class_results = {}
        
        for asset_class in AssetClass:
            symbols = [d.symbol for d in BenchmarkDatasets.get_by_asset_class(asset_class) 
                      if d.symbol in self.symbols]
            
            if not symbols:
                continue
                
            print(f"\n🔍 Analyzing {asset_class.value} ({len(symbols)} symbols)")
            
            class_performance = []
            for symbol in symbols:
                # Run validation for each symbol
                symbol_results = self.validate_symbol(symbol)
                if symbol_results:
                    class_performance.append(symbol_results)
            
            if class_performance:
                # Aggregate results by asset class
                avg_sharpe = np.mean([r['best_sharpe'] for r in class_performance])
                avg_return = np.mean([r['best_return'] for r in class_performance])
                consistency = np.std([r['best_sharpe'] for r in class_performance])
                
                asset_class_results[asset_class.value] = {
                    'symbols': symbols,
                    'avg_sharpe': avg_sharpe,
                    'avg_return': avg_return,
                    'consistency_score': 1 / (1 + consistency),  # Higher is better
                    'symbol_count': len(symbols),
                    'detailed_results': class_performance
                }
        
        return asset_class_results
    
    def validate_symbol(self, symbol: str) -> Optional[Dict]:
        """Validate strategies on a single symbol."""
        print(f"  📊 Validating {symbol}...")
        
        try:
            # Load data for validation period
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = self.validation_periods[0]
            
            data = self.load_benchmark_data(symbol, start_date, end_date)
            if data.empty:
                return None
            
            # Prepare features
            feature_columns = ['rsi', 'macd', 'bb_position', 'volatility', 'volume_ratio']
            X = data[feature_columns].dropna()
            y = data['returns'].loc[X.index]
            
            if len(X) < 100:  # Need minimum data for validation
                print(f"    ⚠️  Insufficient data for {symbol}")
                return None
            
            best_strategy = None
            best_score = -np.inf
            strategy_results = {}
            
            # Test each strategy type
            for strategy_type in self.strategy_types:
                try:
                    strategy = OptionsStrategy(strategy_type)
                    strategy.fit(X, y)
                    score = strategy.score(X, y)
                    
                    strategy_results[strategy_type] = {
                        'sharpe_ratio': score,
                        'params': strategy.get_params()
                    }
                    
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy_type
                        
                except Exception as e:
                    print(f"    ❌ {strategy_type} failed on {symbol}: {e}")
                    continue
            
            if best_strategy:
                # Calculate additional metrics for best strategy
                best_strat = OptionsStrategy(best_strategy)
                best_strat.fit(X, y)
                signals = best_strat.predict(X)
                
                # Calculate returns
                strategy_returns = signals * y.shift(-1)  # Next period returns
                total_return = (1 + strategy_returns).prod() - 1
                
                return {
                    'symbol': symbol,
                    'best_strategy': best_strategy,
                    'best_sharpe': best_score,
                    'best_return': total_return,
                    'asset_class': self.dataset_info[symbol].asset_class.value if symbol in self.dataset_info else 'unknown',
                    'all_strategies': strategy_results,
                    'data_points': len(X)
                }
            
        except Exception as e:
            print(f"    ❌ Validation failed for {symbol}: {e}")
            
        return None
    
    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive validation across all benchmarks."""
        print(f"🚀 Starting comprehensive benchmark validation")
        print(f"📊 Benchmark Suite: {self.benchmark_suite}")
        print(f"🎯 Symbols: {len(self.symbols)} ({', '.join(self.symbols[:10])}{'...' if len(self.symbols) > 10 else ''})")
        print(f"📈 Strategies: {', '.join(self.strategy_types)}")
        
        self.tracker.start_experiment()
        
        # Individual symbol validation
        symbol_results = []
        for symbol in self.symbols:
            result = self.validate_symbol(symbol)
            if result:
                symbol_results.append(result)
        
        # Asset class analysis
        asset_class_results = self.run_asset_class_analysis()
        
        # Overall analysis
        if symbol_results:
            overall_best = max(symbol_results, key=lambda x: x['best_sharpe'])
            
            # Create performance matrix
            self.performance_matrix = self._create_performance_matrix(symbol_results)
            
            # Strategy ranking across all symbols
            strategy_rankings = self._rank_strategies_globally(symbol_results)
            
            results = {
                'validation_summary': {
                    'total_symbols': len(self.symbols),
                    'successful_validations': len(symbol_results),
                    'benchmark_suite': self.benchmark_suite,
                    'validation_date': datetime.now().isoformat()
                },
                'overall_best': overall_best,
                'symbol_results': symbol_results,
                'asset_class_analysis': asset_class_results,
                'strategy_rankings': strategy_rankings,
                'performance_matrix': self.performance_matrix.to_dict()
            }
            
            # Log to MLflow
            self.tracker.log_strategy_config('comprehensive_benchmark', {
                'benchmark_suite': self.benchmark_suite,
                'symbol_count': len(self.symbols)
            })
            
            self.tracker.log_performance_metrics({
                'best_overall_sharpe': overall_best['best_sharpe'],
                'successful_validations': len(symbol_results),
                'validation_success_rate': len(symbol_results) / len(self.symbols)
            })
            
            self.results = results
            return results
        
        else:
            print("❌ No successful validations")
            return {}
    
    def _create_performance_matrix(self, symbol_results: List[Dict]) -> pd.DataFrame:
        """Create performance matrix showing strategy performance across symbols."""
        matrix_data = []
        
        for result in symbol_results:
            row = {'symbol': result['symbol'], 'asset_class': result['asset_class']}
            for strategy_type in self.strategy_types:
                if strategy_type in result['all_strategies']:
                    row[strategy_type] = result['all_strategies'][strategy_type]['sharpe_ratio']
                else:
                    row[strategy_type] = np.nan
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data)
    
    def _rank_strategies_globally(self, symbol_results: List[Dict]) -> Dict:
        """Rank strategies based on performance across all symbols."""
        strategy_performance = {strategy: [] for strategy in self.strategy_types}
        
        for result in symbol_results:
            for strategy_type in self.strategy_types:
                if strategy_type in result['all_strategies']:
                    sharpe = result['all_strategies'][strategy_type]['sharpe_ratio']
                    strategy_performance[strategy_type].append(sharpe)
        
        rankings = {}
        for strategy, sharpes in strategy_performance.items():
            if sharpes:
                rankings[strategy] = {
                    'avg_sharpe': np.mean(sharpes),
                    'median_sharpe': np.median(sharpes),
                    'std_sharpe': np.std(sharpes),
                    'success_rate': len(sharpes) / len(symbol_results),
                    'best_sharpe': max(sharpes),
                    'worst_sharpe': min(sharpes)
                }
        
        # Sort by average Sharpe ratio
        sorted_rankings = dict(sorted(rankings.items(), 
                                    key=lambda x: x[1]['avg_sharpe'], 
                                    reverse=True))
        
        return sorted_rankings
    
    def generate_benchmark_report(self, save_path: str = 'benchmark_validation_report.json'):
        """Generate comprehensive benchmark report."""
        if not self.results:
            print("❌ No results to report. Run validation first.")
            return
        
        # Save detailed results
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Print executive summary
        print(f"\n🎯 COMPREHENSIVE BENCHMARK VALIDATION REPORT")
        print(f"=" * 60)
        
        summary = self.results['validation_summary']
        print(f"📊 Validation Summary:")
        print(f"  • Benchmark Suite: {summary['benchmark_suite']}")
        print(f"  • Total Symbols: {summary['total_symbols']}")
        print(f"  • Successful Validations: {summary['successful_validations']}")
        print(f"  • Success Rate: {summary['successful_validations']/summary['total_symbols']:.1%}")
        
        # Overall best
        best = self.results['overall_best']
        print(f"\n🏆 Overall Best Performance:")
        print(f"  • Strategy: {best['best_strategy']}")
        print(f"  • Symbol: {best['symbol']} ({best['asset_class']})")
        print(f"  • Sharpe Ratio: {best['best_sharpe']:.3f}")
        print(f"  • Return: {best['best_return']:.2%}")
        
        # Strategy rankings
        print(f"\n📈 Strategy Rankings (by avg Sharpe):")
        for i, (strategy, metrics) in enumerate(self.results['strategy_rankings'].items(), 1):
            print(f"  {i}. {strategy}: {metrics['avg_sharpe']:.3f} (success: {metrics['success_rate']:.1%})")
        
        # Asset class performance
        print(f"\n🌍 Asset Class Performance:")
        for asset_class, metrics in self.results['asset_class_analysis'].items():
            print(f"  • {asset_class}: {metrics['avg_sharpe']:.3f} Sharpe ({metrics['symbol_count']} symbols)")
        
        print(f"\n📁 Detailed report saved to: {save_path}")


def main():
    """Run comprehensive benchmark validation."""
    
    # Test different benchmark suites
    suites_to_test = ['core', 'extended_indices', 'sector_rotation', 'international']
    
    for suite in suites_to_test:
        print(f"\n{'='*80}")
        print(f"🔍 TESTING BENCHMARK SUITE: {suite.upper()}")
        print(f"{'='*80}")
        
        validator = ComprehensiveBenchmarkValidator(
            benchmark_suite=suite,
            validation_periods=['2022-01-01', '2023-01-01', '2024-01-01'],
            strategy_types=['iron_condor', 'strangle', 'straddle']
        )
        
        results = validator.run_comprehensive_validation()
        
        if results:
            validator.generate_benchmark_report(f'benchmark_report_{suite}.json')
        
        print(f"\n✅ Completed {suite} benchmark validation")


if __name__ == "__main__":
    main()