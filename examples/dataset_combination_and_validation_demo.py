"""Comprehensive demonstration of dataset combination and overfitting prevention."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from rtradez.data_sources import RTradezDataManager
from rtradez.utils.dataset_combiner import DatasetCombiner, AdvancedFeatureSelector
from rtradez.validation.overfitting_prevention import OverfittingDetector, UnderfittingDetector
from rtradez.validation.time_series_cv import TimeSeriesValidation, WalkForwardCV
from rtradez.methods.strategies import OptionsStrategy

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def demonstrate_dataset_combination():
    """Demonstrate advanced dataset combination capabilities."""
    print("🔗 DATASET COMBINATION DEMO")
    print("=" * 60)
    
    # Initialize data manager
    data_manager = RTradezDataManager()
    
    # Get datasets from different sources
    print("\n📊 Fetching multi-source datasets...")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Collect datasets
    datasets = {}
    
    # Market data
    print("   📈 Market data (SPY)...")
    market_data = data_manager.get_comprehensive_dataset(
        symbol='SPY',
        start_date=start_date,
        end_date=end_date,
        include_sources=['sentiment']
    )
    if not market_data.empty:
        datasets['market'] = market_data.iloc[:, :10]  # Limit columns for demo
    
    # Economic data
    print("   🏛️ Economic indicators...")
    econ_data = data_manager.get_economic_research_dataset(
        start_date=start_date,
        end_date=end_date
    )
    if not econ_data.empty:
        datasets['economic'] = econ_data.iloc[:, :8]  # Limit columns
    
    # Crypto data
    print("   🔗 Crypto data (BTC)...")
    crypto_datasets = data_manager.get_crypto_research_dataset(
        symbols=['BTC-USD'],
        start_date=start_date,
        end_date=end_date
    )
    if crypto_datasets and 'BTC-USD' in crypto_datasets:
        datasets['crypto'] = crypto_datasets['BTC-USD'].iloc[:, :6]  # Limit columns
    
    if not datasets:
        print("   ❌ No datasets available, generating synthetic data...")
        datasets = _generate_synthetic_datasets(start_date, end_date)
    
    print(f"   ✅ Collected {len(datasets)} datasets")
    for name, df in datasets.items():
        print(f"      • {name}: {len(df)} rows, {len(df.columns)} features")
    
    # Initialize dataset combiner
    print("\n🔧 Initializing advanced dataset combiner...")
    combiner = DatasetCombiner(
        alignment_method='inner',
        missing_data_strategy='forward_fill',
        outlier_detection=True,
        feature_scaling='standard',
        max_missing_ratio=0.4
    )
    
    # Combine datasets
    print("\n⚙️ Combining datasets with intelligent preprocessing...")
    combined_data = combiner.combine_datasets(
        datasets=datasets,
        primary_dataset='market',
        feature_prefix=True
    )
    
    if combined_data.empty:
        print("   ❌ Dataset combination failed")
        return None
    
    print(f"   ✅ Combined dataset: {len(combined_data)} rows, {len(combined_data.columns)} features")
    
    # Get combination report
    report = combiner.get_combination_report()
    print(f"   📊 Data coverage: {report['combination_stats']['data_coverage']:.1%}")
    print(f"   💾 Memory usage: {report['combination_stats']['memory_usage_mb']:.1f} MB")
    print(f"   🔄 Feature reduction: {report['combination_stats']['feature_reduction_ratio']:.1%}")
    
    return combined_data, combiner


def demonstrate_feature_selection(combined_data: pd.DataFrame):
    """Demonstrate advanced feature selection."""
    print("\n🎯 FEATURE SELECTION DEMO")
    print("=" * 60)
    
    if combined_data.empty:
        print("   ❌ No data available for feature selection")
        return None, None
    
    # Create target variable (next-day returns)
    target_col = None
    for col in combined_data.columns:
        if 'close' in col.lower() or 'price' in col.lower():
            target_col = col
            break
    
    if target_col is None:
        # Use first numeric column
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
    
    if target_col is None:
        print("   ❌ No suitable target variable found")
        return None, None
    
    # Create returns-based target
    target = combined_data[target_col].pct_change().shift(-1)  # Next-day return
    features = combined_data.drop(columns=[target_col])
    
    # Remove rows with missing target
    valid_idx = ~target.isnull()
    features_clean = features[valid_idx]
    target_clean = target[valid_idx]
    
    print(f"   📊 Feature matrix: {len(features_clean)} samples, {len(features_clean.columns)} features")
    print(f"   🎯 Target variable: {target_col} returns")
    
    # Test different feature selection methods
    selection_methods = ['correlation', 'variance', 'statistical', 'combined']
    selection_results = {}
    
    for method in selection_methods:
        print(f"\n   🔍 Testing {method} feature selection...")
        
        try:
            # Initialize selector
            if method == 'statistical':
                max_features = min(20, len(features_clean.columns) // 2)
            else:
                max_features = None
            
            selector = AdvancedFeatureSelector(
                method=method,
                max_features=max_features,
                correlation_threshold=0.9,
                variance_threshold=0.01
            )
            
            # Fit and transform
            selected_features = selector.fit_transform(features_clean, target_clean)
            
            # Get selection report
            feature_report = selector.get_feature_report()
            
            selection_results[method] = {
                'selected_features': selected_features,
                'n_features': len(selected_features.columns),
                'report': feature_report
            }
            
            print(f"      ✅ Selected {len(selected_features.columns)} features")
            
            # Show feature importance if available
            if hasattr(selector, 'feature_scores_') and selector.feature_scores_:
                top_features = sorted(selector.feature_scores_.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                print(f"      🏆 Top features: {[f[0] for f in top_features]}")
            
        except Exception as e:
            print(f"      ❌ {method} selection failed: {e}")
            selection_results[method] = {'error': str(e)}
    
    # Return best selection method (by number of features selected)
    best_method = None
    best_features = None
    
    for method, result in selection_results.items():
        if 'error' not in result:
            if best_method is None or result['n_features'] > 0:
                best_method = method
                best_features = result['selected_features']
    
    if best_features is not None:
        print(f"\n   🎯 Best selection method: {best_method}")
        print(f"   📊 Final feature set: {len(best_features.columns)} features")
        return best_features, target_clean
    else:
        print("   ❌ All feature selection methods failed")
        return features_clean, target_clean


def demonstrate_overfitting_prevention(X: pd.DataFrame, y: pd.Series):
    """Demonstrate overfitting and underfitting detection."""
    print("\n🛡️ OVERFITTING PREVENTION DEMO")
    print("=" * 60)
    
    if X.empty or y.empty:
        print("   ❌ No data available for overfitting analysis")
        return
    
    # Align data
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    # Remove rows with missing values
    complete_idx = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
    X_clean = X_aligned[complete_idx]
    y_clean = y_aligned[complete_idx]
    
    if len(X_clean) < 100:
        print(f"   ❌ Insufficient clean data: {len(X_clean)} samples")
        return
    
    print(f"   📊 Clean dataset: {len(X_clean)} samples, {len(X_clean.columns)} features")
    
    # Test different model complexities
    models = {
        'simple_ridge': Ridge(alpha=10.0, random_state=42),
        'moderate_ridge': Ridge(alpha=1.0, random_state=42),
        'complex_rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'very_complex_rf': RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
    }
    
    # Initialize detectors
    overfitting_detector = OverfittingDetector(
        min_train_samples=min(100, len(X_clean) // 3),
        validation_horizon=min(50, len(X_clean) // 6),
        step_size=min(20, len(X_clean) // 10)
    )
    
    underfitting_detector = UnderfittingDetector()
    
    analysis_results = {}
    
    for model_name, model in models.items():
        print(f"\n   🔍 Analyzing {model_name}...")
        
        try:
            # Overfitting detection
            print(f"      📈 Overfitting analysis...")
            overfitting_result = overfitting_detector.detect_overfitting(
                model, X_clean, y_clean, model_name
            )
            
            # Underfitting detection
            print(f"      📉 Underfitting analysis...")
            underfitting_result = underfitting_detector.detect_underfitting(
                model, X_clean, y_clean, model_name
            )
            
            analysis_results[model_name] = {
                'overfitting': overfitting_result,
                'underfitting': underfitting_result
            }
            
            # Print key findings
            of_risk = overfitting_result['assessment']['risk_level']
            uf_risk = underfitting_result['assessment']['risk_level']
            
            print(f"      🛡️ Overfitting risk: {of_risk}")
            print(f"      📊 Underfitting risk: {uf_risk}")
            
            # Show top recommendations
            if overfitting_result['assessment']['recommendations']:
                print(f"      💡 Key recommendation: {overfitting_result['assessment']['recommendations'][0]}")
            
        except Exception as e:
            print(f"      ❌ Analysis failed: {e}")
            analysis_results[model_name] = {'error': str(e)}
    
    # Summary
    print(f"\n   📋 ANALYSIS SUMMARY")
    print(f"   " + "=" * 40)
    
    for model_name, result in analysis_results.items():
        if 'error' not in result:
            of_risk = result['overfitting']['assessment']['risk_level']
            uf_risk = result['underfitting']['assessment']['risk_level']
            
            risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}
            
            print(f"   {model_name}:")
            print(f"      Overfitting: {risk_emoji.get(of_risk, '❓')} {of_risk}")
            print(f"      Underfitting: {risk_emoji.get(uf_risk, '❓')} {uf_risk}")
    
    return analysis_results


def demonstrate_time_series_cv(X: pd.DataFrame, y: pd.Series):
    """Demonstrate advanced time series cross-validation."""
    print("\n⏰ TIME SERIES CROSS-VALIDATION DEMO")
    print("=" * 60)
    
    if X.empty or y.empty:
        print("   ❌ No data available for CV analysis")
        return
    
    # Align and clean data
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    complete_idx = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
    X_clean = X_aligned[complete_idx]
    y_clean = y_aligned[complete_idx]
    
    if len(X_clean) < 200:
        print(f"   ❌ Insufficient data for CV: {len(X_clean)} samples")
        return
    
    print(f"   📊 Dataset: {len(X_clean)} samples, {len(X_clean.columns)} features")
    
    # Initialize validation framework
    cv_framework = TimeSeriesValidation(
        cv_methods=['walk_forward', 'purged_group'],
        scoring_metrics=['mse', 'mae', 'r2', 'sharpe']
    )
    
    # Test models
    test_models = {
        'ridge': Ridge(alpha=1.0, random_state=42),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    }
    
    print(f"\n   🧪 Testing {len(test_models)} models with multiple CV methods...")
    
    # Run comparative validation
    try:
        comparison_df = cv_framework.compare_models(
            models=test_models,
            X=X_clean,
            y=y_clean
        )
        
        print(f"\n   📊 CROSS-VALIDATION RESULTS")
        print(f"   " + "=" * 50)
        
        if not comparison_df.empty:
            # Show key metrics
            for _, row in comparison_df.iterrows():
                model_name = row['model']
                print(f"\n   🤖 {model_name.upper()}:")
                
                # Consensus metrics
                if 'r2_mean' in row and not pd.isna(row['r2_mean']):
                    print(f"      R² Score: {row['r2_mean']:.4f} (±{row.get('r2_std', 0):.4f})")
                
                if 'mse_mean' in row and not pd.isna(row['mse_mean']):
                    print(f"      MSE: {row['mse_mean']:.6f} (±{row.get('mse_std', 0):.6f})")
                
                if 'sharpe_mean' in row and not pd.isna(row['sharpe_mean']):
                    print(f"      Sharpe: {row['sharpe_mean']:.4f} (±{row.get('sharpe_std', 0):.4f})")
        
        # Find best model
        if 'r2_mean' in comparison_df.columns:
            best_model_idx = comparison_df['r2_mean'].idxmax()
            if not pd.isna(best_model_idx):
                best_model = comparison_df.loc[best_model_idx, 'model']
                best_r2 = comparison_df.loc[best_model_idx, 'r2_mean']
                print(f"\n   🏆 Best model: {best_model} (R² = {best_r2:.4f})")
        
        return comparison_df
        
    except Exception as e:
        print(f"   ❌ Cross-validation failed: {e}")
        return None


def demonstrate_integrated_pipeline():
    """Demonstrate complete integrated pipeline."""
    print("\n🔄 INTEGRATED PIPELINE DEMO")
    print("=" * 60)
    
    # Step 1: Dataset combination
    combined_data, combiner = demonstrate_dataset_combination()
    
    if combined_data is None:
        print("   ❌ Pipeline terminated: No combined data")
        return
    
    # Step 2: Feature selection
    selected_features, target = demonstrate_feature_selection(combined_data)
    
    if selected_features is None:
        print("   ❌ Pipeline terminated: Feature selection failed")
        return
    
    # Step 3: Overfitting prevention
    overfitting_analysis = demonstrate_overfitting_prevention(selected_features, target)
    
    # Step 4: Time series CV
    cv_results = demonstrate_time_series_cv(selected_features, target)
    
    # Step 5: Pipeline summary
    print(f"\n🎯 PIPELINE SUMMARY")
    print(f"=" * 60)
    print(f"✅ Dataset combination: {len(combined_data)} samples, {len(combined_data.columns)} features")
    print(f"✅ Feature selection: {len(selected_features.columns)} features selected")
    print(f"✅ Overfitting analysis: {len(overfitting_analysis) if overfitting_analysis else 0} models analyzed")
    print(f"✅ Cross-validation: {'Completed' if cv_results is not None else 'Failed'}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print(f"=" * 30)
    
    if combiner and hasattr(combiner, 'combination_stats'):
        data_coverage = combiner.combination_stats.get('data_coverage', 0)
        if data_coverage < 0.8:
            print(f"⚠️ Data coverage is {data_coverage:.1%} - consider improving data quality")
        else:
            print(f"✅ Good data coverage: {data_coverage:.1%}")
    
    if overfitting_analysis:
        high_risk_models = [name for name, result in overfitting_analysis.items() 
                           if 'overfitting' in result and 
                           result['overfitting']['assessment']['risk_level'] == 'HIGH']
        if high_risk_models:
            print(f"🚨 High overfitting risk models: {', '.join(high_risk_models)}")
        else:
            print(f"✅ No high overfitting risk detected")
    
    print(f"\n🎉 Integrated pipeline demonstration complete!")


def _generate_synthetic_datasets(start_date: str, end_date: str) -> dict:
    """Generate synthetic datasets for demonstration."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Business days
    
    np.random.seed(42)
    
    datasets = {}
    
    # Market data
    market_data = pd.DataFrame(index=dates)
    market_data['close'] = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
    market_data['volume'] = np.random.exponential(1000000, len(dates))
    market_data['volatility'] = np.random.lognormal(-2, 0.5, len(dates))
    datasets['market'] = market_data
    
    # Economic data
    econ_data = pd.DataFrame(index=dates)
    econ_data['interest_rate'] = 2.0 + np.cumsum(np.random.normal(0, 0.01, len(dates)))
    econ_data['inflation'] = 3.0 + np.cumsum(np.random.normal(0, 0.005, len(dates)))
    econ_data['unemployment'] = 5.0 + np.cumsum(np.random.normal(0, 0.02, len(dates)))
    datasets['economic'] = econ_data
    
    # Crypto data
    crypto_data = pd.DataFrame(index=dates)
    crypto_data['btc_price'] = 40000 * np.cumprod(1 + np.random.normal(0.001, 0.04, len(dates)))
    crypto_data['btc_volume'] = np.random.exponential(2000000000, len(dates))
    datasets['crypto'] = crypto_data
    
    return datasets


def main():
    """Run comprehensive dataset combination and validation demonstration."""
    print("🌐 RTRADEZ DATASET COMBINATION & VALIDATION DEMO")
    print("=" * 80)
    print("Demonstrating seamless dataset combination and overfitting prevention...")
    print("=" * 80)
    
    try:
        # Run integrated pipeline demonstration
        demonstrate_integrated_pipeline()
        
        print(f"\n✨ DEMO COMPLETE")
        print(f"=" * 80)
        print(f"🎯 Key Features Demonstrated:")
        print(f"   • Multi-source dataset combination with intelligent alignment")
        print(f"   • Advanced feature selection and dimensionality reduction")
        print(f"   • Comprehensive overfitting and underfitting detection")
        print(f"   • Multiple time series cross-validation techniques")
        print(f"   • Integrated pipeline for robust model development")
        print(f"\n🚀 RTradez now provides enterprise-grade data integration")
        print(f"   and model validation capabilities!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()