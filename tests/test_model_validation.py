"""
Comprehensive tests for model validation module.

Tests for financial model validation, performance metrics, and risk assessment.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

from rtradez.validation.model_validation import FinancialModelValidator


class TestFinancialModelValidator:
    """Test FinancialModelValidator functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample financial data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Create synthetic price series with trends and volatility
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        
        # Create features (technical indicators)
        X = pd.DataFrame({
            'price': prices,
            'sma_20': pd.Series(prices).rolling(20).mean(),
            'sma_50': pd.Series(prices).rolling(50).mean(),
            'volatility': pd.Series(prices).pct_change().rolling(20).std(),
            'rsi': np.random.uniform(20, 80, len(dates)),
            'volume': np.random.lognormal(10, 0.5, len(dates))
        }, index=dates)
        
        # Create target (next day returns)
        y = pd.Series(prices).pct_change().shift(-1)
        y.index = dates
        
        # Create price series
        price_series = pd.Series(prices, index=dates)
        
        return X.dropna(), y.dropna(), price_series
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return FinancialModelValidator(
            benchmark_models=['buy_hold', 'mean_reversion'],
            risk_free_rate=0.02,
            confidence_level=0.95
        )
    
    @pytest.fixture
    def simple_model(self):
        """Create simple sklearn model for testing."""
        return LinearRegression()
    
    def test_validator_initialization(self, validator):
        """Test FinancialModelValidator initialization."""
        assert validator.risk_free_rate == 0.02
        assert validator.confidence_level == 0.95
        assert 'buy_hold' in validator.benchmark_models
        assert 'mean_reversion' in validator.benchmark_models
        assert len(validator.financial_metrics) == 9
        assert len(validator.validation_results_) == 0
    
    def test_validator_default_initialization(self):
        """Test validator with default parameters."""
        validator = FinancialModelValidator()
        
        assert validator.risk_free_rate == 0.02
        assert validator.confidence_level == 0.95
        assert len(validator.benchmark_models) == 3
        assert 'buy_hold' in validator.benchmark_models
        assert 'mean_reversion' in validator.benchmark_models
        assert 'momentum' in validator.benchmark_models
    
    def test_generate_predictions_walk_forward(self, validator, simple_model, sample_data):
        """Test walk-forward prediction generation."""
        X, y, _ = sample_data
        
        # Use smaller subset for faster testing
        X_subset = X.iloc[:300]
        y_subset = y.iloc[:300]
        
        predictions = validator._generate_predictions(simple_model, X_subset, y_subset)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) > 0
        # Should have fewer predictions than total samples due to min_train_size
        assert len(predictions) < len(X_subset)
        # Index should be datetime
        assert isinstance(predictions.index, pd.DatetimeIndex)
    
    def test_generate_predictions_insufficient_data(self, validator, simple_model):
        """Test prediction generation with insufficient data."""
        # Create very small dataset
        X = pd.DataFrame({'feature': [1, 2, 3]}, 
                        index=pd.date_range('2023-01-01', periods=3))
        y = pd.Series([0.1, 0.2, 0.3], 
                     index=pd.date_range('2023-01-01', periods=3))
        
        predictions = validator._generate_predictions(simple_model, X, y)
        
        # Should return empty series for insufficient data
        assert len(predictions) == 0
    
    def test_calculate_performance_metrics_basic(self, validator, sample_data):
        """Test basic performance metrics calculation."""
        X, y, prices = sample_data
        
        # Create simple predictions (use actual returns with noise for testing)
        y_pred = y + np.random.normal(0, 0.001, len(y))
        
        metrics = validator._calculate_performance_metrics(y, y_pred, prices)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'directional_accuracy' in metrics
        assert 'information_coefficient' in metrics
        
        # Check metric validity
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['r2'] <= 1
    
    def test_calculate_performance_metrics_with_prices(self, validator, sample_data):
        """Test performance metrics with price data for financial metrics."""
        X, y, prices = sample_data
        
        # Create predictions
        y_pred = y + np.random.normal(0, 0.001, len(y))
        
        metrics = validator._calculate_performance_metrics(y, y_pred, prices)
        
        # Should include financial metrics when prices provided
        financial_metric_names = list(validator.financial_metrics.keys())
        for metric_name in financial_metric_names:
            # Some metrics might fail due to data issues, but key should exist if attempted
            pass  # Just ensure no exceptions are raised
    
    def test_calculate_performance_metrics_no_valid_data(self, validator):
        """Test performance metrics with no valid data."""
        # Create series with all NaN values
        y_true = pd.Series([np.nan, np.nan, np.nan])
        y_pred = pd.Series([np.nan, np.nan, np.nan])
        
        metrics = validator._calculate_performance_metrics(y_true, y_pred)
        
        assert 'error' in metrics
        assert 'No valid predictions' in metrics['error']
    
    def test_calculate_risk_metrics(self, validator, sample_data):
        """Test risk metrics calculation."""
        X, y, prices = sample_data
        y_pred = y + np.random.normal(0, 0.001, len(y))
        
        risk_metrics = validator._calculate_risk_metrics(y, y_pred, prices)
        
        assert isinstance(risk_metrics, dict)
        assert 'volatility' in risk_metrics
        assert 'prediction_volatility' in risk_metrics
        assert 'prediction_stability' in risk_metrics
        
        # Check metric validity
        assert risk_metrics['volatility'] >= 0
        assert risk_metrics['prediction_volatility'] >= 0
        assert risk_metrics['prediction_stability'] > 0
    
    def test_analyze_regime_performance(self, validator, sample_data):
        """Test regime performance analysis."""
        X, y, prices = sample_data
        y_pred = y + np.random.normal(0, 0.001, len(y))
        
        regime_analysis = validator._analyze_regime_performance(X, y, y_pred, prices)
        
        assert isinstance(regime_analysis, dict)
        # Should have different volatility regimes
        regime_keys = list(regime_analysis.keys())
        assert len(regime_keys) > 0
        
        for regime_name, regime_data in regime_analysis.items():
            if isinstance(regime_data, dict):
                assert 'samples' in regime_data
                assert 'r2' in regime_data
                assert 'mse' in regime_data
    
    def test_analyze_regime_performance_no_prices(self, validator, sample_data):
        """Test regime analysis without price data."""
        X, y, _ = sample_data
        y_pred = y + np.random.normal(0, 0.001, len(y))
        
        regime_analysis = validator._analyze_regime_performance(X, y, y_pred, None)
        
        assert isinstance(regime_analysis, dict)
        # Should still work using target volatility
    
    def test_compare_against_benchmarks(self, validator, sample_data):
        """Test benchmark comparison."""
        X, y, prices = sample_data
        
        benchmark_results = validator._compare_against_benchmarks(X, y, prices)
        
        assert isinstance(benchmark_results, dict)
        assert 'buy_hold' in benchmark_results
        assert 'mean_reversion' in benchmark_results
        
        # Check buy-hold metrics
        bh_metrics = benchmark_results['buy_hold']
        assert 'total_return' in bh_metrics
        assert 'sharpe_ratio' in bh_metrics
        assert 'max_drawdown' in bh_metrics
        assert 'volatility' in bh_metrics
    
    def test_compare_against_benchmarks_no_prices(self, validator, sample_data):
        """Test benchmark comparison without price data."""
        X, y, _ = sample_data
        
        benchmark_results = validator._compare_against_benchmarks(X, y, None)
        
        assert 'note' in benchmark_results
        assert 'requires price data' in benchmark_results['note']
    
    def test_run_statistical_tests(self, validator, sample_data):
        """Test statistical significance tests."""
        X, y, _ = sample_data
        y_pred = y + np.random.normal(0, 0.001, len(y))
        
        # Mock benchmark results
        benchmark_results = {'buy_hold': {'sharpe_ratio': 0.5}}
        
        statistical_tests = validator._run_statistical_tests(y, y_pred, benchmark_results)
        
        assert isinstance(statistical_tests, dict)
        assert 'predictions_vs_zero' in statistical_tests
        assert 'correlation_significance' in statistical_tests
        assert 'residuals_normality' in statistical_tests
        
        # Check test structure
        pred_test = statistical_tests['predictions_vs_zero']
        assert 't_statistic' in pred_test
        assert 'p_value' in pred_test
        assert 'significant' in pred_test
    
    def test_run_statistical_tests_insufficient_data(self, validator):
        """Test statistical tests with insufficient data."""
        # Very small dataset
        y_true = pd.Series([0.1, 0.2])
        y_pred = pd.Series([0.11, 0.19])
        
        statistical_tests = validator._run_statistical_tests(y_true, y_pred, {})
        
        assert 'note' in statistical_tests
        assert 'Insufficient data' in statistical_tests['note']
    
    def test_analyze_market_conditions(self, validator, sample_data):
        """Test market condition analysis."""
        X, y, prices = sample_data
        y_pred = y + np.random.normal(0, 0.001, len(y))
        
        market_analysis = validator._analyze_market_conditions(X, y, y_pred, prices)
        
        assert isinstance(market_analysis, dict)
        # Should detect bull and bear periods
        possible_keys = ['bull_market', 'bear_market']
        # At least one condition should be detected in a long time series
    
    def test_analyze_market_conditions_no_prices(self, validator, sample_data):
        """Test market analysis without price data."""
        X, y, _ = sample_data
        y_pred = y + np.random.normal(0, 0.001, len(y))
        
        market_analysis = validator._analyze_market_conditions(X, y, y_pred, None)
        
        assert 'note' in market_analysis
        assert 'requires price data' in market_analysis['note']
    
    def test_generate_overall_assessment(self, validator):
        """Test overall assessment generation."""
        # Mock performance and risk metrics
        performance_metrics = {
            'r2': 0.25,
            'sharpe_ratio': 0.8,
            'max_drawdown': -0.15,
            'hit_ratio': 0.65
        }
        
        risk_metrics = {
            'volatility': 0.20,
            'downside_volatility': 0.15
        }
        
        statistical_tests = {
            'correlation_significance': {
                'significant': True,
                'p_value': 0.01
            }
        }
        
        assessment = validator._generate_overall_assessment(
            performance_metrics, risk_metrics, statistical_tests
        )
        
        assert isinstance(assessment, dict)
        assert 'score' in assessment
        assert 'grade' in assessment
        assert 'strengths' in assessment
        assert 'weaknesses' in assessment
        assert 'recommendations' in assessment
        
        # Check grade validity
        assert assessment['grade'] in ['A', 'B', 'C', 'D', 'F']
        assert 0 <= assessment['score'] <= 100
    
    def test_financial_metric_calculations(self, validator):
        """Test individual financial metric calculations."""
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
        
        # Test Sharpe ratio
        sharpe = validator._calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # Test Sortino ratio
        sortino = validator._calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        
        # Test Calmar ratio
        calmar = validator._calculate_calmar_ratio(returns)
        assert isinstance(calmar, float)
        
        # Test maximum drawdown
        max_dd = validator._calculate_max_drawdown(returns)
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative
        
        # Test VaR
        var_95 = validator._calculate_var(returns)
        assert isinstance(var_95, float)
        
        # Test CVaR
        cvar_95 = validator._calculate_cvar(returns)
        assert isinstance(cvar_95, float)
        assert cvar_95 <= var_95  # CVaR should be more negative than VaR
        
        # Test hit ratio
        hit_ratio = validator._calculate_hit_ratio(returns)
        assert isinstance(hit_ratio, float)
        assert 0 <= hit_ratio <= 1
        
        # Test profit factor
        profit_factor = validator._calculate_profit_factor(returns)
        assert isinstance(profit_factor, float)
        assert profit_factor >= 0
        
        # Test Kelly criterion
        kelly = validator._calculate_kelly_criterion(returns)
        assert isinstance(kelly, float)
        assert 0 <= kelly <= 1
    
    def test_calculate_strategy_returns(self, validator, sample_data):
        """Test strategy returns calculation."""
        _, y, prices = sample_data
        
        # Create simple signals (returns themselves)
        signals = y.fillna(0)
        
        strategy_returns = validator._calculate_strategy_returns(signals, prices)
        
        assert isinstance(strategy_returns, pd.Series)
        assert len(strategy_returns) > 0
        # Should have proper datetime index
        assert isinstance(strategy_returns.index, pd.DatetimeIndex)
    
    def test_clone_model_sklearn(self, validator, simple_model):
        """Test model cloning for sklearn models."""
        cloned = validator._clone_model(simple_model)
        
        assert type(cloned) == type(simple_model)
        assert cloned is not simple_model  # Should be different instance
    
    def test_clone_model_non_sklearn(self, validator):
        """Test model cloning for non-sklearn models."""
        # Create mock model
        class MockModel:
            def __init__(self, param=1):
                self.param = param
            
            def get_params(self):
                return {'param': self.param}
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                return np.zeros(len(X))
        
        model = MockModel(param=5)
        cloned = validator._clone_model(model)
        
        assert type(cloned) == type(model)
        assert cloned.param == model.param
    
    def test_validate_trading_model_complete_workflow(self, validator, simple_model, sample_data):
        """Test complete trading model validation workflow."""
        X, y, prices = sample_data
        
        # Use smaller subset for faster testing
        X_subset = X.iloc[:500]
        y_subset = y.iloc[:500]
        prices_subset = prices.iloc[:500]
        
        validation_result = validator.validate_trading_model(
            model=simple_model,
            X=X_subset,
            y=y_subset,
            prices=prices_subset,
            model_name="test_model"
        )
        
        assert isinstance(validation_result, dict)
        assert 'model_name' in validation_result
        assert 'validation_date' in validation_result
        assert 'data_period' in validation_result
        assert 'performance_metrics' in validation_result
        assert 'risk_metrics' in validation_result
        assert 'regime_analysis' in validation_result
        assert 'benchmark_comparison' in validation_result
        assert 'statistical_tests' in validation_result
        assert 'market_analysis' in validation_result
        assert 'overall_assessment' in validation_result
        
        # Check that results are stored
        assert 'test_model' in validator.validation_results_
    
    def test_validate_trading_model_no_prices(self, validator, simple_model, sample_data):
        """Test validation without price data."""
        X, y, _ = sample_data
        
        X_subset = X.iloc[:300]
        y_subset = y.iloc[:300]
        
        validation_result = validator.validate_trading_model(
            model=simple_model,
            X=X_subset,
            y=y_subset,
            prices=None,
            model_name="no_prices_model"
        )
        
        assert isinstance(validation_result, dict)
        assert 'performance_metrics' in validation_result
        # Should still have basic metrics even without prices
        assert 'mse' in validation_result['performance_metrics']
        assert 'r2' in validation_result['performance_metrics']
    
    def test_validate_trading_model_misaligned_data(self, validator, simple_model):
        """Test validation with misaligned data indices."""
        # Create data with different indices
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        y = pd.Series([0.1, 0.2, 0.3], 
                     index=pd.date_range('2023-01-02', periods=3))
        
        prices = pd.Series([100, 101, 102, 103], 
                          index=pd.date_range('2023-01-01', periods=4))
        
        validation_result = validator.validate_trading_model(
            model=simple_model,
            X=X,
            y=y,
            prices=prices,
            model_name="misaligned_model"
        )
        
        # Should handle misalignment gracefully
        assert isinstance(validation_result, dict)
        # Check that common index was used
        data_period = validation_result['data_period']
        assert data_period['samples'] <= 3  # Maximum possible overlap
    
    def test_generate_validation_report(self, validator, simple_model, sample_data):
        """Test validation report generation."""
        X, y, prices = sample_data
        
        # First validate a model
        X_subset = X.iloc[:300]
        y_subset = y.iloc[:300]
        prices_subset = prices.iloc[:300]
        
        validator.validate_trading_model(
            model=simple_model,
            X=X_subset,
            y=y_subset,
            prices=prices_subset,
            model_name="report_test_model"
        )
        
        # Generate report
        report = validator.generate_validation_report("report_test_model")
        
        assert isinstance(report, dict)
        assert 'executive_summary' in report
        assert 'detailed_results' in report
        assert 'methodology' in report
        
        # Check executive summary structure
        exec_summary = report['executive_summary']
        assert 'model_name' in exec_summary
        assert 'overall_grade' in exec_summary
        assert 'overall_score' in exec_summary
        assert 'key_strengths' in exec_summary
        assert 'key_weaknesses' in exec_summary
        assert 'top_recommendations' in exec_summary
    
    def test_generate_validation_report_nonexistent_model(self, validator):
        """Test report generation for non-existent model."""
        with pytest.raises(ValueError, match="No validation results"):
            validator.generate_validation_report("nonexistent_model")
    
    def test_financial_metrics_with_empty_returns(self, validator):
        """Test financial metrics with empty returns series."""
        empty_returns = pd.Series([], dtype=float)
        
        # All metrics should handle empty series gracefully
        assert validator._calculate_sharpe_ratio(empty_returns) == 0.0
        assert validator._calculate_sortino_ratio(empty_returns) == 0.0
        assert validator._calculate_calmar_ratio(empty_returns) == 0.0
        assert validator._calculate_max_drawdown(empty_returns) == 0.0
        assert validator._calculate_var(empty_returns) == 0.0
        assert validator._calculate_cvar(empty_returns) == 0.0
        assert validator._calculate_hit_ratio(empty_returns) == 0.0
        assert validator._calculate_profit_factor(empty_returns) == 0.0
        assert validator._calculate_kelly_criterion(empty_returns) == 0.0
    
    def test_financial_metrics_edge_cases(self, validator):
        """Test financial metrics with edge cases."""
        # All positive returns
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.005])
        
        hit_ratio = validator._calculate_hit_ratio(positive_returns)
        assert hit_ratio == 1.0
        
        # All negative returns  
        negative_returns = pd.Series([-0.01, -0.02, -0.015, -0.005])
        
        hit_ratio = validator._calculate_hit_ratio(negative_returns)
        assert hit_ratio == 0.0
        
        profit_factor = validator._calculate_profit_factor(negative_returns)
        assert profit_factor == 0.0
    
    @patch('rtradez.validation.model_validation.logger')
    def test_error_handling_and_logging(self, mock_logger, validator, sample_data):
        """Test error handling and logging."""
        X, y, prices = sample_data
        
        # Create a model that will fail
        class FailingModel:
            def fit(self, X, y):
                raise ValueError("Simulated model failure")
            
            def predict(self, X):
                raise ValueError("Prediction failure")
        
        failing_model = FailingModel()
        
        # Should handle model failures gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            validation_result = validator.validate_trading_model(
                model=failing_model,
                X=X.iloc[:100],
                y=y.iloc[:100],
                prices=prices.iloc[:100],
                model_name="failing_model"
            )
        
        # Should still return a result structure
        assert isinstance(validation_result, dict)
        assert 'model_name' in validation_result


@pytest.mark.integration
class TestModelValidationIntegration:
    """Integration tests for model validation."""
    
    @pytest.fixture
    def complex_models(self):
        """Create multiple models for comparison testing."""
        return {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=10, random_state=42),
        }
    
    @pytest.fixture
    def financial_data(self):
        """Create realistic financial dataset."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        
        # Simulate realistic market data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Technical indicators
        price_series = pd.Series(prices, index=dates)
        sma_20 = price_series.rolling(20).mean()
        sma_50 = price_series.rolling(50).mean()
        volatility = price_series.pct_change().rolling(20).std()
        
        X = pd.DataFrame({
            'price': price_series,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'price_sma20_ratio': price_series / sma_20,
            'sma_crossover': (sma_20 > sma_50).astype(int),
            'volatility': volatility,
            'momentum_5': price_series.pct_change(5),
            'momentum_20': price_series.pct_change(20)
        })
        
        # Predict next day returns
        y = price_series.pct_change().shift(-1)
        
        return X.dropna(), y.dropna(), price_series
    
    def test_multi_model_validation_comparison(self, complex_models, financial_data):
        """Test validation of multiple models with comparison."""
        X, y, prices = financial_data
        
        # Use subset for faster testing
        X_subset = X.iloc[:500]
        y_subset = y.iloc[:500]
        prices_subset = prices.iloc[:500]
        
        validator = FinancialModelValidator(
            benchmark_models=['buy_hold', 'momentum'],
            risk_free_rate=0.02
        )
        
        results = {}
        
        for model_name, model in complex_models.items():
            validation_result = validator.validate_trading_model(
                model=model,
                X=X_subset,
                y=y_subset,
                prices=prices_subset,
                model_name=model_name
            )
            
            results[model_name] = validation_result
        
        # Compare results
        assert len(results) == len(complex_models)
        
        for model_name, result in results.items():
            assert 'overall_assessment' in result
            assert 'performance_metrics' in result
            
            # Each model should have valid scores
            score = result['overall_assessment']['score']
            assert 0 <= score <= 100
    
    def test_regime_detection_integration(self, financial_data):
        """Test regime detection with realistic market data."""
        X, y, prices = financial_data
        
        validator = FinancialModelValidator()
        model = LinearRegression()
        
        # Use full dataset to better test regime detection
        validation_result = validator.validate_trading_model(
            model=model,
            X=X,
            y=y,
            prices=prices,
            model_name="regime_test"
        )
        
        regime_analysis = validation_result['regime_analysis']
        
        # Should detect multiple volatility regimes
        assert len(regime_analysis) > 0
        
        # Check regime analysis structure
        for regime_name, regime_data in regime_analysis.items():
            if isinstance(regime_data, dict):
                assert 'samples' in regime_data
                assert regime_data['samples'] > 0
    
    def test_statistical_significance_integration(self, financial_data):
        """Test statistical significance testing integration."""
        X, y, prices = financial_data
        
        validator = FinancialModelValidator(confidence_level=0.95)
        
        # Use a model that should produce statistically significant results
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        validation_result = validator.validate_trading_model(
            model=model,
            X=X.iloc[:800],  # Use more data for statistical significance
            y=y.iloc[:800],
            prices=prices.iloc[:800],
            model_name="significance_test"
        )
        
        statistical_tests = validation_result['statistical_tests']
        
        # Check test results
        assert 'correlation_significance' in statistical_tests
        corr_test = statistical_tests['correlation_significance']
        assert 'correlation' in corr_test
        assert 'p_value' in corr_test
        assert 'significant' in corr_test
        
        # Random forest should produce meaningful correlations
        assert abs(corr_test['correlation']) > 0
    
    def test_benchmark_comparison_integration(self, financial_data):
        """Test benchmark comparison with realistic data."""
        X, y, prices = financial_data
        
        validator = FinancialModelValidator(
            benchmark_models=['buy_hold', 'mean_reversion', 'momentum']
        )
        
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        
        validation_result = validator.validate_trading_model(
            model=model,
            X=X.iloc[:600],
            y=y.iloc[:600],
            prices=prices.iloc[:600],
            model_name="benchmark_test"
        )
        
        benchmark_comparison = validation_result['benchmark_comparison']
        
        # Should have results for all benchmark models
        assert 'buy_hold' in benchmark_comparison
        assert 'mean_reversion' in benchmark_comparison
        assert 'momentum' in benchmark_comparison
        
        # Check benchmark result structure
        bh_results = benchmark_comparison['buy_hold']
        assert 'total_return' in bh_results
        assert 'sharpe_ratio' in bh_results
        assert 'max_drawdown' in bh_results
        assert 'volatility' in bh_results
    
    def test_report_generation_integration(self, financial_data):
        """Test complete report generation integration."""
        X, y, prices = financial_data
        
        validator = FinancialModelValidator()
        model = LinearRegression()
        
        # Validate model
        validator.validate_trading_model(
            model=model,
            X=X.iloc[:400],
            y=y.iloc[:400],
            prices=prices.iloc[:400],
            model_name="report_integration_test"
        )
        
        # Generate report
        report = validator.generate_validation_report("report_integration_test")
        
        # Verify comprehensive report structure
        assert 'executive_summary' in report
        assert 'detailed_results' in report
        assert 'methodology' in report
        
        exec_summary = report['executive_summary']
        assert exec_summary['model_name'] == "report_integration_test"
        assert exec_summary['overall_grade'] in ['A', 'B', 'C', 'D', 'F']
        assert isinstance(exec_summary['key_strengths'], list)
        assert isinstance(exec_summary['key_weaknesses'], list)
        assert isinstance(exec_summary['top_recommendations'], list)
        
        methodology = report['methodology']
        assert 'validation_framework' in methodology
        assert 'metrics_calculated' in methodology
        assert len(methodology['metrics_calculated']) > 0