"""Comprehensive model validation framework for financial time series."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FinancialModelValidator:
    """
    Comprehensive validation framework for financial time series models.
    
    Features:
    - Performance metrics specific to finance
    - Regime-aware validation
    - Risk-adjusted performance measures
    - Statistical significance testing
    - Benchmark comparisons
    """
    
    def __init__(self,
                 benchmark_models: Optional[List[str]] = None,
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95):
        """
        Initialize financial model validator.
        
        Args:
            benchmark_models: List of benchmark model names
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
            confidence_level: Confidence level for statistical tests
        """
        self.benchmark_models = benchmark_models or ['buy_hold', 'mean_reversion', 'momentum']
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        
        # Results storage
        self.validation_results_ = {}
        self.benchmark_results_ = {}
        self.statistical_tests_ = {}
        
        # Financial metrics registry
        self.financial_metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'sortino_ratio': self._calculate_sortino_ratio,
            'calmar_ratio': self._calculate_calmar_ratio,
            'max_drawdown': self._calculate_max_drawdown,
            'var_95': self._calculate_var,
            'cvar_95': self._calculate_cvar,
            'hit_ratio': self._calculate_hit_ratio,
            'profit_factor': self._calculate_profit_factor,
            'kelly_criterion': self._calculate_kelly_criterion
        }
    
    def validate_trading_model(self,
                             model: Any,
                             X: pd.DataFrame,
                             y: pd.Series,
                             prices: Optional[pd.Series] = None,
                             model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive validation for trading models.
        
        Args:
            model: Trained trading model
            X: Feature matrix
            y: Target variable (returns or signals)
            prices: Price series for strategy simulation
            model_name: Name for reporting
            
        Returns:
            Comprehensive validation results
        """
        logger.info(f"Validating trading model '{model_name}'")
        
        # Align data
        common_index = X.index.intersection(y.index)
        if prices is not None:
            common_index = common_index.intersection(prices.index)
        
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        prices_aligned = prices.loc[common_index] if prices is not None else None
        
        # Generate predictions
        predictions = self._generate_predictions(model, X_aligned, y_aligned)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(
            y_aligned, predictions, prices_aligned
        )
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(
            y_aligned, predictions, prices_aligned
        )
        
        # Regime analysis
        regime_analysis = self._analyze_regime_performance(
            X_aligned, y_aligned, predictions, prices_aligned
        )
        
        # Benchmark comparison
        benchmark_comparison = self._compare_against_benchmarks(
            X_aligned, y_aligned, prices_aligned
        )
        
        # Statistical significance tests
        statistical_tests = self._run_statistical_tests(
            y_aligned, predictions, benchmark_comparison
        )
        
        # Market condition analysis
        market_analysis = self._analyze_market_conditions(
            X_aligned, y_aligned, predictions, prices_aligned
        )
        
        # Compile results
        validation_result = {
            'model_name': model_name,
            'validation_date': datetime.now().isoformat(),
            'data_period': {
                'start': common_index.min().isoformat(),
                'end': common_index.max().isoformat(),
                'samples': len(common_index)
            },
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'regime_analysis': regime_analysis,
            'benchmark_comparison': benchmark_comparison,
            'statistical_tests': statistical_tests,
            'market_analysis': market_analysis,
            'overall_assessment': self._generate_overall_assessment(
                performance_metrics, risk_metrics, statistical_tests
            )
        }
        
        # Store results
        self.validation_results_[model_name] = validation_result
        
        return validation_result
    
    def _generate_predictions(self, 
                            model: Any, 
                            X: pd.DataFrame, 
                            y: pd.Series) -> pd.Series:
        """Generate model predictions using walk-forward approach."""
        predictions = []
        prediction_dates = []
        
        # Use walk-forward validation for realistic predictions
        min_train_size = min(252, len(X) // 3)  # 1 year or 1/3 of data
        
        for i in range(min_train_size, len(X)):
            # Training data up to current point
            train_X = X.iloc[:i]
            train_y = y.iloc[:i]
            
            # Current prediction point
            current_X = X.iloc[i:i+1]
            
            try:
                # Clone and fit model
                model_clone = self._clone_model(model)
                model_clone.fit(train_X, train_y)
                
                # Make prediction
                pred = model_clone.predict(current_X)[0]
                predictions.append(pred)
                prediction_dates.append(X.index[i])
                
            except Exception as e:
                logger.warning(f"Prediction failed at index {i}: {e}")
                predictions.append(np.nan)
                prediction_dates.append(X.index[i])
        
        return pd.Series(predictions, index=prediction_dates)
    
    def _calculate_performance_metrics(self,
                                     y_true: pd.Series,
                                     y_pred: pd.Series,
                                     prices: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        # Align series
        common_idx = y_true.index.intersection(y_pred.index)
        y_true_aligned = y_true.loc[common_idx]
        y_pred_aligned = y_pred.loc[common_idx]
        
        # Remove NaN values
        valid_mask = ~(y_true_aligned.isnull() | y_pred_aligned.isnull())
        y_true_clean = y_true_aligned[valid_mask]
        y_pred_clean = y_pred_aligned[valid_mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid predictions'}
        
        metrics = {}
        
        # Basic regression metrics
        try:
            metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
            metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
        except Exception as e:
            logger.warning(f"Failed to calculate basic metrics: {e}")
        
        # Financial performance metrics
        if prices is not None:
            prices_aligned = prices.loc[common_idx][valid_mask]
            strategy_returns = self._calculate_strategy_returns(
                y_pred_clean, prices_aligned
            )
            
            for metric_name, metric_func in self.financial_metrics.items():
                try:
                    metrics[metric_name] = metric_func(strategy_returns, y_true_clean)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
        
        # Directional accuracy
        try:
            if len(y_true_clean) > 1:
                true_direction = np.sign(y_true_clean.diff().dropna())
                pred_direction = np.sign(y_pred_clean.diff().dropna())
                
                common_dir_idx = true_direction.index.intersection(pred_direction.index)
                if len(common_dir_idx) > 0:
                    metrics['directional_accuracy'] = (
                        true_direction[common_dir_idx] == pred_direction[common_dir_idx]
                    ).mean()
        except Exception as e:
            logger.warning(f"Failed to calculate directional accuracy: {e}")
        
        # Information coefficient
        try:
            ic = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
            metrics['information_coefficient'] = ic if not np.isnan(ic) else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate information coefficient: {e}")
        
        return metrics
    
    def _calculate_risk_metrics(self,
                              y_true: pd.Series,
                              y_pred: pd.Series,
                              prices: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate risk-specific metrics."""
        # Align and clean data
        common_idx = y_true.index.intersection(y_pred.index)
        y_pred_aligned = y_pred.loc[common_idx]
        
        valid_mask = ~y_pred_aligned.isnull()
        y_pred_clean = y_pred_aligned[valid_mask]
        
        if len(y_pred_clean) == 0:
            return {'error': 'No valid predictions'}
        
        risk_metrics = {}
        
        if prices is not None:
            prices_aligned = prices.loc[common_idx][valid_mask]
            strategy_returns = self._calculate_strategy_returns(y_pred_clean, prices_aligned)
            
            # Volatility
            risk_metrics['volatility'] = strategy_returns.std() * np.sqrt(252)
            
            # Downside volatility
            negative_returns = strategy_returns[strategy_returns < 0]
            if len(negative_returns) > 0:
                risk_metrics['downside_volatility'] = negative_returns.std() * np.sqrt(252)
            
            # Skewness and kurtosis
            risk_metrics['skewness'] = stats.skew(strategy_returns.dropna())
            risk_metrics['kurtosis'] = stats.kurtosis(strategy_returns.dropna())
            
            # Beta (if market benchmark available)
            market_returns = self._get_market_returns(strategy_returns.index)
            if market_returns is not None:
                aligned_market = market_returns.loc[strategy_returns.index]
                valid_both = ~(strategy_returns.isnull() | aligned_market.isnull())
                
                if valid_both.sum() > 10:
                    covariance = np.cov(strategy_returns[valid_both], aligned_market[valid_both])[0, 1]
                    market_variance = aligned_market[valid_both].var()
                    risk_metrics['beta'] = covariance / market_variance if market_variance > 0 else 0
        
        # Prediction consistency
        risk_metrics['prediction_volatility'] = y_pred_clean.std()
        
        # Model stability (change in predictions)
        pred_changes = y_pred_clean.diff().dropna()
        if len(pred_changes) > 0:
            risk_metrics['prediction_stability'] = 1 / (1 + pred_changes.std())
        
        return risk_metrics
    
    def _analyze_regime_performance(self,
                                  X: pd.DataFrame,
                                  y_true: pd.Series,
                                  y_pred: pd.Series,
                                  prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyze model performance across different market regimes."""
        # Simple regime detection based on volatility
        if prices is not None:
            returns = prices.pct_change().dropna()
            rolling_vol = returns.rolling(20).std()
            
            # Define regimes
            vol_quantiles = rolling_vol.quantile([0.33, 0.67])
            
            low_vol_mask = rolling_vol <= vol_quantiles.iloc[0]
            high_vol_mask = rolling_vol >= vol_quantiles.iloc[1]
            medium_vol_mask = ~(low_vol_mask | high_vol_mask)
            
            regimes = pd.Series(index=rolling_vol.index, dtype=str)
            regimes[low_vol_mask] = 'low_volatility'
            regimes[medium_vol_mask] = 'medium_volatility'
            regimes[high_vol_mask] = 'high_volatility'
        else:
            # Use target volatility as proxy
            target_vol = y_true.rolling(20).std()
            vol_median = target_vol.median()
            
            regimes = pd.Series(index=y_true.index, dtype=str)
            regimes[target_vol <= vol_median] = 'low_volatility'
            regimes[target_vol > vol_median] = 'high_volatility'
        
        # Calculate performance by regime
        regime_performance = {}
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
                
            regime_mask = regimes == regime
            regime_idx = regimes[regime_mask].index
            
            # Subset data for this regime
            regime_true = y_true.loc[regime_idx.intersection(y_true.index)]
            regime_pred = y_pred.loc[regime_idx.intersection(y_pred.index)]
            
            if len(regime_true) > 10 and len(regime_pred) > 10:
                # Calculate metrics for this regime
                common_regime_idx = regime_true.index.intersection(regime_pred.index)
                
                if len(common_regime_idx) > 0:
                    regime_performance[regime] = {
                        'samples': len(common_regime_idx),
                        'r2': r2_score(regime_true[common_regime_idx], 
                                      regime_pred[common_regime_idx]),
                        'mse': mean_squared_error(regime_true[common_regime_idx], 
                                                regime_pred[common_regime_idx])
                    }
                    
                    # Add financial metrics if prices available
                    if prices is not None:
                        regime_prices = prices.loc[regime_idx.intersection(prices.index)]
                        if len(regime_prices) > 0:
                            regime_strategy_returns = self._calculate_strategy_returns(
                                regime_pred[common_regime_idx], 
                                regime_prices[common_regime_idx]
                            )
                            regime_performance[regime]['sharpe'] = self._calculate_sharpe_ratio(
                                regime_strategy_returns
                            )
        
        return regime_performance
    
    def _compare_against_benchmarks(self,
                                  X: pd.DataFrame,
                                  y_true: pd.Series,
                                  prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Compare model performance against benchmark strategies."""
        if prices is None:
            return {'note': 'Benchmark comparison requires price data'}
        
        benchmark_results = {}
        
        # Buy and hold benchmark
        if 'buy_hold' in self.benchmark_models:
            buy_hold_returns = prices.pct_change().dropna()
            benchmark_results['buy_hold'] = {
                'total_return': (prices.iloc[-1] / prices.iloc[0]) - 1,
                'sharpe_ratio': self._calculate_sharpe_ratio(buy_hold_returns),
                'max_drawdown': self._calculate_max_drawdown(buy_hold_returns),
                'volatility': buy_hold_returns.std() * np.sqrt(252)
            }
        
        # Mean reversion benchmark
        if 'mean_reversion' in self.benchmark_models:
            returns = prices.pct_change().dropna()
            mean_reversion_signals = -returns.rolling(5).mean()  # Contrarian signals
            mr_returns = self._calculate_strategy_returns(
                mean_reversion_signals, prices
            )
            
            benchmark_results['mean_reversion'] = {
                'sharpe_ratio': self._calculate_sharpe_ratio(mr_returns),
                'max_drawdown': self._calculate_max_drawdown(mr_returns),
                'volatility': mr_returns.std() * np.sqrt(252)
            }
        
        # Momentum benchmark
        if 'momentum' in self.benchmark_models:
            returns = prices.pct_change().dropna()
            momentum_signals = returns.rolling(10).mean()  # Trend following
            momentum_returns = self._calculate_strategy_returns(
                momentum_signals, prices
            )
            
            benchmark_results['momentum'] = {
                'sharpe_ratio': self._calculate_sharpe_ratio(momentum_returns),
                'max_drawdown': self._calculate_max_drawdown(momentum_returns),
                'volatility': momentum_returns.std() * np.sqrt(252)
            }
        
        return benchmark_results
    
    def _run_statistical_tests(self,
                             y_true: pd.Series,
                             y_pred: pd.Series,
                             benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run statistical significance tests."""
        # Align data
        common_idx = y_true.index.intersection(y_pred.index)
        y_true_aligned = y_true.loc[common_idx]
        y_pred_aligned = y_pred.loc[common_idx]
        
        valid_mask = ~(y_true_aligned.isnull() | y_pred_aligned.isnull())
        y_true_clean = y_true_aligned[valid_mask]
        y_pred_clean = y_pred_aligned[valid_mask]
        
        if len(y_true_clean) < 30:
            return {'note': 'Insufficient data for statistical tests'}
        
        statistical_tests = {}
        
        # Test if predictions are significantly different from zero
        try:
            t_stat, p_value = stats.ttest_1samp(y_pred_clean, 0)
            statistical_tests['predictions_vs_zero'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < (1 - self.confidence_level)
            }
        except Exception as e:
            logger.warning(f"Failed predictions vs zero test: {e}")
        
        # Test correlation significance
        try:
            correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
            n = len(y_true_clean)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            statistical_tests['correlation_significance'] = {
                'correlation': correlation,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < (1 - self.confidence_level)
            }
        except Exception as e:
            logger.warning(f"Failed correlation significance test: {e}")
        
        # Jarque-Bera test for normality of residuals
        try:
            residuals = y_true_clean - y_pred_clean
            jb_stat, jb_p = stats.jarque_bera(residuals)
            
            statistical_tests['residuals_normality'] = {
                'jarque_bera_statistic': jb_stat,
                'p_value': jb_p,
                'residuals_normal': jb_p > (1 - self.confidence_level)
            }
        except Exception as e:
            logger.warning(f"Failed Jarque-Bera test: {e}")
        
        return statistical_tests
    
    def _analyze_market_conditions(self,
                                 X: pd.DataFrame,
                                 y_true: pd.Series,
                                 y_pred: pd.Series,
                                 prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyze performance under different market conditions."""
        if prices is None:
            return {'note': 'Market condition analysis requires price data'}
        
        analysis = {}
        
        # Bull vs bear markets
        returns = prices.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        
        # Simple bull/bear detection: above/below 200-day MA
        ma_200 = prices.rolling(200).mean()
        bull_mask = prices > ma_200
        bear_mask = prices <= ma_200
        
        # Performance in bull markets
        bull_periods = y_pred.index[y_pred.index.isin(bull_mask[bull_mask].index)]
        if len(bull_periods) > 10:
            bull_pred = y_pred.loc[bull_periods]
            bull_true = y_true.loc[bull_periods.intersection(y_true.index)]
            
            if len(bull_true) > 0:
                common_bull = bull_pred.index.intersection(bull_true.index)
                if len(common_bull) > 0:
                    analysis['bull_market'] = {
                        'periods': len(common_bull),
                        'r2': r2_score(bull_true[common_bull], bull_pred[common_bull]),
                        'correlation': np.corrcoef(bull_true[common_bull], bull_pred[common_bull])[0, 1]
                    }
        
        # Performance in bear markets
        bear_periods = y_pred.index[y_pred.index.isin(bear_mask[bear_mask].index)]
        if len(bear_periods) > 10:
            bear_pred = y_pred.loc[bear_periods]
            bear_true = y_true.loc[bear_periods.intersection(y_true.index)]
            
            if len(bear_true) > 0:
                common_bear = bear_pred.index.intersection(bear_true.index)
                if len(common_bear) > 0:
                    analysis['bear_market'] = {
                        'periods': len(common_bear),
                        'r2': r2_score(bear_true[common_bear], bear_pred[common_bear]),
                        'correlation': np.corrcoef(bear_true[common_bear], bear_pred[common_bear])[0, 1]
                    }
        
        return analysis
    
    def _generate_overall_assessment(self,
                                   performance_metrics: Dict[str, float],
                                   risk_metrics: Dict[str, float],
                                   statistical_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall model assessment."""
        assessment = {
            'score': 0.0,
            'grade': 'F',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        score = 0.0
        
        # Performance assessment
        r2 = performance_metrics.get('r2', 0)
        if r2 > 0.3:
            score += 25
            assessment['strengths'].append(f"Good predictive power (R² = {r2:.3f})")
        elif r2 > 0.1:
            score += 15
            assessment['strengths'].append(f"Moderate predictive power (R² = {r2:.3f})")
        else:
            assessment['weaknesses'].append(f"Low predictive power (R² = {r2:.3f})")
        
        # Risk assessment
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        if sharpe > 1.0:
            score += 25
            assessment['strengths'].append(f"Excellent risk-adjusted returns (Sharpe = {sharpe:.3f})")
        elif sharpe > 0.5:
            score += 15
            assessment['strengths'].append(f"Good risk-adjusted returns (Sharpe = {sharpe:.3f})")
        elif sharpe > 0:
            score += 5
            assessment['strengths'].append(f"Positive risk-adjusted returns (Sharpe = {sharpe:.3f})")
        else:
            assessment['weaknesses'].append(f"Poor risk-adjusted returns (Sharpe = {sharpe:.3f})")
        
        # Statistical significance
        if statistical_tests.get('correlation_significance', {}).get('significant', False):
            score += 20
            assessment['strengths'].append("Statistically significant predictions")
        else:
            assessment['weaknesses'].append("Predictions not statistically significant")
        
        # Stability assessment
        max_dd = performance_metrics.get('max_drawdown', 0)
        if abs(max_dd) < 0.1:
            score += 15
            assessment['strengths'].append("Low maximum drawdown")
        elif abs(max_dd) < 0.2:
            score += 10
        else:
            assessment['weaknesses'].append(f"High maximum drawdown ({abs(max_dd):.1%})")
        
        # Additional assessments
        hit_ratio = performance_metrics.get('hit_ratio', 0.5)
        if hit_ratio > 0.6:
            score += 15
            assessment['strengths'].append(f"Good directional accuracy ({hit_ratio:.1%})")
        
        # Assign grade
        assessment['score'] = score
        
        if score >= 80:
            assessment['grade'] = 'A'
        elif score >= 70:
            assessment['grade'] = 'B'
        elif score >= 60:
            assessment['grade'] = 'C'
        elif score >= 50:
            assessment['grade'] = 'D'
        else:
            assessment['grade'] = 'F'
        
        # Generate recommendations
        if r2 < 0.1:
            assessment['recommendations'].append("Improve feature engineering or try more complex models")
        
        if sharpe < 0.5:
            assessment['recommendations'].append("Focus on risk management and position sizing")
        
        if abs(max_dd) > 0.2:
            assessment['recommendations'].append("Implement stop-loss or volatility-based position sizing")
        
        if not statistical_tests.get('correlation_significance', {}).get('significant', False):
            assessment['recommendations'].append("Increase sample size or improve signal quality")
        
        return assessment
    
    # Financial metric calculation methods
    def _calculate_strategy_returns(self, 
                                  signals: pd.Series, 
                                  prices: pd.Series) -> pd.Series:
        """Calculate strategy returns from signals and prices."""
        # Align signals and prices
        common_idx = signals.index.intersection(prices.index)
        signals_aligned = signals.loc[common_idx]
        prices_aligned = prices.loc[common_idx]
        
        # Calculate returns
        price_returns = prices_aligned.pct_change().dropna()
        
        # Align signals with returns (shift signals to avoid look-ahead bias)
        signals_shifted = signals_aligned.shift(1).dropna()
        
        # Calculate strategy returns
        common_return_idx = price_returns.index.intersection(signals_shifted.index)
        strategy_returns = price_returns[common_return_idx] * signals_shifted[common_return_idx]
        
        return strategy_returns.dropna()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd = self._calculate_max_drawdown(returns)
        
        return annual_return / abs(max_dd) if max_dd != 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_var(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate Value at Risk (95%)."""
        if len(returns) == 0:
            return 0.0
        
        return returns.quantile(0.05)
    
    def _calculate_cvar(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate Conditional Value at Risk (95%)."""
        if len(returns) == 0:
            return 0.0
        
        var_95 = self._calculate_var(returns)
        return returns[returns <= var_95].mean()
    
    def _calculate_hit_ratio(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate hit ratio (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
        
        return (returns > 0).mean()
    
    def _calculate_profit_factor(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate profit factor."""
        if len(returns) == 0:
            return 0.0
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        
        return positive_returns / negative_returns if negative_returns > 0 else 0.0
    
    def _calculate_kelly_criterion(self, returns: pd.Series, y_true: Optional[pd.Series] = None) -> float:
        """Calculate Kelly criterion optimal position size."""
        if len(returns) == 0:
            return 0.0
        
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        return max(0, min(kelly, 1))  # Cap between 0 and 1
    
    def _get_market_returns(self, index: pd.DatetimeIndex) -> Optional[pd.Series]:
        """Get market benchmark returns (placeholder)."""
        # This would typically fetch SPY or other market index returns
        # For now, return None
        return None
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model for validation."""
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback for non-sklearn models
            model_class = type(model)
            if hasattr(model, 'get_params'):
                params = model.get_params()
                return model_class(**params)
            else:
                return model_class()
    
    def generate_validation_report(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if model_name not in self.validation_results_:
            raise ValueError(f"No validation results for model '{model_name}'")
        
        results = self.validation_results_[model_name]
        
        return {
            'executive_summary': {
                'model_name': model_name,
                'overall_grade': results['overall_assessment']['grade'],
                'overall_score': results['overall_assessment']['score'],
                'key_strengths': results['overall_assessment']['strengths'][:3],
                'key_weaknesses': results['overall_assessment']['weaknesses'][:3],
                'top_recommendations': results['overall_assessment']['recommendations'][:3]
            },
            'detailed_results': results,
            'methodology': {
                'validation_framework': 'Financial Model Validator',
                'metrics_calculated': list(self.financial_metrics.keys()),
                'confidence_level': self.confidence_level,
                'risk_free_rate': self.risk_free_rate
            }
        }