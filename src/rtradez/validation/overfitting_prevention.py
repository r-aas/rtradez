"""Overfitting and underfitting prevention for financial time series models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class OverfittingDetector:
    """
    Detect overfitting and underfitting in financial time series models.
    
    Features:
    - Walk-forward validation
    - Learning curves analysis  
    - Complexity vs performance curves
    - Statistical significance testing
    - Early stopping detection
    """
    
    def __init__(self, 
                 min_train_samples: int = 252,  # 1 year
                 validation_horizon: int = 63,   # 3 months
                 step_size: int = 21,           # 1 month
                 significance_level: float = 0.05):
        """
        Initialize overfitting detector.
        
        Args:
            min_train_samples: Minimum training samples required
            validation_horizon: Validation period length
            step_size: Step size for walk-forward validation
            significance_level: Statistical significance level
        """
        self.min_train_samples = min_train_samples
        self.validation_horizon = validation_horizon
        self.step_size = step_size
        self.significance_level = significance_level
        
        # Results storage
        self.validation_results_ = {}
        self.learning_curves_ = {}
        self.complexity_curves_ = {}
        self.diagnostic_plots_ = {}
        
    def detect_overfitting(self, 
                          model: Any, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive overfitting detection analysis.
        
        Args:
            model: Trained model to analyze
            X: Feature matrix
            y: Target variable
            model_name: Name for reporting
            
        Returns:
            Overfitting detection report
        """
        logger.info(f"Analyzing overfitting for {model_name}")
        
        # Align data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # Walk-forward validation
        wf_results = self._walk_forward_validation(model, X_aligned, y_aligned)
        
        # Learning curves
        learning_results = self._analyze_learning_curves(model, X_aligned, y_aligned)
        
        # Model complexity analysis
        complexity_results = self._analyze_model_complexity(X_aligned, y_aligned)
        
        # Statistical tests
        statistical_results = self._statistical_significance_tests(wf_results)
        
        # Overall assessment
        assessment = self._assess_overfitting_risk(
            wf_results, learning_results, complexity_results, statistical_results
        )
        
        # Store results
        self.validation_results_[model_name] = {
            'walk_forward': wf_results,
            'learning_curves': learning_results,
            'complexity_analysis': complexity_results,
            'statistical_tests': statistical_results,
            'assessment': assessment
        }
        
        return self.validation_results_[model_name]
    
    def _walk_forward_validation(self, 
                                model: Any, 
                                X: pd.DataFrame, 
                                y: pd.Series) -> Dict[str, Any]:
        """Perform walk-forward validation to detect overfitting."""
        if len(X) < self.min_train_samples + self.validation_horizon:
            raise ValueError(f"Insufficient data: need at least {self.min_train_samples + self.validation_horizon} samples")
        
        results = {
            'train_scores': [],
            'val_scores': [],
            'dates': [],
            'train_sizes': [],
            'predictions': [],
            'actuals': []
        }
        
        # Walk-forward windows
        start_idx = self.min_train_samples
        
        while start_idx + self.validation_horizon <= len(X):
            # Training data (expanding window)
            train_X = X.iloc[:start_idx]
            train_y = y.iloc[:start_idx]
            
            # Validation data
            val_X = X.iloc[start_idx:start_idx + self.validation_horizon]
            val_y = y.iloc[start_idx:start_idx + self.validation_horizon]
            
            try:
                # Clone and train model
                model_clone = self._clone_model(model)
                model_clone.fit(train_X, train_y)
                
                # Predictions
                train_pred = model_clone.predict(train_X)
                val_pred = model_clone.predict(val_X)
                
                # Scores (using MSE)
                train_score = mean_squared_error(train_y, train_pred)
                val_score = mean_squared_error(val_y, val_pred)
                
                # Store results
                results['train_scores'].append(train_score)
                results['val_scores'].append(val_score)
                results['dates'].append(val_X.index[0])
                results['train_sizes'].append(len(train_X))
                results['predictions'].extend(val_pred)
                results['actuals'].extend(val_y.values)
                
            except Exception as e:
                logger.warning(f"Walk-forward validation failed at step {start_idx}: {e}")
                continue
            
            start_idx += self.step_size
        
        # Calculate summary statistics
        if results['train_scores']:
            results['avg_train_score'] = np.mean(results['train_scores'])
            results['avg_val_score'] = np.mean(results['val_scores'])
            results['overfitting_ratio'] = np.mean(results['val_scores']) / np.mean(results['train_scores'])
            results['score_stability'] = np.std(results['val_scores']) / np.mean(results['val_scores'])
            results['degradation_trend'] = self._calculate_performance_trend(results['val_scores'])
        
        return results
    
    def _analyze_learning_curves(self, 
                               model: Any, 
                               X: pd.DataFrame, 
                               y: pd.Series) -> Dict[str, Any]:
        """Analyze learning curves to detect overfitting patterns."""
        # Generate learning curve data points
        train_sizes = np.linspace(self.min_train_samples, len(X) * 0.8, 10).astype(int)
        
        results = {
            'train_sizes': [],
            'train_scores_mean': [],
            'train_scores_std': [],
            'val_scores_mean': [],
            'val_scores_std': [],
            'convergence_analysis': {}
        }
        
        for train_size in train_sizes:
            if train_size >= len(X):
                continue
                
            # Time series split for this train size
            tscv = TimeSeriesSplit(n_splits=3, test_size=self.validation_horizon)
            
            train_scores = []
            val_scores = []
            
            for train_idx, val_idx in tscv.split(X.iloc[:train_size + self.validation_horizon]):
                try:
                    train_X = X.iloc[train_idx]
                    train_y = y.iloc[train_idx]
                    val_X = X.iloc[val_idx]
                    val_y = y.iloc[val_idx]
                    
                    # Train model
                    model_clone = self._clone_model(model)
                    model_clone.fit(train_X, train_y)
                    
                    # Score
                    train_pred = model_clone.predict(train_X)
                    val_pred = model_clone.predict(val_X)
                    
                    train_scores.append(-mean_squared_error(train_y, train_pred))
                    val_scores.append(-mean_squared_error(val_y, val_pred))
                    
                except Exception as e:
                    logger.warning(f"Learning curve failed at size {train_size}: {e}")
                    continue
            
            if train_scores and val_scores:
                results['train_sizes'].append(train_size)
                results['train_scores_mean'].append(np.mean(train_scores))
                results['train_scores_std'].append(np.std(train_scores))
                results['val_scores_mean'].append(np.mean(val_scores))
                results['val_scores_std'].append(np.std(val_scores))
        
        # Analyze convergence
        if len(results['val_scores_mean']) > 3:
            val_scores = np.array(results['val_scores_mean'])
            
            # Check if validation score is improving
            slope, _, r_value, p_value, _ = stats.linregress(
                range(len(val_scores)), val_scores
            )
            
            results['convergence_analysis'] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'is_converging': slope > 0 and p_value < 0.05,
                'final_gap': results['train_scores_mean'][-1] - results['val_scores_mean'][-1]
            }
        
        return results
    
    def _analyze_model_complexity(self, 
                                X: pd.DataFrame, 
                                y: pd.Series) -> Dict[str, Any]:
        """Analyze how model complexity affects performance."""
        complexity_results = {}
        
        # Test different regularization strengths
        alphas = np.logspace(-4, 1, 20)
        
        for model_type, model_class in [
            ('ridge', Ridge),
            ('lasso', Lasso),
            ('elastic_net', ElasticNet)
        ]:
            train_scores = []
            val_scores = []
            
            for alpha in alphas:
                try:
                    if model_type == 'elastic_net':
                        model = model_class(alpha=alpha, l1_ratio=0.5, random_state=42)
                    else:
                        model = model_class(alpha=alpha, random_state=42)
                    
                    # Cross-validation
                    tscv = TimeSeriesSplit(n_splits=5, test_size=self.validation_horizon)
                    cv_scores = cross_val_score(
                        model, X, y, cv=tscv, scoring='neg_mean_squared_error'
                    )
                    
                    # Fit on full training set for training score
                    train_size = int(len(X) * 0.8)
                    model.fit(X.iloc[:train_size], y.iloc[:train_size])
                    train_pred = model.predict(X.iloc[:train_size])
                    train_score = -mean_squared_error(y.iloc[:train_size], train_pred)
                    
                    train_scores.append(train_score)
                    val_scores.append(cv_scores.mean())
                    
                except Exception as e:
                    logger.warning(f"Complexity analysis failed for {model_type}, alpha={alpha}: {e}")
                    continue
            
            if train_scores and val_scores:
                # Find optimal complexity
                val_scores_array = np.array(val_scores)
                optimal_idx = np.argmax(val_scores_array)
                
                complexity_results[model_type] = {
                    'alphas': alphas.tolist(),
                    'train_scores': train_scores,
                    'val_scores': val_scores,
                    'optimal_alpha': alphas[optimal_idx],
                    'optimal_val_score': val_scores[optimal_idx],
                    'overfitting_at_optimal': train_scores[optimal_idx] - val_scores[optimal_idx]
                }
        
        return complexity_results
    
    def _statistical_significance_tests(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical tests on validation results."""
        if not wf_results['val_scores']:
            return {}
        
        val_scores = np.array(wf_results['val_scores'])
        train_scores = np.array(wf_results['train_scores'])
        
        # Paired t-test between train and validation scores
        t_stat, t_p_value = stats.ttest_rel(train_scores, val_scores)
        
        # Test for trend in validation scores (degradation over time)
        periods = range(len(val_scores))
        trend_slope, _, trend_r, trend_p, _ = stats.linregress(periods, val_scores)
        
        # Normality test on validation scores
        shapiro_stat, shapiro_p = stats.shapiro(val_scores)
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, adf_p, _, _, _, _ = adfuller(val_scores)
            stationarity_test = {'adf_statistic': adf_stat, 'adf_p_value': adf_p}
        except ImportError:
            stationarity_test = {'note': 'statsmodels not available for ADF test'}
        
        return {
            'paired_t_test': {
                'statistic': t_stat,
                'p_value': t_p_value,
                'significant_difference': t_p_value < self.significance_level
            },
            'performance_trend': {
                'slope': trend_slope,
                'r_squared': trend_r**2,
                'p_value': trend_p,
                'significant_degradation': trend_slope < 0 and trend_p < self.significance_level
            },
            'normality_test': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > self.significance_level
            },
            'stationarity_test': stationarity_test
        }
    
    def _assess_overfitting_risk(self, 
                               wf_results: Dict[str, Any],
                               learning_results: Dict[str, Any],
                               complexity_results: Dict[str, Any],
                               statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide overall overfitting risk assessment."""
        risk_factors = []
        risk_score = 0.0
        
        # Check overfitting ratio
        if 'overfitting_ratio' in wf_results:
            ratio = wf_results['overfitting_ratio']
            if ratio > 2.0:
                risk_factors.append("High train/validation score ratio")
                risk_score += 0.3
            elif ratio > 1.5:
                risk_factors.append("Moderate train/validation score ratio")
                risk_score += 0.15
        
        # Check score stability
        if 'score_stability' in wf_results:
            stability = wf_results['score_stability']
            if stability > 0.3:
                risk_factors.append("High validation score volatility")
                risk_score += 0.2
            elif stability > 0.15:
                risk_factors.append("Moderate validation score volatility")
                risk_score += 0.1
        
        # Check performance degradation
        if statistical_results.get('performance_trend', {}).get('significant_degradation', False):
            risk_factors.append("Significant performance degradation over time")
            risk_score += 0.25
        
        # Check learning curve convergence
        if learning_results.get('convergence_analysis', {}).get('is_converging', False):
            risk_score -= 0.1  # Good sign
        else:
            risk_factors.append("Validation score not converging")
            risk_score += 0.15
        
        # Check final gap in learning curves
        final_gap = learning_results.get('convergence_analysis', {}).get('final_gap', 0)
        if final_gap > 0.1:
            risk_factors.append("Large gap between train and validation performance")
            risk_score += 0.2
        
        # Determine risk level
        if risk_score >= 0.5:
            risk_level = "HIGH"
        elif risk_score >= 0.25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self._generate_recommendations(risk_level, risk_factors)
        }
    
    def _generate_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on overfitting assessment."""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Consider reducing model complexity (fewer features, more regularization)",
                "Increase training data if possible",
                "Implement cross-validation with larger validation sets",
                "Use ensemble methods to reduce variance",
                "Consider simpler model architectures"
            ])
        
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Monitor performance on fresh data regularly",
                "Consider adding regularization",
                "Validate model on longer out-of-sample periods",
                "Use feature selection to reduce dimensionality"
            ])
        
        else:  # LOW risk
            recommendations.extend([
                "Current model appears well-generalized",
                "Continue monitoring with regular validation",
                "Consider slightly increasing model complexity if needed"
            ])
        
        # Specific recommendations based on risk factors
        for factor in risk_factors:
            if "ratio" in factor.lower():
                recommendations.append("Apply stronger regularization (L1/L2/Elastic Net)")
            elif "volatility" in factor.lower() or "stability" in factor.lower():
                recommendations.append("Use ensemble methods or bagging to reduce variance")
            elif "degradation" in factor.lower():
                recommendations.append("Implement adaptive learning or concept drift detection")
            elif "converging" in factor.lower():
                recommendations.append("Collect more training data or simplify model")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model for cross-validation."""
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
    
    def _calculate_performance_trend(self, scores: List[float]) -> Dict[str, float]:
        """Calculate trend in performance scores."""
        if len(scores) < 3:
            return {'slope': 0.0, 'r_squared': 0.0}
        
        x = np.arange(len(scores))
        slope, _, r_value, p_value, _ = stats.linregress(x, scores)
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value
        }
    
    def plot_validation_results(self, 
                              model_name: str, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot comprehensive validation results."""
        if model_name not in self.validation_results_:
            raise ValueError(f"No results found for model '{model_name}'")
        
        results = self.validation_results_[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Overfitting Analysis: {model_name}', fontsize=16)
        
        # 1. Walk-forward validation scores
        wf_results = results['walk_forward']
        if wf_results['dates']:
            axes[0, 0].plot(wf_results['dates'], wf_results['train_scores'], 
                           label='Training Score', marker='o')
            axes[0, 0].plot(wf_results['dates'], wf_results['val_scores'], 
                           label='Validation Score', marker='s')
            axes[0, 0].set_title('Walk-Forward Validation Scores')
            axes[0, 0].set_ylabel('Score (MSE)')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Learning curves
        learning_results = results['learning_curves']
        if learning_results['train_sizes']:
            train_sizes = learning_results['train_sizes']
            train_mean = learning_results['train_scores_mean']
            train_std = learning_results['train_scores_std']
            val_mean = learning_results['val_scores_mean']
            val_std = learning_results['val_scores_std']
            
            axes[0, 1].plot(train_sizes, train_mean, label='Training Score', marker='o')
            axes[0, 1].fill_between(train_sizes, 
                                   np.array(train_mean) - np.array(train_std),
                                   np.array(train_mean) + np.array(train_std), 
                                   alpha=0.3)
            
            axes[0, 1].plot(train_sizes, val_mean, label='Validation Score', marker='s')
            axes[0, 1].fill_between(train_sizes,
                                   np.array(val_mean) - np.array(val_std),
                                   np.array(val_mean) + np.array(val_std),
                                   alpha=0.3)
            
            axes[0, 1].set_title('Learning Curves')
            axes[0, 1].set_xlabel('Training Set Size')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
        
        # 3. Complexity curves (Ridge regularization)
        complexity_results = results['complexity_analysis']
        if 'ridge' in complexity_results:
            ridge_data = complexity_results['ridge']
            axes[1, 0].semilogx(ridge_data['alphas'], ridge_data['train_scores'], 
                               label='Training Score', marker='o')
            axes[1, 0].semilogx(ridge_data['alphas'], ridge_data['val_scores'], 
                               label='Validation Score', marker='s')
            axes[1, 0].axvline(ridge_data['optimal_alpha'], color='r', linestyle='--', 
                              label='Optimal Alpha')
            axes[1, 0].set_title('Model Complexity (Ridge Regularization)')
            axes[1, 0].set_xlabel('Alpha (Regularization Strength)')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
        
        # 4. Risk assessment
        assessment = results['assessment']
        risk_text = f"Risk Level: {assessment['risk_level']}\\n"
        risk_text += f"Risk Score: {assessment['risk_score']:.3f}\\n\\n"
        risk_text += "Risk Factors:\\n"
        for factor in assessment['risk_factors'][:3]:  # Show top 3
            risk_text += f"• {factor}\\n"
        
        axes[1, 1].text(0.05, 0.95, risk_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_title('Risk Assessment')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive overfitting analysis report."""
        if model_name not in self.validation_results_:
            raise ValueError(f"No results found for model '{model_name}'")
        
        results = self.validation_results_[model_name]
        
        report = {
            'model_name': model_name,
            'analysis_date': datetime.now().isoformat(),
            'summary': results['assessment'],
            'detailed_results': {
                'walk_forward_validation': {
                    'num_validation_periods': len(results['walk_forward']['val_scores']),
                    'avg_training_score': results['walk_forward'].get('avg_train_score'),
                    'avg_validation_score': results['walk_forward'].get('avg_val_score'),
                    'overfitting_ratio': results['walk_forward'].get('overfitting_ratio'),
                    'score_stability': results['walk_forward'].get('score_stability')
                },
                'learning_curves': results['learning_curves']['convergence_analysis'],
                'statistical_tests': results['statistical_tests'],
                'complexity_analysis': {
                    model_type: data['optimal_alpha'] 
                    for model_type, data in results['complexity_analysis'].items()
                }
            },
            'recommendations': results['assessment']['recommendations']
        }
        
        return report


class UnderfittingDetector:
    """
    Detect underfitting in financial time series models.
    
    Focuses on identifying models that are too simple for the data complexity.
    """
    
    def __init__(self, baseline_models: Optional[List[str]] = None):
        """
        Initialize underfitting detector.
        
        Args:
            baseline_models: List of baseline model names to compare against
        """
        self.baseline_models = baseline_models or ['linear', 'ridge', 'random_forest']
        self.detection_results_ = {}
    
    def detect_underfitting(self, 
                          model: Any, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          model_name: str = "model") -> Dict[str, Any]:
        """
        Detect underfitting by comparing against baseline models.
        
        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target variable
            model_name: Name for reporting
            
        Returns:
            Underfitting detection report
        """
        logger.info(f"Analyzing underfitting for {model_name}")
        
        # Align data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # Get model performance
        model_performance = self._evaluate_model_performance(model, X_aligned, y_aligned)
        
        # Compare against baselines
        baseline_comparisons = self._compare_against_baselines(X_aligned, y_aligned)
        
        # Analyze residuals
        residual_analysis = self._analyze_residuals(model, X_aligned, y_aligned)
        
        # Feature importance analysis
        feature_analysis = self._analyze_feature_utilization(model, X_aligned, y_aligned)
        
        # Overall assessment
        assessment = self._assess_underfitting_risk(
            model_performance, baseline_comparisons, residual_analysis, feature_analysis
        )
        
        # Store results
        self.detection_results_[model_name] = {
            'model_performance': model_performance,
            'baseline_comparisons': baseline_comparisons,
            'residual_analysis': residual_analysis,
            'feature_analysis': feature_analysis,
            'assessment': assessment
        }
        
        return self.detection_results_[model_name]
    
    def _evaluate_model_performance(self, 
                                  model: Any, 
                                  X: pd.DataFrame, 
                                  y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5, test_size=63)
        
        metrics = {
            'mse': [],
            'mae': [],
            'r2': [],
            'directional_accuracy': []
        }
        
        for train_idx, val_idx in tscv.split(X):
            train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
            val_X, val_y = X.iloc[val_idx], y.iloc[val_idx]
            
            # Fit and predict
            model_clone = self._clone_model(model)
            model_clone.fit(train_X, train_y)
            val_pred = model_clone.predict(val_X)
            
            # Calculate metrics
            metrics['mse'].append(mean_squared_error(val_y, val_pred))
            metrics['mae'].append(mean_absolute_error(val_y, val_pred))
            metrics['r2'].append(r2_score(val_y, val_pred))
            
            # Directional accuracy
            val_direction = np.sign(val_y.diff().dropna())
            pred_direction = np.sign(pd.Series(val_pred, index=val_y.index).diff().dropna())
            common_idx = val_direction.index.intersection(pred_direction.index)
            if len(common_idx) > 0:
                direction_acc = (val_direction[common_idx] == pred_direction[common_idx]).mean()
                metrics['directional_accuracy'].append(direction_acc)
        
        # Return average metrics
        return {metric: np.mean(values) for metric, values in metrics.items() if values}
    
    def _compare_against_baselines(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Compare performance against baseline models."""
        baseline_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        }
        
        comparisons = {}
        
        for name, baseline_model in baseline_models.items():
            if name in self.baseline_models:
                try:
                    performance = self._evaluate_model_performance(baseline_model, X, y)
                    comparisons[name] = performance
                except Exception as e:
                    logger.warning(f"Failed to evaluate baseline {name}: {e}")
        
        return comparisons
    
    def _analyze_residuals(self, 
                         model: Any, 
                         X: pd.DataFrame, 
                         y: pd.Series) -> Dict[str, Any]:
        """Analyze residuals for patterns indicating underfitting."""
        # Fit model on most recent data
        train_size = int(len(X) * 0.8)
        train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
        test_X, test_y = X.iloc[train_size:], y.iloc[train_size:]
        
        model_clone = self._clone_model(model)
        model_clone.fit(train_X, train_y)
        
        # Generate predictions and residuals
        test_pred = model_clone.predict(test_X)
        residuals = test_y.values - test_pred
        
        # Analyze residual patterns
        analysis = {
            'residual_stats': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            },
            'autocorrelation': self._calculate_autocorrelation(residuals),
            'heteroscedasticity': self._test_heteroscedasticity(residuals, test_pred),
            'normality_test': stats.jarque_bera(residuals)
        }
        
        return analysis
    
    def _analyze_feature_utilization(self, 
                                   model: Any, 
                                   X: pd.DataFrame, 
                                   y: pd.Series) -> Dict[str, Any]:
        """Analyze how well the model utilizes available features."""
        analysis = {}
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            analysis['feature_importances'] = dict(zip(X.columns, importances))
            analysis['num_zero_importance'] = np.sum(importances == 0)
            analysis['importance_concentration'] = np.sum(importances**2)  # Higher = more concentrated
        
        # Check if model has coefficients (linear models)
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            analysis['coefficients'] = dict(zip(X.columns, coefs))
            analysis['num_zero_coefs'] = np.sum(np.abs(coefs) < 1e-10)
            analysis['coef_magnitude_range'] = np.max(np.abs(coefs)) - np.min(np.abs(coefs))
        
        # Feature selection analysis
        analysis['feature_utilization'] = self._assess_feature_utilization(model, X, y)
        
        return analysis
    
    def _assess_feature_utilization(self, 
                                  model: Any, 
                                  X: pd.DataFrame, 
                                  y: pd.Series) -> Dict[str, float]:
        """Assess how effectively the model uses features."""
        # Baseline performance with all features
        baseline_performance = self._evaluate_model_performance(model, X, y)
        
        # Performance with reduced feature sets
        feature_importance_scores = {}
        
        # Random feature subsets
        np.random.seed(42)
        for subset_size in [0.5, 0.3, 0.1]:
            n_features = max(1, int(len(X.columns) * subset_size))
            random_features = np.random.choice(X.columns, n_features, replace=False)
            
            try:
                subset_performance = self._evaluate_model_performance(
                    model, X[random_features], y
                )
                feature_importance_scores[f'random_{int(subset_size*100)}pct'] = \
                    subset_performance.get('r2', 0)
            except Exception as e:
                logger.warning(f"Feature subset evaluation failed: {e}")
        
        return feature_importance_scores
    
    def _calculate_autocorrelation(self, residuals: np.ndarray, max_lags: int = 10) -> Dict[str, float]:
        """Calculate autocorrelation in residuals."""
        autocorrs = {}
        
        for lag in range(1, min(max_lags + 1, len(residuals) // 4)):
            if len(residuals) > lag:
                autocorr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorrs[f'lag_{lag}'] = autocorr if not np.isnan(autocorr) else 0.0
        
        return autocorrs
    
    def _test_heteroscedasticity(self, residuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Test for heteroscedasticity in residuals."""
        # Breusch-Pagan test approximation
        correlation = np.corrcoef(np.abs(residuals), predictions)[0, 1]
        
        return {
            'correlation_abs_residuals_predictions': correlation if not np.isnan(correlation) else 0.0,
            'variance_ratio': np.var(residuals[:len(residuals)//2]) / np.var(residuals[len(residuals)//2:])
        }
    
    def _assess_underfitting_risk(self, 
                                model_performance: Dict[str, float],
                                baseline_comparisons: Dict[str, Dict[str, float]],
                                residual_analysis: Dict[str, Any],
                                feature_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall underfitting risk."""
        risk_factors = []
        risk_score = 0.0
        
        # Check R² score
        model_r2 = model_performance.get('r2', 0)
        if model_r2 < 0.1:
            risk_factors.append("Very low R² score (< 0.1)")
            risk_score += 0.4
        elif model_r2 < 0.3:
            risk_factors.append("Low R² score (< 0.3)")
            risk_score += 0.2
        
        # Compare against baselines
        for baseline_name, baseline_perf in baseline_comparisons.items():
            baseline_r2 = baseline_perf.get('r2', 0)
            if baseline_r2 > model_r2 + 0.1:  # Baseline significantly better
                risk_factors.append(f"Baseline {baseline_name} performs significantly better")
                risk_score += 0.3
        
        # Check residual patterns
        autocorr_values = list(residual_analysis.get('autocorrelation', {}).values())
        if autocorr_values and max(np.abs(autocorr_values)) > 0.3:
            risk_factors.append("Strong autocorrelation in residuals")
            risk_score += 0.2
        
        # Check feature utilization
        if 'num_zero_importance' in feature_analysis:
            zero_importance_ratio = feature_analysis['num_zero_importance'] / len(feature_analysis.get('feature_importances', {}))
            if zero_importance_ratio > 0.7:
                risk_factors.append("Many features have zero importance")
                risk_score += 0.15
        
        # Determine risk level
        if risk_score >= 0.6:
            risk_level = "HIGH"
        elif risk_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self._generate_underfitting_recommendations(risk_level, risk_factors)
        }
    
    def _generate_underfitting_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate recommendations for addressing underfitting."""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Consider more complex models (ensemble methods, neural networks)",
                "Add more features or feature engineering",
                "Increase model capacity (more parameters, deeper models)",
                "Check for data quality issues",
                "Consider non-linear transformations of existing features"
            ])
        
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Try polynomial features or interaction terms",
                "Consider ensemble methods",
                "Experiment with different model architectures",
                "Validate feature engineering approaches"
            ])
        
        else:  # LOW risk
            recommendations.extend([
                "Current model complexity appears appropriate",
                "Monitor performance as new data becomes available",
                "Consider minor complexity increases if justified"
            ])
        
        # Specific recommendations based on risk factors
        for factor in risk_factors:
            if "r²" in factor.lower() or "r2" in factor.lower():
                recommendations.append("Add polynomial features or feature interactions")
            elif "baseline" in factor.lower():
                recommendations.append("Try ensemble methods or more sophisticated algorithms")
            elif "autocorrelation" in factor.lower():
                recommendations.append("Add lagged features or time series specific models")
            elif "zero importance" in factor.lower():
                recommendations.append("Feature engineering: create derived features from existing ones")
        
        return list(set(recommendations))
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model for cross-validation."""
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