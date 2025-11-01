"""Advanced cross-validation techniques for financial time series data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class WalkForwardCV(BaseCrossValidator):
    """
    Walk-forward cross-validation for time series.
    
    Implements expanding and rolling window validation with proper temporal ordering.
    """
    
    def __init__(self,
                 min_train_size: int = 252,  # 1 year
                 test_size: int = 63,        # 3 months  
                 step_size: int = 21,        # 1 month
                 expanding_window: bool = True,
                 max_train_size: Optional[int] = None):
        """
        Initialize walk-forward cross-validator.
        
        Args:
            min_train_size: Minimum training samples required
            test_size: Test set size for each fold
            step_size: Step size between folds
            expanding_window: If True, use expanding window; if False, use rolling window
            max_train_size: Maximum training window size (for rolling window)
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.expanding_window = expanding_window
        self.max_train_size = max_train_size
        
        if not expanding_window and max_train_size is None:
            self.max_train_size = min_train_size * 2  # Default rolling window size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        if X is None:
            return 0
        
        n_samples = len(X)
        if n_samples < self.min_train_size + self.test_size:
            return 0
        
        # Calculate number of possible splits
        available_length = n_samples - self.min_train_size - self.test_size + 1
        n_splits = max(0, available_length // self.step_size)
        
        return n_splits
    
    def split(self, X, y=None, groups=None):
        """Generate indices for train/test splits."""
        n_samples = len(X)
        
        if n_samples < self.min_train_size + self.test_size:
            raise ValueError(
                f"Not enough samples: need at least {self.min_train_size + self.test_size}, "
                f"got {n_samples}"
            )
        
        # Starting position for first test set
        test_start = self.min_train_size
        
        while test_start + self.test_size <= n_samples:
            test_end = test_start + self.test_size
            
            if self.expanding_window:
                # Expanding window: train from beginning to test start
                train_start = 0
                train_end = test_start
            else:
                # Rolling window: fixed-size training window
                train_end = test_start
                train_start = max(0, train_end - self.max_train_size)
            
            # Ensure minimum training size
            if train_end - train_start >= self.min_train_size:
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                
                yield train_indices, test_indices
            
            test_start += self.step_size


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purged group time series split with embargo periods.
    
    Prevents data leakage by adding gaps between training and test sets.
    Useful for high-frequency trading strategies.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 embargo_period: int = 5,  # Days to embargo after training
                 purge_period: int = 2,    # Days to purge before test
                 group_col: Optional[str] = None):
        """
        Initialize purged group time series split.
        
        Args:
            n_splits: Number of splits
            embargo_period: Embargo period after training (in samples)
            purge_period: Purge period before test (in samples)
            group_col: Column name for grouping (e.g., 'date', 'month')
        """
        self.n_splits = n_splits
        self.embargo_period = embargo_period
        self.purge_period = purge_period
        self.group_col = group_col
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """Generate indices for purged splits."""
        n_samples = len(X)
        
        # If groups provided, use group-based splitting
        if groups is not None:
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            
            if n_groups < self.n_splits:
                raise ValueError(f"Number of groups ({n_groups}) < n_splits ({self.n_splits})")
            
            group_size = n_groups // self.n_splits
            
            for i in range(self.n_splits):
                # Test groups
                test_start_group = i * group_size
                test_end_group = (i + 1) * group_size if i < self.n_splits - 1 else n_groups
                test_groups = unique_groups[test_start_group:test_end_group]
                
                # Train groups (before test, with embargo)
                embargo_end_group = max(0, test_start_group - self.embargo_period)
                train_groups = unique_groups[:embargo_end_group]
                
                # Convert groups to indices
                test_mask = np.isin(groups, test_groups)
                train_mask = np.isin(groups, train_groups)
                
                train_indices = np.where(train_mask)[0]
                test_indices = np.where(test_mask)[0]
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices
        
        else:
            # Standard time series splits with purging and embargo
            test_size = n_samples // self.n_splits
            
            for i in range(self.n_splits):
                # Test indices
                test_start = i * test_size
                test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
                
                # Train indices (with purging and embargo)
                train_end = max(0, test_start - self.purge_period)
                train_start = 0
                
                # Apply embargo to previous folds
                if i > 0:
                    prev_test_end = i * test_size
                    train_end = min(train_end, prev_test_end - self.embargo_period)
                
                if train_end > train_start:
                    train_indices = np.arange(train_start, train_end)
                    test_indices = np.arange(test_start, test_end)
                    
                    yield train_indices, test_indices


class MonteCarloCV:
    """
    Monte Carlo cross-validation for time series.
    
    Randomly samples training and test periods while respecting temporal order.
    """
    
    def __init__(self,
                 n_splits: int = 100,
                 train_size: float = 0.7,
                 test_size: float = 0.2,
                 min_gap: int = 5,  # Minimum gap between train and test
                 random_state: Optional[int] = None):
        """
        Initialize Monte Carlo cross-validator.
        
        Args:
            n_splits: Number of random splits
            train_size: Fraction of data for training
            test_size: Fraction of data for testing
            min_gap: Minimum gap between training and test periods
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.min_gap = min_gap
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def split(self, X, y=None, groups=None):
        """Generate random splits."""
        n_samples = len(X)
        
        train_samples = int(n_samples * self.train_size)
        test_samples = int(n_samples * self.test_size)
        
        if train_samples + test_samples + self.min_gap > n_samples:
            raise ValueError("Not enough samples for the specified split configuration")
        
        for _ in range(self.n_splits):
            # Random start for training period
            max_train_start = n_samples - train_samples - test_samples - self.min_gap
            train_start = np.random.randint(0, max_train_start + 1)
            train_end = train_start + train_samples
            
            # Test period after gap
            test_start = train_end + self.min_gap
            test_end = test_start + test_samples
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices


class CombinatorialPurgedCV:
    """
    Combinatorial purged cross-validation.
    
    Implements the CPCV method from "Advances in Financial Machine Learning" by Marcos López de Prado.
    """
    
    def __init__(self,
                 n_splits: int = 6,
                 n_test_groups: int = 2,
                 embargo_period: int = 5,
                 purge_period: int = 2):
        """
        Initialize combinatorial purged cross-validator.
        
        Args:
            n_splits: Number of groups to create
            n_test_groups: Number of groups to use for testing in each split
            embargo_period: Embargo period
            purge_period: Purge period
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_period = embargo_period
        self.purge_period = purge_period
    
    def split(self, X, y=None, groups=None):
        """Generate combinatorial purged splits."""
        n_samples = len(X)
        
        # Create groups
        group_size = n_samples // self.n_splits
        sample_groups = np.repeat(range(self.n_splits), group_size)
        
        # Handle remaining samples
        remainder = n_samples - len(sample_groups)
        if remainder > 0:
            sample_groups = np.concatenate([
                sample_groups, 
                np.repeat(self.n_splits - 1, remainder)
            ])
        
        # Generate all combinations of test groups
        from itertools import combinations
        
        for test_group_combo in combinations(range(self.n_splits), self.n_test_groups):
            test_mask = np.isin(sample_groups, test_group_combo)
            
            # Determine training groups (with embargo and purge)
            train_groups = []
            for group in range(self.n_splits):
                if group not in test_group_combo:
                    # Check if group is too close to test groups
                    min_dist_to_test = min(abs(group - tg) for tg in test_group_combo)
                    if min_dist_to_test > max(self.embargo_period, self.purge_period):
                        train_groups.append(group)
            
            if train_groups:
                train_mask = np.isin(sample_groups, train_groups)
                
                train_indices = np.where(train_mask)[0]
                test_indices = np.where(test_mask)[0]
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices


class TimeSeriesValidation:
    """
    Comprehensive time series validation framework.
    
    Combines multiple validation techniques and provides performance analysis.
    """
    
    def __init__(self,
                 cv_methods: Optional[List[str]] = None,
                 scoring_metrics: Optional[List[str]] = None):
        """
        Initialize time series validation framework.
        
        Args:
            cv_methods: List of CV methods to use
            scoring_metrics: List of metrics to calculate
        """
        self.cv_methods = cv_methods or ['walk_forward', 'purged_group']
        self.scoring_metrics = scoring_metrics or ['mse', 'mae', 'r2', 'sharpe']
        
        # Results storage
        self.validation_results_ = {}
        self.performance_summary_ = {}
        
        # Initialize CV objects
        self.cv_objects = {
            'walk_forward': WalkForwardCV(),
            'purged_group': PurgedGroupTimeSeriesSplit(),
            'monte_carlo': MonteCarloCV(),
            'combinatorial': CombinatorialPurgedCV()
        }
    
    def validate_model(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      groups: Optional[pd.Series] = None,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive model validation using multiple CV techniques.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target variable
            groups: Group labels for group-based CV
            model_name: Name for reporting
            
        Returns:
            Validation results
        """
        logger.info(f"Validating model '{model_name}' using {len(self.cv_methods)} CV methods")
        
        # Align data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        if groups is not None:
            groups_aligned = groups.loc[common_index]
        else:
            groups_aligned = None
        
        results = {}
        
        for cv_method in self.cv_methods:
            logger.info(f"Running {cv_method} cross-validation...")
            
            try:
                cv_results = self._run_cv_method(
                    cv_method, model, X_aligned, y_aligned, groups_aligned
                )
                results[cv_method] = cv_results
                
            except Exception as e:
                logger.warning(f"CV method {cv_method} failed: {e}")
                results[cv_method] = {'error': str(e)}
        
        # Calculate summary statistics
        summary = self._calculate_validation_summary(results)
        
        # Store results
        self.validation_results_[model_name] = results
        self.performance_summary_[model_name] = summary
        
        return {
            'detailed_results': results,
            'summary': summary,
            'model_name': model_name
        }
    
    def _run_cv_method(self,
                      cv_method: str,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      groups: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Run a specific CV method."""
        cv_obj = self.cv_objects[cv_method]
        
        # Convert to numpy for sklearn compatibility
        X_array = X.values
        y_array = y.values
        groups_array = groups.values if groups is not None else None
        
        fold_results = []
        
        try:
            splits = list(cv_obj.split(X_array, y_array, groups_array))
            logger.info(f"{cv_method}: Generated {len(splits)} folds")
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                fold_result = self._evaluate_fold(
                    model, X_array, y_array, train_idx, test_idx, fold_idx
                )
                fold_results.append(fold_result)
            
        except Exception as e:
            logger.error(f"Error in {cv_method}: {e}")
            raise
        
        # Aggregate results
        if fold_results:
            aggregated = self._aggregate_fold_results(fold_results)
            aggregated['n_folds'] = len(fold_results)
            aggregated['cv_method'] = cv_method
            return aggregated
        else:
            return {'error': 'No valid folds generated'}
    
    def _evaluate_fold(self,
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      train_idx: np.ndarray,
                      test_idx: np.ndarray,
                      fold_idx: int) -> Dict[str, Any]:
        """Evaluate a single fold."""
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone and fit model
        model_clone = self._clone_model(model)
        model_clone.fit(X_train, y_train)
        
        # Predictions
        y_pred = model_clone.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        for metric in self.scoring_metrics:
            try:
                if metric == 'mse':
                    metrics[metric] = mean_squared_error(y_test, y_pred)
                elif metric == 'mae':
                    metrics[metric] = mean_absolute_error(y_test, y_pred)
                elif metric == 'r2':
                    metrics[metric] = r2_score(y_test, y_pred)
                elif metric == 'sharpe':
                    # Calculate Sharpe ratio of predictions vs actuals
                    returns_actual = np.diff(y_test)
                    returns_pred = np.diff(y_pred)
                    
                    if len(returns_actual) > 1 and np.std(returns_pred) > 0:
                        correlation = np.corrcoef(returns_actual, returns_pred)[0, 1]
                        sharpe = correlation * np.sqrt(252)  # Annualized
                        metrics[metric] = sharpe if not np.isnan(sharpe) else 0.0
                    else:
                        metrics[metric] = 0.0
                        
            except Exception as e:
                logger.warning(f"Failed to calculate {metric}: {e}")
                metrics[metric] = np.nan
        
        return {
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'metrics': metrics
        }
    
    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across folds."""
        aggregated = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {},
            'train_sizes': [],
            'test_sizes': []
        }
        
        # Extract metrics from all folds
        all_metrics = {}
        for fold_result in fold_results:
            aggregated['fold_metrics'].append(fold_result['metrics'])
            aggregated['train_sizes'].append(fold_result['train_size'])
            aggregated['test_sizes'].append(fold_result['test_size'])
            
            for metric, value in fold_result['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate mean and std for each metric
        for metric, values in all_metrics.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                aggregated['mean_metrics'][metric] = np.mean(valid_values)
                aggregated['std_metrics'][metric] = np.std(valid_values)
            else:
                aggregated['mean_metrics'][metric] = np.nan
                aggregated['std_metrics'][metric] = np.nan
        
        # Summary statistics
        aggregated['avg_train_size'] = np.mean(aggregated['train_sizes'])
        aggregated['avg_test_size'] = np.mean(aggregated['test_sizes'])
        
        return aggregated
    
    def _calculate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary across all CV methods."""
        summary = {
            'cv_methods_used': list(results.keys()),
            'cv_methods_successful': [],
            'consensus_metrics': {},
            'method_comparison': {}
        }
        
        # Collect successful methods and their metrics
        successful_results = {}
        for method, result in results.items():
            if 'error' not in result and 'mean_metrics' in result:
                summary['cv_methods_successful'].append(method)
                successful_results[method] = result['mean_metrics']
        
        if successful_results:
            # Calculate consensus metrics (average across CV methods)
            all_metric_names = set()
            for metrics in successful_results.values():
                all_metric_names.update(metrics.keys())
            
            for metric in all_metric_names:
                method_values = []
                for method_metrics in successful_results.values():
                    if metric in method_metrics and not np.isnan(method_metrics[metric]):
                        method_values.append(method_metrics[metric])
                
                if method_values:
                    summary['consensus_metrics'][metric] = {
                        'mean': np.mean(method_values),
                        'std': np.std(method_values),
                        'methods_available': len(method_values)
                    }
            
            # Method comparison
            summary['method_comparison'] = successful_results
        
        return summary
    
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
    
    def compare_models(self, 
                      models: Dict[str, Any],
                      X: pd.DataFrame,
                      y: pd.Series,
                      groups: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compare multiple models using the same validation framework.
        
        Args:
            models: Dictionary of models {name: model}
            X: Feature matrix
            y: Target variable
            groups: Group labels
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        comparison_results = []
        
        for model_name, model in models.items():
            try:
                validation_result = self.validate_model(model, X, y, groups, model_name)
                
                # Extract key metrics for comparison
                summary = validation_result['summary']
                
                row = {'model': model_name}
                
                # Add consensus metrics
                for metric, values in summary.get('consensus_metrics', {}).items():
                    row[f'{metric}_mean'] = values['mean']
                    row[f'{metric}_std'] = values['std']
                
                # Add method-specific results
                for cv_method in summary.get('cv_methods_successful', []):
                    method_result = validation_result['detailed_results'][cv_method]
                    for metric, value in method_result.get('mean_metrics', {}).items():
                        row[f'{cv_method}_{metric}'] = value
                
                comparison_results.append(row)
                
            except Exception as e:
                logger.error(f"Failed to validate model {model_name}: {e}")
                comparison_results.append({
                    'model': model_name,
                    'error': str(e)
                })
        
        return pd.DataFrame(comparison_results)
    
    def get_validation_report(self, model_name: str) -> Dict[str, Any]:
        """Get detailed validation report for a model."""
        if model_name not in self.validation_results_:
            raise ValueError(f"No validation results found for model '{model_name}'")
        
        return {
            'model_name': model_name,
            'validation_date': datetime.now().isoformat(),
            'detailed_results': self.validation_results_[model_name],
            'summary': self.performance_summary_[model_name],
            'cv_framework': {
                'cv_methods': self.cv_methods,
                'scoring_metrics': self.scoring_metrics
            }
        }