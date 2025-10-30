"""
Experiment tracking and management for RTradez.

Provides comprehensive experiment tracking using MLflow for strategy development,
hyperparameter optimization, and performance monitoring.
"""

import os
import json
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
import logging

# Configure logging
logger = logging.getLogger(__name__)


class RTradezExperimentTracker:
    """
    Comprehensive experiment tracking for RTradez strategies and models.
    
    Features:
    - Strategy performance tracking
    - Hyperparameter logging
    - Model artifacts storage
    - Automated experiment organization
    - Performance comparison and analysis
    """
    
    def __init__(self, tracking_uri: Optional[str] = None,
                 experiment_name: str = "RTradez_Strategies"):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_uri: MLflow tracking URI (defaults to local file store)
            experiment_name: Name of the MLflow experiment
        """
        # Set up MLflow tracking
        if tracking_uri is None:
            tracking_dir = os.path.expanduser("~/.rtradez/mlflow")
            Path(tracking_dir).mkdir(parents=True, exist_ok=True)
            tracking_uri = f"file://{tracking_dir}"
            
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
        logger.info(f"Experiment tracker initialized: {experiment_name}")
        
    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new experiment run."""
        if run_name is None:
            run_name = f"strategy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        tags = tags or {}
        tags.update({
            "rtradez.version": "1.0.0",
            "rtradez.timestamp": datetime.now().isoformat()
        })
        
        return mlflow.start_run(run_name=run_name, tags=tags)
        
    def log_strategy_config(self, strategy_type: str, 
                           parameters: Dict[str, Any]):
        """Log strategy configuration."""
        mlflow.log_param("strategy_type", strategy_type)
        
        for param_name, param_value in parameters.items():
            mlflow.log_param(f"strategy.{param_name}", param_value)
            
    def log_market_data_info(self, symbol: str, start_date: str, 
                           end_date: str, data_shape: tuple):
        """Log market data information."""
        mlflow.log_param("market.symbol", symbol)
        mlflow.log_param("market.start_date", start_date)
        mlflow.log_param("market.end_date", end_date)
        mlflow.log_param("market.data_points", data_shape[0])
        mlflow.log_param("market.features", data_shape[1])
        
    def log_feature_engineering(self, feature_names: List[str],
                              feature_stats: Optional[Dict] = None):
        """Log feature engineering details."""
        mlflow.log_param("features.count", len(feature_names))
        mlflow.log_param("features.names", ",".join(feature_names[:20]))  # Limit length
        
        if feature_stats:
            for stat_name, stat_value in feature_stats.items():
                mlflow.log_metric(f"features.{stat_name}", stat_value)
                
    def log_performance_metrics(self, metrics: Dict[str, float],
                              step: Optional[int] = None):
        """Log strategy performance metrics."""
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"performance.{metric_name}", metric_value, step=step)
            
    def log_backtest_results(self, results: Dict[str, Any]):
        """Log comprehensive backtest results."""
        # Core performance metrics
        core_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
            'profit_factor', 'total_trades', 'avg_trade'
        ]
        
        for metric in core_metrics:
            if metric in results:
                mlflow.log_metric(f"backtest.{metric}", results[metric])
                
        # Log additional metrics
        for key, value in results.items():
            if key not in core_metrics and isinstance(value, (int, float)):
                mlflow.log_metric(f"backtest.{key}", value)
                
    def log_optimization_results(self, study_name: str, 
                               best_params: Dict[str, Any],
                               best_value: float,
                               n_trials: int):
        """Log hyperparameter optimization results."""
        mlflow.log_param("optimization.study_name", study_name)
        mlflow.log_param("optimization.n_trials", n_trials)
        mlflow.log_metric("optimization.best_value", best_value)
        
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"optimization.best_{param_name}", param_value)
            
    def log_model(self, model, model_name: str = "strategy_model",
                  signature: Optional[mlflow.types.Schema] = None):
        """Log trained model as MLflow artifact."""
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature
            )
        except Exception as e:
            # Fallback to pickle if sklearn logging fails
            logger.warning(f"sklearn logging failed, using pickle: {e}")
            mlflow.log_artifact(model, model_name)
            
    def log_dataframe(self, df: pd.DataFrame, name: str):
        """Log DataFrame as artifact."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name)
            mlflow.log_artifact(f.name, f"{name}.csv")
            os.unlink(f.name)
            
    def log_plot(self, fig, name: str):
        """Log matplotlib/plotly figure."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name)
            mlflow.log_artifact(f.name, f"{name}.png")
            os.unlink(f.name)
            
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiment runs."""
        runs_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            run_data = {
                'run_id': run_id,
                'run_name': run.info.run_name,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time
            }
            
            # Add parameters
            for key, value in run.data.params.items():
                run_data[f"param_{key}"] = value
                
            # Add metrics
            for key, value in run.data.metrics.items():
                run_data[f"metric_{key}"] = value
                
            runs_data.append(run_data)
            
        return pd.DataFrame(runs_data)
        
    def get_best_run(self, metric_name: str = "performance.sharpe_ratio",
                     ascending: bool = False) -> Optional[mlflow.entities.Run]:
        """Get best run based on specific metric."""
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"metrics.{metric_name} IS NOT NULL",
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )
            return runs[0] if runs else None
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None
            
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of experiment."""
        experiment = self.client.get_experiment(self.experiment_id)
        runs = self.client.search_runs(experiment_ids=[self.experiment_id])
        
        summary = {
            'experiment_name': experiment.name,
            'experiment_id': self.experiment_id,
            'total_runs': len(runs),
            'successful_runs': len([r for r in runs if r.info.status == RunStatus.FINISHED]),
            'failed_runs': len([r for r in runs if r.info.status == RunStatus.FAILED]),
            'latest_run': max([r.info.start_time for r in runs]) if runs else None
        }
        
        # Get metric statistics
        if runs:
            all_metrics = {}
            for run in runs:
                for metric_name, metric_value in run.data.metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
                    
            summary['metric_stats'] = {}
            for metric_name, values in all_metrics.items():
                summary['metric_stats'][metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                
        return summary


class ExperimentContextManager:
    """Context manager for experiment runs."""
    
    def __init__(self, tracker: RTradezExperimentTracker,
                 run_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags
        self.run = None
        
    def __enter__(self):
        self.run = self.tracker.start_run(self.run_name, self.tags)
        return self.tracker
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(exc_val))
        else:
            mlflow.set_tag("status", "completed")
        mlflow.end_run()


def track_experiment(experiment_name: Optional[str] = None,
                    run_name: Optional[str] = None,
                    tags: Optional[Dict[str, str]] = None):
    """Decorator for automatic experiment tracking."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = RTradezExperimentTracker(experiment_name=experiment_name or "RTradez_Strategies")
            
            with ExperimentContextManager(tracker, run_name, tags) as exp_tracker:
                # Log function name and arguments
                mlflow.log_param("function", func.__name__)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Try to log result if it's a dict with metrics
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                            
                return result
                
        return wrapper
    return decorator


# Global experiment tracker instance
_tracker = None

def get_experiment_tracker() -> RTradezExperimentTracker:
    """Get global experiment tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = RTradezExperimentTracker()
    return _tracker


class StrategyExperimentLogger:
    """Specialized logger for strategy experiments."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.tracker = get_experiment_tracker()
        self.run_start_time = None
        
    def start_experiment(self, tags: Optional[Dict[str, str]] = None):
        """Start a new strategy experiment."""
        run_name = f"{self.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_tags = {"strategy_type": self.strategy_name}
        if tags:
            experiment_tags.update(tags)
            
        self.run = self.tracker.start_run(run_name, experiment_tags)
        self.run_start_time = time.time()
        
    def log_data_info(self, symbol: str, data_shape: tuple, 
                     date_range: tuple):
        """Log market data information."""
        self.tracker.log_market_data_info(
            symbol, date_range[0], date_range[1], data_shape
        )
        
    def log_strategy_params(self, params: Dict[str, Any]):
        """Log strategy parameters."""
        self.tracker.log_strategy_config(self.strategy_name, params)
        
    def log_performance(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.tracker.log_performance_metrics(metrics)
        
    def log_optimization_step(self, trial_number: int, params: Dict[str, Any],
                            objective_value: float):
        """Log optimization trial."""
        mlflow.log_metric("trial_number", trial_number)
        mlflow.log_metric("objective_value", objective_value)
        
        for param_name, param_value in params.items():
            mlflow.log_metric(f"trial_{param_name}", param_value, step=trial_number)
            
    def end_experiment(self, success: bool = True):
        """End the experiment."""
        if self.run_start_time:
            duration = time.time() - self.run_start_time
            mlflow.log_metric("experiment_duration_seconds", duration)
            
        mlflow.set_tag("experiment_status", "success" if success else "failed")
        mlflow.end_run()