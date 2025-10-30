"""
Advanced hyperparameter optimization using Optuna for RTradez.

Provides sophisticated optimization algorithms including Bayesian optimization,
multi-objective optimization, and distributed optimization for strategy tuning.
"""

import os
import json
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import optuna
from optuna.integration import MLflowCallback
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import joblib
import logging

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


class RTradezOptimizer:
    """
    Advanced hyperparameter optimization for RTradez strategies.
    
    Features:
    - Bayesian optimization with TPE sampler
    - Multi-objective optimization
    - Early stopping with pruning
    - Distributed optimization
    - Integration with MLflow experiment tracking
    """
    
    def __init__(self, study_name: Optional[str] = None,
                 storage_url: Optional[str] = None,
                 sampler_type: str = "tpe",
                 pruner_type: str = "median",
                 n_startup_trials: int = 10):
        """
        Initialize optimizer.
        
        Args:
            study_name: Name of the optimization study
            storage_url: Database URL for distributed optimization
            sampler_type: Sampling algorithm ('tpe', 'cmaes', 'random')
            pruner_type: Pruning algorithm ('median', 'successive_halving', 'none')
            n_startup_trials: Number of random trials before using advanced sampling
        """
        self.study_name = study_name or f"rtradez_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_url = storage_url
        
        # Configure sampler
        self.sampler = self._create_sampler(sampler_type, n_startup_trials)
        
        # Configure pruner
        self.pruner = self._create_pruner(pruner_type)
        
        # Initialize study
        self.study = None
        self._create_study()
        
    def _create_sampler(self, sampler_type: str, n_startup_trials: int):
        """Create optimization sampler."""
        if sampler_type == "tpe":
            return TPESampler(n_startup_trials=n_startup_trials, seed=42)
        elif sampler_type == "cmaes":
            return CmaEsSampler(seed=42)
        elif sampler_type == "random":
            return RandomSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
            
    def _create_pruner(self, pruner_type: str):
        """Create pruning algorithm."""
        if pruner_type == "median":
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_type == "successive_halving":
            return SuccessiveHalvingPruner()
        elif pruner_type == "none":
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner type: {pruner_type}")
            
    def _create_study(self):
        """Create or load Optuna study."""
        if self.storage_url:
            # Distributed optimization
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                load_if_exists=True,
                direction="maximize",
                sampler=self.sampler,
                pruner=self.pruner
            )
        else:
            # Local optimization
            self.study = optuna.create_study(
                direction="maximize",
                sampler=self.sampler,
                pruner=self.pruner
            )
            
        logger.info(f"Created study: {self.study_name}")
        
    def optimize_strategy(self, objective_func: Callable,
                         param_space: Dict[str, Dict],
                         n_trials: int = 100,
                         timeout: Optional[int] = None,
                         callbacks: Optional[List] = None) -> optuna.Study:
        """
        Optimize strategy parameters.
        
        Args:
            objective_func: Function to optimize (should return metric to maximize)
            param_space: Parameter space definition
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            callbacks: Additional callbacks for optimization
            
        Returns:
            Completed Optuna study
        """
        def wrapped_objective(trial):
            # Sample parameters from defined space
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config['type']
                
                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
                    
            # Execute objective function
            try:
                score = objective_func(params, trial)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                # Return a very low score for failed trials
                return -1e6
                
        # Set up callbacks
        all_callbacks = callbacks or []
        
        # Skip MLflow callback to avoid session conflicts in this demo
        # Integration handled separately
            
        # Run optimization
        logger.info(f"Starting optimization with {n_trials} trials")
        start_time = time.time()
        
        self.study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=all_callbacks,
            show_progress_bar=True
        )
        
        duration = time.time() - start_time
        logger.info(f"Optimization completed in {duration:.2f} seconds")
        
        return self.study
        
    def multi_objective_optimize(self, objectives: List[Callable],
                               param_space: Dict[str, Dict],
                               n_trials: int = 100,
                               directions: List[str] = None) -> optuna.Study:
        """
        Multi-objective optimization.
        
        Args:
            objectives: List of objective functions
            param_space: Parameter space definition
            n_trials: Number of trials
            directions: Optimization directions for each objective
            
        Returns:
            Multi-objective study
        """
        directions = directions or ["maximize"] * len(objectives)
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=directions,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        def multi_objective_func(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config['type']
                
                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
                    
            # Evaluate all objectives
            results = []
            for objective in objectives:
                try:
                    score = objective(params, trial)
                    results.append(score)
                except Exception as e:
                    logger.warning(f"Objective evaluation failed: {e}")
                    results.append(-1e6)
                    
            return results
            
        study.optimize(multi_objective_func, n_trials=n_trials)
        return study
        
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from optimization."""
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
            
        return self.study.best_params
        
    def get_best_value(self) -> float:
        """Get best objective value."""
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
            
        return self.study.best_value
        
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
            
        trials_data = []
        for trial in self.study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            
            # Add parameters
            for param_name, param_value in trial.params.items():
                trial_data[f'param_{param_name}'] = param_value
                
            trials_data.append(trial_data)
            
        return pd.DataFrame(trials_data)
        
    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import plotly.graph_objects as go
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            # Plot optimization history
            fig1 = plot_optimization_history(self.study)
            fig1.show()
            
            # Plot parameter importances
            fig2 = plot_param_importances(self.study)
            fig2.show()
            
        except ImportError:
            logger.warning("Plotly not available for visualization")
            
    def save_study(self, filepath: str):
        """Save study to file."""
        joblib.dump(self.study, filepath)
        logger.info(f"Study saved to {filepath}")
        
    def load_study(self, filepath: str):
        """Load study from file."""
        self.study = joblib.load(filepath)
        logger.info(f"Study loaded from {filepath}")


class StrategyOptimizerFactory:
    """Factory for creating strategy-specific optimizers."""
    
    @staticmethod
    def create_iron_condor_optimizer() -> RTradezOptimizer:
        """Create optimizer for Iron Condor strategy."""
        return RTradezOptimizer(
            study_name="iron_condor_optimization",
            sampler_type="tpe",
            pruner_type="median"
        )
        
    @staticmethod
    def create_strangle_optimizer() -> RTradezOptimizer:
        """Create optimizer for Strangle strategy."""
        return RTradezOptimizer(
            study_name="strangle_optimization",
            sampler_type="tpe",
            pruner_type="successive_halving"
        )
        
    @staticmethod
    def get_parameter_space(strategy_type: str) -> Dict[str, Dict]:
        """Get parameter space for strategy type."""
        spaces = {
            'iron_condor': {
                'profit_target': {
                    'type': 'float',
                    'low': 0.1,
                    'high': 0.8
                },
                'stop_loss': {
                    'type': 'float',
                    'low': 1.0,
                    'high': 4.0
                },
                'put_strike_distance': {
                    'type': 'int',
                    'low': 3,
                    'high': 15
                },
                'call_strike_distance': {
                    'type': 'int',
                    'low': 3,
                    'high': 15
                }
            },
            'strangle': {
                'profit_target': {
                    'type': 'float',
                    'low': 0.2,
                    'high': 0.9
                },
                'stop_loss': {
                    'type': 'float',
                    'low': 1.5,
                    'high': 5.0
                },
                'put_delta': {
                    'type': 'float',
                    'low': 0.05,
                    'high': 0.3
                },
                'call_delta': {
                    'type': 'float',
                    'low': 0.05,
                    'high': 0.3
                }
            },
            'straddle': {
                'profit_target': {
                    'type': 'float',
                    'low': 0.2,
                    'high': 0.8
                },
                'stop_loss': {
                    'type': 'float',
                    'low': 1.5,
                    'high': 4.0
                },
                'moneyness_tolerance': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.05
                }
            },
            'calendar_spread': {
                'profit_target': {
                    'type': 'float',
                    'low': 0.1,
                    'high': 0.5
                },
                'stop_loss': {
                    'type': 'float',
                    'low': 1.0,
                    'high': 3.0
                },
                'front_dte': {
                    'type': 'int',
                    'low': 14,
                    'high': 45
                },
                'back_dte': {
                    'type': 'int',
                    'low': 45,
                    'high': 90
                }
            }
        }
        
        return spaces.get(strategy_type, {})


class OptunaObjectiveWrapper:
    """Wrapper for creating Optuna-compatible objective functions."""
    
    def __init__(self, strategy_class, X_train, y_train, X_val, y_val,
                 scoring_func: Optional[Callable] = None):
        self.strategy_class = strategy_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scoring_func = scoring_func or self._default_scoring
        
    def _default_scoring(self, strategy, X, y):
        """Default scoring function (Sharpe ratio)."""
        return strategy.score(X, y)
        
    def __call__(self, params: Dict[str, Any], trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        try:
            # Create strategy with trial parameters
            strategy = self.strategy_class(**params)
            
            # Fit on training data
            strategy.fit(self.X_train, self.y_train)
            
            # Evaluate on validation data
            score = self.scoring_func(strategy, self.X_val, self.y_val)
            
            # Report intermediate value for pruning
            trial.report(score, step=0)
            
            return score
            
        except Exception as e:
            logger.warning(f"Objective evaluation failed: {e}")
            return -1e6


def optimize_strategy_with_optuna(strategy_class, strategy_type: str,
                                X_train, y_train, X_val, y_val,
                                n_trials: int = 50,
                                scoring_func: Optional[Callable] = None) -> Tuple[Dict, float, optuna.Study]:
    """
    Convenience function for strategy optimization.
    
    Args:
        strategy_class: Strategy class to optimize
        strategy_type: Type of strategy for parameter space
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_trials: Number of optimization trials
        scoring_func: Custom scoring function
        
    Returns:
        Tuple of (best_params, best_score, study)
    """
    # Create optimizer
    optimizer = RTradezOptimizer(
        study_name=f"{strategy_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Get parameter space
    param_space = StrategyOptimizerFactory.get_parameter_space(strategy_type)
    
    # Create objective function
    objective = OptunaObjectiveWrapper(
        strategy_class, X_train, y_train, X_val, y_val, scoring_func
    )
    
    # Run optimization
    study = optimizer.optimize_strategy(
        objective_func=objective,
        param_space=param_space,
        n_trials=n_trials
    )
    
    return study.best_params, study.best_value, study