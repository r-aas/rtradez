"""Market regime detection for adaptive options strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

from ..base import BaseStrategy, BaseTransformer
from ..utils.caching import cached


class MarketRegimeDetector(BaseTransformer):
    """
    Detect market regimes using multiple methodologies.
    
    Supports:
    - Volatility-based regimes (low/medium/high)
    - Trend-based regimes (bull/bear/sideways)
    - Combined regimes using Gaussian Mixture Models
    """
    
    def __init__(self, 
                 method: str = 'combined',
                 n_regimes: int = 3,
                 lookback_window: int = 252,
                 vol_percentiles: Tuple[float, float] = (33, 67),
                 trend_threshold: float = 0.02):
        """
        Initialize regime detector.
        
        Args:
            method: 'volatility', 'trend', 'combined', or 'hmm'
            n_regimes: Number of regimes to detect
            lookback_window: Window for regime calculation
            vol_percentiles: Percentiles for volatility regime boundaries
            trend_threshold: Minimum trend strength for bull/bear classification
        """
        self.method = method
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.vol_percentiles = vol_percentiles
        self.trend_threshold = trend_threshold
        
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.regime_thresholds_ = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MarketRegimeDetector':
        """
        Fit regime detector to historical data.
        
        Args:
            X: Market data with 'Close', 'High', 'Low', 'Volume' columns
            y: Ignored (for sklearn compatibility)
        """
        if 'Close' not in X.columns:
            raise ValueError("X must contain 'Close' column")
        
        # Calculate regime features
        features = self._calculate_regime_features(X)
        
        if self.method == 'volatility':
            self._fit_volatility_regimes(features)
        elif self.method == 'trend':
            self._fit_trend_regimes(features)
        elif self.method == 'combined':
            self._fit_combined_regimes(features)
        elif self.method == 'hmm':
            self._fit_hmm_regimes(features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to include regime labels.
        
        Args:
            X: Market data
            
        Returns:
            DataFrame with added regime columns
        """
        if not self.is_fitted:
            raise ValueError("Must fit detector before transform")
        
        X_transformed = X.copy()
        features = self._calculate_regime_features(X_transformed)
        
        if self.method == 'volatility':
            regimes = self._predict_volatility_regimes(features)
        elif self.method == 'trend':
            regimes = self._predict_trend_regimes(features)
        elif self.method == 'combined':
            regimes = self._predict_combined_regimes(features)
        elif self.method == 'hmm':
            regimes = self._predict_hmm_regimes(features)
        
        # Add regime labels
        X_transformed['regime'] = regimes
        X_transformed['regime_name'] = self._get_regime_names(regimes)
        
        # Add regime probabilities if available
        if hasattr(self, 'regime_probabilities_'):
            for i in range(self.n_regimes):
                X_transformed[f'regime_{i}_prob'] = self.regime_probabilities_[:, i]
        
        return X_transformed
    
    def _calculate_regime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate features used for regime detection."""
        features = pd.DataFrame(index=X.index)
        
        # Price-based features
        features['returns'] = X['Close'].pct_change()
        features['log_returns'] = np.log(X['Close'] / X['Close'].shift(1))
        
        # Volatility features
        features['realized_vol'] = features['returns'].rolling(20).std() * np.sqrt(252)
        features['parkinson_vol'] = np.sqrt(
            252 / (4 * np.log(2)) * 
            (np.log(X['High'] / X['Low']) ** 2).rolling(20).mean()
        ) if 'High' in X.columns and 'Low' in X.columns else features['realized_vol']
        
        # Trend features
        features['sma_20'] = X['Close'].rolling(20).mean()
        features['sma_50'] = X['Close'].rolling(50).mean()
        features['sma_200'] = X['Close'].rolling(200).mean()
        
        features['trend_20'] = (X['Close'] / features['sma_20'] - 1)
        features['trend_50'] = (X['Close'] / features['sma_50'] - 1) 
        features['trend_200'] = (X['Close'] / features['sma_200'] - 1)
        
        # Momentum features
        features['rsi'] = self._calculate_rsi(X['Close'])
        features['momentum_10'] = X['Close'].pct_change(10)
        features['momentum_20'] = X['Close'].pct_change(20)
        
        # Volume features (if available)
        if 'Volume' in X.columns:
            features['volume_sma'] = X['Volume'].rolling(20).mean()
            features['volume_ratio'] = X['Volume'] / features['volume_sma']
        else:
            features['volume_ratio'] = 1.0
        
        # Market structure features
        features['drawdown'] = self._calculate_drawdown(X['Close'])
        features['volatility_rank'] = features['realized_vol'].rolling(252).rank(pct=True)
        
        return features.fillna(method='bfill').fillna(method='ffill')
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """Calculate running maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max
    
    def _fit_volatility_regimes(self, features: pd.DataFrame):
        """Fit volatility-based regime detector."""
        vol_data = features['realized_vol'].dropna()
        
        # Calculate percentile thresholds
        low_threshold = np.percentile(vol_data, self.vol_percentiles[0])
        high_threshold = np.percentile(vol_data, self.vol_percentiles[1])
        
        self.regime_thresholds_['volatility'] = {
            'low': low_threshold,
            'high': high_threshold
        }
    
    def _predict_volatility_regimes(self, features: pd.DataFrame) -> np.ndarray:
        """Predict volatility-based regimes."""
        vol = features['realized_vol']
        thresholds = self.regime_thresholds_['volatility']
        
        regimes = np.zeros(len(vol))
        regimes[vol <= thresholds['low']] = 0  # Low volatility
        regimes[(vol > thresholds['low']) & (vol <= thresholds['high'])] = 1  # Medium
        regimes[vol > thresholds['high']] = 2  # High volatility
        
        return regimes.astype(int)
    
    def _fit_trend_regimes(self, features: pd.DataFrame):
        """Fit trend-based regime detector."""
        trend_data = features['trend_200'].dropna()
        
        self.regime_thresholds_['trend'] = {
            'bear': -self.trend_threshold,
            'bull': self.trend_threshold
        }
    
    def _predict_trend_regimes(self, features: pd.DataFrame) -> np.ndarray:
        """Predict trend-based regimes."""
        trend = features['trend_200']
        thresholds = self.regime_thresholds_['trend']
        
        regimes = np.ones(len(trend))  # Default to sideways (1)
        regimes[trend <= thresholds['bear']] = 0  # Bear market
        regimes[trend >= thresholds['bull']] = 2  # Bull market
        
        return regimes.astype(int)
    
    def _fit_combined_regimes(self, features: pd.DataFrame):
        """Fit combined regime detector using Gaussian Mixture Model."""
        # Select features for regime detection
        regime_features = [
            'realized_vol', 'trend_200', 'rsi', 'momentum_20', 
            'drawdown', 'volatility_rank'
        ]
        
        feature_data = features[regime_features].dropna()
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Fit Gaussian Mixture Model
        self.gmm.fit(scaled_features)
        
        # Store feature names for prediction
        self.regime_features_ = regime_features
    
    def _predict_combined_regimes(self, features: pd.DataFrame) -> np.ndarray:
        """Predict combined regimes using GMM."""
        feature_data = features[self.regime_features_].fillna(method='ffill')
        scaled_features = self.scaler.transform(feature_data)
        
        regimes = self.gmm.predict(scaled_features)
        self.regime_probabilities_ = self.gmm.predict_proba(scaled_features)
        
        return regimes
    
    def _fit_hmm_regimes(self, features: pd.DataFrame):
        """Fit Hidden Markov Model for regime detection."""
        try:
            from hmmlearn import hmm
        except ImportError:
            warnings.warn("hmmlearn not available, falling back to combined method")
            return self._fit_combined_regimes(features)
        
        # Use returns for HMM
        returns = features['returns'].dropna().values.reshape(-1, 1)
        
        # Fit Gaussian HMM
        self.hmm_model = hmm.GaussianHMM(n_components=self.n_regimes, random_state=42)
        self.hmm_model.fit(returns)
    
    def _predict_hmm_regimes(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regimes using HMM."""
        if not hasattr(self, 'hmm_model'):
            return self._predict_combined_regimes(features)
        
        returns = features['returns'].fillna(method='ffill').values.reshape(-1, 1)
        regimes = self.hmm_model.predict(returns)
        
        return regimes
    
    def _get_regime_names(self, regimes: np.ndarray) -> List[str]:
        """Convert regime numbers to descriptive names."""
        if self.method == 'volatility':
            regime_names = {0: 'Low Vol', 1: 'Medium Vol', 2: 'High Vol'}
        elif self.method == 'trend':
            regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        else:
            regime_names = {i: f'Regime {i}' for i in range(self.n_regimes)}
        
        return [regime_names.get(r, f'Regime {r}') for r in regimes]
    
    def get_regime_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get statistics for each detected regime."""
        X_transformed = self.transform(X)
        
        stats = []
        for regime in range(self.n_regimes):
            regime_data = X_transformed[X_transformed['regime'] == regime]
            
            if len(regime_data) > 0:
                stats.append({
                    'regime': regime,
                    'regime_name': regime_data['regime_name'].iloc[0],
                    'frequency': len(regime_data) / len(X_transformed),
                    'avg_return': regime_data['returns'].mean() * 252,  # Annualized
                    'volatility': regime_data['returns'].std() * np.sqrt(252),
                    'sharpe_ratio': regime_data['returns'].mean() / regime_data['returns'].std() * np.sqrt(252),
                    'max_drawdown': regime_data['drawdown'].min(),
                    'avg_duration': self._calculate_avg_duration(X_transformed['regime'], regime)
                })
        
        return pd.DataFrame(stats)
    
    def _calculate_avg_duration(self, regime_series: pd.Series, regime: int) -> float:
        """Calculate average duration of regime periods."""
        regime_periods = []
        current_period = 0
        
        for r in regime_series:
            if r == regime:
                current_period += 1
            else:
                if current_period > 0:
                    regime_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            regime_periods.append(current_period)
        
        return np.mean(regime_periods) if regime_periods else 0


class RegimeBasedStrategy(BaseStrategy):
    """
    Strategy that adapts based on detected market regimes.
    
    Uses different strategy parameters or types for different market conditions.
    """
    
    def __init__(self,
                 regime_detector: MarketRegimeDetector,
                 regime_strategies: Dict[int, Dict],
                 default_strategy: str = 'iron_condor'):
        """
        Initialize regime-based strategy.
        
        Args:
            regime_detector: Fitted regime detector
            regime_strategies: Dict mapping regime -> strategy config
            default_strategy: Strategy to use if regime not specified
        """
        super().__init__()
        self.regime_detector = regime_detector
        self.regime_strategies = regime_strategies
        self.default_strategy = default_strategy
        self.fitted_strategies_ = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RegimeBasedStrategy':
        """Fit strategies for each regime."""
        # Add regime information
        X_with_regimes = self.regime_detector.transform(X)
        
        # Fit strategy for each regime
        for regime in range(self.regime_detector.n_regimes):
            regime_mask = X_with_regimes['regime'] == regime
            regime_data = X_with_regimes[regime_mask]
            
            if len(regime_data) > 50:  # Minimum data for fitting
                # Get strategy config for this regime
                strategy_config = self.regime_strategies.get(regime, {
                    'strategy_type': self.default_strategy
                })
                
                # Create and fit strategy
                from ..methods.strategies import OptionsStrategy
                strategy = OptionsStrategy(**strategy_config)
                
                regime_X = regime_data[X.columns]  # Original features only
                regime_y = y.loc[regime_X.index] if y is not None else None
                
                strategy.fit(regime_X, regime_y)
                self.fitted_strategies_[regime] = strategy
        
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using regime-appropriate strategies."""
        if not self._fitted:
            raise ValueError("Must fit before predict")
        
        # Add regime information
        X_with_regimes = self.regime_detector.transform(X)
        
        predictions = np.zeros(len(X))
        
        for regime in range(self.regime_detector.n_regimes):
            regime_mask = X_with_regimes['regime'] == regime
            
            if regime in self.fitted_strategies_ and regime_mask.any():
                regime_X = X_with_regimes[regime_mask][X.columns]
                regime_predictions = self.fitted_strategies_[regime].predict(regime_X)
                predictions[regime_mask] = regime_predictions
        
        return predictions
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Score using regime-appropriate strategies."""
        predictions = self.predict(X)
        
        # Calculate regime-aware Sharpe ratio
        strategy_returns = predictions * y.shift(-1)
        return self._calculate_sharpe_ratio(X, strategy_returns)
    
    def get_regime_performance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get performance breakdown by regime."""
        X_with_regimes = self.regime_detector.transform(X)
        predictions = self.predict(X)
        strategy_returns = predictions * y.shift(-1)
        
        performance = []
        for regime in range(self.regime_detector.n_regimes):
            regime_mask = X_with_regimes['regime'] == regime
            regime_returns = strategy_returns[regime_mask]
            
            if len(regime_returns) > 0:
                performance.append({
                    'regime': regime,
                    'regime_name': X_with_regimes[regime_mask]['regime_name'].iloc[0],
                    'frequency': regime_mask.sum() / len(X),
                    'total_return': (1 + regime_returns).prod() - 1,
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'win_rate': (regime_returns > 0).mean(),
                    'avg_return': regime_returns.mean() * 252
                })
        
        return pd.DataFrame(performance)