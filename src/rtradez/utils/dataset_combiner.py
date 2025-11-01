"""Advanced dataset combination and alignment utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class DatasetCombiner:
    """
    Advanced dataset combination and alignment for multiple data sources.
    
    Features:
    - Intelligent temporal alignment
    - Missing data handling strategies
    - Feature engineering and scaling
    - Automatic outlier detection
    - Memory-efficient processing
    """
    
    def __init__(self, 
                 alignment_method: str = 'inner',
                 missing_data_strategy: str = 'forward_fill',
                 outlier_detection: bool = True,
                 feature_scaling: str = 'standard',
                 max_missing_ratio: float = 0.3):
        """
        Initialize dataset combiner.
        
        Args:
            alignment_method: How to align datasets ('inner', 'outer', 'left')
            missing_data_strategy: How to handle missing data ('forward_fill', 'interpolate', 'knn', 'drop')
            outlier_detection: Whether to detect and handle outliers
            feature_scaling: Scaling method ('standard', 'robust', 'none')
            max_missing_ratio: Maximum allowed missing data ratio per feature
        """
        self.alignment_method = alignment_method
        self.missing_data_strategy = missing_data_strategy
        self.outlier_detection = outlier_detection
        self.feature_scaling = feature_scaling
        self.max_missing_ratio = max_missing_ratio
        
        # Initialize processors
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        
        # Track combination statistics
        self.combination_stats = {}
        
    def combine_datasets(self, 
                        datasets: Dict[str, pd.DataFrame],
                        primary_dataset: Optional[str] = None,
                        feature_prefix: bool = True) -> pd.DataFrame:
        """
        Combine multiple datasets with intelligent alignment and preprocessing.
        
        Args:
            datasets: Dictionary of DataFrames by source name
            primary_dataset: Name of primary dataset for alignment
            feature_prefix: Whether to add source prefixes to feature names
        
        Returns:
            Combined and preprocessed DataFrame
        """
        if not datasets:
            raise ValueError("No datasets provided")
        
        logger.info(f"Combining {len(datasets)} datasets")
        
        # Validate datasets
        validated_datasets = self._validate_datasets(datasets)
        
        # Temporal alignment
        aligned_datasets = self._align_temporal_data(validated_datasets, primary_dataset)
        
        # Combine aligned datasets
        combined_data = self._merge_aligned_datasets(aligned_datasets, feature_prefix)
        
        # Handle missing data
        processed_data = self._handle_missing_data(combined_data)
        
        # Outlier detection and handling
        if self.outlier_detection:
            processed_data = self._handle_outliers(processed_data)
        
        # Feature scaling
        if self.feature_scaling != 'none':
            processed_data = self._scale_features(processed_data)
        
        # Update statistics
        self._update_combination_stats(datasets, processed_data)
        
        logger.info(f"Combined dataset: {len(processed_data)} rows, {len(processed_data.columns)} features")
        
        return processed_data
    
    def _validate_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and clean input datasets."""
        validated = {}
        
        for name, df in datasets.items():
            if df.empty:
                logger.warning(f"Dataset '{name}' is empty, skipping")
                continue
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.warning(f"Could not convert index to datetime for '{name}': {e}")
                    continue
            
            # Remove duplicated index entries
            if df.index.duplicated().any():
                logger.warning(f"Removing {df.index.duplicated().sum()} duplicate index entries from '{name}'")
                df = df[~df.index.duplicated(keep='last')]
            
            # Sort by index
            df = df.sort_index()
            
            # Remove all-NaN columns
            df = df.dropna(axis=1, how='all')
            
            if not df.empty:
                validated[name] = df
            else:
                logger.warning(f"Dataset '{name}' became empty after validation")
        
        return validated
    
    def _align_temporal_data(self, 
                           datasets: Dict[str, pd.DataFrame], 
                           primary_dataset: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Align datasets temporally based on common date range."""
        if not datasets:
            return {}
        
        # Determine date range
        if primary_dataset and primary_dataset in datasets:
            # Use primary dataset's date range
            primary_df = datasets[primary_dataset]
            start_date = primary_df.index.min()
            end_date = primary_df.index.max()
            logger.info(f"Using primary dataset '{primary_dataset}' date range: {start_date.date()} to {end_date.date()}")
        else:
            # Find overlapping date range
            if self.alignment_method == 'inner':
                start_date = max(df.index.min() for df in datasets.values())
                end_date = min(df.index.max() for df in datasets.values())
            elif self.alignment_method == 'outer':
                start_date = min(df.index.min() for df in datasets.values())
                end_date = max(df.index.max() for df in datasets.values())
            else:  # left - use first dataset
                first_df = list(datasets.values())[0]
                start_date = first_df.index.min()
                end_date = first_df.index.max()
            
            logger.info(f"Computed date range ({self.alignment_method}): {start_date.date()} to {end_date.date()}")
        
        # Create common business day index
        common_index = pd.bdate_range(start=start_date, end=end_date)
        
        # Align all datasets to common index
        aligned_datasets = {}
        for name, df in datasets.items():
            # Filter to date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df[mask]
            
            if filtered_df.empty:
                logger.warning(f"Dataset '{name}' has no data in common date range")
                continue
            
            # Reindex to common business days
            aligned_df = filtered_df.reindex(common_index)
            
            # Forward fill missing values from reindexing (up to 5 business days)
            aligned_df = aligned_df.fillna(method='ffill', limit=5)
            
            aligned_datasets[name] = aligned_df
            
            data_coverage = (~aligned_df.isnull()).mean().mean()
            logger.info(f"'{name}': {len(aligned_df)} rows, {data_coverage:.1%} data coverage")
        
        return aligned_datasets
    
    def _merge_aligned_datasets(self, 
                              aligned_datasets: Dict[str, pd.DataFrame], 
                              feature_prefix: bool) -> pd.DataFrame:
        """Merge aligned datasets into single DataFrame."""
        if not aligned_datasets:
            return pd.DataFrame()
        
        combined_dfs = []
        
        for name, df in aligned_datasets.items():
            # Add source prefix to column names if requested
            if feature_prefix:
                df_copy = df.copy()
                df_copy.columns = [f"{name}_{col}" for col in df_copy.columns]
                combined_dfs.append(df_copy)
            else:
                combined_dfs.append(df)
        
        # Concatenate horizontally
        combined_data = pd.concat(combined_dfs, axis=1)
        
        return combined_data
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data according to specified strategy."""
        if data.empty:
            return data
        
        # Remove features with too much missing data
        missing_ratios = data.isnull().sum() / len(data)
        features_to_drop = missing_ratios[missing_ratios > self.max_missing_ratio].index
        
        if len(features_to_drop) > 0:
            logger.info(f"Dropping {len(features_to_drop)} features with >{self.max_missing_ratio:.1%} missing data")
            data = data.drop(columns=features_to_drop)
        
        if data.empty:
            logger.warning("All features dropped due to missing data")
            return data
        
        # Apply missing data strategy
        if self.missing_data_strategy == 'forward_fill':
            data = data.fillna(method='ffill', limit=10)  # Limit forward fill
            data = data.fillna(method='bfill', limit=5)   # Backfill remaining
            
        elif self.missing_data_strategy == 'interpolate':
            # Linear interpolation for time series
            data = data.interpolate(method='linear', limit=10)
            data = data.fillna(method='bfill', limit=5)
            
        elif self.missing_data_strategy == 'knn':
            # KNN imputation for remaining missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                self.imputer = KNNImputer(n_neighbors=5)
                data[numeric_columns] = self.imputer.fit_transform(data[numeric_columns])
                
        elif self.missing_data_strategy == 'drop':
            # Drop rows with any missing values
            data = data.dropna()
        
        # Final cleanup - drop rows that are still all NaN
        data = data.dropna(how='all')
        
        remaining_missing = data.isnull().sum().sum()
        if remaining_missing > 0:
            logger.info(f"{remaining_missing} missing values remain after imputation")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using robust statistical methods."""
        if data.empty:
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return data
        
        outlier_counts = {}
        
        for col in numeric_columns:
            series = data[col].dropna()
            if len(series) == 0:
                continue
            
            # Use IQR method for outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds (1.5 * IQR is standard, 3.0 * IQR is more conservative)
            lower_bound = Q1 - 2.0 * IQR
            upper_bound = Q3 + 2.0 * IQR
            
            # Identify outliers
            outliers = (series < lower_bound) | (series > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_counts[col] = outlier_count
                
                # Cap outliers instead of removing them (preserves time series structure)
                data.loc[data[col] < lower_bound, col] = lower_bound
                data.loc[data[col] > upper_bound, col] = upper_bound
        
        if outlier_counts:
            total_outliers = sum(outlier_counts.values())
            logger.info(f"Capped {total_outliers} outliers across {len(outlier_counts)} features")
        
        return data
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features according to specified method."""
        if data.empty:
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return data
        
        # Initialize scaler
        if self.feature_scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.feature_scaling == 'robust':
            self.scaler = RobustScaler()
        else:
            return data
        
        # Fit and transform
        data[numeric_columns] = self.scaler.fit_transform(data[numeric_columns])
        
        logger.info(f"Scaled {len(numeric_columns)} numeric features using {self.feature_scaling} scaling")
        
        return data
    
    def _update_combination_stats(self, 
                                original_datasets: Dict[str, pd.DataFrame], 
                                combined_data: pd.DataFrame):
        """Update combination statistics."""
        self.combination_stats = {
            'original_datasets': len(original_datasets),
            'original_total_features': sum(len(df.columns) for df in original_datasets.values()),
            'combined_features': len(combined_data.columns),
            'combined_rows': len(combined_data),
            'data_coverage': (~combined_data.isnull()).mean().mean(),
            'memory_usage_mb': combined_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'feature_reduction_ratio': 1 - (len(combined_data.columns) / 
                                          sum(len(df.columns) for df in original_datasets.values())),
        }
        
        # Per-dataset statistics
        self.combination_stats['per_dataset'] = {}
        for name, df in original_datasets.items():
            self.combination_stats['per_dataset'][name] = {
                'original_features': len(df.columns),
                'original_rows': len(df),
                'date_range': f"{df.index.min().date()} to {df.index.max().date()}"
            }
    
    def get_combination_report(self) -> Dict[str, Any]:
        """Get detailed report of dataset combination process."""
        return {
            'combination_stats': self.combination_stats,
            'configuration': {
                'alignment_method': self.alignment_method,
                'missing_data_strategy': self.missing_data_strategy,
                'outlier_detection': self.outlier_detection,
                'feature_scaling': self.feature_scaling,
                'max_missing_ratio': self.max_missing_ratio
            },
            'processors_fitted': {
                'scaler': self.scaler is not None,
                'imputer': self.imputer is not None,
                'feature_selector': self.feature_selector is not None
            }
        }


class AdvancedFeatureSelector:
    """
    Advanced feature selection for combined datasets.
    
    Implements multiple selection strategies optimized for financial time series.
    """
    
    def __init__(self, 
                 method: str = 'combined',
                 max_features: Optional[int] = None,
                 correlation_threshold: float = 0.95,
                 variance_threshold: float = 0.01):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('correlation', 'variance', 'statistical', 'pca', 'combined')
            max_features: Maximum number of features to select
            correlation_threshold: Threshold for removing highly correlated features
            variance_threshold: Minimum variance threshold
        """
        self.method = method
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selection_report_ = {}
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit feature selector and transform data.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
        
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit feature selector."""
        if self.method == 'correlation':
            self._fit_correlation_filter(X)
        elif self.method == 'variance':
            self._fit_variance_filter(X)
        elif self.method == 'statistical':
            self._fit_statistical_selection(X, y)
        elif self.method == 'pca':
            self._fit_pca_selection(X)
        elif self.method == 'combined':
            self._fit_combined_selection(X, y)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted selector."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted before transform")
        
        return X[self.selected_features_]
    
    def _fit_correlation_filter(self, X: pd.DataFrame):
        """Remove highly correlated features."""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identify features to remove
        to_remove = set()
        for col in upper_tri.columns:
            correlated = upper_tri.index[upper_tri[col] > self.correlation_threshold].tolist()
            to_remove.update(correlated)
        
        self.selected_features_ = [col for col in X.columns if col not in to_remove]
        
        self.selection_report_['correlation_filter'] = {
            'removed_features': len(to_remove),
            'remaining_features': len(self.selected_features_),
            'removed_feature_names': list(to_remove)
        }
    
    def _fit_variance_filter(self, X: pd.DataFrame):
        """Remove low variance features."""
        # Calculate variance for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        variances = X[numeric_cols].var()
        
        # Select features above threshold
        selected_numeric = variances[variances > self.variance_threshold].index.tolist()
        non_numeric = [col for col in X.columns if col not in numeric_cols]
        
        self.selected_features_ = selected_numeric + non_numeric
        
        self.selection_report_['variance_filter'] = {
            'removed_features': len(numeric_cols) - len(selected_numeric),
            'remaining_features': len(self.selected_features_),
            'variance_threshold': self.variance_threshold
        }
    
    def _fit_statistical_selection(self, X: pd.DataFrame, y: Optional[pd.Series]):
        """Select features using statistical tests."""
        if y is None:
            # Fall back to correlation filter
            self._fit_correlation_filter(X)
            return
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.selected_features_ = list(X.columns)
            return
        
        # Align X and y
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # Remove rows with missing values
        valid_mask = ~(X_aligned[numeric_cols].isnull().any(axis=1) | y_aligned.isnull())
        X_clean = X_aligned[valid_mask]
        y_clean = y_aligned[valid_mask]
        
        if len(X_clean) == 0:
            self.selected_features_ = list(X.columns)
            return
        
        # Use SelectKBest with f_regression
        k = min(self.max_features or len(numeric_cols), len(numeric_cols))
        selector = SelectKBest(score_func=f_regression, k=k)
        
        try:
            selector.fit(X_clean[numeric_cols], y_clean)
            selected_numeric = numeric_cols[selector.get_support()].tolist()
            
            # Add non-numeric columns
            non_numeric = [col for col in X.columns if col not in numeric_cols]
            self.selected_features_ = selected_numeric + non_numeric
            
            self.feature_scores_ = dict(zip(numeric_cols, selector.scores_))
            
            self.selection_report_['statistical_selection'] = {
                'method': 'f_regression',
                'selected_features': len(selected_numeric),
                'total_numeric_features': len(numeric_cols),
                'target_correlation': y_clean.corr(X_clean[selected_numeric].mean(axis=1))
            }
            
        except Exception as e:
            logger.warning(f"Statistical selection failed: {e}, falling back to all features")
            self.selected_features_ = list(X.columns)
    
    def _fit_pca_selection(self, X: pd.DataFrame):
        """Select features using PCA analysis."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.selected_features_ = list(X.columns)
            return
        
        # Remove missing values
        X_clean = X[numeric_cols].dropna()
        
        if len(X_clean) == 0:
            self.selected_features_ = list(X.columns)
            return
        
        # Fit PCA
        n_components = min(self.max_features or len(numeric_cols), len(numeric_cols), len(X_clean))
        pca = PCA(n_components=n_components)
        pca.fit(X_clean)
        
        # Select features with highest loadings on first few components
        n_top_components = min(5, n_components)
        feature_importance = np.abs(pca.components_[:n_top_components]).mean(axis=0)
        
        # Select top features
        top_indices = np.argsort(feature_importance)[-n_components:]
        selected_numeric = numeric_cols[top_indices].tolist()
        
        # Add non-numeric columns
        non_numeric = [col for col in X.columns if col not in numeric_cols]
        self.selected_features_ = selected_numeric + non_numeric
        
        self.selection_report_['pca_selection'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_[:n_top_components].tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'selected_components': n_top_components,
            'selected_features': len(selected_numeric)
        }
    
    def _fit_combined_selection(self, X: pd.DataFrame, y: Optional[pd.Series]):
        """Apply combined selection strategy."""
        # Step 1: Variance filter
        self._fit_variance_filter(X)
        X_var_filtered = X[self.selected_features_]
        
        # Step 2: Correlation filter
        self._fit_correlation_filter(X_var_filtered)
        X_corr_filtered = X_var_filtered[self.selected_features_]
        
        # Step 3: Statistical selection (if target provided)
        if y is not None and len(X_corr_filtered.columns) > 0:
            self._fit_statistical_selection(X_corr_filtered, y)
        
        self.selection_report_['combined_selection'] = {
            'steps': ['variance_filter', 'correlation_filter', 'statistical_selection'],
            'final_features': len(self.selected_features_),
            'original_features': len(X.columns)
        }
    
    def get_feature_report(self) -> Dict[str, Any]:
        """Get detailed feature selection report."""
        return {
            'method': self.method,
            'selected_features': self.selected_features_,
            'feature_scores': self.feature_scores_,
            'selection_report': self.selection_report_,
            'configuration': {
                'max_features': self.max_features,
                'correlation_threshold': self.correlation_threshold,
                'variance_threshold': self.variance_threshold
            }
        }