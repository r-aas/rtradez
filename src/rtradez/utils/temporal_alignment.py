"""Advanced temporal alignment for datasets with different frequencies and shapes."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import warnings
from enum import Enum
import logging
from pydantic import BaseModel, Field, validator, ConfigDict

logger = logging.getLogger(__name__)


class FrequencyType(Enum):
    """Supported data frequencies."""
    TICK = "tick"           # Intraday tick data
    MINUTE = "minute"       # Minute-by-minute
    HOURLY = "hourly"       # Hourly
    DAILY = "daily"         # Daily (business days)
    WEEKLY = "weekly"       # Weekly
    MONTHLY = "monthly"     # Monthly
    QUARTERLY = "quarterly" # Quarterly
    ANNUAL = "annual"       # Annual
    IRREGULAR = "irregular" # Irregular/event-driven


class TemporalProfile(BaseModel):
    """Profile of a dataset's temporal characteristics with validation."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    frequency: FrequencyType = Field(..., description="Data frequency type")
    start_date: datetime = Field(..., description="Starting date of the dataset")
    end_date: datetime = Field(..., description="Ending date of the dataset")
    total_observations: int = Field(..., ge=0, description="Total number of observations")
    missing_periods: int = Field(..., ge=0, description="Number of missing periods")
    regularity_score: float = Field(..., ge=0, le=1, description="Regularity score (0-1)")
    timezone: Optional[str] = Field(None, description="Timezone identifier")
    business_hours_only: bool = Field(False, description="Whether data is business hours only")
    weekends_included: bool = Field(True, description="Whether weekends are included")
    
    @validator('end_date')
    def end_after_start(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('End date must be after start date')
        return v
    
    @validator('missing_periods')
    def missing_not_exceed_total(cls, v, values):
        if 'total_observations' in values and v > values['total_observations']:
            raise ValueError('Missing periods cannot exceed total observations')
        return v
    
    def coverage_ratio(self) -> float:
        """Calculate data coverage ratio."""
        if self.total_observations == 0:
            return 0.0
        expected_periods = self._calculate_expected_periods()
        return min(1.0, self.total_observations / expected_periods) if expected_periods > 0 else 0.0
    
    def _calculate_expected_periods(self) -> int:
        """Calculate expected number of periods for this frequency."""
        if self.frequency == FrequencyType.DAILY:
            total_days = (self.end_date - self.start_date).days + 1
            if not self.weekends_included:
                return total_days * 5 // 7  # Approximate business days
            return total_days
        elif self.frequency == FrequencyType.WEEKLY:
            return ((self.end_date - self.start_date).days // 7) + 1
        elif self.frequency == FrequencyType.MONTHLY:
            return ((self.end_date.year - self.start_date.year) * 12 + 
                   self.end_date.month - self.start_date.month) + 1
        elif self.frequency == FrequencyType.QUARTERLY:
            return ((self.end_date.year - self.start_date.year) * 4 + 
                   (self.end_date.month - 1) // 3 - (self.start_date.month - 1) // 3) + 1
        elif self.frequency == FrequencyType.ANNUAL:
            return self.end_date.year - self.start_date.year + 1
        else:
            return self.total_observations  # For irregular data


class TemporalAlignerConfig(BaseModel):
    """Configuration for temporal alignment with validation."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    target_frequency: FrequencyType = Field(FrequencyType.DAILY, description="Target frequency for alignment")
    alignment_method: str = Field('outer', description="Alignment method")
    fill_method: str = Field('forward_fill', description="Method to fill missing data")
    business_days_only: bool = Field(True, description="Whether to use business days only")
    max_fill_periods: int = Field(5, gt=0, description="Maximum periods to forward fill")
    
    @validator('alignment_method')
    def valid_alignment_method(cls, v):
        valid_methods = ['inner', 'outer', 'left', 'right']
        if v not in valid_methods:
            raise ValueError(f'Alignment method must be one of: {valid_methods}')
        return v
    
    @validator('fill_method')
    def valid_fill_method(cls, v):
        valid_methods = ['forward_fill', 'interpolate', 'carry_forward', 'none', 'zero', 'mean']
        if v not in valid_methods:
            raise ValueError(f'Fill method must be one of: {valid_methods}')
        return v


class TemporalAligner:
    """
    Advanced temporal alignment for multi-frequency, multi-shape datasets.
    
    Handles:
    - Different data frequencies (tick, minute, daily, weekly, monthly, quarterly, annual)
    - Irregular/event-driven data
    - Missing data patterns
    - Time zone conversions
    - Business vs calendar day alignment
    - Forward-fill, interpolation, and resampling strategies
    """
    
    def __init__(self, config: TemporalAlignerConfig):
        """
        Initialize temporal aligner with validated configuration.
        
        Args:
            config: Temporal alignment configuration
        """
        self.config = config
        self.target_frequency = config.target_frequency
        self.alignment_method = config.alignment_method
        self.fill_method = config.fill_method
        self.business_days_only = config.business_days_only
        self.max_fill_periods = config.max_fill_periods
        
        # Storage for analysis
        self.dataset_profiles_ = {}
        self.alignment_report_ = {}
        
    def analyze_temporal_patterns(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, TemporalProfile]:
        """Analyze temporal patterns of all datasets."""
        logger.info(f"Analyzing temporal patterns for {len(datasets)} datasets...")
        
        profiles = {}
        
        for name, df in datasets.items():
            if df.empty:
                continue
                
            profile = self._create_temporal_profile(name, df)
            profiles[name] = profile
            
            logger.info(f"{name}: {profile.frequency.value} frequency, "
                       f"{profile.total_observations} obs, "
                       f"{profile.coverage_ratio():.1%} coverage")
        
        self.dataset_profiles_ = profiles
        return profiles
    
    def _create_temporal_profile(self, name: str, df: pd.DataFrame) -> TemporalProfile:
        """Create temporal profile for a single dataset."""
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except:
                raise ValueError(f"Cannot convert index to datetime for dataset '{name}'")
        
        # Basic stats
        start_date = df.index.min()
        end_date = df.index.max()
        total_obs = len(df)
        
        # Detect frequency
        frequency = self._detect_frequency(df.index)
        
        # Calculate regularity
        regularity_score = self._calculate_regularity(df.index, frequency)
        
        # Count missing periods
        expected_index = self._generate_expected_index(start_date, end_date, frequency)
        missing_periods = len(expected_index) - len(df.index.intersection(expected_index))
        
        # Detect business hours and weekend patterns
        business_hours_only = self._detect_business_hours(df.index)
        weekends_included = self._detect_weekends(df.index)
        
        return TemporalProfile(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            total_observations=total_obs,
            missing_periods=missing_periods,
            regularity_score=regularity_score,
            business_hours_only=business_hours_only,
            weekends_included=weekends_included
        )
    
    def _detect_frequency(self, index: pd.DatetimeIndex) -> FrequencyType:
        """Detect the primary frequency of the data."""
        if len(index) < 2:
            return FrequencyType.IRREGULAR
        
        # Calculate time differences
        diffs = index.to_series().diff().dropna()
        
        # Get the most common difference
        mode_diff = diffs.mode()
        if len(mode_diff) == 0:
            return FrequencyType.IRREGULAR
        
        mode_seconds = mode_diff.iloc[0].total_seconds()
        
        # Map to frequency types
        if mode_seconds <= 60:  # <= 1 minute
            return FrequencyType.TICK
        elif mode_seconds <= 3600:  # <= 1 hour
            return FrequencyType.MINUTE
        elif mode_seconds <= 86400:  # <= 1 day
            return FrequencyType.HOURLY
        elif mode_seconds <= 7 * 86400:  # <= 1 week
            return FrequencyType.DAILY
        elif mode_seconds <= 31 * 86400:  # <= 1 month
            return FrequencyType.WEEKLY
        elif mode_seconds <= 95 * 86400:  # <= ~3 months
            return FrequencyType.MONTHLY
        elif mode_seconds <= 370 * 86400:  # <= ~1 year
            return FrequencyType.QUARTERLY
        else:
            return FrequencyType.ANNUAL
    
    def _calculate_regularity(self, index: pd.DatetimeIndex, frequency: FrequencyType) -> float:
        """Calculate how regular the data frequency is (0-1 score)."""
        if len(index) < 3:
            return 1.0
        
        diffs = index.to_series().diff().dropna()
        
        if frequency == FrequencyType.IRREGULAR:
            return 0.0
        
        # For regular frequencies, check consistency
        mode_diff = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else diffs.median()
        
        # Calculate how many observations match the expected frequency
        tolerance = mode_diff * 0.1  # 10% tolerance
        regular_count = ((diffs >= mode_diff - tolerance) & 
                        (diffs <= mode_diff + tolerance)).sum()
        
        return regular_count / len(diffs)
    
    def _detect_business_hours(self, index: pd.DatetimeIndex) -> bool:
        """Detect if data is limited to business hours."""
        if len(index) == 0:
            return False
        
        # Check if all timestamps are within typical business hours (9 AM - 4 PM)
        hours = index.hour
        business_hour_count = ((hours >= 9) & (hours <= 16)).sum()
        
        return business_hour_count / len(hours) > 0.8
    
    def _detect_weekends(self, index: pd.DatetimeIndex) -> bool:
        """Detect if weekends are included in the data."""
        if len(index) == 0:
            return False
        
        weekend_count = (index.weekday >= 5).sum()
        return weekend_count / len(index) > 0.1
    
    def _generate_expected_index(self, 
                               start_date: datetime, 
                               end_date: datetime, 
                               frequency: FrequencyType) -> pd.DatetimeIndex:
        """Generate expected datetime index for a frequency."""
        if frequency == FrequencyType.DAILY:
            if self.business_days_only:
                return pd.bdate_range(start=start_date, end=end_date)
            else:
                return pd.date_range(start=start_date, end=end_date, freq='D')
        elif frequency == FrequencyType.WEEKLY:
            return pd.date_range(start=start_date, end=end_date, freq='W')
        elif frequency == FrequencyType.MONTHLY:
            return pd.date_range(start=start_date, end=end_date, freq='M')
        elif frequency == FrequencyType.QUARTERLY:
            return pd.date_range(start=start_date, end=end_date, freq='Q')
        elif frequency == FrequencyType.ANNUAL:
            return pd.date_range(start=start_date, end=end_date, freq='Y')
        else:
            # For irregular or high-frequency data, return original
            return pd.date_range(start=start_date, end=end_date, freq='D')
    
    def align_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple datasets to common temporal framework.
        
        Args:
            datasets: Dictionary of DataFrames to align
            
        Returns:
            Dictionary of aligned DataFrames
        """
        logger.info(f"Aligning {len(datasets)} datasets to {self.target_frequency.value} frequency...")
        
        # Analyze temporal patterns first
        profiles = self.analyze_temporal_patterns(datasets)
        
        # Determine target index
        target_index = self._create_target_index(profiles)
        logger.info(f"Target index: {len(target_index)} periods from {target_index.min().date()} to {target_index.max().date()}")
        
        # Align each dataset
        aligned_datasets = {}
        alignment_stats = {}
        
        for name, df in datasets.items():
            if df.empty:
                continue
            
            logger.info(f"Aligning {name}...")
            aligned_df, stats = self._align_single_dataset(df, target_index, profiles[name])
            
            if aligned_df is not None and not aligned_df.empty:
                aligned_datasets[name] = aligned_df
                alignment_stats[name] = stats
                
                logger.info(f"  {name}: {len(df)} -> {len(aligned_df)} observations")
                logger.info(f"  Coverage: {stats['coverage_ratio']:.1%}, Missing filled: {stats['periods_filled']}")
        
        # Store alignment report
        self.alignment_report_ = {
            'target_index_length': len(target_index),
            'datasets_aligned': len(aligned_datasets),
            'alignment_stats': alignment_stats,
            'target_frequency': self.target_frequency.value
        }
        
        return aligned_datasets
    
    def _create_target_index(self, profiles: Dict[str, TemporalProfile]) -> pd.DatetimeIndex:
        """Create target datetime index for alignment."""
        if not profiles:
            return pd.DatetimeIndex([])
        
        # Determine date range based on alignment method
        if self.alignment_method == 'inner':
            # Use intersection of all date ranges
            start_date = max(profile.start_date for profile in profiles.values())
            end_date = min(profile.end_date for profile in profiles.values())
        elif self.alignment_method == 'outer':
            # Use union of all date ranges
            start_date = min(profile.start_date for profile in profiles.values())
            end_date = max(profile.end_date for profile in profiles.values())
        else:  # 'left' - use first dataset
            first_profile = list(profiles.values())[0]
            start_date = first_profile.start_date
            end_date = first_profile.end_date
        
        # Generate target index based on target frequency
        return self._generate_expected_index(start_date, end_date, self.target_frequency)
    
    def _align_single_dataset(self, 
                            df: pd.DataFrame, 
                            target_index: pd.DatetimeIndex,
                            profile: TemporalProfile) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Align a single dataset to target index."""
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='last')].sort_index()
        
        stats = {
            'original_length': len(df),
            'target_length': len(target_index),
            'coverage_ratio': 0.0,
            'periods_filled': 0,
            'resampling_method': 'none'
        }
        
        try:
            # Step 1: Handle frequency conversion if needed
            if profile.frequency != self.target_frequency:
                df = self._resample_to_target_frequency(df, profile.frequency, target_index)
                stats['resampling_method'] = f"{profile.frequency.value}_to_{self.target_frequency.value}"
            
            # Step 2: Reindex to target index
            aligned_df = df.reindex(target_index)
            
            # Step 3: Handle missing values
            if self.fill_method != 'none':
                original_missing = aligned_df.isnull().sum().sum()
                aligned_df = self._fill_missing_values(aligned_df)
                final_missing = aligned_df.isnull().sum().sum()
                stats['periods_filled'] = original_missing - final_missing
            
            # Update stats
            stats['final_length'] = len(aligned_df)
            stats['coverage_ratio'] = (~aligned_df.isnull()).mean().mean()
            
            return aligned_df, stats
            
        except Exception as e:
            logger.warning(f"Failed to align dataset: {e}")
            return None, stats
    
    def _resample_to_target_frequency(self, 
                                    df: pd.DataFrame, 
                                    source_frequency: FrequencyType,
                                    target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Resample dataset to target frequency."""
        
        if source_frequency == self.target_frequency:
            return df
        
        # Define resampling rules
        freq_map = {
            FrequencyType.DAILY: 'D',
            FrequencyType.WEEKLY: 'W',
            FrequencyType.MONTHLY: 'M',
            FrequencyType.QUARTERLY: 'Q',
            FrequencyType.ANNUAL: 'Y'
        }
        
        target_freq_str = freq_map.get(self.target_frequency)
        if target_freq_str is None:
            logger.warning(f"Cannot resample to {self.target_frequency.value}, returning original")
            return df
        
        # Adjust for business days
        if self.target_frequency == FrequencyType.DAILY and self.business_days_only:
            target_freq_str = 'B'
        
        try:
            # Determine resampling method based on frequency conversion
            if self._is_upsampling(source_frequency, self.target_frequency):
                # Upsampling: use forward fill or interpolation
                resampled = df.resample(target_freq_str).ffill()
            else:
                # Downsampling: use appropriate aggregation
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                
                resampled_parts = []
                
                if len(numeric_cols) > 0:
                    # For numeric columns, use mean for prices, last for other metrics
                    agg_dict = {}
                    for col in numeric_cols:
                        if any(price_word in col.lower() for price_word in ['price', 'close', 'open', 'high', 'low']):
                            agg_dict[col] = 'last'  # Use last price for OHLC data
                        elif 'volume' in col.lower():
                            agg_dict[col] = 'sum'   # Sum volume
                        else:
                            agg_dict[col] = 'mean'  # Mean for other metrics
                    
                    numeric_resampled = df[numeric_cols].resample(target_freq_str).agg(agg_dict)
                    resampled_parts.append(numeric_resampled)
                
                if len(non_numeric_cols) > 0:
                    # For non-numeric columns, use last value
                    non_numeric_resampled = df[non_numeric_cols].resample(target_freq_str).last()
                    resampled_parts.append(non_numeric_resampled)
                
                if resampled_parts:
                    resampled = pd.concat(resampled_parts, axis=1)
                else:
                    resampled = df.resample(target_freq_str).last()
            
            return resampled
            
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, returning original data")
            return df
    
    def _is_upsampling(self, source_freq: FrequencyType, target_freq: FrequencyType) -> bool:
        """Check if conversion requires upsampling (higher frequency)."""
        freq_order = {
            FrequencyType.ANNUAL: 0,
            FrequencyType.QUARTERLY: 1,
            FrequencyType.MONTHLY: 2,
            FrequencyType.WEEKLY: 3,
            FrequencyType.DAILY: 4,
            FrequencyType.HOURLY: 5,
            FrequencyType.MINUTE: 6,
            FrequencyType.TICK: 7
        }
        
        source_order = freq_order.get(source_freq, 4)  # Default to daily
        target_order = freq_order.get(target_freq, 4)
        
        return target_order > source_order
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values according to specified method."""
        if self.fill_method == 'none':
            return df
        
        filled_df = df.copy()
        
        if self.fill_method == 'forward_fill':
            # Forward fill with limit
            filled_df = filled_df.fillna(method='ffill', limit=self.max_fill_periods)
            
        elif self.fill_method == 'interpolate':
            # Linear interpolation for numeric columns
            numeric_cols = filled_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                filled_df[numeric_cols] = filled_df[numeric_cols].interpolate(
                    method='linear', limit=self.max_fill_periods
                )
            
            # Forward fill for non-numeric columns
            non_numeric_cols = filled_df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                filled_df[non_numeric_cols] = filled_df[non_numeric_cols].fillna(
                    method='ffill', limit=self.max_fill_periods
                )
        
        elif self.fill_method == 'carry_forward':
            # Carry forward previous value
            filled_df = filled_df.fillna(method='ffill')
        
        return filled_df
    
    def get_alignment_report(self) -> Dict[str, Any]:
        """Get detailed alignment report."""
        return {
            'dataset_profiles': {
                name: {
                    'frequency': profile.frequency.value,
                    'observations': profile.total_observations,
                    'coverage_ratio': profile.coverage_ratio(),
                    'regularity_score': profile.regularity_score,
                    'missing_periods': profile.missing_periods
                }
                for name, profile in self.dataset_profiles_.items()
            },
            'alignment_report': self.alignment_report_,
            'configuration': {
                'target_frequency': self.target_frequency.value,
                'alignment_method': self.alignment_method,
                'fill_method': self.fill_method,
                'business_days_only': self.business_days_only,
                'max_fill_periods': self.max_fill_periods
            }
        }


class DataShapeNormalizer:
    """
    Normalize datasets with different shapes and missing patterns.
    
    Handles:
    - Different column counts and types
    - Missing data patterns
    - Feature scaling and standardization
    - Categorical encoding
    """
    
    def __init__(self,
                 standardize_numeric: bool = True,
                 encode_categorical: bool = True,
                 max_missing_ratio: float = 0.5,
                 min_variance_threshold: float = 1e-6):
        """
        Initialize data shape normalizer.
        
        Args:
            standardize_numeric: Whether to standardize numeric features
            encode_categorical: Whether to encode categorical features
            max_missing_ratio: Maximum allowed missing ratio per feature
            min_variance_threshold: Minimum variance threshold for features
        """
        self.standardize_numeric = standardize_numeric
        self.encode_categorical = encode_categorical
        self.max_missing_ratio = max_missing_ratio
        self.min_variance_threshold = min_variance_threshold
        
        # Storage for fitted transformers
        self.scalers_ = {}
        self.encoders_ = {}
        self.feature_stats_ = {}
        
    def fit_transform(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Fit normalizers and transform datasets."""
        return self.fit(datasets).transform(datasets)
    
    def fit(self, datasets: Dict[str, pd.DataFrame]) -> 'DataShapeNormalizer':
        """Fit normalizers on datasets."""
        logger.info(f"Fitting normalizers for {len(datasets)} datasets...")
        
        for name, df in datasets.items():
            if df.empty:
                continue
            
            logger.info(f"Analyzing {name}: {len(df)} rows, {len(df.columns)} columns")
            
            # Analyze feature statistics
            self.feature_stats_[name] = self._analyze_features(df)
            
            # Fit scalers for numeric columns
            if self.standardize_numeric:
                self._fit_scalers(name, df)
            
            # Fit encoders for categorical columns
            if self.encode_categorical:
                self._fit_encoders(name, df)
        
        return self
    
    def transform(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform datasets using fitted normalizers."""
        transformed_datasets = {}
        
        for name, df in datasets.items():
            if df.empty:
                continue
            
            logger.info(f"Transforming {name}...")
            transformed_df = self._transform_single_dataset(name, df)
            transformed_datasets[name] = transformed_df
        
        return transformed_datasets
    
    def _analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature characteristics."""
        stats = {
            'numeric_columns': [],
            'categorical_columns': [],
            'missing_ratios': {},
            'variance_stats': {},
            'data_types': {}
        }
        
        for col in df.columns:
            # Missing ratio
            missing_ratio = df[col].isnull().sum() / len(df)
            stats['missing_ratios'][col] = missing_ratio
            
            # Data type
            dtype = str(df[col].dtype)
            stats['data_types'][col] = dtype
            
            # Categorize column type
            if pd.api.types.is_numeric_dtype(df[col]):
                stats['numeric_columns'].append(col)
                # Calculate variance for numeric columns
                variance = df[col].var()
                stats['variance_stats'][col] = variance if not pd.isna(variance) else 0.0
            else:
                stats['categorical_columns'].append(col)
        
        return stats
    
    def _fit_scalers(self, dataset_name: str, df: pd.DataFrame):
        """Fit scalers for numeric columns."""
        from sklearn.preprocessing import StandardScaler, RobustScaler
        
        stats = self.feature_stats_[dataset_name]
        numeric_cols = stats['numeric_columns']
        
        if not numeric_cols:
            return
        
        # Filter columns by missing ratio and variance
        valid_numeric_cols = []
        for col in numeric_cols:
            missing_ratio = stats['missing_ratios'][col]
            variance = stats['variance_stats'][col]
            
            if (missing_ratio <= self.max_missing_ratio and 
                variance >= self.min_variance_threshold):
                valid_numeric_cols.append(col)
        
        if valid_numeric_cols:
            # Use RobustScaler for financial data (handles outliers better)
            scaler = RobustScaler()
            
            # Fit on non-null values
            valid_data = df[valid_numeric_cols].dropna()
            if len(valid_data) > 0:
                scaler.fit(valid_data)
                self.scalers_[dataset_name] = {
                    'scaler': scaler,
                    'columns': valid_numeric_cols
                }
                
                logger.info(f"  Fitted scaler for {len(valid_numeric_cols)} numeric columns")
    
    def _fit_encoders(self, dataset_name: str, df: pd.DataFrame):
        """Fit encoders for categorical columns."""
        from sklearn.preprocessing import LabelEncoder
        
        stats = self.feature_stats_[dataset_name]
        categorical_cols = stats['categorical_columns']
        
        if not categorical_cols:
            return
        
        encoders = {}
        
        for col in categorical_cols:
            missing_ratio = stats['missing_ratios'][col]
            
            if missing_ratio <= self.max_missing_ratio:
                # Check if column has reasonable number of unique values
                unique_count = df[col].nunique()
                total_count = len(df)
                
                if unique_count < total_count * 0.5:  # Less than 50% unique values
                    encoder = LabelEncoder()
                    
                    # Fit on non-null values
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        encoder.fit(non_null_values)
                        encoders[col] = encoder
        
        if encoders:
            self.encoders_[dataset_name] = encoders
            logger.info(f"  Fitted encoders for {len(encoders)} categorical columns")
    
    def _transform_single_dataset(self, dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a single dataset."""
        transformed_df = df.copy()
        
        # Apply scaling to numeric columns
        if dataset_name in self.scalers_:
            scaler_info = self.scalers_[dataset_name]
            scaler = scaler_info['scaler']
            columns = scaler_info['columns']
            
            # Transform only columns that exist in current dataset
            existing_columns = [col for col in columns if col in transformed_df.columns]
            
            if existing_columns:
                # Handle missing values before scaling
                for col in existing_columns:
                    # Fill missing values with median before scaling
                    if transformed_df[col].isnull().any():
                        median_val = transformed_df[col].median()
                        transformed_df[col] = transformed_df[col].fillna(median_val)
                
                # Apply scaling
                transformed_df[existing_columns] = scaler.transform(transformed_df[existing_columns])
        
        # Apply encoding to categorical columns
        if dataset_name in self.encoders_:
            encoders = self.encoders_[dataset_name]
            
            for col, encoder in encoders.items():
                if col in transformed_df.columns:
                    # Handle missing values
                    non_null_mask = transformed_df[col].notnull()
                    
                    if non_null_mask.any():
                        # Transform non-null values
                        transformed_values = transformed_df[col].copy()
                        transformed_values[non_null_mask] = encoder.transform(
                            transformed_df[col][non_null_mask]
                        )
                        transformed_df[col] = transformed_values
        
        return transformed_df
    
    def get_normalization_report(self) -> Dict[str, Any]:
        """Get detailed normalization report."""
        return {
            'feature_statistics': self.feature_stats_,
            'scalers_fitted': list(self.scalers_.keys()),
            'encoders_fitted': list(self.encoders_.keys()),
            'configuration': {
                'standardize_numeric': self.standardize_numeric,
                'encode_categorical': self.encode_categorical,
                'max_missing_ratio': self.max_missing_ratio,
                'min_variance_threshold': self.min_variance_threshold
            }
        }