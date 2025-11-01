"""Time bucketing and resampling utilities for multi-frequency financial data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta, time
import warnings
from enum import Enum
import pytz
import logging
from pydantic import BaseModel, Field, validator, ConfigDict

logger = logging.getLogger(__name__)


class BucketType(Enum):
    """Time bucket types for different use cases."""
    CALENDAR = "calendar"           # Calendar-based (daily, weekly, monthly)
    TRADING = "trading"             # Trading session-based
    ROLLING = "rolling"             # Rolling time windows
    EVENT_DRIVEN = "event_driven"   # Event-driven bucketing
    VOLATILITY = "volatility"       # Volatility-based dynamic bucketing
    VOLUME = "volume"               # Volume-based bucketing


class BucketConfig(BaseModel):
    """Configuration for time bucketing with validation."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid', arbitrary_types_allowed=True)
    
    bucket_type: BucketType = Field(..., description="Type of time bucketing")
    bucket_size: str = Field(..., min_length=1, description="Bucket size specification")
    aggregation_method: str = Field("last", description="Aggregation method")
    align_to: str = Field("start", description="Alignment strategy")
    timezone: str = Field("UTC", description="Timezone identifier")
    business_days_only: bool = Field(True, description="Use business days only")
    session_start: time = Field(time(9, 30), description="Market session start time")
    session_end: time = Field(time(16, 0), description="Market session end time")
    
    # Advanced options
    overlap_ratio: float = Field(0.0, ge=0, le=1, description="Overlap ratio for rolling buckets")
    min_observations: int = Field(1, gt=0, description="Minimum observations per bucket")
    handle_gaps: bool = Field(True, description="Handle market gaps and holidays")
    custom_aggregator: Optional[Callable] = Field(None, description="Custom aggregation function")
    
    @validator('aggregation_method')
    def valid_aggregation_method(cls, v):
        valid_methods = ['last', 'first', 'mean', 'median', 'sum', 'min', 'max', 'std', 'var', 'ohlc', 'custom']
        if v not in valid_methods:
            raise ValueError(f'Aggregation method must be one of: {valid_methods}')
        return v
    
    @validator('align_to')
    def valid_align_to(cls, v):
        valid_alignments = ['start', 'end', 'center']
        if v not in valid_alignments:
            raise ValueError(f'Alignment must be one of: {valid_alignments}')
        return v
    
    @validator('timezone')
    def valid_timezone(cls, v):
        try:
            pytz.timezone(v)
        except Exception:
            raise ValueError(f'Invalid timezone: {v}')
        return v
    
    @validator('session_end')
    def session_end_after_start(cls, v, values):
        if 'session_start' in values and v <= values['session_start']:
            raise ValueError('Session end must be after session start')
        return v


class TimeBucketing:
    """
    Advanced time bucketing and resampling for financial data.
    
    Features:
    - Multiple bucket types (calendar, trading session, rolling, event-driven)
    - Intelligent aggregation methods
    - Timezone handling
    - Market hours and holiday awareness
    - Volume and volatility-based dynamic bucketing
    - Custom aggregation functions
    """
    
    def __init__(self, config: BucketConfig):
        """Initialize time bucketing with configuration."""
        self.config = config
        self.timezone = pytz.timezone(config.timezone)
        
        # Market calendar setup
        self.market_holidays = self._get_market_holidays()
        
        # Cache for computed buckets
        self.bucket_cache = {}
        
    def bucket_data(self, 
                   data: pd.DataFrame,
                   timestamp_col: Optional[str] = None) -> pd.DataFrame:
        """
        Bucket time series data according to configuration.
        
        Args:
            data: DataFrame with datetime index or timestamp column
            timestamp_col: Name of timestamp column (if not using index)
            
        Returns:
            Bucketed DataFrame
        """
        logger.info(f"Bucketing data with {self.config.bucket_type.value} method")
        
        # Prepare data
        df = data.copy()
        if timestamp_col:
            df = df.set_index(timestamp_col)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Apply timezone if needed
        if df.index.tz is None:
            df.index = df.index.tz_localize(self.timezone)
        elif df.index.tz != self.timezone:
            df.index = df.index.tz_convert(self.timezone)
        
        # Route to appropriate bucketing method
        if self.config.bucket_type == BucketType.CALENDAR:
            return self._calendar_bucketing(df)
        elif self.config.bucket_type == BucketType.TRADING:
            return self._trading_session_bucketing(df)
        elif self.config.bucket_type == BucketType.ROLLING:
            return self._rolling_bucketing(df)
        elif self.config.bucket_type == BucketType.EVENT_DRIVEN:
            return self._event_driven_bucketing(df)
        elif self.config.bucket_type == BucketType.VOLATILITY:
            return self._volatility_bucketing(df)
        elif self.config.bucket_type == BucketType.VOLUME:
            return self._volume_bucketing(df)
        else:
            raise ValueError(f"Unsupported bucket type: {self.config.bucket_type}")
    
    def _calendar_bucketing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard calendar-based bucketing."""
        logger.info(f"Calendar bucketing with {self.config.bucket_size} intervals")
        
        # Filter to business days if required
        if self.config.business_days_only:
            df = df[df.index.weekday < 5]  # Monday=0, Friday=4
        
        # Create resampling frequency
        freq = self.config.bucket_size
        
        # Apply resampling with appropriate aggregation
        bucketed = self._apply_aggregation(df, freq)
        
        return bucketed
    
    def _trading_session_bucketing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trading session-based bucketing."""
        logger.info("Trading session bucketing")
        
        # Filter to trading hours
        df_trading = self._filter_trading_hours(df)
        
        # Split by trading sessions
        sessions = self._split_trading_sessions(df_trading)
        
        # Bucket within each session
        session_buckets = []
        
        for session_date, session_data in sessions.items():
            if len(session_data) < self.config.min_observations:
                continue
            
            # Create intra-session buckets
            session_freq = self.config.bucket_size
            session_bucketed = self._apply_aggregation(session_data, session_freq)
            
            # Add session metadata
            session_bucketed['session_date'] = session_date
            session_buckets.append(session_bucketed)
        
        if not session_buckets:
            return pd.DataFrame()
        
        # Combine all sessions
        combined = pd.concat(session_buckets)
        
        return combined
    
    def _rolling_bucketing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window bucketing."""
        logger.info(f"Rolling bucketing with {self.config.bucket_size} windows")
        
        # Parse bucket size for rolling windows
        window_size = self._parse_bucket_size_for_rolling(self.config.bucket_size)
        
        if window_size is None:
            raise ValueError(f"Invalid bucket size for rolling: {self.config.bucket_size}")
        
        # Calculate step size based on overlap
        step_size = max(1, int(window_size * (1 - self.config.overlap_ratio)))
        
        # Create rolling buckets
        rolling_buckets = []
        
        for i in range(0, len(df) - window_size + 1, step_size):
            window_data = df.iloc[i:i + window_size]
            
            if len(window_data) < self.config.min_observations:
                continue
            
            # Aggregate window
            if self.config.aggregation_method == "ohlc" and self._has_price_data(window_data):
                bucket_result = self._create_ohlc_bucket(window_data)
            else:
                bucket_result = self._aggregate_window(window_data)
            
            # Set bucket timestamp based on alignment
            if self.config.align_to == "start":
                bucket_timestamp = window_data.index[0]
            elif self.config.align_to == "end":
                bucket_timestamp = window_data.index[-1]
            else:  # center
                mid_idx = len(window_data) // 2
                bucket_timestamp = window_data.index[mid_idx]
            
            bucket_result.name = bucket_timestamp
            rolling_buckets.append(bucket_result)
        
        if not rolling_buckets:
            return pd.DataFrame()
        
        # Combine buckets
        result = pd.DataFrame(rolling_buckets)
        result.index.name = df.index.name
        
        return result
    
    def _event_driven_bucketing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Event-driven bucketing based on significant changes."""
        logger.info("Event-driven bucketing")
        
        if df.empty:
            return df
        
        # Use price or volume changes to define events
        price_cols = [col for col in df.columns if any(word in col.lower() for word in ['price', 'close', 'last'])]
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        
        event_column = None
        if price_cols:
            event_column = price_cols[0]
        elif volume_cols:
            event_column = volume_cols[0]
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                event_column = numeric_cols[0]
        
        if event_column is None:
            logger.warning("No suitable column for event-driven bucketing, falling back to calendar")
            return self._calendar_bucketing(df)
        
        # Calculate percentage changes
        pct_changes = df[event_column].pct_change().fillna(0)
        
        # Define event threshold (e.g., 1% change)
        threshold = self._parse_event_threshold(self.config.bucket_size)
        
        # Identify events
        events = np.abs(pct_changes) >= threshold
        event_indices = df.index[events]
        
        # Create buckets between events
        event_buckets = []
        
        start_idx = 0
        for event_time in event_indices:
            end_idx = df.index.get_loc(event_time) + 1
            
            if end_idx - start_idx >= self.config.min_observations:
                bucket_data = df.iloc[start_idx:end_idx]
                bucket_result = self._aggregate_window(bucket_data)
                bucket_result.name = event_time
                event_buckets.append(bucket_result)
            
            start_idx = end_idx
        
        # Handle remaining data
        if start_idx < len(df):
            bucket_data = df.iloc[start_idx:]
            if len(bucket_data) >= self.config.min_observations:
                bucket_result = self._aggregate_window(bucket_data)
                bucket_result.name = df.index[-1]
                event_buckets.append(bucket_result)
        
        if not event_buckets:
            return pd.DataFrame()
        
        result = pd.DataFrame(event_buckets)
        result.index.name = df.index.name
        
        return result
    
    def _volatility_bucketing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based dynamic bucketing."""
        logger.info("Volatility-based bucketing")
        
        # Find price column for volatility calculation
        price_cols = [col for col in df.columns if any(word in col.lower() for word in ['price', 'close', 'last'])]
        
        if not price_cols:
            logger.warning("No price column found for volatility bucketing, falling back to calendar")
            return self._calendar_bucketing(df)
        
        price_col = price_cols[0]
        
        # Calculate rolling volatility
        returns = df[price_col].pct_change().fillna(0)
        volatility_window = self._parse_volatility_window(self.config.bucket_size)
        rolling_vol = returns.rolling(volatility_window).std()
        
        # Create volatility-based buckets
        vol_quantiles = rolling_vol.quantile([0.33, 0.67])
        
        # Define bucket boundaries based on volatility
        low_vol_mask = rolling_vol <= vol_quantiles.iloc[0]
        high_vol_mask = rolling_vol >= vol_quantiles.iloc[1]
        med_vol_mask = ~(low_vol_mask | high_vol_mask)
        
        # Create buckets for each volatility regime
        vol_buckets = []
        
        for vol_type, mask in [("low", low_vol_mask), ("medium", med_vol_mask), ("high", high_vol_mask)]:
            vol_data = df[mask]
            
            if len(vol_data) >= self.config.min_observations:
                # Further bucket by time within volatility regime
                if len(vol_data) > 100:  # If enough data, create sub-buckets
                    sub_freq = "4H" if vol_type == "high" else "1D"  # Shorter buckets for high vol
                    sub_bucketed = self._apply_aggregation(vol_data, sub_freq)
                    
                    # Add volatility regime info
                    sub_bucketed['volatility_regime'] = vol_type
                    vol_buckets.append(sub_bucketed)
                else:
                    # Single bucket for this regime
                    bucket_result = self._aggregate_window(vol_data)
                    bucket_result['volatility_regime'] = vol_type
                    bucket_result.name = vol_data.index[-1]
                    vol_buckets.append(pd.DataFrame([bucket_result]))
        
        if not vol_buckets:
            return pd.DataFrame()
        
        result = pd.concat(vol_buckets).sort_index()
        
        return result
    
    def _volume_bucketing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based bucketing."""
        logger.info("Volume-based bucketing")
        
        # Find volume column
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        
        if not volume_cols:
            logger.warning("No volume column found for volume bucketing, falling back to calendar")
            return self._calendar_bucketing(df)
        
        volume_col = volume_cols[0]
        
        # Parse volume threshold
        volume_threshold = self._parse_volume_threshold(self.config.bucket_size)
        
        # Create volume-based buckets
        volume_buckets = []
        current_volume = 0
        bucket_start_idx = 0
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_volume += row[volume_col]
            
            if current_volume >= volume_threshold or i == len(df) - 1:
                # Create bucket
                bucket_data = df.iloc[bucket_start_idx:i + 1]
                
                if len(bucket_data) >= self.config.min_observations:
                    bucket_result = self._aggregate_window(bucket_data)
                    bucket_result['total_volume'] = current_volume
                    bucket_result.name = timestamp
                    volume_buckets.append(bucket_result)
                
                # Reset for next bucket
                current_volume = 0
                bucket_start_idx = i + 1
        
        if not volume_buckets:
            return pd.DataFrame()
        
        result = pd.DataFrame(volume_buckets)
        result.index.name = df.index.name
        
        return result
    
    def _apply_aggregation(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Apply aggregation method to resampled data."""
        
        if self.config.aggregation_method == "last":
            return df.resample(freq).last()
        
        elif self.config.aggregation_method == "mean":
            return df.resample(freq).mean()
        
        elif self.config.aggregation_method == "sum":
            return df.resample(freq).sum()
        
        elif self.config.aggregation_method == "ohlc":
            if self._has_price_data(df):
                return self._create_ohlc_resample(df, freq)
            else:
                # Fall back to last for non-price data
                return df.resample(freq).last()
        
        elif self.config.aggregation_method == "custom" and self.config.custom_aggregator:
            return df.resample(freq).apply(self.config.custom_aggregator)
        
        else:
            # Default to last
            return df.resample(freq).last()
    
    def _aggregate_window(self, window_data: pd.DataFrame) -> pd.Series:
        """Aggregate a single window of data."""
        if window_data.empty:
            return pd.Series(dtype=float)
        
        if self.config.aggregation_method == "last":
            return window_data.iloc[-1]
        
        elif self.config.aggregation_method == "mean":
            return window_data.mean()
        
        elif self.config.aggregation_method == "sum":
            return window_data.sum()
        
        elif self.config.aggregation_method == "ohlc":
            if self._has_price_data(window_data):
                return self._create_ohlc_window(window_data)
            else:
                return window_data.iloc[-1]
        
        elif self.config.aggregation_method == "custom" and self.config.custom_aggregator:
            return self.config.custom_aggregator(window_data)
        
        else:
            return window_data.iloc[-1]
    
    def _has_price_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has price-like columns."""
        price_keywords = ['open', 'high', 'low', 'close', 'price']
        return any(any(keyword in col.lower() for keyword in price_keywords) for col in df.columns)
    
    def _create_ohlc_resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Create OHLC data from resampled price data."""
        price_cols = [col for col in df.columns if any(word in col.lower() for word in ['price', 'close', 'last'])]
        
        if not price_cols:
            return df.resample(freq).last()
        
        price_col = price_cols[0]
        
        # Create OHLC for price column
        ohlc = df[price_col].resample(freq).ohlc()
        
        # Add other columns with appropriate aggregation
        other_cols = [col for col in df.columns if col != price_col]
        if other_cols:
            other_data = df[other_cols].resample(freq).agg({
                col: 'sum' if 'volume' in col.lower() else 'last'
                for col in other_cols
            })
            
            result = pd.concat([ohlc, other_data], axis=1)
        else:
            result = ohlc
        
        return result
    
    def _create_ohlc_window(self, window_data: pd.DataFrame) -> pd.Series:
        """Create OHLC data from a window of price data."""
        price_cols = [col for col in window_data.columns if any(word in col.lower() for word in ['price', 'close', 'last'])]
        
        if not price_cols:
            return window_data.iloc[-1]
        
        price_col = price_cols[0]
        prices = window_data[price_col]
        
        result = pd.Series({
            f'{price_col}_open': prices.iloc[0],
            f'{price_col}_high': prices.max(),
            f'{price_col}_low': prices.min(),
            f'{price_col}_close': prices.iloc[-1]
        })
        
        # Add other columns
        other_cols = [col for col in window_data.columns if col != price_col]
        for col in other_cols:
            if 'volume' in col.lower():
                result[col] = window_data[col].sum()
            else:
                result[col] = window_data[col].iloc[-1]
        
        return result
    
    def _create_ohlc_bucket(self, bucket_data: pd.DataFrame) -> pd.Series:
        """Create OHLC bucket from price data."""
        return self._create_ohlc_window(bucket_data)
    
    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to trading hours."""
        trading_mask = (
            (df.index.time >= self.config.session_start) &
            (df.index.time <= self.config.session_end) &
            (df.index.weekday < 5)  # Weekdays only
        )
        
        return df[trading_mask]
    
    def _split_trading_sessions(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data by trading sessions."""
        sessions = {}
        
        for date, group in df.groupby(df.index.date):
            if len(group) > 0:
                sessions[date] = group
        
        return sessions
    
    def _parse_bucket_size_for_rolling(self, bucket_size: str) -> Optional[int]:
        """Parse bucket size for rolling windows."""
        try:
            if bucket_size.endswith('T') or bucket_size.endswith('min'):
                # Time-based rolling not directly supported, convert to approximate row count
                return 10  # Default window size
            elif bucket_size.isdigit():
                return int(bucket_size)
            else:
                # Try to parse as number followed by unit
                import re
                match = re.match(r'(\d+)', bucket_size)
                if match:
                    return int(match.group(1))
                return None
        except:
            return None
    
    def _parse_event_threshold(self, bucket_size: str) -> float:
        """Parse event threshold from bucket size."""
        try:
            if '%' in bucket_size:
                return float(bucket_size.replace('%', '')) / 100
            else:
                # Default threshold
                return 0.01  # 1%
        except:
            return 0.01
    
    def _parse_volatility_window(self, bucket_size: str) -> int:
        """Parse volatility window size."""
        try:
            import re
            match = re.match(r'(\d+)', bucket_size)
            if match:
                return int(match.group(1))
            return 20  # Default window
        except:
            return 20
    
    def _parse_volume_threshold(self, bucket_size: str) -> float:
        """Parse volume threshold from bucket size."""
        try:
            if 'k' in bucket_size.lower():
                return float(bucket_size.lower().replace('k', '')) * 1000
            elif 'm' in bucket_size.lower():
                return float(bucket_size.lower().replace('m', '')) * 1000000
            else:
                return float(bucket_size)
        except:
            return 1000000  # Default 1M
    
    def _get_market_holidays(self) -> List[datetime]:
        """Get market holidays (simplified implementation)."""
        # This would typically integrate with a market calendar service
        # For now, return empty list
        return []
    
    def get_bucket_statistics(self, original_data: pd.DataFrame, bucketed_data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the bucketing process."""
        if original_data.empty or bucketed_data.empty:
            return {}
        
        compression_ratio = len(bucketed_data) / len(original_data)
        
        # Time span analysis
        original_span = original_data.index.max() - original_data.index.min()
        bucketed_span = bucketed_data.index.max() - bucketed_data.index.min()
        
        # Data coverage
        coverage_ratio = bucketed_span / original_span if original_span.total_seconds() > 0 else 0
        
        return {
            'original_observations': len(original_data),
            'bucketed_observations': len(bucketed_data),
            'compression_ratio': compression_ratio,
            'time_coverage_ratio': coverage_ratio,
            'bucket_type': self.config.bucket_type.value,
            'bucket_size': self.config.bucket_size,
            'aggregation_method': self.config.aggregation_method
        }


class TimeResamplingUtility:
    """
    Utility for common time resampling operations.
    
    Provides convenient methods for typical financial data resampling needs.
    """
    
    @staticmethod
    def resample_to_daily(df: pd.DataFrame, method: str = 'last') -> pd.DataFrame:
        """Resample data to daily frequency."""
        config = BucketConfig(
            bucket_type=BucketType.CALENDAR,
            bucket_size='1D',
            aggregation_method=method,
            business_days_only=True
        )
        
        bucketer = TimeBucketing(config)
        return bucketer.bucket_data(df)
    
    @staticmethod
    def resample_to_weekly(df: pd.DataFrame, method: str = 'last') -> pd.DataFrame:
        """Resample data to weekly frequency."""
        config = BucketConfig(
            bucket_type=BucketType.CALENDAR,
            bucket_size='1W',
            aggregation_method=method,
            business_days_only=True
        )
        
        bucketer = TimeBucketing(config)
        return bucketer.bucket_data(df)
    
    @staticmethod
    def resample_to_monthly(df: pd.DataFrame, method: str = 'last') -> pd.DataFrame:
        """Resample data to monthly frequency."""
        config = BucketConfig(
            bucket_type=BucketType.CALENDAR,
            bucket_size='1M',
            aggregation_method=method,
            business_days_only=True
        )
        
        bucketer = TimeBucketing(config)
        return bucketer.bucket_data(df)
    
    @staticmethod
    def create_ohlc_bars(df: pd.DataFrame, frequency: str = '1H') -> pd.DataFrame:
        """Create OHLC bars from tick/minute data."""
        config = BucketConfig(
            bucket_type=BucketType.CALENDAR,
            bucket_size=frequency,
            aggregation_method='ohlc',
            business_days_only=True
        )
        
        bucketer = TimeBucketing(config)
        return bucketer.bucket_data(df)
    
    @staticmethod
    def create_trading_session_buckets(df: pd.DataFrame, 
                                     session_start: time = time(9, 30),
                                     session_end: time = time(16, 0)) -> pd.DataFrame:
        """Create buckets based on trading sessions."""
        config = BucketConfig(
            bucket_type=BucketType.TRADING,
            bucket_size='1H',
            aggregation_method='ohlc',
            session_start=session_start,
            session_end=session_end
        )
        
        bucketer = TimeBucketing(config)
        return bucketer.bucket_data(df)
    
    @staticmethod
    def create_volume_bars(df: pd.DataFrame, volume_threshold: str = '1M') -> pd.DataFrame:
        """Create volume-based bars."""
        config = BucketConfig(
            bucket_type=BucketType.VOLUME,
            bucket_size=volume_threshold,
            aggregation_method='ohlc'
        )
        
        bucketer = TimeBucketing(config)
        return bucketer.bucket_data(df)
    
    @staticmethod
    def align_multiple_timeframes(datasets: Dict[str, pd.DataFrame], 
                                target_frequency: str = '1D') -> Dict[str, pd.DataFrame]:
        """Align multiple datasets to common timeframe."""
        aligned_datasets = {}
        
        for name, df in datasets.items():
            config = BucketConfig(
                bucket_type=BucketType.CALENDAR,
                bucket_size=target_frequency,
                aggregation_method='last',
                business_days_only=True
            )
            
            bucketer = TimeBucketing(config)
            aligned_df = bucketer.bucket_data(df)
            aligned_datasets[name] = aligned_df
        
        return aligned_datasets