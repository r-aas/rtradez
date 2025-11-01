"""Comprehensive demonstration of advanced temporal alignment capabilities."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import warnings
warnings.filterwarnings('ignore')

from rtradez.utils.temporal_alignment import TemporalAligner, DataShapeNormalizer, FrequencyType
from rtradez.utils.time_bucketing import TimeBucketing, BucketConfig, BucketType, TimeResamplingUtility


def create_multi_frequency_datasets():
    """Create datasets with different frequencies for demonstration."""
    print("📊 Creating multi-frequency datasets...")
    
    datasets = {}
    
    # 1. Daily stock data (business days)
    print("   📈 Daily stock data...")
    daily_dates = pd.bdate_range(start='2023-01-01', end='2024-12-31', freq='B')
    np.random.seed(42)
    
    stock_prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(daily_dates)))
    datasets['daily_stocks'] = pd.DataFrame({
        'close': stock_prices,
        'volume': np.random.exponential(1000000, len(daily_dates)),
        'high': stock_prices * (1 + np.abs(np.random.normal(0, 0.01, len(daily_dates)))),
        'low': stock_prices * (1 - np.abs(np.random.normal(0, 0.01, len(daily_dates))))
    }, index=daily_dates)
    
    # 2. Weekly economic data
    print("   🏛️ Weekly economic indicators...")
    weekly_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
    datasets['weekly_economic'] = pd.DataFrame({
        'unemployment_rate': 4.0 + np.cumsum(np.random.normal(0, 0.05, len(weekly_dates))),
        'inflation_rate': 3.0 + np.cumsum(np.random.normal(0, 0.02, len(weekly_dates))),
        'gdp_growth': 2.5 + np.cumsum(np.random.normal(0, 0.1, len(weekly_dates)))
    }, index=weekly_dates)
    
    # 3. Monthly sentiment data
    print("   😊 Monthly sentiment indicators...")
    monthly_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    datasets['monthly_sentiment'] = pd.DataFrame({
        'consumer_confidence': 50 + np.cumsum(np.random.normal(0, 2, len(monthly_dates))),
        'business_sentiment': 50 + np.cumsum(np.random.normal(0, 1.5, len(monthly_dates))),
        'fear_greed_index': 50 + np.random.normal(0, 15, len(monthly_dates))
    }, index=monthly_dates)
    
    # 4. Quarterly earnings data
    print("   💰 Quarterly earnings data...")
    quarterly_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='Q')
    datasets['quarterly_earnings'] = pd.DataFrame({
        'earnings_per_share': 2.0 + np.cumsum(np.random.normal(0.05, 0.2, len(quarterly_dates))),
        'revenue_growth': np.random.normal(0.05, 0.15, len(quarterly_dates)),
        'profit_margin': 0.15 + np.random.normal(0, 0.03, len(quarterly_dates))
    }, index=quarterly_dates)
    
    # 5. Irregular news events
    print("   📰 Irregular news events...")
    np.random.seed(123)
    irregular_dates = pd.to_datetime([
        '2023-03-15', '2023-05-22', '2023-07-10', '2023-09-18', '2023-11-08',
        '2024-01-25', '2024-04-12', '2024-06-30', '2024-08-14', '2024-10-05'
    ])
    datasets['irregular_news'] = pd.DataFrame({
        'news_sentiment_score': np.random.normal(0, 2, len(irregular_dates)),
        'news_count': np.random.poisson(10, len(irregular_dates)),
        'market_impact_score': np.random.uniform(-5, 5, len(irregular_dates))
    }, index=irregular_dates)
    
    # 6. High-frequency intraday data (sample)
    print("   ⚡ Intraday high-frequency data...")
    intraday_start = pd.Timestamp('2024-12-01 09:30:00')
    intraday_end = pd.Timestamp('2024-12-01 16:00:00')
    intraday_dates = pd.date_range(start=intraday_start, end=intraday_end, freq='5min')
    
    base_price = 150
    intraday_returns = np.random.normal(0, 0.001, len(intraday_dates))
    intraday_prices = base_price * np.cumprod(1 + intraday_returns)
    
    datasets['intraday_hf'] = pd.DataFrame({
        'price': intraday_prices,
        'volume': np.random.exponential(10000, len(intraday_dates)),
        'bid_ask_spread': np.random.uniform(0.01, 0.05, len(intraday_dates))
    }, index=intraday_dates)
    
    print(f"✅ Created {len(datasets)} datasets with different frequencies")
    for name, df in datasets.items():
        freq_detected = _detect_simple_frequency(df.index)
        print(f"   • {name}: {len(df)} obs, {freq_detected} frequency")
    
    return datasets


def _detect_simple_frequency(index):
    """Simple frequency detection for display."""
    if len(index) < 2:
        return "unknown"
    
    diff = (index[1] - index[0]).total_seconds()
    
    if diff <= 300:  # 5 minutes
        return "intraday"
    elif diff <= 86400:  # 1 day
        return "daily"
    elif diff <= 7 * 86400:  # 1 week
        return "weekly"
    elif diff <= 31 * 86400:  # 1 month
        return "monthly"
    elif diff <= 95 * 86400:  # ~3 months
        return "quarterly"
    else:
        return "irregular"


def demonstrate_temporal_alignment():
    """Demonstrate temporal alignment capabilities."""
    print("\n🔗 TEMPORAL ALIGNMENT DEMO")
    print("=" * 60)
    
    # Create multi-frequency datasets
    datasets = create_multi_frequency_datasets()
    
    # Initialize temporal aligner
    print("\n⚙️ Initializing temporal aligner for daily alignment...")
    aligner = TemporalAligner(
        target_frequency=FrequencyType.DAILY,
        alignment_method='outer',  # Include all dates
        fill_method='forward_fill',
        business_days_only=True,
        max_fill_periods=5
    )
    
    # Analyze temporal patterns
    print("\n🔍 Analyzing temporal patterns...")
    profiles = aligner.analyze_temporal_patterns(datasets)
    
    for name, profile in profiles.items():
        print(f"   📊 {name}:")
        print(f"      Frequency: {profile.frequency.value}")
        print(f"      Date range: {profile.start_date.date()} to {profile.end_date.date()}")
        print(f"      Observations: {profile.total_observations}")
        print(f"      Coverage: {profile.coverage_ratio():.1%}")
        print(f"      Regularity: {profile.regularity_score:.1%}")
    
    # Perform alignment
    print("\n🎯 Aligning all datasets to daily frequency...")
    aligned_datasets = aligner.align_datasets(datasets)
    
    print(f"\n✅ Alignment complete!")
    print(f"   Original datasets: {len(datasets)}")
    print(f"   Aligned datasets: {len(aligned_datasets)}")
    
    # Show alignment results
    for name, df in aligned_datasets.items():
        if df is not None and not df.empty:
            print(f"   📈 {name}: {len(df)} aligned observations")
            print(f"      Columns: {list(df.columns)}")
            print(f"      Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"      Data coverage: {(~df.isnull()).mean().mean():.1%}")
    
    # Get alignment report
    report = aligner.get_alignment_report()
    print(f"\n📋 ALIGNMENT REPORT")
    print(f"   Target frequency: {report['configuration']['target_frequency']}")
    print(f"   Alignment method: {report['configuration']['alignment_method']}")
    print(f"   Fill method: {report['configuration']['fill_method']}")
    
    return aligned_datasets, report


def demonstrate_shape_normalization(aligned_datasets):
    """Demonstrate data shape normalization."""
    print("\n🔧 DATA SHAPE NORMALIZATION DEMO")
    print("=" * 60)
    
    # Initialize shape normalizer
    print("\n⚙️ Initializing data shape normalizer...")
    normalizer = DataShapeNormalizer(
        standardize_numeric=True,
        encode_categorical=True,
        max_missing_ratio=0.3,
        min_variance_threshold=1e-6
    )
    
    # Fit and transform datasets
    print("\n🎯 Normalizing dataset shapes and scales...")
    normalized_datasets = normalizer.fit_transform(aligned_datasets)
    
    print(f"\n✅ Normalization complete!")
    
    # Show normalization results
    for name, df in normalized_datasets.items():
        if df is not None and not df.empty:
            print(f"   📊 {name}:")
            
            # Check numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                means = df[numeric_cols].mean()
                stds = df[numeric_cols].std()
                print(f"      Numeric features: {len(numeric_cols)}")
                print(f"      Mean range: [{means.min():.3f}, {means.max():.3f}]")
                print(f"      Std range: [{stds.min():.3f}, {stds.max():.3f}]")
            
            # Check categorical columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(categorical_cols) > 0:
                print(f"      Categorical features: {len(categorical_cols)}")
    
    # Get normalization report
    norm_report = normalizer.get_normalization_report()
    print(f"\n📋 NORMALIZATION REPORT")
    print(f"   Datasets processed: {len(norm_report['feature_statistics'])}")
    print(f"   Scalers fitted: {len(norm_report['scalers_fitted'])}")
    print(f"   Encoders fitted: {len(norm_report['encoders_fitted'])}")
    
    return normalized_datasets


def demonstrate_time_bucketing():
    """Demonstrate time bucketing capabilities."""
    print("\n🪣 TIME BUCKETING DEMO")
    print("=" * 60)
    
    # Create sample high-frequency data
    print("\n📊 Creating sample high-frequency data...")
    start_time = pd.Timestamp('2024-12-01 09:30:00')
    end_time = pd.Timestamp('2024-12-01 16:00:00')
    
    # 1-minute data
    minute_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    np.random.seed(42)
    
    base_price = 150
    minute_returns = np.random.normal(0, 0.001, len(minute_index))
    minute_prices = base_price * np.cumprod(1 + minute_returns)
    
    hf_data = pd.DataFrame({
        'price': minute_prices,
        'volume': np.random.exponential(1000, len(minute_index)),
        'bid': minute_prices - np.random.uniform(0.01, 0.05, len(minute_index)),
        'ask': minute_prices + np.random.uniform(0.01, 0.05, len(minute_index))
    }, index=minute_index)
    
    print(f"   ✅ Created {len(hf_data)} minute-level observations")
    
    # Demonstrate different bucketing methods
    bucketing_demos = [
        ("📅 15-minute calendar bars", BucketConfig(
            bucket_type=BucketType.CALENDAR,
            bucket_size='15min',
            aggregation_method='ohlc'
        )),
        ("🏛️ Trading session hourly bars", BucketConfig(
            bucket_type=BucketType.TRADING,
            bucket_size='1H',
            aggregation_method='ohlc',
            session_start=time(9, 30),
            session_end=time(16, 0)
        )),
        ("🔄 50-observation rolling windows", BucketConfig(
            bucket_type=BucketType.ROLLING,
            bucket_size='50',
            aggregation_method='mean',
            overlap_ratio=0.2
        )),
        ("📈 1% event-driven buckets", BucketConfig(
            bucket_type=BucketType.EVENT_DRIVEN,
            bucket_size='1%',
            aggregation_method='last'
        )),
        ("📊 100k volume bars", BucketConfig(
            bucket_type=BucketType.VOLUME,
            bucket_size='100k',
            aggregation_method='ohlc'
        ))
    ]
    
    bucketing_results = {}
    
    for demo_name, config in bucketing_demos:
        print(f"\n{demo_name}")
        
        try:
            bucketer = TimeBucketing(config)
            bucketed_data = bucketer.bucket_data(hf_data)
            
            if not bucketed_data.empty:
                bucketing_results[demo_name] = bucketed_data
                
                # Get statistics
                stats = bucketer.get_bucket_statistics(hf_data, bucketed_data)
                
                print(f"   📊 Original: {stats['original_observations']} observations")
                print(f"   📊 Bucketed: {stats['bucketed_observations']} buckets")
                print(f"   📊 Compression: {stats['compression_ratio']:.1%}")
                print(f"   📊 Coverage: {stats['time_coverage_ratio']:.1%}")
                print(f"   📊 Columns: {list(bucketed_data.columns)}")
            else:
                print("   ❌ No buckets created")
                
        except Exception as e:
            print(f"   ❌ Bucketing failed: {e}")
    
    return bucketing_results


def demonstrate_resampling_utilities():
    """Demonstrate time resampling utilities."""
    print("\n⚙️ TIME RESAMPLING UTILITIES DEMO")
    print("=" * 60)
    
    # Create sample data
    print("\n📊 Creating sample multi-timeframe data...")
    
    # Daily data
    daily_dates = pd.bdate_range(start='2024-01-01', end='2024-12-31')
    np.random.seed(42)
    
    daily_data = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(daily_dates))),
        'volume': np.random.exponential(1000000, len(daily_dates))
    }, index=daily_dates)
    
    # Minute data (sample day)
    minute_start = pd.Timestamp('2024-12-01 09:30:00')
    minute_end = pd.Timestamp('2024-12-01 16:00:00')
    minute_dates = pd.date_range(start=minute_start, end=minute_end, freq='1min')
    
    base_price = daily_data['close'].iloc[-1]
    minute_returns = np.random.normal(0, 0.001, len(minute_dates))
    minute_prices = base_price * np.cumprod(1 + minute_returns)
    
    minute_data = pd.DataFrame({
        'price': minute_prices,
        'volume': np.random.exponential(1000, len(minute_dates))
    }, index=minute_dates)
    
    print(f"   ✅ Daily data: {len(daily_data)} observations")
    print(f"   ✅ Minute data: {len(minute_data)} observations")
    
    # Demonstrate utility functions
    print(f"\n🔧 Demonstrating resampling utilities...")
    
    # 1. Resample daily to weekly
    print(f"   📅 Daily to weekly resampling...")
    weekly_data = TimeResamplingUtility.resample_to_weekly(daily_data, method='last')
    print(f"      {len(daily_data)} daily -> {len(weekly_data)} weekly observations")
    
    # 2. Resample daily to monthly
    print(f"   📅 Daily to monthly resampling...")
    monthly_data = TimeResamplingUtility.resample_to_monthly(daily_data, method='last')
    print(f"      {len(daily_data)} daily -> {len(monthly_data)} monthly observations")
    
    # 3. Create OHLC bars from minute data
    print(f"   📊 Minute to 15-minute OHLC bars...")
    ohlc_15min = TimeResamplingUtility.create_ohlc_bars(minute_data, frequency='15min')
    print(f"      {len(minute_data)} minute -> {len(ohlc_15min)} 15-minute OHLC bars")
    print(f"      OHLC columns: {list(ohlc_15min.columns)}")
    
    # 4. Create trading session buckets
    print(f"   🏛️ Trading session buckets...")
    session_buckets = TimeResamplingUtility.create_trading_session_buckets(minute_data)
    print(f"      {len(minute_data)} minute -> {len(session_buckets)} session buckets")
    
    # 5. Align multiple timeframes
    print(f"   🎯 Aligning multiple timeframes to daily...")
    multi_data = {
        'daily_stocks': daily_data,
        'weekly_econ': weekly_data,
        'monthly_sentiment': monthly_data
    }
    
    aligned_multi = TimeResamplingUtility.align_multiple_timeframes(
        multi_data, target_frequency='1D'
    )
    
    print(f"      Aligned datasets:")
    for name, df in aligned_multi.items():
        print(f"         {name}: {len(df)} observations")
    
    return {
        'weekly': weekly_data,
        'monthly': monthly_data,
        'ohlc_15min': ohlc_15min,
        'session_buckets': session_buckets,
        'aligned_multi': aligned_multi
    }


def demonstrate_combined_pipeline():
    """Demonstrate complete temporal alignment pipeline."""
    print("\n🔄 INTEGRATED TEMPORAL PIPELINE DEMO")
    print("=" * 80)
    
    print("\n1️⃣ Multi-frequency dataset creation and alignment...")
    aligned_datasets, alignment_report = demonstrate_temporal_alignment()
    
    print("\n2️⃣ Data shape normalization...")
    normalized_datasets = demonstrate_shape_normalization(aligned_datasets)
    
    print("\n3️⃣ Advanced time bucketing...")
    bucketing_results = demonstrate_time_bucketing()
    
    print("\n4️⃣ Resampling utilities...")
    resampling_results = demonstrate_resampling_utilities()
    
    # Pipeline summary
    print(f"\n🎯 PIPELINE SUMMARY")
    print(f"=" * 50)
    print(f"✅ Temporal alignment: {len(aligned_datasets)} datasets aligned to daily frequency")
    print(f"✅ Shape normalization: Numeric standardization and categorical encoding applied")
    print(f"✅ Time bucketing: {len(bucketing_results)} bucketing methods demonstrated")
    print(f"✅ Resampling utilities: Multiple frequency conversions demonstrated")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    print(f"=" * 30)
    print(f"🔗 Multi-frequency alignment handles datasets from minute-level to quarterly")
    print(f"📊 Shape normalization ensures consistent scaling across different data sources")
    print(f"🪣 Time bucketing provides flexible aggregation for various analysis needs")
    print(f"⚡ Utility functions simplify common resampling operations")
    print(f"🎯 Integrated pipeline handles complex temporal data challenges seamlessly")
    
    return {
        'aligned_datasets': aligned_datasets,
        'normalized_datasets': normalized_datasets,
        'bucketing_results': bucketing_results,
        'resampling_results': resampling_results,
        'alignment_report': alignment_report
    }


def main():
    """Run comprehensive temporal alignment demonstration."""
    print("⏰ RTRADEZ ADVANCED TEMPORAL ALIGNMENT DEMO")
    print("=" * 80)
    print("Demonstrating handling of different sizes, shapes, and temporal patterns...")
    print("=" * 80)
    
    try:
        # Run integrated pipeline demonstration
        results = demonstrate_combined_pipeline()
        
        print(f"\n✨ DEMO COMPLETE")
        print(f"=" * 80)
        print(f"🎯 Successfully demonstrated:")
        print(f"   • Multi-frequency temporal alignment (tick to quarterly)")
        print(f"   • Different dataset shapes and missing pattern handling")
        print(f"   • Normalization and standardization strategies")
        print(f"   • Advanced time bucketing (calendar, trading, rolling, event, volume)")
        print(f"   • Time resampling utilities for common operations")
        print(f"   • Integrated pipeline for complex temporal data challenges")
        print(f"\n🚀 RTradez temporal alignment handles any dataset combination!")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()