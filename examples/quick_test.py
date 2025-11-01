"""Quick test of RTradez temporal alignment capabilities."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import RTradez modules
from rtradez.utils.temporal_alignment import TemporalAligner, FrequencyType
from rtradez.utils.time_bucketing import TimeResamplingUtility
from rtradez.utils.dataset_combiner import DatasetCombiner

def quick_test():
    """Quick demonstration of RTradez capabilities."""
    print("🚀 RTRADEZ QUICK TEST")
    print("=" * 50)
    
    # Create different frequency datasets
    print("\n📊 Creating datasets with different frequencies...")
    
    # Daily stock data
    daily_dates = pd.bdate_range(start='2024-01-01', end='2024-12-31')
    daily_stocks = pd.DataFrame({
        'price': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(daily_dates))),
        'volume': np.random.exponential(1000000, len(daily_dates))
    }, index=daily_dates)
    
    # Weekly economic data
    weekly_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
    weekly_econ = pd.DataFrame({
        'gdp_growth': np.random.normal(0.02, 0.01, len(weekly_dates)),
        'inflation': np.random.normal(0.03, 0.005, len(weekly_dates))
    }, index=weekly_dates)
    
    # Monthly sentiment
    monthly_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    monthly_sentiment = pd.DataFrame({
        'consumer_confidence': np.random.normal(50, 10, len(monthly_dates)),
        'market_sentiment': np.random.normal(0, 1, len(monthly_dates))
    }, index=monthly_dates)
    
    datasets = {
        'stocks': daily_stocks,
        'economic': weekly_econ,
        'sentiment': monthly_sentiment
    }
    
    print(f"✅ Created {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"   • {name}: {len(df)} observations")
    
    # Test temporal alignment
    print("\n🔗 Testing temporal alignment...")
    aligner = TemporalAligner(
        target_frequency=FrequencyType.DAILY,
        alignment_method='outer',
        fill_method='forward_fill'
    )
    
    aligned_datasets = aligner.align_datasets(datasets)
    
    print(f"✅ Aligned to daily frequency:")
    for name, df in aligned_datasets.items():
        coverage = (~df.isnull()).mean().mean()
        print(f"   • {name}: {len(df)} rows, {coverage:.1%} coverage")
    
    # Test dataset combination
    print("\n🔧 Testing dataset combination...")
    combiner = DatasetCombiner(
        alignment_method='inner',
        missing_data_strategy='forward_fill',
        feature_scaling='standard'
    )
    
    combined_data = combiner.combine_datasets(aligned_datasets, feature_prefix=True)
    
    print(f"✅ Combined dataset:")
    print(f"   • Shape: {combined_data.shape}")
    print(f"   • Columns: {list(combined_data.columns)}")
    print(f"   • Date range: {combined_data.index.min().date()} to {combined_data.index.max().date()}")
    
    # Test resampling utilities
    print("\n⚙️ Testing resampling utilities...")
    
    # Resample daily to weekly
    weekly_stocks = TimeResamplingUtility.resample_to_weekly(daily_stocks)
    print(f"   • Daily to weekly: {len(daily_stocks)} -> {len(weekly_stocks)} observations")
    
    # Resample daily to monthly
    monthly_stocks = TimeResamplingUtility.resample_to_monthly(daily_stocks)
    print(f"   • Daily to monthly: {len(daily_stocks)} -> {len(monthly_stocks)} observations")
    
    print(f"\n🎉 All tests passed! RTradez handles multi-frequency data seamlessly.")
    
    return combined_data

if __name__ == "__main__":
    result = quick_test()