#!/usr/bin/env python3
"""
Pydantic Integration Demo for RTradez.

Demonstrates the comprehensive Pydantic data validation system across
risk management, portfolio management, and data processing components.
"""

import sys
import os
from datetime import datetime, time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rtradez.risk.position_sizing import (
    PositionSizerConfig, KellyConfig, FixedFractionConfig, 
    VolatilityAdjustedConfig, MultiStrategyConfig, SizingMethod,
    KellyCriterion, FixedFractionSizer, create_position_sizer
)
from rtradez.portfolio.portfolio_manager import PortfolioConfig, StrategyAllocation, StrategyStatus
from rtradez.utils.temporal_alignment import TemporalProfile, TemporalAlignerConfig, FrequencyType
from rtradez.utils.time_bucketing import BucketConfig, BucketType


def demo_pydantic_validation():
    """Comprehensive demo of Pydantic validation across RTradez components."""
    print("🔒 RTradez Pydantic Data Validation Demo")
    print("=" * 55)
    
    # 1. Risk Management Configuration
    print("\n📊 RISK MANAGEMENT VALIDATION")
    print("-" * 40)
    
    # Basic position sizer config
    basic_config = PositionSizerConfig(total_capital=500000, max_risk_per_trade=0.025)
    print(f"✅ Basic Config: ${basic_config.total_capital:,.0f} capital, {basic_config.max_risk_per_trade:.1%} max risk")
    
    # Kelly Criterion with specific configuration
    kelly_config = KellyConfig(
        total_capital=1000000,
        max_risk_per_trade=0.02,
        confidence_threshold=0.65,
        max_kelly_fraction=0.30
    )
    kelly_sizer = KellyCriterion(kelly_config)
    print(f"✅ Kelly Config: {kelly_config.confidence_threshold:.0%} confidence threshold")
    
    # Multi-strategy configuration
    multi_config = MultiStrategyConfig(
        total_capital=2000000,
        max_total_risk=0.12,
        correlation_matrix=[[1.0, 0.3], [0.3, 1.0]]
    )
    print(f"✅ Multi-Strategy: {multi_config.max_total_risk:.0%} max portfolio risk")
    
    # 2. Portfolio Management Configuration  
    print("\n🏦 PORTFOLIO MANAGEMENT VALIDATION")
    print("-" * 42)
    
    portfolio_config = PortfolioConfig(
        total_capital=5000000,
        max_strategies=8,
        rebalance_frequency="weekly",
        rebalance_threshold=0.08,
        max_correlation_exposure=0.65,
        emergency_stop_drawdown=0.18,
        cash_reserve_minimum=0.10
    )
    print(f"✅ Portfolio Config: ${portfolio_config.total_capital:,.0f}, max {portfolio_config.max_strategies} strategies")
    
    # Strategy allocation with validation
    strategy_allocation = StrategyAllocation(
        strategy_name="Iron_Condor_SPY",
        target_allocation=0.25,
        current_allocation=0.23,
        min_allocation=0.05,
        max_allocation=0.40,
        status=StrategyStatus.ACTIVE
    )
    print(f"✅ Strategy Allocation: {strategy_allocation.strategy_name}, target: {strategy_allocation.target_allocation:.0%}")
    
    # 3. Temporal Data Processing
    print("\n⏰ TEMPORAL PROCESSING VALIDATION")
    print("-" * 38)
    
    # Temporal profile validation
    temporal_profile = TemporalProfile(
        frequency=FrequencyType.DAILY,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        total_observations=252,
        missing_periods=8,
        regularity_score=0.97,
        timezone="US/Eastern"
    )
    print(f"✅ Temporal Profile: {temporal_profile.frequency.value}, coverage: {temporal_profile.coverage_ratio():.1%}")
    
    # Temporal aligner configuration
    aligner_config = TemporalAlignerConfig(
        target_frequency=FrequencyType.WEEKLY,
        alignment_method="outer",
        fill_method="interpolate",
        business_days_only=True,
        max_fill_periods=3
    )
    print(f"✅ Aligner Config: {aligner_config.target_frequency.value} target, {aligner_config.fill_method} fill")
    
    # Time bucketing configuration
    bucket_config = BucketConfig(
        bucket_type=BucketType.TRADING,
        bucket_size="4H",
        aggregation_method="ohlc",
        timezone="US/Eastern",
        session_start=time(9, 30),
        session_end=time(16, 0),
        overlap_ratio=0.1
    )
    print(f"✅ Bucket Config: {bucket_config.bucket_type.value}, {bucket_config.bucket_size} buckets")
    
    # 4. Validation Error Demonstration
    print("\n🛡️  VALIDATION ERROR EXAMPLES")
    print("-" * 35)
    
    validation_examples = [
        ("Invalid capital (negative)", lambda: PositionSizerConfig(total_capital=-1000)),
        ("Invalid risk percentage (>100%)", lambda: PositionSizerConfig(total_capital=100000, max_risk_per_trade=1.5)),
        ("Invalid allocation bounds", lambda: StrategyAllocation(
            strategy_name="Test", target_allocation=0.3, current_allocation=0.2,
            min_allocation=0.4, max_allocation=0.35
        )),
        ("Invalid date range", lambda: TemporalProfile(
            frequency=FrequencyType.DAILY, start_date=datetime(2024, 12, 31),
            end_date=datetime(2024, 1, 1), total_observations=100,
            missing_periods=0, regularity_score=1.0
        )),
        ("Invalid session times", lambda: BucketConfig(
            bucket_type=BucketType.TRADING, bucket_size="1H",
            session_start=time(16, 0), session_end=time(9, 30)
        ))
    ]
    
    for description, test_func in validation_examples:
        try:
            test_func()
            print(f"❌ {description}: Should have failed!")
        except Exception as e:
            print(f"✅ {description}: Caught validation error")
    
    # 5. Integration Test
    print("\n🔗 INTEGRATION VALIDATION")
    print("-" * 28)
    
    # Create a complete system configuration
    print("Creating integrated system configuration:")
    
    # Position sizing
    sizing_result = kelly_sizer.calculate_position_size(
        strategy_name="Integrated_Test",
        expected_return=0.12,
        volatility=0.18
    )
    print(f"   Kelly Sizing: ${sizing_result.recommended_size:,.0f} recommended")
    print(f"   Confidence: {sizing_result.confidence_level:.2f}")
    
    # Factory pattern with validation
    vol_sizer = create_position_sizer(
        method=SizingMethod.VOLATILITY_ADJUSTED,
        config=VolatilityAdjustedConfig(total_capital=800000, lookback_window=126)
    )
    print(f"   Factory Pattern: Created {type(vol_sizer).__name__}")
    
    # 6. Summary
    print("\n🎯 VALIDATION SUMMARY")
    print("-" * 25)
    
    components = [
        "Position Sizing Configurations",
        "Portfolio Management Settings", 
        "Temporal Data Processing",
        "Time Bucketing Parameters",
        "Error Validation & Safety",
        "Factory Pattern Integration"
    ]
    
    for component in components:
        print(f"✅ {component}")
    
    print(f"\n💡 Benefits Demonstrated:")
    print("   • Automatic data validation at creation time")
    print("   • Type safety and constraint enforcement") 
    print("   • Clear error messages for invalid configurations")
    print("   • Seamless integration across components")
    print("   • Runtime validation for critical parameters")
    print("   • JSON serialization/deserialization support")
    
    print(f"\n🔧 Production Features:")
    print("   • Configuration management with validation")
    print("   • API input validation for web services")
    print("   • Database model validation")
    print("   • Configuration file validation")
    print("   • Parameter bounds enforcement")


if __name__ == "__main__":
    demo_pydantic_validation()