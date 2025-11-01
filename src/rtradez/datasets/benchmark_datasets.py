"""Benchmark datasets for comprehensive options strategy validation."""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class AssetClass(Enum):
    """Asset class categorization for datasets."""
    EQUITY_INDEX = "equity_index"
    SECTOR_ETF = "sector_etf" 
    INTERNATIONAL = "international"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    REAL_ESTATE = "real_estate"
    VOLATILITY = "volatility"
    INDIVIDUAL_STOCK = "individual_stock"


class MarketCap(Enum):
    """Market capitalization categories."""
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    MIXED = "mixed"


@dataclass
class BenchmarkDatasetInfo:
    """Information about a benchmark dataset."""
    symbol: str
    name: str
    asset_class: AssetClass
    market_cap: Optional[MarketCap] = None
    sector: Optional[str] = None
    region: Optional[str] = None
    inception_date: Optional[str] = None
    typical_iv: Optional[float] = None  # Typical implied volatility
    liquidity_score: int = 5  # 1-10 scale for options liquidity
    notes: Optional[str] = None


class BenchmarkDatasets:
    """Comprehensive benchmark datasets for options strategy validation."""
    
    # Core indices (already implemented)
    CORE_INDICES = [
        BenchmarkDatasetInfo('SPY', 'S&P 500 ETF', AssetClass.EQUITY_INDEX, 
                           MarketCap.LARGE_CAP, region='US', liquidity_score=10),
        BenchmarkDatasetInfo('QQQ', 'NASDAQ 100 ETF', AssetClass.EQUITY_INDEX,
                           MarketCap.LARGE_CAP, sector='Technology', region='US', liquidity_score=10),
        BenchmarkDatasetInfo('IWM', 'Russell 2000 ETF', AssetClass.EQUITY_INDEX,
                           MarketCap.SMALL_CAP, region='US', liquidity_score=9),
    ]
    
    # Additional major indices
    ADDITIONAL_INDICES = [
        BenchmarkDatasetInfo('DIA', 'Dow Jones Industrial Average ETF', AssetClass.EQUITY_INDEX,
                           MarketCap.LARGE_CAP, region='US', liquidity_score=8),
        BenchmarkDatasetInfo('MDY', 'S&P Mid-Cap 400 ETF', AssetClass.EQUITY_INDEX,
                           MarketCap.MID_CAP, region='US', liquidity_score=7),
        BenchmarkDatasetInfo('VTI', 'Total Stock Market ETF', AssetClass.EQUITY_INDEX,
                           MarketCap.MIXED, region='US', liquidity_score=6),
    ]
    
    # Sector ETFs for diversification
    SECTOR_ETFS = [
        BenchmarkDatasetInfo('XLK', 'Technology Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Technology', region='US', liquidity_score=9),
        BenchmarkDatasetInfo('XLF', 'Financial Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Financial', region='US', liquidity_score=9),
        BenchmarkDatasetInfo('XLE', 'Energy Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Energy', region='US', liquidity_score=8),
        BenchmarkDatasetInfo('XLV', 'Healthcare Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Healthcare', region='US', liquidity_score=8),
        BenchmarkDatasetInfo('XLI', 'Industrial Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Industrial', region='US', liquidity_score=7),
        BenchmarkDatasetInfo('XLP', 'Consumer Staples Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Consumer Staples', region='US', liquidity_score=7),
        BenchmarkDatasetInfo('XLY', 'Consumer Discretionary Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Consumer Discretionary', region='US', liquidity_score=7),
        BenchmarkDatasetInfo('XLU', 'Utilities Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Utilities', region='US', liquidity_score=6),
        BenchmarkDatasetInfo('XLB', 'Materials Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Materials', region='US', liquidity_score=6),
        BenchmarkDatasetInfo('XLRE', 'Real Estate Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Real Estate', region='US', liquidity_score=6),
        BenchmarkDatasetInfo('XLC', 'Communication Services Select Sector SPDR', AssetClass.SECTOR_ETF,
                           sector='Communication Services', region='US', liquidity_score=6),
    ]
    
    # International exposure
    INTERNATIONAL_ETFS = [
        BenchmarkDatasetInfo('EFA', 'EAFE ETF', AssetClass.INTERNATIONAL,
                           region='Developed Ex-US', liquidity_score=8),
        BenchmarkDatasetInfo('EEM', 'Emerging Markets ETF', AssetClass.INTERNATIONAL,
                           region='Emerging Markets', liquidity_score=8),
        BenchmarkDatasetInfo('FEZ', 'Eurozone ETF', AssetClass.INTERNATIONAL,
                           region='Eurozone', liquidity_score=6),
        BenchmarkDatasetInfo('EWJ', 'Japan ETF', AssetClass.INTERNATIONAL,
                           region='Japan', liquidity_score=6),
        BenchmarkDatasetInfo('MCHI', 'China ETF', AssetClass.INTERNATIONAL,
                           region='China', liquidity_score=5),
    ]
    
    # Alternative asset classes
    ALTERNATIVE_ASSETS = [
        # Fixed Income
        BenchmarkDatasetInfo('TLT', '20+ Year Treasury Bond ETF', AssetClass.FIXED_INCOME,
                           region='US', liquidity_score=9, typical_iv=0.15),
        BenchmarkDatasetInfo('IEF', '7-10 Year Treasury ETF', AssetClass.FIXED_INCOME,
                           region='US', liquidity_score=7),
        BenchmarkDatasetInfo('LQD', 'Investment Grade Corporate Bond ETF', AssetClass.FIXED_INCOME,
                           region='US', liquidity_score=6),
        BenchmarkDatasetInfo('HYG', 'High Yield Corporate Bond ETF', AssetClass.FIXED_INCOME,
                           region='US', liquidity_score=7),
        
        # Commodities
        BenchmarkDatasetInfo('GLD', 'Gold ETF', AssetClass.COMMODITY,
                           liquidity_score=9, typical_iv=0.20),
        BenchmarkDatasetInfo('SLV', 'Silver ETF', AssetClass.COMMODITY,
                           liquidity_score=7, typical_iv=0.35),
        BenchmarkDatasetInfo('USO', 'Oil ETF', AssetClass.COMMODITY,
                           liquidity_score=6, typical_iv=0.45),
        BenchmarkDatasetInfo('UNG', 'Natural Gas ETF', AssetClass.COMMODITY,
                           liquidity_score=5, typical_iv=0.60),
        
        # REITs
        BenchmarkDatasetInfo('VNQ', 'Real Estate ETF', AssetClass.REAL_ESTATE,
                           region='US', liquidity_score=7),
        BenchmarkDatasetInfo('VNQI', 'International Real Estate ETF', AssetClass.REAL_ESTATE,
                           region='International', liquidity_score=5),
    ]
    
    # Volatility products
    VOLATILITY_PRODUCTS = [
        BenchmarkDatasetInfo('VXX', 'VIX Short-Term Futures ETN', AssetClass.VOLATILITY,
                           liquidity_score=8, typical_iv=0.80,
                           notes='High volatility, mean-reverting'),
        BenchmarkDatasetInfo('UVXY', '1.5x VIX Short-Term Futures ETF', AssetClass.VOLATILITY,
                           liquidity_score=7, typical_iv=1.20),
        BenchmarkDatasetInfo('SVXY', 'Short VIX Short-Term Futures ETF', AssetClass.VOLATILITY,
                           liquidity_score=6, typical_iv=1.00),
    ]
    
    # High-volume individual stocks
    INDIVIDUAL_STOCKS = [
        # Mega-cap technology
        BenchmarkDatasetInfo('AAPL', 'Apple Inc.', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Technology', liquidity_score=10),
        BenchmarkDatasetInfo('MSFT', 'Microsoft Corporation', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Technology', liquidity_score=9),
        BenchmarkDatasetInfo('GOOGL', 'Alphabet Inc.', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Technology', liquidity_score=9),
        BenchmarkDatasetInfo('AMZN', 'Amazon.com Inc.', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Consumer Discretionary', liquidity_score=9),
        BenchmarkDatasetInfo('TSLA', 'Tesla Inc.', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Consumer Discretionary', liquidity_score=9),
        
        # Financial
        BenchmarkDatasetInfo('JPM', 'JPMorgan Chase & Co.', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Financial', liquidity_score=8),
        BenchmarkDatasetInfo('BAC', 'Bank of America Corporation', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Financial', liquidity_score=7),
        
        # Other sectors
        BenchmarkDatasetInfo('JNJ', 'Johnson & Johnson', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Healthcare', liquidity_score=7),
        BenchmarkDatasetInfo('XOM', 'Exxon Mobil Corporation', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Energy', liquidity_score=6),
        BenchmarkDatasetInfo('WMT', 'Walmart Inc.', AssetClass.INDIVIDUAL_STOCK,
                           MarketCap.LARGE_CAP, sector='Consumer Staples', liquidity_score=6),
    ]
    
    @classmethod
    def get_all_datasets(cls) -> List[BenchmarkDatasetInfo]:
        """Get all available benchmark datasets."""
        return (cls.CORE_INDICES + cls.ADDITIONAL_INDICES + cls.SECTOR_ETFS + 
                cls.INTERNATIONAL_ETFS + cls.ALTERNATIVE_ASSETS + 
                cls.VOLATILITY_PRODUCTS + cls.INDIVIDUAL_STOCKS)
    
    @classmethod
    def get_by_asset_class(cls, asset_class: AssetClass) -> List[BenchmarkDatasetInfo]:
        """Get datasets filtered by asset class."""
        return [d for d in cls.get_all_datasets() if d.asset_class == asset_class]
    
    @classmethod
    def get_by_liquidity(cls, min_score: int = 7) -> List[BenchmarkDatasetInfo]:
        """Get datasets filtered by minimum liquidity score."""
        return [d for d in cls.get_all_datasets() if d.liquidity_score >= min_score]
    
    @classmethod
    def get_by_sector(cls, sector: str) -> List[BenchmarkDatasetInfo]:
        """Get datasets filtered by sector."""
        return [d for d in cls.get_all_datasets() if d.sector == sector]
    
    @classmethod
    def get_symbols(cls) -> List[str]:
        """Get list of all available symbols."""
        return [d.symbol for d in cls.get_all_datasets()]
    
    @classmethod
    def get_high_priority_symbols(cls) -> List[str]:
        """Get high-priority symbols for initial implementation."""
        high_priority = (cls.CORE_INDICES + cls.ADDITIONAL_INDICES + 
                        cls.SECTOR_ETFS[:6] + cls.INTERNATIONAL_ETFS[:3])
        return [d.symbol for d in high_priority]


# Predefined benchmark suites for different testing scenarios
BENCHMARK_SUITES = {
    'core': ['SPY', 'QQQ', 'IWM'],
    'extended_indices': ['SPY', 'QQQ', 'IWM', 'DIA', 'MDY', 'VTI'],
    'sector_rotation': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU'],
    'international': ['SPY', 'EFA', 'EEM', 'FEZ', 'EWJ'],
    'alternative_assets': ['SPY', 'TLT', 'GLD', 'VNQ', 'USO'],
    'volatility_products': ['SPY', 'VXX', 'UVXY', 'SVXY'],
    'mega_cap_stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    'comprehensive': BenchmarkDatasets.get_high_priority_symbols(),
}