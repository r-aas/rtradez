"""
RTradez Datasets Module

Data ingestion, storage, and management for options trading datasets.

Key Components:
- OptionsDataset: Main dataset class for options data
- DataValidator: Data quality validation
- DataStandardizer: Market data standardization
- VolatilitySurface: Implied volatility surface management
- Sklearn-like transformers: TechnicalIndicatorTransformer, ReturnsTransformer, etc.
"""

from .core import OptionsDataset
from .validators import DataValidator
from .standardizers import DataStandardizer
from .volatility import VolatilitySurface

# Import sklearn-like transformers
try:
    from .transformers import (
        TechnicalIndicatorTransformer,
        ReturnsTransformer, 
        OptionsChainStandardizer,
        VolatilitySurfaceTransformer
    )
    
    __all__ = [
        "OptionsDataset",
        "DataValidator", 
        "DataStandardizer",
        "VolatilitySurface",
        "TechnicalIndicatorTransformer",
        "ReturnsTransformer",
        "OptionsChainStandardizer", 
        "VolatilitySurfaceTransformer",
    ]
except ImportError:
    # Fallback if transformers not available
    __all__ = [
        "OptionsDataset",
        "DataValidator", 
        "DataStandardizer",
        "VolatilitySurface",
    ]