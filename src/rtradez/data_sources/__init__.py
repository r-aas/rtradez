"""Comprehensive data integration framework for RTradez research."""

from .base_provider import BaseDataProvider, DataSourceConfig
from .economic.fred import FREDProvider
from .alternative.sentiment import SentimentProvider
from .crypto.spot_prices import CryptoProvider
from .data_manager import RTradezDataManager

__all__ = [
    'BaseDataProvider', 'DataSourceConfig', 'RTradezDataManager',
    'FREDProvider', 'SentimentProvider', 'CryptoProvider'
]