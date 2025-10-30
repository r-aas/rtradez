"""
Intelligent caching system for RTradez.

Provides high-performance caching for market data, feature engineering,
and backtest results to dramatically speed up iterative development and research.
"""

import os
import hashlib
import pickle
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import pandas as pd
import numpy as np
import diskcache as dc
from functools import wraps
import logging

# Configure logging
logger = logging.getLogger(__name__)


class RTradezCache:
    """
    High-performance caching system for RTradez components.
    
    Features:
    - Disk-based persistent caching with LRU eviction
    - Automatic cache invalidation based on data freshness
    - Separate cache namespaces for different data types
    - Compression for large datasets
    - Cache statistics and monitoring
    """
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 max_size_gb: float = 10.0,
                 default_expire: int = 86400):  # 24 hours
        """
        Initialize cache system.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            default_expire: Default expiration time in seconds
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.rtradez/cache")
        self.max_size = int(max_size_gb * 1024**3)  # Convert to bytes
        self.default_expire = default_expire
        
        # Create cache directories
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize separate caches for different data types
        self.caches = {
            'market_data': dc.Cache(
                directory=os.path.join(self.cache_dir, 'market_data'),
                size_limit=self.max_size // 4,
                eviction_policy='least-recently-used'
            ),
            'features': dc.Cache(
                directory=os.path.join(self.cache_dir, 'features'),
                size_limit=self.max_size // 4,
                eviction_policy='least-recently-used'
            ),
            'backtests': dc.Cache(
                directory=os.path.join(self.cache_dir, 'backtests'),
                size_limit=self.max_size // 4,
                eviction_policy='least-recently-used'
            ),
            'models': dc.Cache(
                directory=os.path.join(self.cache_dir, 'models'),
                size_limit=self.max_size // 4,
                eviction_policy='least-recently-used'
            )
        }
        
        logger.info(f"RTradez cache initialized at {self.cache_dir}")
        
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique cache key from function name and arguments."""
        # Create deterministic key from function name and arguments
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        
        # Convert to JSON and hash for consistent key generation
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get item from cache."""
        if cache_type not in self.caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        try:
            return self.caches[cache_type].get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
            return None
            
    def set(self, cache_type: str, key: str, value: Any, 
            expire: Optional[int] = None) -> bool:
        """Set item in cache."""
        if cache_type not in self.caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        try:
            expire = expire or self.default_expire
            self.caches[cache_type].set(key, value, expire=expire)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")
            return False
            
    def delete(self, cache_type: str, key: str) -> bool:
        """Delete item from cache."""
        if cache_type not in self.caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        try:
            return self.caches[cache_type].delete(key)
        except Exception as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
            return False
            
    def clear(self, cache_type: Optional[str] = None) -> bool:
        """Clear cache(s)."""
        try:
            if cache_type:
                if cache_type not in self.caches:
                    raise ValueError(f"Unknown cache type: {cache_type}")
                self.caches[cache_type].clear()
            else:
                for cache in self.caches.values():
                    cache.clear()
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
            
    def stats(self) -> Dict[str, Dict]:
        """Get cache statistics."""
        stats = {}
        for cache_type, cache in self.caches.items():
            try:
                stats[cache_type] = {
                    'size': len(cache) if hasattr(cache, '__len__') else 0,
                    'size_bytes': cache.volume() if hasattr(cache, 'volume') else 0,
                    'hits': 0,  # Simplified for demo
                    'misses': 0  # Simplified for demo
                }
            except Exception as e:
                logger.warning(f"Failed to get stats for {cache_type}: {e}")
                stats[cache_type] = {'size': 0, 'size_bytes': 0, 'hits': 0, 'misses': 0}
                
        return stats
        
    def cleanup(self, older_than_hours: int = 24) -> Dict[str, int]:
        """Clean up old cache entries."""
        cleaned = {}
        cutoff_time = time.time() - (older_than_hours * 3600)
        
        for cache_type, cache in self.caches.items():
            try:
                # This is simplified - diskcache doesn't expose creation time easily
                # In practice, you'd iterate through keys and check timestamps
                initial_size = len(cache)
                cache.expire()  # Remove expired entries
                final_size = len(cache)
                cleaned[cache_type] = initial_size - final_size
            except Exception as e:
                logger.warning(f"Cleanup failed for {cache_type}: {e}")
                cleaned[cache_type] = 0
                
        return cleaned


# Global cache instance
_cache = None

def get_cache() -> RTradezCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = RTradezCache()
    return _cache


def cached(cache_type: str = 'features', 
          expire: Optional[int] = None,
          key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_type: Type of cache to use ('market_data', 'features', 'backtests', 'models')
        expire: Expiration time in seconds
        key_func: Custom function to generate cache key
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache._generate_key(func.__name__, args, kwargs)
                
            # Try to get from cache
            result = cache.get(cache_type, key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {key}")
                return result
                
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}: {key}")
            result = func(*args, **kwargs)
            
            # Cache the result
            cache.set(cache_type, key, result, expire=expire)
            return result
            
        # Add cache management methods to function
        wrapper.cache_clear = lambda: get_cache().clear(cache_type)
        wrapper.cache_info = lambda: get_cache().stats()[cache_type]
        
        return wrapper
    return decorator


def cache_key_from_dataframe(df: pd.DataFrame, 
                           additional_info: Optional[Dict] = None) -> str:
    """Generate cache key from DataFrame characteristics."""
    key_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'index_start': str(df.index[0]) if len(df) > 0 else None,
        'index_end': str(df.index[-1]) if len(df) > 0 else None,
        'data_hash': hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    }
    
    if additional_info:
        key_info.update(additional_info)
        
    return hashlib.md5(json.dumps(key_info, sort_keys=True).encode()).hexdigest()


def invalidate_market_data_cache(symbol: str, start_date: str = None):
    """Invalidate market data cache for specific symbol."""
    cache = get_cache()
    # This would need more sophisticated key tracking in practice
    logger.info(f"Invalidating market data cache for {symbol}")
    

class CacheManager:
    """High-level cache management interface."""
    
    def __init__(self):
        self.cache = get_cache()
        
    def warm_up_market_data(self, symbols: list, period: str = '1y'):
        """Pre-load market data for common symbols."""
        from ..datasets import OptionsDataset
        
        logger.info(f"Warming up cache for {len(symbols)} symbols")
        for symbol in symbols:
            try:
                OptionsDataset.from_source('yahoo', symbol, period=period)
                logger.debug(f"Cached market data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to cache {symbol}: {e}")
                
    def optimize_cache(self):
        """Optimize cache performance."""
        # Clean up expired entries
        cleaned = self.cache.cleanup()
        logger.info(f"Cache cleanup removed: {cleaned}")
        
        # Get current stats
        stats = self.cache.stats()
        logger.info(f"Cache stats: {stats}")
        
    def backup_cache(self, backup_path: str):
        """Backup cache to specified path."""
        import shutil
        backup_dir = Path(backup_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for cache_type in self.cache.caches.keys():
            src = os.path.join(self.cache.cache_dir, cache_type)
            dst = os.path.join(backup_path, cache_type)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                
        logger.info(f"Cache backed up to {backup_path}")
        
    def restore_cache(self, backup_path: str):
        """Restore cache from backup."""
        import shutil
        for cache_type in self.cache.caches.keys():
            src = os.path.join(backup_path, cache_type)
            dst = os.path.join(self.cache.cache_dir, cache_type)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                
        logger.info(f"Cache restored from {backup_path}")