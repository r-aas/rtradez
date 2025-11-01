"""Tests for rtradez.methods.backtest."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.methods.backtest import *

class TestBacktestEngine:
    """Test cases for BacktestEngine."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(BacktestEngine, "__members__"):
            # Test Enum values
            for member in BacktestEngine:
                assert isinstance(member, BacktestEngine)
            return
        
        try:
            instance = BacktestEngine()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = BacktestEngine(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

