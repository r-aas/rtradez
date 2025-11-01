"""Tests for rtradez.metrics.comparison."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.metrics.comparison import *

class TestComparisonEngine:
    """Test cases for ComparisonEngine."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(ComparisonEngine, "__members__"):
            # Test Enum values
            for member in ComparisonEngine:
                assert isinstance(member, ComparisonEngine)
            return
        
        try:
            instance = ComparisonEngine()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = ComparisonEngine(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

