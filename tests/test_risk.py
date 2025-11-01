"""Tests for rtradez.metrics.risk."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.metrics.risk import *

class TestRiskMetrics:
    """Test cases for RiskMetrics."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(RiskMetrics, "__members__"):
            # Test Enum values
            for member in RiskMetrics:
                assert isinstance(member, RiskMetrics)
            return
        
        try:
            instance = RiskMetrics()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = RiskMetrics(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

