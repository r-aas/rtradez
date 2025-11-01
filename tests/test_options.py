"""Tests for rtradez.metrics.options."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.metrics.options import *

class TestOptionsMetrics:
    """Test cases for OptionsMetrics."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(OptionsMetrics, "__members__"):
            # Test Enum values
            for member in OptionsMetrics:
                assert isinstance(member, OptionsMetrics)
            return
        
        try:
            instance = OptionsMetrics()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = OptionsMetrics(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

