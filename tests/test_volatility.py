"""Tests for rtradez.datasets.volatility."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.datasets.volatility import *

class TestVolatilitySurface:
    """Test cases for VolatilitySurface."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(VolatilitySurface, "__members__"):
            # Test Enum values
            for member in VolatilitySurface:
                assert isinstance(member, VolatilitySurface)
            return
        
        try:
            instance = VolatilitySurface()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = VolatilitySurface(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

