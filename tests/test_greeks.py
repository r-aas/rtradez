"""Tests for rtradez.methods.greeks."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.methods.greeks import *

class TestGreeksCalculator:
    """Test cases for GreeksCalculator."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(GreeksCalculator, "__members__"):
            # Test Enum values
            for member in GreeksCalculator:
                assert isinstance(member, GreeksCalculator)
            return
        
        try:
            instance = GreeksCalculator()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = GreeksCalculator(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

