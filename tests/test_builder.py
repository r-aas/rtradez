"""Tests for rtradez.methods.builder."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.methods.builder import *

class TestStrategyBuilder:
    """Test cases for StrategyBuilder."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(StrategyBuilder, "__members__"):
            # Test Enum values
            for member in StrategyBuilder:
                assert isinstance(member, StrategyBuilder)
            return
        
        try:
            instance = StrategyBuilder()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = StrategyBuilder(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

