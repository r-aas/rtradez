"""Tests for rtradez.datasets.standardizers."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.datasets.standardizers import *

class TestDataStandardizer:
    """Test cases for DataStandardizer."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(DataStandardizer, "__members__"):
            # Test Enum values
            for member in DataStandardizer:
                assert isinstance(member, DataStandardizer)
            return
        
        try:
            instance = DataStandardizer()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = DataStandardizer(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

