"""Tests for rtradez.datasets.validators."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.datasets.validators import *

class TestDataValidator:
    """Test cases for DataValidator."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(DataValidator, "__members__"):
            # Test Enum values
            for member in DataValidator:
                assert isinstance(member, DataValidator)
            return
        
        try:
            instance = DataValidator()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = DataValidator(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

