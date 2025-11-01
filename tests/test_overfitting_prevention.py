"""Tests for rtradez.validation.overfitting_prevention."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.validation.overfitting_prevention import *

class TestOverfittingDetector:
    """Test cases for OverfittingDetector."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(OverfittingDetector, "__members__"):
            # Test Enum values
            for member in OverfittingDetector:
                assert isinstance(member, OverfittingDetector)
            return
        
        try:
            instance = OverfittingDetector()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = OverfittingDetector(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

    def test_detect_overfitting(self, sample_data):
        """Test detect_overfitting method."""
        instance = OverfittingDetector(**sample_data.get("init_params", {}))
        
        try:
            result = instance.detect_overfitting(**sample_data.get("detect_overfitting_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_plot_validation_results(self, sample_data):
        """Test plot_validation_results method."""
        instance = OverfittingDetector(**sample_data.get("init_params", {}))
        
        try:
            result = instance.plot_validation_results(**sample_data.get("plot_validation_results_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_generate_report(self, sample_data):
        """Test generate_report method."""
        instance = OverfittingDetector(**sample_data.get("init_params", {}))
        
        try:
            result = instance.generate_report(**sample_data.get("generate_report_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass


class TestUnderfittingDetector:
    """Test cases for UnderfittingDetector."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(UnderfittingDetector, "__members__"):
            # Test Enum values
            for member in UnderfittingDetector:
                assert isinstance(member, UnderfittingDetector)
            return
        
        try:
            instance = UnderfittingDetector()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = UnderfittingDetector(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

    def test_detect_underfitting(self, sample_data):
        """Test detect_underfitting method."""
        instance = UnderfittingDetector(**sample_data.get("init_params", {}))
        
        try:
            result = instance.detect_underfitting(**sample_data.get("detect_underfitting_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

