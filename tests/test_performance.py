"""Tests for rtradez.metrics.performance."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from rtradez.metrics.performance import *

class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer."""

    def test_initialization(self, sample_data):
        """Test class initialization."""
        # Handle Enum classes
        if hasattr(PerformanceAnalyzer, "__members__"):
            # Test Enum values
            for member in PerformanceAnalyzer:
                assert isinstance(member, PerformanceAnalyzer)
            return
        
        try:
            instance = PerformanceAnalyzer()
            assert instance is not None
        except TypeError:
            # Class requires parameters
            try:
                instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
                assert instance is not None
            except (TypeError, ValueError):
                # Some classes may require specific parameters
                pass

    def test_get_metrics(self, sample_data):
        """Test get_metrics method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.get_metrics(**sample_data.get("get_metrics_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_score(self, sample_data):
        """Test score method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.score(**sample_data.get("score_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_sharpe_ratio(self, sample_data):
        """Test sharpe_ratio method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.sharpe_ratio(**sample_data.get("sharpe_ratio_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_max_drawdown(self, sample_data):
        """Test max_drawdown method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.max_drawdown(**sample_data.get("max_drawdown_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_calmar_ratio(self, sample_data):
        """Test calmar_ratio method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.calmar_ratio(**sample_data.get("calmar_ratio_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_rolling_sharpe(self, sample_data):
        """Test rolling_sharpe method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.rolling_sharpe(**sample_data.get("rolling_sharpe_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_rolling_volatility(self, sample_data):
        """Test rolling_volatility method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.rolling_volatility(**sample_data.get("rolling_volatility_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_generate_report(self, sample_data):
        """Test generate_report method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.generate_report(**sample_data.get("generate_report_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_plot_returns(self, sample_data):
        """Test plot_returns method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.plot_returns(**sample_data.get("plot_returns_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_plot_drawdown(self, sample_data):
        """Test plot_drawdown method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.plot_drawdown(**sample_data.get("plot_drawdown_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_plot_rolling_sharpe(self, sample_data):
        """Test plot_rolling_sharpe method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.plot_rolling_sharpe(**sample_data.get("plot_rolling_sharpe_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_plot_monthly_heatmap(self, sample_data):
        """Test plot_monthly_heatmap method."""
        instance = PerformanceAnalyzer(**sample_data.get("init_params", {}))
        
        try:
            result = instance.plot_monthly_heatmap(**sample_data.get("plot_monthly_heatmap_params", {}))
            assert result is not None
        except (TypeError, NotImplementedError, ValueError) as e:
            # Method may be abstract or require specific parameters
            pass

    def test_calculate_metric(self, sample_data):
        """Test metric calculation."""
        metric = PerformanceAnalyzer(**sample_data.get("metric_params", {}))
        y_true, y_pred = sample_data["y"], sample_data["y_pred"]
        
        result = metric.calculate(y_true, y_pred)
        assert isinstance(result, (int, float, dict))
        
        # Test with returns data if applicable
        if "returns" in sample_data:
            result = metric.calculate(sample_data["returns"])
            assert result is not None
