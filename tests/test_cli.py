"""
Comprehensive tests for CLI interface modules.

Tests for main CLI, trading benchmark CLI, and other command interfaces.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import json
import tempfile
import io
import sys

from typer.testing import CliRunner

# Import CLI modules
from rtradez.cli.main import app as main_app
from rtradez.cli.trading_benchmark import app as trading_benchmark_app
from rtradez.cli import trading_benchmark
from rtradez.trading_benchmarks import BacktestConfig, StrategyBenchmark, BacktestResults, TradingMetrics


class TestMainCLI:
    """Test main CLI application."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_main_app_help(self, runner):
        """Test main app help command."""
        result = runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0
        assert "RTradez" in result.output
        assert "Options Trading Framework" in result.output
    
    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(main_app, ["version"])
        assert result.exit_code == 0
        assert "RTradez" in result.output
        assert "Version:" in result.output
    
    def test_status_command(self, runner):
        """Test status command."""
        result = runner.invoke(main_app, ["status"])
        assert result.exit_code == 0
        assert "RTradez System Status" in result.output
        assert "Component Status" in result.output
        assert "Trading Benchmarks" in result.output
    
    def test_demo_command(self, runner):
        """Test demo command interface."""
        # Test demo help
        result = runner.invoke(main_app, ["demo", "--help"])
        assert result.exit_code == 0
        assert "demo" in result.output.lower()
    
    @patch('rtradez.cli.main.typer.run')
    def test_demo_risk_management(self, mock_typer_run, runner):
        """Test demo risk management selection."""
        # Mock user input for demo selection
        with patch('rtradez.cli.main.Prompt.ask', return_value="1"):
            result = runner.invoke(main_app, ["demo"])
            assert result.exit_code == 0
    
    def test_verbose_flag(self, runner):
        """Test verbose flag functionality."""
        result = runner.invoke(main_app, ["--verbose", "status"])
        assert result.exit_code == 0
        # Should include verbose output
        assert "verbose" in result.output.lower() or "RTradez System Status" in result.output
    
    def test_debug_flag(self, runner):
        """Test debug flag functionality."""
        result = runner.invoke(main_app, ["--debug", "status"])
        assert result.exit_code == 0
        # Should include debug output or normal status
        assert "debug" in result.output.lower() or "RTradez System Status" in result.output


class TestTradingBenchmarkCLI:
    """Test trading benchmark CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_returns_csv(self, temp_dir):
        """Create sample returns CSV file."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        
        csv_file = temp_dir / "sample_returns.csv"
        returns.to_csv(csv_file, header=['returns'])
        return csv_file
    
    @pytest.fixture
    def sample_benchmark_csv(self, temp_dir):
        """Create sample benchmark CSV file."""
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)
        
        csv_file = temp_dir / "sample_benchmark.csv"
        returns.to_csv(csv_file, header=['returns'])
        return csv_file
    
    def test_trading_benchmark_help(self, runner):
        """Test trading benchmark help command."""
        result = runner.invoke(trading_benchmark_app, ["--help"])
        assert result.exit_code == 0
        assert "Trading strategy performance benchmarks" in result.output
    
    @patch('rtradez.cli.trading_benchmark.StrategyBenchmark')
    @patch('rtradez.cli.trading_benchmark.BacktestConfig')
    def test_backtest_command(self, mock_config, mock_benchmark, runner):
        """Test backtest command."""
        # Mock the benchmark and config
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        mock_benchmark_instance = Mock()
        mock_benchmark.return_value = mock_benchmark_instance
        
        # Mock the results
        mock_results = Mock(spec=BacktestResults)
        mock_results.total_return = 0.15
        mock_results.annualized_return = 0.12
        mock_results.sharpe_ratio = 1.2
        mock_results.max_drawdown = -0.10
        mock_results.win_rate = 0.6
        mock_results.profit_factor = 1.5
        mock_results.total_trades = 10
        mock_results.winning_trades = 6
        mock_results.losing_trades = 4
        mock_results.avg_win = 100.0
        mock_results.avg_loss = -50.0
        mock_results.expectancy = 10.0
        
        mock_benchmark_instance.calculate_performance_metrics.return_value = mock_results
        mock_benchmark_instance.run_benchmark_comparison.return_value = {
            'alpha': 0.05,
            'beta': 0.8,
            'correlation': 0.7,
            'information_ratio': 0.4,
            'excess_return': 0.03
        }
        
        result = runner.invoke(trading_benchmark_app, [
            "backtest",
            "--strategy", "covered_call",
            "--start", "2023-01-01",
            "--end", "2023-12-31",
            "--capital", "100000",
            "--no-save"
        ])
        
        assert result.exit_code == 0
        assert "RTradez Strategy Backtesting" in result.output
        
        # Verify mocks were called
        mock_config.assert_called_once()
        mock_benchmark.assert_called_once()
        mock_benchmark_instance.run_simple_covered_call_strategy.assert_called_once()
        mock_benchmark_instance.calculate_performance_metrics.assert_called_once()
    
    def test_backtest_invalid_strategy(self, runner):
        """Test backtest with invalid strategy."""
        result = runner.invoke(trading_benchmark_app, [
            "backtest",
            "--strategy", "invalid_strategy",
            "--start", "2023-01-01",
            "--end", "2023-12-31"
        ])
        
        # Should exit gracefully with error message
        assert "not yet implemented" in result.output or result.exit_code != 0
    
    def test_backtest_invalid_dates(self, runner):
        """Test backtest with invalid date format."""
        result = runner.invoke(trading_benchmark_app, [
            "backtest",
            "--start", "invalid-date",
            "--end", "2023-12-31"
        ])
        
        assert result.exit_code != 0 or "Invalid date format" in result.output
    
    def test_analyze_command(self, runner, sample_returns_csv, sample_benchmark_csv):
        """Test analyze command with CSV files."""
        with patch('rtradez.cli.trading_benchmark.PerformanceAnalyzer') as mock_analyzer:
            # Mock analyzer
            mock_analyzer_instance = Mock()
            mock_analyzer.return_value = mock_analyzer_instance
            
            mock_metrics = Mock(spec=TradingMetrics)
            mock_metrics.total_return = 0.15
            mock_metrics.annualized_return = 0.12
            mock_metrics.volatility = 0.20
            mock_metrics.sharpe_ratio = 0.6
            mock_metrics.sortino_ratio = 0.8
            mock_metrics.max_drawdown = -0.10
            mock_metrics.var_95 = -0.03
            mock_metrics.skewness = 0.1
            mock_metrics.kurtosis = 0.2
            mock_metrics.total_trades = 0
            mock_metrics.win_rate = 0.0
            mock_metrics.profit_factor = 0.0
            mock_metrics.expectancy = 0.0
            mock_metrics.alpha = 0.05
            mock_metrics.beta = 0.8
            mock_metrics.information_ratio = 0.4
            
            mock_analyzer_instance.generate_comprehensive_metrics.return_value = mock_metrics
            
            result = runner.invoke(trading_benchmark_app, [
                "analyze",
                str(sample_returns_csv),
                "--benchmark", str(sample_benchmark_csv),
                "--format", "table"
            ])
            
            assert result.exit_code == 0
            mock_analyzer.assert_called_once()
    
    def test_analyze_missing_file(self, runner):
        """Test analyze command with missing file."""
        result = runner.invoke(trading_benchmark_app, [
            "analyze",
            "nonexistent_file.csv"
        ])
        
        assert result.exit_code != 0 or "not found" in result.output
    
    def test_analyze_invalid_csv(self, runner, temp_dir):
        """Test analyze command with invalid CSV structure."""
        # Create CSV without 'returns' column
        invalid_csv = temp_dir / "invalid.csv"
        pd.DataFrame({'price': [100, 101, 102]}).to_csv(invalid_csv)
        
        result = runner.invoke(trading_benchmark_app, [
            "analyze",
            str(invalid_csv)
        ])
        
        assert result.exit_code != 0 or "must have a 'returns' column" in result.output
    
    def test_compare_command(self, runner, sample_returns_csv, temp_dir):
        """Test compare command with two strategy files."""
        # Create second strategy file
        np.random.seed(456)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        returns2 = pd.Series(np.random.normal(0.0012, 0.025, len(dates)), index=dates)
        strategy2_csv = temp_dir / "strategy2.csv"
        returns2.to_csv(strategy2_csv, header=['returns'])
        
        with patch('rtradez.cli.trading_benchmark.PerformanceAnalyzer') as mock_analyzer:
            mock_analyzer_instance = Mock()
            mock_analyzer.return_value = mock_analyzer_instance
            
            mock_metrics = Mock(spec=TradingMetrics)
            mock_metrics.total_return = 0.15
            mock_metrics.annualized_return = 0.12
            mock_metrics.volatility = 0.20
            mock_metrics.sharpe_ratio = 0.6
            mock_metrics.max_drawdown = -0.10
            mock_metrics.win_rate = 0.6
            
            mock_analyzer_instance.generate_comprehensive_metrics.return_value = mock_metrics
            
            result = runner.invoke(trading_benchmark_app, [
                "compare",
                str(sample_returns_csv),
                str(strategy2_csv),
                "--names", "Strategy1", "Strategy2"
            ])
            
            assert result.exit_code == 0
            assert "Strategy Performance Comparison" in result.output
    
    @patch('rtradez.cli.trading_benchmark.np.random.choice')
    @patch('rtradez.cli.trading_benchmark.pd.read_csv')
    def test_monte_carlo_command(self, mock_read_csv, mock_choice, runner, sample_returns_csv):
        """Test Monte Carlo simulation command."""
        # Mock CSV reading
        np.random.seed(42)
        returns_data = pd.DataFrame({
            'returns': np.random.normal(0.001, 0.02, 252)
        }, index=pd.date_range('2023-01-01', periods=252, freq='D'))
        mock_read_csv.return_value = returns_data
        
        # Mock bootstrap sampling
        mock_choice.side_effect = lambda arr, size, replace: np.random.choice(arr, size, replace)
        
        result = runner.invoke(trading_benchmark_app, [
            "monte-carlo",
            str(sample_returns_csv),
            "--sims", "100",
            "--periods", "252",
            "--method", "bootstrap"
        ])
        
        assert result.exit_code == 0
        assert "Monte Carlo Strategy Analysis" in result.output
        mock_read_csv.assert_called()
    
    def test_monte_carlo_missing_file(self, runner):
        """Test Monte Carlo with missing file."""
        result = runner.invoke(trading_benchmark_app, [
            "monte-carlo",
            "nonexistent.csv"
        ])
        
        assert result.exit_code != 0 or "not found" in result.output
    
    def test_report_command(self, runner, temp_dir):
        """Test report generation command."""
        # Create sample result files
        results_dir = temp_dir / "results"
        results_dir.mkdir()
        
        sample_result = {
            "total_return": 0.15,
            "annualized_return": 0.12,
            "sharpe_ratio": 1.2,
            "config": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            }
        }
        
        result_file = results_dir / "backtest_result.json"
        with open(result_file, 'w') as f:
            json.dump(sample_result, f)
        
        result = runner.invoke(trading_benchmark_app, [
            "report",
            "--dir", str(results_dir),
            "--format", "summary"
        ])
        
        assert result.exit_code == 0
        assert "Trading Performance Report" in result.output
    
    def test_report_no_files(self, runner, temp_dir):
        """Test report with no result files."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        result = runner.invoke(trading_benchmark_app, [
            "report",
            "--dir", str(empty_dir)
        ])
        
        assert result.exit_code != 0 or "No result files found" in result.output


class TestCLIHelpers:
    """Test CLI helper functions."""
    
    def test_display_backtest_results(self):
        """Test backtest results display function."""
        # Mock results
        mock_results = Mock()
        mock_results.total_return = 0.15
        mock_results.annualized_return = 0.12
        mock_results.sharpe_ratio = 1.2
        mock_results.max_drawdown = -0.10
        mock_results.win_rate = 0.6
        mock_results.profit_factor = 1.5
        mock_results.total_trades = 10
        mock_results.winning_trades = 6
        mock_results.losing_trades = 4
        mock_results.avg_win = 100.0
        mock_results.avg_loss = -50.0
        mock_results.expectancy = 10.0
        
        mock_comparison = {
            'alpha': 0.05,
            'beta': 0.8,
            'correlation': 0.7,
            'information_ratio': 0.4,
            'excess_return': 0.03
        }
        
        # Capture console output
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            trading_benchmark._display_backtest_results(mock_results, mock_comparison, verbose=True)
            output = mock_stdout.getvalue()
            
            assert "Strategy Performance Summary" in output
            assert "15.00%" in output  # Total return
            assert "1.20" in output     # Sharpe ratio
    
    def test_display_performance_table(self):
        """Test performance table display function."""
        mock_metrics = Mock(spec=TradingMetrics)
        mock_metrics.total_return = 0.15
        mock_metrics.annualized_return = 0.12
        mock_metrics.volatility = 0.20
        mock_metrics.sharpe_ratio = 0.6
        mock_metrics.sortino_ratio = 0.8
        mock_metrics.calmar_ratio = 1.2
        mock_metrics.max_drawdown = -0.10
        mock_metrics.var_95 = -0.03
        mock_metrics.cvar_95 = -0.04
        mock_metrics.skewness = 0.1
        mock_metrics.kurtosis = 0.2
        mock_metrics.tail_ratio = 1.2
        mock_metrics.total_trades = 10
        mock_metrics.win_rate = 0.6
        mock_metrics.profit_factor = 1.5
        mock_metrics.expectancy = 0.05
        mock_metrics.alpha = 0.05
        mock_metrics.beta = 0.8
        mock_metrics.information_ratio = 0.4
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            trading_benchmark._display_performance_table(mock_metrics)
            output = mock_stdout.getvalue()
            
            assert "Performance Metrics" in output
            assert "Returns" in output
            assert "Risk-Adjusted" in output
    
    def test_display_strategy_comparison(self):
        """Test strategy comparison display function."""
        mock_metrics1 = Mock()
        mock_metrics1.total_return = 0.15
        mock_metrics1.sharpe_ratio = 1.2
        mock_metrics1.max_drawdown = -0.10
        mock_metrics1.volatility = 0.20
        mock_metrics1.win_rate = 0.6
        
        mock_metrics2 = Mock()
        mock_metrics2.total_return = 0.12
        mock_metrics2.sharpe_ratio = 1.0
        mock_metrics2.max_drawdown = -0.08
        mock_metrics2.volatility = 0.18
        mock_metrics2.win_rate = 0.65
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            trading_benchmark._display_strategy_comparison(
                mock_metrics1, mock_metrics2, "Strategy A", "Strategy B", "sharpe"
            )
            output = mock_stdout.getvalue()
            
            assert "Strategy Comparison" in output
            assert "Strategy A" in output
            assert "Strategy B" in output
    
    def test_display_monte_carlo_results(self):
        """Test Monte Carlo results display function."""
        results = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
        mean_return = 0.20
        std_return = 0.08
        lower_ci = 0.12
        upper_ci = 0.28
        var_5 = 0.11
        prob_loss = 0.30
        prob_large_loss = 0.05
        confidence = 0.95
        periods = 252
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            trading_benchmark._display_monte_carlo_results(
                results, mean_return, std_return, lower_ci, upper_ci,
                var_5, prob_loss, prob_large_loss, confidence, periods
            )
            output = mock_stdout.getvalue()
            
            assert "Monte Carlo Results" in output
            assert "20.00%" in output  # Mean return
            assert "95% CI" in output


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI components."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_main_to_subcommand_integration(self, runner):
        """Test integration between main CLI and subcommands."""
        # Test that main CLI can route to subcommands
        result = runner.invoke(main_app, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "Trading strategy benchmarks" in result.output
    
    @patch('rtradez.cli.trading_benchmark.StrategyBenchmark')
    def test_backtest_save_functionality(self, mock_benchmark, runner):
        """Test backtest save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock benchmark
            mock_benchmark_instance = Mock()
            mock_benchmark.return_value = mock_benchmark_instance
            
            mock_results = Mock()
            mock_results.dict.return_value = {"total_return": 0.15}
            mock_benchmark_instance.calculate_performance_metrics.return_value = mock_results
            mock_benchmark_instance.run_benchmark_comparison.return_value = {"alpha": 0.05}
            
            # Change to temp directory
            import os
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                result = runner.invoke(trading_benchmark_app, [
                    "backtest",
                    "--strategy", "covered_call",
                    "--start", "2023-01-01",
                    "--end", "2023-03-01",
                    "--save"
                ])
                
                assert result.exit_code == 0
                
                # Check if file was created
                saved_files = list(Path(temp_dir).glob("backtest_*.json"))
                assert len(saved_files) > 0
                
            finally:
                os.chdir(original_cwd)
    
    def test_cli_error_handling(self, runner):
        """Test CLI error handling."""
        # Test invalid command
        result = runner.invoke(main_app, ["invalid_command"])
        assert result.exit_code != 0
        
        # Test invalid subcommand
        result = runner.invoke(trading_benchmark_app, ["invalid_subcommand"])
        assert result.exit_code != 0
    
    @patch('rtradez.cli.main.subprocess.run')
    def test_demo_integration_workflow(self, mock_subprocess, runner):
        """Test demo integration workflow."""
        # Mock subprocess for integration demo
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('rtradez.cli.main.Prompt.ask', return_value="7"):  # Integration demo
            result = runner.invoke(main_app, ["demo"])
            assert result.exit_code == 0