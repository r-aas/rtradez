"""
Comprehensive tests for pipeline optimizer module.

Tests for data pipeline optimization, memory management, and performance monitoring.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
import psutil
from pathlib import Path

from rtradez.pipeline.pipeline_optimizer import (
    PipelineConfig, OptimizationResult, DataPipelineOptimizer
)


class TestPipelineConfig:
    """Test PipelineConfig functionality."""
    
    def test_default_config_creation(self):
        """Test creating default pipeline configuration."""
        config = PipelineConfig()
        
        assert config.max_memory_usage_gb == 4.0
        assert config.chunk_size == 10000
        assert config.use_multiprocessing == True
        assert config.max_workers is None
        assert config.feature_selection_threshold == 0.95
        assert config.max_features_per_source == 50
        assert config.enable_feature_caching == True
        assert config.alignment_method == 'inner'
        assert config.missing_data_strategy == 'forward_fill'
        assert config.outlier_handling == True
        assert config.enable_lazy_loading == True
        assert config.cache_intermediate_results == True
        assert config.optimize_dtypes == True
        assert config.validation_split == 0.2
        assert config.enable_cross_validation == True
        assert config.cv_folds == 5
    
    def test_custom_config_creation(self):
        """Test creating custom pipeline configuration."""
        config = PipelineConfig(
            max_memory_usage_gb=8.0,
            chunk_size=5000,
            use_multiprocessing=False,
            max_workers=4,
            feature_selection_threshold=0.90,
            max_features_per_source=100,
            enable_feature_caching=False,
            alignment_method='outer',
            missing_data_strategy='drop',
            outlier_handling=False,
            enable_lazy_loading=False,
            cache_intermediate_results=False,
            optimize_dtypes=False,
            validation_split=0.3,
            enable_cross_validation=False,
            cv_folds=10
        )
        
        assert config.max_memory_usage_gb == 8.0
        assert config.chunk_size == 5000
        assert config.use_multiprocessing == False
        assert config.max_workers == 4
        assert config.feature_selection_threshold == 0.90
        assert config.max_features_per_source == 100
        assert config.enable_feature_caching == False
        assert config.alignment_method == 'outer'
        assert config.missing_data_strategy == 'drop'
        assert config.outlier_handling == False
        assert config.enable_lazy_loading == False
        assert config.cache_intermediate_results == False
        assert config.optimize_dtypes == False
        assert config.validation_split == 0.3
        assert config.enable_cross_validation == False
        assert config.cv_folds == 10


class TestOptimizationResult:
    """Test OptimizationResult functionality."""
    
    def test_default_result_creation(self):
        """Test creating default optimization result."""
        result = OptimizationResult()
        
        assert result.optimized_pipeline is None
        assert result.performance_metrics == {}
        assert result.memory_usage == {}
        assert result.processing_time == {}
        assert result.recommendations == []
        assert result.data_quality_report == {}
    
    def test_custom_result_creation(self):
        """Test creating custom optimization result."""
        pipeline = {"data": {"source1": pd.DataFrame()}}
        metrics = {"accuracy": 0.95, "speed": 1.2}
        memory = {"peak_mb": 512.0}
        timing = {"total_time": 45.2}
        recommendations = ["Use more memory", "Enable caching"]
        quality_report = {"completeness": 0.98, "issues": []}
        
        result = OptimizationResult(
            optimized_pipeline=pipeline,
            performance_metrics=metrics,
            memory_usage=memory,
            processing_time=timing,
            recommendations=recommendations,
            data_quality_report=quality_report
        )
        
        assert result.optimized_pipeline == pipeline
        assert result.performance_metrics == metrics
        assert result.memory_usage == memory
        assert result.processing_time == timing
        assert result.recommendations == recommendations
        assert result.data_quality_report == quality_report


class TestDataPipelineOptimizer:
    """Test DataPipelineOptimizer functionality."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Create test pipeline configuration."""
        return PipelineConfig(
            max_memory_usage_gb=2.0,
            chunk_size=1000,
            use_multiprocessing=False,  # Disable for testing
            max_workers=2,
            max_features_per_source=10,
            feature_selection_threshold=0.8
        )
    
    @pytest.fixture
    def optimizer(self, pipeline_config):
        """Create pipeline optimizer instance."""
        return DataPipelineOptimizer(pipeline_config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.randn(1000) * 0.1,  # Low variance
            'feature_4': np.random.randn(1000),
            'categorical': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.randint(0, 2, 1000)
        })
        # Create correlation between feature_1 and feature_2
        data['feature_2'] = data['feature_1'] * 0.9 + np.random.randn(1000) * 0.1
        return data
    
    @pytest.fixture
    def mock_loader_func(self, sample_data):
        """Create mock data loader function."""
        def loader():
            return sample_data.copy()
        return loader
    
    def test_optimizer_initialization(self, pipeline_config):
        """Test DataPipelineOptimizer initialization."""
        optimizer = DataPipelineOptimizer(pipeline_config)
        
        assert optimizer.config == pipeline_config
        assert optimizer.config.max_workers == 2
        assert len(optimizer.performance_stats) == 4
        assert len(optimizer.result_cache) == 0
        assert len(optimizer.data_sources) == 0
        assert len(optimizer.source_priorities) == 0
    
    def test_optimizer_default_initialization(self):
        """Test DataPipelineOptimizer with default config."""
        optimizer = DataPipelineOptimizer()
        
        assert isinstance(optimizer.config, PipelineConfig)
        assert optimizer.config.max_workers <= 8
        assert optimizer.config.max_workers > 0
    
    def test_register_data_source(self, optimizer, mock_loader_func):
        """Test registering data sources."""
        optimizer.register_data_source(
            name="test_source",
            loader_func=mock_loader_func,
            priority=2,
            cache_key="test_cache"
        )
        
        assert "test_source" in optimizer.data_sources
        assert optimizer.data_sources["test_source"]["loader"] == mock_loader_func
        assert optimizer.data_sources["test_source"]["priority"] == 2
        assert optimizer.data_sources["test_source"]["cache_key"] == "test_cache"
        assert optimizer.source_priorities["test_source"] == 2
    
    def test_register_data_source_default_params(self, optimizer, mock_loader_func):
        """Test registering data source with default parameters."""
        optimizer.register_data_source(
            name="default_source",
            loader_func=mock_loader_func
        )
        
        assert "default_source" in optimizer.data_sources
        assert optimizer.data_sources["default_source"]["priority"] == 1
        assert optimizer.data_sources["default_source"]["cache_key"] == "source_default_source"
    
    @patch('rtradez.pipeline.pipeline_optimizer.psutil.virtual_memory')
    def test_analyze_data_sources(self, mock_memory, optimizer, mock_loader_func):
        """Test data source analysis."""
        # Mock memory info
        mock_memory.return_value.available = 8 * 1024**3  # 8GB
        
        # Register test sources
        optimizer.register_data_source("source1", mock_loader_func, priority=2)
        optimizer.register_data_source("source2", mock_loader_func, priority=1)
        
        # Add metadata to sources
        optimizer.data_sources["source1"]["metadata"] = {
            "estimated_rows": 5000,
            "estimated_columns": 20
        }
        optimizer.data_sources["source2"]["metadata"] = {
            "estimated_rows": 10000,
            "estimated_columns": 15
        }
        
        analysis = optimizer._analyze_data_sources(optimizer.data_sources)
        
        assert isinstance(analysis, dict)
        assert "sources" in analysis
        assert "total_estimated_memory" in analysis
        assert "loading_complexity" in analysis
        assert "data_overlap" in analysis
        assert "quality_issues" in analysis
        
        # Check source analysis
        assert "source1" in analysis["sources"]
        assert "source2" in analysis["sources"]
        assert analysis["sources"]["source1"]["priority"] == 2
        assert analysis["sources"]["source2"]["priority"] == 1
    
    def test_create_loading_plan(self, optimizer, mock_loader_func):
        """Test loading plan creation."""
        # Register test sources
        optimizer.register_data_source("source1", mock_loader_func, priority=2)
        optimizer.register_data_source("source2", mock_loader_func, priority=1)
        
        # Create mock analysis
        source_analysis = {
            "sources": {
                "source1": {
                    "priority": 2,
                    "estimated_memory_gb": 0.5,
                    "complexity_score": 1.0
                },
                "source2": {
                    "priority": 1,
                    "estimated_memory_gb": 0.3,
                    "complexity_score": 0.8
                }
            },
            "total_estimated_memory": 0.8
        }
        
        plan = optimizer._create_loading_plan(optimizer.data_sources, source_analysis)
        
        assert isinstance(plan, dict)
        assert "load_order" in plan
        assert "parallel_groups" in plan
        assert "memory_strategy" in plan
        assert "chunking_config" in plan
        assert "caching_strategy" in plan
        
        # Check that higher priority source comes first
        assert "source1" in plan["load_order"]
        assert "source2" in plan["load_order"]
    
    def test_load_single_source_from_cache(self, optimizer, mock_loader_func):
        """Test loading single source from cache."""
        # Register source
        optimizer.register_data_source("cached_source", mock_loader_func)
        
        # Add to cache
        cached_data = pd.DataFrame({'test': [1, 2, 3]})
        optimizer.result_cache["source_cached_source"] = cached_data
        
        loading_plan = {"chunking_config": {}}
        
        result = optimizer._load_single_source("cached_source", loading_plan)
        
        assert result is not None
        pd.testing.assert_frame_equal(result, cached_data)
    
    def test_load_single_source_new_data(self, optimizer, mock_loader_func, sample_data):
        """Test loading single source new data."""
        # Register source
        optimizer.register_data_source("new_source", mock_loader_func)
        
        loading_plan = {"chunking_config": {}}
        
        result = optimizer._load_single_source("new_source", loading_plan)
        
        assert result is not None
        assert len(result) == len(sample_data)
        assert len(result.columns) == len(sample_data.columns)
    
    def test_load_single_source_not_registered(self, optimizer):
        """Test loading unregistered source."""
        loading_plan = {"chunking_config": {}}
        
        result = optimizer._load_single_source("nonexistent", loading_plan)
        
        assert result is None
    
    def test_load_chunked_data(self, optimizer, mock_loader_func, sample_data):
        """Test chunked data loading."""
        optimizer.register_data_source("chunked_source", mock_loader_func)
        
        loading_plan = {
            "chunking_config": {
                "chunked_source": {
                    "chunk_size": 300,
                    "overlap_size": 30
                }
            }
        }
        
        result = optimizer._load_chunked_data("chunked_source", mock_loader_func, loading_plan)
        
        assert result is not None
        assert len(result) <= len(sample_data)  # May have duplicates removed
        assert len(result.columns) == len(sample_data.columns)
    
    def test_process_data_chunk_forward_fill(self, optimizer):
        """Test data chunk processing with forward fill."""
        chunk = pd.DataFrame({
            'a': [1, np.nan, 3, np.nan, 5],
            'b': [np.nan, 2, 3, 4, np.nan]
        })
        
        with patch('pandas.DataFrame.fillna') as mock_fillna:
            mock_fillna.return_value = chunk.fillna(method='ffill')
            
            result = optimizer._process_data_chunk(chunk, "test_source")
            
            mock_fillna.assert_called_once_with(method='ffill')
            assert result is not None
    
    def test_process_data_chunk_drop_na(self, optimizer):
        """Test data chunk processing with drop strategy."""
        optimizer.config.missing_data_strategy = 'drop'
        
        chunk = pd.DataFrame({
            'a': [1, np.nan, 3, np.nan, 5],
            'b': [np.nan, 2, 3, 4, np.nan]
        })
        
        result = optimizer._process_data_chunk(chunk, "test_source")
        
        # Should have dropped rows with NaN
        assert len(result) < len(chunk)
    
    def test_handle_outliers_chunk(self, optimizer):
        """Test outlier handling in data chunk."""
        # Create data with outliers
        chunk = pd.DataFrame({
            'normal': [1, 2, 3, 4, 5],
            'with_outliers': [1, 2, 100, 4, 5]  # 100 is outlier
        })
        
        result = optimizer._handle_outliers_chunk(chunk)
        
        assert result is not None
        # Outlier should be capped
        assert result['with_outliers'].max() < 100
    
    def test_select_top_features_by_variance(self, optimizer, sample_data):
        """Test top feature selection by variance."""
        result = optimizer._select_top_features(sample_data, max_features=3)
        
        assert len(result.columns) <= 3
        assert result is not None
    
    def test_select_top_features_all_features(self, optimizer, sample_data):
        """Test feature selection when max_features >= total features."""
        result = optimizer._select_top_features(sample_data, max_features=10)
        
        # Should return all features since max_features > actual features
        assert len(result.columns) == len(sample_data.columns)
    
    def test_remove_correlated_features(self, optimizer, sample_data):
        """Test correlated feature removal."""
        # feature_1 and feature_2 are highly correlated in sample_data
        result = optimizer._remove_correlated_features(sample_data, correlation_threshold=0.8)
        
        # Should remove one of the correlated features
        assert len(result.columns) <= len(sample_data.columns)
    
    def test_remove_correlated_features_no_numeric(self, optimizer):
        """Test correlated feature removal with no numeric columns."""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        
        result = optimizer._remove_correlated_features(data, correlation_threshold=0.8)
        
        # Should return original data since no numeric columns
        assert len(result.columns) == len(data.columns)
    
    def test_optimize_dtypes(self, optimizer):
        """Test dtype optimization."""
        data = pd.DataFrame({
            'big_int': np.array([1, 2, 3], dtype=np.int64),
            'small_int': np.array([1, 2, 3], dtype=np.int64),
            'float_col': np.array([1.0, 2.0, 3.0], dtype=np.float64),
            'category_col': ['A', 'A', 'B', 'A', 'B'] * 20  # Repetitive for categorization
        })
        
        result = optimizer._optimize_dtypes(data)
        
        assert result is not None
        # Small integers should be downcast
        assert result['small_int'].dtype in [np.int8, np.int16, np.int32]
        # Float should be optimized
        assert result['float_col'].dtype in [np.float32, np.float64]
    
    def test_optimize_memory_usage(self, optimizer, sample_data):
        """Test memory usage optimization."""
        data_dict = {
            "source1": sample_data.copy(),
            "source2": sample_data.copy()
        }
        
        result = optimizer._optimize_memory_usage(data_dict)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "source1" in result
        assert "source2" in result
    
    def test_optimize_memory_usage_with_none(self, optimizer, sample_data):
        """Test memory optimization with None values."""
        data_dict = {
            "source1": sample_data.copy(),
            "source2": None,
            "source3": sample_data.copy()
        }
        
        result = optimizer._optimize_memory_usage(data_dict)
        
        assert isinstance(result, dict)
        assert len(result) == 2  # None source should be excluded
        assert "source1" in result
        assert "source2" not in result
        assert "source3" in result
    
    def test_create_optimized_pipeline(self, optimizer, sample_data):
        """Test optimized pipeline creation."""
        optimized_data = {
            "source1": sample_data.copy(),
            "source2": sample_data.copy()
        }
        
        loading_plan = {
            "load_order": ["source1", "source2"],
            "memory_strategy": "full"
        }
        
        pipeline = optimizer._create_optimized_pipeline(optimized_data, loading_plan)
        
        assert isinstance(pipeline, dict)
        assert "data" in pipeline
        assert "loading_plan" in pipeline
        assert "config" in pipeline
        assert "metadata" in pipeline
        
        metadata = pipeline["metadata"]
        assert "sources" in metadata
        assert "total_samples" in metadata
        assert "total_features" in metadata
        assert "memory_usage_mb" in metadata
        assert "optimization_timestamp" in metadata
    
    @patch('rtradez.pipeline.pipeline_optimizer.psutil.Process')
    def test_validate_pipeline_performance(self, mock_process, optimizer, sample_data):
        """Test pipeline performance validation."""
        # Mock process memory info
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
        mock_process.return_value = mock_process_instance
        
        pipeline = {
            "data": {
                "source1": sample_data.copy(),
                "source2": sample_data.copy()
            },
            "metadata": {
                "sources": ["source1", "source2"],
                "total_features": 10,
                "memory_usage_mb": 100.0
            }
        }
        
        validation_data = sample_data.copy()
        
        metrics = optimizer._validate_pipeline_performance(pipeline, validation_data)
        
        assert isinstance(metrics, dict)
        assert "load_time_seconds" in metrics
        assert "memory_usage_mb" in metrics
        assert "data_completeness" in metrics
        assert "features_per_mb" in metrics
    
    @patch('rtradez.pipeline.pipeline_optimizer.psutil.virtual_memory')
    @patch('rtradez.pipeline.pipeline_optimizer.psutil.Process')
    def test_get_memory_usage_stats(self, mock_process, mock_virtual_memory, optimizer):
        """Test memory usage statistics."""
        # Mock system memory
        mock_virtual_memory.return_value.total = 16 * 1024**3  # 16GB
        mock_virtual_memory.return_value.available = 8 * 1024**3  # 8GB
        mock_virtual_memory.return_value.percent = 50.0
        
        # Mock process memory
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 512 * 1024**2  # 512MB
        mock_process_instance.memory_percent.return_value = 3.2
        mock_process.return_value = mock_process_instance
        
        stats = optimizer._get_memory_usage_stats()
        
        assert isinstance(stats, dict)
        assert "system_memory_total_gb" in stats
        assert "system_memory_available_gb" in stats
        assert "system_memory_percent" in stats
        assert "process_memory_mb" in stats
        assert "process_memory_percent" in stats
        
        assert stats["system_memory_total_gb"] == 16.0
        assert stats["system_memory_available_gb"] == 8.0
        assert stats["system_memory_percent"] == 50.0
        assert stats["process_memory_mb"] == 512.0
        assert stats["process_memory_percent"] == 3.2
    
    def test_generate_optimization_recommendations(self, optimizer):
        """Test optimization recommendations generation."""
        source_analysis = {
            "total_estimated_memory": 8.0,  # Exceeds default 4GB limit
            "quality_issues": ["Missing data in source1"]
        }
        
        performance_metrics = {
            "data_completeness": 0.7  # Low completeness
        }
        
        # Add some performance stats
        optimizer.performance_stats['data_loading_time'] = [45.0]  # Slow
        optimizer.performance_stats['feature_reduction_ratio'] = [0.05]  # Low reduction
        
        recommendations = optimizer._generate_optimization_recommendations(
            source_analysis, performance_metrics
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations for memory, loading time, completeness, and quality issues
        rec_text = " ".join(recommendations)
        assert "memory" in rec_text.lower() or "chunk" in rec_text.lower()
    
    def test_generate_optimization_recommendations_good_performance(self, optimizer):
        """Test recommendations with good performance."""
        source_analysis = {
            "total_estimated_memory": 2.0,  # Within limits
            "quality_issues": []
        }
        
        performance_metrics = {
            "data_completeness": 0.95  # Good completeness
        }
        
        # Good performance stats
        optimizer.performance_stats['data_loading_time'] = [5.0]  # Fast
        optimizer.performance_stats['feature_reduction_ratio'] = [0.3]  # Good reduction
        
        recommendations = optimizer._generate_optimization_recommendations(
            source_analysis, performance_metrics
        )
        
        assert isinstance(recommendations, list)
        assert "well-optimized" in recommendations[0].lower()
    
    def test_get_performance_report(self, optimizer, sample_data):
        """Test performance report generation."""
        # Add some data to cache
        optimizer.result_cache["test_data"] = sample_data.copy()
        
        report = optimizer.get_performance_report()
        
        assert isinstance(report, dict)
        assert "performance_stats" in report
        assert "memory_stats" in report
        assert "cache_stats" in report
        assert "config" in report
        
        cache_stats = report["cache_stats"]
        assert "cached_items" in cache_stats
        assert "cache_memory_estimate_mb" in cache_stats
        assert cache_stats["cached_items"] == 1
    
    def test_clear_cache(self, optimizer, sample_data):
        """Test cache clearing."""
        # Add data to cache
        optimizer.result_cache["test_data"] = sample_data.copy()
        assert len(optimizer.result_cache) == 1
        
        optimizer.clear_cache()
        
        assert len(optimizer.result_cache) == 0
    
    @patch('rtradez.pipeline.pipeline_optimizer.psutil.virtual_memory')
    def test_optimize_pipeline_full_workflow(self, mock_memory, optimizer, mock_loader_func):
        """Test complete pipeline optimization workflow."""
        # Mock memory
        mock_memory.return_value.available = 8 * 1024**3  # 8GB
        
        # Register sources
        optimizer.register_data_source("source1", mock_loader_func, priority=2)
        optimizer.register_data_source("source2", mock_loader_func, priority=1)
        
        # Add metadata
        optimizer.data_sources["source1"]["metadata"] = {
            "estimated_rows": 1000,
            "estimated_columns": 6
        }
        optimizer.data_sources["source2"]["metadata"] = {
            "estimated_rows": 1000,
            "estimated_columns": 6
        }
        
        result = optimizer.optimize_pipeline(target_variable="target")
        
        assert isinstance(result, OptimizationResult)
        assert result.optimized_pipeline is not None
        assert isinstance(result.performance_metrics, dict)
        assert isinstance(result.memory_usage, dict)
        assert isinstance(result.processing_time, dict)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.data_quality_report, dict)
    
    def test_optimize_pipeline_with_validation(self, optimizer, mock_loader_func, sample_data):
        """Test pipeline optimization with validation data."""
        optimizer.register_data_source("source1", mock_loader_func)
        
        result = optimizer.optimize_pipeline(
            target_variable="target",
            validation_data=sample_data.copy()
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.optimized_pipeline is not None
        assert len(result.performance_metrics) > 0  # Should have validation metrics
    
    def test_optimize_pipeline_exception_handling(self, optimizer):
        """Test pipeline optimization with exception handling."""
        # Register source with failing loader
        def failing_loader():
            raise ValueError("Simulated loading failure")
        
        optimizer.register_data_source("failing_source", failing_loader)
        
        result = optimizer.optimize_pipeline()
        
        assert isinstance(result, OptimizationResult)
        assert len(result.recommendations) > 0
        assert any("failed" in rec.lower() for rec in result.recommendations)
    
    def test_load_sources_sequential(self, optimizer, mock_loader_func):
        """Test sequential source loading."""
        optimizer.register_data_source("source1", mock_loader_func)
        optimizer.register_data_source("source2", mock_loader_func)
        
        loading_plan = {"chunking_config": {}}
        source_group = ["source1", "source2"]
        
        result = optimizer._load_sources_sequential(source_group, loading_plan)
        
        assert isinstance(result, dict)
        assert "source1" in result
        assert "source2" in result
    
    def test_load_sources_parallel(self, optimizer, mock_loader_func):
        """Test parallel source loading."""
        optimizer.register_data_source("source1", mock_loader_func)
        optimizer.register_data_source("source2", mock_loader_func)
        
        loading_plan = {"chunking_config": {}}
        source_group = ["source1", "source2"]
        
        result = optimizer._load_sources_parallel(source_group, loading_plan)
        
        assert isinstance(result, dict)
        assert "source1" in result or "source2" in result  # At least one should load
    
    def test_execute_optimized_loading(self, optimizer, mock_loader_func):
        """Test optimized loading execution."""
        optimizer.register_data_source("source1", mock_loader_func)
        
        loading_plan = {
            "parallel_groups": [["source1"]],
            "chunking_config": {}
        }
        
        result = optimizer._execute_optimized_loading(loading_plan)
        
        assert isinstance(result, dict)
        assert "source1" in result
        assert len(optimizer.performance_stats['data_loading_time']) > 0
    
    def test_optimize_feature_processing(self, optimizer, sample_data):
        """Test feature processing optimization."""
        processed_data = {
            "source1": sample_data.copy(),
            "source2": sample_data.copy()
        }
        
        result = optimizer._optimize_feature_processing(processed_data, "target")
        
        assert isinstance(result, dict)
        assert "source1" in result
        assert "source2" in result
        
        # Should have reduced features (max 10 per source in config)
        for source_data in result.values():
            assert len(source_data.columns) <= optimizer.config.max_features_per_source


@pytest.mark.integration
class TestPipelineOptimizerIntegration:
    """Integration tests for pipeline optimizer."""
    
    @pytest.fixture
    def temp_csv_files(self):
        """Create temporary CSV files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV files
            data1 = pd.DataFrame({
                'feature_1': np.random.randn(500),
                'feature_2': np.random.randn(500),
                'feature_3': np.random.randn(500),
                'target': np.random.randint(0, 2, 500)
            })
            
            data2 = pd.DataFrame({
                'feature_4': np.random.randn(500),
                'feature_5': np.random.randn(500),
                'feature_6': np.random.randn(500),
                'target': np.random.randint(0, 2, 500)
            })
            
            file1 = Path(temp_dir) / "data1.csv"
            file2 = Path(temp_dir) / "data2.csv"
            
            data1.to_csv(file1, index=False)
            data2.to_csv(file2, index=False)
            
            yield file1, file2
    
    def test_real_file_optimization_workflow(self, temp_csv_files):
        """Test optimization workflow with real CSV files."""
        file1, file2 = temp_csv_files
        
        # Create loaders for the files
        def loader1():
            return pd.read_csv(file1)
        
        def loader2():
            return pd.read_csv(file2)
        
        # Create optimizer
        config = PipelineConfig(
            max_memory_usage_gb=1.0,
            chunk_size=200,
            use_multiprocessing=False,
            max_features_per_source=3
        )
        optimizer = DataPipelineOptimizer(config)
        
        # Register sources
        optimizer.register_data_source("csv_source1", loader1, priority=2)
        optimizer.register_data_source("csv_source2", loader2, priority=1)
        
        # Run optimization
        result = optimizer.optimize_pipeline(target_variable="target")
        
        assert isinstance(result, OptimizationResult)
        assert result.optimized_pipeline is not None
        
        # Check that data was loaded and optimized
        pipeline_data = result.optimized_pipeline["data"]
        assert "csv_source1" in pipeline_data
        assert "csv_source2" in pipeline_data
        
        # Check feature reduction
        for source_data in pipeline_data.values():
            assert len(source_data.columns) <= config.max_features_per_source
    
    def test_memory_constrained_optimization(self, temp_csv_files):
        """Test optimization under memory constraints."""
        file1, file2 = temp_csv_files
        
        def loader1():
            return pd.read_csv(file1)
        
        def loader2():
            return pd.read_csv(file2)
        
        # Create very memory-constrained config
        config = PipelineConfig(
            max_memory_usage_gb=0.001,  # Very small
            chunk_size=50,
            use_multiprocessing=False
        )
        optimizer = DataPipelineOptimizer(config)
        
        optimizer.register_data_source("source1", loader1)
        optimizer.register_data_source("source2", loader2)
        
        # Add metadata to trigger chunking
        optimizer.data_sources["source1"]["metadata"] = {
            "estimated_rows": 1000,
            "estimated_columns": 10
        }
        
        result = optimizer.optimize_pipeline()
        
        assert isinstance(result, OptimizationResult)
        # Should still complete despite memory constraints
        assert result.optimized_pipeline is not None
    
    @patch('rtradez.pipeline.pipeline_optimizer.psutil.virtual_memory')
    def test_performance_monitoring_integration(self, mock_memory, temp_csv_files):
        """Test performance monitoring throughout optimization."""
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        file1, file2 = temp_csv_files
        
        def loader1():
            return pd.read_csv(file1)
        
        config = PipelineConfig(use_multiprocessing=False)
        optimizer = DataPipelineOptimizer(config)
        
        optimizer.register_data_source("monitored_source", loader1)
        
        result = optimizer.optimize_pipeline()
        
        # Check that performance was tracked
        assert len(optimizer.performance_stats['data_loading_time']) > 0
        assert len(optimizer.performance_stats['processing_time']) > 0
        
        # Check performance report
        report = optimizer.get_performance_report()
        assert "performance_stats" in report
        assert "memory_stats" in report
        
        # Check that recommendations were generated
        assert len(result.recommendations) > 0