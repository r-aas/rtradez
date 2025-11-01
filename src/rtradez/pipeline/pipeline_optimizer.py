"""Pipeline optimization for handling multiple data sources efficiently."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
import psutil
import gc
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline optimization."""
    
    # Data processing
    max_memory_usage_gb: float = 4.0
    chunk_size: int = 10000
    use_multiprocessing: bool = True
    max_workers: Optional[int] = None
    
    # Feature processing
    feature_selection_threshold: float = 0.95  # Correlation threshold
    max_features_per_source: int = 50
    enable_feature_caching: bool = True
    
    # Data alignment
    alignment_method: str = 'inner'  # 'inner', 'outer', 'left'
    missing_data_strategy: str = 'forward_fill'
    outlier_handling: bool = True
    
    # Performance optimization
    enable_lazy_loading: bool = True
    cache_intermediate_results: bool = True
    optimize_dtypes: bool = True
    
    # Validation
    validation_split: float = 0.2
    enable_cross_validation: bool = True
    cv_folds: int = 5


@dataclass
class OptimizationResult:
    """Result of pipeline optimization."""
    
    optimized_pipeline: Any = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    processing_time: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    data_quality_report: Dict[str, Any] = field(default_factory=dict)


class DataPipelineOptimizer:
    """
    Optimize data processing pipelines for multiple sources and large datasets.
    
    Features:
    - Memory-efficient processing
    - Parallel data loading and processing
    - Intelligent feature selection
    - Automated dtype optimization
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline optimizer.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Set max workers based on system resources
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count(), 8)
        
        # Performance tracking
        self.performance_stats = {
            'data_loading_time': [],
            'processing_time': [],
            'memory_peak': [],
            'feature_reduction_ratio': []
        }
        
        # Cache for intermediate results
        self.result_cache = {}
        
        # Data source registry
        self.data_sources = {}
        self.source_priorities = {}
        
    def register_data_source(self,
                           name: str,
                           loader_func: Callable,
                           priority: int = 1,
                           cache_key: Optional[str] = None):
        """
        Register a data source with the optimizer.
        
        Args:
            name: Name of the data source
            loader_func: Function to load data from this source
            priority: Priority for processing (higher = more important)
            cache_key: Optional cache key for this source
        """
        self.data_sources[name] = {
            'loader': loader_func,
            'priority': priority,
            'cache_key': cache_key or f"source_{name}",
            'metadata': {}
        }
        self.source_priorities[name] = priority
        
        logger.info(f"Registered data source '{name}' with priority {priority}")
    
    def optimize_pipeline(self,
                         data_sources: Optional[Dict[str, Any]] = None,
                         target_variable: Optional[str] = None,
                         validation_data: Optional[pd.DataFrame] = None) -> OptimizationResult:
        """
        Optimize the entire data pipeline.
        
        Args:
            data_sources: Dictionary of data sources to process
            target_variable: Name of target variable for optimization
            validation_data: Validation dataset for performance testing
            
        Returns:
            Optimization result with optimized pipeline and metrics
        """
        logger.info("Starting pipeline optimization...")
        start_time = time.time()
        
        # Use registered sources if none provided
        if data_sources is None:
            data_sources = self.data_sources
        
        result = OptimizationResult()
        
        try:
            # Step 1: Analyze data sources
            source_analysis = self._analyze_data_sources(data_sources)
            result.data_quality_report = source_analysis
            
            # Step 2: Optimize data loading
            loading_plan = self._create_loading_plan(data_sources, source_analysis)
            
            # Step 3: Load and process data efficiently
            processed_data = self._execute_optimized_loading(loading_plan)
            
            # Step 4: Optimize feature processing
            optimized_features = self._optimize_feature_processing(
                processed_data, target_variable
            )
            
            # Step 5: Memory optimization
            optimized_data = self._optimize_memory_usage(optimized_features)
            
            # Step 6: Create optimized pipeline
            result.optimized_pipeline = self._create_optimized_pipeline(
                optimized_data, loading_plan
            )
            
            # Step 7: Validate pipeline performance
            if validation_data is not None:
                validation_results = self._validate_pipeline_performance(
                    result.optimized_pipeline, validation_data
                )
                result.performance_metrics.update(validation_results)
            
            # Step 8: Generate optimization report
            optimization_time = time.time() - start_time
            result.processing_time['total_optimization'] = optimization_time
            result.memory_usage = self._get_memory_usage_stats()
            result.recommendations = self._generate_optimization_recommendations(
                source_analysis, result.performance_metrics
            )
            
            logger.info(f"Pipeline optimization completed in {optimization_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline optimization failed: {e}")
            result.recommendations.append(f"Optimization failed: {str(e)}")
        
        return result
    
    def _analyze_data_sources(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data sources for optimization opportunities."""
        logger.info("Analyzing data sources...")
        
        analysis = {
            'sources': {},
            'total_estimated_memory': 0,
            'loading_complexity': {},
            'data_overlap': {},
            'quality_issues': []
        }
        
        # Analyze each source
        for source_name, source_config in data_sources.items():
            source_analysis = {
                'priority': source_config.get('priority', 1),
                'estimated_size': 0,
                'estimated_features': 0,
                'complexity_score': 1.0,
                'quality_score': 1.0
            }
            
            try:
                # Try to get metadata without full loading
                if 'metadata' in source_config:
                    metadata = source_config['metadata']
                    source_analysis['estimated_size'] = metadata.get('estimated_rows', 10000)
                    source_analysis['estimated_features'] = metadata.get('estimated_columns', 10)
                
                # Estimate memory usage
                estimated_memory = (source_analysis['estimated_size'] * 
                                  source_analysis['estimated_features'] * 8) / 1024**3  # GB
                source_analysis['estimated_memory_gb'] = estimated_memory
                analysis['total_estimated_memory'] += estimated_memory
                
                # Calculate complexity score based on size and features
                size_factor = min(source_analysis['estimated_size'] / 100000, 5.0)
                feature_factor = min(source_analysis['estimated_features'] / 100, 3.0)
                source_analysis['complexity_score'] = size_factor * feature_factor
                
            except Exception as e:
                logger.warning(f"Failed to analyze source {source_name}: {e}")
                analysis['quality_issues'].append(f"Analysis failed for {source_name}: {str(e)}")
            
            analysis['sources'][source_name] = source_analysis
        
        # Check for memory constraints
        available_memory = psutil.virtual_memory().available / 1024**3  # GB
        if analysis['total_estimated_memory'] > available_memory * 0.8:
            analysis['quality_issues'].append(
                f"Estimated memory usage ({analysis['total_estimated_memory']:.1f}GB) "
                f"exceeds 80% of available memory ({available_memory:.1f}GB)"
            )
        
        return analysis
    
    def _create_loading_plan(self,
                           data_sources: Dict[str, Any],
                           source_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized data loading plan."""
        logger.info("Creating optimized loading plan...")
        
        plan = {
            'load_order': [],
            'parallel_groups': [],
            'memory_strategy': 'chunk' if source_analysis['total_estimated_memory'] > self.config.max_memory_usage_gb else 'full',
            'chunking_config': {},
            'caching_strategy': {}
        }
        
        # Sort sources by priority and complexity
        sources_by_priority = sorted(
            source_analysis['sources'].items(),
            key=lambda x: (x[1]['priority'], -x[1]['complexity_score']),
            reverse=True
        )
        
        # Group sources for parallel loading
        current_group = []
        current_group_memory = 0
        max_group_memory = self.config.max_memory_usage_gb / 2  # Conservative estimate
        
        for source_name, source_info in sources_by_priority:
            source_memory = source_info['estimated_memory_gb']
            
            if (current_group_memory + source_memory <= max_group_memory and 
                len(current_group) < self.config.max_workers):
                current_group.append(source_name)
                current_group_memory += source_memory
            else:
                if current_group:
                    plan['parallel_groups'].append(current_group)
                current_group = [source_name]
                current_group_memory = source_memory
        
        if current_group:
            plan['parallel_groups'].append(current_group)
        
        # Set up chunking for large sources
        for source_name, source_info in source_analysis['sources'].items():
            if source_info['estimated_memory_gb'] > self.config.max_memory_usage_gb / 4:
                chunk_size = max(
                    self.config.chunk_size,
                    int(source_info['estimated_size'] / 10)  # Divide into 10 chunks
                )
                plan['chunking_config'][source_name] = {
                    'chunk_size': chunk_size,
                    'overlap_size': chunk_size // 10  # 10% overlap
                }
        
        # Set up caching strategy
        for source_name in data_sources.keys():
            if self.config.enable_feature_caching:
                plan['caching_strategy'][source_name] = {
                    'cache_raw': source_analysis['sources'][source_name]['estimated_memory_gb'] < 1.0,
                    'cache_processed': True,
                    'cache_features': True
                }
        
        plan['load_order'] = [source for group in plan['parallel_groups'] for source in group]
        
        logger.info(f"Created loading plan with {len(plan['parallel_groups'])} parallel groups")
        
        return plan
    
    def _execute_optimized_loading(self, loading_plan: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Execute optimized data loading according to plan."""
        logger.info("Executing optimized data loading...")
        
        loaded_data = {}
        loading_start = time.time()
        
        for group_idx, source_group in enumerate(loading_plan['parallel_groups']):
            logger.info(f"Loading group {group_idx + 1}/{len(loading_plan['parallel_groups'])}: {source_group}")
            
            # Load sources in parallel within group
            if len(source_group) > 1 and self.config.use_multiprocessing:
                group_data = self._load_sources_parallel(source_group, loading_plan)
            else:
                group_data = self._load_sources_sequential(source_group, loading_plan)
            
            loaded_data.update(group_data)
            
            # Memory cleanup between groups
            gc.collect()
        
        loading_time = time.time() - loading_start
        self.performance_stats['data_loading_time'].append(loading_time)
        
        logger.info(f"Data loading completed in {loading_time:.2f} seconds")
        
        return loaded_data
    
    def _load_sources_parallel(self,
                             source_group: List[str],
                             loading_plan: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load sources in parallel."""
        group_data = {}
        
        with ThreadPoolExecutor(max_workers=min(len(source_group), self.config.max_workers)) as executor:
            # Submit loading tasks
            future_to_source = {
                executor.submit(self._load_single_source, source_name, loading_plan): source_name
                for source_name in source_group
            }
            
            # Collect results
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    source_data = future.result()
                    if source_data is not None:
                        group_data[source_name] = source_data
                        logger.info(f"Loaded {source_name}: {len(source_data)} rows, {len(source_data.columns)} columns")
                except Exception as e:
                    logger.error(f"Failed to load {source_name}: {e}")
        
        return group_data
    
    def _load_sources_sequential(self,
                               source_group: List[str],
                               loading_plan: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load sources sequentially."""
        group_data = {}
        
        for source_name in source_group:
            try:
                source_data = self._load_single_source(source_name, loading_plan)
                if source_data is not None:
                    group_data[source_name] = source_data
                    logger.info(f"Loaded {source_name}: {len(source_data)} rows, {len(source_data.columns)} columns")
            except Exception as e:
                logger.error(f"Failed to load {source_name}: {e}")
        
        return group_data
    
    def _load_single_source(self,
                          source_name: str,
                          loading_plan: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load a single data source."""
        if source_name not in self.data_sources:
            logger.warning(f"Source {source_name} not registered")
            return None
        
        source_config = self.data_sources[source_name]
        cache_key = source_config['cache_key']
        
        # Check cache first
        if self.config.cache_intermediate_results and cache_key in self.result_cache:
            logger.info(f"Loading {source_name} from cache")
            return self.result_cache[cache_key]
        
        # Load data
        try:
            loader_func = source_config['loader']
            
            # Check if chunking is needed
            if source_name in loading_plan['chunking_config']:
                data = self._load_chunked_data(source_name, loader_func, loading_plan)
            else:
                data = loader_func()
            
            # Optimize data types
            if self.config.optimize_dtypes and data is not None:
                data = self._optimize_dtypes(data)
            
            # Cache result
            if self.config.cache_intermediate_results and data is not None:
                self.result_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load source {source_name}: {e}")
            return None
    
    def _load_chunked_data(self,
                          source_name: str,
                          loader_func: Callable,
                          loading_plan: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load data in chunks for memory efficiency."""
        chunk_config = loading_plan['chunking_config'][source_name]
        
        logger.info(f"Loading {source_name} in chunks of {chunk_config['chunk_size']}")
        
        chunks = []
        try:
            # This is a simplified chunking - real implementation would depend on the data source
            # For now, we'll load full data and then chunk it
            full_data = loader_func()
            
            if full_data is None or len(full_data) == 0:
                return None
            
            chunk_size = chunk_config['chunk_size']
            overlap_size = chunk_config['overlap_size']
            
            for start_idx in range(0, len(full_data), chunk_size - overlap_size):
                end_idx = min(start_idx + chunk_size, len(full_data))
                chunk = full_data.iloc[start_idx:end_idx].copy()
                
                # Process chunk
                processed_chunk = self._process_data_chunk(chunk, source_name)
                chunks.append(processed_chunk)
            
            # Combine chunks
            combined_data = pd.concat(chunks, ignore_index=True)
            
            # Remove duplicates from overlaps
            if overlap_size > 0:
                combined_data = combined_data.drop_duplicates()
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Chunked loading failed for {source_name}: {e}")
            return None
    
    def _process_data_chunk(self, chunk: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Process a single data chunk."""
        # Basic preprocessing
        processed_chunk = chunk.copy()
        
        # Handle missing values
        if self.config.missing_data_strategy == 'forward_fill':
            processed_chunk = processed_chunk.fillna(method='ffill')
        elif self.config.missing_data_strategy == 'drop':
            processed_chunk = processed_chunk.dropna()
        
        # Handle outliers if enabled
        if self.config.outlier_handling:
            processed_chunk = self._handle_outliers_chunk(processed_chunk)
        
        return processed_chunk
    
    def _handle_outliers_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in a data chunk."""
        numeric_columns = chunk.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in chunk.columns:
                Q1 = chunk[col].quantile(0.25)
                Q3 = chunk[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 2.0 * IQR
                upper_bound = Q3 + 2.0 * IQR
                
                # Cap outliers instead of removing them
                chunk[col] = chunk[col].clip(lower=lower_bound, upper=upper_bound)
        
        return chunk
    
    def _optimize_feature_processing(self,
                                   processed_data: Dict[str, pd.DataFrame],
                                   target_variable: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Optimize feature processing across all data sources."""
        logger.info("Optimizing feature processing...")
        
        optimized_data = {}
        processing_start = time.time()
        
        for source_name, data in processed_data.items():
            if data is None or data.empty:
                continue
            
            # Limit features per source
            if len(data.columns) > self.config.max_features_per_source:
                optimized_features = self._select_top_features(
                    data, self.config.max_features_per_source, target_variable
                )
            else:
                optimized_features = data
            
            # Remove highly correlated features
            if self.config.feature_selection_threshold < 1.0:
                optimized_features = self._remove_correlated_features(
                    optimized_features, self.config.feature_selection_threshold
                )
            
            optimized_data[source_name] = optimized_features
            
            reduction_ratio = 1 - (len(optimized_features.columns) / len(data.columns))
            self.performance_stats['feature_reduction_ratio'].append(reduction_ratio)
            
            logger.info(f"Optimized {source_name}: {len(data.columns)} -> {len(optimized_features.columns)} features "
                       f"({reduction_ratio:.1%} reduction)")
        
        processing_time = time.time() - processing_start
        self.performance_stats['processing_time'].append(processing_time)
        
        return optimized_data
    
    def _select_top_features(self,
                           data: pd.DataFrame,
                           max_features: int,
                           target_variable: Optional[str] = None) -> pd.DataFrame:
        """Select top features using various criteria."""
        if len(data.columns) <= max_features:
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return data.iloc[:, :max_features]
        
        # Calculate feature importance scores
        feature_scores = {}
        
        # Variance-based scoring
        for col in numeric_columns:
            variance_score = data[col].var()
            feature_scores[col] = variance_score
        
        # Select top features by score
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_columns = [col for col, _ in top_features]
        
        # Add non-numeric columns if there's space
        non_numeric = [col for col in data.columns if col not in numeric_columns]
        remaining_slots = max_features - len(selected_columns)
        selected_columns.extend(non_numeric[:remaining_slots])
        
        return data[selected_columns]
    
    def _remove_correlated_features(self,
                                  data: pd.DataFrame,
                                  correlation_threshold: float) -> pd.DataFrame:
        """Remove highly correlated features."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return data
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identify features to remove
        to_remove = set()
        for col in upper_tri.columns:
            correlated_features = upper_tri.index[upper_tri[col] > correlation_threshold].tolist()
            to_remove.update(correlated_features)
        
        # Keep non-numeric columns and non-correlated numeric columns
        columns_to_keep = [col for col in data.columns if col not in to_remove]
        
        return data[columns_to_keep]
    
    def _optimize_memory_usage(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Optimize memory usage of processed data."""
        logger.info("Optimizing memory usage...")
        
        optimized_data = {}
        memory_before = sum(df.memory_usage(deep=True).sum() for df in data_dict.values() if df is not None)
        
        for source_name, data in data_dict.items():
            if data is None:
                continue
            
            optimized_df = self._optimize_dtypes(data)
            optimized_data[source_name] = optimized_df
        
        memory_after = sum(df.memory_usage(deep=True).sum() for df in optimized_data.values())
        memory_reduction = (memory_before - memory_after) / memory_before if memory_before > 0 else 0
        
        logger.info(f"Memory optimization: {memory_before / 1024**2:.1f}MB -> {memory_after / 1024**2:.1f}MB "
                   f"({memory_reduction:.1%} reduction)")
        
        return optimized_data
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type == 'object':
                # Try to convert to categorical if beneficial
                num_unique = optimized_df[col].nunique()
                if num_unique < len(optimized_df) * 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
            
            elif 'int' in str(col_type):
                # Downcast integers
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            
            elif 'float' in str(col_type):
                # Downcast floats
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df
    
    def _create_optimized_pipeline(self,
                                 optimized_data: Dict[str, pd.DataFrame],
                                 loading_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create the optimized pipeline object."""
        pipeline = {
            'data': optimized_data,
            'loading_plan': loading_plan,
            'config': self.config,
            'metadata': {
                'sources': list(optimized_data.keys()),
                'total_samples': sum(len(df) for df in optimized_data.values() if df is not None),
                'total_features': sum(len(df.columns) for df in optimized_data.values() if df is not None),
                'memory_usage_mb': sum(df.memory_usage(deep=True).sum() for df in optimized_data.values() if df is not None) / 1024**2,
                'optimization_timestamp': datetime.now().isoformat()
            }
        }
        
        return pipeline
    
    def _validate_pipeline_performance(self,
                                     pipeline: Dict[str, Any],
                                     validation_data: pd.DataFrame) -> Dict[str, float]:
        """Validate pipeline performance."""
        logger.info("Validating pipeline performance...")
        
        metrics = {}
        
        try:
            # Test loading speed
            load_start = time.time()
            # Simulate reloading pipeline data
            for source_name in pipeline['metadata']['sources']:
                if source_name in pipeline['data']:
                    _ = pipeline['data'][source_name].head()
            load_time = time.time() - load_start
            metrics['load_time_seconds'] = load_time
            
            # Test memory efficiency
            current_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            metrics['memory_usage_mb'] = current_memory
            
            # Test data quality
            total_missing = 0
            total_cells = 0
            
            for df in pipeline['data'].values():
                if df is not None:
                    total_missing += df.isnull().sum().sum()
                    total_cells += df.size
            
            metrics['data_completeness'] = 1 - (total_missing / total_cells) if total_cells > 0 else 0
            
            # Test feature efficiency
            metrics['features_per_mb'] = (pipeline['metadata']['total_features'] / 
                                        pipeline['metadata']['memory_usage_mb'])
            
        except Exception as e:
            logger.warning(f"Pipeline validation failed: {e}")
            metrics['validation_error'] = str(e)
        
        return metrics
    
    def _get_memory_usage_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'system_memory_total_gb': memory_info.total / 1024**3,
            'system_memory_available_gb': memory_info.available / 1024**3,
            'system_memory_percent': memory_info.percent,
            'process_memory_mb': process.memory_info().rss / 1024**2,
            'process_memory_percent': process.memory_percent()
        }
    
    def _generate_optimization_recommendations(self,
                                            source_analysis: Dict[str, Any],
                                            performance_metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Memory recommendations
        if source_analysis['total_estimated_memory'] > self.config.max_memory_usage_gb:
            recommendations.append(
                f"Consider increasing chunk size or reducing max_features_per_source. "
                f"Estimated memory usage: {source_analysis['total_estimated_memory']:.1f}GB"
            )
        
        # Performance recommendations
        avg_load_time = np.mean(self.performance_stats['data_loading_time'])
        if avg_load_time > 30:  # seconds
            recommendations.append(
                f"Data loading is slow ({avg_load_time:.1f}s). Consider enabling chunking or reducing data size."
            )
        
        # Data quality recommendations
        if performance_metrics.get('data_completeness', 1.0) < 0.8:
            recommendations.append(
                f"Data completeness is low ({performance_metrics['data_completeness']:.1%}). "
                f"Consider improving missing data handling."
            )
        
        # Feature efficiency recommendations
        avg_reduction = np.mean(self.performance_stats['feature_reduction_ratio'])
        if avg_reduction < 0.1:
            recommendations.append(
                "Low feature reduction achieved. Consider stricter correlation thresholds or feature selection."
            )
        
        # Quality issues from analysis
        for issue in source_analysis['quality_issues']:
            recommendations.append(f"Data quality issue: {issue}")
        
        if not recommendations:
            recommendations.append("Pipeline is well-optimized. No major issues detected.")
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'performance_stats': self.performance_stats,
            'memory_stats': self._get_memory_usage_stats(),
            'cache_stats': {
                'cached_items': len(self.result_cache),
                'cache_memory_estimate_mb': sum(
                    df.memory_usage(deep=True).sum() / 1024**2 
                    for df in self.result_cache.values() 
                    if isinstance(df, pd.DataFrame)
                )
            },
            'config': self.config
        }
    
    def clear_cache(self):
        """Clear the result cache to free memory."""
        self.result_cache.clear()
        gc.collect()
        logger.info("Pipeline cache cleared")