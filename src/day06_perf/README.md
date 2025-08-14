# Day 6: Performance Engineering and Production Reliability

## Learning Objectives

This module covers advanced performance engineering and reliability practices essential for enterprise geospatial systems operating at scale. By the end of this module, you will understand:

- **Performance measurement and profiling** techniques for identifying bottlenecks in geospatial applications
- **Memory optimization strategies** for processing large-scale spatial datasets efficiently
- **Concurrency and parallelism patterns** optimized for spatial operations and I/O-bound workloads  
- **Caching architectures** with spatial awareness and geographic distribution strategies
- **Production observability** including metrics, logging, tracing, and alerting for geospatial services
- **Reliability engineering patterns** including circuit breakers, bulkheads, and graceful degradation
- **Capacity planning and auto-scaling** for geospatial workloads with variable spatial distributions
- **Database performance optimization** for spatial queries and large geographic datasets

## Theoretical Foundation

### Performance Characteristics of Geospatial Systems

**Spatial Data Access Patterns:**
Geospatial applications exhibit unique performance characteristics:
- **Spatial locality**: Queries often cluster around geographic regions
- **Multi-scale access**: Same data accessed at different zoom levels and resolutions
- **Temporal patterns**: Usage follows human activity patterns (diurnal, seasonal)
- **Hotspot phenomena**: Popular locations create uneven load distribution

**Performance Bottlenecks in Spatial Systems:**
- **I/O bound operations**: Database queries, file system access, network tile fetching
- **CPU bound operations**: Coordinate transformations, geometric computations, spatial indexing
- **Memory bound operations**: Large dataset processing, spatial joins, buffer operations
- **Network bound operations**: Tile serving, real-time location updates, distributed queries

**Scaling Challenges:**
- **Geographic load balancing**: Distributing work based on spatial boundaries
- **Cache efficiency**: Balancing hit rates with storage costs across geographic regions
- **Data locality**: Minimizing network overhead for spatially-related data
- **Query optimization**: Spatial query plans with varying selectivity ratios

### Site Reliability Engineering for Geospatial Services

**Availability Requirements:**
- **Consumer applications**: 99.9% availability (8.77 hours/year downtime)
- **Enterprise systems**: 99.95% availability (4.38 hours/year downtime) 
- **Safety-critical systems**: 99.99% availability (52.6 minutes/year downtime)
- **Global mapping services**: 99.999% availability (5.26 minutes/year downtime)

**Error Budgets and SLOs:**
- **Latency SLOs**: p95 < 100ms for interactive mapping, p99 < 500ms for routing
- **Throughput SLOs**: 10,000+ QPS for tile serving, 1,000+ QPS for geocoding
- **Error rate SLOs**: < 0.1% for critical path operations, < 1% for non-critical features

**Reliability Patterns:**
- **Circuit breakers**: Prevent cascade failures between geospatial services
- **Bulkhead isolation**: Separate resource pools for different geographic regions
- **Graceful degradation**: Fallback to cached or simplified data during outages
- **Chaos engineering**: Systematic testing of failure scenarios in spatial systems

## Performance Measurement and Profiling

### Advanced Profiling Techniques

```python
import cProfile
import pstats
import tracemalloc
import psutil
import time
from typing import Dict, List, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for geospatial operations."""
    duration_seconds: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    spatial_operations_count: int
    cache_hit_rate: float

class GeospatialProfiler:
    """Advanced profiler for geospatial operations."""
    
    def __init__(self):
        self.metrics_history = []
        self.active_profiles = {}
        self.process = psutil.Process()
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile a geospatial operation with comprehensive metrics."""
        # Start resource monitoring
        start_time = time.time()
        tracemalloc.start()
        
        # Baseline measurements
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        baseline_io = self.process.io_counters()
        baseline_net = psutil.net_io_counters()
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            yield self
        finally:
            # Stop profiling
            profiler.disable()
            
            # Collect final measurements
            end_time = time.time()
            final_memory = self.process.memory_info().rss / 1024 / 1024
            final_io = self.process.io_counters()
            final_net = psutil.net_io_counters()
            
            # Memory profiling
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            metrics = PerformanceMetrics(
                duration_seconds=end_time - start_time,
                memory_peak_mb=peak_mem / 1024 / 1024,
                memory_current_mb=current_mem / 1024 / 1024,
                cpu_percent=self.process.cpu_percent(),
                disk_io_read_mb=(final_io.read_bytes - baseline_io.read_bytes) / 1024 / 1024,
                disk_io_write_mb=(final_io.write_bytes - baseline_io.write_bytes) / 1024 / 1024,
                network_bytes_sent=final_net.bytes_sent - baseline_net.bytes_sent,
                network_bytes_recv=final_net.bytes_recv - baseline_net.bytes_recv,
                spatial_operations_count=getattr(self, '_spatial_ops_count', 0),
                cache_hit_rate=getattr(self, '_cache_hit_rate', 0.0)
            )
            
            # Store profile results
            self.active_profiles[operation_name] = {
                'metrics': metrics,
                'profiler': profiler,
                'timestamp': end_time
            }
            
            self.metrics_history.append((operation_name, metrics))
    
    def get_hotspots(self, operation_name: str, top_n: int = 10) -> List[Dict]:
        """Get performance hotspots for an operation."""
        if operation_name not in self.active_profiles:
            return []
        
        profiler = self.active_profiles[operation_name]['profiler']
        stats = pstats.Stats(profiler)
        
        # Sort by cumulative time
        stats.sort_stats('cumulative')
        
        # Extract top functions
        hotspots = []
        for func_info in stats.get_stats_profile().func_profiles.values():
            if len(hotspots) >= top_n:
                break
                
            hotspots.append({
                'function': func_info.func_name,
                'filename': func_info.filename,
                'line_number': func_info.line_number,
                'cumulative_time': func_info.cumulative_time,
                'internal_time': func_info.internal_time,
                'call_count': func_info.call_count
            })
        
        return hotspots
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {}
        
        # Aggregate metrics
        total_operations = len(self.metrics_history)
        avg_duration = sum(m.duration_seconds for _, m in self.metrics_history) / total_operations
        max_memory = max(m.memory_peak_mb for _, m in self.metrics_history)
        total_disk_io = sum(m.disk_io_read_mb + m.disk_io_write_mb for _, m in self.metrics_history)
        
        # Identify bottlenecks
        slow_operations = [
            (name, metrics) for name, metrics in self.metrics_history 
            if metrics.duration_seconds > avg_duration * 2
        ]
        
        memory_intensive = [
            (name, metrics) for name, metrics in self.metrics_history
            if metrics.memory_peak_mb > max_memory * 0.8
        ]
        
        return {
            'summary': {
                'total_operations': total_operations,
                'average_duration_seconds': avg_duration,
                'max_memory_mb': max_memory,
                'total_disk_io_mb': total_disk_io,
                'overall_cache_hit_rate': sum(m.cache_hit_rate for _, m in self.metrics_history) / total_operations
            },
            'bottlenecks': {
                'slow_operations': [(name, m.duration_seconds) for name, m in slow_operations],
                'memory_intensive': [(name, m.memory_peak_mb) for name, m in memory_intensive]
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        avg_memory = sum(m.memory_peak_mb for _, m in self.metrics_history) / len(self.metrics_history)
        avg_cache_hit = sum(m.cache_hit_rate for _, m in self.metrics_history) / len(self.metrics_history)
        
        if avg_memory > 1000:  # > 1GB average
            recommendations.append("Consider implementing streaming/chunked processing for large datasets")
        
        if avg_cache_hit < 0.7:  # < 70% cache hit rate
            recommendations.append("Optimize caching strategy - consider larger cache size or better eviction policies")
        
        high_io_ops = [m for _, m in self.metrics_history if m.disk_io_read_mb + m.disk_io_write_mb > 100]
        if len(high_io_ops) > len(self.metrics_history) * 0.3:
            recommendations.append("High disk I/O detected - consider SSD storage or better I/O optimization")
        
        return recommendations

def performance_monitor(operation_type: str = "spatial_operation"):
    """Decorator for monitoring performance of spatial operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = GeospatialProfiler()
            with profiler.profile_operation(f"{operation_type}_{func.__name__}"):
                result = await func(*args, **kwargs)
                
                # Log performance metrics
                metrics = profiler.metrics_history[-1][1]
                logger.info(f"Performance metrics for {func.__name__}", extra={
                    'duration_seconds': metrics.duration_seconds,
                    'memory_peak_mb': metrics.memory_peak_mb,
                    'operation_type': operation_type
                })
                
                return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            profiler = GeospatialProfiler()
            with profiler.profile_operation(f"{operation_type}_{func.__name__}"):
                result = func(*args, **kwargs)
                
                # Log performance metrics
                metrics = profiler.metrics_history[-1][1]
                logger.info(f"Performance metrics for {func.__name__}", extra={
                    'duration_seconds': metrics.duration_seconds,
                    'memory_peak_mb': metrics.memory_peak_mb,
                    'operation_type': operation_type
                })
                
                return result
        
        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
```

### Memory Optimization Strategies

```python
import gc
import weakref
from typing import Optional, Dict, Any, Iterator
from collections import OrderedDict
import numpy as np
from shapely.geometry import Geometry
import gzip
import pickle

class SpatialMemoryManager:
    """Advanced memory management for geospatial operations."""
    
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.geometry_cache = SpatialLRUCache(max_size_mb=1000)
        self.compression_ratio = 0.3  # Estimated compression ratio
        self.weak_refs = weakref.WeakValueDictionary()
    
    @contextmanager
    def memory_limit_context(self, operation_name: str):
        """Context manager for memory-limited operations."""
        initial_memory = self._get_memory_usage()
        
        try:
            yield self
        finally:
            current_memory = self._get_memory_usage()
            memory_delta = current_memory - initial_memory
            
            if memory_delta > self.max_memory_bytes * 0.8:
                logger.warning(f"High memory usage in {operation_name}: {memory_delta / 1024 / 1024:.1f} MB")
                self._aggressive_cleanup()
    
    def optimize_geometry_collection(self, geometries: List[Geometry]) -> 'OptimizedGeometryCollection':
        """Optimize a collection of geometries for memory efficiency."""
        # Analyze geometry characteristics
        total_coords = sum(len(geom.coords) if hasattr(geom, 'coords') else 0 for geom in geometries)
        avg_complexity = total_coords / len(geometries) if geometries else 0
        
        if avg_complexity > 1000:  # High complexity geometries
            return self._create_compressed_collection(geometries)
        elif len(geometries) > 10000:  # Many simple geometries
            return self._create_indexed_collection(geometries)
        else:
            return self._create_standard_collection(geometries)
    
    def _create_compressed_collection(self, geometries: List[Geometry]) -> 'CompressedGeometryCollection':
        """Create memory-efficient compressed geometry collection."""
        compressed_data = []
        
        for geom in geometries:
            # Serialize and compress geometry
            wkb_data = geom.wkb
            compressed = gzip.compress(wkb_data)
            compressed_data.append(compressed)
        
        return CompressedGeometryCollection(compressed_data, self.compression_ratio)
    
    def _create_indexed_collection(self, geometries: List[Geometry]) -> 'IndexedGeometryCollection':
        """Create spatially-indexed geometry collection."""
        # Build spatial index with minimal memory footprint
        spatial_index = rtree.index.Index()
        geometry_store = {}
        
        for i, geom in enumerate(geometries):
            # Store only bounding box in index
            spatial_index.insert(i, geom.bounds)
            
            # Store geometry with weak reference
            geometry_store[i] = geom
            self.weak_refs[f"geom_{i}"] = geom
        
        return IndexedGeometryCollection(spatial_index, geometry_store)
    
    def stream_large_dataset(self, dataset_path: str, 
                           chunk_size: int = 10000) -> Iterator[List[Geometry]]:
        """Stream large datasets with memory management."""
        with open(dataset_path, 'rb') as f:
            while True:
                with self.memory_limit_context(f"stream_chunk"):
                    chunk = self._read_chunk(f, chunk_size)
                    if not chunk:
                        break
                    
                    # Process chunk
                    processed_chunk = self._process_geometry_chunk(chunk)
                    yield processed_chunk
                    
                    # Explicit cleanup
                    del chunk, processed_chunk
                    gc.collect()
    
    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        # Clear caches
        self.geometry_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear weak references
        self.weak_refs.clear()
        
        # Log memory recovery
        current_memory = self._get_memory_usage()
        logger.info(f"Memory cleanup completed. Current usage: {current_memory / 1024 / 1024:.1f} MB")

class SpatialLRUCache:
    """LRU cache optimized for spatial data."""
    
    def __init__(self, max_size_mb: int = 1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.cache = OrderedDict()
        self.size_tracker = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache and move to end (most recently used)."""
        if key in self.cache:
            # Move to end
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    def put(self, key: str, value: Any, estimated_size: int = None):
        """Put item in cache with size tracking."""
        if estimated_size is None:
            estimated_size = self._estimate_size(value)
        
        # Remove existing item if present
        if key in self.cache:
            self.current_size -= self.size_tracker[key]
            del self.cache[key]
            del self.size_tracker[key]
        
        # Evict items if necessary
        while self.current_size + estimated_size > self.max_size_bytes and self.cache:
            self._evict_lru()
        
        # Add new item
        self.cache[key] = value
        self.size_tracker[key] = estimated_size
        self.current_size += estimated_size
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.cache:
            key, _ = self.cache.popitem(last=False)  # FIFO - least recently used
            size = self.size_tracker.pop(key)
            self.current_size -= size
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value[:10])  # Sample first 10
            else:
                return 1024  # Conservative estimate
```

### Concurrency and Parallelism Optimization

```python
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, Any, Awaitable
import numpy as np
from functools import partial

class SpatialConcurrencyManager:
    """Optimized concurrency management for spatial operations."""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.io_thread_pool = ThreadPoolExecutor(max_workers=min(32, self.cpu_count * 4))
        self.cpu_process_pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        self.semaphores = {
            'disk_io': asyncio.Semaphore(4),  # Limit concurrent disk operations
            'network_io': asyncio.Semaphore(20),  # Network operations
            'cpu_intensive': asyncio.Semaphore(self.cpu_count),  # CPU-bound operations
            'memory_intensive': asyncio.Semaphore(2)  # Memory-intensive operations
        }
    
    async def parallel_spatial_operation(self, 
                                       geometries: List[Geometry], 
                                       operation: Callable,
                                       chunk_size: int = None,
                                       operation_type: str = 'cpu_intensive') -> List[Any]:
        """Execute spatial operations in parallel with optimal batching."""
        
        if chunk_size is None:
            chunk_size = max(1, len(geometries) // (self.cpu_count * 2))
        
        # Split geometries into chunks
        chunks = [geometries[i:i + chunk_size] for i in range(0, len(geometries), chunk_size)]
        
        # Choose execution strategy based on operation type
        if operation_type == 'cpu_intensive':
            return await self._cpu_parallel_execution(chunks, operation)
        elif operation_type == 'io_bound':
            return await self._io_parallel_execution(chunks, operation)
        else:
            return await self._mixed_parallel_execution(chunks, operation)
    
    async def _cpu_parallel_execution(self, chunks: List[List[Geometry]], 
                                    operation: Callable) -> List[Any]:
        """Execute CPU-intensive operations using process pool."""
        loop = asyncio.get_event_loop()
        
        # Prepare serializable operation
        if hasattr(operation, '__self__'):
            # Handle bound methods by converting to function
            operation = partial(operation.__func__, operation.__self__)
        
        tasks = []
        for chunk in chunks:
            async with self.semaphores['cpu_intensive']:
                task = loop.run_in_executor(
                    self.cpu_process_pool,
                    self._process_geometry_chunk,
                    chunk, operation
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        flattened_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"CPU operation failed: {result}")
                continue
            flattened_results.extend(result)
        
        return flattened_results
    
    async def _io_parallel_execution(self, chunks: List[List[Geometry]], 
                                   operation: Callable) -> List[Any]:
        """Execute I/O-bound operations using thread pool."""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for chunk in chunks:
            async with self.semaphores['network_io']:
                task = loop.run_in_executor(
                    self.io_thread_pool,
                    self._process_geometry_chunk,
                    chunk, operation
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and filter results
        flattened_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"I/O operation failed: {result}")
                continue
            flattened_results.extend(result)
        
        return flattened_results
    
    def _process_geometry_chunk(self, chunk: List[Geometry], operation: Callable) -> List[Any]:
        """Process a chunk of geometries with the given operation."""
        try:
            return [operation(geom) for geom in chunk]
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return []
    
    async def adaptive_batch_processing(self, 
                                      data_stream: AsyncIterator[Geometry],
                                      processor: Callable,
                                      target_latency_ms: float = 100) -> AsyncIterator[Any]:
        """Adaptive batch processing that adjusts batch size based on performance."""
        
        batch_size = 10  # Initial batch size
        batch = []
        processing_times = []
        
        async for item in data_stream:
            batch.append(item)
            
            if len(batch) >= batch_size:
                # Process batch and measure time
                start_time = time.time()
                
                async with self.semaphores['cpu_intensive']:
                    results = await self.parallel_spatial_operation(
                        batch, processor, chunk_size=batch_size//2
                    )
                
                processing_time = (time.time() - start_time) * 1000  # ms
                processing_times.append(processing_time)
                
                # Yield results
                for result in results:
                    yield result
                
                # Adapt batch size based on performance
                batch_size = self._adapt_batch_size(
                    batch_size, processing_time, target_latency_ms, processing_times
                )
                
                batch = []
        
        # Process remaining items
        if batch:
            async with self.semaphores['cpu_intensive']:
                results = await self.parallel_spatial_operation(batch, processor)
                for result in results:
                    yield result
    
    def _adapt_batch_size(self, current_size: int, last_time: float, 
                         target_time: float, history: List[float]) -> int:
        """Adapt batch size based on performance metrics."""
        if len(history) < 3:
            return current_size
        
        avg_time = sum(history[-3:]) / 3
        
        if avg_time > target_time * 1.2:  # Too slow
            return max(1, int(current_size * 0.8))
        elif avg_time < target_time * 0.5:  # Too fast, can increase
            return min(1000, int(current_size * 1.2))
        else:
            return current_size  # Good performance, keep current size

class SpatialWorkloadBalancer:
    """Load balancer for spatial workloads across multiple workers."""
    
    def __init__(self, worker_count: int = None):
        self.worker_count = worker_count or mp.cpu_count()
        self.workers = []
        self.workload_stats = {}
        
    async def distribute_spatial_workload(self, 
                                        spatial_tasks: List[Dict],
                                        distribution_strategy: str = 'geographic') -> List[Any]:
        """Distribute spatial workload across workers."""
        
        if distribution_strategy == 'geographic':
            task_groups = self._group_by_geography(spatial_tasks)
        elif distribution_strategy == 'complexity':
            task_groups = self._group_by_complexity(spatial_tasks)
        else:
            task_groups = self._round_robin_distribution(spatial_tasks)
        
        # Execute task groups in parallel
        tasks = []
        for i, task_group in enumerate(task_groups):
            worker_id = i % self.worker_count
            task = self._execute_on_worker(worker_id, task_group)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect and flatten results
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Worker task failed: {result}")
                continue
            all_results.extend(result)
        
        return all_results
    
    def _group_by_geography(self, tasks: List[Dict]) -> List[List[Dict]]:
        """Group tasks by geographic proximity."""
        # Implementation would use spatial clustering
        # For now, simple bbox-based grouping
        groups = [[] for _ in range(self.worker_count)]
        
        for i, task in enumerate(tasks):
            if 'bbox' in task:
                # Simple hash-based assignment based on bbox center
                center_x = (task['bbox'][0] + task['bbox'][2]) / 2
                center_y = (task['bbox'][1] + task['bbox'][3]) / 2
                group_id = hash((int(center_x * 1000), int(center_y * 1000))) % self.worker_count
                groups[group_id].append(task)
            else:
                groups[i % self.worker_count].append(task)
        
        return groups
```

### Advanced Caching Strategies

```python
import redis
import json
import hashlib
from typing import Optional, Dict, Any, Union
import geojson
from shapely.geometry import shape, box

class SpatialCacheManager:
    """Multi-tier spatial caching with geographic awareness."""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.local_cache = SpatialLRUCache(max_size_mb=500)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_pressure_events': 0
        }
    
    async def get_spatial_data(self, bbox: Tuple[float, float, float, float],
                              zoom_level: int, layer: str) -> Optional[Dict]:
        """Get spatial data with multi-tier caching."""
        cache_key = self._generate_spatial_key(bbox, zoom_level, layer)
        
        # L1: Check local memory cache
        local_result = self.local_cache.get(cache_key)
        if local_result:
            self.cache_stats['hits'] += 1
            return local_result
        
        # L2: Check Redis cache
        redis_result = await self._get_from_redis(cache_key)
        if redis_result:
            # Store in local cache for faster future access
            self.local_cache.put(cache_key, redis_result)
            self.cache_stats['hits'] += 1
            return redis_result
        
        self.cache_stats['misses'] += 1
        return None
    
    async def set_spatial_data(self, bbox: Tuple[float, float, float, float],
                              zoom_level: int, layer: str, data: Dict,
                              ttl_seconds: int = 3600):
        """Set spatial data in multi-tier cache."""
        cache_key = self._generate_spatial_key(bbox, zoom_level, layer)
        
        # Optimize data for caching
        optimized_data = self._optimize_for_cache(data, zoom_level)
        
        # Store in both cache tiers
        self.local_cache.put(cache_key, optimized_data)
        await self._set_in_redis(cache_key, optimized_data, ttl_seconds)
    
    def _generate_spatial_key(self, bbox: Tuple[float, float, float, float],
                             zoom_level: int, layer: str) -> str:
        """Generate cache key for spatial data."""
        # Normalize bbox to reduce cache fragmentation
        normalized_bbox = self._normalize_bbox(bbox, zoom_level)
        
        key_data = {
            'bbox': normalized_bbox,
            'zoom': zoom_level,
            'layer': layer
        }
        
        # Create hash for consistent key generation
        key_string = json.dumps(key_data, sort_keys=True)
        return f"spatial:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _normalize_bbox(self, bbox: Tuple[float, float, float, float],
                       zoom_level: int) -> Tuple[float, float, float, float]:
        """Normalize bbox to grid boundaries for better cache hit rates."""
        # Grid size based on zoom level
        grid_size = 1.0 / (2 ** max(0, zoom_level - 10))
        
        # Snap to grid
        min_x = math.floor(bbox[0] / grid_size) * grid_size
        min_y = math.floor(bbox[1] / grid_size) * grid_size
        max_x = math.ceil(bbox[2] / grid_size) * grid_size
        max_y = math.ceil(bbox[3] / grid_size) * grid_size
        
        return (min_x, min_y, max_x, max_y)
    
    def _optimize_for_cache(self, data: Dict, zoom_level: int) -> Dict:
        """Optimize spatial data for caching based on zoom level."""
        if 'features' not in data:
            return data
        
        optimized_features = []
        simplification_tolerance = self._get_simplification_tolerance(zoom_level)
        
        for feature in data['features']:
            if 'geometry' in feature:
                # Simplify geometry based on zoom level
                geom = shape(feature['geometry'])
                simplified = geom.simplify(simplification_tolerance, preserve_topology=True)
                
                optimized_feature = {
                    'type': 'Feature',
                    'geometry': simplified.__geo_interface__,
                    'properties': self._filter_properties(feature.get('properties', {}), zoom_level)
                }
                optimized_features.append(optimized_feature)
        
        return {
            'type': 'FeatureCollection',
            'features': optimized_features
        }
    
    async def invalidate_spatial_region(self, bbox: Tuple[float, float, float, float],
                                       layer: str = None):
        """Invalidate cache for a spatial region."""
        # Find all cache keys that intersect with the bbox
        pattern = f"spatial:*"
        if layer:
            pattern = f"spatial:*{layer}*"
        
        # This is a simplified approach - production systems would use
        # spatial indexing of cache keys for efficient invalidation
        keys_to_invalidate = []
        async for key in self.redis_client.scan_iter(pattern):
            if await self._key_intersects_bbox(key, bbox):
                keys_to_invalidate.append(key)
        
        # Remove from both cache levels
        if keys_to_invalidate:
            await self.redis_client.delete(*keys_to_invalidate)
            for key in keys_to_invalidate:
                if key in self.local_cache.cache:
                    del self.local_cache.cache[key]

class SpatialPreloadManager:
    """Intelligent preloading of spatial data based on usage patterns."""
    
    def __init__(self, cache_manager: SpatialCacheManager):
        self.cache_manager = cache_manager
        self.usage_patterns = {}
        self.preload_queue = asyncio.Queue()
        self.preload_worker_running = False
    
    async def track_access_pattern(self, bbox: Tuple[float, float, float, float],
                                  zoom_level: int, layer: str):
        """Track spatial data access patterns for intelligent preloading."""
        pattern_key = f"{layer}:{zoom_level}"
        
        if pattern_key not in self.usage_patterns:
            self.usage_patterns[pattern_key] = {
                'access_count': 0,
                'bboxes': [],
                'last_access': time.time()
            }
        
        pattern = self.usage_patterns[pattern_key]
        pattern['access_count'] += 1
        pattern['bboxes'].append(bbox)
        pattern['last_access'] = time.time()
        
        # Keep only recent bboxes
        if len(pattern['bboxes']) > 1000:
            pattern['bboxes'] = pattern['bboxes'][-500:]
        
        # Trigger preloading for hot patterns
        if pattern['access_count'] % 10 == 0:
            await self._schedule_preload(pattern_key)
    
    async def _schedule_preload(self, pattern_key: str):
        """Schedule preloading based on access patterns."""
        if not self.preload_worker_running:
            asyncio.create_task(self._preload_worker())
            self.preload_worker_running = True
        
        pattern = self.usage_patterns[pattern_key]
        
        # Predict next likely access areas
        predicted_areas = self._predict_access_areas(pattern['bboxes'])
        
        for bbox, confidence in predicted_areas:
            if confidence > 0.7:  # High confidence predictions
                layer, zoom_str = pattern_key.split(':')
                zoom_level = int(zoom_str)
                
                await self.preload_queue.put({
                    'bbox': bbox,
                    'zoom_level': zoom_level,
                    'layer': layer,
                    'priority': confidence
                })
    
    def _predict_access_areas(self, recent_bboxes: List[Tuple]) -> List[Tuple[Tuple, float]]:
        """Predict likely next access areas based on spatial patterns."""
        if len(recent_bboxes) < 5:
            return []
        
        predictions = []
        
        # Simple spatial trend analysis
        recent = recent_bboxes[-10:]  # Last 10 accesses
        
        # Calculate movement vector
        if len(recent) >= 2:
            start_center = self._bbox_center(recent[0])
            end_center = self._bbox_center(recent[-1])
            
            dx = end_center[0] - start_center[0]
            dy = end_center[1] - start_center[1]
            
            # Predict next area based on movement
            if abs(dx) > 0.001 or abs(dy) > 0.001:  # Significant movement
                last_bbox = recent[-1]
                predicted_center = (
                    self._bbox_center(last_bbox)[0] + dx,
                    self._bbox_center(last_bbox)[1] + dy
                )
                
                bbox_size = (
                    last_bbox[2] - last_bbox[0],
                    last_bbox[3] - last_bbox[1]
                )
                
                predicted_bbox = (
                    predicted_center[0] - bbox_size[0]/2,
                    predicted_center[1] - bbox_size[1]/2,
                    predicted_center[0] + bbox_size[0]/2,
                    predicted_center[1] + bbox_size[1]/2
                )
                
                predictions.append((predicted_bbox, 0.8))
        
        return predictions
    
    async def _preload_worker(self):
        """Background worker for preloading spatial data."""
        while True:
            try:
                # Wait for preload tasks
                preload_task = await asyncio.wait_for(
                    self.preload_queue.get(), timeout=60.0
                )
                
                # Check if data is already cached
                cache_key = self.cache_manager._generate_spatial_key(
                    preload_task['bbox'],
                    preload_task['zoom_level'],
                    preload_task['layer']
                )
                
                existing_data = await self.cache_manager.get_spatial_data(
                    preload_task['bbox'],
                    preload_task['zoom_level'],
                    preload_task['layer']
                )
                
                if existing_data is None:
                    # Fetch and cache data
                    # This would call your actual data source
                    data = await self._fetch_spatial_data(
                        preload_task['bbox'],
                        preload_task['zoom_level'],
                        preload_task['layer']
                    )
                    
                    if data:
                        await self.cache_manager.set_spatial_data(
                            preload_task['bbox'],
                            preload_task['zoom_level'],
                            preload_task['layer'],
                            data
                        )
                
            except asyncio.TimeoutError:
                # No preload tasks, worker can exit
                self.preload_worker_running = False
                break
            except Exception as e:
                logger.error(f"Preload worker error: {e}")
```

## Production Monitoring and Observability

### Comprehensive Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Prometheus metrics for spatial operations
SPATIAL_OPERATIONS_TOTAL = Counter(
    'spatial_operations_total',
    'Total number of spatial operations',
    ['operation_type', 'layer', 'result_status']
)

SPATIAL_QUERY_DURATION = Histogram(
    'spatial_query_duration_seconds',
    'Time spent on spatial queries',
    ['query_type', 'complexity_bucket'],
    buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
)

SPATIAL_CACHE_OPERATIONS = Counter(
    'spatial_cache_operations_total',
    'Cache operations',
    ['cache_type', 'operation', 'result']
)

SPATIAL_MEMORY_USAGE = Gauge(
    'spatial_memory_usage_bytes',
    'Memory usage for spatial operations',
    ['component']
)

SPATIAL_INDEX_SIZE = Gauge(
    'spatial_index_size_total',
    'Number of entries in spatial indices',
    ['index_type']
)

class SpatialObservabilityManager:
    """Comprehensive observability for spatial systems."""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.tracer = trace.get_tracer(__name__)
        self.active_spans = {}
        
        # Setup distributed tracing
        trace.set_tracer_provider(TracerProvider())
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
    
    @contextmanager
    def trace_spatial_operation(self, operation_name: str, 
                               bbox: Tuple = None, layer: str = None):
        """Trace spatial operations with distributed tracing."""
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add spatial context to span
            if bbox:
                span.set_attribute("spatial.bbox.min_x", bbox[0])
                span.set_attribute("spatial.bbox.min_y", bbox[1])
                span.set_attribute("spatial.bbox.max_x", bbox[2])
                span.set_attribute("spatial.bbox.max_y", bbox[3])
                span.set_attribute("spatial.bbox.area", 
                                 (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            
            if layer:
                span.set_attribute("spatial.layer", layer)
            
            # Record operation start
            start_time = time.time()
            SPATIAL_OPERATIONS_TOTAL.labels(
                operation_type=operation_name,
                layer=layer or 'unknown',
                result_status='started'
            ).inc()
            
            try:
                yield span
                
                # Record successful completion
                SPATIAL_OPERATIONS_TOTAL.labels(
                    operation_type=operation_name,
                    layer=layer or 'unknown',
                    result_status='success'
                ).inc()
                
            except Exception as e:
                # Record error
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                SPATIAL_OPERATIONS_TOTAL.labels(
                    operation_type=operation_name,
                    layer=layer or 'unknown',
                    result_status='error'
                ).inc()
                
                raise
            finally:
                # Record timing
                duration = time.time() - start_time
                complexity = self._classify_complexity(bbox, operation_name)
                
                SPATIAL_QUERY_DURATION.labels(
                    query_type=operation_name,
                    complexity_bucket=complexity
                ).observe(duration)
    
    def log_spatial_event(self, event_type: str, **context):
        """Log spatial events with structured logging."""
        self.logger.info(
            event_type,
            **context,
            timestamp=time.time()
        )
    
    def record_cache_operation(self, cache_type: str, operation: str, 
                             result: str, response_time: float = None):
        """Record cache operation metrics."""
        SPATIAL_CACHE_OPERATIONS.labels(
            cache_type=cache_type,
            operation=operation,
            result=result
        ).inc()
        
        if response_time:
            SPATIAL_QUERY_DURATION.labels(
                query_type=f"cache_{operation}",
                complexity_bucket="simple"
            ).observe(response_time)
    
    def update_memory_metrics(self, component: str, memory_bytes: int):
        """Update memory usage metrics."""
        SPATIAL_MEMORY_USAGE.labels(component=component).set(memory_bytes)
    
    def update_index_metrics(self, index_type: str, entry_count: int):
        """Update spatial index metrics."""
        SPATIAL_INDEX_SIZE.labels(index_type=index_type).set(entry_count)
    
    def _classify_complexity(self, bbox: Tuple, operation: str) -> str:
        """Classify operation complexity for metrics."""
        if not bbox:
            return "unknown"
        
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if area < 0.001:
            return "simple"
        elif area < 0.1:
            return "medium"
        else:
            return "complex"

class SpatialHealthChecker:
    """Health monitoring for spatial system components."""
    
    def __init__(self, spatial_service, cache_manager, database_pool):
        self.spatial_service = spatial_service
        self.cache_manager = cache_manager
        self.database_pool = database_pool
        self.health_history = []
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_status = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check spatial service health
        try:
            service_health = await self._check_spatial_service()
            health_status['components']['spatial_service'] = service_health
        except Exception as e:
            health_status['components']['spatial_service'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check cache health
        try:
            cache_health = await self._check_cache_health()
            health_status['components']['cache'] = cache_health
        except Exception as e:
            health_status['components']['cache'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check database health
        try:
            db_health = await self._check_database_health()
            health_status['components']['database'] = db_health
        except Exception as e:
            health_status['components']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['overall_status'] = 'unhealthy'
        
        # Store health history
        self.health_history.append(health_status)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-50:]
        
        return health_status
    
    async def _check_spatial_service(self) -> Dict[str, Any]:
        """Check spatial service health."""
        start_time = time.time()
        
        # Test basic query
        test_bbox = (-122.5, 37.7, -122.4, 37.8)  # San Francisco area
        results = await self.spatial_service.query_bbox(*test_bbox)
        
        response_time = time.time() - start_time
        
        return {
            'status': 'healthy' if response_time < 5.0 else 'slow',
            'response_time_seconds': response_time,
            'result_count': len(results) if results else 0
        }
    
    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache system health."""
        # Test cache operations
        test_key = f"health_check_{time.time()}"
        test_data = {'test': True, 'timestamp': time.time()}
        
        # Write test
        start_time = time.time()
        await self.cache_manager.set_spatial_data(
            (-1, -1, 1, 1), 10, "health_check", test_data, ttl_seconds=60
        )
        write_time = time.time() - start_time
        
        # Read test
        start_time = time.time()
        retrieved_data = await self.cache_manager.get_spatial_data(
            (-1, -1, 1, 1), 10, "health_check"
        )
        read_time = time.time() - start_time
        
        # Calculate cache hit rate
        hit_rate = (
            self.cache_manager.cache_stats['hits'] / 
            max(1, self.cache_manager.cache_stats['hits'] + self.cache_manager.cache_stats['misses'])
        )
        
        return {
            'status': 'healthy' if write_time < 0.1 and read_time < 0.05 else 'slow',
            'write_time_seconds': write_time,
            'read_time_seconds': read_time,
            'hit_rate': hit_rate,
            'data_consistency': retrieved_data == test_data
        }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        start_time = time.time()
        
        async with self.database_pool.acquire() as connection:
            # Test simple query
            result = await connection.fetch("SELECT 1 as health_check")
            
        response_time = time.time() - start_time
        
        return {
            'status': 'healthy' if response_time < 1.0 else 'slow',
            'response_time_seconds': response_time,
            'connection_pool_size': len(self.database_pool._holders),
            'active_connections': len([h for h in self.database_pool._holders if h._con])
        }
```

## Professional Development Exercises

### Exercise 1: Build a Performance Analysis Dashboard
Create a comprehensive performance monitoring system:
- Real-time metrics visualization for spatial operations
- Performance regression detection and alerting
- Resource utilization tracking and capacity planning
- Automated performance benchmarking and reporting

### Exercise 2: Implement Auto-Scaling for Spatial Workloads
Design an auto-scaling system that:
- Monitors spatial query patterns and load distribution
- Scales compute resources based on geographic demand
- Implements predictive scaling for known traffic patterns
- Balances cost optimization with performance requirements

### Exercise 3: Create a Multi-Region Caching Strategy
Build a globally distributed caching system:
- Geographic cache distribution and replication
- Intelligent cache warming based on usage patterns
- Cross-region cache invalidation and consistency
- Performance optimization for global user bases

### Exercise 4: Develop a Chaos Engineering Framework
Implement chaos engineering for spatial systems:
- Network partition simulation between geographic regions
- Database failover testing with spatial data consistency
- Cache failure scenarios and graceful degradation
- Load spike simulation and recovery testing

## Industry Context and Real-World Applications

### Production Performance Requirements

**Google Maps Performance Standards:**
- Tile serving: p99 < 50ms globally
- Search queries: p95 < 200ms
- Route calculation: p99 < 2 seconds
- 99.9% availability with global redundancy

**Uber's Spatial Computing Scale:**
- 15+ billion location updates daily
- Sub-second H3 spatial indexing
- Real-time driver matching algorithms
- 99.99% availability for safety-critical operations

**Tesla's Autonomous Driving Requirements:**
- Real-time HD map processing: < 10ms latency
- Sensor fusion with map data: 60+ FPS
- Neural network inference: < 5ms
- Zero tolerance for safety-critical failures

### Enterprise Performance Optimization

**Financial Services (Trading Systems):**
- Geospatial market data: microsecond latency requirements
- Regulatory compliance with audit trails
- Multi-region disaster recovery
- Real-time risk calculations with geographic factors

**Logistics and Supply Chain:**
- Route optimization: millions of calculations per second
- Real-time package tracking at global scale
- Warehouse automation with spatial optimization
- Predictive analytics for demand forecasting

**Smart Cities and IoT:**
- Sensor data processing: millions of updates per minute
- Real-time traffic optimization
- Emergency response coordination
- Energy grid optimization with spatial factors

## Resources

### Performance Engineering
- [High Performance Python by Micha Gorelick](https://www.oreilly.com/library/view/high-performance-python/9781449361747/)
- [Site Reliability Engineering by Google](https://sre.google/books/)
- [Designing Data-Intensive Applications by Martin Kleppmann](https://dataintensive.net/)

### Monitoring and Observability
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Grafana Visualization](https://grafana.com/docs/)
- [Jaeger Distributed Tracing](https://www.jaegertracing.io/docs/)

### Database Performance
- [PostGIS Performance Tuning](https://postgis.net/workshops/postgis-intro/performance.html)
- [PostgreSQL Performance Optimization](https://www.postgresql.org/docs/current/performance-tips.html)
- [Spatial Database Optimization](https://www.amazon.com/Spatial-Databases-Shashi-Shekhar/dp/0136859674)

### Industry Performance Practices
- [Uber's Engineering Blog](https://eng.uber.com/)
- [Google's SRE Practices](https://sre.google/workbook/table-of-contents/)
- [Netflix's Performance Engineering](https://netflixtechblog.com/)

This module provides the foundation for building and operating high-performance, reliable geospatial systems that can scale to serve millions of users while maintaining strict performance and availability requirements.