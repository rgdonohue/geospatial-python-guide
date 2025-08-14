# Day 1: Async I/O and Bounded Concurrency

## Learning Objectives

This module introduces the foundational concepts of asynchronous programming and concurrent I/O operations in the context of distributed geospatial systems. By the end of this module, you will understand:

- **Event-driven architecture** and its role in high-throughput geospatial data pipelines
- **Bounded concurrency patterns** for sustainable resource utilization in distributed tile services
- **Backpressure mechanisms** and adaptive rate limiting for external API consumption
- **Performance measurement methodologies** for latency-sensitive geospatial applications
- **Production-ready error handling** including exponential backoff and circuit breaker patterns

## Theoretical Foundation

### Concurrency vs Parallelism in Geospatial Context

In geospatial systems, we frequently deal with:
- **I/O-bound operations**: Fetching tiles from CDNs, querying spatial databases, reading large raster files
- **CPU-bound operations**: Geometric computations, coordinate transformations, spatial indexing
- **Network-bound operations**: API calls to geocoding services, real-time positioning data

**Concurrency** allows us to efficiently handle multiple I/O operations without blocking, making it ideal for tile fetching, API aggregation, and real-time data streams. **Parallelism** is better suited for computationally intensive tasks like spatial analysis across large datasets.

### Event Loop Architecture

The asyncio event loop provides a single-threaded, non-blocking execution model:

```
┌─────────────────────────────────┐
│         Event Loop             │
├─────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌────┐ │
│  │Task1│ │Task2│ │Task3│ │... │ │
│  └─────┘ └─────┘ └─────┘ └────┘ │
├─────────────────────────────────┤
│       I/O Multiplexing          │
│    (select/epoll/kqueue)        │
└─────────────────────────────────┘
```

**Benefits for Geospatial Applications:**
- **Resource efficiency**: Single thread handles thousands of tile requests
- **Predictable performance**: No context switching overhead
- **Simplified reasoning**: No race conditions or locks needed

### Bounded Concurrency Theory

Unbounded concurrency can lead to:
- **Resource exhaustion**: Too many open file descriptors/connections
- **Server overwhelming**: Violating rate limits on tile servers
- **Memory pressure**: Accumulating response buffers faster than processing

**Semaphore Pattern** provides controlled resource access:
```python
semaphore = asyncio.Semaphore(max_concurrency)

async def bounded_operation():
    async with semaphore:  # Acquire permit
        # Only max_concurrency operations run simultaneously
        await actual_work()
    # Permit automatically released
```

### Retry Strategies and Resilience

**Exponential Backoff** prevents cascading failures:
- **Linear backoff**: 1s, 2s, 3s, 4s... (can cause thundering herd)
- **Exponential backoff**: 1s, 2s, 4s, 8s... (better distribution)
- **Jittered exponential**: Add randomness to prevent synchronized retries

**Error Classification** for geospatial services:
- **Transient errors** (5xx, timeouts): Retryable
- **Client errors** (4xx, invalid coordinates): Non-retryable
- **Rate limiting** (429): Retryable with longer backoff

## System Architecture Context

### Geospatial Data Pipeline Integration

This module's patterns apply across geospatial pipeline components:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │    │   Tile       │    │   Spatial   │    │  Real-time  │
│ Ingestion   │───▶│  Processing  │───▶│   Analysis  │───▶│  Delivery   │
│             │    │              │    │             │    │             │
│• ETL jobs   │    │• Tile gen    │    │• Queries    │    │• WebSockets │
│• Async I/O  │    │• Async proc  │    │• Async DB   │    │• Push notif │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
```

**Key Integration Points:**
- **Data ingestion**: Concurrent downloads from multiple sources
- **Tile processing**: Parallel tile generation with bounded workers
- **Spatial analysis**: Async database queries with connection pooling
- **Real-time delivery**: WebSocket fan-out with backpressure handling

### Production Deployment Considerations

**Infrastructure Scaling:**
- **Horizontal scaling**: Multiple async workers behind load balancer
- **Vertical scaling**: Tune concurrency limits based on machine resources
- **CDN integration**: Reduce load through intelligent caching strategies

**Monitoring and Observability:**
- **Latency percentiles**: p50, p95, p99 response times
- **Throughput metrics**: Requests/second, concurrent connections
- **Error rates**: By error type and upstream service
- **Resource utilization**: Memory, file descriptors, connection pools

## Code Architecture Deep Dive

### Core Components Analysis

```python
async def fetch_tile(
    client: httpx.AsyncClient,
    z: int, x: int, y: int,
    timeout_seconds: float = 10.0,
    retries: int = 2,
    backoff_base: float = 0.2,
    backoff_cap: float = 2.0,
) -> bytes:
```

**Design Decisions:**
- **Dependency injection**: `client` parameter allows reuse and testing
- **Explicit timeouts**: Prevent hanging requests in production
- **Configurable retry logic**: Adaptable to different service characteristics
- **Exponential backoff with cap**: Prevents excessive wait times

### Concurrency Control Implementation

```python
async def fetch_tiles_concurrently(
    tiles: Iterable[Tuple[int, int, int]],
    max_concurrency: int = 10,
    # ...
) -> List[bytes]:
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _bounded_fetch(z: int, x: int, y: int) -> bytes:
        async with semaphore:
            return await fetch_tile(client_obj, z, x, y, ...)
```

**Advanced Patterns:**
- **Semaphore as resource pool**: Controls concurrent operations
- **Context manager protocol**: Ensures proper resource cleanup
- **Task creation vs gathering**: `create_task()` schedules immediately, `gather()` waits for completion

### Performance Measurement Framework

```python
def benchmark(
    tile_count: int = 20,
    max_concurrency: int = 10,
    timeout_seconds: float = 10.0,
    retries: int = 2,
    global_timeout: Optional[float] = None,
) -> None:
```

**Metrics Collection:**
- **Individual request latencies**: Measure per-request performance
- **End-to-end timing**: Total execution time including coordination overhead
- **Percentile analysis**: Understanding tail latency behavior
- **Throughput calculation**: Effective requests per second

## Advanced Implementation Patterns

### Connection Pool Management

```python
class TileClientPool:
    def __init__(self, max_connections: int = 100):
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=20
            ),
            timeout=httpx.Timeout(10.0)
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
```

### Adaptive Rate Limiting

```python
class AdaptiveRateLimiter:
    def __init__(self, initial_rate: float = 10.0):
        self.current_rate = initial_rate
        self.success_count = 0
        self.error_count = 0
    
    async def acquire(self):
        await asyncio.sleep(1.0 / self.current_rate)
    
    def on_success(self):
        self.success_count += 1
        if self.success_count > 10:
            self.current_rate *= 1.1  # Increase rate
            self.success_count = 0
    
    def on_error(self, error_type: str):
        if error_type == "rate_limit":
            self.current_rate *= 0.5  # Decrease rate
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
```

## Performance Optimization Strategies

### Memory Efficiency

```python
async def stream_tiles_to_disk(tiles: List[Tuple[int, int, int]]):
    """Process tiles without loading all into memory simultaneously."""
    async with httpx.AsyncClient() as client:
        async for tile_data in bounded_tile_stream(client, tiles):
            # Process one tile at a time
            await save_tile_to_disk(tile_data)
```

### Batch Processing Optimization

```python
async def fetch_tile_batch(
    client: httpx.AsyncClient,
    tile_batch: List[Tuple[int, int, int]],
    batch_size: int = 50
) -> AsyncGenerator[bytes, None]:
    """Process tiles in optimally-sized batches."""
    for i in range(0, len(tile_batch), batch_size):
        batch = tile_batch[i:i + batch_size]
        results = await asyncio.gather(
            *[fetch_tile(client, z, x, y) for z, x, y in batch],
            return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception):
                # Handle individual failures
                continue
            yield result
```

## Production Deployment Patterns

### Configuration Management

```python
@dataclass
class TileFetcherConfig:
    max_concurrency: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENCY", "10")))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("TIMEOUT_SECONDS", "10.0")))
    retries: int = field(default_factory=lambda: int(os.getenv("RETRIES", "2")))
    base_url: str = field(default_factory=lambda: os.getenv("TILE_BASE_URL", "https://tile.openstreetmap.org"))
    
    def __post_init__(self):
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
```

### Health Monitoring

```python
class TileServiceHealth:
    def __init__(self):
        self.stats = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "avg_latency_ms": 0.0,
            "current_concurrency": 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Endpoint for load balancer health checks."""
        success_rate = self.stats["requests_successful"] / max(self.stats["requests_total"], 1)
        
        return {
            "status": "healthy" if success_rate > 0.95 else "degraded",
            "uptime_seconds": time.time() - self.start_time,
            "stats": self.stats
        }
```

## Running the Module

### Basic Usage
```bash
# Activate environment
source .venv/bin/activate

# Run with default settings
python -m src.day01_concurrency.tile_fetcher

# Tune for your environment
python -m src.day01_concurrency.tile_fetcher \
  --tile-count 100 \
  --max-concurrency 20 \
  --timeout 5 \
  --retries 3
```

### Production Configuration
```bash
# Environment-based configuration
export TILE_BASE_URL="https://your-tile-server.com"
export MAX_CONCURRENCY=50
export TIMEOUT_SECONDS=15
export RETRIES=3

python -m src.day01_concurrency.tile_fetcher
```

### Local Development with Mock Server
```bash
# Terminal 1: Start mock tile server
make run-mock-tiles

# Terminal 2: Run benchmark against local server
TILE_BASE_URL=http://127.0.0.1:8001 \
  python -m src.day01_concurrency.tile_fetcher \
  --tile-count 100 \
  --max-concurrency 50 \
  --timeout 2
```

## Performance Analysis Framework

### Benchmarking Methodology

1. **Baseline Measurement**: Sequential execution as performance floor
2. **Concurrency Scaling**: Test different concurrency levels (1, 5, 10, 20, 50)
3. **Latency Distribution**: Analyze p50, p95, p99 percentiles
4. **Throughput Analysis**: Requests/second vs resource utilization
5. **Error Rate Impact**: Performance under various failure rates

### Expected Results Analysis

**Typical Performance Characteristics:**
- **Sequential**: ~1-2 tiles/second (limited by round-trip time)
- **Concurrent (10)**: ~8-15 tiles/second (network bandwidth limited)
- **Concurrent (50)**: ~20-30 tiles/second (may hit rate limits)

**Performance Scaling Laws:**
- **Amdahl's Law**: Speedup limited by non-parallelizable components
- **Little's Law**: Throughput = Concurrency / Average Latency
- **Universal Scalability Law**: Accounts for contention and coherency costs

## Professional Development Exercises

### Exercise 1: Production-Ready Error Handling
Implement comprehensive error handling with:
- Structured logging with correlation IDs
- Metrics collection for different error types
- Graceful degradation strategies
- Dead letter queue for failed tiles

### Exercise 2: Advanced Backpressure Control
Design a dynamic concurrency controller that:
- Monitors downstream service health
- Adjusts concurrency based on error rates
- Implements exponential backoff with jitter
- Provides circuit breaker functionality

### Exercise 3: Performance Optimization
Profile and optimize the tile fetcher:
- Measure memory allocation patterns
- Implement streaming for large tile sets
- Add connection pooling with keep-alive
- Benchmark different serialization formats

### Exercise 4: Integration Testing
Create comprehensive integration tests:
- Mock tile server with configurable latency/errors
- Property-based testing with Hypothesis
- Load testing with realistic traffic patterns
- Chaos engineering scenarios

## Industry Context and Best Practices

### Real-World Applications

**Mapping Services:**
- Google Maps: Massive tile serving infrastructure with global CDN
- OpenStreetMap: Community-driven tile generation and distribution
- Mapbox: Dynamic styling and real-time data integration

**Autonomous Vehicles:**
- High-definition map tile streaming for real-time navigation
- Low-latency requirements for safety-critical applications
- Redundant data sources with automatic failover

**Location-Based Services:**
- Real-time asset tracking with geofencing
- Location analytics with privacy-preserving aggregation
- Mobile app optimization for battery and bandwidth

### Ethical Considerations

**Resource Responsibility:**
- Respect rate limits and terms of service
- Implement proper caching to reduce server load
- Consider carbon footprint of unnecessary requests

**Data Privacy:**
- Avoid logging sensitive location information
- Implement proper data retention policies
- Consider GDPR/CCPA compliance for user data

### Industry Standards

**OGC Standards:**
- **WMTS**: Web Map Tile Service specification
- **TMS**: Tile Map Service for standardized tile access
- **WMS**: Web Map Service for dynamic map generation

**Performance Benchmarks:**
- **Sub-100ms**: Interactive mapping applications
- **Sub-50ms**: Real-time navigation systems
- **Sub-10ms**: Autonomous vehicle safety systems

## Further Reading and Resources

### Technical References
- [asyncio Official Documentation](https://docs.python.org/3/library/asyncio.html)
- [httpx Advanced Usage](https://www.python-httpx.org/advanced/)
- [Concurrency in Python by Matthew Fowler](https://realpython.com/async-io-python/)

### Performance Engineering
- [High Performance Python by Micha Gorelick](https://www.oreilly.com/library/view/high-performance-python/9781449361747/)
- [Site Reliability Engineering by Google](https://sre.google/books/)

### Geospatial Engineering
- [PostGIS in Action by Regina Obe](https://www.manning.com/books/postgis-in-action-third-edition)
- [Tile Map Service Specification](https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification)

### Production Operations
- [Designing Data-Intensive Applications by Martin Kleppmann](https://dataintensive.net/)
- [Building Microservices by Sam Newman](https://samnewman.io/books/building_microservices_2nd_edition/)

This module provides the foundation for understanding how asynchronous programming patterns enable scalable, resilient geospatial systems. The patterns learned here will be applied throughout the remaining modules as we build increasingly sophisticated geospatial services.