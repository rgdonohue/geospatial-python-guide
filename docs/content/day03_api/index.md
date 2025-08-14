# Day 03 - Api

## Learning Objectives

This module covers enterprise-grade API design and implementation for geospatial services, focusing on scalability, reliability, and maintainability patterns used in production mapping and location-based systems. By the end of this module, you will understand:

- **RESTful API design principles** for geospatial resource modeling and hypermedia controls
- **OpenAPI specification mastery** including advanced schema composition and documentation generation
- **Streaming architecture patterns** for real-time geospatial data delivery and backpressure handling
- **Production observability** with structured logging, metrics collection, and distributed tracing
- **Geospatial-specific API patterns** including tiling protocols, spatial query optimization, and CRS handling
- **Service mesh integration** for microservices communication and traffic management
- **API versioning strategies** for evolving geospatial schemas and backward compatibility

## Theoretical Foundation

### Geospatial API Architecture Patterns

**Resource-Oriented Design:**
Geospatial APIs model spatial entities as first-class resources with clear hierarchies and relationships:

```
/datasets/{dataset_id}                    # Spatial dataset resource
├── /features                            # Feature collection
│   ├── /{feature_id}                   # Individual feature
│   └── /bbox/{bbox}                    # Spatial query
├── /tiles/{z}/{x}/{y}                  # Tile hierarchy
└── /metadata                           # Dataset metadata
```

**Streaming-First Architecture:**
Large geospatial datasets require streaming patterns to prevent memory exhaustion and improve time-to-first-byte:
- **NDJSON streams** for feature collections
- **Chunked tile responses** for large raster data
- **Server-sent events** for real-time location updates
- **WebSocket protocols** for bidirectional spatial data exchange

**Spatial Query Optimization:**
- **Bounding box queries** with spatial indexing
- **Multi-resolution tiling** for efficient data serving
- **Coordinate reference system (CRS) transformations** at API boundaries
- **Geometry simplification** based on zoom levels and client capabilities

### Production API Design Principles

**Idempotency and Consistency:**
Geospatial updates must maintain spatial integrity and referential consistency across distributed systems.

**Eventual Consistency Models:**
Location-based services often require eventual consistency for performance, using patterns like:
- **CQRS (Command Query Responsibility Segregation)** for read/write optimization
- **Event sourcing** for audit trails and temporal queries
- **Conflict-free replicated data types (CRDTs)** for distributed spatial editing

**Rate Limiting and Fair Usage:**
Tile servers and geocoding APIs implement sophisticated rate limiting:
- **Token bucket algorithms** for burst handling
- **Geographic rate limiting** based on request density
- **Adaptive throttling** based on downstream service health

## Core Concepts

### 1. FastAPI Overview
FastAPI uses type hints to validate inputs and auto-generate OpenAPI docs. It’s async-first and great for streaming responses.

### 2. Pydantic Models

Pydantic provides data validation using Python type annotations. It's the foundation for FastAPI's request/response validation.

```python
from pydantic import BaseModel, Field

class TilePath(BaseModel):
    z: int = Field(ge=0, le=22, description="Zoom level (0-22)")
    x: int = Field(description="Tile X coordinate")
    y: int = Field(description="Tile Y coordinate")
```

**Validation Features:**
- Type checking and conversion
- Constraint validation (min/max values, regex patterns)
- Custom validators
- Automatic serialization/deserialization

### 3. Streaming Responses

For large datasets, streaming responses prevent memory issues and improve user experience:

```python
from fastapi.responses import StreamingResponse

async def stream_large_dataset():
    async def generate():
        for item in large_dataset:
            yield f"{item}\n"
    
    return StreamingResponse(generate(), media_type="text/plain")
```

## Code Walkthrough (this repo)

### 1. Basic FastAPI App Structure

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Spatial API Drills")
```

**Key Components:**
- `FastAPI()`: Main application instance
- `title`: Sets the API title in OpenAPI docs
- Additional metadata can be added for better documentation

### 2. Tile Endpoint

```python
@app.get("/tiles/{z}/{x}/{y}.mvt")
async def get_tile(z: int, x: int, y: int):
    if z < 0 or z > 22:
        raise HTTPException(status_code=400, detail="Invalid zoom")
    
    # Demo stream of empty tile
    async def streamer():
        yield b""  # replace with file streaming in drill
    
    return StreamingResponse(
        streamer(), 
        media_type="application/vnd.mapbox-vector-tile"
    )
```

**Features:**
- **Path Parameters**: `{z}`, `{x}`, `{y}` are automatically parsed and validated
- **Validation**: Zoom level is checked (0-22 range)
- **Error Handling**: `HTTPException` returns proper HTTP status codes
- **Streaming**: Uses `StreamingResponse` for efficient data delivery

### 3. Bounding Box Endpoint

```python
@app.get("/stream-features")
async def stream_features(
    min_lat: float, 
    min_lon: float, 
    max_lat: float, 
    max_lon: float
):
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="Invalid bbox")
    
    async def streamer():
        # demo NDJSON
        yield b"{}\n"
    
    return StreamingResponse(
        streamer(), 
        media_type="application/x-ndjson"
    )
```

**Features:**
- **Query Parameters**: Automatically parsed from URL query string
- **Validation**: Checks that bounding box coordinates are valid
- **NDJSON Format**: Newline-delimited JSON for streaming

## Run

### 1. Basic Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API (dev)
uvicorn src.day03_api.app:app --reload
```

### Access
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- OpenAPI Schema: http://localhost:8000/openapi.json
- Prometheus Metrics: http://localhost:8000/metrics

### Smoke it with curl
```bash
# Test tile endpoint
curl "http://localhost:8000/tiles/5/10/12.mvt"

# Test bbox endpoint
curl "http://localhost:8000/stream-features?min_lat=37.0&min_lon=-122.0&max_lat=38.0&max_lon=-121.0"
```

## Exercises

### 1) Implement File Streaming

Replace the placeholder tile streaming with actual file reading:

```python
import os
from pathlib import Path

@app.get("/tiles/{z}/{x}/{y}.mvt")
async def get_tile(z: int, x: int, y: int):
    if z < 0 or z > 22:
        raise HTTPException(status_code=400, detail="Invalid zoom")
    
    # Construct file path
    tile_path = Path(f"tiles/{z}/{x}/{y}.mvt")
    
    if not tile_path.exists():
        raise HTTPException(status_code=404, detail="Tile not found")
    
    async def stream_tile():
        with open(tile_path, "rb") as f:
            while chunk := f.read(8192):  # 8KB chunks
                yield chunk
    
    return StreamingResponse(
        stream_tile(),
        media_type="application/vnd.mapbox-vector-tile",
        headers={"Content-Length": str(tile_path.stat().st_size)}
    )
```

### 2) Add Pydantic Request Models

Create proper request models for validation:

```python
from pydantic import BaseModel, Field, validator
from typing import Optional

class BoundingBox(BaseModel):
    min_lat: float = Field(..., ge=-90, le=90, description="Minimum latitude")
    min_lon: float = Field(..., ge=-180, le=180, description="Minimum longitude")
    max_lat: float = Field(..., ge=-90, le=90, description="Maximum latitude")
    max_lon: float = Field(..., ge=-180, le=180, description="Maximum longitude")
    limit: Optional[int] = Field(100, gt=0, le=1000, description="Maximum features to return")
    offset: Optional[int] = Field(0, ge=0, description="Number of features to skip")
    
    @validator('max_lat')
    def max_lat_must_be_greater(cls, v, values):
        if 'min_lat' in values and v <= values['min_lat']:
            raise ValueError('max_lat must be greater than min_lat')
        return v
    
    @validator('max_lon')
    def max_lon_must_be_greater(cls, v, values):
        if 'min_lon' in values and v <= values['min_lon']:
            raise ValueError('max_lon must be greater than min_lon')
        return v

@app.post("/stream-features")
async def stream_features_post(bbox: BoundingBox):
    # Now we have validated bbox data
    async def streamer():
        # TODO: Implement actual feature streaming
        yield f'{{"bbox": {bbox.dict()}}}\n'
    
    return StreamingResponse(streamer(), media_type="application/x-ndjson")
```

### 3) Implement Real Feature Streaming

```python
import json
from shapely.geometry import box
from shapely.wkt import loads

@app.get("/stream-features")
async def stream_features(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float,
    limit: int = 100, offset: int = 0
):
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="Invalid bbox")
    
    # Load features (in practice, this would come from a database)
    features = load_features_from_source()
    
    # Filter by bounding box
    query_bbox = box(min_lon, min_lat, max_lon, max_lat)
    filtered_features = [
        f for f in features 
        if query_bbox.intersects(loads(f["geometry"]))
    ]
    
    # Apply pagination
    paginated_features = filtered_features[offset:offset + limit]
    
    async def streamer():
        for feature in paginated_features:
            yield json.dumps(feature) + "\n"
    
    return StreamingResponse(
        streamer(),
        media_type="application/x-ndjson",
        headers={
            "X-Total-Count": str(len(filtered_features)),
            "X-Page-Size": str(len(paginated_features)),
            "X-Page-Offset": str(offset)
        }
    )
```

## Advanced

### 1) Response Models

Define explicit response models for better documentation:

```python
from typing import List, Optional

class Feature(BaseModel):
    type: str = "Feature"
    properties: dict
    geometry: dict

class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[Feature]
    total_count: int
    page_size: int
    page_offset: int

@app.get("/features", response_model=FeatureCollection)
async def get_features(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float,
    limit: int = 100, offset: int = 0
):
    # Implementation here
    pass
```

### 2) Custom Response Classes

Create custom response types for specific use cases:

```python
from fastapi.responses import Response
import json

class GeoJSONResponse(Response):
    def __init__(self, content, **kwargs):
        super().__init__(
            content=json.dumps(content),
            media_type="application/geo+json",
            **kwargs
        )

@app.get("/features/geojson")
async def get_features_geojson():
    features = {"type": "FeatureCollection", "features": []}
    return GeoJSONResponse(features)
```

### 3) Middleware and Dependencies

Add cross-cutting concerns:

```python
from fastapi import Depends, Request
import time

async def log_request(request: Request):
    start_time = time.time()
    yield
    process_time = time.time() - start_time
    print(f"{request.method} {request.url.path} took {process_time:.3f}s")

@app.get("/tiles/{z}/{x}/{y}.mvt", dependencies=[Depends(log_request)])
async def get_tile(z: int, x: int, y: int):
    # Implementation here
    pass
```

## Best Practices

### 1. Error Handling
```python
from fastapi import HTTPException, status

@app.get("/tiles/{z}/{x}/{y}.mvt")
async def get_tile(z: int, x: int, y: int):
    try:
        # Implementation
        pass
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tile {z}/{x}/{y} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

### 2. Input Validation
```python
from pydantic import validator

class TileRequest(BaseModel):
    z: int = Field(..., ge=0, le=22)
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    
    @validator('x', 'y')
    def validate_coordinates(cls, v, values):
        if 'z' in values:
            max_coord = 2 ** values['z'] - 1
            if v > max_coord:
                raise ValueError(f'Coordinate {v} exceeds maximum for zoom {values["z"]}')
        return v
```

### 3. Performance Considerations
```python
# Use async for I/O operations
async def load_tile_data(z: int, x: int, y: int):
    # Async file reading or database query
    pass

# Use streaming for large responses
async def stream_large_dataset():
    async def generator():
        async for item in async_data_source():
            yield process_item(item)
    
    return StreamingResponse(generator())
```

## Common Pitfalls

### 1. Forgetting to Await
```python
# ❌ This won't work
@app.get("/data")
async def get_data():
    result = fetch_from_database()  # Missing await
    return result

# ✅ Correct async usage
@app.get("/data")
async def get_data():
    result = await fetch_from_database()
    return result
```

### 2. Blocking Operations
```python
# ❌ This blocks the event loop
@app.get("/slow")
async def slow_endpoint():
    time.sleep(10)  # Blocks everything
    return {"message": "done"}

# ✅ Use asyncio.sleep or run in thread
@app.get("/slow")
async def slow_endpoint():
    await asyncio.sleep(10)  # Non-blocking
    return {"message": "done"}
```

### 3. Memory Issues with Large Data
```python
# ❌ Loads everything into memory
@app.get("/large-dataset")
async def get_large_dataset():
    data = load_all_data()  # Could be GB of data
    return data

# ✅ Stream the data
@app.get("/large-dataset")
async def get_large_dataset():
    async def stream():
        async for item in load_data_stream():
            yield json.dumps(item) + "\n"
    
    return StreamingResponse(stream())
```

## Next Steps

After completing this day:
1. Implement actual file streaming for tiles
2. Add database integration for features
3. Implement proper error handling and logging
4. Add authentication and rate limiting
5. Consider adding caching (Redis, etc.)

## Enterprise Production Patterns

### Microservices Architecture for Geospatial Systems

```python
# Service discovery and configuration
class ServiceRegistry:
    def __init__(self):
        self._services = {}
        self._health_checks = {}
    
    async def register_service(self, name: str, endpoint: str, 
                              health_check: Callable):
        self._services[name] = endpoint
        self._health_checks[name] = health_check
    
    async def discover_service(self, name: str) -> Optional[str]:
        if name in self._services:
            # Check health before returning
            is_healthy = await self._health_checks[name]()
            return self._services[name] if is_healthy else None
        return None

# Circuit breaker for downstream services
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, 
                 timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = "HALF_OPEN"
            else:
                raise HTTPException(503, "Service temporarily unavailable")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

# Geospatial-specific middleware
@app.middleware("http")
async def geospatial_middleware(request: Request, call_next):
    # Add spatial context to request
    if "bbox" in request.query_params:
        request.state.spatial_context = parse_bbox(request.query_params["bbox"])
    
    # Add coordinate system info
    crs = request.headers.get("Accept-CRS", "EPSG:4326")
    request.state.target_crs = crs
    
    response = await call_next(request)
    
    # Add spatial headers to response
    if hasattr(request.state, "spatial_context"):
        response.headers["Content-CRS"] = crs
        response.headers["Content-Bbox"] = format_bbox(request.state.spatial_context)
    
    return response
```

### Advanced Error Handling and Resilience

```python
class GeospatialError(Exception):
    """Base exception for geospatial operations."""
    def __init__(self, message: str, error_code: str, 
                 details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class InvalidGeometryError(GeospatialError):
    """Raised when geometry is invalid or malformed."""
    pass

class SpatialIndexError(GeospatialError):
    """Raised when spatial index operations fail."""
    pass

@app.exception_handler(GeospatialError)
async def geospatial_error_handler(request: Request, exc: GeospatialError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.error_code,
                "message": str(exc),
                "details": exc.details,
                "request_id": request.headers.get("X-Request-ID"),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

# Retry decorator for external service calls
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (httpx.RequestError, asyncio.TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator
```

### Advanced Streaming and Real-Time Patterns

```python
class SpatialEventStream:
    """Real-time spatial event streaming."""
    
    def __init__(self):
        self._subscribers: Dict[str, Set[Queue]] = {}
        self._spatial_index = rtree.index.Index()
    
    async def subscribe_to_region(self, client_queue: Queue, 
                                  bbox: Tuple[float, float, float, float]):
        """Subscribe client to events in a spatial region."""
        region_id = f"bbox_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        
        if region_id not in self._subscribers:
            self._subscribers[region_id] = set()
        
        self._subscribers[region_id].add(client_queue)
        
        # Add region to spatial index
        self._spatial_index.insert(id(client_queue), bbox)
    
    async def publish_spatial_event(self, event: Dict, location: Point):
        """Publish event to all subscribers in the area."""
        # Find subscribers whose regions intersect with event location
        bbox = (location.x, location.y, location.x, location.y)
        intersecting_queues = list(self._spatial_index.intersection(bbox))
        
        for queue_id in intersecting_queues:
            # Find the actual queue object and send event
            for region_subscribers in self._subscribers.values():
                for queue in region_subscribers:
                    if id(queue) == queue_id:
                        try:
                            await queue.put(event)
                        except:
                            # Clean up disconnected clients
                            region_subscribers.discard(queue)

@app.websocket("/ws/spatial-events")
async def spatial_event_websocket(websocket: WebSocket):
    await websocket.accept()
    client_queue = asyncio.Queue()
    
    try:
        # Get initial subscription parameters
        bbox_data = await websocket.receive_json()
        bbox = tuple(bbox_data["bbox"])
        
        # Subscribe to spatial events
        await spatial_stream.subscribe_to_region(client_queue, bbox)
        
        # Send events to client
        while True:
            event = await client_queue.get()
            await websocket.send_json(event)
            
    except WebSocketDisconnect:
        # Clean up subscription
        pass
```

### Security and Authentication Patterns

```python
# JWT-based authentication with spatial scopes
class SpatialJWTBearer(HTTPBearer):
    def __init__(self, spatial_authority: str):
        super().__init__(auto_error=True)
        self.spatial_authority = spatial_authority
    
    async def __call__(self, request: Request):
        credentials = await super().__call__(request)
        token = credentials.credentials
        
        # Decode and validate JWT
        try:
            payload = jwt.decode(token, SPATIAL_SECRET_KEY, algorithms=["HS256"])
            
            # Check spatial permissions
            spatial_scopes = payload.get("spatial_scopes", [])
            required_scope = self._determine_spatial_scope(request)
            
            if required_scope not in spatial_scopes:
                raise HTTPException(403, "Insufficient spatial permissions")
            
            return payload
            
        except jwt.PyJWTError:
            raise HTTPException(401, "Invalid authentication token")
    
    def _determine_spatial_scope(self, request: Request) -> str:
        """Determine required spatial scope based on request."""
        if "bbox" in request.query_params:
            bbox = parse_bbox(request.query_params["bbox"])
            if self._is_sensitive_area(bbox):
                return "sensitive_area_access"
        
        return "general_spatial_access"

# API key rate limiting with geographic awareness
class GeographicRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(self, api_key: str, 
                              location: Optional[Point] = None) -> bool:
        """Check rate limits with geographic weighting."""
        base_key = f"rate_limit:{api_key}"
        
        # Standard rate limiting
        current_count = await self.redis.get(f"{base_key}:count")
        if current_count and int(current_count) > 1000:  # 1000 req/hour
            return False
        
        # Geographic rate limiting for high-value areas
        if location and self._is_high_traffic_area(location):
            geo_key = f"{base_key}:geo:{self._get_grid_cell(location)}"
            geo_count = await self.redis.get(geo_key)
            if geo_count and int(geo_count) > 100:  # 100 req/hour per grid cell
                return False
        
        return True
```

### Performance Optimization and Caching

```python
class SpatialCache:
    """Multi-level spatial caching system."""
    
    def __init__(self, redis_client, memory_cache_size: int = 1000):
        self.redis = redis_client
        self.memory_cache = LRUCache(memory_cache_size)
        self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.1)
    
    async def get_features_in_bbox(self, bbox: Tuple[float, float, float, float], 
                                   zoom_level: int) -> Optional[List[Dict]]:
        """Get cached features with spatial and zoom-level awareness."""
        cache_key = f"features:{self._hash_bbox(bbox)}:z{zoom_level}"
        
        # Check bloom filter first
        if cache_key not in self.bloom_filter:
            return None
        
        # Check memory cache
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check Redis cache
        cached_data = await self.redis.get(cache_key)
        if cached_data:
            features = json.loads(cached_data)
            self.memory_cache[cache_key] = features
            return features
        
        return None
    
    async def cache_features(self, bbox: Tuple[float, float, float, float],
                           zoom_level: int, features: List[Dict],
                           ttl_seconds: int = 3600):
        """Cache features with appropriate TTL based on data volatility."""
        cache_key = f"features:{self._hash_bbox(bbox)}:z{zoom_level}"
        
        # Add to bloom filter
        self.bloom_filter.add(cache_key)
        
        # Cache in memory
        self.memory_cache[cache_key] = features
        
        # Cache in Redis with TTL
        await self.redis.setex(
            cache_key, 
            ttl_seconds, 
            json.dumps(features, cls=GeospatialJSONEncoder)
        )

# Geometry simplification middleware
@app.middleware("http")
async def geometry_optimization_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # Simplify geometries based on zoom level and viewport
    if (response.headers.get("content-type") == "application/geo+json" and
        "zoom" in request.query_params):
        
        zoom_level = int(request.query_params["zoom"])
        tolerance = calculate_simplification_tolerance(zoom_level)
        
        # Simplify response geometries
        simplified_content = simplify_geojson_response(
            response.content, tolerance
        )
        
        return Response(
            content=simplified_content,
            media_type=response.media_type,
            headers=dict(response.headers)
        )
    
    return response
```

### Observability and Monitoring

```python
# Structured logging for geospatial operations
class SpatialLogger:
    def __init__(self):
        self.logger = structlog.get_logger()
    
    def log_spatial_query(self, bbox: Tuple[float, float, float, float],
                         feature_count: int, query_time_ms: float,
                         user_id: str, request_id: str):
        self.logger.info(
            "spatial_query_executed",
            bbox=bbox,
            feature_count=feature_count,
            query_time_ms=query_time_ms,
            user_id=user_id,
            request_id=request_id,
            bbox_area=calculate_bbox_area(bbox)
        )

# Distributed tracing for spatial operations
async def trace_spatial_operation(operation_name: str):
    with tracer.start_as_current_span(operation_name) as span:
        # Add spatial context to trace
        if hasattr(request.state, "spatial_context"):
            bbox = request.state.spatial_context
            span.set_attribute("spatial.bbox", str(bbox))
            span.set_attribute("spatial.area", calculate_bbox_area(bbox))
        
        yield span

# Custom metrics for geospatial APIs
SPATIAL_QUERY_HISTOGRAM = Histogram(
    "spatial_query_duration_seconds",
    "Time spent executing spatial queries",
    labelnames=["query_type", "zoom_level", "result_size_category"]
)

SPATIAL_CACHE_HIT_COUNTER = Counter(
    "spatial_cache_hits_total",
    "Number of spatial cache hits",
    labelnames=["cache_level", "zoom_level"]
)

@app.middleware("http")
async def spatial_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Record spatial query metrics
    if "/bbox" in request.url.path:
        zoom_level = request.query_params.get("zoom", "unknown")
        result_count = response.headers.get("X-Feature-Count", "0")
        result_category = categorize_result_size(int(result_count))
        
        SPATIAL_QUERY_HISTOGRAM.labels(
            query_type="bbox",
            zoom_level=zoom_level,
            result_size_category=result_category
        ).observe(duration)
    
    return response
```

## Real-World Integration Examples

### Integration with PostGIS

```python
class PostGISSpatialService:
    """Production-ready PostGIS integration."""
    
    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool
    
    async def query_features_in_bbox(
        self, 
        bbox: Tuple[float, float, float, float],
        table_name: str,
        limit: int = 1000,
        offset: int = 0,
        geometry_column: str = "geom"
    ) -> List[Dict]:
        """Efficient spatial query with PostGIS."""
        
        query = f"""
        SELECT 
            ST_AsGeoJSON({geometry_column}) as geometry,
            jsonb_build_object(
                'id', id,
                'properties', properties
            ) as feature
        FROM {table_name}
        WHERE ST_Intersects(
            {geometry_column},
            ST_MakeEnvelope($1, $2, $3, $4, 4326)
        )
        ORDER BY ST_Area({geometry_column}) DESC
        LIMIT $5 OFFSET $6
        """
        
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                query, bbox[0], bbox[1], bbox[2], bbox[3], limit, offset
            )
            
            features = []
            for row in rows:
                geometry = json.loads(row["geometry"])
                feature_data = json.loads(row["feature"])
                
                features.append({
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": feature_data["properties"],
                    "id": feature_data["id"]
                })
            
            return features
```

### Cloud Storage Integration

```python
class CloudOptimizedGeoTIFFService:
    """Service for streaming Cloud Optimized GeoTIFF data."""
    
    def __init__(self, s3_client):
        self.s3 = s3_client
    
    async def stream_cog_tile(
        self,
        bucket: str,
        key: str,
        z: int, x: int, y: int,
        bands: List[int] = None
    ) -> StreamingResponse:
        """Stream tile from Cloud Optimized GeoTIFF."""
        
        # Calculate byte range for tile
        tile_bounds = calculate_tile_bounds(z, x, y)
        byte_range = await self._calculate_cog_byte_range(
            bucket, key, tile_bounds, bands
        )
        
        # Stream partial object from S3
        async def tile_generator():
            response = await self.s3.get_object(
                Bucket=bucket,
                Key=key,
                Range=f"bytes={byte_range[0]}-{byte_range[1]}"
            )
            
            async for chunk in response["Body"]:
                yield chunk
        
        return StreamingResponse(
            tile_generator(),
            media_type="image/tiff",
            headers={
                "Content-Length": str(byte_range[1] - byte_range[0] + 1),
                "Accept-Ranges": "bytes"
            }
        )
```

## Industry Standards and Compliance

### OGC Compliance

```python
class OGCCompliantFeatureService:
    """OGC Web Feature Service (WFS) compliant implementation."""
    
    @app.get("/wfs", response_class=XMLResponse)
    async def get_capabilities():
        """Return WFS capabilities document."""
        capabilities = generate_wfs_capabilities()
        return XMLResponse(content=capabilities)
    
    @app.get("/wfs/features", response_model=FeatureCollection)
    async def get_features(
        service: str = Query("WFS"),
        version: str = Query("2.0.0"),
        request: str = Query("GetFeature"),
        typeName: str = Query(...),
        bbox: Optional[str] = Query(None),
        maxFeatures: int = Query(1000),
        outputFormat: str = Query("application/geopackage+sqlite3")
    ):
        """OGC WFS GetFeature operation."""
        
        if service != "WFS":
            raise HTTPException(400, "Invalid service parameter")
        
        # Validate and parse parameters according to OGC spec
        parsed_bbox = parse_ogc_bbox(bbox) if bbox else None
        
        # Execute spatial query
        features = await spatial_service.query_features(
            type_name=typeName,
            bbox=parsed_bbox,
            limit=maxFeatures
        )
        
        return FeatureCollection(features=features)
```

## Professional Development Exercises

### Exercise 1: Build a Multi-Tenant Geospatial API
Design and implement a multi-tenant spatial data API:
- Implement tenant isolation at the database level
- Add role-based access control for spatial operations
- Create tenant-specific rate limiting
- Implement cross-tenant spatial analytics (where permitted)

### Exercise 2: Real-Time Location Tracking Service
Create a real-time location tracking and geofencing service:
- WebSocket-based real-time location updates
- Spatial event triggers and notifications
- Historical trajectory storage and querying
- Privacy controls and data retention policies

### Exercise 3: Distributed Tile Caching System
Build a distributed tile caching and generation system:
- Multi-level caching (CDN, Redis, local)
- On-demand tile generation with queuing
- Cache invalidation strategies
- Performance monitoring and auto-scaling

### Exercise 4: Geospatial Data Pipeline API
Design an API for managing geospatial ETL pipelines:
- Job scheduling and dependency management
- Progress tracking and error handling
- Data quality validation and reporting
- Integration with external data sources

## Resources

### Technical Standards
- [OGC Web Feature Service (WFS) 2.0](https://www.ogc.org/standards/wfs)
- [OGC API - Features](https://ogcapi.ogc.org/features/)
- [Tile Map Service (TMS) Specification](https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification)
- [Cloud Optimized GeoTIFF](https://www.cogeo.org/)

### FastAPI and Python
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [OpenAPI Specification](https://swagger.io/specification/)
- [Streaming Responses in FastAPI](https://fastapi.tiangolo.com/advanced/custom-response/)

### Geospatial Technologies
- [PostGIS Documentation](https://postgis.net/documentation/)
- [GDAL/OGR Python Bindings](https://gdal.org/python/)
- [Shapely Documentation](https://shapely.readthedocs.io/)
- [GeoJSON Specification](https://geojson.org/)

### Production Operations
- [Site Reliability Engineering](https://sre.google/books/)
- [Microservices Patterns by Chris Richardson](https://microservices.io/)
- [Building Event-Driven Microservices](https://www.oreilly.com/library/view/building-event-driven-microservices/9781492057888/)

This module provides the foundation for building production-grade geospatial APIs that can scale to serve millions of users while maintaining data integrity, performance, and reliability standards expected in enterprise environments.
