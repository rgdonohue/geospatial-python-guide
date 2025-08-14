# Day 07 - Mock

## Graduate-Level Learning Objectives

This capstone project synthesizes all concepts from the previous modules into a comprehensive enterprise-grade geospatial service. By the end of this module, you will have demonstrated:

- **Full-stack geospatial application development** with production-ready architecture patterns
- **API design and implementation** following RESTful principles and OpenAPI specifications
- **Spatial data modeling and query optimization** for large-scale road network datasets
- **Performance engineering and scalability** considerations for high-throughput spatial services
- **Testing strategies and quality assurance** for mission-critical geospatial applications
- **Documentation and maintainability** practices for enterprise software development
- **Integration with enterprise infrastructure** including monitoring, logging, and deployment

## Project Overview: Real-Time Road Network Intelligence Service

### Business Context

You are building a core infrastructure service for a logistics company that needs real-time access to road network data for:
- **Route optimization algorithms** requiring sub-second response times
- **Fleet management systems** tracking thousands of vehicles
- **Dynamic pricing models** based on traffic and road conditions
- **Regulatory compliance** for commercial vehicle routing
- **Emergency response coordination** requiring highest availability

### Technical Requirements

**Performance Targets:**
- **Query latency**: p95 < 100ms, p99 < 200ms for bounding box queries
- **Throughput**: 1,000+ QPS sustained load, 5,000+ QPS peak capacity
- **Availability**: 99.9% uptime with graceful degradation
- **Data consistency**: Strong consistency for updates, eventual consistency for reads
- **Scalability**: Horizontal scaling to support 10x traffic growth

**Functional Requirements:**
- **Spatial queries**: Efficient bounding box queries with pagination
- **Network topology**: Connected road analysis for routing algorithms
- **Dynamic updates**: Real-time road condition and metadata updates
- **Multi-format support**: GeoJSON for web clients, Protocol Buffers for internal services
- **Caching strategy**: Multi-tier caching with spatial awareness

## Architecture Deep Dive

### System Design Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Web Clients   │
│   (HAProxy)     │────│   (Nginx)       │────│   (React SPA)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │              ┌─────────────────┐
         │                       │              │  Mobile Apps    │
         │                       │──────────────│  (iOS/Android)  │
         │                       │              └─────────────────┘
         │                       │
         │              ┌─────────────────┐
         │              │   CDN/Edge      │
         │              │   (CloudFlare)  │
         │              └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application Layer                     │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   Spatial API   │   Network API   │   Updates API   │ Admin API │
│   Service       │   Service       │   Service       │ Service   │
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│   Spatial       │   Graph         │   Event         │ Config    │
│   Queries       │   Analysis      │   Processing    │ Mgmt      │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │   PostgreSQL    │    │   Monitoring    │
│   (L1/L2)       │    │   + PostGIS     │    │   (Prometheus)  │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│• Spatial tiles  │    │• Road segments  │    │• Metrics        │
│• Query results  │    │• Spatial index  │    │• Tracing        │
│• Metadata       │    │• Audit log      │    │• Alerting       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Model Design

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from geoalchemy2 import Geometry
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

Base = declarative_base()

class RoadSegment(Base):
    """Enterprise-grade road segment model with comprehensive metadata."""
    
    __tablename__ = 'road_segments'
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(100), unique=True, nullable=False)  # From data provider
    
    # Basic attributes
    name = Column(String(255), nullable=True)
    road_type = Column(String(50), nullable=False)  # highway, arterial, residential, etc.
    speed_limit_kmh = Column(Integer, nullable=True)
    direction = Column(String(20), nullable=False)  # bidirectional, forward, backward
    
    # Geometry (using PostGIS)
    geometry = Column(Geometry('LINESTRING', srid=4326), nullable=False)
    length_meters = Column(Float, nullable=True)
    
    # Network topology
    start_node_id = Column(UUID(as_uuid=True), nullable=True)
    end_node_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Traffic and conditions
    traffic_level = Column(String(20), default='unknown')  # low, medium, high, blocked
    surface_type = Column(String(50), nullable=True)  # asphalt, concrete, gravel, etc.
    lanes_count = Column(Integer, nullable=True)
    
    # Restrictions and regulations
    truck_restricted = Column(Boolean, default=False)
    hazmat_restricted = Column(Boolean, default=False)
    height_limit_meters = Column(Float, nullable=True)
    weight_limit_kg = Column(Float, nullable=True)
    
    # Quality and metadata
    data_source = Column(String(100), nullable=False)
    accuracy_meters = Column(Float, nullable=True)
    confidence_score = Column(Integer, default=50)  # 0-100
    
    # Extended attributes (flexible schema)
    attributes = Column(JSONB, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100), nullable=True)
    updated_by = Column(String(100), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    
    # Soft delete
    is_active = Column(Boolean, default=True, nullable=False)
    deleted_at = Column(DateTime, nullable=True)
    
    # Spatial indexes for performance
    __table_args__ = (
        Index('idx_road_segments_geometry', geometry, postgresql_using='gist'),
        Index('idx_road_segments_road_type', road_type),
        Index('idx_road_segments_traffic_level', traffic_level),
        Index('idx_road_segments_external_id', external_id),
        Index('idx_road_segments_active', is_active),
        Index('idx_road_segments_bbox', 
              'ST_Envelope(geometry)', postgresql_using='gist'),
    )

class RoadNode(Base):
    """Junction/intersection points for network topology."""
    
    __tablename__ = 'road_nodes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    geometry = Column(Geometry('POINT', srid=4326), nullable=False)
    node_type = Column(String(50), nullable=False)  # intersection, terminal, bridge
    elevation_meters = Column(Float, nullable=True)
    
    # Traffic control
    has_traffic_light = Column(Boolean, default=False)
    has_stop_sign = Column(Boolean, default=False)
    is_roundabout = Column(Boolean, default=False)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_road_nodes_geometry', geometry, postgresql_using='gist'),
        Index('idx_road_nodes_type', node_type),
    )

class TrafficEvent(Base):
    """Real-time traffic events and incidents."""
    
    __tablename__ = 'traffic_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    road_segment_id = Column(UUID(as_uuid=True), nullable=False)
    
    event_type = Column(String(50), nullable=False)  # accident, construction, closure
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    description = Column(Text, nullable=True)
    
    # Temporal extent
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    estimated_duration_minutes = Column(Integer, nullable=True)
    
    # Spatial extent
    affected_geometry = Column(Geometry('LINESTRING', srid=4326), nullable=True)
    impact_radius_meters = Column(Float, nullable=True)
    
    # Impact assessment
    delay_minutes = Column(Integer, nullable=True)
    speed_reduction_percent = Column(Integer, nullable=True)
    lanes_blocked = Column(Integer, nullable=True)
    
    # Source and verification
    source = Column(String(100), nullable=False)
    verified = Column(Boolean, default=False)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_traffic_events_road_segment', road_segment_id),
        Index('idx_traffic_events_time_range', start_time, end_time),
        Index('idx_traffic_events_severity', severity),
        Index('idx_traffic_events_active', 
              "start_time <= NOW() AND (end_time IS NULL OR end_time >= NOW())"),
    )
```

### Service Layer Architecture

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
from contextlib import asynccontextmanager

@dataclass
class BoundingBox:
    """Type-safe bounding box with validation."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    
    def __post_init__(self):
        if not (-90 <= self.min_lat <= 90) or not (-90 <= self.max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= self.min_lon <= 180) or not (-180 <= self.max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be less than max_lon")
    
    @property
    def area(self) -> float:
        """Calculate bounding box area in square degrees."""
        return (self.max_lat - self.min_lat) * (self.max_lon - self.min_lon)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point as (lat, lon)."""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2
        )

@dataclass
class SpatialQueryOptions:
    """Configuration for spatial queries."""
    limit: int = 1000
    offset: int = 0
    include_geometry: bool = True
    simplify_tolerance: Optional[float] = None
    buffer_meters: Optional[float] = None
    
    def __post_init__(self):
        if self.limit <= 0 or self.limit > 10000:
            raise ValueError("Limit must be between 1 and 10000")
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")

class RoadNetworkService:
    """Enterprise road network service with comprehensive functionality."""
    
    def __init__(self, 
                 database_pool: asyncpg.Pool,
                 cache_manager: SpatialCacheManager,
                 metrics_collector: SpatialObservabilityManager):
        self.db_pool = database_pool
        self.cache = cache_manager
        self.metrics = metrics_collector
        self.graph_analyzer = RoadNetworkAnalyzer()
    
    async def query_roads_in_bbox(self, 
                                 bbox: BoundingBox,
                                 options: SpatialQueryOptions = None) -> Dict:
        """Query roads within bounding box with enterprise features."""
        options = options or SpatialQueryOptions()
        
        with self.metrics.trace_spatial_operation("bbox_query", 
                                                 bbox=(bbox.min_lon, bbox.min_lat, 
                                                      bbox.max_lon, bbox.max_lat)):
            # Check cache first
            cache_key = self._generate_cache_key(bbox, options)
            cached_result = await self.cache.get_spatial_data(
                (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
                zoom_level=self._infer_zoom_level(bbox.area),
                layer="roads"
            )
            
            if cached_result:
                self.metrics.record_cache_operation("spatial", "get", "hit")
                return cached_result
            
            # Query database
            self.metrics.record_cache_operation("spatial", "get", "miss")
            result = await self._execute_spatial_query(bbox, options)
            
            # Cache result
            await self.cache.set_spatial_data(
                (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
                zoom_level=self._infer_zoom_level(bbox.area),
                layer="roads",
                data=result,
                ttl_seconds=self._calculate_cache_ttl(bbox.area)
            )
            
            return result
    
    async def find_connected_roads(self, road_id: str, 
                                  max_distance_km: float = 5.0) -> List[str]:
        """Find roads connected to a given road segment."""
        
        with self.metrics.trace_spatial_operation("connectivity_analysis"):
            # Get the target road segment
            async with self.db_pool.acquire() as conn:
                road_query = """
                SELECT id, start_node_id, end_node_id, ST_AsGeoJSON(geometry) as geometry
                FROM road_segments 
                WHERE id = $1 AND is_active = true
                """
                road = await conn.fetchrow(road_query, road_id)
                
                if not road:
                    raise ValueError(f"Road segment {road_id} not found")
                
                # Find connected roads through shared nodes
                connected_query = """
                WITH target_nodes AS (
                    SELECT start_node_id as node_id FROM road_segments WHERE id = $1
                    UNION
                    SELECT end_node_id as node_id FROM road_segments WHERE id = $1
                ),
                connected_roads AS (
                    SELECT DISTINCT rs.id, rs.name, 
                           ST_Distance(rs.geometry::geography, target.geometry::geography) as distance_meters
                    FROM road_segments rs
                    CROSS JOIN (SELECT geometry FROM road_segments WHERE id = $1) target
                    WHERE rs.is_active = true 
                      AND rs.id != $1
                      AND (rs.start_node_id IN (SELECT node_id FROM target_nodes)
                           OR rs.end_node_id IN (SELECT node_id FROM target_nodes)
                           OR ST_DWithin(rs.geometry::geography, target.geometry::geography, $2))
                )
                SELECT id FROM connected_roads 
                WHERE distance_meters <= $2
                ORDER BY distance_meters
                LIMIT 100
                """
                
                connected_roads = await conn.fetch(
                    connected_query, road_id, max_distance_km * 1000
                )
                
                return [str(row['id']) for row in connected_roads]
    
    async def update_road_metadata(self, road_id: str, updates: Dict) -> Dict:
        """Update road segment metadata with validation and audit trail."""
        
        with self.metrics.trace_spatial_operation("road_update"):
            # Validate updates
            allowed_fields = {
                'speed_limit_kmh', 'traffic_level', 'surface_type',
                'truck_restricted', 'hazmat_restricted', 'attributes'
            }
            
            invalid_fields = set(updates.keys()) - allowed_fields
            if invalid_fields:
                raise ValueError(f"Invalid fields: {invalid_fields}")
            
            # Validate values
            if 'speed_limit_kmh' in updates:
                speed_limit = updates['speed_limit_kmh']
                if not isinstance(speed_limit, int) or speed_limit < 0 or speed_limit > 200:
                    raise ValueError("Speed limit must be between 0 and 200 km/h")
            
            if 'traffic_level' in updates:
                valid_levels = {'low', 'medium', 'high', 'blocked'}
                if updates['traffic_level'] not in valid_levels:
                    raise ValueError(f"Traffic level must be one of: {valid_levels}")
            
            # Perform update with optimistic locking
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Get current version
                    current = await conn.fetchrow(
                        "SELECT version FROM road_segments WHERE id = $1 AND is_active = true",
                        road_id
                    )
                    
                    if not current:
                        raise ValueError(f"Road segment {road_id} not found")
                    
                    # Build update query
                    set_clauses = []
                    params = [road_id, current['version']]
                    param_idx = 3
                    
                    for field, value in updates.items():
                        set_clauses.append(f"{field} = ${param_idx}")
                        params.append(value)
                        param_idx += 1
                    
                    set_clauses.extend([
                        f"updated_at = NOW()",
                        f"version = version + 1"
                    ])
                    
                    update_query = f"""
                    UPDATE road_segments 
                    SET {', '.join(set_clauses)}
                    WHERE id = $1 AND version = $2 AND is_active = true
                    RETURNING id, version, updated_at
                    """
                    
                    result = await conn.fetchrow(update_query, *params)
                    
                    if not result:
                        raise ValueError("Update failed - segment may have been modified")
                    
                    # Invalidate cache
                    await self._invalidate_road_cache(road_id)
                    
                    # Log audit event
                    self.metrics.log_spatial_event(
                        "road_updated",
                        road_id=road_id,
                        updates=updates,
                        new_version=result['version']
                    )
                    
                    return {
                        'id': str(result['id']),
                        'version': result['version'],
                        'updated_at': result['updated_at'].isoformat()
                    }
    
    async def stream_roads_in_region(self, 
                                   bbox: BoundingBox,
                                   batch_size: int = 1000) -> AsyncIterator[List[Dict]]:
        """Stream roads in large regions efficiently."""
        
        with self.metrics.trace_spatial_operation("streaming_query"):
            offset = 0
            
            while True:
                options = SpatialQueryOptions(
                    limit=batch_size,
                    offset=offset,
                    include_geometry=True
                )
                
                batch = await self.query_roads_in_bbox(bbox, options)
                features = batch.get('features', [])
                
                if not features:
                    break
                
                yield features
                
                if len(features) < batch_size:
                    break
                
                offset += batch_size
                
                # Prevent runaway queries
                if offset > 100000:
                    raise ValueError("Query too large - use smaller bounding box")
    
    async def _execute_spatial_query(self, bbox: BoundingBox, 
                                   options: SpatialQueryOptions) -> Dict:
        """Execute optimized spatial query against database."""
        
        query = """
        SELECT 
            id,
            external_id,
            name,
            road_type,
            speed_limit_kmh,
            direction,
            traffic_level,
            surface_type,
            lanes_count,
            truck_restricted,
            hazmat_restricted,
            length_meters,
            confidence_score,
            attributes,
            created_at,
            updated_at
        """
        
        if options.include_geometry:
            if options.simplify_tolerance:
                query += f", ST_AsGeoJSON(ST_Simplify(geometry, {options.simplify_tolerance})) as geometry"
            else:
                query += ", ST_AsGeoJSON(geometry) as geometry"
        
        query += """
        FROM road_segments
        WHERE is_active = true
          AND geometry && ST_MakeEnvelope($1, $2, $3, $4, 4326)
          AND ST_Intersects(geometry, ST_MakeEnvelope($1, $2, $3, $4, 4326))
        ORDER BY road_type, name
        LIMIT $5 OFFSET $6
        """
        
        # Count query for total
        count_query = """
        SELECT COUNT(*) as total
        FROM road_segments
        WHERE is_active = true
          AND geometry && ST_MakeEnvelope($1, $2, $3, $4, 4326)
          AND ST_Intersects(geometry, ST_MakeEnvelope($1, $2, $3, $4, 4326))
        """
        
        async with self.db_pool.acquire() as conn:
            # Execute queries in parallel
            roads_task = conn.fetch(
                query, 
                bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat,
                options.limit, options.offset
            )
            
            count_task = conn.fetchval(
                count_query,
                bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
            )
            
            roads, total_count = await asyncio.gather(roads_task, count_task)
            
            # Convert to GeoJSON format
            features = []
            for road in roads:
                feature = {
                    'type': 'Feature',
                    'id': str(road['id']),
                    'properties': {
                        'external_id': road['external_id'],
                        'name': road['name'],
                        'road_type': road['road_type'],
                        'speed_limit_kmh': road['speed_limit_kmh'],
                        'direction': road['direction'],
                        'traffic_level': road['traffic_level'],
                        'surface_type': road['surface_type'],
                        'lanes_count': road['lanes_count'],
                        'truck_restricted': road['truck_restricted'],
                        'hazmat_restricted': road['hazmat_restricted'],
                        'length_meters': road['length_meters'],
                        'confidence_score': road['confidence_score'],
                        'attributes': road['attributes'],
                        'created_at': road['created_at'].isoformat() if road['created_at'] else None,
                        'updated_at': road['updated_at'].isoformat() if road['updated_at'] else None
                    }
                }
                
                if options.include_geometry and road['geometry']:
                    import json
                    feature['geometry'] = json.loads(road['geometry'])
                
                features.append(feature)
            
            return {
                'type': 'FeatureCollection',
                'features': features,
                'count': len(features),
                'total': total_count,
                'bbox': [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat],
                'limit': options.limit,
                'offset': options.offset
            }
```

### API Implementation

```python
from fastapi import FastAPI, HTTPException, Depends, Query, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
import json

class SpatialQueryRequest(BaseModel):
    """Request model for spatial queries with validation."""
    
    min_lat: float = Field(..., ge=-90, le=90, description="Minimum latitude")
    min_lon: float = Field(..., ge=-180, le=180, description="Minimum longitude")
    max_lat: float = Field(..., ge=-90, le=90, description="Maximum latitude")
    max_lon: float = Field(..., ge=-180, le=180, description="Maximum longitude")
    
    limit: int = Field(1000, ge=1, le=10000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")
    
    include_geometry: bool = Field(True, description="Include geometry in response")
    simplify_tolerance: Optional[float] = Field(None, ge=0, le=0.1, 
                                               description="Geometry simplification tolerance")
    
    @validator('max_lat')
    def validate_lat_range(cls, v, values):
        if 'min_lat' in values and v <= values['min_lat']:
            raise ValueError('max_lat must be greater than min_lat')
        return v
    
    @validator('max_lon')
    def validate_lon_range(cls, v, values):
        if 'min_lon' in values and v <= values['min_lon']:
            raise ValueError('max_lon must be greater than min_lon')
        return v

class RoadUpdateRequest(BaseModel):
    """Request model for road updates."""
    
    speed_limit_kmh: Optional[int] = Field(None, ge=0, le=200)
    traffic_level: Optional[str] = Field(None, regex='^(low|medium|high|blocked)$')
    surface_type: Optional[str] = Field(None, max_length=50)
    truck_restricted: Optional[bool] = None
    hazmat_restricted: Optional[bool] = None
    attributes: Optional[Dict[str, Any]] = None

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool = True
    data: Any = None
    message: str = ""
    request_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# FastAPI application setup
app = FastAPI(
    title="Road Network Intelligence Service",
    description="Enterprise-grade road network API for logistics and navigation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global dependencies
async def get_road_service() -> RoadNetworkService:
    """Dependency injection for road network service."""
    # This would be initialized with proper database pool and cache
    return RoadNetworkService(db_pool, cache_manager, metrics_collector)

@app.get("/api/v1/roads/bbox", 
         response_model=APIResponse,
         summary="Query roads in bounding box",
         description="Retrieve road segments within a geographic bounding box")
async def query_roads_bbox(
    min_lat: float = Query(..., ge=-90, le=90, description="Minimum latitude"),
    min_lon: float = Query(..., ge=-180, le=180, description="Minimum longitude"), 
    max_lat: float = Query(..., ge=-90, le=90, description="Maximum latitude"),
    max_lon: float = Query(..., ge=-180, le=180, description="Maximum longitude"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset"),
    include_geometry: bool = Query(True, description="Include geometry"),
    simplify_tolerance: Optional[float] = Query(None, ge=0, le=0.1),
    service: RoadNetworkService = Depends(get_road_service)
):
    """Query roads within a bounding box with comprehensive options."""
    
    try:
        # Validate bounding box
        if max_lat <= min_lat:
            raise HTTPException(400, "max_lat must be greater than min_lat")
        if max_lon <= min_lon:
            raise HTTPException(400, "max_lon must be greater than min_lon")
        
        # Check area limit to prevent abuse
        area = (max_lat - min_lat) * (max_lon - min_lon)
        if area > 1.0:  # Roughly 111km x 111km at equator
            raise HTTPException(400, "Bounding box too large - maximum 1 square degree")
        
        bbox = BoundingBox(min_lat, min_lon, max_lat, max_lon)
        options = SpatialQueryOptions(
            limit=limit,
            offset=offset,
            include_geometry=include_geometry,
            simplify_tolerance=simplify_tolerance
        )
        
        result = await service.query_roads_in_bbox(bbox, options)
        
        return APIResponse(
            data=result,
            message=f"Found {result['count']} roads (total: {result['total']})"
        )
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Bbox query failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/v1/roads/{road_id}/connected",
         response_model=APIResponse,
         summary="Find connected roads",
         description="Find roads connected to a specific road segment")
async def get_connected_roads(
    road_id: str = Path(..., description="Road segment ID"),
    max_distance_km: float = Query(5.0, ge=0.1, le=50.0, 
                                  description="Maximum search distance in kilometers"),
    service: RoadNetworkService = Depends(get_road_service)
):
    """Find roads connected to a given road segment."""
    
    try:
        connected_roads = await service.find_connected_roads(road_id, max_distance_km)
        
        return APIResponse(
            data={
                'road_id': road_id,
                'connected_roads': connected_roads,
                'count': len(connected_roads),
                'max_distance_km': max_distance_km
            },
            message=f"Found {len(connected_roads)} connected roads"
        )
        
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Connected roads query failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.patch("/api/v1/roads/{road_id}",
           response_model=APIResponse,
           summary="Update road metadata",
           description="Update road segment metadata and attributes")
async def update_road(
    road_id: str = Path(..., description="Road segment ID"),
    updates: RoadUpdateRequest = ...,
    service: RoadNetworkService = Depends(get_road_service)
):
    """Update road segment metadata with validation and audit trail."""
    
    try:
        # Convert to dict, excluding None values
        update_data = {k: v for k, v in updates.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(400, "No valid updates provided")
        
        result = await service.update_road_metadata(road_id, update_data)
        
        return APIResponse(
            data=result,
            message="Road updated successfully"
        )
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Road update failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/v1/roads/stream/bbox",
         summary="Stream roads in bounding box",
         description="Stream large result sets efficiently")
async def stream_roads_bbox(
    min_lat: float = Query(..., ge=-90, le=90),
    min_lon: float = Query(..., ge=-180, le=180),
    max_lat: float = Query(..., ge=-90, le=90),
    max_lon: float = Query(..., ge=-180, le=180),
    batch_size: int = Query(1000, ge=100, le=5000),
    service: RoadNetworkService = Depends(get_road_service)
):
    """Stream roads for large datasets using NDJSON format."""
    
    try:
        bbox = BoundingBox(min_lat, min_lon, max_lat, max_lon)
        
        async def generate_stream():
            yield '{"type": "stream_start", "bbox": ' + json.dumps([min_lon, min_lat, max_lon, max_lat]) + '}\n'
            
            batch_count = 0
            feature_count = 0
            
            async for batch in service.stream_roads_in_region(bbox, batch_size):
                batch_count += 1
                feature_count += len(batch)
                
                for feature in batch:
                    yield json.dumps(feature) + '\n'
                
                # Progress indicator every 10 batches
                if batch_count % 10 == 0:
                    yield f'{{"type": "progress", "batches": {batch_count}, "features": {feature_count}}}\n'
            
            yield f'{{"type": "stream_end", "total_features": {feature_count}, "total_batches": {batch_count}}}\n'
        
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f"attachment; filename=roads_{min_lat}_{min_lon}_{max_lat}_{max_lon}.ndjson"
            }
        )
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Streaming query failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/health",
         summary="Health check",
         description="Service health status")
async def health_check(service: RoadNetworkService = Depends(get_road_service)):
    """Comprehensive health check endpoint."""
    
    health_status = await service.check_system_health()
    
    status_code = 200 if health_status['overall_status'] == 'healthy' else 503
    
    return APIResponse(
        data=health_status,
        message=f"Service is {health_status['overall_status']}"
    ), status_code

@app.get("/metrics",
         summary="Prometheus metrics",
         description="Metrics endpoint for monitoring")
async def get_metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Application lifecycle
@app.on_event("startup")
async def startup_event():
    """Initialize application resources."""
    logger.info("Starting Road Network Intelligence Service")
    
    # Initialize database pool
    # Initialize cache manager  
    # Initialize metrics collector
    # Run health checks
    
    logger.info("Service startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup application resources."""
    logger.info("Shutting down Road Network Intelligence Service")
    
    # Close database connections
    # Close cache connections
    # Flush metrics
    
    logger.info("Service shutdown completed")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

## Performance Engineering and Optimization

### Database Optimization

```sql
-- Comprehensive indexing strategy for spatial queries
CREATE INDEX CONCURRENTLY idx_road_segments_spatial_optimized 
ON road_segments USING GIST (geometry) 
WHERE is_active = true;

CREATE INDEX CONCURRENTLY idx_road_segments_composite_lookup
ON road_segments (road_type, traffic_level, is_active, created_at)
WHERE is_active = true;

-- Materialized view for frequently accessed road statistics
CREATE MATERIALIZED VIEW road_network_stats AS
SELECT 
    road_type,
    COUNT(*) as segment_count,
    AVG(length_meters) as avg_length,
    SUM(length_meters) as total_length,
    ST_Extent(geometry) as overall_bounds,
    AVG(confidence_score) as avg_confidence
FROM road_segments 
WHERE is_active = true 
GROUP BY road_type;

CREATE UNIQUE INDEX ON road_network_stats (road_type);

-- Optimized query for bbox searches with explain analyze
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT id, name, road_type, ST_AsGeoJSON(geometry) as geometry
FROM road_segments
WHERE is_active = true
  AND geometry && ST_MakeEnvelope(-122.5, 37.7, -122.4, 37.8, 4326)
  AND ST_Intersects(geometry, ST_MakeEnvelope(-122.5, 37.7, -122.4, 37.8, 4326))
ORDER BY road_type, name
LIMIT 1000;

-- Partitioning strategy for large datasets
CREATE TABLE road_segments_partitioned (
    LIKE road_segments INCLUDING ALL
) PARTITION BY RANGE (created_at);

CREATE TABLE road_segments_2024_q1 PARTITION OF road_segments_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE road_segments_2024_q2 PARTITION OF road_segments_partitioned  
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
```

### Caching Strategy Implementation

```python
class RoadNetworkCacheManager:
    """Specialized caching for road network data."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_cache = {}
        self.cache_stats = defaultdict(int)
    
    async def cache_road_segment(self, road_id: str, road_data: Dict, 
                               ttl_seconds: int = 3600):
        """Cache individual road segment with spatial indexing."""
        
        # Store in Redis with spatial information
        cache_key = f"road:{road_id}"
        
        # Add spatial metadata for cache invalidation
        if 'geometry' in road_data:
            geom = shape(road_data['geometry'])
            bbox = geom.bounds
            
            # Store road in spatial grid cells for invalidation
            grid_cells = self._get_grid_cells(bbox)
            
            pipeline = self.redis.pipeline()
            
            # Store road data
            pipeline.setex(cache_key, ttl_seconds, json.dumps(road_data))
            
            # Add to spatial grid cells
            for cell_id in grid_cells:
                pipeline.sadd(f"grid:{cell_id}", road_id)
                pipeline.expire(f"grid:{cell_id}", ttl_seconds + 300)
            
            await pipeline.execute()
    
    async def invalidate_spatial_region(self, bbox: Tuple[float, float, float, float]):
        """Invalidate cache for all roads in a spatial region."""
        
        grid_cells = self._get_grid_cells(bbox)
        roads_to_invalidate = set()
        
        # Collect all roads in affected grid cells
        for cell_id in grid_cells:
            cell_roads = await self.redis.smembers(f"grid:{cell_id}")
            roads_to_invalidate.update(cell_roads)
        
        # Remove from cache
        if roads_to_invalidate:
            cache_keys = [f"road:{road_id}" for road_id in roads_to_invalidate]
            await self.redis.delete(*cache_keys)
            
            self.cache_stats['invalidated'] += len(cache_keys)
    
    def _get_grid_cells(self, bbox: Tuple[float, float, float, float], 
                       grid_size: float = 0.01) -> List[str]:
        """Calculate grid cells that intersect with bounding box."""
        min_x, min_y, max_x, max_y = bbox
        
        cells = []
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                cell_id = f"{int(x/grid_size)}:{int(y/grid_size)}"
                cells.append(cell_id)
                y += grid_size
            x += grid_size
        
        return cells

class ConnectionPoolManager:
    """Optimized database connection management."""
    
    def __init__(self):
        self.pools = {}
        self.pool_stats = defaultdict(int)
    
    async def create_pool(self, database_url: str, pool_name: str = "default"):
        """Create optimized connection pool for spatial queries."""
        
        pool = await asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=30,
            server_settings={
                'jit': 'off',  # Disable JIT for consistent performance
                'application_name': f'road_network_service_{pool_name}',
                'shared_preload_libraries': 'pg_stat_statements,auto_explain',
                'auto_explain.log_min_duration': '1000ms',
                'auto_explain.log_analyze': 'true'
            }
        )
        
        self.pools[pool_name] = pool
        return pool
    
    @asynccontextmanager
    async def get_connection(self, pool_name: str = "default"):
        """Get connection with automatic retry and monitoring."""
        
        if pool_name not in self.pools:
            raise ValueError(f"Pool {pool_name} not found")
        
        pool = self.pools[pool_name]
        start_time = time.time()
        
        try:
            async with pool.acquire() as conn:
                # Set spatial query optimizations
                await conn.execute("SET enable_seqscan = false")
                await conn.execute("SET work_mem = '256MB'")
                await conn.execute("SET effective_cache_size = '4GB'")
                
                self.pool_stats[f'{pool_name}_acquired'] += 1
                yield conn
                
        except Exception as e:
            self.pool_stats[f'{pool_name}_errors'] += 1
            raise
        finally:
            acquisition_time = time.time() - start_time
            self.pool_stats[f'{pool_name}_avg_acquisition_time'] = (
                self.pool_stats.get(f'{pool_name}_avg_acquisition_time', 0) * 0.9 +
                acquisition_time * 0.1
            )
```

## Production Deployment and Operations

### Docker Configuration

```dockerfile
# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Install system dependencies for spatial libraries
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV GEOS_CONFIG=/usr/bin/geos-config

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgdal28 \
    libgeos-c1v5 \
    libproj19 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY alembic.ini .
COPY alembic/ ./alembic/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "src.day07_mock.api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: road-network-service
  labels:
    app: road-network-service
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: road-network-service
  template:
    metadata:
      labels:
        app: road-network-service
        version: v1.0.0
    spec:
      containers:
      - name: api
        image: road-network-service:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: road-network-config
---
apiVersion: v1
kind: Service
metadata:
  name: road-network-service
spec:
  selector:
    app: road-network-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: road-network-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.roadnetwork.company.com
    secretName: tls-secret
  rules:
  - host: api.roadnetwork.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: road-network-service
            port:
              number: 80
```

## Testing Strategy and Quality Assurance

### Comprehensive Test Suite

```python
import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
import json

class TestRoadNetworkAPI:
    """Comprehensive API test suite."""
    
    @pytest.fixture
    async def client(self):
        """Test client with mocked dependencies."""
        
        # Mock database pool
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock cache manager
        mock_cache = AsyncMock()
        
        # Mock metrics collector
        mock_metrics = AsyncMock()
        
        with patch('src.day07_mock.api.get_road_service') as mock_service:
            service_instance = RoadNetworkService(mock_pool, mock_cache, mock_metrics)
            mock_service.return_value = service_instance
            
            async with AsyncClient(app=app, base_url="http://test") as ac:
                yield ac, service_instance, mock_conn
    
    @pytest.mark.asyncio
    async def test_bbox_query_success(self, client):
        """Test successful bounding box query."""
        
        ac, service, mock_conn = client
        
        # Mock database response
        mock_conn.fetch.return_value = [
            {
                'id': '123e4567-e89b-12d3-a456-426614174000',
                'external_id': 'road_001',
                'name': 'Main Street',
                'road_type': 'arterial',
                'speed_limit_kmh': 50,
                'direction': 'bidirectional',
                'traffic_level': 'medium',
                'surface_type': 'asphalt',
                'lanes_count': 2,
                'truck_restricted': False,
                'hazmat_restricted': False,
                'length_meters': 1500.0,
                'confidence_score': 85,
                'attributes': {'surface_condition': 'good'},
                'created_at': datetime(2024, 1, 1),
                'updated_at': datetime(2024, 1, 1),
                'geometry': '{"type":"LineString","coordinates":[[-122.4,37.7],[-122.39,37.71]]}'
            }
        ]
        
        mock_conn.fetchval.return_value = 1  # Total count
        
        response = await ac.get("/api/v1/roads/bbox", params={
            'min_lat': 37.7,
            'min_lon': -122.4,
            'max_lat': 37.8,
            'max_lon': -122.3,
            'limit': 100
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert 'data' in data
        assert data['data']['type'] == 'FeatureCollection'
        assert len(data['data']['features']) == 1
        
        feature = data['data']['features'][0]
        assert feature['type'] == 'Feature'
        assert feature['properties']['name'] == 'Main Street'
        assert feature['properties']['road_type'] == 'arterial'
        assert 'geometry' in feature
    
    @pytest.mark.asyncio
    async def test_bbox_query_validation_errors(self, client):
        """Test bounding box validation errors."""
        
        ac, _, _ = client
        
        # Invalid latitude range
        response = await ac.get("/api/v1/roads/bbox", params={
            'min_lat': 91,  # Invalid
            'min_lon': -122.4,
            'max_lat': 37.8,
            'max_lon': -122.3
        })
        assert response.status_code == 422
        
        # Invalid bounding box (min >= max)
        response = await ac.get("/api/v1/roads/bbox", params={
            'min_lat': 37.8,
            'min_lon': -122.3,
            'max_lat': 37.7,  # Less than min_lat
            'max_lon': -122.4
        })
        assert response.status_code == 400
        
        # Bounding box too large
        response = await ac.get("/api/v1/roads/bbox", params={
            'min_lat': 37.0,
            'min_lon': -123.0,
            'max_lat': 38.5,  # More than 1 degree difference
            'max_lon': -121.5
        })
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_connected_roads_query(self, client):
        """Test connected roads functionality."""
        
        ac, service, mock_conn = client
        
        road_id = '123e4567-e89b-12d3-a456-426614174000'
        
        # Mock road exists check
        mock_conn.fetchrow.return_value = {
            'id': road_id,
            'start_node_id': 'node_1',
            'end_node_id': 'node_2',
            'geometry': '{"type":"LineString","coordinates":[[-122.4,37.7],[-122.39,37.71]]}'
        }
        
        # Mock connected roads
        mock_conn.fetch.return_value = [
            {'id': 'connected_road_1'},
            {'id': 'connected_road_2'}
        ]
        
        response = await ac.get(f"/api/v1/roads/{road_id}/connected")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert data['data']['road_id'] == road_id
        assert len(data['data']['connected_roads']) == 2
        assert 'connected_road_1' in data['data']['connected_roads']
    
    @pytest.mark.asyncio
    async def test_road_update_success(self, client):
        """Test successful road update."""
        
        ac, service, mock_conn = client
        
        road_id = '123e4567-e89b-12d3-a456-426614174000'
        
        # Mock current version check
        mock_conn.fetchrow.side_effect = [
            {'version': 1},  # Current version
            {  # Update result
                'id': road_id,
                'version': 2,
                'updated_at': datetime(2024, 1, 2)
            }
        ]
        
        update_data = {
            'speed_limit_kmh': 60,
            'traffic_level': 'high'
        }
        
        response = await ac.patch(f"/api/v1/roads/{road_id}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert data['data']['id'] == road_id
        assert data['data']['version'] == 2
    
    @pytest.mark.asyncio
    async def test_road_update_validation_errors(self, client):
        """Test road update validation."""
        
        ac, _, _ = client
        
        road_id = '123e4567-e89b-12d3-a456-426614174000'
        
        # Invalid speed limit
        response = await ac.patch(f"/api/v1/roads/{road_id}", json={
            'speed_limit_kmh': 250  # Too high
        })
        assert response.status_code == 422
        
        # Invalid traffic level
        response = await ac.patch(f"/api/v1/roads/{road_id}", json={
            'traffic_level': 'invalid_level'
        })
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint."""
        
        ac, service, _ = client
        
        # Mock health check response
        with patch.object(service, 'check_system_health') as mock_health:
            mock_health.return_value = {
                'overall_status': 'healthy',
                'components': {
                    'database': {'status': 'healthy'},
                    'cache': {'status': 'healthy'}
                }
            }
            
            response = await ac.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is True
            assert data['data']['overall_status'] == 'healthy'

class TestSpatialQueries:
    """Spatial query performance and correctness tests."""
    
    @pytest.mark.asyncio
    async def test_large_bbox_query_performance(self):
        """Test performance with large bounding box queries."""
        
        # This would use a real test database with sample data
        service = await create_test_service()
        
        bbox = BoundingBox(37.0, -122.5, 38.0, -121.5)
        options = SpatialQueryOptions(limit=5000)
        
        start_time = time.time()
        result = await service.query_roads_in_bbox(bbox, options)
        duration = time.time() - start_time
        
        # Performance assertion
        assert duration < 1.0  # Should complete within 1 second
        assert result['count'] <= 5000
        assert 'features' in result
    
    @pytest.mark.asyncio  
    async def test_spatial_accuracy(self):
        """Test spatial query accuracy and edge cases."""
        
        service = await create_test_service()
        
        # Test exact boundary conditions
        bbox = BoundingBox(37.7749, -122.4194, 37.7750, -122.4193)
        result = await service.query_roads_in_bbox(bbox, SpatialQueryOptions())
        
        # Verify all returned roads actually intersect the bbox
        for feature in result['features']:
            geom = shape(feature['geometry'])
            bbox_geom = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
            assert geom.intersects(bbox_geom)

# Load testing with locust
class RoadNetworkLoadTest(HttpUser):
    """Load testing scenarios for road network API."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup for load test user."""
        self.sf_bbox = {
            'min_lat': 37.7,
            'min_lon': -122.5,
            'max_lat': 37.8,
            'max_lon': -122.4
        }
    
    @task(3)
    def query_small_bbox(self):
        """Simulate small bounding box queries (most common)."""
        
        # Random small area in San Francisco
        center_lat = random.uniform(37.75, 37.78)
        center_lon = random.uniform(-122.45, -122.42)
        
        params = {
            'min_lat': center_lat - 0.001,
            'min_lon': center_lon - 0.001,
            'max_lat': center_lat + 0.001,
            'max_lon': center_lon + 0.001,
            'limit': 100
        }
        
        with self.client.get("/api/v1/roads/bbox", params=params) as response:
            if response.status_code != 200:
                response.failure(f"Expected 200, got {response.status_code}")
    
    @task(1)
    def query_connected_roads(self):
        """Simulate connected roads queries."""
        
        # Use known road ID for testing
        road_id = "test_road_123"
        
        with self.client.get(f"/api/v1/roads/{road_id}/connected") as response:
            if response.status_code not in [200, 404]:  # 404 acceptable for missing roads
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Simulate health checks."""
        
        with self.client.get("/health") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
```

## Professional Development Assessment

### Capstone Evaluation Criteria

**Technical Excellence (40%):**
- Code quality, organization, and documentation
- Proper use of design patterns and architectural principles
- Performance optimization and scalability considerations
- Error handling and edge case management

**API Design (25%):**
- RESTful design principles and consistency
- Request/response validation and error messages
- OpenAPI documentation completeness
- Proper HTTP status codes and headers

**Spatial Functionality (20%):**
- Accuracy of spatial queries and operations
- Efficient use of spatial indexes and databases
- Proper coordinate system handling
- Network topology analysis implementation

**Production Readiness (15%):**
- Logging, monitoring, and observability
- Configuration management and environment handling
- Security considerations and input validation
- Testing coverage and quality

### Deployment and Demonstration

**Live Demo Requirements:**
1. **Service deployment** on cloud infrastructure (AWS/GCP/Azure)
2. **Database setup** with sample road network data
3. **Monitoring dashboard** showing real-time metrics
4. **Load testing results** demonstrating performance targets
5. **API documentation** with interactive examples

**Code Review Checklist:**
- [ ] Clean, readable, and well-documented code
- [ ] Comprehensive error handling and validation
- [ ] Efficient database queries with proper indexing
- [ ] Multi-level caching implementation
- [ ] Comprehensive test coverage (>80%)
- [ ] Production-ready logging and monitoring
- [ ] Security best practices followed
- [ ] Performance targets met under load

## Industry Context and Career Advancement

### Real-World Applications

**Logistics and Transportation:**
- **UPS ORION**: Route optimization using real-time road data
- **Uber/Lyft**: Dynamic routing and ETA calculations  
- **FedEx**: Fleet management with traffic-aware routing
- **Amazon Logistics**: Last-mile delivery optimization

**Autonomous Vehicles:**
- **Tesla Autopilot**: Real-time map updates for navigation
- **Waymo**: HD mapping for autonomous driving
- **Cruise**: Urban environment mapping and navigation

**Smart Cities:**
- **Traffic Management**: Real-time traffic optimization
- **Emergency Services**: Optimal routing for emergency vehicles
- **Urban Planning**: Infrastructure analysis and planning

### Career Progression

**Entry Level (0-2 years):**
- Junior GIS Developer
- Spatial Data Analyst  
- Backend Developer (geospatial focus)

**Mid Level (2-5 years):**
- Senior GIS Engineer
- Geospatial Software Engineer
- Location Intelligence Engineer
- Mapping Platform Developer

**Senior Level (5+ years):**
- Principal Geospatial Engineer
- Geospatial Architecture Lead
- Location Services Technical Lead
- Director of Spatial Engineering

**Specialized Roles:**
- Cartographic Engineer
- Spatial Database Administrator
- GIS Solutions Architect
- Autonomous Vehicle Mapping Engineer

## Further Learning and Resources

### Advanced Topics
- **Spatial Machine Learning**: ML algorithms for geospatial data
- **Real-time Streaming**: Apache Kafka with spatial data
- **Distributed Computing**: Spark and Dask for large-scale geospatial processing
- **Computer Vision**: Satellite imagery analysis and processing

### Industry Certifications
- **AWS Certified Solutions Architect** (with geospatial focus)
- **Google Cloud Professional Data Engineer**
- **ESRI Technical Certification**
- **Open Source Geospatial Foundation (OSGeo) Certification**

### Professional Organizations
- **Open Source Geospatial Foundation (OSGeo)**
- **Urban and Regional Information Systems Association (URISA)**
- **American Society for Photogrammetry and Remote Sensing (ASPRS)**
- **Association of Geographic Information Laboratories in Europe (AGILE)**

This capstone project demonstrates mastery of enterprise geospatial engineering principles and prepares you for senior roles in the growing field of location intelligence and spatial computing.
