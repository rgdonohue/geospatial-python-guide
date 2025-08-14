# Day 2: Object-Oriented Design and Enterprise Architecture Patterns

## Learning Objectives

This module introduces enterprise-grade software architecture patterns essential for building scalable, maintainable geospatial data processing systems. By the end of this module, you will understand:

- **Interface Segregation and Dependency Inversion** principles in distributed geospatial systems
- **Provider Pattern implementation** for multi-format data source abstraction
- **Factory and Strategy patterns** for runtime behavior selection and configuration
- **Plugin architecture design** enabling extensible data processing pipelines
- **Clean Architecture principles** for domain-driven geospatial service design
- **Enterprise integration patterns** for heterogeneous geospatial data ecosystems

## Theoretical Foundation

### SOLID Principles in Geospatial Context

**Single Responsibility Principle (SRP):**
Each provider handles exactly one data format concern, making the system easier to test, debug, and maintain.

**Open/Closed Principle (OCP):**
The provider architecture is open for extension (new data formats) but closed for modification (existing providers remain unchanged).

**Liskov Substitution Principle (LSP):**
Any MapDataProvider implementation can be substituted for another without breaking client code.

**Interface Segregation Principle (ISP):**
Small, focused interfaces prevent clients from depending on methods they don't use.

**Dependency Inversion Principle (DIP):**
High-level geospatial processing logic depends on abstractions, not concrete implementations.

### Enterprise Architecture Context

In large-scale geospatial systems, we encounter:
- **Multiple data formats**: GeoJSON, Shapefile, MVT, PostGIS, Oracle Spatial
- **Varying data sources**: Files, databases, APIs, streaming services
- **Different access patterns**: Batch processing, real-time streaming, random access
- **Evolving requirements**: New formats, changing schemas, performance optimizations

**The Provider Pattern** enables:
- **Runtime format selection**: Choose optimal format based on use case
- **Zero-downtime migrations**: Gradually migrate between data sources
- **A/B testing**: Compare performance between different implementations
- **Vendor independence**: Avoid lock-in to specific data formats or services

## Goals
- Design **enterprise-grade interfaces** with minimal coupling and maximal cohesion
- Implement **pluggable architecture** supporting runtime format selection
- Apply **dependency injection patterns** for testable, configurable systems
- Understand **clean architecture boundaries** between domain logic and infrastructure

## Core Concepts

### 1) Interface via ABC

ABCs define a contract implemented by concrete providers.

```python
from abc import ABC, abstractmethod

class MapDataProvider(ABC):
    @abstractmethod
    def get_feature_by_id(self, feature_id: str):
        """Must be implemented by subclasses."""
        pass
```

Benefits:
- Enforces consistency; enables polymorphism and easy mocking.
- Documents expectations; keeps call sites stable as implementations evolve.

### 2) Factory and Strategy

Factory creates the right provider for a given content type; Strategy lets behavior vary (e.g., coordinate transforms) without touching callers.

```python
class ProviderFactory:
    @staticmethod
    def create_provider(content_type: str) -> MapDataProvider:
        if content_type == "geojson":
            return GeoJSONProvider()
        elif content_type == "mvt":
            return MVTProvider()
        else:
            raise ValueError(f"Unknown content type: {content_type}")
```

Strategy example:

```python
class CoordinateTransformer:
    def __init__(self, strategy: TransformationStrategy):
        self.strategy = strategy
    
    def transform(self, coordinates):
        return self.strategy.transform(coordinates)
```

### 3) Package Layout

Organize for clarity and import hygiene:

```
src/day02_oop/
├── __init__.py          # Package initialization
├── providers/           # Subpackage for providers
│   ├── __init__.py     # Exports main classes
│   ├── base.py         # Abstract base class
│   ├── geojson_provider.py
│   └── mvt_provider.py
└── README.md
```

## Code Walkthrough

### 1) Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Any

class MapDataProvider(ABC):
    @abstractmethod
    def get_feature_by_id(self, feature_id: str) -> Any:
        """Retrieve a feature by its unique identifier."""
        raise NotImplementedError

    @abstractmethod
    def stream_features(self) -> Any:
        """Stream all features from the data source."""
        raise NotImplementedError
```

**Key Points:**
- `@abstractmethod` decorator marks methods that must be implemented
- `raise NotImplementedError` provides a clear error if not implemented
- Type hints help with IDE support and documentation

### 2. Concrete Implementations

#### GeoJSON Provider
```python
import json
from pathlib import Path
from typing import Any, Iterable
from .base import MapDataProvider

class GeoJSONProvider(MapDataProvider):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data = json.loads(self._path.read_text())
        # Build an index for fast lookups
        self._index = {
            f["properties"].get("id"): f 
            for f in self._data.get("features", [])
        }

    def get_feature_by_id(self, feature_id: str) -> Any:
        return self._index.get(feature_id)

    def stream_features(self) -> Iterable[dict]:
        yield from self._data.get("features", [])
```

**Features:**
- **Indexing**: Builds a lookup dictionary for O(1) feature retrieval
- **Path handling**: Uses `pathlib.Path` for cross-platform compatibility
- **Memory efficiency**: Yields features one at a time

#### MVT Provider
```python
from pathlib import Path
from typing import Any, Iterable
from .base import MapDataProvider

class MVTProvider(MapDataProvider):
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def get_feature_by_id(self, feature_id: str) -> Any:
        # MVT files don't have feature IDs - would need external index
        return None

    def stream_features(self) -> Iterable[bytes]:
        # Yield raw MVT bytes from directory
        for p in self._directory.rglob("*.mvt"):
            yield p.read_bytes()
```

**Features:**
- **Directory scanning**: Uses `rglob` to find all `.mvt` files
- **Binary handling**: Returns raw bytes for MVT data
- **Placeholder implementation**: Shows where external indexing would be needed

### 3. Package Exports

```python
# providers/__init__.py
from .base import MapDataProvider
from .geojson_provider import GeoJSONProvider
from .mvt_provider import MVTProvider

# This allows users to import directly from the package
# from src.day02_oop.providers import MapDataProvider
```

## Usage
Create and use providers via the common interface. Note: `MVTProvider.stream_features()` yields raw bytes for demo purposes; `GeoJSONProvider` and `CSVProvider` yield dict features.
```python
from src.day02_oop.providers import GeoJSONProvider, MVTProvider, CSVProvider

# Create providers
geojson_provider = GeoJSONProvider("data/roads.geojson")
mvt_provider = MVTProvider("data/tiles/")
csv_provider = CSVProvider("src/day02_oop/providers/data/roads_sample.csv")

# Use polymorphically
for provider in [geojson_provider, mvt_provider, csv_provider]:
    features = list(provider.stream_features())
    print(f"Provider {type(provider).__name__}: {len(features)} features")
```

### Testing the Interface
```python
def test_provider_interface(provider: MapDataProvider):
    """Test that any provider implements the required interface."""
    # This will work with any MapDataProvider subclass
    features = list(provider.stream_features())
    assert len(features) >= 0
    
    # Test feature lookup if supported
    if provider.get_feature_by_id("test_id") is not None:
        print("Provider supports feature lookup")
```

## Exercises

### 1) Implement a New Provider

Create a `CSVProvider` that reads from CSV files:

```python
import csv
from pathlib import Path
from typing import Any, Iterable
from .base import MapDataProvider

class CSVProvider(MapDataProvider):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        # TODO: Implement CSV reading logic
    
    def get_feature_by_id(self, feature_id: str) -> Any:
        # TODO: Implement CSV lookup
        pass
    
    def stream_features(self) -> Iterable[dict]:
        # TODO: Implement CSV streaming
        pass
```

### 2) Add a Factory Method

```python
class ProviderFactory:
    _providers = {
        "geojson": GeoJSONProvider,
        "mvt": MVTProvider,
        "csv": CSVProvider,  # Add your new provider
    }
    
    @classmethod
    def create_provider(cls, content_type: str, **kwargs) -> MapDataProvider:
        if content_type not in cls._providers:
            raise ValueError(f"Unknown provider type: {content_type}")
        
        provider_class = cls._providers[content_type]
        return provider_class(**kwargs)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Allow runtime registration of new providers."""
        cls._providers[name] = provider_class

Or use the built-in minimal factory:
```python
from src.day02_oop.providers import ProviderFactory

csv = ProviderFactory.create("csv", path="src/day02_oop/providers/data/roads_sample.csv")
geojson = ProviderFactory.create("geojson", path="data/roads.geojson")
```
```

### 3) Add Caching for Hot Paths

```python
from functools import lru_cache

class CachedGeoJSONProvider(GeoJSONProvider):
    @lru_cache(maxsize=1000)
    def get_feature_by_id(self, feature_id: str) -> Any:
        return super().get_feature_by_id(feature_id)
```

## Advanced

### 1) Protocols (Structural Typing)

Instead of ABCs, you can use protocols for structural typing:

```python
from typing import Protocol

class MapDataProvider(Protocol):
    def get_feature_by_id(self, feature_id: str) -> Any: ...
    def stream_features(self) -> Iterable[Any]: ...

# Any class with these methods automatically implements the protocol
```

### 2) Generics for Type Safety

Make providers more type-safe with generics:

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class MapDataProvider(ABC, Generic[T]):
    @abstractmethod
    def get_feature_by_id(self, feature_id: str) -> T | None:
        pass
    
    @abstractmethod
    def stream_features(self) -> Iterable[T]:
        pass

class GeoJSONProvider(MapDataProvider[dict]):
    # Now the return types are properly typed
    pass
```

### 3) Context Managers

Add resource management to providers:

```python
class ManagedGeoJSONProvider(GeoJSONProvider):
    def __enter__(self):
        # Could open file handles, database connections, etc.
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources
        pass
```

## Best Practices

### 1) Interface Design
- Keep interfaces small and focused
- Use descriptive method names
- Document expected behavior clearly
- Consider backward compatibility

### 2) Error Handling
```python
class GeoJSONProvider(MapDataProvider):
    def get_feature_by_id(self, feature_id: str) -> Any:
        if not hasattr(self, '_index'):
            raise RuntimeError("Provider not properly initialized")
        
        if not feature_id:
            raise ValueError("Feature ID cannot be empty")
        
        return self._index.get(feature_id)
```

### 3) Testing
```python
import pytest
from unittest.mock import Mock

def test_provider_contract():
    """Test that providers follow the contract."""
    mock_provider = Mock(spec=MapDataProvider)
    
    # These should work without errors
    mock_provider.get_feature_by_id("test")
    list(mock_provider.stream_features())
```

## Common Pitfalls

### 1) Over-Abstraction
```python
# ❌ Too many abstract methods
class MapDataProvider(ABC):
    @abstractmethod
    def get_feature_by_id(self, feature_id: str): pass
    @abstractmethod
    def stream_features(self): pass
    @abstractmethod
    def get_feature_count(self): pass  # Maybe not needed?
    @abstractmethod
    def get_feature_bounds(self): pass  # Maybe not needed?

# ✅ Focus on core functionality
class MapDataProvider(ABC):
    @abstractmethod
    def get_feature_by_id(self, feature_id: str): pass
    @abstractmethod
    def stream_features(self): pass
```

### 2) Ignoring Resource Management
```python
# ❌ File handles not properly closed
def stream_features(self):
    with open(self._path) as f:
        return json.load(f)["features"]

# ✅ Proper resource management
def stream_features(self):
    with open(self._path) as f:
        data = json.load(f)
        yield from data["features"]
```

## Next Steps

- Decide ABC vs Protocol for your team’s typing philosophy.
- Implement and test `CSVProvider` or a PostGIS-backed provider.
- Add a simple factory and registration mechanism.
- Validate/normalize feature schemas at the boundary.

## Advanced Enterprise Patterns

### Hexagonal Architecture Implementation

```python
# Domain Layer - Core business logic
class SpatialQuery:
    def __init__(self, bbox: BoundingBox, filters: Dict[str, Any]):
        self.bbox = bbox
        self.filters = filters

# Application Layer - Use cases
class GeospatialService:
    def __init__(self, data_provider: MapDataProvider, 
                 spatial_index: SpatialIndex):
        self._provider = data_provider
        self._index = spatial_index
    
    async def query_features(self, query: SpatialQuery) -> FeatureCollection:
        # Pure business logic, no infrastructure concerns
        candidate_ids = self._index.query_bbox(query.bbox)
        features = []
        for feature_id in candidate_ids:
            feature = await self._provider.get_feature_by_id(feature_id)
            if self._matches_filters(feature, query.filters):
                features.append(feature)
        return FeatureCollection(features)

# Infrastructure Layer - External concerns
class PostGISProvider(MapDataProvider):
    def __init__(self, connection_pool: asyncpg.Pool):
        self._pool = connection_pool
```

### Plugin Architecture with Registry

```python
class ProviderRegistry:
    """Registry for managing provider plugins at runtime."""
    
    def __init__(self):
        self._providers: Dict[str, Type[MapDataProvider]] = {}
        self._configurations: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, provider_class: Type[MapDataProvider], 
                 config: Dict[str, Any] = None):
        """Register a provider with optional configuration."""
        self._providers[name] = provider_class
        self._configurations[name] = config or {}
    
    def create(self, name: str, **override_config) -> MapDataProvider:
        """Create provider instance with configuration."""
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}")
        
        provider_class = self._providers[name]
        config = {**self._configurations[name], **override_config}
        return provider_class(**config)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

# Usage
registry = ProviderRegistry()
registry.register("geojson", GeoJSONProvider, {"buffer_size": 8192})
registry.register("postgis", PostGISProvider, {"pool_size": 20})

# Runtime provider selection
provider = registry.create("geojson", path="/data/roads.geojson")
```

### Configuration-Driven Architecture

```python
@dataclass
class ProviderConfig:
    """Type-safe configuration for providers."""
    provider_type: str
    connection_string: Optional[str] = None
    file_path: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    
    def __post_init__(self):
        if self.provider_type in ["postgis", "oracle"] and not self.connection_string:
            raise ValueError(f"{self.provider_type} requires connection_string")
        if self.provider_type in ["geojson", "shapefile"] and not self.file_path:
            raise ValueError(f"{self.provider_type} requires file_path")

class ConfigurableProviderFactory:
    """Factory that creates providers from configuration."""
    
    @staticmethod
    def from_config(config: ProviderConfig) -> MapDataProvider:
        if config.provider_type == "geojson":
            provider = GeoJSONProvider(config.file_path)
        elif config.provider_type == "postgis":
            provider = PostGISProvider(config.connection_string)
        else:
            raise ValueError(f"Unsupported provider: {config.provider_type}")
        
        if config.cache_enabled:
            provider = CachedProvider(provider, ttl=config.cache_ttl_seconds)
        
        if config.retry_attempts > 0:
            provider = RetryingProvider(provider, max_attempts=config.retry_attempts)
        
        return provider
```

### Decorator Pattern for Cross-Cutting Concerns

```python
class CachedProvider(MapDataProvider):
    """Decorator that adds caching to any provider."""
    
    def __init__(self, wrapped_provider: MapDataProvider, 
                 cache: Optional[Dict] = None, ttl: int = 300):
        self._provider = wrapped_provider
        self._cache = cache or {}
        self._ttl = ttl
        self._timestamps = {}
    
    def get_feature_by_id(self, feature_id: str) -> Any:
        now = time.time()
        
        # Check cache validity
        if (feature_id in self._cache and 
            feature_id in self._timestamps and 
            now - self._timestamps[feature_id] < self._ttl):
            return self._cache[feature_id]
        
        # Fetch from wrapped provider
        feature = self._provider.get_feature_by_id(feature_id)
        
        # Update cache
        self._cache[feature_id] = feature
        self._timestamps[feature_id] = now
        
        return feature

class MetricsProvider(MapDataProvider):
    """Decorator that adds metrics collection."""
    
    def __init__(self, wrapped_provider: MapDataProvider, 
                 metrics_collector: 'MetricsCollector'):
        self._provider = wrapped_provider
        self._metrics = metrics_collector
    
    async def get_feature_by_id(self, feature_id: str) -> Any:
        start_time = time.time()
        try:
            result = await self._provider.get_feature_by_id(feature_id)
            self._metrics.record_success("get_feature_by_id", time.time() - start_time)
            return result
        except Exception as e:
            self._metrics.record_error("get_feature_by_id", str(e))
            raise

# Usage: Composable decorators
base_provider = GeoJSONProvider("data.geojson")
cached_provider = CachedProvider(base_provider, ttl=600)
monitored_provider = MetricsProvider(cached_provider, metrics_collector)
```

### Command Pattern for Data Operations

```python
class DataOperation(ABC):
    """Abstract command for data operations."""
    
    @abstractmethod
    async def execute(self) -> Any:
        pass
    
    @abstractmethod
    async def undo(self) -> None:
        pass

class BulkInsertOperation(DataOperation):
    def __init__(self, provider: MapDataProvider, features: List[Dict]):
        self._provider = provider
        self._features = features
        self._inserted_ids = []
    
    async def execute(self) -> List[str]:
        for feature in self._features:
            feature_id = await self._provider.insert_feature(feature)
            self._inserted_ids.append(feature_id)
        return self._inserted_ids
    
    async def undo(self) -> None:
        for feature_id in reversed(self._inserted_ids):
            await self._provider.delete_feature(feature_id)

class DataOperationExecutor:
    """Executor that manages operations with undo capability."""
    
    def __init__(self):
        self._history: List[DataOperation] = []
    
    async def execute(self, operation: DataOperation) -> Any:
        result = await operation.execute()
        self._history.append(operation)
        return result
    
    async def undo_last(self) -> None:
        if self._history:
            operation = self._history.pop()
            await operation.undo()
```

### Event-Driven Architecture

```python
class DataProviderEvent:
    """Base class for provider events."""
    def __init__(self, provider_name: str, timestamp: datetime):
        self.provider_name = provider_name
        self.timestamp = timestamp

class FeatureUpdatedEvent(DataProviderEvent):
    def __init__(self, provider_name: str, feature_id: str, 
                 old_feature: Dict, new_feature: Dict):
        super().__init__(provider_name, datetime.now())
        self.feature_id = feature_id
        self.old_feature = old_feature
        self.new_feature = new_feature

class EventDrivenProvider(MapDataProvider):
    """Provider that emits events for all operations."""
    
    def __init__(self, wrapped_provider: MapDataProvider, 
                 event_bus: 'EventBus'):
        self._provider = wrapped_provider
        self._event_bus = event_bus
    
    async def update_feature(self, feature_id: str, updates: Dict) -> Dict:
        old_feature = await self._provider.get_feature_by_id(feature_id)
        new_feature = await self._provider.update_feature(feature_id, updates)
        
        event = FeatureUpdatedEvent(
            provider_name=self.__class__.__name__,
            feature_id=feature_id,
            old_feature=old_feature,
            new_feature=new_feature
        )
        await self._event_bus.publish(event)
        
        return new_feature
```

## Industry Integration Patterns

### Multi-Tenant Architecture

```python
class TenantAwareProvider(MapDataProvider):
    """Provider that implements multi-tenancy."""
    
    def __init__(self, base_provider: MapDataProvider, 
                 tenant_resolver: 'TenantResolver'):
        self._base_provider = base_provider
        self._tenant_resolver = tenant_resolver
    
    async def get_feature_by_id(self, feature_id: str) -> Any:
        tenant_id = self._tenant_resolver.get_current_tenant()
        scoped_feature_id = f"{tenant_id}:{feature_id}"
        return await self._base_provider.get_feature_by_id(scoped_feature_id)
    
    async def stream_features(self) -> AsyncGenerator[Dict, None]:
        tenant_id = self._tenant_resolver.get_current_tenant()
        async for feature in self._base_provider.stream_features():
            if self._belongs_to_tenant(feature, tenant_id):
                yield feature
```

### Data Lineage and Audit Trail

```python
class AuditableProvider(MapDataProvider):
    """Provider that maintains audit trail."""
    
    def __init__(self, wrapped_provider: MapDataProvider, 
                 audit_store: 'AuditStore'):
        self._provider = wrapped_provider
        self._audit_store = audit_store
    
    async def update_feature(self, feature_id: str, updates: Dict) -> Dict:
        # Record audit entry
        audit_entry = AuditEntry(
            operation="UPDATE",
            feature_id=feature_id,
            changes=updates,
            user_id=self._get_current_user(),
            timestamp=datetime.now()
        )
        
        result = await self._provider.update_feature(feature_id, updates)
        await self._audit_store.record(audit_entry)
        return result
```

### Geospatial-Specific Enterprise Patterns

```python
class CoordinateSystemAwareProvider(MapDataProvider):
    """Provider that handles coordinate system transformations."""
    
    def __init__(self, base_provider: MapDataProvider, 
                 source_crs: int, target_crs: int):
        self._provider = base_provider
        self._transformer = Transformer.from_crs(source_crs, target_crs)
    
    async def get_feature_by_id(self, feature_id: str) -> Any:
        feature = await self._provider.get_feature_by_id(feature_id)
        return self._transform_feature(feature)
    
    def _transform_feature(self, feature: Dict) -> Dict:
        geometry = feature["geometry"]
        if geometry["type"] == "Point":
            coords = geometry["coordinates"]
            x, y = self._transformer.transform(coords[0], coords[1])
            geometry["coordinates"] = [x, y]
        # Handle other geometry types...
        return feature

class SpatialIndexedProvider(MapDataProvider):
    """Provider with built-in spatial indexing."""
    
    def __init__(self, base_provider: MapDataProvider):
        self._provider = base_provider
        self._spatial_index = rtree.index.Index()
        self._features_cache = {}
        self._build_index()
    
    async def query_bbox(self, min_x: float, min_y: float, 
                        max_x: float, max_y: float) -> List[Dict]:
        """Efficient bounding box query using spatial index."""
        candidate_ids = list(self._spatial_index.intersection(
            (min_x, min_y, max_x, max_y)
        ))
        
        features = []
        for feature_id in candidate_ids:
            if feature_id in self._features_cache:
                feature = self._features_cache[feature_id]
                if self._intersects_bbox(feature, min_x, min_y, max_x, max_y):
                    features.append(feature)
        
        return features
```

## Testing Enterprise Architectures

### Contract Testing

```python
class ProviderContractTest:
    """Base test class that validates provider contracts."""
    
    def test_feature_retrieval_contract(self, provider: MapDataProvider):
        """Test that provider correctly implements feature retrieval."""
        # Arrange
        test_feature_id = "test_feature_123"
        
        # Act
        feature = provider.get_feature_by_id(test_feature_id)
        
        # Assert
        if feature is not None:
            assert "properties" in feature
            assert "geometry" in feature
            assert feature["type"] == "Feature"
    
    def test_streaming_contract(self, provider: MapDataProvider):
        """Test that provider correctly implements streaming."""
        feature_count = 0
        for feature in provider.stream_features():
            feature_count += 1
            assert isinstance(feature, dict)
            if feature_count >= 10:  # Test first 10 features
                break
        
        assert feature_count > 0  # Should have at least some features

# Test different providers against the same contract
class TestGeoJSONProvider(ProviderContractTest):
    @pytest.fixture
    def provider(self):
        return GeoJSONProvider("test_data.geojson")

class TestPostGISProvider(ProviderContractTest):
    @pytest.fixture
    def provider(self):
        return PostGISProvider("postgresql://test:test@localhost/testdb")
```

## Professional Development Exercises

### Exercise 1: Design a Multi-Source Data Aggregator
Create a provider that aggregates data from multiple sources:
- Combine real-time API data with static file data
- Handle inconsistent schemas across sources
- Implement conflict resolution strategies
- Add performance monitoring and caching

### Exercise 2: Build a Plugin Ecosystem
Design a plugin system for custom data transformations:
- Create a transformation plugin interface
- Implement a plugin discovery mechanism
- Add configuration management for plugins
- Create sample plugins for common transformations

### Exercise 3: Implement Event Sourcing
Design an event-sourced provider that:
- Captures all data mutations as events
- Rebuilds state from event history
- Supports point-in-time queries
- Implements event replay for debugging

### Exercise 4: Create a Multi-Tenant GIS Service
Build a multi-tenant spatial data service:
- Implement tenant isolation at the data layer
- Add role-based access control
- Create tenant-specific configuration
- Implement cross-tenant analytics (where permitted)

## Industry Context and Real-World Applications

### Enterprise GIS Systems
- **ESRI ArcGIS Enterprise**: Multi-tier architecture with service-oriented design
- **Oracle Spatial**: Database-integrated spatial processing with pluggable engines
- **PostGIS/PostgreSQL**: Open-source spatial database with extension architecture

### Cloud-Native Geospatial Services
- **AWS Location Services**: Managed geospatial services with provider abstractions
- **Google Maps Platform**: API-driven architecture with multiple data sources
- **Microsoft Azure Maps**: Cloud-first design with hybrid deployment options

### Microservices Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Ingestion│    │  Spatial Query  │    │  Visualization  │
│   Microservice  │    │  Microservice   │    │  Microservice   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│• Provider Fabric│    │• Query Engine   │    │• Renderer       │
│• ETL Pipeline   │    │• Cache Layer    │    │• Style Engine   │
│• Data Validation│    │• Index Manager  │    │• Export Service │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Provider       │
                    │  Registry       │
                    │  Service        │
                    └─────────────────┘
```

## Resources

### Books and Publications
- **Clean Architecture by Robert C. Martin**: Fundamental principles of software architecture
- **Patterns of Enterprise Application Architecture by Martin Fowler**: Essential enterprise patterns
- **Domain-Driven Design by Eric Evans**: Domain modeling and bounded contexts
- **Building Microservices by Sam Newman**: Microservices architecture patterns

### Technical References
- [Python ABCs Documentation](https://docs.python.org/3/library/abc.html)
- [Protocol-Based Typing](https://typing.readthedocs.io/en/latest/spec/protocol.html)
- [Python Packaging Guide](https://packaging.python.org/)
- [SOLID Principles in Python](https://realpython.com/solid-principles-python/)

### Industry Standards
- [OGC Web Feature Service (WFS)](https://www.ogc.org/standards/wfs)
- [GeoPackage Specification](https://www.geopackage.org/)
- [Cloud Optimized GeoTIFF](https://www.cogeo.org/)

This module establishes the architectural foundation for building enterprise-grade geospatial systems. The patterns and principles learned here scale from small applications to global, distributed geospatial infrastructures.
