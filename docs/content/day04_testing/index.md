# Day 04 - Testing

## Learning Objectives

This module covers comprehensive testing strategies essential for enterprise geospatial systems, emphasizing quality engineering practices that ensure reliability, performance, and maintainability at scale. By the end of this module, you will understand:

- **Test-Driven Development (TDD) and Behavior-Driven Development (BDD)** for geospatial domain modeling
- **Advanced pytest patterns** including fixtures, parametrization, plugins, and custom test discovery
- **Property-based testing with Hypothesis** for robust geospatial algorithm validation
- **Integration testing strategies** for distributed geospatial services and external API dependencies
- **Performance testing methodologies** including load, stress, spike, and endurance testing
- **Mutation testing and fuzzing** for ensuring test quality and edge case coverage
- **Test automation in CI/CD pipelines** with geographic data and spatial analysis workflows
- **Chaos engineering principles** for testing system resilience under failure conditions

## Theoretical Foundation

### Quality Engineering in Geospatial Systems

**Spatial Data Complexity:**
Geospatial systems face unique testing challenges:
- **Coordinate precision**: Floating-point arithmetic and transformation accuracy
- **Topology validation**: Ensuring geometric integrity across operations  
- **Scale variability**: Testing from millimeter precision to global datasets
- **Temporal dimensions**: Time-series spatial data and change detection
- **Multi-format support**: Ensuring consistency across GeoJSON, WKT, MVT, etc.

**Distributed System Testing:**
Modern geospatial architectures require sophisticated testing approaches:
- **Service mesh testing**: Inter-service communication and failure scenarios
- **Event-driven architecture validation**: Async message processing and ordering
- **Spatial indexing correctness**: R-tree, quadtree, and H3 index validation
- **Cache coherence testing**: Multi-level caching with spatial invalidation

### Testing Pyramid for Geospatial Applications

```
                    /\
                   /  \     Manual Exploratory Testing
                  /____\    (Map visual validation, UX testing)
                 /      \   
                /        \  End-to-End Integration Tests
               /__________\ (Full pipeline, external services)
              /            \
             /              \ Contract/API Tests  
            /________________\ (Service boundaries, protocols)
           /                  \
          /                    \ Component/Service Tests
         /______________________\ (Business logic, spatial operations)
        /                        \
       /                          \ Unit Tests
      /____________________________\ (Pure functions, algorithms, data structures)
```

**Optimal Distribution (Geospatial Systems):**
- **Unit Tests (70%)**: Spatial algorithms, coordinate transformations, geometric operations
- **Component Tests (20%)**: Provider integrations, spatial queries, cache behavior  
- **Integration Tests (8%)**: API contracts, database transactions, external service mocks
- **E2E Tests (2%)**: Critical user workflows, visual map rendering validation

## Core Concepts

### 1) TDD Cycle
Red → Green → Refactor. Target small, observable units and keep test names behavior-focused.

### 2) pytest Essentials

pytest is a powerful testing framework that makes testing simple and scalable:

```python
import pytest

def test_simple():
    assert 2 + 2 == 4

def test_with_fixture(sample_data):
    assert len(sample_data) > 0

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6)
])
def test_doubling(input, expected):
    assert input * 2 == expected
```

### 3) Testing Pyramid

```
    /\
   /  \     E2E Tests (Few, Slow)
  /____\    
 /      \   Integration Tests (Some, Medium)
/________\  Unit Tests (Many, Fast)
```

- **Unit Tests**: Test individual functions/methods in isolation
- **Integration Tests**: Test how components work together
- **End-to-End Tests**: Test complete user workflows

## Code Walkthrough (this repo)

### 1) API Smoke Tests
`src/day04_testing/tests/test_smoke.py` verifies the Day 3 API routes.

```python
from fastapi.testclient import TestClient
from src.day03_api.app import app

def test_tiles_smoke():
    client = TestClient(app)
    resp = client.get("/tiles/0/0/0.mvt")
    assert resp.status_code == 200

def test_stream_features_invalid_bbox():
    client = TestClient(app)
    resp = client.get("/stream-features", params={
        "min_lat": 1, "min_lon": 0, 
        "max_lat": 0, "max_lon": 1
    })
    assert resp.status_code == 400
```

**Key Points:**
- `TestClient`: FastAPI's testing utility that simulates HTTP requests
- **Arrange-Act-Assert**: Clear test structure
- **Descriptive names**: Test names explain what they're testing

### 2) Test Client Usage

```python
from fastapi.testclient import TestClient

# Create a test client
client = TestClient(app)

# Test GET requests
response = client.get("/endpoint")
response = client.get("/endpoint", params={"param": "value"})

# Test POST requests
response = client.post("/endpoint", json={"data": "value"})

# Test headers
response = client.get("/endpoint", headers={"Authorization": "Bearer token"})

# Assertions
assert response.status_code == 200
assert response.json()["key"] == "expected_value"
assert "expected_header" in response.headers
```

## Running Tests
```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest src/day04_testing/tests/test_smoke.py

# Run specific test function
pytest src/day04_testing/tests/test_smoke.py::test_tiles_smoke
```

### Options
```bash
# Run tests and show coverage
pytest --cov=src

# Run tests in parallel
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run only tests matching a pattern
pytest -k "bbox"
```

## Exercises

### 1) Unit Tests for RoadNetwork

Create `test_road_network.py` to test the core data processing:

```python
import pytest
from src.day07_mock.mock_test.road_network import RoadNetwork
from unittest.mock import patch, mock_open
import csv

class TestRoadNetwork:
    @pytest.fixture
    def sample_csv_data(self):
        return """road_id,name,geometry,speed_limit,road_type,last_updated
R001,El Camino Real,"LINESTRING(-122.123 37.456, -122.124 37.457)",50,arterial,2024-01-15T10:00:00Z
R002,Page Mill Rd,"LINESTRING(-122.124 37.457, -122.125 37.458)",40,residential,2024-01-15T10:00:00Z"""

    @pytest.fixture
    def road_network(self, sample_csv_data):
        with patch("builtins.open", mock_open(read_data=sample_csv_data)):
            with patch("csv.DictReader") as mock_reader:
                mock_reader.return_value = [
                    {
                        "road_id": "R001",
                        "name": "El Camino Real",
                        "geometry": "LINESTRING(-122.123 37.456, -122.124 37.457)",
                        "speed_limit": "50",
                        "road_type": "arterial",
                        "last_updated": "2024-01-15T10:00:00Z"
                    },
                    {
                        "road_id": "R002",
                        "name": "Page Mill Rd",
                        "geometry": "LINESTRING(-122.124 37.457, -122.125 37.458)",
                        "speed_limit": "40",
                        "road_type": "residential",
                        "last_updated": "2024-01-15T10:00:00Z"
                    }
                ]
                network = RoadNetwork("dummy_path.csv")
                return network

    def test_loads_roads_correctly(self, road_network):
        assert len(road_network._roads) == 2
        assert "R001" in road_network._roads
        assert "R002" in road_network._roads

    def test_find_roads_in_bbox(self, road_network):
        roads = road_network.find_roads_in_bbox(37.45, -122.13, 37.46, -122.12)
        assert len(roads) == 2

    def test_get_connected_roads(self, road_network):
        connected = road_network.get_connected_roads("R001")
        assert "R002" in connected

    def test_update_road(self, road_network):
        updated = road_network.update_road("R001", speed_limit=60)
        assert updated.speed_limit == 60
        assert updated.road_type == "arterial"  # Unchanged
```

### 2) API Tests

Create comprehensive API tests:

```python
import pytest
from fastapi.testclient import TestClient
from src.day07_mock.mock_test.api import app
import json

class TestAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_roads_bbox_valid(self, client):
        response = client.get("/roads/bbox", params={
            "min_lat": 37.0, "min_lon": -122.0,
            "max_lat": 38.0, "max_lon": -121.0
        })
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert "features" in data

    def test_roads_bbox_invalid(self, client):
        response = client.get("/roads/bbox", params={
            "min_lat": 38.0, "min_lon": -122.0,
            "max_lat": 37.0, "max_lon": -121.0  # Invalid bbox
        })
        assert response.status_code == 400

    def test_roads_bbox_pagination(self, client):
        response = client.get("/roads/bbox", params={
            "min_lat": 37.0, "min_lon": -122.0,
            "max_lat": 38.0, "max_lon": -121.0,
            "limit": 5, "offset": 0
        })
        assert response.status_code == 200

    def test_connected_roads(self, client):
        response = client.get("/roads/R001/connected")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_update_road(self, client):
        update_data = {"speed_limit": 55}
        response = client.post("/roads/R001/update", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["properties"]["speed_limit"] == 55

    def test_update_road_invalid(self, client):
        update_data = {"speed_limit": -5}  # Invalid speed
        response = client.post("/roads/R001/update", json=update_data)
        assert response.status_code == 422  # Validation error
```

### 3) Property-Based Testing with Hypothesis

```python
import pytest
from hypothesis import given, strategies as st
from shapely.geometry import box
from src.day07_mock.mock_test.road_network import RoadNetwork

class TestRoadNetworkProperties:
    @given(
        min_lat=st.floats(min_value=-90, max_value=90),
        min_lon=st.floats(min_value=-180, max_value=180),
        max_lat=st.floats(min_value=-90, max_value=90),
        max_lon=st.floats(min_value=-180, max_value=180)
    )
    def test_bbox_query_always_returns_list(self, min_lat, min_lon, max_lat, max_lon):
        # Skip invalid bboxes
        if min_lat > max_lat or min_lon > max_lon:
            return
        
        # Create a minimal network for testing
        network = RoadNetwork("dummy_path.csv")
        result = network.find_roads_in_bbox(min_lat, min_lon, max_lat, max_lon)
        assert isinstance(result, list)

    @given(
        road_id=st.text(min_size=1, max_size=10)
    )
    def test_connected_roads_always_returns_list(self, road_id):
        network = RoadNetwork("dummy_path.csv")
        result = network.get_connected_roads(road_id)
        assert isinstance(result, list)
        assert all(isinstance(x, str) for x in result)
```

## Advanced Techniques

### 1) Fixtures and Dependency Injection

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db = Mock()
    db.query.return_value.filter.return_value.all.return_value = [
        {"id": 1, "name": "Test Road"}
    ]
    return db

@pytest.fixture
def sample_geojson():
    """Sample GeoJSON data for testing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "R001", "name": "Test Road"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-122.123, 37.456], [-122.124, 37.457]]
                }
            }
        ]
    }

def test_with_fixtures(mock_database, sample_geojson):
    # Use the fixtures in your test
    assert mock_database.query.called
    assert len(sample_geojson["features"]) == 1
```

### 2) Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("bbox,expected_count", [
    ((37.0, -122.0, 38.0, -121.0), 2),  # Large bbox
    ((37.45, -122.13, 37.46, -122.12), 2),  # Small bbox
    ((0.0, 0.0, 1.0, 1.0), 0),  # No roads in this area
])
def test_bbox_queries(bbox, expected_count):
    min_lat, min_lon, max_lat, max_lon = bbox
    network = RoadNetwork("test_data.csv")
    roads = network.find_roads_in_bbox(min_lat, min_lon, max_lat, max_lon)
    assert len(roads) == expected_count

@pytest.mark.parametrize("invalid_bbox", [
    (38.0, -122.0, 37.0, -121.0),  # min_lat > max_lat
    (37.0, -121.0, 38.0, -122.0),  # min_lon > max_lon
])
def test_invalid_bbox_raises_error(invalid_bbox):
    min_lat, min_lon, max_lat, max_lon = invalid_bbox
    network = RoadNetwork("test_data.csv")
    
    with pytest.raises(ValueError):
        network.find_roads_in_bbox(min_lat, min_lon, max_lat, max_lon)
```

### 3) Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock
import httpx

def test_tile_fetching_with_mock():
    with patch("httpx.AsyncClient") as mock_client:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = b"fake_tile_data"
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        # Test the function
        from src.day01_concurrency.tile_fetcher import fetch_tile
        import asyncio
        
        result = asyncio.run(fetch_tile(mock_client_instance, 5, 10, 12))
        assert result == b"fake_tile_data"
```

## Load Testing (lightweight)

### 1) pytest-benchmark

```python
import pytest
from fastapi.testclient import TestClient
from src.day07_mock.mock_test.api import app

def test_api_performance(benchmark):
    client = TestClient(app)
    
    def make_request():
        return client.get("/roads/bbox", params={
            "min_lat": 37.0, "min_lon": -122.0,
            "max_lat": 38.0, "max_lon": -121.0
        })
    
    result = benchmark(make_request)
    assert result.status_code == 200

def test_concurrent_requests(benchmark):
    import asyncio
    import httpx
    
    async def make_concurrent_requests():
        async with httpx.AsyncClient() as client:
            tasks = [
                client.get("http://localhost:8000/roads/bbox", params={
                    "min_lat": 37.0, "min_lon": -122.0,
                    "max_lat": 38.0, "max_lon": -121.0
                })
                for _ in range(10)
            ]
            responses = await asyncio.gather(*tasks)
            return responses
    
    # This would need a running server
    # result = benchmark(lambda: asyncio.run(make_concurrent_requests()))
```

### 2. Load Testing with Locust

Create `locustfile.py` for more sophisticated load testing:

```python
from locust import HttpUser, task, between

class RoadNetworkUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def test_bbox_query(self):
        self.client.get("/roads/bbox", params={
            "min_lat": 37.0, "min_lon": -122.0,
            "max_lat": 38.0, "max_lon": -121.0
        })
    
    @task(1)
    def test_connected_roads(self):
        self.client.get("/roads/R001/connected")
    
    @task(1)
    def test_update_road(self):
        self.client.post("/roads/R001/update", json={"speed_limit": 55})
```

## Test Coverage

### 1. Measuring Coverage

```bash
# Install coverage tools
pip install pytest-cov

# Run tests with coverage
pytest --cov=src --cov-report=html

# Generate detailed coverage report
pytest --cov=src --cov-report=term-missing
```

### 2. Coverage Configuration

Create `.coveragerc` file:

```ini
[run]
source = src
omit = 
    */tests/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Best Practices

### 1. Test Organization
```
tests/
├── unit/           # Unit tests
│   ├── test_road_network.py
│   └── test_api.py
├── integration/    # Integration tests
│   └── test_end_to_end.py
└── conftest.py    # Shared fixtures
```

### 2. Test Naming
```python
# ✅ Descriptive test names
def test_find_roads_in_bbox_returns_empty_list_for_area_with_no_roads():
    pass

def test_update_road_speed_limit_updates_correct_field():
    pass

# ❌ Unclear test names
def test_bbox():
    pass

def test_update():
    pass
```

### 3. Test Data Management
```python
@pytest.fixture(scope="session")
def sample_roads_data():
    """Load test data once for all tests."""
    return load_test_data_from_file("sample_roads.csv")

@pytest.fixture
def road_network(sample_roads_data):
    """Create fresh network for each test."""
    with patch("builtins.open", mock_open(read_data=sample_roads_data)):
        return RoadNetwork("dummy_path.csv")
```

## Common Pitfalls

### 1. Testing Implementation Details
```python
# ❌ Testing internal state
def test_internal_index():
    network = RoadNetwork("data.csv")
    assert "_index" in network.__dict__  # Too specific

# ✅ Testing public behavior
def test_can_find_road_by_id():
    network = RoadNetwork("data.csv")
    road = network.get_road_by_id("R001")
    assert road is not None
```

### 2. Over-Mocking
```python
# ❌ Mocking everything
@patch("pathlib.Path")
@patch("csv.DictReader")
@patch("shapely.wkt.loads")
def test_over_mocked(mock_loads, mock_reader, mock_path):
    # Test becomes brittle and hard to maintain

# ✅ Mock only external dependencies
def test_with_minimal_mocking():
    with patch("builtins.open", mock_open(read_data=sample_csv)):
        network = RoadNetwork("data.csv")
        # Test the actual logic
```

### 3. Ignoring Edge Cases
```python
# ❌ Only testing happy path
def test_bbox_query():
    roads = network.find_roads_in_bbox(37.0, -122.0, 38.0, -121.0)
    assert len(roads) > 0

# ✅ Test edge cases too
def test_bbox_query_empty_area():
    roads = network.find_roads_in_bbox(0.0, 0.0, 1.0, 1.0)
    assert len(roads) == 0

def test_bbox_query_invalid_coordinates():
    with pytest.raises(ValueError):
        network.find_roads_in_bbox(38.0, -122.0, 37.0, -121.0)
```

## Next Steps

After completing this day:
1. Achieve >80% test coverage on core functionality
2. Add integration tests with real data
3. Implement performance benchmarks
4. Set up continuous integration (CI) pipeline
5. Add property-based tests for complex logic

## Advanced Testing Patterns

### Chaos Engineering for Geospatial Systems

```python
class SpatialChaosExperiment:
    """Chaos engineering for testing spatial system resilience."""
    
    def __init__(self, spatial_service: SpatialService):
        self.service = spatial_service
        self.failures_injected = []
    
    async def inject_coordinate_drift(self, drift_factor: float = 0.001):
        """Inject small coordinate drifts to test precision handling."""
        original_transform = self.service.coordinate_transformer.transform
        
        def drifted_transform(x: float, y: float) -> Tuple[float, float]:
            drift_x = x + (random.random() - 0.5) * drift_factor
            drift_y = y + (random.random() - 0.5) * drift_factor
            return original_transform(drift_x, drift_y)
        
        self.service.coordinate_transformer.transform = drifted_transform
        self.failures_injected.append("coordinate_drift")
    
    async def simulate_spatial_index_corruption(self):
        """Simulate partial spatial index corruption."""
        # Randomly remove some entries from spatial index
        if hasattr(self.service, '_spatial_index'):
            index = self.service._spatial_index
            entries = list(index.intersection(index.bounds))
            corrupted_entries = random.sample(entries, len(entries) // 10)
            
            for entry_id in corrupted_entries:
                try:
                    index.delete(entry_id, index.get_bounds(entry_id))
                except:
                    pass  # Index might already be corrupted
        
        self.failures_injected.append("index_corruption")
    
    async def induce_memory_pressure(self, target_mb: int = 100):
        """Create memory pressure to test resource handling."""
        memory_hog = []
        try:
            # Allocate memory in chunks
            for _ in range(target_mb):
                memory_hog.append(b'x' * (1024 * 1024))  # 1MB chunks
            
            # Hold memory for test duration
            await asyncio.sleep(1)
        finally:
            del memory_hog
        
        self.failures_injected.append("memory_pressure")

@pytest.fixture
async def chaos_experiment(spatial_service):
    experiment = SpatialChaosExperiment(spatial_service)
    yield experiment
    
    # Cleanup: restore original state if possible
    if hasattr(spatial_service, '_reset_to_clean_state'):
        await spatial_service._reset_to_clean_state()

class TestSpatialResilience:
    """Test suite for spatial system resilience under chaos conditions."""
    
    @pytest.mark.asyncio
    async def test_bbox_query_under_coordinate_drift(self, chaos_experiment):
        """Test that bbox queries remain stable under coordinate drift."""
        # Inject coordinate drift
        await chaos_experiment.inject_coordinate_drift(drift_factor=0.0001)
        
        # Original query
        bbox = (37.0, -122.0, 38.0, -121.0)
        results_before = await chaos_experiment.service.query_bbox(bbox)
        
        # Query with drift should still return reasonable results
        results_after = await chaos_experiment.service.query_bbox(bbox)
        
        # Results should be similar (allowing for some drift tolerance)
        assert abs(len(results_before) - len(results_after)) <= 2
    
    @pytest.mark.asyncio
    async def test_spatial_operations_during_memory_pressure(self, chaos_experiment):
        """Test spatial operations continue working under memory pressure."""
        # Create memory pressure
        memory_task = asyncio.create_task(
            chaos_experiment.induce_memory_pressure(target_mb=50)
        )
        
        try:
            # Perform spatial operations during memory pressure
            bbox = (37.0, -122.0, 38.0, -121.0)
            results = await chaos_experiment.service.query_bbox(bbox)
            
            # Should not fail completely, might have reduced performance
            assert isinstance(results, list)
            
        finally:
            await memory_task
```

### Mutation Testing for Spatial Algorithms

```python
# Install: pip install mutmut
# Usage: mutmut run --paths-to-mutate=src/spatial_algorithms/

class TestSpatialAlgorithmRobustness:
    """Test suite designed to catch mutations in spatial algorithms."""
    
    def test_distance_calculation_edge_cases(self):
        """Comprehensive tests to catch distance calculation mutations."""
        from src.spatial_algorithms import calculate_distance
        
        # Test identical points
        assert calculate_distance(0, 0, 0, 0) == 0
        
        # Test symmetric property
        d1 = calculate_distance(1, 1, 2, 2)
        d2 = calculate_distance(2, 2, 1, 1)
        assert abs(d1 - d2) < 1e-10
        
        # Test triangle inequality
        p1, p2, p3 = (0, 0), (1, 0), (1, 1)
        d12 = calculate_distance(*p1, *p2)
        d23 = calculate_distance(*p2, *p3)
        d13 = calculate_distance(*p1, *p3)
        assert d12 + d23 >= d13 - 1e-10
        
        # Test known distances
        assert abs(calculate_distance(0, 0, 3, 4) - 5.0) < 1e-10
        
        # Test extreme values
        huge_val = 1e10
        assert calculate_distance(0, 0, huge_val, 0) == huge_val
    
    def test_bbox_intersection_mutations(self):
        """Tests designed to catch bbox intersection logic mutations."""
        from src.spatial_algorithms import bbox_intersects
        
        # Test complete overlap
        bbox1 = (0, 0, 2, 2)
        bbox2 = (1, 1, 3, 3)
        assert bbox_intersects(bbox1, bbox2) == True
        
        # Test no overlap
        bbox1 = (0, 0, 1, 1)
        bbox2 = (2, 2, 3, 3)
        assert bbox_intersects(bbox1, bbox2) == False
        
        # Test edge touching (depends on implementation)
        bbox1 = (0, 0, 1, 1)
        bbox2 = (1, 1, 2, 2)
        result = bbox_intersects(bbox1, bbox2)
        assert isinstance(result, bool)  # Should not crash
        
        # Test degenerate bboxes
        bbox1 = (0, 0, 0, 0)  # Point
        bbox2 = (0, 0, 1, 1)
        result = bbox_intersects(bbox1, bbox2)
        assert isinstance(result, bool)
```

### Fuzzing for Geospatial Input Validation

```python
import hypothesis.strategies as st
from hypothesis import given, assume, settings, HealthCheck

class TestGeospatialInputFuzzing:
    """Fuzz testing for geospatial input handling."""
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False)
    )
    def test_coordinate_validation_never_crashes(self, lat, lon):
        """Coordinate validation should never crash, regardless of input."""
        from src.spatial_validation import validate_coordinate
        
        # Should either return True or raise a specific exception
        try:
            result = validate_coordinate(lat, lon)
            assert isinstance(result, bool)
        except (ValueError, TypeError) as e:
            # Acceptable to raise validation errors
            assert str(e)  # Error message should not be empty
    
    @given(
        coords=st.lists(
            st.tuples(
                st.floats(min_value=-180, max_value=180, allow_nan=False),
                st.floats(min_value=-90, max_value=90, allow_nan=False)
            ),
            min_size=3,
            max_size=1000
        )
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=5000)
    def test_polygon_validation_fuzzing(self, coords):
        """Polygon validation should handle arbitrary coordinate sequences."""
        from src.spatial_validation import validate_polygon
        
        # Ensure polygon is closed
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        
        try:
            result = validate_polygon(coords)
            assert isinstance(result, bool)
            
            # If valid, polygon should have basic properties
            if result:
                assert len(coords) >= 4  # Minimum for closed polygon
                assert coords[0] == coords[-1]  # Must be closed
                
        except (ValueError, TypeError, OverflowError) as e:
            # Acceptable exceptions for invalid input
            pass
    
    @given(
        geojson_like=st.recursive(
            st.one_of(
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(),
                st.booleans(),
                st.none()
            ),
            lambda children: st.one_of(
                st.lists(children, max_size=20),
                st.dictionaries(st.text(max_size=50), children, max_size=10)
            ),
            max_leaves=100
        )
    )
    def test_geojson_parser_robustness(self, geojson_like):
        """GeoJSON parser should handle arbitrary JSON-like structures."""
        from src.geojson_parser import parse_geojson
        
        try:
            result = parse_geojson(geojson_like)
            # If parsing succeeds, result should be valid
            assert hasattr(result, 'type') or result is None
        except (ValueError, TypeError, KeyError) as e:
            # Expected for invalid GeoJSON
            pass
```

### Performance and Load Testing Framework

```python
class SpatialLoadTestFramework:
    """Framework for load testing spatial services."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = httpx.AsyncClient()
        self.metrics = {
            'requests_sent': 0,
            'requests_succeeded': 0,
            'requests_failed': 0,
            'total_latency': 0,
            'min_latency': float('inf'),
            'max_latency': 0,
            'latencies': []
        }
    
    async def generate_spatial_workload(
        self,
        concurrent_users: int = 50,
        requests_per_user: int = 100,
        test_duration_seconds: int = 300
    ):
        """Generate realistic spatial query workload."""
        
        async def user_session(user_id: int):
            """Simulate individual user behavior."""
            requests_made = 0
            start_time = time.time()
            
            while (requests_made < requests_per_user and 
                   time.time() - start_time < test_duration_seconds):
                
                # Generate realistic spatial query
                bbox = self._generate_realistic_bbox()
                
                request_start = time.time()
                try:
                    response = await self.session.get(
                        f"{self.base_url}/api/features/bbox",
                        params={
                            'min_lat': bbox[0], 'min_lon': bbox[1],
                            'max_lat': bbox[2], 'max_lon': bbox[3]
                        },
                        timeout=30.0
                    )
                    
                    latency = time.time() - request_start
                    self._record_success(latency)
                    
                    if response.status_code != 200:
                        self._record_failure(f"HTTP {response.status_code}")
                    
                except Exception as e:
                    self._record_failure(str(e))
                
                requests_made += 1
                
                # Realistic user think time
                await asyncio.sleep(random.uniform(0.1, 2.0))
        
        # Run concurrent user sessions
        tasks = [
            asyncio.create_task(user_session(i)) 
            for i in range(concurrent_users)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        return self._generate_report()
    
    def _generate_realistic_bbox(self) -> Tuple[float, float, float, float]:
        """Generate realistic bounding boxes based on typical usage patterns."""
        # Focus on populated areas with varying zoom levels
        city_centers = [
            (37.7749, -122.4194),  # San Francisco
            (40.7128, -74.0060),   # New York
            (51.5074, -0.1278),    # London
            (35.6762, 139.6503),   # Tokyo
        ]
        
        center = random.choice(city_centers)
        
        # Random zoom level affects bbox size
        zoom_level = random.randint(10, 18)
        size = 1.0 / (2 ** (zoom_level - 10))  # Smaller at higher zoom
        
        return (
            center[0] - size,
            center[1] - size,
            center[0] + size,
            center[1] + size
        )

@pytest.mark.asyncio
@pytest.mark.load_test
async def test_spatial_api_under_load():
    """Load test for spatial API endpoints."""
    framework = SpatialLoadTestFramework("http://localhost:8000")
    
    report = await framework.generate_spatial_workload(
        concurrent_users=25,
        requests_per_user=50,
        test_duration_seconds=60
    )
    
    # Performance assertions
    assert report['success_rate'] > 0.95  # 95% success rate
    assert report['p95_latency'] < 2.0    # 95th percentile under 2 seconds
    assert report['avg_latency'] < 0.5    # Average under 500ms
```

### CI/CD Integration Patterns

```python
# pytest.ini configuration for geospatial testing
"""
[tool:pytest]
minversion = 6.0
addopts = 
    -ra 
    -q 
    --strict-markers 
    --strict-config
    --cov=src 
    --cov-branch 
    --cov-report=term-missing 
    --cov-report=html
    --cov-fail-under=80
    --hypothesis-show-statistics
markers =
    unit: Unit tests
    integration: Integration tests
    load_test: Load and performance tests
    chaos: Chaos engineering tests
    slow: Tests that take more than 5 seconds
    spatial: Tests that require spatial data/operations
    external: Tests that require external services
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
"""

class GeospatialTestConfig:
    """Configuration for geospatial test environments."""
    
    @staticmethod
    def setup_test_database():
        """Set up PostGIS test database."""
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        # Create test database with PostGIS
        conn = psycopg2.connect(
            host="localhost",
            user="postgres", 
            password="password"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute("DROP DATABASE IF EXISTS test_geospatial")
        cursor.execute("CREATE DATABASE test_geospatial")
        
        # Connect to test database and enable PostGIS
        test_conn = psycopg2.connect(
            host="localhost",
            user="postgres",
            password="password",
            database="test_geospatial"
        )
        test_cursor = test_conn.cursor()
        test_cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        test_conn.commit()
        
        return "postgresql://postgres:password@localhost/test_geospatial"
    
    @staticmethod
    def setup_test_data():
        """Generate test spatial datasets."""
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
        
        # Generate synthetic spatial data
        test_features = []
        for i in range(1000):
            # Random points around San Francisco
            lat = 37.7749 + random.uniform(-0.1, 0.1)
            lon = -122.4194 + random.uniform(-0.1, 0.1)
            
            feature = {
                'id': f'feature_{i}',
                'geometry': Point(lon, lat),
                'properties': {
                    'name': f'Test Feature {i}',
                    'category': random.choice(['restaurant', 'shop', 'park']),
                    'rating': random.uniform(1.0, 5.0)
                }
            }
            test_features.append(feature)
        
        # Create GeoDataFrame and save as test file
        gdf = gpd.GeoDataFrame(test_features)
        gdf.to_file('test_data/synthetic_features.geojson', driver='GeoJSON')
        
        return gdf

# GitHub Actions workflow for geospatial testing
"""
name: Geospatial Testing Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgis/postgis:13-3.1
        env:
          POSTGRES_PASSWORD: password
          POSTGRES_DB: test_geospatial
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install GDAL
      run: |
        sudo apt-get update
        sudo apt-get install -y gdal-bin libgdal-dev
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: pytest tests/ -m "unit" --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: pytest tests/ -m "integration" 
      env:
        DATABASE_URL: postgresql://postgres:password@localhost:5432/test_geospatial
    
    - name: Run spatial accuracy tests
      run: pytest tests/ -m "spatial" --hypothesis-show-statistics
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
"""
```

## Data-Driven Testing for Geospatial Systems

### Test Data Management

```python
class SpatialTestDataManager:
    """Manages test datasets for geospatial testing."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.datasets = {}
    
    def register_dataset(self, name: str, generator_func: Callable):
        """Register a test dataset generator."""
        self.datasets[name] = generator_func
    
    def get_dataset(self, name: str, **kwargs) -> Any:
        """Get or generate a test dataset."""
        cache_key = f"{name}_{hash(str(kwargs))}"
        cache_file = self.data_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Generate dataset
        dataset = self.datasets[name](**kwargs)
        
        # Cache for future use
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        return dataset

# Test data generators
def generate_road_network(num_roads: int = 100, area_bounds: Tuple = None):
    """Generate synthetic road network for testing."""
    if area_bounds is None:
        area_bounds = (37.7, -122.5, 37.8, -122.4)  # SF area
    
    roads = []
    for i in range(num_roads):
        # Generate random road segments
        start_lat = random.uniform(area_bounds[0], area_bounds[2])
        start_lon = random.uniform(area_bounds[1], area_bounds[3])
        
        # Random direction and length
        bearing = random.uniform(0, 360)
        length = random.uniform(0.001, 0.01)  # degrees
        
        end_lat = start_lat + length * math.cos(math.radians(bearing))
        end_lon = start_lon + length * math.sin(math.radians(bearing))
        
        road = {
            'id': f'road_{i}',
            'geometry': f'LINESTRING({start_lon} {start_lat}, {end_lon} {end_lat})',
            'properties': {
                'name': f'Test Road {i}',
                'speed_limit': random.choice([25, 35, 45, 55]),
                'road_type': random.choice(['residential', 'arterial', 'highway'])
            }
        }
        roads.append(road)
    
    return roads

# Usage in tests
@pytest.fixture(scope="session")
def spatial_test_data():
    manager = SpatialTestDataManager(Path("test_data"))
    manager.register_dataset("road_network", generate_road_network)
    return manager

class TestWithSpatialData:
    def test_road_network_loading(self, spatial_test_data):
        """Test road network loading with different dataset sizes."""
        small_network = spatial_test_data.get_dataset("road_network", num_roads=10)
        large_network = spatial_test_data.get_dataset("road_network", num_roads=1000)
        
        assert len(small_network) == 10
        assert len(large_network) == 1000
```

## Resources

### Testing Frameworks and Tools
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Coverage Plugin](https://pytest-cov.readthedocs.io/)
- [Hypothesis Property-Based Testing](https://hypothesis.readthedocs.io/)
- [mutmut Mutation Testing](https://mutmut.readthedocs.io/)
- [locust Load Testing](https://locust.io/)

### Geospatial Testing
- [PostGIS Testing Guide](https://postgis.net/documentation/)
- [GDAL/OGR Test Suite](https://gdal.org/development/testing.html)
- [Shapely Testing Patterns](https://shapely.readthedocs.io/)

### Quality Engineering
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Test-Driven Development by Example](https://www.amazon.com/Test-Driven-Development-Kent-Beck/dp/0321146530)
- [Growing Object-Oriented Software, Guided by Tests](https://www.amazon.com/Growing-Object-Oriented-Software-Guided-Tests/dp/0321503627)
- [Building Quality Software by NASA](https://software.nasa.gov/)

### Chaos Engineering
- [Principles of Chaos Engineering](https://principlesofchaos.org/)
- [Chaos Monkey for Spatial Systems](https://netflix.github.io/chaosmonkey/)

This comprehensive testing module ensures that geospatial systems are robust, reliable, and performant under all conditions, from normal operation to extreme failure scenarios.
