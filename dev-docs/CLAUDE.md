# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running Applications
- Day 3 API (main): `uvicorn src.day03_api.app:app --reload`
- Day 7 mock API: `uvicorn src.day07_mock.mock_test.api:app --reload`
- Day 1 tile server (for testing): `uvicorn src.day01_concurrency.mock_tile_server:app --reload --port 8001`

### Testing
- Run all tests: `pytest -q`
- Run specific test module: `pytest -q src/day04_testing/tests/test_smoke.py`
- Run specific test: `pytest -q src/day04_testing/tests/test_smoke.py::test_tiles_smoke`

### Benchmarking
- Day 1 benchmark (OSM): `python -m src.day01_concurrency.tile_fetcher --tile-count 20 --max-concurrency 10`
- Local benchmark: `TILE_BASE_URL=http://127.0.0.1:8001 python -m src.day01_concurrency.tile_fetcher --tile-count 50 --max-concurrency 20 --timeout 2 --retries 2`

### Makefile Commands
- `make install` - Install dependencies
- `make test` - Run all tests
- `make run-api` - Start Day 3 API
- `make run-mock-tiles` - Start mock tile server
- `make bench` - Run OSM benchmark
- `make bench-local` - Run local benchmark

## Code Architecture

### Module Structure
This is a 7-day Python geospatial engineering practice repository with each day focused on specific concepts:

- **Day 1 (`src/day01_concurrency/`)**: Async I/O patterns with bounded concurrency for tile fetching
- **Day 2 (`src/day02_oop/`)**: Provider abstractions using ABC pattern with factory for data source selection
- **Day 3 (`src/day03_api/`)**: Production FastAPI with streaming responses, validation, and Prometheus metrics
- **Day 4 (`src/day04_testing/`)**: pytest-based testing with fixtures and property-based testing
- **Day 5 (`src/day05_data/`)**: Geospatial data processing with Protobuf and spatial indexing
- **Day 6 (`src/day06_perf/`)**: Performance optimization and observability patterns
- **Day 7 (`src/day07_mock/`)**: Complete mock screening test implementation

### Key Patterns

**Async Concurrency**: Uses `asyncio.Semaphore` for bounded concurrency with exponential backoff retry logic. The tile fetcher demonstrates proper async patterns with context managers and resource cleanup.

**Provider Pattern**: Abstract base classes define `MapDataProvider` interface with concrete implementations for different data formats (CSV, GeoJSON, MVT). Factory pattern selects providers by content type.

**API Design**: FastAPI apps use Pydantic models for validation, streaming responses for large data, and middleware for metrics collection. Endpoints include proper error handling and OpenAPI documentation.

**Geospatial Processing**: Integration with Shapely for geometry operations, coordinate system handling (WGS84/Web Mercator), and spatial indexing with R-tree for efficient bbox queries.

### Testing Approach
- Uses pytest with async support (`pytest-asyncio`)
- Property-based testing with `hypothesis` for edge cases
- Integration tests with `TestClient` for API endpoints
- Smoke tests validate basic functionality across modules

### Dependencies
Key libraries: FastAPI, httpx, Shapely, rtree, pyproj, mapbox-vector-tile, prometheus-client, pytest, hypothesis

## Documentation Files
- `docs/PLAN.md` - Detailed 7-day learning plan with concepts and success criteria
- `docs/AGENTS.md` - Repository guidelines and coding conventions
- `docs/MOCK_TEST.md` - Final capstone project specification (90-minute road network service)