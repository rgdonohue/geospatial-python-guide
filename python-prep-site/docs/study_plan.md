# Python Geospatial Key Practices

### For Geospatial Engineering – Map Integration and Backend Systems

---

## Prerequisites and Environment

- Python 3.11+
- Use a virtualenv with pip (no conda). From the repo root:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Code scaffold lives under `src/` with one folder per day.

---

## Day 1 – Core Python + Concurrency

**Focus:** Async I/O patterns for high-throughput tile fetching and backpressure.

**Concepts:**
- Data structures and comprehensions; iterators/generators
- `dataclasses` vs `NamedTuple`
- Type hints and mypy-friendly APIs
- `async`/`await`, task groups, backpressure with `asyncio.Semaphore`

**Drill:**
1. Implement async tile fetcher for `https://tile.openstreetmap.org/{z}/{x}/{y}.png`
2. Run sequential vs concurrent benchmarks; add bounded concurrency
3. Record results and observations

**Artifacts:** `src/day01_concurrency/`

**Success Criteria:** Measurable speedup with concurrency; clean cancellation and timeout handling.

**→ [Go to Concurrency Module](day01_concurrency/index.md)**

---

## Day 2 – OOP, Design Patterns & Project Structure

**Focus:** Clean abstractions for data providers.

**Concepts:**
- Abstract base classes, composition over inheritance
- Factory and Strategy patterns
- Package layout, `__init__.py`, tests organization

**Drill:**
1. Define `MapDataProvider` ABC
2. Implement `GeoJSONProvider` and `MVTProvider`
3. Simple factory to select provider by content type

**Artifacts:** `src/day02_oop/`

**Success Criteria:** Providers interchangeable behind a single interface with minimal tests.

**→ [Go to OOP Module](day02_oop/index.md)**

---

## Day 3 – Production-Ready APIs

**Focus:** Robust FastAPI endpoints for spatial data.

**Concepts:**
- Pydantic models, validation, response models
- Streaming responses, ranged responses
- OpenAPI docs quality

**Drill:**
1. Implement `/tiles/{z}/{x}/{y}.mvt` with validation and streaming from disk
2. Implement `/stream-features` filtering large GeoJSON by bbox; stream NDJSON

**Artifacts:** `src/day03_api/`

**Success Criteria:** Endpoints validated, documented, and locally runnable.

**→ [Go to APIs Module](day03_api/index.md)**

---

## Day 4 – Testing & Integration

**Focus:** TDD flow and integration with external services.

**Concepts:**
- `pytest` fixtures, parametrization, and mocking
- Property-based testing with `hypothesis`
- Basic load testing and latency measurement

**Drill:**
1. Unit tests for Day 3 endpoints using `TestClient`
2. Integration tests with sample data
3. Light load test against `/tiles` and `/stream-features`

**Artifacts:** `src/day04_testing/`

**Success Criteria:** >80% coverage on core code; latency numbers captured.

**→ [Go to Testing Module](day04_testing/index.md)**

---

## Day 5 – Data Processing, Geospatial & Automotive Standards

**Focus:** Vector tiles, coordinate systems, efficient spatial queries.

**Concepts:**
- Shapely, CRS (WGS84, Web Mercator), local ENU basics
- Mapbox Vector Tile; Protobuf wire format
- Spatial indexing with R-tree

**Drills:**
1. Protobuf: define `RoadSegment` schema, serialize/deserialize, compare to JSON
2. Spatial index: build R-tree, query bbox vs naive loop, benchmark

**Artifacts:** `src/day05_data/`

**Success Criteria:** Clear benchmarks; correctness validated by cross-checks.

**→ [Go to Data Module](day05_data/index.md)**

---

## Day 6 – Performance, Reliability & Observability

**Focus:** Production-grade reliability patterns.

**Concepts:**
- `cProfile`, `timeit`, memory profiling
- Connection pooling and retries with backoff
- Circuit breaker; Prometheus metrics

**Drill:**
1. Add Prometheus metrics to Day 3 API (request count, latency, cache hit/miss)
2. Simulate upstream failure and return degraded response
3. Profile and optimize a slow path from Day 5

**Artifacts:** `src/day06_perf/`

**Success Criteria:** Metrics exposed; graceful degradation demonstrated; perf delta measured.

**→ [Go to Performance Module](day06_perf/index.md)**

---

## Day 7 – Full Mock Screening

**Focus:** Simulate test conditions end-to-end.

**Tasks:**
- Parse CSV/GeoJSON; filter by bbox & attribute
- Serve results via FastAPI; unit tests; README
- Timebox: 90 minutes; conduct post-mortem

**Artifacts:** `src/day07_mock/mock_test/` (+ see `MOCK_TEST.md`)

**Success Criteria:** Working app with tests and clear documentation in the timebox.

**→ [Go to Mock Project Module](day07_mock/index.md)**


