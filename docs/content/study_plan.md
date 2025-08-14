# Study Plan

A 7‑day plan to build production‑ready skills for software engineering with geospatial data. Each day links to an overview page in this site and to the runnable code modules in the repository.

## Prerequisites
- Python 3.11+
- Create a virtualenv and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Project code lives under `src/` with one folder per day; docs live under `docs/content/`

---

## Day 01 – Core Python + Concurrency
- **Overview**: [Day 01](day01_concurrency/index.md)
- **Code**: [src/day01_concurrency](https://github.com/rgdonohue/geospatial-python-guide/tree/main/src/day01_concurrency)
- **Focus**: Async I/O for high‑throughput tile fetching with backpressure.
- **Concepts**: `async`/`await`, task groups, timeouts, `asyncio.Semaphore`.
- **Drill**: Implement and benchmark a bounded‑concurrency tile fetcher; compare sequential vs concurrent.
- **Success**: Measurable speedup; clean cancellation and retry/timeout behavior.

---

## Day 02 – OOP & Provider Abstractions
- **Overview**: [Day 02](day02_oop/index.md)
- **Code**: [src/day02_oop](https://github.com/rgdonohue/geospatial-python-guide/tree/main/src/day02_oop)
- **Focus**: Interchangeable data providers behind clear interfaces.
- **Concepts**: ABCs, composition, Factory/Strategy patterns, package layout.
- **Drill**: Define a `MapDataProvider`; implement `GeoJSONProvider` and `MVTProvider`; simple factory.
- **Success**: Providers swap seamlessly behind a single interface with light tests.

---

## Day 03 – Production‑Ready APIs
- **Overview**: [Day 03](day03_api/index.md)
- **Code**: [src/day03_api](https://github.com/rgdonohue/geospatial-python-guide/tree/main/src/day03_api)
- **Focus**: Robust FastAPI endpoints for spatial data.
- **Concepts**: Pydantic models, validation, streaming responses, OpenAPI quality.
- **Drill**: Tiles endpoint with validation and streaming; bbox‑filtered NDJSON stream.
- **Success**: Validated endpoints with docs and metrics; runs locally.

---

## Day 04 – Testing & Integration
- **Overview**: [Day 04](day04_testing/index.md)
- **Code**: [src/day04_testing/tests](https://github.com/rgdonohue/geospatial-python-guide/tree/main/src/day04_testing/tests)
- **Focus**: TDD flow plus integration against sample data/services.
- **Concepts**: `pytest` fixtures/parametrization, mocking, property‑based tests.
- **Drill**: Unit tests for Day 03, integration tests with data, small load test.
- **Success**: >80% coverage on core paths; latency numbers captured.

---

## Day 05 – Data & Geospatial Fundamentals
- **Overview**: [Day 05](day05_data/index.md)
- **Code**: [src/day05_data](https://github.com/rgdonohue/geospatial-python-guide/tree/main/src/day05_data)
- **Focus**: Efficient vector data formats and spatial indexing.
- **Concepts**: Protobuf vs JSON, MVT, CRS (WGS84, Web Mercator), R‑tree.
- **Drill**: Define `RoadSegment` protobuf; serialize/deserialize; benchmark R‑tree vs naive bbox.
- **Success**: Clear perf delta; correctness validated by cross‑checks.

---

## Day 06 – Performance & Reliability
- **Overview**: [Day 06](day06_perf/index.md)
- **Code**: [src/day06_perf](https://github.com/rgdonohue/geospatial-python-guide/tree/main/src/day06_perf)
- **Focus**: Profiling, resiliency, and observability for services.
- **Concepts**: `cProfile`, connection pooling, retries with backoff, circuit breaker, Prometheus.
- **Drill**: Add metrics to Day 03; simulate upstream failure; profile and optimize a slow path.
- **Success**: Metrics exposed; graceful degradation; perf improvement demonstrated.

---

## Day 07 – End‑to‑End Mock Screen
- **Overview**: [Day 07](day07_mock/index.md)
- **Code**: [src/day07_mock/mock_test](https://github.com/rgdonohue/geospatial-python-guide/tree/main/src/day07_mock/mock_test)
- **Focus**: Build a small geospatial API under time constraints with tests and docs.
- **Tasks**: Parse CSV/GeoJSON, bbox/attribute filter, serve via FastAPI, test and document.
- **Success**: Working app with tests and concise README within a 90‑minute timebox.

---

## Key Resources
- FastAPI Docs
- Pytest Docs
- Mapbox Vector Tile Spec
- PostGIS in Action
- Fluent Python
