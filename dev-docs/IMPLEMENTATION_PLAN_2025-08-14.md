# Implementation Plan

This plan balances production readiness with a coherent, engaging learning narrative. Work is staged in phases so each module delivers a runnable, teachable path without overwhelming detail.

## Phase 1 — Day 3: API Core

Deliverables:
- Implement tile file streaming in `src/day03_api/app.py` with `Content-Length`, `404` on missing, and correct media type `application/vnd.mapbox-vector-tile`.
- Add tile path validation: `0 <= x,y < 2**z`; introduce a bbox request model with validators for `/stream-features`.
- Minimal structured logging (request id, bbox area, feature count) alongside existing Prometheus metrics.

Acceptance:
- Tests pass for tile 200/404, x,y/bbox validation, and NDJSON line streaming.
- Manual curl checks verified; `make run-api` works end-to-end.

## Phase 2 — Day 7: Capstone Data + Caching

Deliverables:
- Include a `roads.csv` seed (or a generator) and document schema.
- Add rtree-based bbox querying in `RoadNetwork` (fallback to O(n) if disabled or missing rtree).
- Integrate Redis cache for bbox responses with stale-while-revalidate (SWR); response headers indicate staleness.
- Basic token-bucket rate limiting and `/healthz` endpoint.
- Docker Compose to run API + Redis locally.

Acceptance:
- “Capstone flow” works (bbox → connected → update) with observable cache hits and SWR via metrics.
- `docker compose up` launches services; sample requests documented and reproducible.

## Phase 3 — Day 4: Testing & CI

Deliverables:
- Expand tests: tile bounds, bbox pagination, ProviderFactory errors, metrics emission, and Day 1 retry/backoff classification.
- Add `pytest.ini`; include bounded Hypothesis settings for property tests (NDJSON well-formedness, bbox invariants).
- GitHub Actions workflow: run tests + coverage, upload artifacts.

Acceptance:
- CI green for Python 3.11 (optional matrix includes 3.12).
- Enforced coverage threshold (target: ≥ 80%).

## Phase 4 — Day 2: Providers Enhancements

Deliverables:
- Extend `MapDataProvider` with `iter_features(bbox=None, limit=None, offset=None)`, `count()`, optional `crs` attribute.
- CSV/GeoJSON providers support lazy streaming; configurable ID property; return GeoJSON-compatible Feature dicts.
- README “recipes” showing GeoPandas/Dask interoperability (short narrative, minimal code).

Acceptance:
- Unit tests pass for ID lookups, bbox filter correctness, and `count()`.
- A small example script demonstrates factory usage and streaming.

## Phase 5 — Day 6: Performance Harness

Deliverables:
- Scripts to profile API and Day 1 worker (cProfile + tracemalloc) and sweep concurrency vs p95 latency.
- Prometheus recording rules for p95/p99 and a minimal Grafana dashboard JSON.
- Redis cache drill: hit/miss metrics and observed impact on latency.

Acceptance:
- Benchmark artifacts saved to `benchmarks/day06/` with environment metadata.
- Dashboard loads with sensible latency/throughput/error panels; rules evaluate on sample metrics.

## Phase 6 — Day 5: Data Demos

Deliverables:
- Protobuf roundtrip demo (encode/decode vs JSON size/time) and a Make target for code generation.
- Document an “advanced” schema (coords + optional Z/T + CRS) alongside the minimal schema; keep Protobuf as the runnable path.
- One small indexing example (R-tree) with correctness check vs brute-force.

Acceptance:
- Scripts run locally; outputs show size/throughput comparisons.
- Tests pass for Protobuf roundtrip and bbox correctness.

## Phase 7 — Cross-Cutting Polish

Deliverables:
- Centralized configuration (env-driven dataclass or `pydantic-settings`) introduced where appropriate (Day 3+).
- Error taxonomy and FastAPI exception handlers for `InvalidTile`, `InvalidBBox`, and `ProviderError`.
- Caching headers (`ETag`, `Last-Modified`, `Cache-Control`) where applicable.
- Readmes synced with code: each day explains how to run, observe, and how it connects to prior days.

Acceptance:
- Consistent configuration usage and standardized error responses.
- Documentation accurately reflects runnable code and expected outcomes.

## Dependencies & Sequencing

1) Day 3 API (unlock testing and flows)
2) Day 7 data + caching (end-to-end value quickly)
3) Day 4 tests + CI (stabilize)
4) Day 2 providers (support API/capstone)
5) Day 6 performance harness
6) Day 5 data demos
7) Cross-cutting polish and documentation sync

## Risks & Mitigations

- Scope creep: keep one strong runnable path per phase; defer optional extras.
- Environment drift: capture environment metadata in benchmarks; pin test tool versions.
- Teaching complexity: maintain a forward narrative with tangible wins each phase.

