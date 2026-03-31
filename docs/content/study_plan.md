# Python Geospatial Engineering Study Plan

## Audience

This guide is for:

- Python and backend engineers moving into geospatial systems
- data engineers who need to work with spatial data, APIs, and query services
- engineers who want code, services, testing, and production-oriented drills rather than GIS desktop workflows

Assumptions:

- you already know Python, HTTP APIs, testing, and basic software design
- you do not need beginner Python instruction
- you may be new to CRS, geometry validity, tile systems, and spatial query behavior

## What You Will Build

By the end of the core path, you should be able to:

- validate geospatial inputs and document CRS and coordinate-order assumptions
- define clean provider boundaries around spatial data formats
- compare naive spatial filtering to indexed lookup
- build a small FastAPI service for spatial queries and tile-style endpoints
- use concurrency where it helps at I/O boundaries
- test geo-specific edge cases instead of relying on smoke tests alone
- measure latency, profile slow paths, and expose basic service metrics
- complete a small road-network query service that is credible as portfolio work

## Core Path

The core path is intentionally narrow:

1. Learn the correctness rules that keep geospatial systems from silently lying
2. Put stable data contracts around formats and providers
3. Work with realistic query patterns and indexing
4. Build APIs and tile delivery on top of that foundation
5. Add concurrency where it improves integrations and upstream access
6. Harden what you built with testing, profiling, and observability
7. Finish with a realistic capstone service

## Module 1: Spatial Correctness for Engineers

**Purpose:** Make geospatial correctness the starting point, not a cleanup step.

**Focus:**

- lon/lat vs lat/lon
- WGS84 vs Web Mercator
- degrees vs meters
- bbox semantics
- geometry validity and precision

**What you do:**

- validate bbox inputs and coordinate ranges
- inspect simple geometries with Shapely
- write down project conventions for CRS and coordinate order
- trace a few common failure modes before moving on

**Current repo anchor:**

- this module is mostly a framing and correctness layer for the rest of the repo
- the capstone brief already includes important conventions in `dev-docs/MOCK_TEST.md`

**Why it comes first:**

- bad geospatial assumptions make every later module wrong, even if the code is otherwise clean

## Module 2: Data Contracts and Provider Boundaries

**Purpose:** Keep data-source-specific logic from leaking through the whole system.

**Focus:**

- provider interfaces
- feature shape consistency
- IDs, schema normalization, and boundary error handling
- format-specific tradeoffs across CSV, GeoJSON, and MVT-style outputs

**What you build:**

- provider implementations behind a common interface
- tests for lookup, streaming, and interface behavior
- clearer expectations for what a provider returns

**Current repo anchor:**

- `src/day02_oop/`

**Why it comes second:**

- once correctness rules are clear, the next job is to define clean boundaries before adding storage or API complexity

## Module 3: Storage and Querying

**Purpose:** Move beyond file scans toward realistic spatial query behavior.

**Focus:**

- bbox query cost
- spatial indexing basics
- R-tree as the current runnable path
- PostGIS as the next core production direction

**What you build:**

- a small indexed query example
- a benchmark comparing naive filtering to indexed lookup
- a clearer path from file-backed examples to database-backed geo services

**Current repo anchor:**

- `src/day05_data/`

**Current reality:**

- the repo currently supports file- and index-oriented examples better than it supports full PostGIS workflows
- PostGIS should be treated here as the next planned realism step, not as something already fully implemented

**Why it comes third:**

- APIs should sit on top of realistic query and storage assumptions, not raw in-memory scans alone

## Module 4: APIs and Tile Delivery

**Purpose:** Build service endpoints that look like real geospatial backend work.

**Focus:**

- FastAPI validation for spatial inputs
- streaming feature responses
- tile path validation and tile response behavior
- pagination, limits, content types, and cache-friendly API habits

**What you build:**

- a bbox endpoint with validation
- a tile-style endpoint with stronger path checking
- minimal service metrics
- better API behavior around response shape and input limits

**Current repo anchor:**

- `src/day03_api/`

**Current reality:**

- the repo already has a small FastAPI app and metrics endpoint
- tile serving is still basic and should be hardened rather than oversold

**Why it comes fourth:**

- after correctness, provider boundaries, and query behavior are in place, API design becomes much more grounded

## Module 5: Concurrency and External Integrations

**Purpose:** Use concurrency where it pays off: upstream calls, ingestion, and I/O-heavy boundaries.

**Focus:**

- async I/O
- bounded concurrency
- retries, backoff, and timeouts
- respecting upstream limits

**What you build:**

- a sequential vs concurrent fetch comparison
- retry and backoff behavior under controlled failures
- a clearer mental model for when async helps and when it does not

**Current repo anchor:**

- `src/day01_concurrency/`

**Why it comes here:**

- concurrency is important, but it should support a geospatial system you already understand rather than act as the conceptual entry point

## Module 6: Testing Geospatial Systems

**Purpose:** Make spatial correctness and service behavior testable.

**Focus:**

- API validation tests
- unit tests for provider and query logic
- geo-specific edge cases
- confidence in bbox, connectivity, and response-contract behavior

**What you build:**

- tests for provider behavior
- tests for API validation and content types
- stronger checks around bbox invariants and road-network logic

**Current repo anchor:**

- `src/day04_testing/`

**Important note:**

- testing should happen throughout the guide
- this module exists to concentrate the patterns, not to imply that earlier modules should wait to be tested

**Why it comes here:**

- by this point there are enough concrete artifacts to test in a meaningful way

## Module 7: Performance and Observability

**Purpose:** Measure and improve systems that already exist.

**Focus:**

- latency and throughput
- profiling and memory inspection
- basic metrics and structured logs
- understanding slow paths and cache behavior

**What you build:**

- metrics around API paths
- a small before/after performance comparison
- a profiling pass on one slow path

**Current repo anchor:**

- `src/day06_perf/`

**Current reality:**

- the strongest near-term observability path in this repo is basic metrics, profiling, and logging
- avoid reading this module as a promise of a full production observability stack

**Why it comes here:**

- performance work is most useful when there is already a service or query path worth measuring

## Module 8: Capstone

**Purpose:** Integrate the core path into one realistic backend geospatial project.

**Project:**

- road-network query service

**Minimum scope:**

- load and validate road data
- support bbox queries with pagination
- support connected-road lookup
- support a small update path with validation
- include tests, documentation, and a brief performance note

**Current repo anchor:**

- `src/day07_mock/mock_test/`
- `dev-docs/MOCK_TEST.md`

**Why this capstone:**

- it is small enough to finish
- it exercises correctness, querying, API design, testing, and operational judgment
- it is closer to real backend geospatial work than a notebook-only project

## Elective / Advanced Topics

These are useful extensions, but they should not dominate the core path:

- deeper PostGIS work: loading, indexing, EXPLAIN plans, and database-backed APIs
- GeoParquet and Arrow workflows
- tile packaging and delivery formats beyond the current repo path
- raster, COG, and STAC workflows
- H3 or S2
- deployment packaging and containerization
- authentication, authorization, and rate limiting
- production debugging labs with richer observability tooling

Use these after the core path is stable and code-backed.

## What Is Implemented Now vs Planned Expansion

### Implemented now

- async tile fetching with retries, bounded concurrency, and a mock tile server
- provider examples for several formats
- a small FastAPI app with basic spatial endpoints and metrics
- initial tests around API behavior and concurrency
- a simple road-network mock project

### Planned expansion

- stronger correctness-specific exercises at the start of the curriculum
- more realistic storage and query work, with PostGIS as a core next step
- better tile validation and delivery behavior
- stronger provider consistency and streaming behavior
- more substantial testing around geo edge cases
- a hardened capstone with clearer performance expectations

Rule for future additions:

- if a topic is presented as core, it should map to runnable code in this repo or be explicitly labeled as planned expansion

## Suggested Pace

- work one module at a time
- spend more time implementing and testing than reading
- keep notes on assumptions, tradeoffs, and geo-specific mistakes you corrected
- treat the capstone as the place where portfolio signal is created

## Key References

- FastAPI documentation
- Pytest documentation
- Shapely documentation
- PostGIS documentation
- Mapbox Vector Tile specification
- GeoParquet specification
