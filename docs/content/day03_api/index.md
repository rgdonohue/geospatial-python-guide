# Day 03 - APIs and Tile Delivery

This module turns the repo's small FastAPI app into a practical lesson in geospatial API behavior.

The goal is not to simulate a large platform. The goal is to build a small service that:

- validates spatial inputs
- returns predictable content types
- exposes a tile-style endpoint and a feature-stream endpoint
- emits basic metrics
- gives you a realistic place to practice the API habits that matter in geospatial backend work

## What This Module Builds Now

In the current repo, this module gives you:

- a FastAPI app
- a `/metrics` endpoint with Prometheus counters and latency histograms
- a tile-style endpoint at `/tiles/{z}/{x}/{y}.mvt`
- a simple streamed feature endpoint at `/stream-features`

This is intentionally small. It is a drill app, not a full geospatial service.

## Current Repo Anchor

- `src/day03_api/app.py`
- `src/day04_testing/tests/test_smoke.py`
- `src/day04_testing/tests/test_day03_api_validation.py`

## Current Reality

What is implemented now:

- zoom validation on the tile endpoint
- basic bbox shape validation on the feature-stream endpoint
- correct content types for MVT-style and NDJSON responses
- request count and latency metrics

What is **not** fully implemented yet:

- real tile file serving
- tile coordinate validation for `x` and `y` based on `z`
- bounded or paginated feature responses
- structured request logging
- PostGIS-backed feature queries
- cache headers and stronger tile delivery behavior

Read this module as a practical API foundation, not as a finished service.

## Why This Module Comes Here

In the study plan, APIs come after:

1. spatial correctness
2. provider and data-contract boundaries
3. realistic query and storage thinking

That order matters. If your CRS assumptions, coordinate order, or query model are wrong, the API can look clean and still return bad results.

## Learning Goals

By the end of this module, you should be able to:

- explain what the current FastAPI app does and does not do
- validate common spatial inputs at the API boundary
- return the right content type for streamed features and tile-like responses
- describe the difference between a placeholder tile endpoint and a real tile-serving endpoint
- identify the next code changes needed to make the app more production-like

## Geospatial API Design in This Repo

This module uses two endpoint shapes because they represent common backend patterns:

- `/tiles/{z}/{x}/{y}.mvt`
- `/stream-features?min_lat=...&min_lon=...&max_lat=...&max_lon=...`

They are small, but they force you to think about:

- path validation vs query validation
- tile coordinate rules
- bbox inversion bugs
- streaming vs materializing responses
- content types that clients can rely on
- how to observe request volume and latency

## Code Walkthrough

### 1) FastAPI App Setup

The app is defined in `src/day03_api/app.py`:

```python
app = FastAPI(title="Spatial API Drills")
```

This is enough to give you:

- route registration
- generated OpenAPI docs
- a simple place to add validation and middleware

### 2) Metrics Middleware

The app already records:

- request counts by path, method, and status
- request latency by path and method

That is a good start for this repo. It gives you visibility without pretending the service already has a full observability stack.

```python
REQS = Counter("api_requests_total", "Requests", ["path", "method", "status"])
LAT = Histogram("api_request_seconds", "Request latency seconds", ["path", "method"])
```

And it exposes them at:

```python
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### 3) Tile Endpoint

Current endpoint:

```python
@app.get("/tiles/{z}/{x}/{y}.mvt")
async def get_tile(z: int, x: int, y: int):
    if z < 0 or z > 22:
        raise HTTPException(status_code=400, detail="Invalid zoom")

    async def streamer():
        yield b""

    return StreamingResponse(
        streamer(),
        media_type="application/vnd.mapbox-vector-tile",
    )
```

What this does well:

- uses a recognizable tile path shape
- validates zoom range
- returns the correct MVT media type

What it does **not** do yet:

- validate `x` and `y` against `0 <= coord < 2**z`
- check whether a tile exists
- stream a real file or byte payload
- return `404` for missing tiles
- set useful response headers such as `Content-Length` or cache headers

Treat it as a tile contract drill, not a real tile server.

### 4) Feature Streaming Endpoint

Current endpoint:

```python
@app.get("/stream-features")
async def stream_features(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="Invalid bbox")

    async def streamer():
        yield b"{}\n"

    return StreamingResponse(streamer(), media_type="application/x-ndjson")
```

What this does well:

- accepts bbox-style query parameters
- rejects inverted bounding boxes
- returns NDJSON content type

What it does **not** do yet:

- validate latitude and longitude ranges
- cap result size or support pagination
- stream real features
- attach feature counts or other useful response metadata

Still, it is a good drill for boundary validation and response shape.

## Running the Module

```bash
source .venv/bin/activate
uvicorn src.day03_api.app:app --reload
```

Useful URLs:

- API docs: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`
- Metrics: `http://localhost:8000/metrics`

Quick checks:

```bash
curl "http://localhost:8000/tiles/0/0/0.mvt"
curl "http://localhost:8000/stream-features?min_lat=0&min_lon=0&max_lat=1&max_lon=1"
curl "http://localhost:8000/metrics"
```

## Practical API Guidance

### Spatial Input Validation

At a minimum, geospatial APIs should validate:

- coordinate ranges
- bbox ordering
- tile zoom bounds
- tile coordinate bounds for the given zoom
- request size limits where large responses are possible

In this repo, the current app only validates part of that. The next step is to harden the boundary rather than add new endpoints.

### Bounding Box Handling

For bbox inputs, be explicit about:

- parameter order
- accepted coordinate system
- whether edges are inclusive
- whether large boxes are allowed

Current repo convention:

- query params are named `min_lat`, `min_lon`, `max_lat`, `max_lon`
- the capstone brief treats input coordinates as WGS84

That convention should stay consistent across docs and code.

### Response Shape and Content Types

The current app uses two response styles:

- `application/vnd.mapbox-vector-tile` for the tile-style endpoint
- `application/x-ndjson` for streamed feature output

That is a useful distinction:

- use explicit media types so downstream clients know what they are receiving
- avoid returning large spatial feature collections as unbounded JSON blobs
- prefer bounded responses or streaming when payload size can grow quickly

### Pagination or Bounded Responses

The current `/stream-features` endpoint is only a stub, but the design lesson is still important:

- spatial APIs should not assume “return everything”
- bounding boxes can still select very large result sets
- plan for `limit`, `offset`, or another bounded-response strategy once the endpoint returns real data

### Tile Path Validation

Tile endpoints need more than zoom validation.

For XYZ-style paths:

- `z` should be in a supported range
- `x` and `y` should satisfy `0 <= x,y < 2**z`

Without that check, the API accepts impossible tile coordinates and does unnecessary work.

### Metrics and Basic Observability

For this repo, basic observability means:

- request counts
- latency histograms
- stable route labels

That is enough for this stage. It gives you useful feedback without pretending you already need tracing, distributed log pipelines, or elaborate SRE tooling.

## Common Geospatial API Bugs

These are the bugs to watch for first:

- swapping lon/lat and lat/lon
- accepting an inverted bbox
- treating degrees like meters
- returning huge unbounded responses
- accepting invalid tile coordinates for a zoom level
- using the wrong content type for a response
- silently changing coordinate system assumptions between endpoints

If you fix those early, the rest of the API design gets much easier.

## Tests in This Repo

Current tests already check:

- tile zoom validation
- basic bbox validation
- response content types
- metrics exposure

See:

- `src/day04_testing/tests/test_smoke.py`
- `src/day04_testing/tests/test_day03_api_validation.py`

The next useful tests for this module are:

- tile `x` and `y` bounds by zoom
- bbox latitude and longitude range validation
- real NDJSON line assertions with multiple features
- `404` behavior for missing tiles once tile files are introduced

## Planned Expansion

These are good next steps, but they are not fully implemented in the current repo:

- request models for bbox validation
- real tile file streaming
- tile existence checks and `404` responses
- bounded feature responses with pagination
- structured logging for request context
- cache headers for tile responses
- PostGIS-backed feature queries

Those improvements would make this module a much more realistic geospatial API lesson without turning it into a large systems-design document.

## Exercises

### 1) Harden Tile Validation

Add validation for `x` and `y` based on `z`.

Success looks like:

- impossible tile coordinates return `400`
- tests cover boundary values for multiple zoom levels

### 2) Turn the Tile Stub Into Real File Streaming

Replace the empty byte stream with real file-backed behavior.

Success looks like:

- existing tiles stream bytes
- missing tiles return `404`
- response headers include at least `Content-Length`

### 3) Strengthen Bbox Validation

Move from inversion-only validation to full coordinate validation.

Success looks like:

- latitudes outside `[-90, 90]` return validation errors
- longitudes outside `[-180, 180]` return validation errors
- docs and code agree on bbox parameter names and order

### 4) Bound the Feature Response

Keep `/stream-features` from becoming an unbounded dump.

Success looks like:

- the endpoint accepts a simple limit or similar bound
- tests prove the response stays well-formed NDJSON

## Bottom Line

This module is a small but useful API drill.

It already gives you:

- a FastAPI app
- real middleware
- metrics
- geospatially-shaped endpoints

It does **not** yet give you a full tile server or a real feature query service. That is fine. The right next move is to harden the current boundary behavior before adding more complexity.
