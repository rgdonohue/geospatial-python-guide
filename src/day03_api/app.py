from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Spatial API Drills")

MAX_ZOOM = 22
DEFAULT_FEATURE_LIMIT = 100
MAX_FEATURE_LIMIT = 1000

# Prometheus metrics
REQS = Counter("api_requests_total", "Requests", ["path", "method", "status"])
LAT = Histogram("api_request_seconds", "Request latency seconds", ["path", "method"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    route = request.scope.get("route")
    path_label = getattr(route, "path", request.url.path)
    method_label = request.method
    with LAT.labels(path_label, method_label).time():
        response = await call_next(request)
    REQS.labels(path_label, method_label, str(response.status_code)).inc()
    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _validate_tile_path(z: int, x: int, y: int) -> None:
    max_coord = 2**z
    if x >= max_coord:
        raise HTTPException(status_code=400, detail="Invalid tile x for zoom")
    if y >= max_coord:
        raise HTTPException(status_code=400, detail="Invalid tile y for zoom")


def _validate_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> None:
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="Invalid bbox")


@app.get("/tiles/{z}/{x}/{y}.mvt")
async def get_tile(
    z: Annotated[int, Path(ge=0, le=MAX_ZOOM)],
    x: Annotated[int, Path(ge=0)],
    y: Annotated[int, Path(ge=0)],
):
    _validate_tile_path(z, x, y)

    # Demo stream of empty tile
    async def streamer():
        # This short stub is finite; real or longer-lived streams should
        # include cooperative cancellation points while yielding data.
        yield b""  # replace with file streaming in drill

    return StreamingResponse(streamer(), media_type="application/vnd.mapbox-vector-tile")


@app.get("/stream-features")
async def stream_features(
    min_lat: Annotated[float, Query(ge=-90.0, le=90.0)],
    min_lon: Annotated[float, Query(ge=-180.0, le=180.0)],
    max_lat: Annotated[float, Query(ge=-90.0, le=90.0)],
    max_lon: Annotated[float, Query(ge=-180.0, le=180.0)],
    limit: Annotated[int, Query(ge=1, le=MAX_FEATURE_LIMIT)] = DEFAULT_FEATURE_LIMIT,
):
    _validate_bbox(min_lat, min_lon, max_lat, max_lon)

    async def streamer():
        # demo NDJSON bounded by limit
        for _ in range(limit):
            yield b"{}\n"

    return StreamingResponse(streamer(), media_type="application/x-ndjson")
