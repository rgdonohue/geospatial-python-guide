from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Spatial API Drills")


class TilePath(BaseModel):
    z: int = Field(ge=0, le=22)
    x: int
    y: int


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


@app.get("/tiles/{z}/{x}/{y}.mvt")
async def get_tile(z: int, x: int, y: int):
    if z < 0 or z > 22:
        raise HTTPException(status_code=400, detail="Invalid zoom")
    # Demo stream of empty tile
    async def streamer():
        yield b""  # replace with file streaming in drill

    return StreamingResponse(streamer(), media_type="application/vnd.mapbox-vector-tile")


@app.get("/stream-features")
async def stream_features(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="Invalid bbox")

    async def streamer():
        # demo NDJSON
        yield b"{}\n"

    return StreamingResponse(streamer(), media_type="application/x-ndjson")

