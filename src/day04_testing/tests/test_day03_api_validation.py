from fastapi.testclient import TestClient
from src.day03_api.app import app


def test_tile_invalid_zoom_low():
    client = TestClient(app)
    resp = client.get("/tiles/-1/0/0.mvt")
    assert resp.status_code == 400


def test_tile_invalid_zoom_high():
    client = TestClient(app)
    resp = client.get("/tiles/23/0/0.mvt")
    assert resp.status_code == 400


def test_tile_content_type():
    client = TestClient(app)
    resp = client.get("/tiles/0/0/0.mvt")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/vnd.mapbox-vector-tile")


def test_stream_features_valid_ndjson():
    client = TestClient(app)
    resp = client.get(
        "/stream-features",
        params={"min_lat": 0, "min_lon": 0, "max_lat": 1, "max_lon": 1},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    # One line of NDJSON
    assert resp.text.endswith("\n")


def test_metrics_endpoint_exposes_counters():
    client = TestClient(app)
    # Hit a route to increment counters
    client.get("/tiles/0/0/0.mvt")
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "api_requests_total" in metrics.text

