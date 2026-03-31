from fastapi.testclient import TestClient
from src.day03_api.app import app


client = TestClient(app)


def test_tile_invalid_zoom_low():
    resp = client.get("/tiles/-1/0/0.mvt")
    assert resp.status_code == 422


def test_tile_invalid_zoom_high():
    resp = client.get("/tiles/23/0/0.mvt")
    assert resp.status_code == 422


def test_tile_invalid_x_for_zoom():
    resp = client.get("/tiles/1/2/0.mvt")
    assert resp.status_code == 400


def test_tile_invalid_y_for_zoom():
    resp = client.get("/tiles/1/0/2.mvt")
    assert resp.status_code == 400


def test_tile_negative_x_rejected():
    resp = client.get("/tiles/1/-1/0.mvt")
    assert resp.status_code == 422


def test_tile_negative_y_rejected():
    resp = client.get("/tiles/1/0/-1.mvt")
    assert resp.status_code == 422


def test_tile_content_type():
    resp = client.get("/tiles/0/0/0.mvt")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/vnd.mapbox-vector-tile")


def test_stream_features_valid_ndjson():
    resp = client.get(
        "/stream-features",
        params={"min_lat": 0, "min_lon": 0, "max_lat": 1, "max_lon": 1, "limit": 2},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    assert resp.text == "{}\n{}\n"


def test_stream_features_invalid_latitude_range():
    resp = client.get(
        "/stream-features",
        params={"min_lat": -91, "min_lon": 0, "max_lat": 1, "max_lon": 1},
    )
    assert resp.status_code == 422


def test_stream_features_invalid_longitude_range():
    resp = client.get(
        "/stream-features",
        params={"min_lat": 0, "min_lon": -181, "max_lat": 1, "max_lon": 1},
    )
    assert resp.status_code == 422


def test_stream_features_invalid_limit_low():
    resp = client.get(
        "/stream-features",
        params={"min_lat": 0, "min_lon": 0, "max_lat": 1, "max_lon": 1, "limit": 0},
    )
    assert resp.status_code == 422


def test_stream_features_invalid_limit_high():
    resp = client.get(
        "/stream-features",
        params={"min_lat": 0, "min_lon": 0, "max_lat": 1, "max_lon": 1, "limit": 1001},
    )
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_counters():
    # Hit a route to increment counters
    client.get("/tiles/0/0/0.mvt")
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "api_requests_total" in metrics.text
