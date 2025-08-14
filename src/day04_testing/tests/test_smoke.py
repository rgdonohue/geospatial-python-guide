from fastapi.testclient import TestClient
from src.day03_api.app import app


def test_tiles_smoke():
    client = TestClient(app)
    resp = client.get("/tiles/0/0/0.mvt")
    assert resp.status_code == 200


def test_stream_features_invalid_bbox():
    client = TestClient(app)
    resp = client.get("/stream-features", params={"min_lat": 1, "min_lon": 0, "max_lat": 0, "max_lon": 1})
    assert resp.status_code == 400


