from fastapi.testclient import TestClient
from src.day07_mock.mock_test.api import app


def test_bbox_validation():
    client = TestClient(app)
    r = client.get("/roads/bbox", params={"min_lat": 1, "min_lon": 0, "max_lat": 0, "max_lon": 1})
    assert r.status_code == 400


