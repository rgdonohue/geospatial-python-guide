from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .road_network import RoadNetwork, Road

DATA_PATH = Path(__file__).parent / "roads.csv"


def road_to_feature(road: Road) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "road_id": road.road_id,
            "name": road.name,
            "speed_limit": road.speed_limit,
            "road_type": road.road_type,
        },
        "geometry": road.geometry.__geo_interface__,
    }


@lru_cache(maxsize=1)
def get_network() -> RoadNetwork:
    return RoadNetwork(DATA_PATH)


class UpdateBody(BaseModel):
    speed_limit: int | None = Field(default=None, gt=0)
    road_type: Literal["highway", "arterial", "residential", "service"] | None = None


app = FastAPI(title="Mock Test API")


@app.get("/roads/bbox")
def roads_bbox(
    min_lat: float = Query(...),
    min_lon: float = Query(...),
    max_lat: float = Query(...),
    max_lon: float = Query(...),
    limit: int = Query(100, gt=0, le=1000),
    offset: int = Query(0, ge=0),
):
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="Invalid bbox")
    roads = get_network().find_roads_in_bbox(min_lat, min_lon, max_lat, max_lon)
    page = roads[offset : offset + limit]
    features = [road_to_feature(r) for r in page]
    return {"type": "FeatureCollection", "features": features, "count": len(roads)}


@app.get("/roads/{road_id}/connected")
def connected(road_id: str):
    ids = get_network().get_connected_roads(road_id)
    return ids


@app.post("/roads/{road_id}/update")
def update_road(road_id: str, body: UpdateBody):
    updated = get_network().update_road(road_id, speed_limit=body.speed_limit, road_type=body.road_type)
    if not updated:
        raise HTTPException(status_code=404, detail="Road not found")
    return road_to_feature(updated)


