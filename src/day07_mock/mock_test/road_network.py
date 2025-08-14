from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from shapely.geometry import LineString, box
from shapely.wkt import loads as load_wkt


@dataclass(frozen=True)
class Road:
    road_id: str
    name: str
    geometry: LineString
    speed_limit: int
    road_type: str
    last_updated: str


class RoadNetwork:
    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)
        self._roads: Dict[str, Road] = {}
        self._endpoints_index: Dict[Tuple[float, float], List[str]] = {}
        self._load()

    def _load(self) -> None:
        with self._csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    geom = load_wkt(row["geometry"]).simplify(0)
                    speed = int(row["speed_limit"]) if row.get("speed_limit") else 0
                    road = Road(
                        road_id=row["road_id"],
                        name=row.get("name", ""),
                        geometry=geom, 
                        speed_limit=speed,
                        road_type=row.get("road_type", "unknown"),
                        last_updated=row.get("last_updated", ""),
                    )
                    self._roads[road.road_id] = road
                    # index endpoints
                    start = tuple(road.geometry.coords[0])
                    end = tuple(road.geometry.coords[-1])
                    self._endpoints_index.setdefault(start, []).append(road.road_id)
                    self._endpoints_index.setdefault(end, []).append(road.road_id)
                except Exception:
                    continue

    def find_roads_in_bbox(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> List[Road]:
        query = box(min_lon, min_lat, max_lon, max_lat)
        return [r for r in self._roads.values() if r.geometry.intersects(query)]

    @lru_cache(maxsize=1024)
    def get_connected_roads(self, road_id: str) -> List[str]:
        road = self._roads.get(road_id)
        if not road:
            return []
        start = tuple(road.geometry.coords[0])
        end = tuple(road.geometry.coords[-1])
        connected = set(self._endpoints_index.get(start, [])) | set(self._endpoints_index.get(end, []))
        connected.discard(road_id)
        return sorted(connected)

    def update_road(self, road_id: str, speed_limit: int | None = None, road_type: str | None = None) -> Road | None:
        road = self._roads.get(road_id)
        if not road:
            return None
        new_speed = speed_limit if speed_limit is not None else road.speed_limit
        new_type = road_type if road_type is not None else road.road_type
        updated = Road(road_id=road.road_id, name=road.name, geometry=road.geometry, speed_limit=new_speed, road_type=new_type, last_updated=road.last_updated)
        self._roads[road_id] = updated
        self.get_connected_roads.cache_clear()
        return updated


