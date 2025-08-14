import json
from pathlib import Path
from typing import Any, Iterable

from .base import MapDataProvider


class GeoJSONProvider(MapDataProvider):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data = json.loads(self._path.read_text())
        self._index = {f["properties"].get("id"): f for f in self._data.get("features", [])}

    def get_feature_by_id(self, feature_id: str) -> Any:
        return self._index.get(feature_id)

    def stream_features(self) -> Iterable[dict]:
        yield from self._data.get("features", [])


