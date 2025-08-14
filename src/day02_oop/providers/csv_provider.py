import csv
from pathlib import Path
from typing import Any, Dict, Iterable

from .base import MapDataProvider


class CSVProvider(MapDataProvider):
    """Simple CSV-backed provider.

    Expects a header row and an ID column (default: 'id'). Rows are exposed as dicts.
    """

    def __init__(self, path: str | Path, id_field: str = "id") -> None:
        self._path = Path(path)
        self._id_field = id_field
        self._rows: list[Dict[str, Any]] = []
        self._index: dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        with self._path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._rows.append(row)
                key = str(row.get(self._id_field, ""))
                if key:
                    self._index[key] = row

    def get_feature_by_id(self, feature_id: str) -> Any:
        return self._index.get(str(feature_id))

    def stream_features(self) -> Iterable[dict]:
        yield from self._rows

