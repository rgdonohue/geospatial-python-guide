from pathlib import Path
from typing import Any, Iterable

from .base import MapDataProvider


class MVTProvider(MapDataProvider):
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def get_feature_by_id(self, feature_id: str) -> Any:
        # Placeholder: usually requires an attribute index external to the tile store.
        return None

    def stream_features(self) -> Iterable[bytes]:
        # Placeholder: yield raw MVT bytes from a directory for demo purposes.
        for p in self._directory.rglob("*.mvt"):
            yield p.read_bytes()


