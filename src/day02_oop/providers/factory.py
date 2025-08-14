from __future__ import annotations

from typing import Any, Callable, Dict, Type

from .base import MapDataProvider
from .geojson_provider import GeoJSONProvider
from .mvt_provider import MVTProvider
from .csv_provider import CSVProvider


class ProviderFactory:
    """Minimal registry-based factory for providers.

    Example:
        provider = ProviderFactory.create("csv", path="roads.csv", id_field="id")
    """

    _registry: Dict[str, Type[MapDataProvider]] = {
        "geojson": GeoJSONProvider,
        "mvt": MVTProvider,
        "csv": CSVProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: Type[MapDataProvider]) -> None:
        cls._registry[name] = provider_cls

    @classmethod
    def create(cls, name: str, /, **kwargs: Any) -> MapDataProvider:
        try:
            provider_cls = cls._registry[name]
        except KeyError as e:
            raise ValueError(f"Unknown provider type: {name}") from e
        return provider_cls(**kwargs)

