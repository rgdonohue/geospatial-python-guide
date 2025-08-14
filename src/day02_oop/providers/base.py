from abc import ABC, abstractmethod
from typing import Any


class MapDataProvider(ABC):
    @abstractmethod
    def get_feature_by_id(self, feature_id: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def stream_features(self) -> Any:
        raise NotImplementedError


