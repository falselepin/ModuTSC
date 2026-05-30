from abc import ABC, abstractmethod
from typing import Dict, List, Any


class Actor(ABC):

    @abstractmethod
    def setup(self, cfg: dict) -> None: ...

    @abstractmethod
    def translate(self, acts: List[dict]) -> Dict[str, int]: ...

    def dim(self) -> int:
        return 4

    def mask(self, raw: list) -> List[List[bool]]:
        return []
