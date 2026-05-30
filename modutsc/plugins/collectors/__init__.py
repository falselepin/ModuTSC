from abc import ABC, abstractmethod
from typing import Optional


class Collector(ABC):

    @abstractmethod
    def setup(self, cfg: dict) -> None: ...

    @abstractmethod
    def push(self, transition: dict) -> None: ...

    @abstractmethod
    def ready(self) -> bool: ...

    @abstractmethod
    def pull(self) -> Optional[dict]: ...

    @abstractmethod
    def size(self) -> int: ...

    def clear(self) -> None:
        pass
