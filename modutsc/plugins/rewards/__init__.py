from abc import ABC, abstractmethod
from typing import List


class Reward(ABC):

    @abstractmethod
    def setup(self, cfg: dict) -> None: ...

    @abstractmethod
    def compute(self, env) -> List[float]: ...

    def reset(self) -> None:
        pass
