from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np


class Observer(ABC):
    __output_type__ = np.ndarray

    @abstractmethod
    def setup(self, cfg: dict) -> None: ...

    @abstractmethod
    def observe(self, env) -> List[dict]: ...

    def dim(self) -> int:
        return 1

    def reset(self) -> None:
        pass
