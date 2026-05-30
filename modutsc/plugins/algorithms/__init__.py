from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class Algorithm(ABC):
    __input_type__ = np.ndarray

    @abstractmethod
    def setup(self, cfg: dict) -> None: ...

    @abstractmethod
    def act(self, obs: dict) -> dict: ...

    def learn(self, batch: dict) -> dict:
        return {}

    def sync(self, tau: float = 1.0) -> None:
        pass

    def params(self) -> dict:
        return {}

    def load(self, p: dict) -> None:
        pass

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def bind_topology(self, env, jid: Optional[str] = None) -> None:
        pass
