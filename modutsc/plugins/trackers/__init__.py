from abc import ABC, abstractmethod
from typing import Any, Dict


class Tracker(ABC):
    @abstractmethod
    def log(self, metrics: dict, step: int, *, ref_kind: str = "train") -> None: ...
    @abstractmethod
    def close(self) -> None: ...

    def reset_episode_stats(self, env: Any) -> None:
        pass

    def accumulate_step(self, env: Any) -> None:
        pass

    def episode_kpi_dict(self) -> dict:
        return {}

    def episode_header(self, *, global_ep: int, epoch: int, full_cfg: Dict[str, Any]) -> None:
        pass

    def note(self, message: str) -> None:
        pass
