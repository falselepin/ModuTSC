import numpy as np
from typing import Dict, List, Optional
from modutsc.plugins.collectors import Collector
from modutsc.scheduling.registry import register


def _empty_buffer() -> dict:
    return {
        "obs": [],
        "actions": [],
        "rewards": [],
        "next_obs": [],
        "terminated": [],
    }


@register("collector", "replay")
class ReplayCollector(Collector):

    def setup(self, cfg: dict) -> None:
        self._capacity = cfg.get("capacity", 6000)
        self._batch_size = cfg.get("batch_size", 64)
        self._per_agent = bool(cfg.get("per_agent", False))
        self._buffer = _empty_buffer()
        self._buffers: Dict[str, dict] = {}

    @property
    def per_agent_mode(self) -> bool:
        return self._per_agent

    def push(self, transition: dict) -> None:
        if self._per_agent:
            aid = transition.get("agent_id")
            if aid is None:
                raise ValueError("per_agent replay requires transition['agent_id']")
            aid = str(aid)
            buf = self._buffers.setdefault(aid, _empty_buffer())
            for key in buf:
                buf[key].append(transition[key])
            if len(buf["obs"]) > self._capacity:
                for key in buf:
                    buf[key].pop(0)
            return
        for key in self._buffer:
            self._buffer[key].append(transition[key])
        if len(self._buffer["obs"]) > self._capacity:
            for key in self._buffer:
                self._buffer[key].pop(0)

    def ready(self) -> bool:
        if self._per_agent:
            return any(self.ready_for(aid) for aid in self._buffers)
        return len(self._buffer["obs"]) >= self._batch_size

    def ready_for(self, agent_id: str) -> bool:
        if not self._per_agent:
            return self.ready()
        buf = self._buffers.get(str(agent_id))
        return buf is not None and len(buf["obs"]) >= self._batch_size

    def pull(self) -> Optional[dict]:
        if self._per_agent:
            raise RuntimeError(
                "ReplayCollector.pull() is not supported in per_agent mode. "
                "Use pull_for(agent_id) instead."
            )
        if not self.ready():
            return None
        indices = np.random.choice(
            len(self._buffer["obs"]), self._batch_size, replace=False
        )
        batch = {}
        for key in self._buffer:
            batch[key] = np.array(
                [self._buffer[key][i] for i in indices], dtype=np.float32
            )
        return batch

    def pull_for(self, agent_id: str) -> Optional[dict]:
        if not self._per_agent:
            return self.pull()
        aid = str(agent_id)
        if not self.ready_for(aid):
            return None
        buf = self._buffers[aid]
        n = len(buf["obs"])
        indices = np.random.choice(n, self._batch_size, replace=False)
        batch = {}
        for key in buf:
            batch[key] = np.array(
                [buf[key][i] for i in indices], dtype=np.float32
            )
        return batch

    def size(self) -> int:
        if self._per_agent:
            return sum(len(b["obs"]) for b in self._buffers.values())
        return len(self._buffer["obs"])

    def clear(self) -> None:
        for key in self._buffer:
            self._buffer[key].clear()
        self._buffers.clear()
