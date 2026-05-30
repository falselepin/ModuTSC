import numpy as np
from typing import Optional, List
from modutsc.plugins.collectors import Collector
from modutsc.scheduling.registry import register


@register("collector", "ma2c")
class OnPolicyCollector(Collector):
    """??????????????????????????????????????"""

    def setup(self, cfg: dict) -> None:
        self._batch_size = cfg.get("batch_size", 120)
        self._n_agent = cfg.get("n_agent", 1)
        self._buffer = {
            'wave': [],
            'wait': [],
            'fp': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': []
        }

    def push(self, transition: dict) -> None:
        """transition ???????? agent ??????????"""
        for key in self._buffer:
            self._buffer[key].append(transition[key])

    def ready(self) -> bool:
        return len(self._buffer['dones']) >= self._batch_size

    def pull(self) -> Optional[List[dict]]:
        if not self.ready():
            return None

        T = self._batch_size
        # ???? per?agent ????
        batch_per_agent = []
        for i in range(self._n_agent):
            traj = {}
            # ?????? agent ???
            traj['wave'] = np.stack([self._buffer['wave'][t][i] for t in range(T)])
            traj['wait'] = np.stack([self._buffer['wait'][t][i] for t in range(T)])
            traj['fp'] = np.stack([self._buffer['fp'][t][i] for t in range(T)])
            traj['actions'] = np.array([self._buffer['actions'][t][i] for t in range(T)], dtype=np.int64)
            traj['rewards'] = np.array([self._buffer['rewards'][t][i] for t in range(T)], dtype=np.float32)
            traj['values'] = np.array([self._buffer['values'][t][i] for t in range(T)], dtype=np.float32)
            traj['dones'] = np.array([self._buffer['dones'][t] for t in range(T)], dtype=np.float32)
            batch_per_agent.append(traj)

        self.clear()
        return batch_per_agent

    def clear(self) -> None:
        for key in self._buffer:
            self._buffer[key].clear()

    def flush(self) -> Optional[List[dict]]:
        T = self.size()
        if T == 0:
            return None
        batch_per_agent = []
        for i in range(self._n_agent):
            traj = {}
            traj['wave'] = np.stack([self._buffer['wave'][t][i] for t in range(T)])
            traj['wait'] = np.stack([self._buffer['wait'][t][i] for t in range(T)])
            traj['fp'] = np.stack([self._buffer['fp'][t][i] for t in range(T)])
            traj['actions'] = np.array([self._buffer['actions'][t][i] for t in range(T)], dtype=np.int64)
            traj['rewards'] = np.array([self._buffer['rewards'][t][i] for t in range(T)], dtype=np.float32)
            traj['values'] = np.array([self._buffer['values'][t][i] for t in range(T)], dtype=np.float32)
            traj['dones'] = np.array([self._buffer['dones'][t] for t in range(T)], dtype=np.float32)
            batch_per_agent.append(traj)
        self.clear()
        return batch_per_agent

    def size(self) -> int:
        return len(self._buffer['dones'])
    
class LinearScheduler:
    def __init__(self, init_val, min_val=0.0, total_step=1):
        self.init_val = init_val
        self.min_val = min_val
        self.total_step = total_step
        self.cur_step = 0

    def get(self, step):
        self.cur_step += step
        frac = min(1.0, self.cur_step / self.total_step)
        return self.init_val - frac * (self.init_val - self.min_val)