from typing import List
import numpy as np
from modutsc.plugins.rewards import Reward
from modutsc.scheduling.registry import register


@register("reward", "composite")
class CompositeReward(Reward):

    def setup(self, cfg: dict) -> None:
        raw = cfg.get("metrics", {"waiting": -1.0})
        self._metrics = {k: v for k, v in raw.items() if v is not None} if isinstance(raw, dict) else {"waiting": -1.0}
        self._norm = cfg.get("reward_norm", 1.0) or 1.0
        self._prev_num: dict = {}
        self._prev_waiting: dict = {}

    def compute(self, env) -> List[float]:
        rewards = []
        lane_states = env.all_incoming_lane_states()
        for jid in env.ids():
            r = 0.0
            lanes = list(lane_states.get(jid, {}).values())

            for metric, weight in self._metrics.items():
                if metric in ("waiting", "queue"):
                    r += weight * sum(l.get("waiting", 0) for l in lanes)

                elif metric == "num":
                    r += weight * sum(l.get("num", 0) for l in lanes)

                elif metric == "wait_time":
                    r += weight * sum(l.get("wait_time", 0.0) for l in lanes)

                elif metric == "speed":
                    speeds = [l.get("speed", 0.0) for l in lanes if l.get("num", 0) > 0]
                    r += weight * (np.mean(speeds) if speeds else 0.0)

                elif metric == "pressure":
                    total_waiting = sum(l.get("waiting", 0) for l in lanes)
                    prev_w = self._prev_waiting.get(jid, total_waiting)
                    r += weight * (prev_w - total_waiting)
                    self._prev_waiting[jid] = total_waiting

                elif metric == "delay":
                    r += weight * sum(l.get("wait_time", 0.0) for l in lanes)

                elif metric == "throughput":
                    total_num = sum(l.get("num", 0) for l in lanes)
                    prev_n = self._prev_num.get(jid, total_num)
                    r += weight * max(0, prev_n - total_num)
                    self._prev_num[jid] = total_num

            r = r / max(self._norm, 1.0)
            rewards.append(float(r))
        return rewards

    def reset(self) -> None:
        self._prev_num.clear()
        self._prev_waiting.clear()
