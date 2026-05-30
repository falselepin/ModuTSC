from typing import List
import numpy as np
from modutsc.plugins.observers import Observer
from modutsc.scheduling.registry import register


class FlatLaneObserver(Observer):
    __default_features__ = ["num", "waiting", "wait_time"]
    __prepend_phase__ = False

    @staticmethod
    def __config_defaults__() -> dict:
        return {
            "num":               {"div": 10.0, "clip_min": 0.0, "clip_max": 5.0},
            "waiting":           {"div": 5.0,  "clip_min": 0.0, "clip_max": 5.0},
            "wait_time":         {"div": 100.0,"clip_min": 0.0, "clip_max": 5.0},
            "speed":             {"div": 15.0, "clip_min": 0.0, "clip_max": 2.0},
        }

    def setup(self, cfg: dict) -> None:
        self._features = cfg.get("features", self.__default_features__)
        self._prepend_phase = cfg.get("prepend_phase", self.__prepend_phase__)
        self._norm = self._resolve_norm(self._features, cfg)

    def observe(self, env) -> List[dict]:
        n_feat = len(self._features)
        results = []
        lane_states = env.all_incoming_lane_states()
        for jid in env.ids():
            feats = []
            if self._prepend_phase:
                cur_phase = env.traffic_light_get_phase(jid)
                n_phase = env.phase_count(jid)
                for i in range(n_phase):
                    feats.append(1.0 if i == cur_phase else 0.0)
            j_lanes = sorted(lane_states.get(jid, {}).items())
            for lid, lane in j_lanes:
                for fn in self._features:
                    v = float(lane.get(fn, 0.0))
                    v = self._normalize(fn, v, self._norm)
                    feats.append(v)
            results.append({"id": jid, "features": np.array(feats, dtype=np.float32)})
        return results

    def dim(self) -> int:
        return 0

    def _resolve_norm(self, features: list, cfg: dict) -> dict:
        result = {}
        user_norm = cfg.get("norm") or {}
        for fn in features:
            default = self.__config_defaults__().get(fn, {})
            override = user_norm.get(fn, {})
            result[fn] = {**default, **override}
        return result

    def _normalize(self, field_name: str, value: float,
                   norm_table: dict) -> float:
        meta = norm_table.get(field_name, {})
        if "div" in meta:
            value = value / meta["div"]
        if "mean" in meta:
            value = (value - meta["mean"]) / meta.get("std", 1.0)
        lo = meta.get("clip_min")
        hi = meta.get("clip_max")
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value


for _kind, _name in [
    ("observer", "flat_lane"),
    ("observer", "standard"),
    ("observer", "colight"),
    ("observer", "ma2c"),
]:
    register(_kind, _name)(FlatLaneObserver)
