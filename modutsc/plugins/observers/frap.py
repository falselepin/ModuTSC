from typing import List
import numpy as np
from modutsc.plugins.observers import Observer
from modutsc.scheduling.registry import register


@register("observer", "frap")
class Frap(Observer):

    @staticmethod
    def _norm_defaults() -> dict:
        return {
            "num": {"div": 10.0, "clip_min": 0.0, "clip_max": 5.0},
            "waiting": {"div": 5.0, "clip_min": 0.0, "clip_max": 5.0},
            "wait_time": {"div": 100.0, "clip_min": 0.0, "clip_max": 5.0},
        }

    def setup(self, cfg: dict, env=None) -> None:
        self._features = cfg.get("features", ["num"])
        self._norm = self._resolve_norm(self._features, cfg)
        
        # 如果提供了 env，从环境获取拓扑参数
        if env is not None:
            ids = env.ids()
            if ids:
                self._num_phase = max(len(env.green_phase_indices(j)) for j in ids)
                self._num_lanelink = max(
                    len(env.traffic_light_controlled_links(j)) for j in ids
                )
            else:
                self._num_phase = cfg.get("num_phase", 4)
                self._num_lanelink = cfg.get("max_lanelinks", 4)
        else:
            self._num_phase = cfg.get("num_phase", 4)
            self._num_lanelink = cfg.get("max_lanelinks", 4)

    def observe(self, env) -> List[dict]:
        results: List[dict] = []
        lane_states = env.all_incoming_lane_states()
        
        # 使用固定的拓扑参数，确保维度一致性
        n_act = getattr(self, '_num_phase', 4)
        n_links = getattr(self, '_num_lanelink', n_act)
        
        for jid in env.ids():
            gp = env.green_phase_indices(jid)
            cur_phase = env.traffic_light_get_phase(jid)
            state_str = env.traffic_light_state_string(jid)
            is_yellow = 'y' in state_str.lower()
            phase_oh = [0.0] * n_act
            if cur_phase in gp:
                local = gp.index(cur_phase)
                if local < n_act:
                    phase_oh[local] = 1.0
            elif not is_yellow and gp:
                phase_oh[0] = 1.0
            controlled_links = env.traffic_light_controlled_links(jid)
            link_lanes: List[str] = []
            for grp in controlled_links[:n_links]:
                link_lanes.append(grp[0][0] if grp else "")
            # 如果实际 link 数少于 n_links，填充空字符串
            while len(link_lanes) < n_links:
                link_lanes.append("")
            j_lanes = lane_states.get(jid, {})
            link_feats: List[float] = []
            for lid in link_lanes:
                lane = j_lanes.get(lid, {}) if lid else {}
                for fn in self._features:
                    v = float(lane.get(fn, 0.0)) if lane else 0.0
                    v = self._normalize(fn, v)
                    link_feats.append(v)
            feats = np.array(phase_oh + link_feats, dtype=np.float32)
            mask = np.ones(n_act, dtype=np.float32)
            results.append({"id": jid, "features": feats, "mask": mask})
        return results

    def dim(self) -> int:
        # 返回动态计算的维度：每个信号灯的相位数 + link数 * 特征数
        return 0  # 保持向后兼容，实际维度在运行时确定

    def _resolve_norm(self, features: list, cfg: dict) -> dict:
        result = {}
        user_norm = cfg.get("norm") or {}
        for fn in features:
            default = self._norm_defaults().get(fn, {})
            override = user_norm.get(fn, {})
            result[fn] = {**default, **override}
        return result

    def _normalize(self, field_name: str, value: float) -> float:
        meta = self._norm.get(field_name, {})
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
