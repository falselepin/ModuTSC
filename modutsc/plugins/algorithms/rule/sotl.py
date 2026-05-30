from modutsc.plugins.algorithms import Algorithm
from modutsc.scheduling.registry import register


@register("algorithm", "sotl")
class SotlController(Algorithm):

    def setup(self, cfg: dict) -> None:
        self._act_dim = cfg.get("num_phase", 4)
        self._min_duration = cfg.get("min_duration", 10)
        self._min_green_veh = cfg.get("min_green_veh", 20)
        self._max_red_veh = cfg.get("max_red_veh", 0)
        self._step = 0
        self._last = 0

    def act(self, obs: dict) -> dict:
        raw = obs.get("extras", {}).get("raw", None)
        phase_2_lane_mask = obs.get("extras", {}).get("phase_2_lane_mask", None)

        if raw is None:
            return {"agent_id": obs["id"], "value": self._last}

        if self._step < self._min_duration:
            self._step += 1
            return {"agent_id": obs["id"], "value": self._last}

        self._step += 1
        action = self._last

        in_lanes = sorted(raw.in_lanes.keys())
        veh_counts = []
        for lid in in_lanes:
            lane = raw.in_lanes.get(lid)
            veh_counts.append(lane.num if lane else 0)

        if phase_2_lane_mask and len(phase_2_lane_mask) > self._last:
            mask = phase_2_lane_mask[self._last]
            green_count = sum(veh_counts[i] for i in range(min(len(veh_counts), len(mask))) if mask[i])
            red_count = sum(veh_counts[i] for i in range(min(len(veh_counts), len(mask))) if not mask[i])

            if green_count <= self._min_green_veh and red_count > self._max_red_veh:
                action = (action + 1) % self._act_dim

        self._last = action
        return {"agent_id": obs["id"], "value": action}

    def params(self) -> dict:
        return {"step": self._step, "last": self._last}

    def load(self, p: dict) -> None:
        self._step = p.get("step", 0)
        self._last = p.get("last", 0)
