from modutsc.plugins.algorithms import Algorithm
from modutsc.scheduling.registry import register


@register("algorithm", "max_pressure")
class MaxPressureController(Algorithm):

    def setup(self, cfg: dict) -> None:
        self._act_dim = cfg.get("num_phase", 4)
        self._min_duration = cfg.get("min_duration", 10)
        self._step = 0
        self._last = 0

    def act(self, obs: dict) -> dict:
        env = obs.get("extras", {}).get("env", None)
        if env is None:
            return {"agent_id": obs["id"], "value": self._last}

        if self._step < self._min_duration:
            self._step += 1
            return {"agent_id": obs["id"], "value": self._last}

        self._step += 1

        best_phase = 0
        best_pressure = -float("inf")
        j_lanes = env.all_incoming_lane_states().get(obs["id"], {})
        in_lanes = list(sorted(j_lanes.keys()))

        for phase_idx in range(self._act_dim):
            pressure = 0.0
            for i, lid in enumerate(in_lanes):
                lane = j_lanes.get(lid, {})
                if i % self._act_dim == phase_idx % self._act_dim:
                    pressure += lane.get("num", 0) + lane.get("waiting", 0)
                else:
                    if lane.get("waiting", 0) > 0:
                        pressure -= lane.get("waiting", 0) * 0.5
            if pressure > best_pressure:
                best_pressure = pressure
                best_phase = phase_idx

        self._last = best_phase
        return {"agent_id": obs["id"], "value": best_phase}

    def params(self) -> dict:
        return {"step": self._step, "last": self._last}

    def load(self, p: dict) -> None:
        self._step = p.get("step", 0)
        self._last = p.get("last", 0)
