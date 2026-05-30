from modutsc.plugins.algorithms import Algorithm
from modutsc.scheduling.registry import register


@register("algorithm", "fixed_time")
class FixedTimeController(Algorithm):

    def setup(self, cfg: dict) -> None:
        self._act_dim = cfg.get("num_phase", 4)
        self._interval = cfg.get("interval", 20)
        self._step = 0
        self._last = 0

    def act(self, obs: dict) -> dict:
        if self._step % self._interval == 0 and self._act_dim > 1:
            self._last = (self._step // self._interval) % self._act_dim
        self._step += 1
        return {"agent_id": obs["id"], "value": self._last}

    def params(self) -> dict:
        return {"step": self._step, "last": self._last}

    def load(self, p: dict) -> None:
        self._step = p.get("step", 0)
        self._last = p.get("last", 0)
