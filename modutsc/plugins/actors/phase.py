from typing import Dict, List
from modutsc.plugins.actors import Actor
from modutsc.scheduling.registry import register


@register("actor", "phase")
class PhaseActor(Actor):

    def setup(self, cfg: dict, env=None) -> None:
        # 如果传入了 env，优先从环境中获取真实的拓扑参数
        if env is not None:
            ids = env.ids()
            if ids:
                self._max_phase = max(env.phase_count(j) for j in ids)
            else:
                self._max_phase = cfg.get("max_phase", 4)
        else:
            self._max_phase = cfg.get("max_phase", 4)

    def translate(self, acts: List[dict]) -> Dict[str, int]:
        result = {}
        for a in acts:
            val = int(a["value"])
            result[a["agent_id"]] = val
        return result

    def dim(self) -> int:
        return self._max_phase
