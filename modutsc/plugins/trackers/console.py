import os
from typing import Any, Dict, Optional
from modutsc.plugins.trackers import Tracker
from modutsc.scheduling.registry import register


def _aggregate_controlled_incoming_halting(env) -> int:
    total = 0
    for jid in env.ids():
        lanes = env.incoming_lanes(jid) if hasattr(env, 'incoming_lanes') else []
        for lid in lanes:
            total += int(env.lane_halting_count(lid))
    return total


def _trip_kpis_from_aggregates(
    total_queue: float,
    total_arrived_veh: int,
    total_departed_veh: int,
    decision_steps: int,
    n_lanes: int,
    decision_interval: float,
) -> Dict[str, float]:
    d_interval = max(float(decision_interval), 1.0)
    sim_steps = max(decision_steps * d_interval, 1.0)
    total_delay = total_queue * d_interval
    ds = max(decision_steps, 1)
    nl = max(n_lanes, 1)

    ATT = total_delay / max(total_departed_veh, 1) if total_departed_veh else 0.0
    AQL = total_queue / max(decision_steps * nl, 1)
    Throughput = total_arrived_veh / ds
    RealDelay = total_delay / total_arrived_veh if total_arrived_veh else 0.0
    TripFlow = total_arrived_veh / sim_steps

    return {
        "ATT": ATT,
        "AQL": AQL,
        "Throughput": Throughput,
        "RealDelay": RealDelay,
        "TripFlow": TripFlow,
        "departed": float(total_departed_veh),
        "arrived": float(total_arrived_veh),
    }


def _cfg_enable_traffic_kpi(cfg: dict) -> bool:
    if "traffic_episode_kpis" in cfg:
        return bool(cfg["traffic_episode_kpis"])
    return bool(cfg.get("sumo_episode_kpis", True))


class _SumoStepKpiAccumulator:
    """本模块内：按决策步在 SumoEnv 上累加 ATT/AQL 等指标。"""

    def __init__(self) -> None:
        self._total_queue = 0.0
        self._total_arrived = 0
        self._total_departed = 0
        self._decision_steps = 0
        self._n_lanes = 1
        self._d_interval = 1.0

    def reset(self, env) -> None:
        self._total_queue = 0.0
        self._total_arrived = 0
        self._total_departed = 0
        self._decision_steps = 0
        total_lanes = 0
        if env:
            for jid in env.ids():
                lanes = env.incoming_lanes(jid) if hasattr(env, 'incoming_lanes') else []
                total_lanes += len(lanes)
        self._n_lanes = max(total_lanes, 1)
        self._d_interval = float(getattr(env, '_decision_interval', 1.0))

    def accumulate_step(self, env) -> None:
        if env is None:
            return
        self._total_departed += len(env.recent_departed_ids())
        self._total_arrived += len(env.recent_arrived_ids())
        self._total_queue += float(_aggregate_controlled_incoming_halting(env))
        self._decision_steps += 1

    def as_dict(self) -> Dict[str, float]:
        return _trip_kpis_from_aggregates(
            self._total_queue,
            self._total_arrived,
            self._total_departed,
            self._decision_steps,
            self._n_lanes,
            self._d_interval,
        )


@register("tracker", "console")
class ConsoleTracker(Tracker):
    def __init__(self):
        self._prefix = ""
        self._flush = True
        self._traffic = None

    def setup(self, cfg: dict) -> None:
        self._prefix = cfg.get("prefix", "")
        self._flush = cfg.get("flush", True)
        self._traffic = (
            _SumoStepKpiAccumulator() if _cfg_enable_traffic_kpi(cfg) else None
        )

    _REF_LABEL = {"train": "train_step", "episode": "episode", "env_steps": "env_steps"}

    def log(self, metrics: dict, step: int, *, ref_kind: str = "train") -> None:
        label = self._REF_LABEL.get(ref_kind, "train_step")
        parts = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items())
        if self._prefix:
            msg = f"[{self._prefix} {label}={step}] {parts}"
        else:
            msg = f"[{label}={step}] {parts}"
        print(msg, flush=self._flush)

    def close(self) -> None:
        pass

    def reset_episode_stats(self, env) -> None:
        if self._traffic is not None:
            self._traffic.reset(env)

    def accumulate_step(self, env) -> None:
        if self._traffic is not None:
            self._traffic.accumulate_step(env)

    def episode_kpi_dict(self) -> dict:
        return self._traffic.as_dict() if self._traffic is not None else {}

    def episode_header(self, *, global_ep: int, epoch: int, full_cfg: Dict[str, Any]) -> None:
        exp = full_cfg.get("experiment") or {}
        env_c = (full_cfg.get("environment") or {}).get("config") or {}
        ename = exp.get("name", "")
        road = env_c.get("roadnet_file", "")
        flow = env_c.get("flow_file", "")
        sumo_cfg = env_c.get("sumo_cfg") or env_c.get("config_file")
        bits = []
        if ename:
            bits.append(f"experiment={ename}")
        if road:
            bits.append(f"roadnet={os.path.basename(str(road))}")
        if flow:
            bits.append(f"flow={os.path.basename(str(flow))}")
        if sumo_cfg:
            bits.append(f"sumo_cfg={os.path.basename(str(sumo_cfg))}")
        tag = " | ".join(bits) if bits else "environment.config 中无 roadnet/flow 路径"
        if self._prefix:
            print(f"[{self._prefix}] Episode {global_ep} (epoch={epoch})  {tag}", flush=self._flush)
        else:
            print(f"Episode {global_ep} (epoch={epoch})  {tag}", flush=self._flush)

    def note(self, message: str) -> None:
        print(message, flush=self._flush)
