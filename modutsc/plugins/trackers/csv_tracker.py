import csv
import os
from typing import Dict

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


@register("tracker", "csv")
class CsvTracker(Tracker):
    def __init__(self):
        self._file = None
        self._writer = None
        self._header_written = False
        self._flush = False
        self._traffic = None

    def setup(self, cfg: dict) -> None:
        path = cfg.get("path", "metrics.csv")
        self._flush = cfg.get("flush", False)
        self._traffic = (
            _SumoStepKpiAccumulator() if _cfg_enable_traffic_kpi(cfg) else None
        )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=None)
        self._header_written = False

    def log(self, metrics: dict, step: int, *, ref_kind: str = "train") -> None:
        if self._writer is None:
            return
        row = {"ref_kind": ref_kind, "step": step, **metrics}
        if not self._header_written:
            self._writer.fieldnames = list(row.keys())
            self._writer.writeheader()
            self._header_written = True
        self._writer.writerow(row)
        if self._flush:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()

    def reset_episode_stats(self, env) -> None:
        if self._traffic is not None:
            self._traffic.reset(env)

    def accumulate_step(self, env) -> None:
        if self._traffic is not None:
            self._traffic.accumulate_step(env)

    def episode_kpi_dict(self) -> dict:
        return self._traffic.as_dict() if self._traffic is not None else {}
