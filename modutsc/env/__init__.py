from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any


class Env(ABC):
    """原子查询/控制接口。所有抽象方法是一对一 TraCI 查询。

    Observer / Reward / Tracker 在编排器的协调下通过此接口读取仿真数据。
    每个组件不持有 Env 引用——它由编排器在调用时传入。
    """

    @abstractmethod
    def ids(self) -> List[str]: ...

    @abstractmethod
    def phase_count(self, jid: str) -> int: ...

    @abstractmethod
    def launch(self, cfg: dict) -> None: ...

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None: ...

    @abstractmethod
    def step(self, cmds: Dict[str, int]) -> dict: ...

    @abstractmethod
    def time(self) -> float: ...

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def sim_min_expected(self) -> int: ...

    @abstractmethod
    def sim_arrived_count(self) -> int: ...

    @abstractmethod
    def sim_arrived_ids(self) -> List[str]: ...

    @abstractmethod
    def sim_departed_ids(self) -> List[str]: ...

    @abstractmethod
    def sim_departed_count(self) -> int: ...

    @abstractmethod
    def lane_vehicle_ids(self, lane_id: str) -> List[str]: ...

    @abstractmethod
    def lane_vehicle_count(self, lane_id: str) -> int: ...

    @abstractmethod
    def lane_halting_count(self, lane_id: str) -> int: ...

    @abstractmethod
    def lane_mean_speed(self, lane_id: str) -> float: ...

    @abstractmethod
    def lane_length(self, lane_id: str) -> float: ...

    @abstractmethod
    def lane_links(self, lane_id: str) -> List: ...

    @abstractmethod
    def vehicle_speed(self, veh_id: str) -> float: ...

    @abstractmethod
    def vehicle_waiting_time(self, veh_id: str) -> float: ...

    @abstractmethod
    def vehicle_lane_pos(self, veh_id: str) -> float: ...

    @abstractmethod
    def vehicle_gps_pos(self, veh_id: str) -> Tuple[float, float]: ...

    @abstractmethod
    def vehicle_depart_time(self, veh_id: str) -> float: ...

    @abstractmethod
    def all_vehicle_ids(self) -> List[str]: ...

    @abstractmethod
    def vehicle_total_count(self) -> int: ...

    @abstractmethod
    def tl_set_raw_state(self, jid: str, state_str: str) -> None: ...

    @abstractmethod
    def junction_pos(self, jid: str) -> Tuple[float, float]: ...

    @abstractmethod
    def all_edge_ids(self) -> List[str]: ...

    @abstractmethod
    def edge_lane_count(self, edge_id: str) -> int: ...

    @abstractmethod
    def controlled_lanes(self, jid: str) -> List[str]: ...

    @abstractmethod
    def incoming_lanes(self, jid: str) -> List[str]: ...

    @abstractmethod
    def recent_arrived_ids(self) -> List[str]: ...

    @abstractmethod
    def recent_departed_ids(self) -> List[str]: ...

    @abstractmethod
    def traffic_light_controlled_links(
        self, jid: str
    ) -> List[List[Tuple[str, str]]]:
        """TraCI: trafficlight.getControlledLinks"""

    @abstractmethod
    def traffic_light_state_string(self, jid: str) -> str:
        """TraCI: trafficlight.getRedYellowGreenState"""

    @abstractmethod
    def traffic_light_get_phase(self, jid: str) -> int:
        """TraCI: trafficlight.getPhase"""

    @abstractmethod
    def green_phase_indices(self, jid: str) -> List[int]: ...

    @abstractmethod
    def lane_waiting_time(self, lane_id: str) -> float:
        """TraCI: lane.getWaitingTime"""

    def all_incoming_lane_states(self) -> Dict[str, Dict[str, dict]]:
        """便捷批量查询：所有路口进口道的计数。

        返回 {jid: {lid: {num, waiting, wait_time, speed, length}, ...}, ...}
        默认实现循环调原子方法；仿真器实现可 override 做真正的批量优化。

        单个 lane 查询失败时静默跳过并打印一次警告——大型真实路网中
        偶有 SUMO 内部 lane 索引不一致，跳过不影响其余 lane 和 episode 继续。
        """
        result: Dict[str, Dict[str, dict]] = {}
        warned: set = set()
        for jid in self.ids():
            j_info: Dict[str, dict] = {}
            for lid in self.incoming_lanes(jid):
                try:
                    j_info[lid] = {
                        "num":      self.lane_vehicle_count(lid),
                        "waiting":  self.lane_halting_count(lid),
                        "wait_time": self.lane_waiting_time(lid),
                        "speed":    self.lane_mean_speed(lid),
                        "length":   self.lane_length(lid),
                    }
                except Exception as e:
                    if lid not in warned:
                        message = f"[env] lane query failed for '{lid}' ({e}); skipping"
                        print(message)
                        warned.add(lid)
            result[jid] = j_info
        return result