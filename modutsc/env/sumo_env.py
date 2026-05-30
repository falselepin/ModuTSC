import os, sys, traci
import numpy as np
from sumolib import checkBinary
from typing import Dict, List, Optional, Tuple
from modutsc.env import Env
from modutsc.scheduling.registry import register


@register("environment", "sumo")
class SumoEnv(Env):

    def __init__(self):
        self._tls_ids: List[str] = []
        self._controlled_lanes: Dict[str, List[str]] = {}
        self._in_lanes: Dict[str, List[str]] = {}
        self._out_lanes: Dict[str, List[str]] = {}
        self._yellow_duration = 3
        self._min_green = 5
        self._decision_interval = 5
        self._max_time = 3600.0
        self._sumo_cfg: str = ""
        self._roadnet_file: str = ""
        self._flow_file: str = ""
        self._gui: bool = False

        # phase mapping
        self._green_phases: Dict[str, List[int]] = {}       # jid ?? [0,2,4,...] ?????????????
        self._phase_states: Dict[str, List[str]] = {}        # jid ?? ["GGgrrr", ...]
        self._all_phases: Dict[str, List[dict]] = {}         # jid ?? [{"state":...,"dur":...}]

        # yellow management
        self._green_to_yellow: Dict[str, Dict[int, int]] = {} # jid ?? {green_idx: yellow_idx}
        self._yellow_state: Dict[str, str] = {}               # jid ?? current yellow state string

        # per-junction state
        self._current_phase: Dict[str, int] = {}              # jid ?? current SUMO phase idx
        self._green_elapsed: Dict[str, int] = {}              # jid ?? steps since last green switch
        self._is_in_yellow: Dict[str, bool] = {}
        self._pending_phase: Dict[str, Optional[int]] = {}    # jid ?? target green phase after yellow

        # per-decision-step aggregated arrivals (accumulated across simulation steps)
        self._recent_arrived: List[str] = []
        self._recent_departed: List[str] = []

        self._tls_ids_filter: Optional[List[str]] = None

    # ===================================================================
    #  existing core methods
    # ===================================================================

    def ids(self) -> List[str]:
        return self._tls_ids

    def phase_count(self, jid: str) -> int:
        gp = self._green_phases.get(jid, [])
        return max(len(gp), 1)

    def launch(self, cfg: dict) -> None:
        self._max_time = cfg.get("sim_max_time", 3600)
        self._yellow_duration = cfg.get("yellow_duration", 3)
        self._min_green = cfg.get("min_green", 5)
        self._decision_interval = cfg.get("decision_interval", 5)
        self._sumo_cfg = cfg.get("sumo_cfg", "")
        self._roadnet_file = cfg.get("roadnet_file", "")
        self._flow_file = cfg.get("flow_file", "")
        self._tls_ids_filter = cfg.get("tls_ids")
        self._gui = cfg.get("gui", False)

        if 'SUMO_HOME' not in os.environ:
            raise EnvironmentError("SUMO_HOME not set")
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

        # 先清理任何现有的连接和残留进程
        self.close()

        binary = checkBinary('sumo-gui' if self._gui else 'sumo')
        if self._sumo_cfg:
            cmd = [binary, "-c", self._sumo_cfg]
        else:
            cmd = [binary]
        cmd += ["--start", "--quit-on-end",
                "--no-warnings", "--no-step-log", "--time-to-teleport", "-1"]
        if self._roadnet_file:
            cmd += ["-n", self._roadnet_file]
        if self._flow_file:
            cmd += ["-r", self._flow_file]

        traci.start(cmd)
        self._tls_ids = traci.trafficlight.getIDList()
        self._apply_tls_ids_filter()
        self._build_lane_topology()
        self._build_phase_mapping()

    def _apply_tls_ids_filter(self) -> None:
        if not self._tls_ids_filter:
            return
        want = {str(x) for x in self._tls_ids_filter}
        self._tls_ids = [j for j in self._tls_ids if j in want]
        if not self._tls_ids:
            raise ValueError("environment.config.tls_ids matched no traffic lights")

    def _build_lane_topology(self):
        in_map: Dict[str, List[str]] = {}
        out_map: Dict[str, List[str]] = {}
        all_out: set = set()
        for jid in self._tls_ids:
            try:
                controlled = traci.trafficlight.getControlledLanes(jid)
            except traci.exceptions.TraCIException:
                controlled = []
            in_map[jid] = []
            out_map[jid] = []
            for lid in controlled:
                try:
                    links = traci.lane.getLinks(lid)
                    for link in links:
                        to_lane = link[0]
                        if to_lane != "":
                            all_out.add(to_lane)
                except traci.exceptions.TraCIException:
                    pass
        for jid in self._tls_ids:
            try:
                controlled = traci.trafficlight.getControlledLanes(jid)
            except traci.exceptions.TraCIException:
                controlled = []
            for lid in controlled:
                if lid in all_out:
                    out_map[jid].append(lid)
                else:
                    in_map[jid].append(lid)
        self._controlled_lanes = {jid: in_map[jid] + out_map[jid]
                                   for jid in self._tls_ids}
        self._in_lanes = in_map
        self._out_lanes = out_map

    def _build_phase_mapping(self):
        self._green_phases = {}
        self._phase_states = {}
        self._all_phases = {}
        self._green_to_yellow = {}
        self._current_phase = {}
        self._green_elapsed = {}
        self._is_in_yellow = {}
        self._pending_phase = {}

        for jid in self._tls_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(jid)
            except Exception:
                logics = []
            if not logics:
                self._green_phases[jid] = list(range(4))
                self._green_to_yellow[jid] = {}
                self._current_phase[jid] = 0
                self._green_elapsed[jid] = 0
                self._is_in_yellow[jid] = False
                self._pending_phase[jid] = None
                continue

            phases = logics[0].getPhases()
            all_states = [p.state for p in phases]
            self._phase_states[jid] = all_states

            gp = []
            for i, s in enumerate(all_states):
                if any(ch in 'Gg' for ch in s) and not any(ch in 'yY' for ch in s):
                    gp.append(i)
            self._green_phases[jid] = gp if gp else [0]

            mapping = {}
            for gi in gp:
                state = all_states[gi]
                green_pos = [pos for pos, ch in enumerate(state) if ch in 'Gg']
                best_yellow = None
                best_overlap = 0
                for j, y_state in enumerate(all_states):
                    if any(ch in 'yY' for ch in y_state):
                        yellow_pos = [pos for pos, ch in enumerate(y_state) if ch in 'yY']
                        overlap = len(set(green_pos) & set(yellow_pos))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_yellow = j
                mapping[gi] = best_yellow if best_overlap > 0 else None
            self._green_to_yellow[jid] = mapping

            self._current_phase[jid] = gp[0]
            self._green_elapsed[jid] = 0
            self._is_in_yellow[jid] = False
            self._pending_phase[jid] = None

    def _get_yellow_state_str(self, jid: str, from_green: int) -> str:
        states = self._phase_states.get(jid, [])
        if from_green < len(states):
            return ''.join(['y' if ch in 'Gg' else ch for ch in states[from_green]])
        return ''

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            import random; random.seed(seed)
            import numpy as np; np.random.seed(seed)
        load_args = ["--start", "--no-warnings", "--no-step-log",
                     "--time-to-teleport", "-1"]
        if self._sumo_cfg:
            load_args = ["-c", self._sumo_cfg, "--start",
                         "--no-warnings", "--no-step-log",
                         "--time-to-teleport", "-1"]
        else:
            if self._roadnet_file:
                load_args += ["-n", self._roadnet_file]
            if self._flow_file:
                load_args += ["-r", self._flow_file]

        try:
            traci.load(load_args)
        except Exception as e:
            print(f"[sumo_env] reset error: {e}, attempting to reconnect...")
            try:
                traci.close()
            except Exception:
                pass
            # 重新启动 SUMO
            self.launch({
                "roadnet_file": self._roadnet_file,
                "flow_file": self._flow_file,
                "sumo_cfg": self._sumo_cfg,
                "gui": self._gui,
                "sim_max_time": self._max_time,
                "yellow_duration": self._yellow_duration,
                "min_green": self._min_green,
                "decision_interval": self._decision_interval,
            })
        self._tls_ids = traci.trafficlight.getIDList()
        self._apply_tls_ids_filter()
        self._build_lane_topology()
        self._build_phase_mapping()
        for jid in self._tls_ids:
            gp = self._green_phases.get(jid, [0])
            init_phase = gp[0] if gp else 0
            states = self._phase_states.get(jid, [])
            if init_phase < len(states):
                try:
                    traci.trafficlight.setRedYellowGreenState(jid, states[init_phase])
                except traci.exceptions.TraCIException:
                    pass
                self._current_phase[jid] = init_phase
        return None

    def step(self, cmds: Dict[str, int]) -> dict:
        decision_interval = self._decision_interval
        yellow_dur = self._yellow_duration
        states = self._phase_states
        self._recent_arrived.clear()
        self._recent_departed.clear()

        # 1. ???????? ?? ??? SUMO ????
        targets: Dict[str, int] = {}
        for jid, action in cmds.items():
            if jid not in self._green_phases:
                continue
            gp = self._green_phases[jid]
            action = int(action)
            if action < 0 or action >= len(gp):
                continue
            targets[jid] = gp[action]

        # 2. ??????????????????? & min_green ????
        switches: Dict[str, int] = {}
        for jid, target in targets.items():
            gp = self._green_phases.get(jid, [0])
            current = self._current_phase.get(jid, gp[0])
            s_list = states.get(jid, [""])
            cs = s_list[current] if current < len(s_list) else ""
            ts = s_list[target] if target < len(s_list) else ""
            if cs != ts and self._green_elapsed.get(jid, 0) >= self._min_green:
                switches[jid] = target

        # 3. ?????? ?? ???????????
        #    ???????? simulationStep ?????? setRedYellowGreenState??
        #    ???? SUMO ?????????????????????????????????????
        try:
            for sec in range(decision_interval):
                is_yellow = bool(switches) and sec < yellow_dur
                for jid in self._tls_ids:
                    if jid in switches:
                        if is_yellow:
                            s = self._get_yellow_state_str(jid, self._current_phase.get(jid, 0))
                        else:
                            target = switches[jid]
                            s_list = states.get(jid, [])
                            s = s_list[target] if target < len(s_list) else None
                    else:
                        cur = self._current_phase.get(jid, self._green_phases.get(jid, [0])[0])
                        s_list = states.get(jid, [])
                        s = s_list[cur] if cur < len(s_list) else None
                    if s is not None:
                        try:
                            traci.trafficlight.setRedYellowGreenState(jid, s)
                        except traci.exceptions.TraCIException:
                            pass
                traci.simulationStep()
                self._recent_departed.extend(traci.simulation.getDepartedIDList())
                self._recent_arrived.extend(traci.simulation.getArrivedIDList())
        except Exception as e:
            print(f"[sumo_env] step error: {e}")
            return {"terminated": True, "truncated": True}

        # 4. ??????????????
        if switches:
            for jid in switches:
                self._current_phase[jid] = switches[jid]
                self._green_elapsed[jid] = 0
        for jid in self._tls_ids:
            if jid not in switches:
                self._green_elapsed[jid] = self._green_elapsed.get(jid, 0) + decision_interval

        try:
            terminated = traci.simulation.getMinExpectedNumber() <= 0
            truncated = self.time() >= self._max_time
        except Exception as e:
            print(f"[sumo_env] get termination error: {e}")
            return {"terminated": True, "truncated": True}
        return {"terminated": terminated, "truncated": truncated}

    def green_phase_indices(self, jid: str) -> List[int]:
        return list(self._green_phases.get(jid, [0]))

    def traffic_light_get_phase(self, jid: str) -> int:
        try:
            return traci.trafficlight.getPhase(jid)
        except traci.exceptions.TraCIException:
            gp = self._green_phases.get(jid, [0])
            return gp[0] if gp else 0

    def lane_waiting_time(self, lid: str) -> float:
        try:
            return traci.lane.getWaitingTime(lid)
        except traci.exceptions.TraCIException:
            return 0.0

    def all_incoming_lane_states(self) -> Dict[str, Dict[str, dict]]:
        result: Dict[str, Dict[str, dict]] = {}
        for jid in self._tls_ids:
            j_info: Dict[str, dict] = {}
            for lid in self._in_lanes.get(jid, []):
                j_info[lid] = {
                    "num":      self.lane_vehicle_count(lid),
                    "waiting":  self.lane_halting_count(lid),
                    "wait_time": self.lane_waiting_time(lid),
                    "speed":    self.lane_mean_speed(lid),
                    "length":   self.lane_length(lid),
                }
            result[jid] = j_info
        return result

    def time(self) -> float:
        try:
            return traci.simulation.getTime()
        except Exception:
            return 0.0

    def traffic_light_controlled_links(
        self, jid: str
    ) -> List[List[Tuple[str, str]]]:
        try:
            raw = traci.trafficlight.getControlledLinks(jid)
        except traci.exceptions.TraCIException:
            return []
        out: List[List[Tuple[str, str]]] = []
        for group in raw:
            row: List[Tuple[str, str]] = []
            for link in group:
                if not link:
                    continue
                inc = str(link[0]) if link[0] else ""
                out_lane = str(link[1]) if len(link) > 1 and link[1] else ""
                row.append((inc, out_lane))
            if row:
                out.append(row)
        return out

    def traffic_light_state_string(self, jid: str) -> str:
        try:
            return traci.trafficlight.getRedYellowGreenState(jid)
        except traci.exceptions.TraCIException:
            return ""

    def close(self) -> None:
        try:
            traci.close(wait=True)
        except Exception:
            pass
        if os.name == 'nt':
            try:
                import subprocess
                subprocess.run(
                    ['taskkill', '/F', '/IM', 'sumo.exe'],
                    capture_output=True, text=True
                )
                subprocess.run(
                    ['taskkill', '/F', '/IM', 'sumo-gui.exe'],
                    capture_output=True, text=True
                )
            except Exception:
                pass

    def done(self) -> bool:
        """是否应结束回合：由 sim_min_expected 与配置 sim_max_time 组合判断（非单一 TraCI）。"""
        if traci.simulation.getMinExpectedNumber() <= 0:
            return True
        if self.time() >= self._max_time:
            return True
        return False

    # ---- simulation query ----
    def sim_min_expected(self) -> int:
        return traci.simulation.getMinExpectedNumber()
    def sim_arrived_count(self) -> int:
        return traci.simulation.getArrivedNumber()
    def sim_arrived_ids(self) -> List[str]:
        return traci.simulation.getArrivedIDList()
    def sim_departed_ids(self) -> List[str]:
        return traci.simulation.getDepartedIDList()
    def sim_departed_count(self) -> int:
        return traci.simulation.getDepartedNumber()

    # ---- lane query ----
    def lane_vehicle_ids(self, lane_id: str) -> List[str]:
        return list(traci.lane.getLastStepVehicleIDs(lane_id))
    def lane_vehicle_count(self, lane_id: str) -> int:
        return traci.lane.getLastStepVehicleNumber(lane_id)
    def lane_halting_count(self, lane_id: str) -> int:
        return traci.lane.getLastStepHaltingNumber(lane_id)
    def lane_mean_speed(self, lane_id: str) -> float:
        return traci.lane.getLastStepMeanSpeed(lane_id)
    def lane_length(self, lane_id: str) -> float:
        return traci.lane.getLength(lane_id)
    def lane_links(self, lane_id: str) -> List:
        return list(traci.lane.getLinks(lane_id))

    # ---- vehicle query ----
    def vehicle_speed(self, veh_id: str) -> float:
        return traci.vehicle.getSpeed(veh_id)
    def vehicle_waiting_time(self, veh_id: str) -> float:
        return traci.vehicle.getWaitingTime(veh_id)
    def vehicle_lane_pos(self, veh_id: str) -> float:
        return traci.vehicle.getLanePosition(veh_id)
    def vehicle_gps_pos(self, veh_id: str) -> Tuple[float, float]:
        x, y = traci.vehicle.getPosition(veh_id)
        return (float(x), float(y))
    def vehicle_depart_time(self, veh_id: str) -> float:
        return traci.vehicle.getTimeSinceDeparture(veh_id)
    def all_vehicle_ids(self) -> List[str]:
        return list(traci.vehicle.getIDList())
    def vehicle_total_count(self) -> int:
        return traci.vehicle.getIDCount()

    # ---- tl query ----
    def tl_set_raw_state(self, jid: str, state_str: str) -> None:
        traci.trafficlight.setRedYellowGreenState(jid, state_str)

    # ---- topology query ----
    def junction_pos(self, jid: str) -> Tuple[float, float]:
        x, y = traci.junction.getPosition(jid)
        return (float(x), float(y))
    def all_edge_ids(self) -> List[str]:
        return list(traci.edge.getIDList())
    def edge_lane_count(self, edge_id: str) -> int:
        return traci.edge.getLaneNumber(edge_id)
    def controlled_lanes(self, jid: str) -> List[str]:
        return list(self._controlled_lanes.get(jid, []))

    def incoming_lanes(self, jid: str) -> List[str]:
        return list(self._in_lanes.get(jid, []))

    def recent_arrived_ids(self) -> List[str]:
        """本决策步内各子步 getArrivedIDList 的累积（非单次 TraCI 快照）。"""
        return list(self._recent_arrived)

    def recent_departed_ids(self) -> List[str]:
        """本决策步内各子步 getDepartedIDList 的累积。"""
        return list(self._recent_departed)
