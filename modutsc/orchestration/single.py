import random
import numpy as np
from typing import Dict, Any, List, Optional
from modutsc.orchestration import Orchestrator
from modutsc.env import Env
from modutsc.plugins.observers import Observer
from modutsc.plugins.actors import Actor
from modutsc.plugins.rewards import Reward
from modutsc.plugins.collectors import Collector
from modutsc.scheduling.registry import register
from modutsc.plugins.trackers import Tracker


def _aggregate_learn_infos(
    infos: List[Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """多路口各自 learn：对本决策步内各 agent 的 loss 取平均（仅输出 mean）。"""
    rows = [x for x in infos if x]
    losses: List[float] = []
    for x in rows:
        if "loss" not in x:
            continue
        try:
            losses.append(float(x["loss"]))
        except (TypeError, ValueError):
            continue
    if not losses:
        return dict(rows[-1]) if rows else {}
    return {"loss": sum(losses) / len(losses)}


@register("orchestrator", "single")
class SingleOrchestrator(Orchestrator):

    __compatible_plugins__ = {
        "observer": ["frap", "flat_lane", "standard", "colight", "ma2c", "igrl_graph"],
        "actor": ["phase"],
        "algorithm": ["frap", "dqn", "colight", "ma2c_agent", "fixed_time", "max_pressure", "sotl"],
        "collector": ["replay"],
        "reward": ["composite", "queue"],
    }

    def setup(self, env: Env, observer: Observer, actor: Actor,
              reward: Reward, collector: Collector,
              algorithms: list, cfg: dict, tracker: Optional[Tracker] = None,
              **kwargs) -> None:
        self._env = env
        self._observer = observer
        self._actor = actor
        self._reward = reward
        self._collector = collector
        self._algos = algorithms
        self._cfg = cfg
        self._tracker = tracker

        n_alg, n_tl = len(self._algos), len(env.ids())
        if n_alg > 1 and n_alg != n_tl:
            raise ValueError(
                f"算法实例数 ({n_alg}) 与信号灯数 ({n_tl}) 不一致"
            )
        if (
            n_alg > 1
            and n_alg == n_tl
            and not getattr(collector, "per_agent_mode", False)
        ):
            raise ValueError(
                "每路口独立网络时请在 replay collector 中设置 per_agent: true"
            )

        self._learn_every = cfg.get("learn_every", 1)
        self._sync_every = cfg.get("sync_every", 100)
        if "max_decision_steps" in cfg:
            self._episode_decision_cap = int(cfg["max_decision_steps"])
        elif "max_steps" in cfg:
            self._episode_decision_cap = int(cfg["max_steps"])
        else:
            self._episode_decision_cap = 720
        # 仿真秒：>0 时按间隔打 train 日志；0 表示每次 learn 都打（调试用）
        self._train_log_interval_sim = float(
            cfg.get("train_log_interval_sim_sec", 300)
        )
        self._epsilon = cfg.get("epsilon", 1.0)
        self._eps_min = cfg.get("epsilon_min", 0.01)
        self._eps_decay = cfg.get("epsilon_decay", 0.995)
        self._step = 0
        self._ep_num = 0
        self._bind_algorithms_topology()

    def _bind_algorithms_topology(self) -> None:
        """单智能体：每个算法实例绑定环境拓扑，并校正 obs_dim。"""
        if not self._env.ids():
            return
        for algo in self._algos:
            bind = getattr(algo, "bind_topology", None)
            if bind is not None:
                bind(self._env)

        try:
            obs_list = self._observer.observe(self._env)
            if obs_list:
                real_obs_dim = obs_list[0]["features"].shape[-1]
                for algo in self._algos:
                    cur = getattr(algo, "_obs_dim", None)
                    if cur is not None and cur != real_obs_dim:
                        reinit = getattr(algo, "reinitialize_with_obs_dim", None)
                        if reinit is not None:
                            reinit(real_obs_dim)
        except Exception as e:
            print(f"[bind_topology] obs_dim correction skipped: {e}")

    def _learn_from_collector(self) -> List[Dict[str, Any]]:
        """单智能体从各路口经验中学习；per_agent 仅用于分路口存 buffer。"""
        infos: List[Dict[str, Any]] = []
        if len(self._algos) != 1:
            tl_ids = self._env.ids()
            if getattr(self._collector, "per_agent_mode", False):
                for idx, algo in enumerate(self._algos):
                    if idx >= len(tl_ids):
                        break
                    jid = tl_ids[idx]
                    if self._collector.ready_for(jid):
                        batch = self._collector.pull_for(jid)
                        if batch is not None:
                            info = algo.learn(batch)
                            if info:
                                infos.append(info)
            elif self._collector.ready():
                batch = self._collector.pull()
                if batch is not None:
                    for algo in self._algos:
                        info = algo.learn(batch)
                        if info:
                            infos.append(info)
            return infos

        algo = self._algos[0]
        if getattr(self._collector, "per_agent_mode", False):
            for jid in self._env.ids():
                if self._collector.ready_for(jid):
                    batch = self._collector.pull_for(jid)
                    if batch is not None:
                        info = algo.learn(batch)
                        if info:
                            infos.append(info)
        elif self._collector.ready():
            batch = self._collector.pull()
            if batch is not None:
                info = algo.learn(batch)
                if info:
                    infos.append(info)
        return infos

    def warmup(self, steps: int) -> dict:
        self._env.reset()
        collected = 0
        while collected < steps:
            if self._should_stop():
                break
            obs_list = self._observer.observe(self._env)
            actions = [self._random_action(obs) for obs in obs_list]
            result = self._env.step(self._actor.translate(actions))
            rews = self._reward.compute(self._env)
            next_obs = self._observer.observe(self._env)
            for obs, act, r, no in zip(obs_list, actions, rews, next_obs):
                self._collector.push({
                    "obs": obs["features"], "actions": act["value"],
                    "rewards": r, "next_obs": no["features"],
                    "terminated": 1.0 if result["terminated"] else 0.0,
                    "agent_id": obs["id"],
                })
                collected += 1
                if collected >= steps:
                    break
            if result["terminated"] or result["truncated"]:
                self._env.reset()
        return {"warmup_steps": collected, "buffer_size": self._collector.size()}

    def episode(self) -> dict:
        self._env.reset()
        self._reward.reset()
        if self._tracker:
            self._tracker.reset_episode_stats(self._env)
        total_reward = 0.0
        steps = 0
        self._ep_num += 1
        last_learn: Dict[str, Any] = {}
        pending_loss_sum = 0.0
        pending_loss_n = 0
        iv = self._train_log_interval_sim
        next_log_sim = iv if iv > 0 else None

        def register_learn(agg: Dict[str, Any]) -> None:
            nonlocal last_learn, pending_loss_sum, pending_loss_n, next_log_sim
            last_learn = agg
            if not self._tracker or "loss" not in agg:
                return
            lv = float(agg["loss"])
            sim_t = float(self._env.time())
            if iv <= 0:
                self._tracker.log(
                    {"loss": lv, "sim_time": int(sim_t)},
                    self._step,
                    ref_kind="train",
                )
                return
            pending_loss_sum += lv
            pending_loss_n += 1
            while next_log_sim is not None and sim_t >= next_log_sim:
                if pending_loss_n > 0:
                    mean = pending_loss_sum / pending_loss_n
                    self._tracker.log(
                        {"loss": mean, "sim_time": int(sim_t)},
                        self._step,
                        ref_kind="train",
                    )
                    pending_loss_sum = 0.0
                    pending_loss_n = 0
                next_log_sim += iv

        result = None
        try:
            while True:
                if self._should_stop():
                    break
                obs_list = self._observer.observe(self._env)

                actions = []
                for idx, obs in enumerate(obs_list):
                    if random.random() < self._epsilon:
                        count = self._env.phase_count(obs["id"])
                        a = {"agent_id": obs["id"], "value": random.randint(0, max(0, count - 1))}
                    else:
                        for algo in self._algos:
                            algo.eval()
                        pick = self._algos[0] if len(self._algos) == 1 else self._algos[idx]
                        a = pick.act(obs)
                        for algo in self._algos:
                            algo.train()
                    actions.append(a)

                result = self._env.step(self._actor.translate(actions))
                rews = self._reward.compute(self._env)
                next_obs = self._observer.observe(self._env)

                if self._tracker:
                    self._tracker.accumulate_step(self._env)

                # 将车辆位置和信号灯状态写入 shared_state
                if hasattr(self, '_shared_state') and self._shared_state is not None and steps % 5 == 0:
                    self._shared_state['vehicles'] = self._collect_vehicles()
                    self._shared_state['signals'] = self._collect_signals()

                for obs, act, r, no in zip(obs_list, actions, rews, next_obs):
                    self._collector.push({
                        "obs": obs["features"], "actions": act["value"],
                        "rewards": r, "next_obs": no["features"],
                        "terminated": 1.0 if result["terminated"] else 0.0,
                        "agent_id": obs["id"],
                    })
                total_reward += sum(rews)
                steps += 1
                self._step += 1

                if self._step % self._learn_every == 0:
                    infos = self._learn_from_collector()
                    if infos:
                        register_learn(_aggregate_learn_infos(infos))

                if self._step % self._sync_every == 0:
                    for algo in self._algos:
                        algo.sync()

                if result["terminated"] or result["truncated"]:
                    break
                if self._episode_decision_cap > 0 and steps >= self._episode_decision_cap:
                    break
        except Exception as e:
            print(f"[episode] error at step {steps}: {e}")
            result = {"terminated": True, "truncated": True}
            self._env_error = True
        else:
            self._env_error = False

        if not self._env_error:
            if (
                self._tracker
                and iv > 0
                and pending_loss_n > 0
            ):
                mean = pending_loss_sum / pending_loss_n
                self._tracker.log(
                    {
                        "loss": mean,
                        "sim_time": int(self._env.time()),
                    },
                    self._step,
                    ref_kind="train",
                )

            self._epsilon = max(self._eps_min, self._epsilon * self._eps_decay)

            n_lanes = 0
            try:
                if hasattr(self._env, 'lane_halting_count') and hasattr(self._env, 'incoming_lanes'):
                    qh = sum(
                        self._env.lane_halting_count(lid)
                        for jid in self._env.ids()
                        for lid in self._env.incoming_lanes(jid)
                    )
                    n_lanes = sum(
                        len(self._env.incoming_lanes(jid))
                        for jid in self._env.ids()
                    )
                else:
                    qh = 0
                    n_lanes = 1
            except Exception:
                qh = 0
                n_lanes = 1
            kpi = self._tracker.episode_kpi_dict() if self._tracker else {}

            sim_time_end = float(self._env.time())
        else:
            self._epsilon = max(self._eps_min, self._epsilon * self._eps_decay)
            n_lanes = 1
            qh = 0
            kpi = {}
            sim_time_end = 0.0
        metrics = {
            **kpi,
            "avg_reward": total_reward / max(steps, 1),
            "steps": steps,
            "sim_time": sim_time_end,
            "queue": qh / max(n_lanes, 1),
            "epsilon": self._epsilon,
            "episode": self._ep_num,
        }
        if last_learn:
            if "loss" in last_learn:
                metrics["total_loss"] = last_learn["loss"]
            metrics.update({k: v for k, v in last_learn.items()
                            if k in ("policy_loss", "value_loss", "entropy", "grad_norm")})
        if self._tracker:
            self._tracker.log(metrics, self._ep_num, ref_kind="episode")
        return metrics

    def evaluate(self, steps: int) -> dict:
        self._env.reset()
        if self._tracker:
            self._tracker.reset_episode_stats(self._env)
        total_reward = 0.0
        decision_steps = 0
        n_junctions = max(len(self._env.ids()), 1)
        for algo in self._algos:
            algo.eval()
        for _ in range(steps):
            if self._should_stop():
                break
            obs_list = self._observer.observe(self._env)
            actions = [
                self._algos[0].act(obs) if len(self._algos) == 1
                else self._algos[i].act(obs)
                for i, obs in enumerate(obs_list)
            ]
            result = self._env.step(self._actor.translate(actions))
            rews = self._reward.compute(self._env)
            total_reward += sum(rews)
            decision_steps += 1
            if self._tracker:
                self._tracker.accumulate_step(self._env)
            if result["terminated"] or result["truncated"]:
                break
        for algo in self._algos:
            algo.train()
        kpi = self._tracker.episode_kpi_dict() if self._tracker else {}
        ds = max(decision_steps, 1)
        return {
            **kpi,
            "avg_queue": kpi.get("AQL", 0.0),
            "avg_reward": total_reward / ds,
        }

    def _random_action(self, obs):
        count = self._env.phase_count(obs["id"])
        return {"agent_id": obs["id"], "value": random.randint(0, max(0, count - 1))}

    def _collect_vehicles(self) -> list:
        """收集所有车辆的位置信息，供前端仿真视图渲染。"""
        vehicles = []
        env = self._env
        try:
            import traci
            for veh_id in traci.vehicle.getIDList():
                try:
                    pos = traci.vehicle.getPosition(veh_id)
                    angle = traci.vehicle.getAngle(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    vehicles.append({
                        "id": veh_id,
                        "x": pos[0],
                        "y": pos[1],
                        "angle": angle,
                        "speed": speed,
                    })
                except Exception:
                    pass
        except Exception:
            pass
        return vehicles

    def _collect_signals(self) -> list:
        """收集所有信号灯的当前状态，供前端仿真视图渲染。"""
        signals = []
        env = self._env
        try:
            import traci
            # 使用 traci.trafficlight.getIDList() 获取真正的信号灯ID列表
            for tl_id in traci.trafficlight.getIDList():
                try:
                    state = traci.trafficlight.getRedYellowGreenState(tl_id)
                    # 统计各颜色数量，取最多的作为当前相位
                    g_count = state.count('G') + state.count('g')
                    y_count = state.count('y')
                    r_count = state.count('r')
                    if y_count > 0:
                        phase = "黄"
                    elif g_count > r_count:
                        phase = "绿"
                    else:
                        phase = "红"
                    signals.append({
                        "id": tl_id,
                        "phase": phase,
                        "state": state,
                    })
                except Exception:
                    pass
        except Exception:
            pass
        return signals

    def save(self, path: str) -> None:
        import pickle
        state = {"step": self._step, "epsilon": self._epsilon, "ep_num": self._ep_num,
                 "algos": [a.params() for a in self._algos]}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._step = state["step"]; self._epsilon = state["epsilon"]; self._ep_num = state["ep_num"]
        for a, p in zip(self._algos, state["algos"]):
            a.load(p)

    def teardown(self) -> None:
        self._env.close()
