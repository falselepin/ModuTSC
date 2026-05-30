"""自定义编排器: test_single — 简化版 SingleOrchestrator，用于系统测试。"""
import random
from typing import Dict, Any, List, Optional

from modutsc.orchestration import Orchestrator
from modutsc.scheduling.registry import register


@register("orchestrator", "test_single")
class TestSingleOrchestrator(Orchestrator):
    """简化版单智能体编排器，用于验证自定义插件上传→装配→启动全流程。"""

    def setup(self, env, observer, actor, reward, collector, algorithms, cfg,
              tracker=None, **kwargs) -> None:
        self._env = env
        self._observer = observer
        self._actor = actor
        self._reward = reward
        self._collector = collector
        self._algos = algorithms
        self._cfg = cfg
        self._tracker = tracker

        self._learn_every = cfg.get("learn_every", 1)
        self._sync_every = cfg.get("sync_every", 100)
        self._max_steps = cfg.get("max_decision_steps", cfg.get("max_steps", 720))
        self._epsilon = cfg.get("epsilon", 1.0)
        self._eps_min = cfg.get("epsilon_min", 0.01)
        self._eps_decay = cfg.get("epsilon_decay", 0.995)
        self._step = 0
        self._ep_num = 0

        # 绑定拓扑
        for algo in self._algos:
            bind = getattr(algo, "bind_topology", None)
            if bind is not None:
                bind(self._env)

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
        last_loss = None

        while True:
            if self._should_stop():
                break
            obs_list = self._observer.observe(self._env)

            # epsilon-greedy 选动作
            actions = []
            for obs in obs_list:
                if random.random() < self._epsilon:
                    a = self._random_action(obs)
                else:
                    for algo in self._algos:
                        algo.eval()
                    a = self._algos[0].act(obs)
                    for algo in self._algos:
                        algo.train()
                actions.append(a)

            result = self._env.step(self._actor.translate(actions))
            rews = self._reward.compute(self._env)
            next_obs = self._observer.observe(self._env)

            if self._tracker:
                self._tracker.accumulate_step(self._env)

            # 每 5 步采集一次车辆位置和信号灯状态供前端仿真视图渲染
            if hasattr(self, '_shared_state') and self._shared_state is not None and steps % 5 == 0:
                try:
                    import traci
                    vehicles = []
                    for veh_id in traci.vehicle.getIDList():
                        try:
                            pos = traci.vehicle.getPosition(veh_id)
                            vehicles.append({
                                "id": veh_id,
                                "x": pos[0],
                                "y": pos[1],
                                "angle": traci.vehicle.getAngle(veh_id),
                                "speed": traci.vehicle.getSpeed(veh_id),
                            })
                        except Exception:
                            pass
                    self._shared_state['vehicles'] = vehicles
                    signals = []
                    for tl_id in traci.trafficlight.getIDList():
                        try:
                            state = traci.trafficlight.getRedYellowGreenState(tl_id)
                            g_count = state.count('G') + state.count('g')
                            y_count = state.count('y')
                            r_count = state.count('r')
                            phase = "黄" if y_count > 0 else ("绿" if g_count > r_count else "红")
                            signals.append({"id": tl_id, "phase": phase, "state": state})
                        except Exception:
                            pass
                    self._shared_state['signals'] = signals
                except Exception:
                    pass

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

            # 定期学习
            if self._step % self._learn_every == 0:
                info = self._learn_batch()
                if info and "loss" in info:
                    last_loss = info["loss"]

            # 定期同步
            if self._step % self._sync_every == 0:
                for algo in self._algos:
                    algo.sync()

            if result["terminated"] or result["truncated"]:
                break
            if self._max_steps > 0 and steps >= self._max_steps:
                break

        self._epsilon = max(self._eps_min, self._epsilon * self._eps_decay)
        kpi = self._tracker.episode_kpi_dict() if self._tracker else {}

        metrics = {
            **kpi,
            "avg_reward": total_reward / max(steps, 1),
            "steps": steps,
            "sim_time": float(self._env.time()),
            "epsilon": self._epsilon,
            "episode": self._ep_num,
        }
        if last_loss is not None:
            metrics["total_loss"] = last_loss

        if self._tracker:
            self._tracker.log(metrics, self._ep_num, ref_kind="episode")
        return metrics

    def evaluate(self, steps: int) -> dict:
        self._env.reset()
        if self._tracker:
            self._tracker.reset_episode_stats(self._env)
        total_reward = 0.0
        decision_steps = 0
        for algo in self._algos:
            algo.eval()
        for _ in range(steps):
            if self._should_stop():
                break
            obs_list = self._observer.observe(self._env)
            actions = [self._algos[0].act(obs) for obs in obs_list]
            result = self._env.step(self._actor.translate(actions))
            rews = self._reward.compute(self._env)
            if self._tracker:
                self._tracker.accumulate_step(self._env)
            total_reward += sum(rews)
            decision_steps += 1
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

    def save(self, path: str) -> None:
        import pickle
        state = {
            "step": self._step, "epsilon": self._epsilon,
            "ep_num": self._ep_num,
            "algos": [a.params() for a in self._algos],
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._step = state["step"]
        self._epsilon = state["epsilon"]
        self._ep_num = state["ep_num"]
        for a, p in zip(self._algos, state["algos"]):
            a.load(p)

    def _learn_batch(self):
        """从收集器中拉取数据并训练，兼容 per_agent 模式。"""
        algo = self._algos[0]
        if getattr(self._collector, "per_agent_mode", False):
            for jid in self._env.ids():
                if self._collector.ready_for(jid):
                    batch = self._collector.pull_for(jid)
                    if batch is not None:
                        return algo.learn(batch)
        elif self._collector.ready():
            batch = self._collector.pull()
            if batch is not None:
                return algo.learn(batch)
        return None

    def _random_action(self, obs):
        count = self._env.phase_count(obs["id"])
        return {"agent_id": obs["id"], "value": random.randint(0, max(0, count - 1))}
