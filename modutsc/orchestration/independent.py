"""
Independent multi-agent orchestrator (IQL-style).

Each traffic light is one agent: own observation, own policy, own replay buffer.
No neighbor observation, fingerprint, or reward mixing (unlike MA2C).
"""
import copy
import random
import numpy as np
from typing import Dict, Any, List, Optional

from modutsc.orchestration import Orchestrator
from modutsc.orchestration.single import _aggregate_learn_infos
from modutsc.env import Env
from modutsc.plugins.observers import Observer
from modutsc.plugins.actors import Actor
from modutsc.plugins.rewards import Reward
from modutsc.plugins.collectors import Collector
from modutsc.scheduling.registry import register
from modutsc.plugins.trackers import Tracker


@register("orchestrator", "independent")
class IndependentOrchestrator(Orchestrator):
    """经典独立多智能体编排：无通信，每路口独立网络与训练。"""

    __compatible_plugins__ = {
        "observer": ["frap", "flat_lane", "standard", "colight", "igrl_graph"],
        "actor": ["phase"],
        "algorithm": ["frap", "dqn", "colight"],
        "collector": ["replay"],
        "reward": ["composite", "queue"],
    }

    def setup(
        self,
        env: Env,
        observer: Observer,
        actor: Actor,
        reward: Reward,
        collector: Collector,
        algorithms: list,
        cfg: dict,
        tracker: Optional[Tracker] = None,
        **kwargs,
    ) -> None:
        self._env = env
        self._observer = observer
        self._actor = actor
        self._reward = reward
        self._collector = collector
        self._cfg = cfg
        self._tracker = tracker

        tl_ids = list(env.ids())
        n_tl = len(tl_ids)
        if n_tl < 1:
            raise ValueError("independent 编排器需要至少一个信号灯")

        if not getattr(collector, "per_agent_mode", False):
            raise ValueError(
                "independent 编排器要求 replay collector 设置 per_agent: true"
            )

        if not algorithms:
            raise ValueError("independent 编排器需要至少一个 algorithm 插件实例")

        n_alg = len(algorithms)
        if n_alg > 1 and n_alg != n_tl:
            raise ValueError(
                f"algorithm 实例数 ({n_alg}) 与信号灯数 ({n_tl}) 不一致；"
                "可只配置 1 个 algorithm（将按路口克隆），或配置 {n_tl} 个实例"
            )

        self._algos = self._build_agents(algorithms, tl_ids)

        self._learn_every = cfg.get("learn_every", 1)
        self._sync_every = cfg.get("sync_every", 100)
        if "max_decision_steps" in cfg:
            self._episode_decision_cap = int(cfg["max_decision_steps"])
        elif "max_steps" in cfg:
            self._episode_decision_cap = int(cfg["max_steps"])
        else:
            self._episode_decision_cap = 720
        self._train_log_interval_sim = float(
            cfg.get("train_log_interval_sim_sec", 300)
        )
        self._epsilon = cfg.get("epsilon", 1.0)
        self._eps_min = cfg.get("epsilon_min", 0.01)
        self._eps_decay = cfg.get("epsilon_decay", 0.995)
        self._step = 0
        self._ep_num = 0

    def _build_agents(self, algorithms: list, tl_ids: List[str]) -> list:
        n_tl = len(tl_ids)
        agents: list = []
        if len(algorithms) == 1:
            template = algorithms[0]
            for i, jid in enumerate(tl_ids):
                ag = template if i == 0 else copy.deepcopy(template)
                bind = getattr(ag, "bind_topology", None)
                if bind is not None:
                    bind(self._env, jid)
                agents.append(ag)
            return agents

        for ag, jid in zip(algorithms, tl_ids):
            bind = getattr(ag, "bind_topology", None)
            if bind is not None:
                bind(self._env, jid)
            agents.append(ag)
        return agents

    def _learn_from_collector(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for idx, algo in enumerate(self._algos):
            jid = self._env.ids()[idx]
            if not self._collector.ready_for(jid):
                continue
            batch = self._collector.pull_for(jid)
            if batch is None:
                continue
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
                    "obs": obs["features"],
                    "actions": act["value"],
                    "rewards": r,
                    "next_obs": no["features"],
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

                for algo in self._algos:
                    algo.eval()
                actions = [
                    self._algos[i].act(obs)
                    for i, obs in enumerate(obs_list)
                ]
                for algo in self._algos:
                    algo.train()

                result = self._env.step(self._actor.translate(actions))
                rews = self._reward.compute(self._env)
                next_obs = self._observer.observe(self._env)

                if self._tracker:
                    self._tracker.accumulate_step(self._env)

                for obs, act, r, no in zip(obs_list, actions, rews, next_obs):
                    self._collector.push({
                        "obs": obs["features"],
                        "actions": act["value"],
                        "rewards": r,
                        "next_obs": no["features"],
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
            print(f"[independent episode] error at step {steps}: {e}")
            result = {"terminated": True, "truncated": True}

        if self._tracker and iv > 0 and pending_loss_n > 0:
            mean = pending_loss_sum / pending_loss_n
            self._tracker.log(
                {"loss": mean, "sim_time": int(self._env.time())},
                self._step,
                ref_kind="train",
            )

        self._epsilon = max(self._eps_min, self._epsilon * self._eps_decay)

        n_j = max(len(self._env.ids()), 1)
        try:
            qh = sum(
                self._env.lane_halting_count(lid)
                for jid in self._env.ids()
                for lid in self._env.incoming_lanes(jid)
            )
        except Exception:
            qh = 0
        kpi = self._tracker.episode_kpi_dict() if self._tracker else {}

        metrics = {
            **kpi,
            "avg_reward": total_reward / max(steps, 1),
            "steps": steps,
            "sim_time": float(self._env.time()),
            "queue": qh / max(n_j, 1),
            "epsilon": self._epsilon,
            "episode": self._ep_num,
            "n_agents": len(self._algos),
        }
        if last_learn:
            if "loss" in last_learn:
                metrics["total_loss"] = last_learn["loss"]
            metrics.update(
                {
                    k: v
                    for k, v in last_learn.items()
                    if k in ("policy_loss", "value_loss", "entropy", "grad_norm")
                }
            )
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
            actions = [
                self._algos[i].act(obs)
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

    def _random_action(self, obs: dict) -> dict:
        count = self._env.phase_count(obs["id"])
        return {
            "agent_id": obs["id"],
            "value": random.randint(0, max(0, count - 1)),
        }

    def save(self, path: str) -> None:
        import pickle

        state = {
            "step": self._step,
            "epsilon": self._epsilon,
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
        for ag, params in zip(self._algos, state["algos"]):
            ag.load(params)

    def teardown(self) -> None:
        self._env.close()
