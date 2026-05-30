from typing import Optional
from modutsc.orchestration import Orchestrator
from modutsc.env import Env
from modutsc.plugins.observers import Observer
from modutsc.plugins.actors import Actor
from modutsc.plugins.rewards import Reward
from modutsc.plugins.collectors import Collector
from modutsc.scheduling.registry import register
from modutsc.plugins.trackers import Tracker


@register("orchestrator", "rule")
class RuleOrchestrator(Orchestrator):

    __compatible_plugins__ = {
        "observer": ["frap", "flat_lane", "standard", "colight", "ma2c", "igrl_graph"],
        "actor": ["phase"],
        "algorithm": ["fixed_time", "max_pressure", "sotl"],
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
        self._algos = algorithms
        self._cfg = cfg
        self._tracker = tracker
        self._max_steps = cfg.get("max_steps", 720)

        for algo in self._algos:
            algo.bind_topology(self._env)

    def warmup(self, steps: int) -> dict:
        return {"warmup_steps": 0}

    def episode(self) -> dict:
        self._env.reset()
        if self._tracker:
            self._tracker.reset_episode_stats(self._env)
        steps = 0
        total_reward = 0.0

        result = None
        try:
            while steps < self._max_steps:
                obs_list = self._observer.observe(self._env)

                for obs in obs_list:
                    obs.get("extras", {})["env"] = self._env

                actions = [a.act(o) for a, o in zip(self._algos, obs_list)]
                result = self._env.step(self._actor.translate(actions))
                rews = self._reward.compute(self._env)
                total_reward += sum(rews)
                steps += 1
                if self._tracker:
                    self._tracker.accumulate_step(self._env)

                if result["terminated"] or result["truncated"]:
                    break
        except Exception as e:
            print(f"[rule episode] error at step {steps}: {e}")
            result = {"terminated": True, "truncated": True}

        n_j = max(len(self._env.ids()), 1)
        try:
            if hasattr(self._env, 'lane_halting_count') and hasattr(self._env, 'incoming_lanes'):
                qh = sum(
                    self._env.lane_halting_count(lid)
                    for jid in self._env.ids()
                    for lid in self._env.incoming_lanes(jid)
                )
            else:
                qh = 0
        except Exception:
            qh = 0
        kpi = self._tracker.episode_kpi_dict() if self._tracker else {}

        metrics = {
            **kpi,
            "avg_reward": total_reward / max(steps, 1),
            "steps": steps,
            "queue": qh / max(n_j, 1),
        }
        if self._tracker:
            self._tracker.log(metrics, steps, ref_kind="env_steps")
        return metrics

    def evaluate(self, steps: int) -> dict:
        self._env.reset()
        if self._tracker:
            self._tracker.reset_episode_stats(self._env)
        total_reward = 0.0
        decision_steps = 0
        try:
            for _ in range(steps):
                obs_list = self._observer.observe(self._env)
                for obs in obs_list:
                    obs.get("extras", {})["env"] = self._env
                actions = [a.act(o) for a, o in zip(self._algos, obs_list)]
                result = self._env.step(self._actor.translate(actions))
                rews = self._reward.compute(self._env)
                total_reward += sum(rews)
                decision_steps += 1
                if self._tracker:
                    self._tracker.accumulate_step(self._env)
                if result["terminated"] or result["truncated"]:
                    break
        except Exception as e:
            print(f"[rule evaluate] error at step {decision_steps}: {e}")
        kpi = self._tracker.episode_kpi_dict() if self._tracker else {}
        return {
            **kpi,
            "avg_reward": total_reward / max(decision_steps, 1),
            "steps": decision_steps,
        }

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({}, f)

    def load(self, path: str) -> None:
        pass

    def teardown(self) -> None:
        self._env.close()
