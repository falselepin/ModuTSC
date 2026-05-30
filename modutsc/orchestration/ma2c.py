import copy
import csv
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from modutsc.orchestration import Orchestrator
from modutsc.env import Env
from modutsc.plugins.observers import Observer
from modutsc.plugins.actors import Actor
from modutsc.plugins.rewards import Reward
from modutsc.plugins.collectors import Collector
from modutsc.scheduling.registry import register
from modutsc.plugins.trackers import Tracker


class LinearScheduler:
    def __init__(self, init_val, min_val=0.0, total_step=1):
        self.init_val = init_val
        self.min_val = min_val
        self.total_step = total_step
        self.cur_step = 0

    def get(self, step):
        self.cur_step += step
        if self.total_step <= 0:
            return self.init_val
        frac = min(1.0, self.cur_step / self.total_step)
        return self.init_val - frac * (self.init_val - self.min_val)


@register("orchestrator", "ma2c")
class Ma2cOrchestrator(Orchestrator):
    """MA2C 多智能体编排器，基于 A2C 架构的协作式多路口信号控制。"""

    __compatible_plugins__ = {
        "observer": ["ma2c"],
        "actor": ["phase"],
        "algorithm": ["ma2c_agent"],
        "collector": ["ma2c"],
        "reward": ["composite"],
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
        self._tracker = tracker
        self._cfg = cfg

        # ????????
        self._gamma = cfg.get('gamma', 0.99)
        self._alpha = cfg.get('coop_gamma', 0.75)
        self._max_steps = cfg.get('max_steps', 720)
        self._total_step = cfg.get('total_step', 2000000)
        self._batch_size = cfg.get('batch_size', 120)
        self._reward_norm = cfg.get("reward_norm", 1.0)
        self._reward_clip = cfg.get("reward_clip", 2.0)

        self._neighbor_map = cfg.get('neighbor_map', None)
        self._neighbor_topk = cfg.get('neighbor_topk', None)

        n_agent = len(env.ids())
        self._n_agent = n_agent

        collector._n_agent = n_agent

        self._neighbor_mask = self._build_neighbor_mask()
        self._max_neighbors = max(1, int(self._neighbor_mask.sum(axis=1).max()))

        own_dim = observer.dim()
        max_action_dim = max(env.phase_count(jid) for jid in env.ids()) if env.ids() else actor.dim()

        self._own_dim = own_dim
        self._wave_dim = own_dim * self._max_neighbors
        self._wait_dim = 0
        self._fp_dim = (max_action_dim - 1) * self._max_neighbors
        self._max_action_dim = max_action_dim

        if not algorithms:
            raise ValueError("Ma2cOrchestrator requires at least one algorithm")
        template = algorithms[0]
        template.setup(
            {
                "wave_dim": self._wave_dim,
                "wait_dim": self._wait_dim,
                "fp_dim": self._fp_dim,
                "num_lstm": cfg.get("num_lstm", 64),
                "n_fc_wave": cfg.get("n_fc_wave", 128),
                "n_fc_wait": cfg.get("n_fc_wait", 64),
                "n_fc_fp": cfg.get("n_fc_fp", 32),
                "learning_rate": cfg.get("learning_rate", 1e-4),
                "critic_learning_rate": cfg.get("critic_learning_rate", 1e-4),
                "rmsp_alpha": cfg.get("rmsp_alpha", 0.99),
                "rmsp_epsilon": cfg.get("rmsp_epsilon", 5e-5),
                "max_grad_norm": cfg.get("max_grad_norm", 40),
                "gamma": self._gamma,
                "value_coef": cfg.get("value_coef", 1.0),
                "device": cfg.get("device", "cpu"),
                "obs_dim": self._wave_dim + self._wait_dim + self._fp_dim,
                "act_dim": max_action_dim,
            }
        )
        self._algos = [template]
        for i in range(1, n_agent):
            clone = copy.deepcopy(template)
            self._algos.append(clone)

        self._current_pis = []
        for i in range(n_agent):
            dim = env.phase_count(env.ids()[i])
            self._current_pis.append(np.ones(dim) / dim)

        lr_decay = cfg.get('lr_decay', 'linear')
        entropy_decay = cfg.get('entropy_decay', 'linear')
        lr_init = cfg.get("learning_rate", 5e-4)
        if lr_decay == 'constant':
            self._lr_scheduler = LinearScheduler(lr_init, lr_init, total_step=0)
        else:
            lr_min = cfg.get("lr_min", lr_init * 0.1)
            self._lr_scheduler = LinearScheduler(lr_init, lr_min, self._total_step)
        beta_init = cfg.get("entropy_coef_init", 0.01)
        if entropy_decay == 'constant':
            self._beta_scheduler = LinearScheduler(beta_init, beta_init, total_step=0)
        else:
            beta_min = cfg.get("entropy_coef_min", 0.0)
            self._beta_scheduler = LinearScheduler(beta_init, beta_min, int(self._total_step * 0.8))

        self._states_bw_actor = [np.zeros(self._algos[i]._n_lstm * 2, dtype=np.float32)
                                 for i in range(n_agent)]
        self._states_bw_critic = [np.zeros(self._algos[i]._n_lstm * 2, dtype=np.float32)
                                  for i in range(n_agent)]
        self._step_count = 0
        self._ep_num = 0

        output_dir = cfg.get("output_dir", "output/ma2c_monaco")
        os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        self._scalars_path = os.path.join(output_dir, "log", "all_scalars.csv")
        self._train_reward_path = os.path.join(output_dir, "data", "train_reward.csv")
        self._scalar_keys = [
            "loss/fplstm_0a_policy_loss", "loss/fplstm_0a_value_loss",
            "loss/fplstm_0a_total_loss", "train/fplstm_0a_gradnorm",
            "train_reward", "test_reward"
        ]
        with open(self._scalars_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "step"] + self._scalar_keys)
        with open(self._train_reward_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "agent", "step", "test_id", "avg_reward", "std_reward"])

    # --------------------------------------------------------------
    def _learn_on_buffer(self, batch_per_agent, last_done):
        losses = {}
        grad_norm = float("nan")
        if not last_done:
            current_raw = self._env
            current_obs = self._observer.observe(current_raw)
            wave_last, wait_last, fp_last = self._build_aug_inputs(current_obs)
            self._inject_extras(current_obs, wave_last, wait_last, fp_last)
            for i, ag in enumerate(self._algos):
                action_obj = ag.act(current_obs[i])
                batch_per_agent[i]['bootstrap_value'] = action_obj.get("extras", {}).get("value", 0.0)
        else:
            for i in range(self._n_agent):
                batch_per_agent[i]['bootstrap_value'] = 0.0

        cur_lr = self._lr_scheduler.get(self._batch_size)
        cur_beta = self._beta_scheduler.get(self._batch_size)
        for i, ag in enumerate(self._algos):
            for pg in ag._optimizer.param_groups:
                pg['lr'] = cur_lr
            batch_per_agent[i]['entropy_coef'] = cur_beta
            batch_per_agent[i]['init_states_actor'] = self._states_bw_actor[i].copy()
            batch_per_agent[i]['init_states_critic'] = self._states_bw_critic[i].copy()
            ag.train()
            info = ag.learn(batch_per_agent[i])
            ag.eval()
            self._states_bw_actor[i] = ag._state_actor.squeeze(0).cpu().numpy()
            self._states_bw_critic[i] = ag._state_critic.squeeze(0).cpu().numpy()
            if i == 0:
                losses = info
                grad_norm = info.get("grad_norm", float("nan"))

        # per-batch CSV log
        row = {
            "loss/fplstm_0a_policy_loss":   f"{losses.get('policy_loss', 0):.6f}",
            "loss/fplstm_0a_value_loss":    f"{losses.get('value_loss', 0):.6f}",
            "loss/fplstm_0a_total_loss":    f"{losses.get('total_loss', 0):.6f}",
            "train/fplstm_0a_gradnorm":     f"{grad_norm:.6f}" if not np.isnan(grad_norm) else "0.0",
            "train_reward": "",
            "test_reward": "",
        }
        with open(self._scalars_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self._step_count] + [row[k] for k in self._scalar_keys])

        if self._tracker:
            self._tracker.log(losses, self._step_count, ref_kind="train")
        return losses, grad_norm

    def _build_neighbor_mask(self) -> np.ndarray:
        ids = self._env.ids()
        n = len(ids)
        id_to_idx = {jid: i for i, jid in enumerate(ids)}
        if self._neighbor_map is not None:
            return self._mask_from_neighbor_map(id_to_idx, n)
        if self._neighbor_topk is not None:
            return self._mask_from_coordinates(id_to_idx, n)
        return self._mask_ring(n)

    def _mask_from_neighbor_map(self, id_to_idx, n):
        mask = np.zeros((n, n), dtype=int)
        np.fill_diagonal(mask, 1)
        for jid, neighbors in self._neighbor_map.items():
            if jid not in id_to_idx:
                continue
            i = id_to_idx[jid]
            for nb in neighbors:
                if nb in id_to_idx:
                    mask[i, id_to_idx[nb]] = 1
        return mask

    def _mask_from_coordinates(self, id_to_idx, n):
        try:
            positions = [self._env.junction_pos(jid) for jid in id_to_idx]
            coords = np.array(positions)
        except Exception:
            return self._mask_ring(n)
        dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))
        topk = min(self._neighbor_topk, n)
        mask = np.zeros((n, n), dtype=int)
        for i in range(n):
            nearest = np.argsort(dist[i])[:topk]
            mask[i, nearest] = 1
        return mask

    def _mask_ring(self, n):
        mask = np.eye(n, dtype=int)
        for i in range(n):
            mask[i, (i + 1) % n] = 1
            mask[i, (i - 1) % n] = 1
        return mask

    def _spatial_discount(self, raw_rewards):
        n = self._n_agent
        dist = np.full((n, n), 1e6)
        np.fill_diagonal(dist, 0)
        for i in range(n):
            for j in np.where(self._neighbor_mask[i])[0]:
                dist[i, j] = 1 if i != j else 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        weight = np.power(self._alpha, dist)
        raw = np.array(raw_rewards)
        disc = weight @ raw
        return disc.tolist()

    def _process_rewards(self, raw_rewards):
        raw = np.array(raw_rewards)
        n = self._n_agent
        disc = []
        for i in range(n):
            cur = raw[i]
            nbr_indices = [j for j in range(n) if self._neighbor_mask[i, j] and j != i]
            for j in nbr_indices:
                cur += self._alpha * raw[j]
            n_node = 1 + len(nbr_indices)
            cur /= (n_node * self._reward_norm)
            disc.append(cur)
        if self._reward_clip:
            disc = [max(-self._reward_clip, min(self._reward_clip, r)) for r in disc]
        return disc

    def _build_aug_inputs(self, obs_list):
        n = self._n_agent
        own = np.stack([o["features"] for o in obs_list])

        def _to_fixed(v: np.ndarray, target: int) -> np.ndarray:
            v = np.asarray(v, dtype=np.float32).ravel()
            if v.size > target:
                return v[:target].copy()
            if v.size < target:
                return np.pad(v, (0, target - v.size))
            return v.copy()

        wave_all = []
        fp_all = []
        for i in range(n):
            # wave: self + α×neighbors (original env.py _get_state for MA2C)
            wave_parts = [own[i]]
            fps = []
            for j in range(n):
                if self._neighbor_mask[i, j] and j != i:
                    wave_parts.append(own[j] * self._alpha)
                    # fingerprint = policy[:-1] (original update_fingerprint)
                    pi = self._current_pis[j][:-1]
                    fps.append(pi)
            wave_vec = np.concatenate(wave_parts)
            fp_vec = np.concatenate(fps) if fps else np.zeros(0, dtype=np.float32)

            wave_vec = _to_fixed(wave_vec, self._wave_dim)
            fp_vec = _to_fixed(fp_vec, self._fp_dim)
            wave_all.append(wave_vec)
            fp_all.append(fp_vec)

        wave = np.stack(wave_all)
        fp = np.stack(fp_all)
        wait = np.zeros((n, 0), dtype=np.float32)
        return wave, wait, fp

    def _inject_extras(self, obs_list, wave, wait, fp):
        for i, obs in enumerate(obs_list):
            obs.get("extras", {})["wave"] = wave[i]
            obs.get("extras", {})["wait"] = wait[i]
            obs.get("extras", {})["fp"] = fp[i]

    def _emit_note(self, message: str) -> None:
        """与 Orchestrator.run 一致：有 tracker 走 note，否则 print。"""
        if self._tracker:
            self._tracker.note(message)
        else:
            print(message)

    def warmup(self, steps):
        self._env.reset()
        for s in range(steps):
            if self._should_stop():
                break
            obs_list = self._observer.observe(self._env)
            wave, wait, fp = self._build_aug_inputs(obs_list)
            self._inject_extras(obs_list, wave, wait, fp)
            actions = [np.random.randint(self._env.phase_count(self._env.ids()[i]))
                      for i in range(self._n_agent)]
            result = self._env.step(self._actor.translate(
                [{"agent_id": self._env.ids()[i], "value": a} for i, a in enumerate(actions)]
            ))
            if result["terminated"] or result["truncated"]:
                self._env.reset()
                self._reward.reset()
        self._collector.clear()
        return {"warmup_steps": steps}

    def episode(self):
        self._env.reset()
        self._reward.reset()
        self._collector.clear()
        for ag in self._algos:
            ag.reset_state()
        # Reinitialize fingerprints to match current SUMO phase counts
        for i in range(self._n_agent):
            dim = self._env.phase_count(self._env.ids()[i])
            self._current_pis[i] = np.ones(dim) / dim
        for i in range(self._n_agent):
            self._states_bw_actor[i].fill(0.0)
            self._states_bw_critic[i].fill(0.0)

        total_reward = 0.0
        total_raw_reward = 0.0
        sim_max_time = float(getattr(self._env, '_max_time', 3600))
        decision_steps = 0
        next_progress = 1200

        losses = {}
        grad_norm = float("nan")

        if self._tracker:
            self._tracker.reset_episode_stats(self._env)
        self._ep_num += 1

        while decision_steps < self._max_steps:
            if self._should_stop():
                break
            cur_time = self._env.time()

            if cur_time >= next_progress:
                self._emit_note(
                    f"  Sim time {int(next_progress)}/{int(sim_max_time)} seconds"
                )
                next_progress += 1200

            obs_list = self._observer.observe(self._env)
            wave, wait, fp = self._build_aug_inputs(obs_list)
            self._inject_extras(obs_list, wave, wait, fp)

            actions, values = [], []
            for i, (ag, obs) in enumerate(zip(self._algos, obs_list)):
                action_obj = ag.act(obs)
                pi = action_obj.get("extras", {}).get("pi")
                if pi is not None:
                    jid = self._env.ids()[i]
                    n_phases = self._env.phase_count(jid)
                    pi = pi.copy()
                    pi = pi[:n_phases]
                    s = pi.sum()
                    if s > 0:
                        pi = pi / s
                    else:
                        pi[:] = 1.0 / n_phases
                    self._current_pis[i] = pi
                    act = int(np.random.choice(n_phases, p=pi))
                else:
                    act = action_obj["value"]
                actions.append(act)
                values.append(action_obj.get("extras", {}).get("value", 0.0))

            result = self._env.step(self._actor.translate(
                [{"agent_id": self._env.ids()[i], "value": a} for i, a in enumerate(actions)]
            ))
            raw_rewards = self._reward.compute(self._env)
            disc_rewards = self._process_rewards(raw_rewards)
            self._collector.push({
                'wave': wave, 'wait': wait, 'fp': fp,
                'actions': actions, 'rewards': disc_rewards,
                'values': values, 'dones': result["terminated"]
            })
            total_reward += sum(disc_rewards)
            total_raw_reward += sum(raw_rewards)
            if self._tracker:
                self._tracker.accumulate_step(self._env)

            decision_steps += 1
            self._step_count += 1

            if self._collector.ready():
                batch_per_agent = self._collector.pull()
                last_done = batch_per_agent[0]['dones'][-1]
                losses, grad_norm = self._learn_on_buffer(batch_per_agent, last_done)

            if result["terminated"] or result["truncated"]:
                for ag in self._algos:
                    ag.reset_state()
                break

        if self._collector.size() > 0:
            batch_per_agent = self._collector.flush()
            last_done = batch_per_agent[0]['dones'][-1]
            losses, grad_norm = self._learn_on_buffer(batch_per_agent, last_done)

        end_time = self._env.time()

        kpi = self._tracker.episode_kpi_dict() if self._tracker else {}

        avg_reward = total_raw_reward / max(decision_steps, 1)

        metrics = {
            **kpi,
            "departed": int(kpi.get("departed", 0)),
            "arrived": int(kpi.get("arrived", 0)),
            "sim_time": float(end_time),
            "steps": decision_steps,
            "avg_reward": avg_reward,
            "avg_raw_reward": total_raw_reward / max(decision_steps, 1),
            "grad_norm": grad_norm,
            **losses,
        }
        if self._tracker:
            self._tracker.log(metrics, self._ep_num, ref_kind="episode")
        else:
            print(f"  Average Reward: {avg_reward:.4f}")
            print(
                "  Total departed (approx): "
                f"{int(kpi.get('departed', 0))}, Total arrived: "
                f"{int(kpi.get('arrived', 0))}"
            )

        # per-episode train_reward.csv
        with open(self._train_reward_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self._ep_num, "ma2c", self._step_count, -1, f"{avg_reward:.6f}", "0.0"])

        return metrics

    def evaluate(self, steps):
        self._env.reset()
        if self._tracker:
            self._tracker.reset_episode_stats(self._env)
        for i in range(self._n_agent):
            dim = self._env.phase_count(self._env.ids()[i])
            self._current_pis[i] = np.ones(dim) / dim
        for ag in self._algos:
            ag.reset_state()
            ag.eval()
        for _ in range(steps):
            if self._should_stop():
                break
            obs_list = self._observer.observe(self._env)
            wave, wait, fp = self._build_aug_inputs(obs_list)
            self._inject_extras(obs_list, wave, wait, fp)
            actions = []
            for i, (ag, obs) in enumerate(zip(self._algos, obs_list)):
                action_obj = ag.act(obs)
                pi = action_obj.get("extras", {}).get("pi")
                if pi is not None:
                    jid = self._env.ids()[i]
                    n_phases = self._env.phase_count(jid)
                    pi = pi.copy()
                    pi = pi[:n_phases]
                    s = pi.sum()
                    if s > 0:
                        pi = pi / s
                    else:
                        pi = np.ones(n_phases) / n_phases
                    self._current_pis[i] = pi
                    actions.append(int(np.argmax(pi)))
                else:
                    actions.append(action_obj["value"])
            result = self._env.step(self._actor.translate(
                [{"agent_id": self._env.ids()[i], "value": a} for i, a in enumerate(actions)]
            ))
            if self._tracker:
                self._tracker.accumulate_step(self._env)
            if self._env.done():
                break
        for ag in self._algos:
            ag.train()
        ek = self._tracker.episode_kpi_dict() if self._tracker else {}
        return {**ek, "evaluate_steps": steps}

    def save(self, path):
        import torch
        torch.save(
            {i: ag._net.state_dict() for i, ag in enumerate(self._algos)},
            path
        )

    def load(self, path):
        import torch
        ckpt = torch.load(path)
        for i, ag in enumerate(self._algos):
            ag._net.load_state_dict(ckpt[i])

    def teardown(self):
        self._env.close()