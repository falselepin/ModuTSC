import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from modutsc.plugins.algorithms import Algorithm
from modutsc.scheduling.registry import register


@register("algorithm", "dqn")
class DqnAlgorithm(Algorithm):

    def setup(self, cfg: dict, env=None) -> None:
        if env is not None:
            ids = env.ids()
            if ids:
                self._act_dim = max(env.phase_count(j) for j in ids)
                num_lanelinks = max(
                    len(env.traffic_light_controlled_links(j)) for j in ids
                )
                self._obs_dim = self._act_dim + num_lanelinks
            else:
                self._act_dim = cfg.get("num_phase", 4)
                self._obs_dim = cfg.get("act_in_dim", 8)
        else:
            self._act_dim = cfg.get("num_phase", 4)
            self._obs_dim = cfg.get("act_in_dim", 8)

        self._gamma = cfg.get("gamma", 0.95)
        self._lr = cfg.get("lr", 1e-4)
        self._tau = cfg.get("tau", 0.005)
        self._device = cfg.get("device", "cpu")
        self._hidden_size = cfg.get("hidden_size", 128)

        self._q_net = MLP(self._obs_dim, self._act_dim, self._hidden_size).to(self._device)
        self._target_net = MLP(self._obs_dim, self._act_dim, self._hidden_size).to(self._device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._optim = optim.Adam(self._q_net.parameters(), lr=self._lr)
        self._loss_fn = nn.MSELoss()
        self._step = 0

    def bind_topology(self, env, jid=None) -> None:
        ids = env.ids()
        if not ids:
            return
        self._act_dim = max(env.phase_count(j) for j in ids)
        num_lanelinks = max(
            len(env.traffic_light_controlled_links(j)) for j in ids
        )
        new_obs_dim = self._act_dim + num_lanelinks
        if new_obs_dim != self._obs_dim:
            self._obs_dim = new_obs_dim
            self._reinit_networks()

    def reinitialize_with_obs_dim(self, obs_dim: int) -> None:
        if obs_dim == self._obs_dim:
            return
        self._obs_dim = obs_dim
        self._reinit_networks()

    def _reinit_networks(self) -> None:
        self._q_net = MLP(self._obs_dim, self._act_dim, self._hidden_size).to(self._device)
        self._target_net = MLP(self._obs_dim, self._act_dim, self._hidden_size).to(self._device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._optim = optim.Adam(self._q_net.parameters(), lr=self._lr)

    def act(self, obs: dict) -> dict:
        x = torch.FloatTensor(obs["features"]).unsqueeze(0).to(self._device)
        with torch.no_grad():
            q = self._q_net(x).squeeze(0).cpu().numpy()
        if obs.get("mask") is not None:
            try:
                q = np.where(obs["mask"] > 0, q, -1e9)
            except ValueError:
                n = min(len(q), len(obs["mask"]))
                q[:n] = np.where(obs["mask"][:n] > 0, q[:n], -1e9)
        return {"agent_id": obs["id"], "value": int(np.argmax(q))}

    def learn(self, batch: dict) -> dict:
        obs_t = torch.FloatTensor(batch["obs"]).to(self._device)
        acts = torch.LongTensor(batch["actions"]).unsqueeze(1).to(self._device)
        rews = torch.FloatTensor(batch["rewards"]).unsqueeze(1).to(self._device)
        next_obs_t = torch.FloatTensor(batch["next_obs"]).to(self._device)
        terminated = torch.FloatTensor(batch.get("terminated", batch.get("dones", 0))).unsqueeze(1).to(self._device)

        q_eval = self._q_net(obs_t).gather(1, acts)
        with torch.no_grad():
            q_next = self._target_net(next_obs_t).max(1, keepdim=True)[0]
            q_target = rews + self._gamma * q_next * (1 - terminated)

        loss = self._loss_fn(q_eval, q_target)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        self._step += 1

        return {"loss": float(loss.item()), "step": self._step}

    def sync(self, tau: float = 1.0) -> None:
        t = tau if tau < 1.0 else self._tau
        for tp, sp in zip(self._target_net.parameters(), self._q_net.parameters()):
            tp.data.copy_(t * sp.data + (1 - t) * tp.data)

    def params(self) -> dict:
        return {k: v.cpu() for k, v in self._q_net.state_dict().items()}

    def load(self, p: dict) -> None:
        self._q_net.load_state_dict(p)
        self._target_net.load_state_dict(p)

    def train(self) -> None: self._q_net.train()
    def eval(self) -> None: self._q_net.eval()


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
