import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from modutsc.plugins.algorithms import Algorithm
from modutsc.scheduling.registry import register


@register("algorithm", "colight")
class ColightAlgorithm(Algorithm):

    def setup(self, cfg: dict, env=None) -> None:
        # 如果传入了 env，优先从环境中获取真实的拓扑参数
        if env is not None:
            ids = env.ids()
            if ids:
                self._num_phase = max(env.phase_count(j) for j in ids)
                self._num_lane = max(
                    len(env.traffic_light_controlled_links(j)) for j in ids
                )
            else:
                self._num_phase = cfg.get("num_phase", 4)
                self._num_lane = int(cfg.get("max_lanelinks", self._num_phase))
        else:
            self._num_phase = cfg.get("num_phase", 4)
            self._num_lane = int(cfg.get("max_lanelinks", self._num_phase))

        self._act_dim = self._num_phase
        self._gamma = cfg.get("gamma", 0.95)
        self._lr = cfg.get("lr", 1e-4)
        self._tau = cfg.get("tau", 0.005)
        self._device = cfg.get("device", "cpu")

        self._q_net = MultiHeadAttentionNetwork(
            self._num_lane, self._num_phase,
            dim_embed=32, num_head=5
        ).to(self._device)
        self._target_net = copy.deepcopy(self._q_net).to(self._device)
        self._optim = optim.RMSprop(self._q_net.parameters(), lr=self._lr)
        self._loss_fn = nn.MSELoss()
        self._step = 0

    def act(self, obs: dict) -> dict:
        x = torch.FloatTensor(obs["features"]).unsqueeze(0).to(self._device)
        with torch.no_grad():
            q = self._q_net(x).squeeze(0).cpu().numpy()
        return {"agent_id": obs["id"], "value": int(np.argmax(q))}

    def learn(self, batch: dict) -> dict:
        obs_t = torch.FloatTensor(batch["obs"]).to(self._device)
        acts = torch.LongTensor(batch["actions"]).unsqueeze(1).to(self._device)
        rews = torch.FloatTensor(batch["rewards"]).unsqueeze(1).to(self._device)
        next_obs_t = torch.FloatTensor(batch["next_obs"]).to(self._device)
        terminated = torch.FloatTensor(
            batch.get("terminated", np.zeros(len(batch["obs"])))
        ).unsqueeze(1).to(self._device)

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
        tconst = tau if tau < 1.0 else self._tau
        for tp, sp in zip(self._target_net.parameters(), self._q_net.parameters()):
            tp.data.copy_(tconst * sp.data + (1 - tconst) * tp.data)

    def params(self) -> dict:
        return {k: v.cpu() for k, v in self._q_net.state_dict().items()}

    def load(self, p: dict) -> None:
        self._q_net.load_state_dict(p)
        self._target_net.load_state_dict(p)

    def train(self) -> None: self._q_net.train()
    def eval(self) -> None: self._q_net.eval()


class MultiHeadAttentionNetwork(nn.Module):
    """CoLight network: multi-head self-attention over phase+lanelink features."""

    def __init__(self, num_lane, num_phase, dim_embed=32, num_head=5):
        super().__init__()
        self.num_phase = num_phase
        self.num_lane = num_lane
        self.num_head = num_head
        self.dim_embed = dim_embed

        # Input embedding
        self.embedding = nn.Sequential(
            nn.Linear(num_phase + num_lane, dim_embed), nn.ReLU(),
            nn.Linear(dim_embed, dim_embed), nn.ReLU(),
        )

        # Self-attention heads
        self.q_proj = nn.ModuleList([
            nn.Linear(dim_embed, dim_embed) for _ in range(num_head)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(dim_embed, dim_embed) for _ in range(num_head)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(dim_embed, dim_embed) for _ in range(num_head)
        ])

        # Output head
        self.output = nn.Sequential(
            nn.Linear(dim_embed, dim_embed), nn.ReLU(),
            nn.Linear(dim_embed, num_phase),
        )

    def forward(self, x):
        emb = self.embedding(x)

        head_outputs = []
        for h in range(self.num_head):
            q = self.q_proj[h](emb)
            k = self.k_proj[h](emb)
            v = self.v_proj[h](emb)
            attn = torch.softmax((q * k).sum(dim=-1, keepdim=True), dim=0)
            head_outputs.append(attn * v)

        pooled = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
        return self.output(pooled)
