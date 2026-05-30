import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, List, Optional
from modutsc.plugins.algorithms import Algorithm
from modutsc.scheduling.registry import register


def expand_raw_p2l_to_obs_layout(
    raw: np.ndarray, act_dim: int, obs_dim: int, n_feat: int
) -> np.ndarray:
    n_cols = obs_dim - act_dim
    if n_cols <= 0:
        return np.zeros((act_dim, 0), dtype=np.float32)
    mx_pad = n_cols // max(n_feat, 1)
    out = np.zeros((act_dim, n_cols), dtype=np.float32)
    g, L = raw.shape
    g_use = min(g, act_dim)
    L_use = min(L, mx_pad)
    for li in range(L_use):
        sl = slice(li * n_feat, (li + 1) * n_feat)
        out[:g_use, sl] = raw[:g_use, li : li + 1]
    return out


@register("algorithm", "frap")
class Frap(Algorithm):

    def setup(self, cfg: dict, env=None) -> None:
        # 如果传入了 env，优先从环境中获取真实的拓扑参数
        if env is not None:
            ids = env.ids()
            if ids:
                self._num_phase = max(env.phase_count(j) for j in ids)
                self._num_lanelink = max(
                    len(env.traffic_light_controlled_links(j)) for j in ids
                )
            else:
                self._num_phase = cfg.get("num_phase", 4)
                self._num_lanelink = int(cfg.get("max_lanelinks", self._num_phase))
        else:
            self._num_phase = cfg.get("num_phase", 4)
            self._num_lanelink = int(cfg.get("max_lanelinks", self._num_phase))

        self._act_dim = self._num_phase
        self._gamma = cfg.get("gamma", 0.95)
        self._lr = cfg.get("lr", 1e-4)
        self._tau = cfg.get("tau", 0.005)
        self._device = cfg.get("device", "cpu")

        self._frap_tls_id: Optional[str] = cfg.get("frap_tls_id")
        manual = cfg.get("phase_2_passable_lanelink", None)
        self._manual_p2l = manual is not None

        if self._manual_p2l:
            p2l_arr = np.array(manual, dtype=np.float32)
            self._p2l = torch.from_numpy(p2l_arr).to(self._device)
        else:
            self._p2l = torch.eye(self._num_phase, self._num_lanelink)[
                : self._num_phase, : self._num_lanelink
            ].to(self._device)

        self._init_networks()

    def _init_networks(self) -> None:
        self._q_net = FrapNet(self._num_lanelink, self._num_phase, self._p2l).to(
            self._device
        )
        self._target_net = FrapNet(
            self._num_lanelink, self._num_phase, self._p2l
        ).to(self._device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._optim = optim.Adam(self._q_net.parameters(), lr=self._lr)
        self._loss_fn = nn.MSELoss()
        self._step = 0

    def _init_networks_with_p2l(self) -> None:
        """根据 p2l 矩阵的实际维度重新初始化网络。
        
        当 bind_topology() 从环境获取真实拓扑后，p2l 矩阵的维度可能与配置值不同。
        此方法使用 p2l 的实际维度来初始化网络，确保维度一致性。
        """
        p2l_rows, p2l_cols = self._p2l.shape
        num_lanelink = p2l_cols
        num_phase = p2l_rows  # 从 p2l 矩阵获取实际的相位数
        
        self._q_net = FrapNet(num_lanelink, num_phase, self._p2l).to(self._device)
        self._target_net = FrapNet(num_lanelink, num_phase, self._p2l).to(self._device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._optim = optim.Adam(self._q_net.parameters(), lr=self._lr)
        self._loss_fn = nn.MSELoss()
        self._step = 0

    def bind_topology(self, env, jid: Optional[str] = None) -> None:
        if self._manual_p2l:
            return
        ids = env.ids()
        if not ids:
            return

        n_green = max(len(env.green_phase_indices(j)) for j in ids)
        n_green = max(n_green, 1)
        n_links = max(len(env.traffic_light_controlled_links(j)) for j in ids)
        n_links = max(n_links, 1)

        jid = jid or self._frap_tls_id or ids[0]
        ctrl_links = env.traffic_light_controlled_links(jid)
        states = []
        try:
            states = env.traffic_light_state_string(jid)
        except Exception:
            pass
        gp = env.green_phase_indices(jid)

        if ctrl_links and states:
            p2l = np.zeros((n_green, n_links), dtype=np.float32)
            for row, prog_idx in enumerate(gp):
                if row >= n_green:
                    break
                st = states[prog_idx] if prog_idx < len(states) else ""
                for li in range(min(n_links, len(st))):
                    if st[li] in "Gg":
                        p2l[row, li] = 1.0
        else:
            p2l = np.zeros((n_green, n_links), dtype=np.float32)
            for r in range(min(n_green, n_links)):
                p2l[r, r] = 1.0
            for r in range(n_links, n_green):
                p2l[r, n_links - 1] = 1.0

        self._num_phase = n_green
        self._num_lanelink = n_links
        self._act_dim = self._num_phase

        obs_dim = self._num_phase + self._num_lanelink
        p2l_arr = expand_raw_p2l_to_obs_layout(
            p2l.astype(np.float32),
            self._num_phase,
            obs_dim,
            1,
        )
        self._p2l = torch.from_numpy(p2l_arr).to(self._device)
        self._init_networks_with_p2l()

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
        return {"loss": float(loss.item())}

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


class FrapNet(nn.Module):
    def __init__(self, num_lanelink, num_phase, p2l):
        super().__init__()
        self.num_lanelink = num_lanelink
        self.num_phase = num_phase
        self.register_buffer('p2l', p2l)
        self.register_buffer('l2p', p2l.permute(1, 0))
        self.register_buffer('comp_mask', self._build_comp_mask(p2l))

        d_emb = 4
        d_hid = 16
        d_conv = 20

        self.phase_emb = nn.Sequential(nn.Linear(1, d_emb), nn.ReLU())
        self.veh_emb = nn.Sequential(nn.Linear(1, d_emb), nn.ReLU())
        self.lanelink_emb = nn.Sequential(nn.Linear(d_emb * 2, d_hid), nn.ReLU())
        self.rel_emb = nn.Embedding(2, d_emb)
        self.conv_cube = nn.Sequential(nn.Conv2d(d_hid * 2, d_conv, 1), nn.ReLU())
        self.conv_rel = nn.Sequential(nn.Conv2d(d_emb, d_conv, 1), nn.ReLU())
        self.tail = nn.Sequential(
            nn.Conv2d(d_conv, d_conv, 1), nn.ReLU(),
            nn.Conv2d(d_conv, 1, 1),
        )

    def _build_comp_mask(self, p2l):
        mask = torch.zeros((self.num_phase, self.num_phase - 1), dtype=torch.int64, device=p2l.device)
        for pi in range(self.num_phase):
            cj = 0
            for pj in range(self.num_phase):
                if pi == pj: continue
                for li in range(self.num_lanelink):
                    if p2l[pi, li] == 1 and p2l[pj, li] == 1:
                        mask[pi, cj] = 1
                cj += 1
        return mask

    def forward(self, obs):
        B = obs.shape[0]
        phase_feat = obs[:, :self.num_phase]
        ll_feat = obs[:, self.num_phase:self.num_phase + self.num_lanelink]

        passable = torch.matmul(phase_feat.float(), self.p2l.float())

        e1 = self.phase_emb(passable.unsqueeze(-1))
        e2 = self.veh_emb(ll_feat.unsqueeze(-1))
        e3 = torch.cat([e1, e2], dim=2)
        e3 = self.lanelink_emb(e3).permute(0, 2, 1)

        phase_emb = torch.matmul(e3, self.l2p.float()).permute(0, 2, 1)

        cube = torch.zeros((B, 32, self.num_phase, self.num_phase - 1), device=obs.device)
        for pi in range(self.num_phase):
            cj = 0
            for pj in range(self.num_phase):
                if pi == pj: continue
                cube[:, :, pi, cj] = torch.cat([phase_emb[:, pi, :], phase_emb[:, pj, :]], dim=1)
                cj += 1

        pc = self.conv_cube(cube)
        rc = self.conv_rel(self.rel_emb(self.comp_mask).permute(2, 0, 1).unsqueeze(0))
        cf = pc * rc
        bm = self.tail(cf)
        q = torch.sum(bm, dim=3).squeeze(1)
        return q
