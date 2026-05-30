import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from modutsc.plugins.algorithms import Algorithm
from modutsc.scheduling.registry import register


try:
    import dgl
    _DGL_GRAPH_TYPE = dgl.heterograph.DGLHeteroGraph
except ImportError:
    _DGL_GRAPH_TYPE = np.ndarray


@register("algorithm", "igrl")
class IgrlAlgorithm(Algorithm):
    __input_type__ = _DGL_GRAPH_TYPE

    def setup(self, cfg: dict, env=None) -> None:
        # 如果传入了 env，优先从环境中获取真实的拓扑参数
        if env is not None:
            ids = env.ids()
            if ids:
                self._act_dim = max(env.phase_count(j) for j in ids)
            else:
                self._act_dim = cfg.get("num_phase", 4)
        else:
            self._act_dim = cfg.get("num_phase", 4)

        self._obs_dim = cfg.get("act_in_dim", 8)
        self._gamma = cfg.get("gamma", 0.95)
        self._lr = cfg.get("lr", 1e-4)
        self._tau = cfg.get("tau", 0.005)
        self._device = cfg.get("device", "cpu")
        hidden = cfg.get("hidden_size", 128)

        self._use_gcn = False
        self._q_graph = None
        self._q_cached = None

        try:
            import dgl
            from modutsc.plugins.algorithms.drl.igrl.model import (
                Convolutional_Message_Passing_Framework
            )
            self._q_graph = Convolutional_Message_Passing_Framework(
                policy='discrete', noisy=False, num_attention_heads=1,
                bias_before_aggregation=True, nonlinearity_before_aggregation=True,
                share_initial_params_between_actions=False,
                multidimensional_attention=False, state_first_dim_only=False,
                gaussian_mixture=False, n_gaussians=1, value_model_based=False,
                use_attention=False, separate_actor_critic=False,
                n_actions=self._act_dim, rl_learner_type='Q_Learning',
                std_attention=False, state_vars=1, n_convolutional_layers=2,
                num_nodes_types=3, nodes_types_num_bases=3,
                node_state_dim=cfg.get("node_state_dim", 10),
                node_embedding_dim=cfg.get("node_embedding_dim", 32),
                num_rels=6, n_hidden_message=32, n_hidden_aggregation=32,
                n_hidden_prediction=32, hidden_layers_size=2,
                prediction_size=self._act_dim, activation=torch.nn.functional.relu,
            ).to(self._device)
            self._target_graph = copy.deepcopy(self._q_graph).to(self._device)
            self._use_gcn = True
            self._optim = optim.Adam(self._q_graph.parameters(), lr=self._lr)

        except (ImportError, ModuleNotFoundError) as e:
            print(f"[IGRL] DGL not available, falling back to MLP: {e}")

        if not self._use_gcn:
            if self._obs_dim <= 0:
                self._obs_dim = 48
            self._q_net = MLP(self._obs_dim, self._act_dim, hidden).to(self._device)
            self._target_net = MLP(self._obs_dim, self._act_dim, hidden).to(self._device)
            self._target_net.load_state_dict(self._q_net.state_dict())
            self._optim = optim.Adam(self._q_net.parameters(), lr=self._lr)

        self._loss_fn = nn.MSELoss()
        self._step = 0

    def act(self, obs: dict) -> dict:
        if self._use_gcn and isinstance(obs["features"], dgl_heterograph_type()):
            return self._act_gcn(obs)
        else:
            return self._act_mlp(obs)

    def _act_mlp(self, obs: dict) -> dict:
        x = torch.FloatTensor(obs["features"]).unsqueeze(0).to(self._device)
        with torch.no_grad():
            q = self._q_net(x).squeeze(0).cpu().numpy()
        return {"agent_id": obs["id"], "value": int(np.argmax(q))}

    def _act_gcn(self, obs: dict) -> dict:
        """Extract per-junction Q-values from global graph forward pass."""
        g = obs["features"]

        if g is not self._last_graph:
            self._q_graph.eval()
            self._q_graph = self._q_graph.to(self._device)
            try:
                with torch.no_grad():
                    g = g.to(self._device)
                    q_output = self._q_graph.forward_graph(g)
            except Exception:
                q_output = self._q_graph.forward_graph(g)
            if isinstance(q_output, (list, tuple)):
                q_output = q_output[-1] if len(q_output) > 0 else q_output[0]
            self._q_cached = q_output.cpu().numpy()
            self._last_graph = g
            self._q_graph.train()

        idx = obs.get("extras", {}).get("tsc_idx", abs(hash(obs["id"])) % self._act_dim)
        if idx >= len(self._q_cached):
            idx = idx % len(self._q_cached)
        q = self._q_cached[idx]

        return {"agent_id": obs["id"], "value": int(np.argmax(q))}

    def learn(self, batch: dict) -> dict:
        if self._use_gcn and self._is_graph_batch(batch):
            return self._learn_gcn(batch)
        else:
            return self._learn_mlp(batch)

    def _learn_mlp(self, batch: dict) -> dict:
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

    def _learn_gcn(self, batch: dict) -> dict:
        graphs = batch["obs"]
        acts = batch["actions"]
        rews = batch["rewards"]
        next_graphs = batch["next_obs"]
        terminated = batch.get("terminated", [0] * len(graphs))
        tsc_indices = batch.get("tsc_indices", [0] * len(graphs))

        losses = []
        for i, (g, a, r, ng, t, ti) in enumerate(
            zip(graphs, acts, rews, next_graphs, terminated, tsc_indices)
        ):
            g = g.to(self._device)
            ng = ng.to(self._device)
            a = int(a) if not isinstance(a, (int, np.integer)) else int(a)
            r = float(r) if not isinstance(r, (float, np.floating)) else float(r)
            t = float(t) if not isinstance(t, (float, np.floating)) else float(t)
            ti = int(ti) if not isinstance(ti, (int, np.integer)) else int(ti)

            try:
                q_out = self._q_graph.forward_graph(g)
            except Exception:
                q_out = self._q_graph(g)
            if isinstance(q_out, (list, tuple)):
                q_out = q_out[-1]

            if isinstance(q_out, torch.Tensor):
                if ti < q_out.size(0):
                    q_eval = q_out[ti, a].unsqueeze(0)
                else:
                    continue
            else:
                continue

            with torch.no_grad():
                try:
                    q_next_out = self._target_graph.forward_graph(ng)
                except Exception:
                    q_next_out = self._target_graph(ng)
                if isinstance(q_next_out, (list, tuple)):
                    q_next_out = q_next_out[-1]
                q_next = q_next_out[ti].max().detach()

            q_target = torch.tensor([r + self._gamma * q_next * (1 - t)],
                                     device=self._device, dtype=torch.float32)

            loss = self._loss_fn(q_eval, q_target)
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
            losses.append(float(loss.item()))
            self._step += 1

        return {"loss": np.mean(losses) if losses else 0.0, "step": self._step}

    def sync(self, tau: float = 1.0) -> None:
        tconst = tau if tau < 1.0 else self._tau
        if self._use_gcn:
            for tp, sp in zip(self._target_graph.parameters(), self._q_graph.parameters()):
                tp.data.copy_(tconst * sp.data + (1 - tconst) * tp.data)
        else:
            for tp, sp in zip(self._target_net.parameters(), self._q_net.parameters()):
                tp.data.copy_(tconst * sp.data + (1 - tconst) * tp.data)

    def params(self) -> dict:
        net = self._q_graph if self._use_gcn else self._q_net
        return {k: v.cpu() for k, v in net.state_dict().items()}

    def load(self, p: dict) -> None:
        if self._use_gcn:
            self._q_graph.load_state_dict(p)
            self._target_graph.load_state_dict(p)
        else:
            self._q_net.load_state_dict(p)
            self._target_net.load_state_dict(p)

    def train(self) -> None:
        (self._q_graph if self._use_gcn else self._q_net).train()

    def eval(self) -> None:
        (self._q_graph if self._use_gcn else self._q_net).eval()

    def _is_graph_batch(self, batch: dict) -> bool:
        obs_list = batch.get("obs", [])
        if not obs_list:
            return False
        sample = obs_list[0] if isinstance(obs_list, list) else obs_list
        return hasattr(sample, 'nodes') or hasattr(sample, 'ndata')

    @property
    def _last_graph(self):
        if not hasattr(self, '_last_graph_'):
            self._last_graph_ = None
        return self._last_graph_

    @_last_graph.setter
    def _last_graph(self, v):
        self._last_graph_ = v


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x): return self.net(x)


def dgl_heterograph_type():
    try:
        import dgl
        return dgl.heterograph.DGLHeteroGraph
    except ImportError:
        return type(None)
