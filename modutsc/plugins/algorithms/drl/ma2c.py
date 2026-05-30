import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from modutsc.plugins.algorithms import Algorithm
from modutsc.scheduling.registry import register

# ────────────────────────────────
#  EXACT REPLICA OF TF1 layers from agents/utils.py
# ────────────────────────────────

def svd_ortho_init(shape, scale=np.sqrt(2)):
    """Exact replica of TF1 ortho_init() in agents/utils.py."""
    shape = tuple(shape)
    if len(shape) == 2:
        flat_shape = shape
    else:
        flat_shape = (int(np.prod(shape[:-1])), shape[-1])
    a = np.random.standard_normal(flat_shape).astype(np.float64)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return (scale * q).astype(np.float32)


class FCLayer(nn.Module):
    """Exact replica of TF1 fc() + ortho_init — SVD-based weight, zero bias."""
    def __init__(self, in_features, out_features):
        super().__init__()
        w_np = svd_ortho_init((in_features, out_features))
        self.weight = nn.Parameter(torch.from_numpy(w_np))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class CustomLSTM(nn.Module):
    """Exact replica of agents/utils.py lstm() — SVD-init wx/wh, zero bias."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        wx_np = svd_ortho_init((input_size, hidden_size * 4))
        wh_np = svd_ortho_init((hidden_size, hidden_size * 4))
        self.wx = nn.Parameter(torch.from_numpy(wx_np))
        self.wh = nn.Parameter(torch.from_numpy(wh_np))
        self.b = nn.Parameter(torch.zeros(hidden_size * 4))

    def forward(self, x, state):
        c, h = state
        z = torch.matmul(x, self.wx) + torch.matmul(h, self.wh) + self.b
        i, f, o, u = z.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        new_c = f * c + i * u
        new_h = o * torch.tanh(new_c)
        return new_h, (new_c, new_h)


# ────────────────────────────────
#  MA2C NETWORK
# ────────────────────────────────

class MA2CNet(nn.Module):
    def __init__(self, wave_dim: int, wait_dim: int, fp_dim: int, action_dim: int,
                 n_fc_wave: int = 128, n_fc_wait: int = 32, n_fc_fp: int = 64,
                 n_lstm: int = 64):
        super().__init__()
        self.n_lstm = n_lstm
        self._has_wait = wait_dim > 0
        lstm_input = n_fc_wave + (n_fc_wait if self._has_wait else 0) + n_fc_fp

        # ---- Actor network ----
        self.actor_fc_wave = FCLayer(wave_dim, n_fc_wave)
        if self._has_wait:
            self.actor_fc_wait = FCLayer(wait_dim, n_fc_wait)
        self.actor_fc_fp   = FCLayer(fp_dim, n_fc_fp)
        self.actor_lstm     = CustomLSTM(lstm_input, n_lstm)
        self.actor_head     = FCLayer(n_lstm, action_dim)

        # ---- Critic network ----
        self.critic_fc_wave = FCLayer(wave_dim, n_fc_wave)
        if self._has_wait:
            self.critic_fc_wait = FCLayer(wait_dim, n_fc_wait)
        self.critic_fc_fp   = FCLayer(fp_dim, n_fc_fp)
        self.critic_lstm     = CustomLSTM(lstm_input, n_lstm)
        self.critic_head     = FCLayer(n_lstm, 1)

    def forward(self, wave, wait, fp, states_actor, states_critic):
        # Actor
        h_a_wave = F.relu(self.actor_fc_wave(wave))
        h_a_fp   = F.relu(self.actor_fc_fp(fp))
        if self._has_wait:
            h_a_wait = F.relu(self.actor_fc_wait(wait))
            act_in = torch.cat([h_a_wave, h_a_wait, h_a_fp], dim=-1)
        else:
            act_in = torch.cat([h_a_wave, h_a_fp], dim=-1)
        c_a, h_a = states_actor.chunk(2, dim=-1)
        new_h_a, (new_c_a, _) = self.actor_lstm(act_in, (c_a, h_a))
        new_states_actor = torch.cat([new_c_a, new_h_a], dim=-1)
        pi = F.softmax(self.actor_head(new_h_a), dim=-1)

        # Critic
        h_c_wave = F.relu(self.critic_fc_wave(wave))
        h_c_fp   = F.relu(self.critic_fc_fp(fp))
        if self._has_wait:
            h_c_wait = F.relu(self.critic_fc_wait(wait))
            crt_in = torch.cat([h_c_wave, h_c_wait, h_c_fp], dim=-1)
        else:
            crt_in = torch.cat([h_c_wave, h_c_fp], dim=-1)
        c_c, h_c = states_critic.chunk(2, dim=-1)
        new_h_c, (new_c_c, _) = self.critic_lstm(crt_in, (c_c, h_c))
        new_states_critic = torch.cat([new_c_c, new_h_c], dim=-1)
        v = self.critic_head(new_h_c).squeeze(-1)

        return pi, v, new_states_actor, new_states_critic


# ────────────────────────────────
#  MA2C AGENT
# ────────────────────────────────

class TF1RMSprop(torch.optim.Optimizer):
    """Exact replica of TF1 RMSPropOptimizer: v_0=1, eps inside sqrt."""
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['square_avg'] = torch.ones_like(p, memory_format=torch.preserve_format)
                square_avg = state['square_avg']
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                # TF1: var -= lr * grad / sqrt(square_avg + eps)
                p.addcdiv_(grad, square_avg.add(eps).sqrt_(), value=-lr)
        return loss


@register("algorithm", "ma2c_agent")
class MA2CAgent(Algorithm):

    def setup(self, cfg: dict, env=None) -> None:
        self._device = cfg.get("device", "cpu")

        # 如果传入了 env，优先从环境中获取真实的拓扑参数
        if env is not None:
            ids = env.ids()
            if ids:
                self._action_dim = max(env.phase_count(j) for j in ids)
            else:
                self._action_dim = cfg.get("num_phase", 4)
        else:
            self._action_dim = cfg.get("num_phase", 4)

        self._wave_dim = cfg.get("wave_dim", 16)
        self._wait_dim = cfg.get("wait_dim", 0)
        self._fp_dim = cfg.get("fp_dim", 0)
        self._n_lstm = cfg.get("num_lstm", 64)
        self._n_fc_wave = cfg.get("n_fc_wave", 128)
        self._n_fc_wait = cfg.get("n_fc_wait", 32)
        self._n_fc_fp = cfg.get("n_fc_fp", 64)
        self._gamma = cfg.get("gamma", 0.99)
        self._max_grad_norm = cfg.get("max_grad_norm", 40)
        self._value_coef = cfg.get("value_coef", 1.0)

        if self._wave_dim + self._wait_dim + self._fp_dim == 0:
            self._wave_dim = 16

        self._net = MA2CNet(
            wave_dim=self._wave_dim,
            wait_dim=self._wait_dim,
            fp_dim=self._fp_dim,
            action_dim=self._action_dim,
            n_fc_wave=self._n_fc_wave,
            n_fc_wait=self._n_fc_wait,
            n_fc_fp=self._n_fc_fp,
            n_lstm=self._n_lstm,
        ).to(self._device)

        actor_lr = cfg.get("learning_rate", 1e-4)
        critic_lr = cfg.get("critic_learning_rate", actor_lr)
        rmsp_alpha = cfg.get("rmsp_alpha", 0.99)
        rmsp_eps = cfg.get("rmsp_epsilon", 1e-5)

        actor_params, critic_params = [], []
        for name, p in self._net.named_parameters():
            if 'critic' in name:
                critic_params.append(p)
            else:
                actor_params.append(p)
        self._optimizer = TF1RMSprop([
            {'params': actor_params, 'lr': actor_lr},
            {'params': critic_params, 'lr': critic_lr},
        ], alpha=rmsp_alpha, eps=rmsp_eps)

        self._state_actor = torch.zeros(1, self._n_lstm * 2, device=self._device)
        self._state_critic = torch.zeros(1, self._n_lstm * 2, device=self._device)

    def forward(self, wave, wait, fp) -> Tuple[np.ndarray, float]:
        wave_t = torch.tensor(wave, dtype=torch.float, device=self._device).unsqueeze(0)
        wait_t = torch.tensor(wait, dtype=torch.float, device=self._device).unsqueeze(0)
        fp_t = torch.tensor(fp, dtype=torch.float, device=self._device).unsqueeze(0)
        with torch.no_grad():
            pi, v, self._state_actor, self._state_critic = self._net(
                wave_t, wait_t, fp_t, self._state_actor, self._state_critic
            )
        return pi.squeeze(0).cpu().numpy(), v.item()

    def reset_state(self):
        self._state_actor.zero_()
        self._state_critic.zero_()

    def act(self, obs: dict) -> dict:
        wave = obs.get("extras", {}).get("wave")
        wait = obs.get("extras", {}).get("wait")
        fp = obs.get("extras", {}).get("fp")
        if wave is None or wait is None or fp is None:
            raise ValueError("MA2CAgent requires wave/wait/fp in Obs.extras")
        pi, v = self.forward(wave, wait, fp)
        s = pi.sum()
        if s > 0:
            pi = pi / s
        else:
            pi = np.ones_like(pi) / len(pi)
        act = int(np.random.choice(len(pi), p=pi))
        return {"agent_id": obs["id"], "value": act, "extras": {"pi": pi, "value": v}}

    def learn(self, batch: dict) -> dict:
        device = self._device
        T = batch['wave'].shape[0]

        wave = torch.tensor(batch['wave'], dtype=torch.float, device=device)
        wait = torch.tensor(batch['wait'], dtype=torch.float, device=device)
        fp = torch.tensor(batch['fp'], dtype=torch.float, device=device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float, device=device)
        values = torch.tensor(batch['values'], dtype=torch.float, device=device)
        dones = torch.tensor(batch['dones'], dtype=torch.float, device=device)
        bootstrap = batch.get('bootstrap_value', 0.0)
        init_states_actor = torch.tensor(batch['init_states_actor'], dtype=torch.float,
                                         device=device).unsqueeze(0)
        init_states_critic = torch.tensor(batch['init_states_critic'], dtype=torch.float,
                                          device=device).unsqueeze(0)
        cur_beta = batch['entropy_coef']

        R = bootstrap
        Rs = torch.zeros(T, device=device)
        Advs = torch.zeros(T, device=device)
        for t in reversed(range(T)):
            R = rewards[t] + self._gamma * R * (1.0 - dones[t])
            Adv = R - values[t]
            Rs[t] = R
            Advs[t] = Adv

        states_a = init_states_actor
        states_c = init_states_critic
        pi_list, v_list = [], []
        for t in range(T):
            if t > 0 and dones[t-1] > 0.5:
                states_a = torch.zeros_like(states_a)
                states_c = torch.zeros_like(states_c)
            pi, v, states_a, states_c = self._net(
                wave[t].unsqueeze(0), wait[t].unsqueeze(0), fp[t].unsqueeze(0),
                states_a, states_c
            )
            pi_list.append(pi.squeeze(0))
            v_list.append(v)
        pi_seq = torch.stack(pi_list)
        v_seq = torch.stack(v_list).squeeze(-1)

        log_pi = torch.log(pi_seq + 1e-8)
        selected_log_pi = log_pi[range(T), actions]
        policy_loss = - (selected_log_pi * Advs.detach()).mean()

        entropy = - (pi_seq * log_pi).sum(dim=1).mean()
        entropy_loss = - cur_beta * entropy

        value_loss = self._value_coef * 0.5 * F.mse_loss(v_seq, Rs.detach())

        loss = policy_loss + value_loss + entropy_loss

        self._optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self._net.parameters(), self._max_grad_norm)
        self._optimizer.step()

        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "grad_norm": float(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm),
        }

    def train(self) -> None:
        self._net.train()

    def eval(self) -> None:
        self._net.eval()

    def params(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self._net.state_dict().items()}

    def load(self, p: dict) -> None:
        for k, v in p.items():
            p[k] = torch.tensor(v)
        self._net.load_state_dict(p)
