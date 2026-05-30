from abc import ABC, abstractmethod
import math
import os
from typing import Dict, Any, Optional
from modutsc.env import Env
from modutsc.plugins.observers import Observer
from modutsc.plugins.actors import Actor
from modutsc.plugins.rewards import Reward
from modutsc.plugins.collectors import Collector
from modutsc.plugins.algorithms import Algorithm
from modutsc.plugins.trackers import Tracker


def _fmt_train_scalar(m: Dict[str, Any], key: str, fmt: str) -> str:
    if key not in m:
        return "n/a"
    v = m[key]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def _fmt_loss_total(m: Dict[str, Any]) -> str:
    if "total_loss" in m:
        return _fmt_train_scalar(m, "total_loss", ".4f")
    if "loss" in m:
        return _fmt_train_scalar(m, "loss", ".4f")
    return "n/a"


def _fallback_episode_header(cfg: dict, global_ep: int, epoch: int) -> None:
    exp = cfg.get("experiment") or {}
    env_c = (cfg.get("environment") or {}).get("config") or {}
    ename = exp.get("name", "")
    road = env_c.get("roadnet_file", "")
    flow = env_c.get("flow_file", "")
    bits = []
    if ename:
        bits.append(f"experiment={ename}")
    if road:
        bits.append(f"roadnet={os.path.basename(str(road))}")
    if flow:
        bits.append(f"flow={os.path.basename(str(flow))}")
    tag = " | ".join(bits) if bits else ""
    extra = f"  {tag}" if tag else ""
    print(f"Episode {global_ep} (epoch={epoch}){extra}")


def _frontend_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """将后端 metrics 转换为前端兼容格式（添加小写别名，数值保留1位小数）。"""
    result = {}
    for k, v in m.items():
        if isinstance(v, float):
            result[k] = round(v, 1)
        else:
            result[k] = v
    if "Throughput" in result:
        result["throughput"] = result["Throughput"]
    if "arrived" in result:
        result["completed"] = result["arrived"]
    if "total_loss" in result:
        result["loss"] = result["total_loss"]
    elif "loss" in result:
        result["total_loss"] = result["loss"]
    return result


class Orchestrator(ABC):

    @abstractmethod
    def setup(self, env: Env, observer: Observer, actor: Actor,
              reward: Reward, collector: Collector,
              algorithms: list, cfg: dict, tracker: Optional[Tracker] = None,
              **kwargs) -> None: ...

    @abstractmethod
    def warmup(self, steps: int) -> dict: ...

    @abstractmethod
    def episode(self) -> dict: ...

    @abstractmethod
    def evaluate(self, steps: int) -> dict: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    def teardown(self) -> None:
        pass

    def _should_stop(self) -> bool:
        return getattr(self, "_stop_event", None) is not None and self._stop_event.is_set()

    def run(self, cfg: dict, on_episode=None):
        training_cfg = cfg.get("training", {})
        eval_cfg = cfg.get("evaluation", {})

        tr = getattr(self, "_tracker", None)

        warmup_steps = training_cfg.get("warmup_steps", 1000)
        if warmup_steps > 0:
            if self._should_stop():
                msg = "[Training] Stopped by user before warmup"
                if tr is not None:
                    tr.note(msg)
                else:
                    print(msg)
                self.teardown()
                return {"training": [], "stopped": True}
            warmup_result = self.warmup(warmup_steps)
            msg = f"[Warmup] {warmup_result}"
            if tr is not None:
                tr.note(msg)
            else:
                print(msg)

        if self._should_stop():
            msg = "[Training] Stopped by user after warmup"
            if tr is not None:
                tr.note(msg)
            else:
                print(msg)
            self.teardown()
            return {"training": [], "stopped": True}

        num_epochs = training_cfg.get("num_epochs", 100)
        eps_per_epoch = training_cfg.get("episodes_per_epoch", 1)
        eval_freq = eval_cfg.get("eval_frequency", 10)
        eval_steps = eval_cfg.get("eval_steps", 3600)
        checkpoint_dir = eval_cfg.get("checkpoint_dir", "checkpoints/")

        all_metrics = []
        stopped = False
        for epoch in range(num_epochs):
            if self._should_stop():
                msg = f"[Training] Stopped by user at epoch {epoch}"
                if tr is not None:
                    tr.note(msg)
                else:
                    print(msg)
                stopped = True
                break

            for ep in range(eps_per_epoch):
                if self._should_stop():
                    stopped = True
                    break

                global_ep = epoch * eps_per_epoch + ep + 1
                if tr is not None:
                    tr.episode_header(global_ep=global_ep, epoch=epoch, full_cfg=cfg)
                else:
                    _fallback_episode_header(cfg, global_ep, epoch)
                try:
                    m = self.episode()
                except Exception as e:
                    print(f"[run] episode failed: {e}")
                    m = {"terminated": True, "episode": global_ep}
                m["epoch"] = epoch
                all_metrics.append(m)
                
                # 检查连接是否正常
                if m.get("sim_time", 0) == 0 and m.get("steps", 0) > 0:
                    print(f"[run] Warning: sim_time is 0 but steps > 0, connection may be lost")
                    # 尝试重新启动环境并继续训练
                    try:
                        self._env.close()
                    except Exception:
                        pass
                    try:
                        print(f"[run] Attempting to restart environment...")
                        self._env.reset()
                        print(f"[run] Environment restarted successfully, continuing training")
                    except Exception as e:
                        print(f"[run] Failed to restart environment: {e}")
                        print(f"[run] Environment connection lost, stopping training")
                        stopped = True
                        break

                # 将训练指标写入 shared_state，供前端轮询获取
                if hasattr(self, '_shared_state') and self._shared_state is not None:
                    self._shared_state['current_epoch'] = epoch + 1
                    self._shared_state['total_epochs'] = num_epochs
                    self._shared_state.setdefault('reward_curves', []).append(m)
                    self._shared_state['reward_curve'] = [
                        point.get('avg_reward', 0) for point in self._shared_state['reward_curves']
                    ]
                    self._shared_state['last_episode_metrics'] = _frontend_metrics(m)

                if on_episode is not None:
                    try:
                        on_episode(global_ep, m)
                    except Exception:
                        pass
                print(
                    f"  ATT={_fmt_train_scalar(m, 'ATT', '.2f')}; "
                    f"AQL={_fmt_train_scalar(m, 'AQL', '.4f')}; "
                    f"Throughput={_fmt_train_scalar(m, 'Throughput', '.2f')}; "
                    f"RealDelay={_fmt_train_scalar(m, 'RealDelay', '.2f')}; "
                    f"TripFlow={_fmt_train_scalar(m, 'TripFlow', '.4f')}; "
                    f"decision_steps={m.get('steps', 'n/a')}; "
                    f"sim_time={_fmt_train_scalar(m, 'sim_time', '.0f')}s"
                )
                print(
                    f"  Loss: total={_fmt_loss_total(m)} "
                    f"policy={_fmt_train_scalar(m, 'policy_loss', '.4f')} "
                    f"value={_fmt_train_scalar(m, 'value_loss', '.4f')} "
                    f"entropy={_fmt_train_scalar(m, 'entropy', '.4f')} "
                    f"grad_preclip={_fmt_train_scalar(m, 'grad_norm', '.2f')}"
                )

            if stopped:
                break

            if (epoch + 1) % eval_freq == 0:
                if self._should_stop():
                    stopped = True
                    break
                import os as _os
                _os.makedirs(checkpoint_dir, exist_ok=True)
                eval_result = self.evaluate(eval_steps)
                ev_msg = f"[Eval epoch {epoch + 1:3d}] {eval_result}"
                if tr is not None:
                    tr.note(ev_msg)
                else:
                    print(ev_msg)

                # 将评估指标写入 shared_state
                if hasattr(self, '_shared_state') and self._shared_state is not None:
                    self._shared_state['last_eval_metrics'] = _frontend_metrics(eval_result)

                self.save(f"{checkpoint_dir}/ckpt_epoch_{epoch + 1}.pkl")

        self.teardown()
        result = {"training": all_metrics}
        if stopped:
            result["stopped"] = True
        return result
