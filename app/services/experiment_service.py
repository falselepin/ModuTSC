# app/services/experiment_service.py

import threading
import uuid
import os
import yaml
import traceback
from datetime import datetime
from pathlib import Path
from app.config import PROJECT_ROOT, CONFIGS_DIR
from app.services.file_service import save_index, save_experiment_detail
from app.config import DEFAULT_TRAINING, DEFAULT_EVALUATION

experiments = []
active_experiments = {}  # exp_id -> { ... }
latest_frame = None      # 用于存储最新的截图数据

def detect_device():
    """自动检测可用设备，优先使用 GPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def _algorithm_defaults(model_id: str) -> dict:
    defaults = {
        "dqn": {"lr": 0.001, "gamma": 0.99, "hidden_size": 64},
        "ma2c": {"lr": 5e-4, "gamma": 0.99, "tau": 0.1, "hidden_size": 64},
        "frap": {"lr": 0.001, "gamma": 0.95},
        "colight": {"lr": 0.0001, "gamma": 0.95, "tau": 0.005},
        "fixed_time": {},
        "max_pressure": {},
    }
    return defaults.get(model_id, {})


def get_default_config(model_id: str, dataset_id: str) -> dict:
    from modutsc.api import scaffold_config
    selections = {
        "environment": "sumo",
        "observer": "frap",
        "actor": "phase",
        "reward": "composite",
        "collector": "replay",
        "algorithm": model_id,
    }
    cfg = scaffold_config("single", selections, {
        "algorithm": _algorithm_defaults(model_id),
    })
    cfg["experiment"]["name"] = f"{model_id}_{dataset_id}_exp"
    cfg["environment"]["config"]["roadnet_file"] = f"data/{dataset_id}/roadnet.net.xml"
    cfg["environment"]["config"]["flow_file"] = f"data/{dataset_id}/flow_0.rou.xml"
    cfg["evaluation"]["checkpoint_dir"] = f"checkpoints/{model_id}_{dataset_id}/"
    return cfg


def start_experiment_thread(exp_id, dataset_id, model_id, full_config, gui_enabled, shared_state):
    """启动训练线程，返回线程对象"""
    # 提前获取实验名称和停止事件
    exp_name = active_experiments[exp_id]["name"]
    stop_event = active_experiments[exp_id]["stop_event"]

    def run_training():
        try:
            from modutsc.api import run_experiment

            detected_device = detect_device()
            active_experiments[exp_id]["device"] = detected_device

            overrides = {
                "environment": {
                    "config": {
                        "roadnet_file": (PROJECT_ROOT / "data" / dataset_id / "roadnet.net.xml").as_posix(),
                        "flow_file": (PROJECT_ROOT / "data" / dataset_id / "flow_0.rou.xml").as_posix()
                    }
                }
            }

            if "algorithm" in full_config:
                for i, algo in enumerate(full_config["algorithm"]):
                    if isinstance(algo, dict) and "config" in algo:
                        if "device" not in algo["config"]:
                            algo["config"]["device"] = detected_device
                            if i < len(overrides.setdefault("algorithm", [])):
                                overrides["algorithm"][i]["config"]["device"] = detected_device
                            else:
                                overrides["algorithm"].append({"config": {"device": detected_device}})

            result = run_experiment(
                str(CONFIGS_DIR / f"auto_{exp_id}.yaml"),
                stop_event=stop_event,
                overrides=overrides,
                shared_state=shared_state
            )

            if result.get("error"):
                active_experiments[exp_id]["status"] = "failed"
                active_experiments[exp_id]["error"] = result["error"]
                return

            was_stopped = result.get("stopped", False) or active_experiments[exp_id]["status"] == "cancelling"

            if not was_stopped:
                all_metrics = result.get("training", [])
                total_reward = sum(float(ep.get("avg_reward", 0)) for ep in all_metrics)
                avg_reward = total_reward / max(len(all_metrics), 1)
                exp_summary = {
                    "id": exp_id,
                    "name": exp_name,
                    "datasetName": dataset_id,
                    "modelName": model_id,
                    "status": "completed",
                    "avgReward": round(avg_reward, 2),
                    "avgSpeed": 0,
                    "createdAt": datetime.now().isoformat(),
                    "progress": 100
                }
                experiments.append(exp_summary)
                save_index(experiments)
                reward_curves = shared_state.get("reward_curves", [])
                if not reward_curves:
                    reward_curves = result.get("training", [])
                # 保存配置文件到 results 目录
                config_path = f"config_{exp_id}.yaml"
                config_full_path = PROJECT_ROOT / "results" / config_path
                with open(config_full_path, "w", encoding="utf-8") as f:
                    yaml.dump(full_config, f, allow_unicode=True, sort_keys=False)
                detail = {
                    "id": exp_id,
                    "name": exp_name,
                    "datasetName": dataset_id,
                    "modelName": model_id,
                    "metrics": shared_state.get("last_eval_metrics") or shared_state.get("last_episode_metrics", {}),
                    "reward_curve": [point["avg_reward"] for point in reward_curves],
                    "signals": shared_state.get("signals", []),
                    "vehicles": shared_state.get("vehicles", []),
                    "road_network": shared_state.get("road_network", None),
                    "last_epoch": shared_state.get("current_epoch", "?"),
                    "gui_enabled": gui_enabled,
                    "config_path": config_path,
                }
                save_experiment_detail(exp_id, detail)
                active_experiments[exp_id].update({"status": "completed", "progress": 100})
            else:
                active_experiments[exp_id].update({"status": "cancelled", "progress": 0})
        except Exception as e:
            traceback.print_exc()
            print(f"Experiment {exp_id} failed: {e}")
            active_experiments[exp_id] = {
                "status": "failed",
                "progress": 0,
                "error": str(e)
            }
        finally:
            temp_yaml = CONFIGS_DIR / f"auto_{exp_id}.yaml"
            if temp_yaml.exists():
                os.remove(temp_yaml)

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return thread


def start_screenshot_thread(exp_id):
    """启动截图线程"""
    def capture_loop():
        global latest_frame      
        import time
        import pygetwindow as gw
        from PIL import ImageGrab
        import io

        sumo_window = None
        # 等待窗口出现（最多10秒）
        for _ in range(20):
            for win in gw.getAllWindows():
                if 'sumo' in win.title.lower():
                    sumo_window = win
                    break
            if sumo_window:
                break
            time.sleep(0.5)

        while active_experiments[exp_id]["status"] not in ("completed", "cancelled", "failed"):
            try:
                if not sumo_window or not sumo_window.visible or sumo_window.isMinimized:
                    # 重新查找窗口
                    sumo_window = None
                    for win in gw.getAllWindows():
                        if 'sumo' in win.title.lower():
                            sumo_window = win
                            break
                    time.sleep(0.2)
                    continue
                left, top = sumo_window.topleft
                right, bottom = sumo_window.bottomright
                if right - left < 100 or bottom - top < 100:
                    time.sleep(0.2)
                    continue
                img = ImageGrab.grab(bbox=(left, top, right, bottom))
                img = img.resize((800, 500))
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=60)
                latest_frame = buf.getvalue()
            except Exception:
                pass
            time.sleep(0.2)

    threading.Thread(target=capture_loop, daemon=True).start()