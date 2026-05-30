import uuid
import os
import yaml
import threading
import importlib
import traceback
from fastapi import APIRouter, HTTPException
from app.config import PROJECT_ROOT, CONFIGS_DIR, RESULTS_DIR
from app.services.experiment_service import (
    experiments, active_experiments,
    get_default_config,
    start_experiment_thread, start_screenshot_thread
)
from app.services.file_service import save_index, load_index, save_experiment_detail, load_experiment_detail, delete_experiment_files
from app.utils.road_network import extract_road_network
from datetime import datetime

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

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

@router.get("/")
def get_experiments(status: str = None):
    result = experiments.copy()
    for eid, info in active_experiments.items():
        if info["status"] in ("running", "failed", "cancelled"):
            result.append({
                "id": eid,
                "name": info.get("name", f"exp_{eid}"),
                "datasetName": info.get("datasetName", ""),
                "modelName": info.get("modelName", ""),
                "status": info["status"],
                "avgReward": info.get("avgReward", 0),
                "throughput": info.get("throughput", 0),
                "createdAt": info.get("createdAt", ""),
                "progress": info.get("progress", 0)
            })
    if status:
        result = [exp for exp in result if exp["status"] == status]
    return result

@router.post("/start_custom")
async def start_custom_experiment(data: dict):
    dataset_id = data.get("dataset_id")
    gui_enabled = data.get("gui_enabled", False)
    custom_config = data.get("config")  # 前端已拼装好的完整 YAML 字典

    if not dataset_id or not custom_config:
        raise HTTPException(status_code=400, detail="缺少必要参数")

    # 校验路网和流量文件
    roadnet_file = custom_config.get("environment", {}).get("config", {}).get("roadnet_file", "")
    flow_file = custom_config.get("environment", {}).get("config", {}).get("flow_file", "")
    if not roadnet_file or not flow_file:
        raise HTTPException(status_code=400, detail="配置中缺少路网或流量文件路径")

    # 转换为绝对路径并检查（与 start_experiment 相同的路径检查逻辑）
    roadnet_abs = (PROJECT_ROOT / roadnet_file).as_posix()
    flow_abs = (PROJECT_ROOT / flow_file).as_posix()
    print(f"Looking for roadnet at: {roadnet_abs}")
    print(f"Looking for flow at: {flow_abs}")

    if not os.path.isfile(roadnet_abs):
        raise HTTPException(status_code=400, detail="路网文件不存在")
    if not os.path.isfile(flow_abs):
        raise HTTPException(status_code=400, detail="车流文件不存在")

    # 更新配置中的路径为绝对路径
    custom_config["environment"]["config"]["roadnet_file"] = roadnet_abs
    custom_config["environment"]["config"]["flow_file"] = flow_abs
    custom_config["environment"]["config"]["gui"] = gui_enabled

    exp_id = str(uuid.uuid4())[:8]
    # 设置实验名称，若用户未提供则使用默认名称
    exp_name = custom_config.get("experiment", {}).get("name", f"custom_{exp_id}")
    custom_config.setdefault("experiment", {})
    custom_config["experiment"]["name"] = exp_name

    # 提取算法插件标识（用于日志/状态展示）
    algorithm_plugin = "custom"
    if "algorithm" in custom_config and custom_config["algorithm"]:
        algo = custom_config["algorithm"]
        if isinstance(algo, list):
            algo = algo[0] if algo else {}
        algorithm_plugin = algo.get("plugin", "custom") if algo else "custom"

    # 写入临时 YAML
    CONFIGS_DIR.mkdir(exist_ok=True)
    temp_yaml = CONFIGS_DIR / f"auto_{exp_id}.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(custom_config, f)

    # 初始化运行时信息（与 start_experiment 类似）
    stop_event = threading.Event()
    shared_state = {}
    active_experiments[exp_id] = {
        "status": "running",
        "progress": 0,
        "name": exp_name,
        "datasetName": dataset_id,
        "modelName": algorithm_plugin,        # 用算法插件名作为模型标识
        "createdAt": datetime.now().isoformat(),
        "stop_event": stop_event,
        "shared_state": shared_state,
        "gui_enabled": gui_enabled
    }

    def run_custom():
        try:
            from modutsc.api import run_experiment

            detected_device = detect_device()
            active_experiments[exp_id]["device"] = detected_device

            overrides = {}
            if "algorithm" in custom_config:
                algos = custom_config.get("algorithm", [])
                if isinstance(algos, dict):
                    algos = [algos]
                for i, algo in enumerate(algos):
                    if isinstance(algo, dict) and "config" in algo:
                        if "device" not in algo["config"]:
                            algo["config"]["device"] = detected_device
                            if i < len(overrides.setdefault("algorithm", [])):
                                overrides["algorithm"][i]["config"]["device"] = detected_device
                            else:
                                overrides["algorithm"].append({"config": {"device": detected_device}})

            result = run_experiment(
                str(temp_yaml),
                stop_event=stop_event,
                overrides=overrides,
                shared_state=shared_state
            )

            if result.get("error"):
                print(f"[experiment] Launcher error: {result['error']}")
                active_experiments[exp_id]["status"] = "failed"
                active_experiments[exp_id]["error"] = result["error"]
                return

            all_metrics = result.get("training", [])
            was_stopped = result.get("stopped", False) or active_experiments[exp_id]["status"] == "cancelling"
            if was_stopped:
                active_experiments[exp_id]["status"] = "cancelled"
                active_experiments[exp_id]["progress"] = 0
            else:
                total_reward = sum(float(ep.get("avg_reward", 0)) for ep in all_metrics)
                avg_reward = total_reward / max(len(all_metrics), 1)
                total_throughput = sum(float(ep.get("Throughput", 0)) for ep in all_metrics)
                avg_throughput = total_throughput / max(len(all_metrics), 1)
                exp_summary = {
                    "id": exp_id,
                    "name": exp_name,
                    "datasetName": dataset_id,
                    "modelName": algorithm_plugin,
                    "status": "completed",
                    "avgReward": round(avg_reward, 2),
                    "throughput": round(avg_throughput, 2),
                    "createdAt": datetime.now().isoformat(),
                    "progress": 100
                }
                experiments.append(exp_summary)
                save_index(experiments)
                # reward_curve: 优先从 shared_state 获取，fallback 从 result["training"] 提取
                reward_curves = shared_state.get("reward_curves", [])
                if not reward_curves:
                    reward_curves = result.get("training", [])
                # 保存配置文件到 results 目录
                from app.config import PROJECT_ROOT
                config_path = f"config_{exp_id}.yaml"
                config_full_path = PROJECT_ROOT / "results" / config_path
                with open(config_full_path, "w", encoding="utf-8") as f:
                    yaml.dump(custom_config, f, allow_unicode=True, sort_keys=False)
                detail = {
                    "id": exp_id,
                    "name": exp_name,
                    "datasetName": dataset_id,
                    "modelName": algorithm_plugin,
                    "metrics": shared_state.get("last_eval_metrics") or shared_state.get("last_episode_metrics", {}),
                    "reward_curve": [point.get("avg_reward", 0) for point in all_metrics],
                    "episode_metrics": all_metrics,
                    "signals": shared_state.get("signals", []),
                    "vehicles": shared_state.get("vehicles", []),
                    "road_network": shared_state.get("road_network", None),
                    "last_epoch": shared_state.get("current_epoch", "?"),
                    "gui_enabled": gui_enabled,
                    "config_path": config_path,
                }
                save_experiment_detail(exp_id, detail)
                active_experiments[exp_id]["status"] = "completed"
                active_experiments[exp_id]["avgReward"] = round(avg_reward, 2)
                active_experiments[exp_id]["progress"] = 100
        except Exception as e:
            import traceback
            print(f"[experiment] FATAL: {e}")
            traceback.print_exc()
            active_experiments[exp_id]["status"] = "failed"
            active_experiments[exp_id]["error"] = str(e)
        finally:
            if temp_yaml.exists():
                os.remove(temp_yaml)
    thread = threading.Thread(target=run_custom, daemon=True)
    thread.start()

    if gui_enabled:
        start_screenshot_thread(exp_id)

    return {"experiment_id": exp_id, "message": "自定义实验已启动"}

@router.post("/start")
async def start_experiment(data: dict):
    # 检查并发
    running = any(info["status"] == "running" for info in active_experiments.values())
    if running:
        raise HTTPException(status_code=409, detail="当前已有实验在运行")
    dataset_id = data.get("dataset_id")
    model_id = data.get("model_id")
    gui_enabled = data.get("config_overrides", {}).get("gui_enabled", False)
    user_overrides = data.get("config_overrides", {})
    training_override = user_overrides.get("training", {})
    eval_override = user_overrides.get("evaluation", {})

    full_config = get_default_config(model_id, dataset_id)
    full_config["training"].update(training_override)
    full_config["evaluation"].update(eval_override)
    full_config["environment"]["config"]["gui"] = gui_enabled

    # 路径转换
    roadnet_rel = full_config["environment"]["config"]["roadnet_file"]
    flow_rel = full_config["environment"]["config"]["flow_file"]
    roadnet_abs = (PROJECT_ROOT / roadnet_rel).as_posix()
    flow_abs = (PROJECT_ROOT / flow_rel).as_posix()
    print(f"Looking for roadnet at: {roadnet_abs}")
    print(f"Looking for flow at: {flow_abs}")
    if not os.path.isfile(roadnet_abs):
        raise HTTPException(status_code=400, detail="路网文件不存在")
    if not os.path.isfile(flow_abs):
        raise HTTPException(status_code=400, detail="车流文件不存在")
    full_config["environment"]["config"]["roadnet_file"] = roadnet_abs
    full_config["environment"]["config"]["flow_file"] = flow_abs

    exp_id = str(uuid.uuid4())[:8]
    exp_name = f"{model_id}_{dataset_id}_{exp_id}"
    full_config["experiment"]["name"] = exp_name

    CONFIGS_DIR.mkdir(exist_ok=True)
    temp_yaml = CONFIGS_DIR / f"auto_{exp_id}.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(full_config, f)

    stop_event = threading.Event()
    shared_state = {}
    active_experiments[exp_id] = {
        "status": "running", "progress": 0,
        "name": exp_name, "datasetName": dataset_id,
        "modelName": model_id, "createdAt": datetime.now().isoformat(),
        "stop_event": stop_event, "shared_state": shared_state,
        "gui_enabled": gui_enabled
    }

    start_experiment_thread(exp_id, dataset_id, model_id, full_config, gui_enabled, shared_state)
    if gui_enabled:
        start_screenshot_thread(exp_id)

    return {"experiment_id": exp_id, "message": "训练已启动"}

@router.delete("/clear_all")
def clear_all_experiments():
    global experiments
    experiments = []
    save_index([])
    for file in RESULTS_DIR.glob("exp_*.json"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    to_remove = [eid for eid, info in active_experiments.items()
                 if info["status"] in ("completed", "cancelled", "failed")]
    for eid in to_remove:
        del active_experiments[eid]
    return {"message": "所有已完成实验已清空"}

@router.get("/{exp_id}/monitor")
def monitor_experiment(exp_id: str):
    info = active_experiments.get(exp_id)
    if not info:
        exp = next((e for e in experiments if e["id"] == exp_id), None)
        if exp:
            detail = load_experiment_detail(exp_id)
            if detail:
                return {
                    "status": exp["status"],
                    "experiment": {
                        "name": exp['name'],
                        "datasetName": exp['datasetName'],
                        "modelName": exp['modelName'],
                        "epoch": detail.get("last_epoch", "?"),
                        "last_epoch": detail.get("last_epoch", "?")
                    },
                    "message": "实验已完成",
                    "metrics": detail.get("metrics", {}),
                    "signals": detail.get("signals", []),
                    "vehicles": detail.get("vehicles", []),
                    "road_network": detail.get("road_network"),
                    "gui_enabled": detail.get("gui_enabled", False),
                    "reward_curve": detail.get("reward_curve", [])
                }
            else:
                return {"status":"completed", "experiment": exp, "message": "实验已完成"}
        raise HTTPException(status_code=404, detail="实验不存在")
    
    status = info.get("status", "unknown")
    shared = info.get("shared_state", {})
    
    if status == "running":
        return {
            "status": "running",
            "message": "模型还在训练中，请稍后...",
            "experiment": {
                "name": info.get("name"),
                "datasetName": info.get("datasetName"),
                "modelName": info.get("modelName"),
            },
            "current_epoch": shared.get("current_epoch"),
            "total_epochs": shared.get("total_epochs"),
            "metrics": shared.get("last_episode_metrics", {}),
            "reward_curve": shared.get("reward_curve", []),
            "vehicles": shared.get("vehicles", []),
            "signals": shared.get("signals", []),
            "gui_enabled": info.get("gui_enabled", False)
        }
    elif status == "evaluating":
        env = shared.get("_env")
        road_network = None
        if env:
            road_network = extract_road_network(env)
        return {
            "current_epoch": shared.get("current_epoch"),
            "total_epochs": shared.get("total_epochs"),
            "status": "evaluating",
            "experiment": {
                "name": info.get("name"),
                "datasetName": info.get("datasetName"),
                "modelName": info.get("modelName"),
            },
            "metrics": shared.get("last_eval_metrics", {}),
            "signals": shared.get("signals", []),
            "vehicles": shared.get("vehicles", []),
            "road_network": road_network,
            "reward_curve": shared.get("reward_curve", []),
            "gui_enabled": info.get("gui_enabled", False)
        }
    elif status in ("completed", "cancelled", "failed"):
        detail = load_experiment_detail(exp_id)
        return {
            "status": status,
            "experiment": {
                "name":info.get("name"),
                "datasetName": info.get("datasetName"),
                "modelName": info.get("modelName"),
            },
            "message": "实验已结束",
            "metrics": detail.get("metrics", {}) if detail else {},
            "signals": detail.get("signals", []) if detail else [],
            "vehicles": detail.get("vehicles", []) if detail else [],
            "road_network": detail.get("road_network") if detail else None,
            "gui_enabled": detail.get("gui_enabled", False) if detail else info.get("gui_enabled", False),
            "reward_curve": detail.get("reward_curve", []) if detail else [],
            "epoch": detail.get("last_epoch", "?") if detail else "?",
            "last_epoch": detail.get("last_epoch", "?") if detail else "?"
        }

@router.post("/{exp_id}/stop")
def stop_experiment(exp_id: str):
    if exp_id in active_experiments and active_experiments[exp_id]["status"] == "running":
        stop_event = active_experiments[exp_id].get("stop_event")
        if stop_event:
            stop_event.set()
        active_experiments[exp_id]["status"] = "cancelling"
        return {"message": "已请求停止，训练循环将在当前步完成后优雅退出"}
    raise HTTPException(status_code=404, detail="实验不存在或不在运行中")

@router.get("/{exp_id}")
def get_experiment_detail(exp_id: str):
    for exp in experiments:
        if exp["id"] == exp_id:
            return exp
    if exp_id in active_experiments:
        info = active_experiments[exp_id]
        return {
            "id": exp_id,
            "name": info.get("name", f"exp_{exp_id}"),
            "datasetName": info.get("datasetName", ""),
            "modelName": info.get("modelName", ""),
            "status": info["status"],
            "avgReward": info.get("avgReward", 0),
            "throughput": info.get("throughput", 0),
            "createdAt": info.get("createdAt", ""),
            "progress": info.get("progress", 0)
        }
    raise HTTPException(status_code=404, detail="实验不存在")

@router.delete("/{exp_id}")
def delete_experiment(exp_id: str):
    global experiments
    experiments = [e for e in experiments if e["id"] != exp_id]
    save_index(experiments)
    delete_experiment_files(exp_id)
    if exp_id in active_experiments:
        del active_experiments[exp_id]
    return {"message": "删除成功"}


@router.put("/{exp_id}/rename")
def rename_experiment(exp_id: str, data: dict):
    global experiments
    name = data.get("name")
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="名称不能为空")
    for exp in experiments:
        if exp["id"] == exp_id:
            exp["name"] = name.strip()
            save_index(experiments)
            return {"message": "重命名成功"}
    raise HTTPException(status_code=404, detail="实验不存在")


def _start_experiment_with_config(config_id: str, gui_enabled: bool = False):
    """使用保存的配置文件启动实验"""
    config_path = RESULTS_DIR / f"config_{config_id}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="配置文件不存在")
    
    with open(config_path, "r", encoding="utf-8") as f:
        custom_config = yaml.safe_load(f)
    
    exp_id = str(uuid.uuid4())[:8]
    exp_name = f"retrain_{config_id}"
    
    dataset_id = custom_config.get("environment", {}).get("config", {}).get("roadnet_file", "unknown")
    if dataset_id != "unknown":
        dataset_id = dataset_id.split("/")[1]
    
    algorithm_plugin = "custom"
    if "algorithm" in custom_config and custom_config["algorithm"]:
        algo = custom_config["algorithm"]
        if isinstance(algo, list):
            algo = algo[0] if algo else {}
        algorithm_plugin = algo.get("plugin", "custom") if algo else "custom"
    
    temp_yaml = CONFIGS_DIR / f"auto_{exp_id}.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(custom_config, f)
    
    stop_event = threading.Event()
    shared_state = {}
    active_experiments[exp_id] = {
        "status": "running",
        "progress": 0,
        "name": exp_name,
        "datasetName": dataset_id,
        "modelName": algorithm_plugin,
        "createdAt": datetime.now().isoformat(),
        "stop_event": stop_event,
        "shared_state": shared_state,
        "gui_enabled": gui_enabled
    }
    
    def run_retrain():
        try:
            from modutsc.api import run_experiment
            
            detected_device = detect_device()
            active_experiments[exp_id]["device"] = detected_device
            
            overrides = {}
            if "algorithm" in custom_config:
                algos = custom_config.get("algorithm", [])
                if isinstance(algos, dict):
                    algos = [algos]
                for i, algo in enumerate(algos):
                    if isinstance(algo, dict) and "config" in algo:
                        if "device" not in algo["config"]:
                            algo["config"]["device"] = detected_device
                            if i < len(overrides.setdefault("algorithm", [])):
                                overrides["algorithm"][i]["config"]["device"] = detected_device
                            else:
                                overrides["algorithm"].append({"config": {"device": detected_device}})
            
            result = run_experiment(
                str(temp_yaml),
                stop_event=stop_event,
                overrides=overrides,
                shared_state=shared_state
            )
            
            if result.get("error"):
                active_experiments[exp_id]["status"] = "failed"
                active_experiments[exp_id]["error"] = result["error"]
                return
            
            all_metrics = result.get("training", [])
            was_stopped = result.get("stopped", False) or active_experiments[exp_id]["status"] == "cancelling"
            if was_stopped:
                active_experiments[exp_id]["status"] = "cancelled"
                active_experiments[exp_id]["progress"] = 0
            else:
                total_reward = sum(float(ep.get("avg_reward", 0)) for ep in all_metrics)
                avg_reward = total_reward / max(len(all_metrics), 1)
                total_throughput = sum(float(ep.get("Throughput", 0)) for ep in all_metrics)
                avg_throughput = total_throughput / max(len(all_metrics), 1)
                exp_summary = {
                    "id": exp_id,
                    "name": exp_name,
                    "datasetName": dataset_id,
                    "modelName": algorithm_plugin,
                    "status": "completed",
                    "avgReward": round(avg_reward, 2),
                    "throughput": round(avg_throughput, 2),
                    "createdAt": datetime.now().isoformat(),
                    "progress": 100
                }
                experiments.append(exp_summary)
                save_index(experiments)
                
                reward_curves = shared_state.get("reward_curves", [])
                if not reward_curves:
                    reward_curves = result.get("training", [])
                
                new_config_path = f"config_{exp_id}.yaml"
                new_config_full_path = PROJECT_ROOT / "results" / new_config_path
                with open(new_config_full_path, "w", encoding="utf-8") as f:
                    yaml.dump(custom_config, f, allow_unicode=True, sort_keys=False)
                
                detail = {
                    "id": exp_id,
                    "name": exp_name,
                    "datasetName": dataset_id,
                    "modelName": algorithm_plugin,
                    "metrics": shared_state.get("last_eval_metrics") or shared_state.get("last_episode_metrics", {}),
                    "reward_curve": [point["avg_reward"] for point in reward_curves],
                    "signals": shared_state.get("signals", []),
                    "vehicles": shared_state.get("vehicles", []),
                    "road_network": shared_state.get("road_network", None),
                    "last_epoch": shared_state.get("current_epoch", "?"),
                    "gui_enabled": gui_enabled,
                    "config_path": new_config_path,
                }
                save_experiment_detail(exp_id, detail)
                active_experiments[exp_id]["status"] = "completed"
                active_experiments[exp_id]["avgReward"] = round(avg_reward, 2)
                active_experiments[exp_id]["progress"] = 100
        except Exception as e:
            traceback.print_exc()
            active_experiments[exp_id]["status"] = "failed"
            active_experiments[exp_id]["error"] = str(e)
        finally:
            if temp_yaml.exists():
                os.remove(temp_yaml)
    
    thread = threading.Thread(target=run_retrain, daemon=True)
    thread.start()
    
    if gui_enabled:
        start_screenshot_thread(exp_id)
    
    return {"experiment_id": exp_id, "message": "使用保存的配置重新训练已启动"}