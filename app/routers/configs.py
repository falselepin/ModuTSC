from fastapi import APIRouter, HTTPException
import os
import yaml
from pathlib import Path
from datetime import datetime
from app.config import PROJECT_ROOT, RESULTS_DIR

router = APIRouter(prefix="/api/configs", tags=["configs"])


def _extract_dataset_name(roadnet_file: str) -> str:
    """从 roadnet_file 路径中提取数据集名称"""
    if not roadnet_file or roadnet_file == "unknown":
        return "unknown"
    # 支持绝对路径和相对路径
    parts = Path(roadnet_file).parts
    # 找到 "data" 目录后的第一个子目录名
    for i, p in enumerate(parts):
        if p == "data" and i + 1 < len(parts):
            return parts[i + 1]
    # fallback: 尝试 split
    return roadnet_file.split("/")[-2] if "/" in roadnet_file else "unknown"


def _extract_model_name(algo_cfg) -> str:
    """从 algorithm 配置中提取模型名称"""
    if not algo_cfg:
        return "custom"
    if isinstance(algo_cfg, list):
        algo_cfg = algo_cfg[0] if algo_cfg else {}
    if isinstance(algo_cfg, dict):
        return algo_cfg.get("plugin", "custom")
    return "custom"


@router.get("/")
def list_saved_configs():
    """获取所有已保存的配置文件列表"""
    configs = []
    results_dir = RESULTS_DIR
    if not results_dir.exists():
        return []
    
    for file in results_dir.glob("config_*.yaml"):
        exp_id = file.stem.replace("config_", "")
        try:
            with open(file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not cfg:
                continue
            roadnet_file = cfg.get("environment", {}).get("config", {}).get("roadnet_file", "")
            dataset_name = _extract_dataset_name(roadnet_file)
            model_name = _extract_model_name(cfg.get("algorithm"))
            config_info = {
                "id": exp_id,
                "filename": file.name,
                "datasetName": dataset_name,
                "modelName": model_name,
                "createdAt": file.stat().st_mtime,
                "config": cfg,
            }
            configs.append(config_info)
        except Exception as e:
            print(f"Error reading config {file}: {e}")
    
    configs.sort(key=lambda x: x["createdAt"], reverse=True)
    return configs


@router.delete("/{config_id}")
def delete_config(config_id: str):
    """删除指定的配置文件"""
    config_path = RESULTS_DIR / f"config_{config_id}.yaml"
    detail_path = RESULTS_DIR / f"detail_{config_id}.json"
    
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="配置文件不存在")
    
    try:
        os.remove(config_path)
        if detail_path.exists():
            os.remove(detail_path)
        return {"success": True, "message": "配置文件已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.post("/{config_id}/retrain")
def retrain_with_config(config_id: str, gui_enabled: bool = False):
    """使用保存的配置重新训练"""
    from app.routers.experiments import _start_experiment_with_config
    return _start_experiment_with_config(config_id, gui_enabled)