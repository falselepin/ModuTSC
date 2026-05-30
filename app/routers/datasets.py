import os
from fastapi import APIRouter
from app.config import DATA_DIR, DATASET_META
from app.utils.flow_parser import parse_dataset_flows
from app.utils.roadnet_parser import parse_roadnet

router = APIRouter(prefix="/api", tags=["datasets"])

# 缓存：避免每次请求都重新解析
_curve_cache = {}
_roadnet_cache = {}


@router.get("/datasets")
def get_datasets():
    datasets = []
    if not os.path.exists(DATA_DIR):
        return []
    for folder in sorted(os.listdir(DATA_DIR)):
        full_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(full_path):
            continue
        meta = DATASET_META.get(folder, {"typeTag": "未知", "description": "无描述"})

        # 从缓存获取或解析真实流量数据
        if folder not in _curve_cache:
            _curve_cache[folder] = parse_dataset_flows(full_path)
        flow_data = _curve_cache[folder]

        datasets.append({
            "id": folder,
            "name": folder,
            "typeTag": meta["typeTag"],
            "description": meta["description"],
            "vehicleCurve": flow_data["vehicleCurve"],
            "totalVehicles": flow_data["total_vehicles"],
            "simDurationSec": flow_data["sim_duration_sec"],
            "flowFiles": flow_data["flow_files"],
        })
    return datasets


@router.get("/datasets/{dataset_id}/flow-detail")
def get_dataset_flow_detail(dataset_id: str):
    """获取数据集的详细流量信息，供前端展示更丰富的图表。"""
    dataset_dir = os.path.join(DATA_DIR, dataset_id)
    if not os.path.isdir(dataset_dir):
        return {"error": "数据集不存在"}

    if dataset_id not in _curve_cache:
        _curve_cache[dataset_id] = parse_dataset_flows(dataset_dir)
    flow_data = _curve_cache[dataset_id]

    meta = DATASET_META.get(dataset_id, {"typeTag": "未知", "description": "无描述"})

    return {
        "id": dataset_id,
        "typeTag": meta["typeTag"],
        "description": meta["description"],
        "vehicleCurve": flow_data["vehicleCurve"],
        "totalVehicles": flow_data["total_vehicles"],
        "simDurationSec": flow_data["sim_duration_sec"],
        "flowFiles": flow_data["flow_files"],
    }


@router.get("/datasets/{dataset_id}/roadnet")
def get_dataset_roadnet(dataset_id: str):
    """获取数据集的路网拓扑数据，供前端 Three.js 渲染。"""
    dataset_dir = os.path.join(DATA_DIR, dataset_id)
    if not os.path.isdir(dataset_dir):
        return {"error": "数据集不存在"}

    if dataset_id not in _roadnet_cache:
        # 查找 .net.xml 文件
        net_file = None
        for f in os.listdir(dataset_dir):
            if f.endswith(".net.xml"):
                net_file = os.path.join(dataset_dir, f)
                break
        if net_file is None:
            _roadnet_cache[dataset_id] = {
                "nodes": [], "edges": [], "traffic_lights": [],
                "bounds": {"xMin": 0, "yMin": 0, "xMax": 1000, "yMax": 1000},
            }
        else:
            _roadnet_cache[dataset_id] = parse_roadnet(net_file)

    return _roadnet_cache[dataset_id]
