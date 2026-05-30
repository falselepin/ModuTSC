import uuid
from fastapi import APIRouter, HTTPException
from modutsc.api import (
    assembly_requirements, assembly_iterative,
    get_plugin_config_keys, create_constraint_session,
    list_datasets, get_dataset_topo, match_datasets_by_topo,
    list_modules_with_params,
    get_orchestrator_kinds, get_orchestrator_wiring,
    filter_plugins_by_dimension,
    scaffold_config as _scaffold_config,
)
from typing import Dict, Any, Optional

router = APIRouter(prefix="/api/assemble", tags=["assemble"])

_constraint_sessions: Dict[str, Any] = {}


@router.get("/modules")
def get_modules():
    return list_modules_with_params()


@router.get("/recommend")
def get_recommendation(orchestrator: str):
    plan = assembly_requirements(orchestrator)
    recommended = {}
    for kind, slot in plan.get("slots", {}).items():
        opts = slot.get("options", [])
        if opts:
            recommended[kind] = opts[0]["plugin"]
    return recommended


@router.get("/wiring")
def get_wiring(orchestrator: str):
    try:
        return get_orchestrator_wiring(orchestrator)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/orchestrator-kinds")
def get_orch_kinds_endpoint(orchestrator: str):
    return {"orchestrator": orchestrator, "kinds": get_orchestrator_kinds(orchestrator)}


@router.get("/iterative")
def get_iterative_assembly(selections: Optional[str] = None):
    sel = {}
    if selections:
        for pair in selections.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                sel[k.strip()] = v.strip()
    return assembly_iterative(sel)


@router.get("/config-keys")
def get_config_keys(kind: str, plugin: str):
    try:
        keys = get_plugin_config_keys(kind, plugin)
        return {"kind": kind, "plugin": plugin, "config_keys": keys}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/dimension-filter")
def get_dimension_filter(orchestrator: str, dataset_path: str):
    try:
        return filter_plugins_by_dimension(orchestrator, dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"维度兼容性检测失败: {str(e)}")


@router.get("/datasets-topo")
def get_datasets_with_topo():
    ds_list = list_datasets()
    return ds_list


@router.get("/dataset-topo")
def get_single_dataset_topo(path: str):
    topo = get_dataset_topo(path)
    if not topo:
        raise HTTPException(status_code=404, detail="数据集拓扑未找到")
    return topo


@router.post("/constraint-session")
def create_constraint_session_endpoint(data: dict):
    orch_name = data.get("orchestrator")
    selections = data.get("selections", {})
    dataset_path = data.get("dataset_path")

    if not orch_name:
        raise HTTPException(status_code=400, detail="缺少编排器名称")

    try:
        session = create_constraint_session(orch_name, selections, dataset_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"创建约束会话失败: {str(e)}")

    session_id = str(uuid.uuid4())[:8]
    _constraint_sessions[session_id] = session

    state = session.get_state()
    return {"session_id": session_id, "state": state}


@router.get("/constraint-session/{session_id}/state")
def get_constraint_state(session_id: str):
    session = _constraint_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session.get_state()


@router.post("/constraint-session/{session_id}/set-value")
def set_constraint_value(session_id: str, data: dict):
    session = _constraint_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    kind = data.get("kind")
    key = data.get("key")
    value = data.get("value")

    if not kind or not key:
        raise HTTPException(status_code=400, detail="缺少 kind 或 key")

    delta = session.set_value(kind, key, value)
    return {"delta": delta, "state": session.get_state()}


@router.post("/constraint-session/{session_id}/recommend")
def get_constraint_recommend(session_id: str, data: dict):
    session = _constraint_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    kind = data.get("kind")
    key = data.get("key")

    if not kind or not key:
        raise HTTPException(status_code=400, detail="缺少 kind 或 key")

    return session.recommend(kind, key)


@router.post("/generate-config")
def generate_config(data: dict):
    orch_name = data.get("orchestrator", "")
    selections = data.get("selections", {})
    config_params = data.get("config_params", {})
    experiment_name = data.get("experiment_name", "auto_exp")
    dataset_id = data.get("dataset_id", "")
    gui_enabled = data.get("gui_enabled", False)

    if not orch_name or not selections:
        raise HTTPException(status_code=400, detail="缺少 orchestrator 或 selections")

    full_selections = dict(selections)
    full_selections["environment"] = full_selections.get("environment", "sumo")

    cfg = _scaffold_config(orch_name, full_selections, config_params)

    cfg["experiment"]["name"] = experiment_name
    cfg["environment"]["config"]["gui"] = gui_enabled

    if dataset_id:
        cfg["environment"]["config"]["roadnet_file"] = f"data/{dataset_id}/roadnet.net.xml"
        cfg["environment"]["config"]["flow_file"] = f"data/{dataset_id}/flow_0.rou.xml"

    training = config_params.get("training", {})
    evaluation = config_params.get("evaluation", {})
    if training:
        cfg.setdefault("training", {}).update(training)
    if evaluation:
        cfg.setdefault("evaluation", {}).update(evaluation)

    return cfg
