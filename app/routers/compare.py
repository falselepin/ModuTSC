from fastapi import APIRouter
from app.services.file_service import load_experiment_detail
from app.routers.models import AVAILABLE_METRICS

router = APIRouter(prefix="/api/experiments", tags=["compare"])

def _resolve_metric(detail: dict, metric: dict) -> list:
    source = metric.get("source", "")
    episodes = detail.get("episode_metrics", [])

    if source == "reward_curve":
        return detail.get("reward_curve", [])

    if episodes:
        parts = source.split(".", 1)
        key = parts[1] if len(parts) == 2 else metric["value"]
        return [ep.get(key, ep.get(key.lower(), 0)) for ep in episodes]

    parts = source.split(".", 1)
    if len(parts) == 2:
        return detail.get(parts[0], {}).get(parts[1], 0)
    return 0

@router.get("/compare")
def compare_experiments(ids: str, metrics: str):
    id_list = ids.split(",")
    metric_list = metrics.split(",")
    result = []
    for eid in id_list:
        detail = load_experiment_detail(eid)
        if not detail:
            continue
        curves = {}
        for m_name in metric_list:
            metric_meta = next((m for m in AVAILABLE_METRICS if m["value"] == m_name), None)
            if metric_meta:
                curves[m_name] = _resolve_metric(detail, metric_meta)
        result.append({
            "id": eid,
            "name": detail.get("name", eid),
            "modelName": detail.get("modelName", ""),
            "datasetName": detail.get("datasetName", ""),
            "curves": curves
        })
    return result
