import os
from pathlib import Path


def _query_topo_from_env(env_cls, roadnet_path: str) -> dict:
    _ensure_sumo_shutdown()
    env = env_cls()
    env.launch({
        "roadnet_file": roadnet_path,
        "gui": False,
    })
    ids = env.ids()
    topo = {
        "num_tsc": len(ids),
        "num_phase": max((env.phase_count(j) for j in ids), default=1),
    }
    incoming = []
    controlled = []
    green_counts = []
    for j in ids:
        il = env.incoming_lanes(j)
        incoming.append(len(il))
        cl = env.controlled_lanes(j)
        controlled.append(len(cl))
        green_counts.append(len(env.green_phase_indices(j)))
    topo["max_lanelinks"] = max(incoming, default=1) if incoming else 1
    topo["total_incoming_lanes"] = sum(incoming)
    topo["max_controlled_lanes"] = max(controlled, default=1) if controlled else 1
    topo["max_green_phases"] = max(green_counts, default=2) if green_counts else 2
    if hasattr(env, "all_edge_ids"):
        topo["total_edges"] = len(env.all_edge_ids())
    env.close()
    _ensure_sumo_shutdown()
    return topo


def _ensure_sumo_shutdown():
    import time
    import subprocess
    waited = 0
    while waited < 10:
        try:
            result = subprocess.run(
                ["tasklist", "/fi", "imagename eq sumo.exe"],
                capture_output=True, text=True, timeout=5,
                encoding='utf-8', errors='ignore'
            )
            if "sumo.exe" not in result.stdout:
                return
        except Exception:
            pass
        time.sleep(1)
        waited += 1


def ensure_cache(data_dir: str, env_cls, cache_path: str = "data/datasets_index.yaml") -> dict:
    index = load_index(cache_path)
    data_dir_path = Path(data_dir)
    for root, _, files in os.walk(data_dir_path):
        for f in files:
            if f.endswith(".net.xml"):
                if "_temp" in f:
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, Path(cache_path).parent).replace("\\", "/")
                if rel in index:
                    continue
                try:
                    index[rel] = _query_topo_from_env(env_cls, full)
                except Exception:
                    continue
    save_index(cache_path, index)
    return index


def rebuild_cache(data_dir: str, env_cls, cache_path: str = "data/datasets_index.yaml") -> dict:
    index = {}
    data_dir_path = Path(data_dir)
    for root, _, files in os.walk(data_dir_path):
        for f in files:
            if f.endswith(".net.xml"):
                if "_temp" in f:
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, Path(cache_path).parent).replace("\\", "/")
                try:
                    topo = _query_topo_from_env(env_cls, full)
                    index[rel] = topo
                    print(f"  indexed: {rel} -> {topo}")
                except Exception as e:
                    print(f"  skip: {rel}: {e}")
    save_index(cache_path, index)
    return index


def save_index(cache_path: str, index: dict) -> None:
    out = Path(cache_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(out, "w", encoding="utf-8") as fp:
        yaml.safe_dump(index, fp, allow_unicode=True, sort_keys=False)


def load_index(index_path: str) -> dict:
    if not os.path.exists(index_path):
        return {}
    import yaml
    with open(index_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_topo(roadnet: str, cache_path: str = "data/datasets_index.yaml") -> dict:
    if not roadnet:
        return {}
    index = load_index(cache_path)
    target = roadnet.replace("\\", "/")

    for path, meta in index.items():
        if target == path:
            return dict(meta)

    target_norm = target.rstrip("/")
    for path, meta in index.items():
        path_norm = path.rstrip("/")
        if target_norm == path_norm or target_norm.endswith("/" + path_norm) or path_norm.endswith("/" + target_norm):
            return dict(meta)

    target_tail = "/".join(target_norm.rsplit("/", 2)[-2:]) if target_norm.count("/") >= 1 else target_norm
    for path, meta in index.items():
        if path.rstrip("/").endswith(target_tail):
            return dict(meta)

    return {}


def match_datasets(constraints: dict, index: dict = None) -> list:
    if index is None:
        index = load_index("data/datasets_index.yaml")
    matches = []
    for path, meta in index.items():
        ok = True
        for key, spec in constraints.items():
            if key not in meta:
                ok = False
                break
            if not _match_spec(meta[key], spec):
                ok = False
                break
        if ok:
            matches.append({"path": path, **meta})
    return matches


def _match_spec(value, spec):
    if spec is None:
        return True
    if isinstance(spec, (int, float)):
        return value == spec
    if isinstance(spec, str) and ".." in spec:
        lo, hi = spec.split("..", 1)
        try:
            lo = int(lo) if lo else 0
            hi = int(hi) if hi else float("inf")
        except (ValueError, OverflowError):
            return False
        return lo <= value <= hi
    if isinstance(spec, str) and spec.startswith(">="):
        try:
            return value >= int(spec[2:])
        except (ValueError, OverflowError):
            return False
    if isinstance(spec, str) and spec.startswith("<="):
        try:
            return value <= int(spec[2:])
        except (ValueError, OverflowError):
            return False
    return value == spec
