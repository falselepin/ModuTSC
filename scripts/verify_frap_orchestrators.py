"""Smoke-verify FRAP with single / independent orchestrators."""
import copy
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _patch_config(orch_plugin: str) -> dict:
    import yaml

    path = os.path.join(ROOT, "configs", "frap.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["orchestrator"]["plugin"] = orch_plugin
    cfg["training"]["warmup_steps"] = 0
    cfg["training"]["num_epochs"] = 1
    cfg["training"]["episodes_per_epoch"] = 1
    cfg["evaluation"]["eval_steps"] = 3
    cfg["orchestrator"]["config"]["max_decision_steps"] = 15
    cfg["collector"]["config"]["batch_size"] = 8
    tmp = os.path.join(ROOT, "configs", f"_verify_frap_{orch_plugin}.yaml")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return tmp


def verify_build(orch_plugin: str) -> dict:
    from modutsc.scheduling.launcher import Launcher
    from modutsc.scheduling.registry import discover, find

    discover()
    if find("orchestrator", orch_plugin) is None:
        raise RuntimeError(f"orchestrator '{orch_plugin}' not registered")

    cfg_path = _patch_config(orch_plugin)
    launcher = Launcher(cfg_path)
    orch = launcher.build()
    env = orch._env
    obs = orch._observer
    algos = orch._algos

    ids = env.ids()
    obs_list = obs.observe(env)
    dim_obs = len(obs_list[0]["features"]) if obs_list else 0
    dim_cfg = obs.dim()

    checks = {
        "orchestrator": orch_plugin,
        "n_tls": len(ids),
        "n_algos": len(algos),
        "observer_dim": dim_cfg,
        "obs_feature_len": dim_obs,
        "resolved_num_phase": launcher.resolved_config.get("algorithm", [{}])[0]
        if isinstance(launcher.resolved_config.get("algorithm"), list)
        else launcher.resolved_config.get("algorithm", {}).get("config", {}).get("num_phase"),
    }
    if isinstance(launcher.resolved_config.get("algorithm"), list):
        ac = launcher.resolved_config["algorithm"][0].get("config", {})
    else:
        ac = (launcher.resolved_config.get("algorithm") or {}).get("config", {})
    checks["resolved_max_lanelinks"] = ac.get("max_lanelinks")

    algo0 = algos[0]
    if hasattr(algo0, "_num_phase"):
        checks["algo_num_phase"] = algo0._num_phase
        checks["algo_max_lanelink"] = algo0._num_lanelink
    if hasattr(algo0, "_p2l"):
        checks["p2l_shape"] = tuple(algo0._p2l.shape)
        checks["p2l_is_eye"] = bool(
            __import__("torch").allclose(
                algo0._p2l,
                __import__("torch").eye(
                    algo0._p2l.shape[0], algo0._p2l.shape[1]
                )[: algo0._p2l.shape[0], : algo0._p2l.shape[1]].to(algo0._p2l.device),
            )
        )

    if dim_obs != dim_cfg:
        raise AssertionError(f"obs dim mismatch: observe={dim_obs} dim()={dim_cfg}")

    for i, o in enumerate(obs_list):
        a = algos[0].act(o) if len(algos) == 1 else algos[i].act(o)
        if a["agent_id"] != o["id"]:
            raise AssertionError(f"agent id mismatch at {i}")

    orch.teardown()
    try:
        os.remove(cfg_path)
    except OSError:
        pass
    return checks


def verify_run(orch_plugin: str) -> dict:
    from modutsc.scheduling.launcher import Launcher
    from modutsc.scheduling.registry import discover

    discover()
    cfg_path = _patch_config(orch_plugin)
    launcher = Launcher(cfg_path)
    orch = launcher.build()
    w = orch.warmup(10)
    ep = orch.episode()
    ev = orch.evaluate(3)
    orch.teardown()
    try:
        os.remove(cfg_path)
    except OSError:
        pass
    return {"warmup": w, "episode_steps": ep.get("steps"), "eval": ev}


def main():
    if not os.environ.get("SUMO_HOME"):
        print("SKIP runtime: SUMO_HOME not set")
        return 2

    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"SKIP runtime: missing dependency: {e}")
        return 2

    from modutsc.scheduling.registry import discover

    discover()
    print("=== Static registry ===")
    from modutsc.scheduling.registry import find

    for name in ("single", "independent"):
        cls = find("orchestrator", name)
        print(f"  {name}: {'OK' if cls else 'MISSING'}")
        if cls:
            print(f"    compatible algorithm: {cls.__compatible_plugins__.get('algorithm')}")

    ok = True
    for orch in ("single", "independent"):
        print(f"\n=== Build check: {orch} ===")
        try:
            info = verify_build(orch)
            for k, v in info.items():
                print(f"  {k}: {v}")
        except Exception:
            ok = False
            traceback.print_exc()

        print(f"\n=== Run check: {orch} ===")
        try:
            info = verify_run(orch)
            print(f"  {info}")
        except Exception:
            ok = False
            traceback.print_exc()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
