"""Launch an experiment from a YAML config file.

Usage:
  py -m modutsc run configs/xxx.yaml
  py -m modutsc run
  py -m modutsc index --rebuild       (rebuild datasets cache via simulator)
"""

import sys
import warnings

warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")

from modutsc.scheduling.launcher import Launcher
from modutsc.scheduling.registry import find, discover, list_all


def main():
    if len(sys.argv) < 2:
        print("Usage: py -m modutsc run [config_path]")
        print("       py -m modutsc index --rebuild")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "index":
        do_index(sys.argv[2:])
        return

    if cmd != "run":
        print("Usage: py -m modutsc run [config_path]")
        print("       py -m modutsc index --rebuild")
        sys.exit(1)

    config_path = sys.argv[2] if len(sys.argv) >= 3 else None
    if config_path is None:
        print("Usage: py -m modutsc run <config_path>")
        sys.exit(1)

    print(f"Loading: {config_path}")
    launcher = Launcher(config_path)
    orch = launcher.build()
    results = orch.run(launcher.config)
    print("Done.", results.get("training", [])[-1] if results.get("training") else results)
    if launcher.resolved_config:
        resolved_path = config_path.replace(".yaml", "_resolved.yaml")
        print(f"Full config: {resolved_path}")


def do_index(args):
    if not args or args[0] != "--rebuild":
        print("Usage: py -m modutsc index --rebuild")
        print("       Rebuild datasets_index.yaml by launching the simulator")
        print("       for each .net.xml in data/")
        sys.exit(1)

    discover()
    envs = list_all("environment")
    if not envs:
        print("[index] no environment plugin found; please check SUMO_HOME")
        sys.exit(1)
    env_cls = find("environment", envs[0])

    from modutsc.scheduling.dataset_index import rebuild_cache
    print("[index] rebuilding datasets_index.yaml via simulator ...")
    index = rebuild_cache("data", env_cls)
    print(f"[index] done, {len(index)} datasets indexed")


if __name__ == "__main__":
    main()
