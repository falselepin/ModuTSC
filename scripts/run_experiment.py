import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modutsc.scheduling.launcher import Launcher


def main():
    config = sys.argv[1] if len(sys.argv) >= 2 else "configs/dqn_monaco.yaml"
    print(f"Loading: {config}")
    launcher = Launcher(config)
    orch = launcher.build()
    results = orch.run(launcher.config)
    print("Done.", results)


if __name__ == "__main__":
    main()
