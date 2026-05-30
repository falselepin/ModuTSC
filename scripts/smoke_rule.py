import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

def main():
    config = "configs/fixed_time_monaco.yaml"
    print(f"Loading: {config}", flush=True)
    launcher = Launcher(config)
    print("Building...", flush=True)
    orch = launcher.build()
    print("Build done.", flush=True)

    print("Running episode...", flush=True)
    m = orch.episode()
    print(f"Episode: {m}", flush=True)

    print("Running evaluate...", flush=True)
    e = orch.evaluate(20)
    print(f"Eval: {e}", flush=True)

    orch.teardown()
    print("FIXEDTIME DONE", flush=True)

if __name__ == "__main__":
    main()
