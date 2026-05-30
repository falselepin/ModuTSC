import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

def main():
    config = "configs/igrl_monaco.yaml"
    print(f"Loading: {config}", flush=True)
    launcher = Launcher(config)
    print("Building...", flush=True)
    orch = launcher.build()
    print("Build done.", flush=True)

    print("Warmup 30...", flush=True)
    w = orch.warmup(30)
    print(f"Warmup: {w}", flush=True)

    print("Episode 1...", flush=True)
    m = orch.episode()
    print(f"Episode: {m}", flush=True)

    print("Eval 15...", flush=True)
    e = orch.evaluate(15)
    print(f"Eval: {e}", flush=True)

    orch.teardown()
    print("IGRL DONE", flush=True)

if __name__ == "__main__":
    main()
