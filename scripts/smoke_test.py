import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

def main():
    config = "configs/dqn_monaco.yaml"
    print(f"Loading: {config}", flush=True)
    launcher = Launcher(config)
    print("Launcher created, building...", flush=True)
    orch = launcher.build()
    print("Build done, running...", flush=True)

    print("Running warmup...", flush=True)
    warmup = orch.warmup(50)
    print(f"Warmup done: {warmup}", flush=True)

    print("Running 1 episode...", flush=True)
    metrics = orch.episode()
    print(f"Episode done: {metrics}", flush=True)

    print("Running evaluate...", flush=True)
    eval_result = orch.evaluate(50)
    print(f"Eval done: {eval_result}", flush=True)

    orch.teardown()
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
