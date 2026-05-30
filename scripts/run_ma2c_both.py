import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

def run_ma2c(config_name, label):
    config = f"configs/{config_name}.yaml"
    print(f"\n{'='*60}", flush=True)
    print(f"MA2C on {label}", flush=True)
    print(f"{'='*60}", flush=True)

    launcher = Launcher(config)
    orch = launcher.build()
    print("Build done.", flush=True)

    print("Warmup 100 steps...", flush=True)
    w = orch.warmup(100)
    print(f"Warmup: {w}", flush=True)

    print("\nEpisode 1:", flush=True)
    m = orch.episode()

    print("\nEpisode 1 results:", flush=True)
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nEvaluate 30 steps:", flush=True)
    e = orch.evaluate(30)
    for k, v in e.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    orch.teardown()
    print(f"\n{label} DONE", flush=True)
    return m


if __name__ == "__main__":
    results = {}
    results["Monaco"] = run_ma2c("ma2c_monaco", "Monaco")

    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY: First Episode Metrics", flush=True)
    print(f"{'='*60}", flush=True)
    for label, ep in results.items():
        print(f"\n--- {label} ---", flush=True)
        print(f"  avg_reward:       {ep.get('avg_reward', 'N/A'):.4f}", flush=True)
        print(f"  avg_travel_time:  {ep.get('avg_travel_time', 'N/A'):.2f}s", flush=True)
        print(f"  throughput/step:  {ep.get('throughput_per_step', 'N/A'):.4f}", flush=True)
        print(f"  avg_delay:        {ep.get('avg_delay', 'N/A'):.4f}s", flush=True)
        print(f"  completed_trips:  {ep.get('completed_trips', 'N/A')}", flush=True)
        print(f"  steps:            {ep.get('steps', 'N/A')}", flush=True)
