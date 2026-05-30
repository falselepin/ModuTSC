import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

config = "configs/ma2c_monaco.yaml"
print(f"Loading: {config}", flush=True)
launcher = Launcher(config)
orch = launcher.build()
print("Build done. SUMO GUI will appear.", flush=True)

print("Warmup 200 steps...", flush=True)
w = orch.warmup(200)
print(f"Warmup: {w}", flush=True)

print("Episode running (720 steps, ~3600s sim time)...", flush=True)
m = orch.episode()
print("=== Episode Results ===", flush=True)
for k, v in sorted(m.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

orch.teardown()
print("DONE", flush=True)
