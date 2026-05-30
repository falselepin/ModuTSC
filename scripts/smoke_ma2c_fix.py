import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

launcher = Launcher("configs/ma2c_monaco.yaml")
orch = launcher.build()
print("Build done", flush=True)
print("Warmup 6...", flush=True)
w = orch.warmup(6)
print(f"Warmup: {w}", flush=True)
print("Episode 6 steps...", flush=True)
m = orch.episode()
print("Results:", flush=True)
for k, v in sorted(m.items()):
    if isinstance(v, float): print(f"  {k}: {v:.4f}", flush=True)
    else: print(f"  {k}: {v}", flush=True)
orch.teardown()
print("DONE (no phase errors = fix works)", flush=True)
