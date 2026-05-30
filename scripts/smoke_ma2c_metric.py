import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

config = "configs/ma2c_monaco.yaml"
print(f"Loading: {config}", flush=True)
launcher = Launcher(config)
orch = launcher.build()
print("Build done.", flush=True)

print("Warmup 50...", flush=True)
w = orch.warmup(50)
print(f"Warmup: {w}", flush=True)

print("Episode...", flush=True)
m = orch.episode()
print(f"Episode done", flush=True)

print("Results:", flush=True)
for k, v in sorted(m.items()):
    print(f"  {k}: {v}", flush=True)

print("Eval 20...", flush=True)
e = orch.evaluate(20)
print(f"Eval done", flush=True)
for k, v in sorted(e.items()):
    print(f"  {k}: {v}", flush=True)

orch.teardown()
print("ALL DONE", flush=True)
