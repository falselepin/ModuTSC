import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

launcher = Launcher("configs/ma2c_monaco.yaml")
orch = launcher.build()

orch.warmup(6)
m = orch.episode()

result = {}
for k, v in sorted(m.items()):
    if isinstance(v, float):
        result[k] = round(v, 4)
    else:
        result[k] = v

with open("step_result.json", "w") as f:
    json.dump(result, f, indent=2)

orch.teardown()
