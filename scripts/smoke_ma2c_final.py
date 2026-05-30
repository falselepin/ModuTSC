import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = open("ma2c_final.txt", "w")
def p(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

from modutsc.scheduling.launcher import Launcher

launcher = Launcher("configs/ma2c_monaco.yaml")
orch = launcher.build()
p("Build done")

p("Warmup 50...")
w = orch.warmup(50)
p(f"Warmup: {w}")

p("Episode 1...")
m = orch.episode()
p("=== Episode Results ===")
for k, v in sorted(m.items()):
    if isinstance(v, float):
        p(f"  {k}: {v:.4f}")
    else:
        p(f"  {k}: {v}")

p("Eval 20...")
e = orch.evaluate(20)
p("=== Eval Results ===")
for k, v in sorted(e.items()):
    if isinstance(v, float):
        p(f"  {k}: {v:.4f}")
    else:
        p(f"  {k}: {v}")

orch.teardown()
p("DONE")
log.close()
