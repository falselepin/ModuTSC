import sys, os, time, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modutsc.scheduling.launcher import Launcher

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ma2c_monaco_run.log")

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

def main():
    config = "configs/ma2c_monaco.yaml"
    log(f"Loading: {config}")
    launcher = Launcher(config)
    log("Building...")
    orch = launcher.build()
    log(f"Build done. Agents: {len(orch._env.ids())}")

    t0 = time.time()
    cfg = launcher.config

    # Modify training to run with progress logging
    training_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    warmup_steps = training_cfg.get("warmup_steps", 200)
    if warmup_steps > 0:
        log(f"Starting warmup ({warmup_steps} steps)...")
        result = orch.warmup(warmup_steps)
        log(f"Warmup done: {result}")

    num_epochs = training_cfg.get("num_epochs", 1400)
    eps_per_epoch = training_cfg.get("episodes_per_epoch", 1)
    eval_freq = eval_cfg.get("eval_frequency", 20)
    eval_steps = eval_cfg.get("eval_steps", 720)
    checkpoint_dir = eval_cfg.get("checkpoint_dir", "checkpoints/ma2c_monaco/")

    for epoch in range(num_epochs):
        for ep in range(eps_per_epoch):
            m = orch.episode()
            m["epoch"] = epoch
            log(f"[Epoch {epoch:3d}/{num_epochs}] avg_reward={m['avg_reward']:.4f} "
                f"steps={m.get('steps', '?')} "
                f"v_loss={m.get('value_loss', float('nan')):.4f} "
                f"p_loss={m.get('policy_loss', float('nan')):.4f}")

        if (epoch + 1) % eval_freq == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            eval_result = orch.evaluate(eval_steps)
            log(f"[Eval epoch {epoch:3d}] {eval_result}")
            orch.save(f"{checkpoint_dir}/ckpt_epoch_{epoch + 1}.pkl")

    elapsed = time.time() - t0
    orch.teardown()
    log(f"Experiment completed in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    log("DONE")

if __name__ == "__main__":
    main()
