#!/usr/bin/env python3
"""W&B-sweep entrypoint for the MemoryEnv R2-Dreamer trainings.

Recycles the cluster's existing wandb-agent + SLURM-array sweep mechanism
(`job_memory.slurm` runs `wandb agent <sweep>`; this file is the sweep
`program`). The swept grid parameter is the cue set (2cue/3cue/4cue); this
wrapper translates it into the r2dreamer hydra command and runs the training as
a subprocess, so the hydra entrypoint and wandb stay cleanly separated.

The wandb run is just a dispatch/record; r2dreamer writes its own
tensorboard logs + the `latest.pt` checkpoint under
``external/r2dreamer/runs/memory_<cue>/``.
"""
from __future__ import annotations
import os
import subprocess
import sys

import wandb


def main() -> int:
    run = wandb.init()                       # picks up the sweep-injected config
    cfg = run.config
    cue = str(cfg["cue"])                     # "2cue" | "3cue" | "4cue"
    model = str(cfg.get("model", "size25M"))
    steps = str(cfg.get("steps", "10e6"))
    seed = int(cfg.get("seed", 0))

    repo = os.environ.get("PROJECT_DIR", os.getcwd())
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}/src" + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable, "external/r2dreamer/train.py",
        "env=memory", f"env.task=memory_{cue}",
        f"model={model}", f"env.steps={steps}",
        "device=cuda:0", f"seed={seed}",
        f"logdir=external/r2dreamer/runs/memory_{cue}",
    ]
    print("[sweep_train] cue=%s model=%s steps=%s seed=%d" % (cue, model, steps, seed),
          flush=True)
    print("[sweep_train] cmd:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=repo, env=env).returncode
    wandb.finish(exit_code=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
