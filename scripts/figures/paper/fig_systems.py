#!/usr/bin/env python3
"""Systems figure for the training chapter, from measured runs on an RTX 3090.

(a) end-to-end PPO training throughput, the PyTorch pipeline (host NumPy env,
    CleanRL-style recurrent PPO) against the matched PureJaxRL-style JAX pipeline
    (on-device env + jitted rollout/update), versus the number of parallel
    environments -- from scripts/bridge_tunnel/profile_ppo.py (PyTorch) and
    scripts/bridge_tunnel/profile_ppo_jax.py (JAX). Both are ~2.0M-param
    recurrent actor-critics on the same fork_wall task.
(b) the PyTorch PPO iteration wall-clock, rollout against update, with the
    rollout split into host env stepping, policy forward, and transfer
    -- from scripts/bridge_tunnel/profile_ppo.py.

  PYTHONPATH=src python scripts/figures/paper/fig_systems.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "paper/figures/forkwall_paper"

# --- measured: end-to-end PPO training throughput (env-steps/s), RTX 3090 ---
# both ~2.0M-param recurrent PPO, 128-step rollout, 4 epochs x 4 minibatches.
PT_B = [32, 128, 512]
PT = [2_003, 4_015, 5_306]                   # PyTorch, host env (saturates)
JX_B = [32, 256, 1024, 2048]
JX = [10_696, 36_858, 51_956, 59_989]        # JAX, on-device

# --- measured: PyTorch PPO iteration breakdown (ms/iter), 32 envs x 128 ---
PPO_ENV, PPO_FWD, PPO_XFER, PPO_UPD = 596.6, 174.3, 16.8, 1244.8
PPO_ROLL = PPO_ENV + PPO_FWD + PPO_XFER
PPO_ITER = PPO_ROLL + PPO_UPD

C_PT, C_JAX = "#ee4c2c", "#0ea5e9"           # PyTorch orange, JAX blue
C_ENV, C_FWD, C_XFER, C_UPD = "#ef4444", "#f59e0b", "#a3a3a3", "#2563eb"

rc = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9}
with plt.rc_context(rc):
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.0),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))

    # ---- panel (a): PyTorch vs JAX PPO training throughput ----
    a = ax[0]
    ptb = [b for b, v in zip(PT_B, PT) if v]
    ptv = [v for v in PT if v]
    a.loglog(ptb, ptv, "o-", color=C_PT, lw=2, ms=7, label="PyTorch PPO (host env)")
    a.loglog(JX_B, JX, "s-", color=C_JAX, lw=2, ms=7, label="JAX PPO (on-device)")
    # speedup at the matched batch B=32
    if PT[0]:
        a.annotate(f"{JX[0]/PT[0]:.1f}$\\times$ at 32 envs", xy=(32, JX[0]),
                   xytext=(6, 8), textcoords="offset points", fontsize=8.5, color=C_JAX)
    a.set_xlabel("parallel environments (batch $B$)")
    a.set_ylabel("training throughput (env-steps / s)")
    a.set_title("(a) End-to-end PPO training: PyTorch vs JAX (both $\\approx$2.0M params)",
                loc="left", fontsize=9.5)
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    a.grid(True, which="both", alpha=.15)

    # ---- panel (b): PPO iteration breakdown ----
    b = ax[1]
    # stacked horizontal bars: rollout (env/fwd/xfer) and update
    y = 1
    left = 0
    for val, col, lab in [(PPO_ENV, C_ENV, "host env step"),
                          (PPO_FWD, C_FWD, "policy forward"),
                          (PPO_XFER, C_XFER, "transfer")]:
        b.barh(y, val, left=left, color=col, edgecolor="white", label=lab)
        left += val
    b.barh(0, PPO_UPD, color=C_UPD, edgecolor="white", label="update (BPTT)")
    b.set_yticks([0, 1]); b.set_yticklabels(["update", "rollout"])
    b.set_xlabel("wall-clock per iteration (ms)")
    b.set_title("(b) PPO iteration: rollout is host-bound\n"
                "RTX 3090, 2.0M params, 32$\\times$128", loc="left", fontsize=9.5)
    b.annotate(f"{PPO_ROLL:.0f} ms  ({100*PPO_ROLL/PPO_ITER:.0f}%)",
               xy=(PPO_ROLL, 1), xytext=(6, 0), textcoords="offset points",
               va="center", fontsize=8)
    b.annotate(f"{PPO_UPD:.0f} ms  ({100*PPO_UPD/PPO_ITER:.0f}%)",
               xy=(PPO_UPD, 0), xytext=(6, 0), textcoords="offset points",
               va="center", fontsize=8)
    b.text(PPO_ENV / 2, 1.34, f"{100*PPO_ENV/PPO_ROLL:.0f}% of rollout",
           ha="center", va="center", fontsize=7.5, color=C_ENV)
    b.set_xlim(0, PPO_UPD * 1.25); b.set_ylim(-0.6, 1.7)
    b.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=1)

    fig.tight_layout()
    fig.savefig(OUT / "fig_res_systems.png", bbox_inches="tight")
    print("wrote fig_res_systems.png")
