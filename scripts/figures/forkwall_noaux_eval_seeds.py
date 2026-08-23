#!/usr/bin/env python3
"""Held-out success statistics for the fork_wall no-aux seed sweep.

Training-time success is measured on freshly generated maps with a stochastic
policy, which is fine for curves but is not a clean final number. This runs the
proper held-out evaluation (seeds >= 10000, disjoint from training) on each
seed's final.pt and reports, per seed and aggregated mean +/- std:

  success (category-correct door) / wrong-door / timeout, overall and per
  category, plus the map -> door choice matrix that shows WHETHER a seed
  conditions on the belief at all or collapsed into the constant-door basin.

    python scripts/figures/forkwall_noaux_eval_seeds.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bridge_tunnel"))

from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy, batched_rollout, _door_of  # noqa: E402

DOORS = ["top", "bottom", "none"]
CORRECT_DOOR = {"lakes": "bottom", "rocky": "top", "balanced": "either"}
CAT_COLOR = {"balanced": "#5C6B57", "lakes": "#1E6FA6", "rocky": "#A3572A"}


def eval_one(ckpt: Path, device, n_maps, n_traj, seed_start, max_steps):
    policy, cargs, view_size, env_size, env_width = _load_policy(ckpt, device)
    commit = False if cargs.get("no_commit", False) else None
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    rows, door_mat = {}, np.zeros((3, 3))
    for ci, cat in enumerate(CATEGORIES):
        succ, wrong, tout = [], [], []
        for j in range(n_maps):
            rec = generate_commit_map(size=env_size, width=env_width, seed=seed_start + j,
                                      category=cat, tree_frac=cargs.get("tree_frac", 0.03),
                                      goal_half=gh, fork_wall=True,
                                      passage_half=cargs.get("passage_half", 1),
                                      wall_margin=cargs.get("wall_margin", 1))
            o = batched_rollout(policy, rec, n_traj, view_size, max_steps, device, commit=commit)
            succ.extend(o["success"].tolist())
            wrong.extend((o["reached_any"] & ~o["success"]).tolist())
            tout.extend((~o["reached_any"]).tolist())
            for p in o["final_pos"]:
                d = _door_of(rec, p)
                door_mat[ci, DOORS.index(d if d in ("top", "bottom") else "none")] += 1
        rows[cat] = {"success": float(np.mean(succ)), "wrong_door": float(np.mean(wrong)),
                     "timeout": float(np.mean(tout)), "n": len(succ)}
    door_mat = door_mat / door_mat.sum(axis=1, keepdims=True).clip(min=1)
    overall = {k: float(np.mean([rows[c][k] for c in CATEGORIES]))
               for k in ("success", "wrong_door", "timeout")}
    # a seed is "conditioning" if it does NOT send every category to one door
    top_share = door_mat[:, 0]
    conditions = bool(top_share.max() - top_share.min() > 0.5)
    return {"per_category": rows, "overall": overall,
            "door_matrix": door_mat.tolist(), "conditions_on_belief": conditions}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="ppo_gru_forkwall_noaux_seed*")
    p.add_argument("--ckpt-root", type=Path, default=REPO / "outputs/ppo_checkpoints")
    p.add_argument("--out", type=Path, default=REPO / "paper/figures/forkwall_noaux_eval.png")
    p.add_argument("--eval-maps", type=int, default=16)
    p.add_argument("--eval-traj", type=int, default=24)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    ckpts = sorted(args.ckpt_root.glob(f"{args.glob}/final.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints match {args.glob}/final.pt under {args.ckpt_root}")

    results = {}
    for c in ckpts:
        name = c.parent.name
        seed = int(name.rsplit("seed", 1)[-1])
        print(f"evaluating {name} ...", flush=True)
        results[seed] = eval_one(c, device, args.eval_maps, args.eval_traj,
                                 args.eval_seed_start, args.max_steps)
        r = results[seed]
        print(f"  overall success {r['overall']['success']:.3f}  "
              f"wrong {r['overall']['wrong_door']:.3f}  timeout {r['overall']['timeout']:.3f}  "
              f"conditions={r['conditions_on_belief']}")

    seeds = sorted(results)
    print(f"\n{'seed':>5s} {'success':>8s} {'wrong':>8s} {'timeout':>8s}  " +
          " ".join(f"{c:>9s}" for c in CATEGORIES) + "   conditions")
    for s in seeds:
        r = results[s]
        print(f"{s:>5d} {r['overall']['success']:>8.3f} {r['overall']['wrong_door']:>8.3f} "
              f"{r['overall']['timeout']:>8.3f}  " +
              " ".join(f"{r['per_category'][c]['success']:>9.3f}" for c in CATEGORIES) +
              f"   {r['conditions_on_belief']}")
    su = [results[s]["overall"]["success"] for s in seeds]
    print(f"{'mean':>5s} {np.mean(su):>8.3f}")
    print(f"{'std':>5s} {np.std(su):>8.3f}")
    n_cond = sum(results[s]["conditions_on_belief"] for s in seeds)
    print(f"\nseeds conditioning on the belief: {n_cond}/{len(seeds)}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    w = 0.8 / len(seeds)
    for i, s in enumerate(seeds):
        vals = [results[s]["per_category"][c]["success"] for c in CATEGORIES]
        ax.bar(np.arange(3) + i * w - 0.4 + w / 2, vals, width=w,
               color=[CAT_COLOR[c] for c in CATEGORIES], alpha=0.5 + 0.1 * i,
               edgecolor="white", linewidth=0.6, label=f"seed {s}")
    ax.set_xticks(range(3)); ax.set_xticklabels(CATEGORIES); ax.set_ylim(0, 1.02)
    ax.axhline(2/3, color="#B4791E", ls="--", lw=1.2, label="constant-door (2/3)")
    ax.set_title("held-out success by category", fontsize=11); ax.set_ylabel("success")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.15, axis="y")

    ax = axes[1]
    mu = [np.mean([results[s]["overall"][k] for s in seeds])
          for k in ("success", "wrong_door", "timeout")]
    sd = [np.std([results[s]["overall"][k] for s in seeds])
          for k in ("success", "wrong_door", "timeout")]
    ax.bar(["success", "wrong door", "timeout"], mu, yerr=sd, capsize=6,
           color=["#2F8F63", "#B4791E", "#8c8c8c"], alpha=0.85,
           edgecolor="white", linewidth=0.8)
    for i, (m, s_) in enumerate(zip(mu, sd)):
        ax.text(i, m + s_ + 0.02, f"{m:.2f}±{s_:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_title(f"outcome, mean ± std over {len(seeds)} seeds", fontsize=11)
    ax.grid(alpha=0.15, axis="y")

    ax = axes[2]
    dm = np.mean([np.array(results[s]["door_matrix"]) for s in seeds], axis=0)
    im = ax.imshow(dm, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["top door", "bottom door", "no door"], fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(CATEGORIES, fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{dm[i,j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if dm[i, j] < 0.6 else "black", fontweight="bold")
    ax.set_title("mean map → door matrix", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"fork_wall no-aux — held-out eval ({args.eval_maps} maps × "
                 f"{args.eval_traj} rollouts/category, seeds ≥ {args.eval_seed_start})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved {args.out}")
    jp = args.out.with_suffix(".json")
    jp.write_text(json.dumps({"seeds": seeds, "results": results,
                              "n_conditioning": n_cond}, indent=2))
    print(f"saved {jp}")


if __name__ == "__main__":
    main()
