#!/usr/bin/env python3
"""Before/after skill-mean steering on fork_wall: stochastic trajectory grids
(baseline vs steered) with success rate, plus the mean belief-over-time — the
variable we are NOT steering (we steer the build/mine behavior; the archetype
belief is the side effect).

Uses the build->mine behavior direction saved by
eval_bridge_tunnel_forkwall_steered.py.

  python scripts/bridge_tunnel/forkwall_steer_traj_belief.py \
      --checkpoint released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt \
      --direction outputs/bridge_tunnel_forkwall/skillmean_balanced_steer_dir_skill-mean.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel import generate_commit_map, tiles as T  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered  # noqa: E402
from eval_bridge_tunnel_commit_ppo import _draw_commit_path  # noqa: E402
from eval_bridge_tunnel_commit_ppo_steered import (  # noqa: E402
    make_steer_fn, cat_alpha, BELIEF2I)

CATS = ["rocky", "lakes"]
CAT_COL = {"lakes": "#1f77b4", "rocky": "#8c564b"}


DOOR_COL = {"top": "#17becf", "bottom": "#ff7f0e", "none": "#888888"}


def draw_traj(ax, rec, out, title):
    ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
    # colour each stochastic path by the DOOR it reached (the decision that moves)
    for tr, door in zip(out["trajs"], out["doors"]):
        arr = np.asarray(tr)
        ax.plot(arr[:, 1], arr[:, 0], color=DOOR_COL.get(door, "#888"),
                lw=1.1, alpha=0.55, zorder=4)
    top_ok = rec.correct_target in ("top", "either")
    bot_ok = rec.correct_target in ("bottom", "either")
    for cells, ok in ((rec.top_goal_cells, top_ok), (rec.bottom_goal_cells, bot_ok)):
        if cells:
            ys = [r for r, c in cells]; xs = [c for r, c in cells]
            ax.scatter(xs, ys, c=("lime" if ok else "red"), s=34, marker="s",
                       edgecolors="k", zorder=6)
    ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=26, marker="o",
               edgecolors="k", zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt"))
    ap.add_argument("--direction", type=Path,
                    default=Path("outputs/bridge_tunnel_forkwall/skillmean_balanced_steer_dir_skill-mean.npy"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--traj-n", type=int, default=28, help="stochastic rollouts on the shown map")
    ap.add_argument("--belief-maps", type=int, default=8)
    ap.add_argument("--belief-traj", type=int, default=16)
    ap.add_argument("--map-seed", type=int, default=10_000)
    ap.add_argument("--belief-seed-start", type=int, default=11_000)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--out", default="outputs/bridge_tunnel_forkwall/steer_traj_belief.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    policy, cargs, view_size, env_size, env_width = _load_policy(a.checkpoint, device)
    direction = torch.from_numpy(np.load(a.direction).astype(np.float32)).to(device)
    commit = False if cargs.get("no_commit", False) else None
    ph = cargs.get("passage_half", 1); wm = cargs.get("wall_margin", 1)
    gh = cargs.get("goal_half", 0); gh = gh if (gh is not None and gh >= 0) else None

    def mk(seed, cat):
        return generate_commit_map(size=env_size, width=env_width, seed=seed, category=cat,
                                   tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                   fork_wall=True, passage_half=ph, wall_margin=wm)

    def steer_for(cat):
        return make_steer_fn("skill-mean", cat, policy, direction,
                             cat_alpha((a.alpha, a.alpha), cat), 0, 10**9,
                             10, 0.01, 0.0, 0.05, 0.5, 0.75)

    fig = plt.figure(figsize=(18, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.3, 1.05])

    for ri, cat in enumerate(CATS):
        # ── trajectory grid on one representative held-out map ────────────
        rec = mk(a.map_seed, cat)
        base = batched_rollout_steered(policy, rec, a.traj_n, view_size, a.max_steps,
                                       device, commit=commit)
        steer = batched_rollout_steered(policy, rec, a.traj_n, view_size, a.max_steps,
                                        device, commit=commit, steer_fn=steer_for(cat))
        for cj, (out, lab) in enumerate([(base, "baseline"), (steer, "steered")]):
            ax = fig.add_subplot(gs[ri, cj])
            succ = out["success"].mean()
            wrong = (out["reached_any"] & ~out["success"]).mean()
            top = np.mean([d == "top" for d in out["doors"]])
            bot = np.mean([d == "bottom" for d in out["doors"]])
            draw_traj(ax, rec, out,
                      f"{cat} · {lab}   succ {succ:.0%} / wrong {wrong:.0%}\n"
                      f"door: top {top:.0%} · bottom {bot:.0%}   |   "
                      f"crossing: mine {out['n_mines'].mean():.1f} / "
                      f"build {out['n_builds'].mean():.1f} per ep")

        # ── belief over time (aggregated over many maps) ──────────────────
        bidx = BELIEF2I[cat]
        agg = {"baseline": [], "steered": []}
        for j in range(a.belief_maps):
            r2 = mk(a.belief_seed_start + j, cat)
            b = batched_rollout_steered(policy, r2, a.belief_traj, view_size,
                                        a.max_steps, device, commit=commit)
            s = batched_rollout_steered(policy, r2, a.belief_traj, view_size,
                                        a.max_steps, device, commit=commit,
                                        steer_fn=steer_for(cat))
            agg["baseline"].append(b["belief_probs"][:, :a.horizon, bidx])
            agg["steered"].append(s["belief_probs"][:, :a.horizon, bidx])
        axb = fig.add_subplot(gs[ri, 2])
        tt = np.arange(a.horizon)
        for lab, ls in [("baseline", "-"), ("steered", "--")]:
            arr = np.concatenate(agg[lab])                       # (N, horizon)
            mean = np.nanmean(arr, axis=0); sd = np.nanstd(arr, axis=0)
            axb.plot(tt, mean, ls, color=CAT_COL[cat], lw=2, label=lab)
            axb.fill_between(tt, mean - sd, mean + sd, color=CAT_COL[cat], alpha=0.12)
        axb.axhline(1 / 3, ls=":", c="#999", lw=0.8)
        axb.set_ylim(-0.02, 1.02); axb.set_xlim(0, a.horizon)
        axb.set_xlabel("timestep"); axb.set_ylabel(f"P({cat} | h)")
        axb.set_title(f"{cat}: belief we are NOT steering", fontsize=9)
        axb.legend(fontsize=8, loc="center right")

    fig.suptitle("fork_wall — steering the build/mine BEHAVIOR: the crossing barely changes "
                 "(same mine/build counts), only the DOOR flips\n"
                 "paths coloured by door reached (cyan=top, orange=bottom); lime/red squares = "
                 "correct/decoy door; right column = the un-steered belief drifting",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
