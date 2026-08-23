#!/usr/bin/env python
"""Roll out the best fork_wall DreamerV3 agent with its STOCHASTIC policy and draw,
per map, the full terrain with the trajectories overlaid as transparent poly-lines
whose colour encodes the action taken at each step. With --n-rollouts > 1 several
independent samples are overlaid at low alpha, so the coloured density shows the
policy's path distribution and the door-decision consistency under stochasticity.

  python scripts/bridge_tunnel/viz_rollout_actions.py \
      --checkpoint external/r2dreamer/runs/fw_sw_25M_bl64_h15/latest.pt \
      --model-size size25M --maps data/bridge_tunnel/forkwall6k/test.pkl \
      --per-cat 2 --n-rollouts 32 --seed 1 --out paper/figures/cogniland/rollout_actions.png
"""
from __future__ import annotations
import argparse, pathlib, pickle, sys
from collections import defaultdict, Counter
import numpy as np, torch, gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src")); sys.path.insert(0, str(_REPO / "external/r2dreamer"))
sys.path.insert(0, str(_REPO / "scripts/bridge_tunnel"))
from hydra import compose, initialize_config_dir
import dreamer_belief_report_r2d as R
from dreamer import Dreamer
from cogniland.bridge_tunnel.env import BridgeTunnelEnv
from cogniland.bridge_tunnel.tiles import TILE_COLORS
from tensordict import TensorDict

CATS = ["balanced", "lakes", "rocky"]
# action id -> (label, colour). 0/1/2/3 = up/down/left/right, 4/5 = build.
ACTIONS = [
    ("up",            "#4C78A8"),
    ("down",          "#F58518"),
    ("left",          "#54A24B"),
    ("right",         "#E45756"),
    ("build_raft",    "#B279A2"),
    ("build_harness", "#9D755D"),
]


def load(checkpoint, device, model_size):
    cfg_dir = str((_REPO / "external/r2dreamer/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="configs", overrides=[
            "env=bridge_tunnel_forkwall", "env.task=bridgetunnel_forkwall",
            f"model={model_size}", "model.rep_loss=dreamer", f"device={device}",
            "model.compile=False"])
    vd = R.VIEW * R.VIEW * R.NUM_TILES + R.N_SCALARS
    obs = gym.spaces.Dict({"vector": gym.spaces.Box(-np.inf, np.inf, (vd,), np.float32),
        "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "is_first": gym.spaces.Box(0, 1, (), bool), "is_last": gym.spaces.Box(0, 1, (), bool),
        "is_terminal": gym.spaces.Box(0, 1, (), bool)})
    class _OH(gym.spaces.Box): discrete = True
    ag = Dreamer(cfg.model, obs, _OH(0, 1, (6,), np.float32)).to(device)
    ag.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["agent_state_dict"], strict=False)
    ag.eval(); return ag


@torch.no_grad()
def rollout(agent, device, rec):
    """One STOCHASTIC-policy episode on a fixed map."""
    env = BridgeTunnelEnv(**{**R.ENV_KW, "categories": (rec.category,)})
    env._fixed_record = rec; raw, info = env.reset()
    terrain = env._terrain.copy()
    st = agent.get_initial_state(1); first = True
    acts = []
    for t in range(env.max_steps):
        vec = R.flatten_obs(raw)
        trans = TensorDict({"vector": torch.as_tensor(vec, device=device, dtype=torch.float32)[None],
                            "is_first": torch.tensor([first], device=device)}, batch_size=(1,))
        a, st = agent.act(trans, st, eval=False)          # <-- STOCHASTIC (rsample)
        first = False
        ai = int(a.argmax(-1)); acts.append(ai)
        raw, r, term, trunc, info = env.step(ai)
        if term or trunc: break
    pos = np.array(env._traj, dtype=float)                # (T+1, 2) = (row, col)
    reached = bool(info.get("reached_any_target", False))
    door = ("top" if env._traj[-1][0] < env.height / 2 else "bottom") if reached else "timeout"
    return dict(pos=pos, acts=np.array(acts, dtype=int),
                success=bool(info.get("reached_target", False)), door=door)


def draw_map(ax, terrain, cat, eps, alpha):
    """Overlay all rollouts `eps` (same map) as action-coloured transparent lines."""
    H, W = terrain.shape
    ax.imshow(TILE_COLORS[terrain], interpolation="nearest", zorder=0)
    all_segs, all_cols, end_xy, spawn = [], [], [], None
    for ep in eps:
        pos, acts = ep["pos"], ep["acts"]
        xy = np.column_stack([pos[:, 1], pos[:, 0]])      # (x=col, y=row)
        spawn = xy[0]; end_xy.append(xy[-1])
        for t in range(len(acts)):
            p0, p1 = xy[t], xy[t + 1]
            if np.allclose(p0, p1):                       # blocked / build -> no move
                continue
            all_segs.append([p0, p1])
            all_cols.append(ACTIONS[acts[t]][1] if acts[t] < len(ACTIONS) else "#000000")
    if all_segs:
        ax.add_collection(LineCollection(all_segs, colors=all_cols, linewidths=2.6,
                                         alpha=alpha, capstyle="round", zorder=3))
    end_xy = np.array(end_xy)
    ax.scatter(end_xy[:, 0], end_xy[:, 1], s=22, facecolors="black", edgecolors="none",
               alpha=min(1.0, 6 * alpha), zorder=4)        # where episodes ended (door)
    ax.scatter([spawn[0]], [spawn[1]], s=110, facecolors="none", edgecolors="white",
               linewidths=2.4, zorder=5)                   # spawn
    want = {"lakes": "bottom", "rocky": "top", "balanced": "either"}[cat]
    dc = Counter(e["door"] for e in eps)
    dist = "  ".join(f"{k}:{v}" for k, v in dc.most_common())
    succ = sum(e["success"] for e in eps)
    ax.set_title(f"{cat} (wants {want}) — {succ}/{len(eps)} correct   doors[{dist}]",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="external/r2dreamer/runs/fw_sw_25M_bl64_h15/latest.pt")
    ap.add_argument("--model-size", default="size25M")
    ap.add_argument("--maps", default="data/bridge_tunnel/forkwall6k/test.pkl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-cat", type=int, default=2)
    ap.add_argument("--n-rollouts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=-1.0, help="<0 -> auto by n-rollouts")
    ap.add_argument("--out", default="paper/figures/cogniland/rollout_actions.png")
    args = ap.parse_args()
    alpha = args.alpha if args.alpha > 0 else float(np.clip(3.0 / args.n_rollouts, 0.06, 0.6))

    torch.manual_seed(args.seed)
    agent = load(args.checkpoint, args.device, args.model_size)
    recs = pickle.load(open(args.maps, "rb"))
    by = defaultdict(list)
    for r in recs: by[r.category].append(r)
    rng = np.random.default_rng(args.seed)

    maps = []
    for c in CATS:
        idx = rng.choice(len(by[c]), size=min(args.per_cat, len(by[c])), replace=False)
        for i in idx: maps.append(by[c][i])

    ncol, nrow = args.per_cat, len(CATS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 7.2, nrow * 3.7), squeeze=False)
    for k, rec in enumerate(maps):
        eps = [rollout(agent, args.device, rec) for _ in range(args.n_rollouts)]
        terrain = BridgeTunnelEnv(**{**R.ENV_KW, "categories": (rec.category,)})
        terrain._fixed_record = rec; terrain.reset(); terr = terrain._terrain.copy()
        draw_map(axes[k // ncol][k % ncol], terr, rec.category, eps, alpha)

    handles = [Line2D([0], [0], color=col, lw=4, label=lab) for lab, col in ACTIONS[:4]]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                       markeredgecolor="white", markersize=10, lw=0, label="spawn"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                       markersize=9, lw=0, label="episode end")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    ck = pathlib.Path(args.checkpoint).parts[-2]
    fig.suptitle(f"Stochastic-policy rollouts — {ck}  ({args.n_rollouts} samples/map, "
                 f"line colour = action, alpha={alpha:.2f})",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    tot = args.n_rollouts * len(maps)
    print("wrote", out, f" maps={len(maps)} rollouts/map={args.n_rollouts} total={tot}")


if __name__ == "__main__":
    main()
