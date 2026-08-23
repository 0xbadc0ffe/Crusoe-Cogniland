#!/usr/bin/env python3
"""Figure 6: 24 stochastic rollouts per (agent, map), drawn as translucent lines.

Overlap is the message: a single run is faint but legible, and where many runs
agree the ink accumulates into a bright highway. Lines are continuous and
sub-cell jittered so that coincident paths still visibly stack instead of
collapsing into one hard stroke.

Run once per agent (each needs its own interpreter, see paper_rollouts.py):
  PYTHONPATH=src python scripts/figures/paper_traj_density.py --agent ppo
  PYTHONPATH=src:r2dreamer_model  ... --agent dreamer
  (from STORM_model/) PYTHONPATH=.:..:../src python ../scripts/figures/paper_traj_density.py --agent storm
Then, with all three json files present:
  python scripts/figures/paper_traj_density.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from cogniland.bridge_tunnel.tiles import TILE_COLORS  # noqa: E402

AGENTS = ["ppo", "dreamer", "storm"]
LABEL = {"ppo": "PPO + GRU", "dreamer": "DreamerV3", "storm": "STORM"}
COL = {"ppo": "#f59e0b", "dreamer": "#60a5fa", "storm": "#34d399"}


def collect(agent, map_ids, n, args):
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS, make_dreamer, make_ppo, make_storm

    if agent == "ppo":
        act, reset = make_ppo(args.ppo_ckpt)
    elif agent == "storm":
        act, reset = make_storm(args.storm_bundle, args.storm_step)
    else:
        act, reset = make_dreamer(args.dreamer_ckpt, args.device, args.dreamer_size, sampled=True)

    with open(args.maps, "rb") as f:
        pool = pickle.load(f)

    out = {}
    for mid in map_ids:
        rec = pool[mid]
        runs = []
        for k in range(n):
            env = BridgeTunnelEnv(seed=k, map_record=rec, **FORKWALL_KWARGS)
            obs, _ = env.reset()
            reset()
            traj = [env._pos]
            for t in range(FORKWALL_KWARGS["max_steps"]):
                obs, r, term, trunc, _ = env.step(act(obs, False))
                traj.append(env._pos)
                if term or trunc:
                    break
            runs.append(dict(traj=[(int(a), int(b)) for a, b in traj],
                             success=bool(env._pos in (env._correct_cells or set())),
                             steps=len(traj) - 1))
        out[str(mid)] = runs
        ok = sum(r["success"] for r in runs)
        print(f"  map {mid:5d} {rec.category:9s} {ok}/{n} success")
    return out


def plot(dirpath, map_ids, maps_path):
    with open(maps_path, "rb") as f:
        pool = pickle.load(f)
    data = {}
    for a in AGENTS:
        f = dirpath / f"traj_density_{a}.json"
        if f.exists():
            data[a] = json.loads(f.read_text())
    if not data:
        print("no traj_density_*.json yet")
        return

    rc = {"figure.dpi": 140, "savefig.dpi": 140, "font.size": 8.5, "axes.titlesize": 9}
    rng = np.random.default_rng(0)
    with plt.rc_context(rc):
        nrow, ncol = len(map_ids), len(data)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.45 * ncol, 2.25 * nrow))
        axes = np.asarray(axes).reshape(nrow, ncol)
        for r, mid in enumerate(map_ids):
            rec = pool[mid]
            for c, agent in enumerate([a for a in AGENTS if a in data]):
                ax = axes[r, c]
                ax.imshow(TILE_COLORS[rec.terrain], interpolation="nearest")
                H, W = rec.terrain.shape
                # a run that ends inside the door cell can push the autoscale one
                # column past the map and letterbox the panel
                ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5)
                for cells, name in ((rec.top_goal_cells, "top"),
                                    (rec.bottom_goal_cells, "bottom")):
                    good = rec.correct_target in ("either", name)
                    for (rr, cc) in cells:
                        ax.add_patch(Rectangle((cc - .5, rr - .5), 1, 1, fill=False,
                                               edgecolor="#22c55e" if good else "#ef4444",
                                               lw=1.6, zorder=8))
                runs = data[agent].get(str(mid), [])
                ok = sum(x["success"] for x in runs)
                for run in runs:
                    t = np.asarray(run["traj"], dtype=float)
                    # sub-cell jitter: identical paths still stack visibly
                    t = t + rng.normal(0, .16, t.shape)
                    # one line reads about as strongly as eight did at .16
                    # (1-(1-.16)**8 = .75); thinner strokes keep bundles legible
                    ax.plot(t[:, 1], t[:, 0], color=COL[agent], lw=.85,
                            alpha=.72, solid_capstyle="round", zorder=6)
                ax.plot([], [], color=COL[agent], lw=2, label="one episode")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{LABEL[agent]} — {ok}/{len(runs)} reach the right door",
                             loc="left", fontsize=8.5)
                if c == 0:
                    ax.set_ylabel(f"map {mid}\n({rec.category})", fontsize=8)
        fig.suptitle("24 stochastic episodes per panel — where runs agree the ink "
                     "stacks into a highway; single deviations stay faint but visible",
                     y=1.005)
        fig.tight_layout()
        fig.savefig(dirpath / "fig_trajectories.png", bbox_inches="tight")
        plt.close(fig)
    print("wrote", dirpath / "fig_trajectories.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=AGENTS)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--map-ids", default="0,5,7")
    p.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    p.add_argument("--ppo-ckpt", default=str(REPO / "final_models/ppo/ppo_plain.pt"))
    p.add_argument("--storm-bundle", default=str(REPO / "final_models/storm"))
    p.add_argument("--storm-step", type=int, default=624489)
    p.add_argument("--dreamer-ckpt", default=str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"))
    p.add_argument("--dreamer-size", default="size25M")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in args.map_ids.split(",")]
    if not args.plot_only:
        if not args.agent:
            p.error("--agent is required unless --plot-only")
        res = collect(args.agent, ids, args.n, args)
        (out / f"traj_density_{args.agent}.json").write_text(json.dumps(res))
    plot(out, ids, args.maps)


if __name__ == "__main__":
    main()
