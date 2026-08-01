#!/usr/bin/env python3
"""Chapter figures 7 & 9 for a fork_wall agent: what the recurrent state
encodes, and how the belief forms along a trajectory.

(7A) PCA of GRU states coloured by the NEXT action
(7B) mean activation of the most action-selective units, by next action
(7C) linear probes on the GRU state vs majority/chance baselines:
       map category (3) | next action (6) | terrain ahead (4) |
       final door (2)   | x-position (8 bins)
(9)  single stochastic rollouts per category, with the probe-decoded
     P(category | h_t) underneath — belief accumulating as evidence arrives

All probes use a GROUPED split over map id (train and test maps disjoint), so
nothing can be won by memorising a map.

    python scripts/figures/forkwall_hidden_state_analysis.py \
        --checkpoint outputs/ppo_checkpoints/ppo_gru_forkwall_noaux_seed1/final.pt
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
sys.path.insert(0, str(REPO / "scripts" / "mechinterp"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy, _door_of  # noqa: E402
from train_belief_probe import load_belief_probe  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ACTIONS = ["up", "down", "left", "right", "build", "mine"]
ACT_COLOR = ["#7fb3d5", "#5499c7", "#f5b041", "#e67e22", "#8e44ad", "#c0392b"]
CAT_COLOR = {"balanced": "#5C6B57", "lakes": "#1E6FA6", "rocky": "#A3572A"}
FACE_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@torch.no_grad()
def collect(policy, rec, n_traj, view_size, max_steps, device, commit):
    """Per-step: gru_h, next action, terrain ahead, x-position; per-episode: door."""
    H, W = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=H, width=W, view_size=view_size,
                                  max_steps=max_steps, commit=commit) for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    done = torch.zeros(n_traj, device=device)
    active = np.ones(n_traj, dtype=bool)
    rows = [[] for _ in range(n_traj)]
    final_pos = [None] * n_traj
    beliefs = [[] for _ in range(n_traj)]
    paths = [[tuple(e._pos)] for e in envs]

    for _ in range(max_steps):
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        _, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        x = h.squeeze(0)
        logits, _ = policy._heads(x)
        bel = torch.softmax(policy.belief(x), dim=-1).cpu().numpy()
        xnp = x.cpu().numpy()
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            a = int(acts[i])
            r0, c0 = e._pos
            fr, fc = FACE_DELTA[a if a < 4 else e._facing]
            tr, tc = r0 + fr, c0 + fc
            ahead = int(rec.terrain[tr, tc]) if (0 <= tr < H and 0 <= tc < W) else T.OOB
            ahead_cls = {T.WATER: 1, T.ROCK: 2, T.TREE: 3}.get(ahead, 0)
            rows[i].append((xnp[i].copy(), a, ahead_cls, c0 / max(W - 1, 1)))
            beliefs[i].append(bel[i].copy())
            o, _, term, trunc, info = e.step(a)
            obs[i] = o
            paths[i].append(tuple(e._pos))
            if term:
                final_pos[i] = e._pos; active[i] = False
            elif trunc:
                active[i] = False
        done = torch.zeros(n_traj, device=device)
        if not active.any():
            break
    doors = [_door_of(rec, p) for p in final_pos]
    return rows, [d if d in ("top", "bottom") else "none" for d in doors], beliefs, paths


def probe(X, y, groups, seed=0):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups))
    pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=3000))])
    pipe.fit(X[tr], y[tr])
    acc = float(balanced_accuracy_score(y[te], pipe.predict(X[te])))
    vals, cnt = np.unique(y[te], return_counts=True)
    return acc, float(cnt.max() / cnt.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=REPO / "outputs/ppo_checkpoints/ppo_gru_forkwall_noaux_seed1/final.pt")
    p.add_argument("--probe", type=Path, default=None)
    p.add_argument("--maps", type=int, default=14)
    p.add_argument("--traj", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=80_000)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out-prefix", type=Path, default=REPO / "paper/figures/forkwall_noaux")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    bel_lin, pmeta = load_belief_probe(
        args.probe or args.checkpoint.parent / "belief_probe.pt", device)
    policy.belief = bel_lin
    commit = False if cargs.get("no_commit", False) else None
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    torch.manual_seed(0)

    def make_map(seed, cat):
        return generate_commit_map(size=env_size, width=env_width, seed=seed, category=cat,
                                   tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                   fork_wall=True, passage_half=cargs.get("passage_half", 1),
                                   wall_margin=cargs.get("wall_margin", 1))

    X, y_cat, y_act, y_ahead, y_x, y_door, groups = [], [], [], [], [], [], []
    demo = {}
    for ci, cat in enumerate(CATEGORIES):
        for j in range(args.maps):
            seed = args.seed_start + ci * 500 + j
            rec = make_map(seed, cat)
            rows, doors, beliefs, paths = collect(policy, rec, args.traj, view_size,
                                                  args.max_steps, device, commit)
            if j == 0:
                k = int(np.argmax([len(r) for r in rows]))
                demo[cat] = (rec, paths[k], np.array(beliefs[k]), doors[k])
            for i, ep in enumerate(rows):
                for (hh, a, ah, xp) in ep:
                    X.append(hh); y_cat.append(ci); y_act.append(a); y_ahead.append(ah)
                    y_x.append(min(int(xp * 8), 7))
                    y_door.append(0 if doors[i] == "top" else 1)
                    groups.append(f"{cat}_{seed}")
        print(f"  collected {cat}", flush=True)

    X = np.asarray(X, dtype=np.float64)
    g = np.asarray(groups)
    print(f"dataset: {X.shape[0]} steps x {X.shape[1]} dims")

    targets = [("map category (3)", np.asarray(y_cat)), ("next action (6)", np.asarray(y_act)),
               ("terrain ahead (4)", np.asarray(y_ahead)), ("final door (2)", np.asarray(y_door)),
               ("x-position (8)", np.asarray(y_x))]
    accs, bases = [], []
    for name, yy in targets:
        a, b = probe(X, yy, g)
        accs.append(a); bases.append(b)
        print(f"  probe {name:20s} balanced-acc {a:.3f}   majority {b:.3f}")

    # ---------- figure 7 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    sub = np.random.default_rng(0).choice(len(Z), size=min(4000, len(Z)), replace=False)
    ya = np.asarray(y_act)
    for a in range(6):
        m = sub[ya[sub] == a]
        axes[0].scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.45, color=ACT_COLOR[a],
                        label=ACTIONS[a], linewidths=0)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].set_title("(A) GRU state PCA, coloured by NEXT action", fontsize=10)
    axes[0].legend(fontsize=7, markerscale=2.2, loc="best")

    means = np.stack([X[ya == a].mean(0) for a in range(6)])
    sel = np.argsort(means.std(0))[::-1][:40]
    mm_ = means[:, sel]
    mm_ = (mm_ - mm_.mean(0)) / (mm_.std(0) + 1e-8)
    im = axes[1].imshow(mm_, aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)
    axes[1].set_yticks(range(6)); axes[1].set_yticklabels(ACTIONS, fontsize=9)
    axes[1].set_xlabel("GRU unit (top-40 action-selective)")
    axes[1].set_title("(B) mean activation by next action", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)

    xs = np.arange(len(targets)); w = 0.38
    axes[2].bar(xs - w/2, accs, w, color="#2F8F63", label="probe (held-out maps)")
    axes[2].bar(xs + w/2, bases, w, color="#bbbbbb", label="majority / chance")
    for i, a in enumerate(accs):
        axes[2].text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([t[0] for t in targets], rotation=20, ha="right", fontsize=8)
    axes[2].set_ylim(0, 1.08); axes[2].set_ylabel("balanced accuracy")
    axes[2].set_title("(C) linear probes on the GRU state", fontsize=10)
    axes[2].legend(fontsize=8)

    fig.suptitle(f"What the recurrent state encodes — {args.checkpoint.parent.name} "
                 f"(no aux belief loss; belief read by probe, "
                 f"balanced acc {pmeta['balanced_accuracy']:.2f})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    o1 = Path(str(args.out_prefix) + "_hidden_state.png")
    o1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o1, dpi=150); print(f"saved {o1}")

    # ---------- figure 9 ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.4),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    for k, cat in enumerate(CATEGORIES):
        rec, path, bel, door = demo[cat]
        ax = axes[0, k]
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        pth = np.array(path, dtype=float)
        ax.scatter(pth[:, 1], pth[:, 0], c=np.arange(len(pth)), cmap="cividis",
                   s=5, linewidths=0)
        for cells, ok in ((rec.top_goal_cells, rec.correct_target in ("top", "either")),
                          (rec.bottom_goal_cells, rec.correct_target in ("bottom", "either"))):
            if cells:
                ys = [r for r, _ in cells]; xsv = [c for _, c in cells]
                ax.scatter(xsv, ys, c="lime" if ok else "red", s=26, marker="s",
                           edgecolors="k", zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{cat}: door={door} (dark→light = time)", fontsize=9)

        ax = axes[1, k]
        for ci2, c2 in enumerate(CATEGORIES):
            ax.plot(bel[:, ci2], color=CAT_COLOR[c2], lw=1.5, label=c2)
        ax.axhline(1/3, color="gray", lw=0.7, ls=":")
        ax.set_ylim(0, 1.02); ax.set_xlabel("timestep")
        if k == 0:
            ax.set_ylabel("probe-decoded P(category | $h_t$)"); ax.legend(fontsize=8)
    fig.suptitle("Trajectories and belief formation — belief accumulates as terrain "
                 "evidence enters the view", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    o2 = Path(str(args.out_prefix) + "_belief_formation.png")
    fig.savefig(o2, dpi=150); print(f"saved {o2}")

    jp = Path(str(args.out_prefix) + "_hidden_state.json")
    jp.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "n_steps": int(X.shape[0]),
        "probe_balanced_accuracy_belief_head": pmeta["balanced_accuracy"],
        "probes": {t[0]: {"acc": a, "majority": b}
                   for t, a, b in zip(targets, accs, bases)}}, indent=2))
    print(f"saved {jp}")


if __name__ == "__main__":
    main()
