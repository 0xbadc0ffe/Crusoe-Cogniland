#!/usr/bin/env python3
"""Belief space for bridge_tunnel_commit: belief = map category (balanced/lakes/rocky).

The generalization test of the MemoryEnv neural-geometry pipeline: same
construction (linear category probe on GRU states -> Hellinger map -> PCA),
different environment. Here the belief is NOT written by a single cue tile —
the agent must accumulate noisy terrain evidence while navigating, so belief
formation should be gradual and the simplex interior genuinely occupied.

Probe is trained on half the map seeds and evaluated/plotted on held-out seeds
(tests the category concept, not map memorization).

  python scripts/bridge_tunnel/belief_space_bt.py \
      --checkpoint released_models/bridge_tunnel_commit/ppo_gru_commit.pt
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
from cogniland.bridge_tunnel import generate_commit_map  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel.mapgen import CATEGORIES  # noqa: E402
from eval_bridge_tunnel_commit_ppo import _load_policy  # noqa: E402

CAT_COL = {"balanced": "#8a8a8a", "lakes": "#3b6fb6", "rocky": "#b5651d"}


@torch.no_grad()
def collect(policy, rec, n_traj, view_size, max_steps, device):
    """Lockstep stochastic rollouts on one map; returns per-step arrays."""
    H, W = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=H, width=W,
                                  view_size=view_size, max_steps=max_steps)
            for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    active = np.ones(n_traj, dtype=bool)
    Hs, Prog, Commit, Tt, Alive = [], [], [], [], []
    for t in range(max_steps):
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        done = torch.zeros(1, n_traj, device=device)
        gru_out, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done, h)
        feat = gru_out.squeeze(0)
        logits, _ = policy._heads(feat)
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        scal = np.stack([o["scalars"] for o in obs])
        Hs.append(feat.cpu().numpy())
        Prog.append(np.abs(scal[:, 0]) + np.abs(scal[:, 1]))   # |compass| ~ dist-to-goal
        Commit.append(np.array([2 * o["scalars"][2] for o in obs]))  # active_obj
        Tt.append(np.full(n_traj, t))
        Alive.append(active.copy())
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, r, term, trunc, info = e.step(int(acts[i]))
            obs[i] = o
            if term or trunc:
                active[i] = False
        if not active.any():
            break
    return (np.concatenate([a[m] for a, m in zip(Hs, Alive)]),
            np.concatenate([a[m] for a, m in zip(Prog, Alive)]),
            np.concatenate([a[m] for a, m in zip(Commit, Alive)]),
            np.concatenate([a[m] for a, m in zip(Tt, Alive)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("released_models/bridge_tunnel_commit/ppo_gru_commit.pt"))
    ap.add_argument("--maps-per-cat", type=int, default=8)
    ap.add_argument("--traj-per-map", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed-start", type=int, default=10_000)
    ap.add_argument("--out", default="outputs/report_geometry/bt_belief_hellinger.png")
    ap.add_argument("--categories", nargs="+", default=None,
                    help="subset of map categories (default: all three)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    policy, cargs, view_size, env_size, env_width = _load_policy(a.checkpoint, device)
    gh = cargs.get("goal_half", 1)
    cats = a.categories or list(CATEGORIES)

    X, CAT, PROG, COMMIT, TT, MAP = [], [], [], [], [], []
    for ci, cat in enumerate(cats):
        for j in range(a.maps_per_cat):
            rec = generate_commit_map(size=env_size, width=env_width,
                                      seed=a.seed_start + j, category=cat,
                                      tree_frac=cargs.get("tree_frac", 0.03),
                                      goal_half=(gh if (gh is not None and gh >= 0) else None))
            Hs, Pr, Cm, Tt = collect(policy, rec, a.traj_per_map, view_size,
                                     a.max_steps, device)
            X.append(Hs); PROG.append(Pr); COMMIT.append(Cm); TT.append(Tt)
            CAT.append(np.full(len(Hs), ci)); MAP.append(np.full(len(Hs), j))
            print(f"[collect] {cat} map {j}: {len(Hs)} steps", flush=True)
    X, CAT, PROG, COMMIT, TT, MAP = map(np.concatenate, (X, CAT, PROG, COMMIT, TT, MAP))

    # probe: train on even map seeds, evaluate on odd (held-out maps)
    from sklearn.linear_model import LogisticRegression
    tr = MAP % 2 == 0
    te = ~tr
    sub = np.random.default_rng(0).permutation(tr.sum())[:40000]
    clf = LogisticRegression(max_iter=3000).fit(X[tr][sub], CAT[tr][sub])
    acc = float(clf.score(X[te][::5], CAT[te][::5]))
    print(f"[probe] held-out-map accuracy: {acc:.3f} (chance 0.33)")
    P = clf.predict_proba(X[te])

    # Hellinger + PCA on held-out states
    rng = np.random.default_rng(2)
    idx = rng.permutation(len(P))[:3000]
    Q = np.sqrt(P[idx]); Qc = Q - Q.mean(0)
    _, S_, Vt = np.linalg.svd(Qc, full_matrices=False)
    co = Qc @ Vt[:2].T
    ev = (S_ ** 2 / (S_ ** 2).sum())
    print("[hellinger] expl var:", np.round(ev, 3))

    fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.4))
    for ci, cat in enumerate(cats):
        m = CAT[te][idx] == ci
        axs[0].scatter(co[m, 0], co[m, 1], s=7, alpha=.45, lw=0,
                       c=CAT_COL[cat], label=cat)
    axs[0].legend(fontsize=8, markerscale=2)
    evs = "/".join(f"{v:.0%}" for v in ev[:3])
    axs[0].set_title(f"bridge_tunnel belief space — Hellinger + PCA\n"
                     f"(held-out maps; probe acc {acc:.2f}; EV {evs})")
    tnorm = np.clip(TT[te][idx] / 150.0, 0, 1)
    sc2 = axs[1].scatter(co[:, 0], co[:, 1], s=7, alpha=.5, lw=0,
                         c=tnorm, cmap="viridis")
    plt.colorbar(sc2, ax=axs[1], label="episode time (t/150, clipped)")
    axs[1].set_title("same embedding, colored by time")
    # belief formation: P(true category) vs time, per category
    for ci, cat in enumerate(cats):
        m = CAT[te] == ci
        tt = TT[te][m]
        pt = P[m][:, ci]
        bins = np.arange(0, 200, 10)
        mb = [pt[(tt >= b) & (tt < b + 10)].mean() for b in bins]
        axs[2].plot(bins + 5, mb, "-o", ms=3.5, lw=2, c=CAT_COL[cat], label=cat)
    axs[2].axhline(1 / len(cats), ls=":", c="#999")
    axs[2].set_ylim(0, 1.02)
    axs[2].set_xlabel("timestep"); axs[2].set_ylabel("P(true category | h)")
    axs[2].set_title("belief formation: gradual evidence accumulation")
    axs[2].legend(fontsize=8)
    for ax_ in axs[:2]:
        ax_.set_xlabel("Hellinger PC1"); ax_.set_ylabel("Hellinger PC2")
        ax_.set_aspect("equal")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=145)
    print(f"[belief_space_bt] wrote {a.out}")


if __name__ == "__main__":
    main()
