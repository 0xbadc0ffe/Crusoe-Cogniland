#!/usr/bin/env python3
"""Psychometric curve on the fork_wall env: clamp the GRU belief at fractions
along the lakes<->rocky class-mean axis (Hellinger-arc pullback) and measure
door choice. Does P(top door) vary smoothly with the implanted belief?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel.mapgen import generate_commit_map  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy, _door_of  # noqa: E402

CKPT = Path("released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt")
CATS = ["lakes", "rocky"]          # lakes -> bottom door, rocky -> top door
TBIN, NBINS = 10, 13


@torch.no_grad()
def rollout(policy, rec, n, view_size, max_steps, device, cargs,
            U=None, RHO=None):
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=rec.terrain.shape[0],
                                  width=rec.terrain.shape[1], view_size=view_size,
                                  max_steps=max_steps,
                                  fork_wall=cargs.get("fork_wall", True),
                                  commit=cargs.get("commit", False))
            for _ in range(n)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n, policy.gru_hidden, device=device)
    active = np.ones(n, bool)
    feats, alive_tr = [], []
    doors = ["none"] * n
    for t in range(max_steps):
        if U is not None:
            b = min(t // TBIN, NBINS - 1)
            u = torch.from_numpy(U[b]).to(device, torch.float32)
            win = torch.from_numpy(active).to(device)
            delta = (float(RHO[b]) - (h[0] @ u)).unsqueeze(-1) * u.unsqueeze(0)
            h[0] = torch.where(win.unsqueeze(-1), h[0] + delta, h[0])
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        g, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                   torch.zeros(1, n, device=device), h)
        feats.append(g.squeeze(0).cpu().numpy()); alive_tr.append(active.copy())
        logits, _ = policy._heads(g.squeeze(0))
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, r, term, trunc, info = e.step(int(acts[i]))
            obs[i] = o
            if term or trunc:
                active[i] = False
                doors[i] = _door_of(rec, e._pos)
        if not active.any():
            break
    return np.stack(feats), np.stack(alive_tr), doors


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cargs, view_size, env_size, env_width = _load_policy(CKPT, device)
    gh = cargs.get("goal_half", 1)

    def mk(cat, seed):
        return generate_commit_map(size=env_size, width=env_width, seed=seed,
                                   category=cat, tree_frac=cargs.get("tree_frac", 0.03),
                                   goal_half=gh, fork_wall=True)

    # train data (even seeds): probe + t-binned class means
    from sklearn.linear_model import LogisticRegression
    X, CAT, TT = [], [], []
    for ci, cat in enumerate(CATS):
        for j in range(10):
            F, A, _ = rollout(policy, mk(cat, 30000 + 2 * j), 6, view_size, 300,
                              device, cargs)
            T_, N_ = A.shape
            X.append(F[A]); CAT.append(np.full(A.sum(), ci))
            TT.append(np.broadcast_to(np.arange(T_)[:, None], A.shape)[A])
    X, CAT, TT = map(np.concatenate, (X, CAT, TT))
    clf = LogisticRegression(max_iter=3000).fit(X[::4], CAT[::4])
    MU = np.zeros((2, NBINS, X.shape[1]), np.float32)
    for b in range(NBINS):
        mb = np.minimum(TT // TBIN, NBINS - 1) == b
        for ci in range(2):
            m = mb & (CAT == ci)
            MU[ci, b] = X[m].mean(0) if m.sum() >= 20 else MU[ci, b - 1]
    print("[setup] ready", flush=True)

    FR = np.linspace(0, 1, 9)
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for ai, src in enumerate([0, 1]):                 # sweep from each category
        tgt = 1 - src
        Uc = np.stack([MU[tgt, b] - MU[src, b] for b in range(NBINS)])
        Uc /= (np.linalg.norm(Uc, axis=1, keepdims=True) + 1e-9)
        top_r, bot_r, pr_t = [], [], []
        for f in FR:
            RHO = np.array([(1 - f) * (MU[src, b] @ Uc[b]) + f * (MU[tgt, b] @ Uc[b])
                            for b in range(NBINS)], np.float32)
            drs, Ps = [], []
            for j in range(5):
                F, A, doors = rollout(policy, mk(CATS[src], 30001 + 2 * j), 12,
                                      view_size, 300, device, cargs,
                                      U=Uc.astype(np.float32), RHO=RHO)
                drs += doors
                Ps.append(clf.predict_proba(F[A])[:, tgt].mean())
            drs = np.array(drs)
            top_r.append(float((drs == "top").mean()))
            bot_r.append(float((drs == "bottom").mean()))
            pr_t.append(float(np.mean(Ps)))
            print(f"[{CATS[src]}->{CATS[tgt]}] f={f:.2f} P_probe(tgt)={pr_t[-1]:.2f} "
                  f"top={top_r[-1]:.2f} bottom={bot_r[-1]:.2f}", flush=True)
        ax = axs[ai]
        ax.plot(FR, pr_t, "-o", ms=4, c="#777", label="probe P(target cat)")
        tgt_door = top_r if CATS[tgt] == "rocky" else bot_r
        src_door = bot_r if CATS[tgt] == "rocky" else top_r
        ax.plot(FR, tgt_door, "-s", ms=4, lw=2, c="#8e44ad", label="P(target-category door)")
        ax.plot(FR, src_door, "-^", ms=4, lw=2, c="#1b9e77", label="P(source-category door)")
        ax.axhline(0.5, ls=":", c="#bbb", lw=0.8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"fork_wall: {CATS[src]} maps, clamp toward {CATS[tgt]}")
        ax.set_xlabel("clamp fraction f along class-mean axis")
        ax.legend(fontsize=8)
    fig.suptitle("BT fork_wall psychometric: door choice vs belief-arc clamp position",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "outputs/report_geometry/bt_forkwall_psychometric.png"
    fig.savefig(out, dpi=145)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
