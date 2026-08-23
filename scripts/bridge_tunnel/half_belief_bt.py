#!/usr/bin/env python3
"""Belief-clamp interventions on bridge_tunnel_commit (aux-belief agent).

Clamp the GRU carry along the time-local lakes<->rocky class-mean axis to
  frac=0.5  : the Hellinger arc apex (50/50 half-belief), or
  frac=1.0  : the OPPOSITE category's mean (full swap),
from episode start until the agent commits or t=CLAMP_T; then release.

Questions: does commitment stall under a half-belief? which way does it
collapse? does the implanted belief persist after release, or does the
ever-present terrain evidence re-correct it (contrast with MemoryEnv, where
the cue is gone and implants persist)? and do full swaps cause irreversible
wrong commitments?

  python scripts/bridge_tunnel/half_belief_bt.py
"""
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
from cogniland.bridge_tunnel import generate_commit_map  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from eval_bridge_tunnel_commit_ppo import _load_policy  # noqa: E402
from belief_space_bt import collect  # noqa: E402

CKPT = Path("released_models/bridge_tunnel_commit/ppo_gru_commit_aux_belief.pt")
CATS = ["lakes", "rocky"]
CAT_COL = {"lakes": "#3b6fb6", "rocky": "#b5651d"}
TBIN = 10
NBINS = 13          # bins 0..120+, last bin catches the rest
CLAMP_T = 60


def tbin(t):
    return min(t // TBIN, NBINS - 1)


@torch.no_grad()
def clamped_rollout(policy, rec, n_traj, view_size, max_steps, device,
                    U=None, RHO=None):
    """Rollout with an optional carry clamp h <- h + (rho - h.u) u, active
    until the env commits or t >= CLAMP_T. Returns per-step probe features,
    commit info and outcomes."""
    H, W = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=H, width=W,
                                  view_size=view_size, max_steps=max_steps)
            for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    active = np.ones(n_traj, dtype=bool)
    committed = np.zeros(n_traj, dtype=bool)
    commit_t = np.full(n_traj, -1)
    final_commit = np.zeros(n_traj, dtype=np.int64)
    reached = np.zeros(n_traj, dtype=bool)
    feats, alive_tr = [], []
    for t in range(max_steps):
        if U is not None and t < CLAMP_T:
            b = tbin(t)
            u = torch.from_numpy(U[b]).to(device, torch.float32)
            rho = float(RHO[b])
            win = torch.from_numpy(active & ~committed).to(device)
            proj = (h[0] @ u)
            delta = (rho - proj).unsqueeze(-1) * u.unsqueeze(0)
            h[0] = torch.where(win.unsqueeze(-1), h[0] + delta, h[0])
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        gru_out, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                         torch.zeros(1, n_traj, device=device), h)
        feats.append(gru_out.squeeze(0).cpu().numpy())
        alive_tr.append(active.copy())
        logits, _ = policy._heads(gru_out.squeeze(0))
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, r, term, trunc, info = e.step(int(acts[i]))
            obs[i] = o
            if info["committed_now"]:
                committed[i] = True
                commit_t[i] = t
            final_commit[i] = info["commit"]
            if term:
                reached[i] = True; active[i] = False
            elif trunc:
                active[i] = False
        if not active.any():
            break
    return (np.stack(feats), np.stack(alive_tr), committed, commit_t,
            final_commit, reached)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cargs, view_size, env_size, env_width = _load_policy(CKPT, device)
    gh = cargs.get("goal_half", 1)

    def make_map(cat, seed):
        return generate_commit_map(size=env_size, width=env_width, seed=seed,
                                   category=cat, tree_frac=cargs.get("tree_frac", 0.03),
                                   goal_half=(gh if (gh is not None and gh >= 0) else None))

    # ── training data (even seeds): probe + time-binned class means ────────
    X, CAT, TT = [], [], []
    for ci, cat in enumerate(CATS):
        for j in range(0, 12, 1):
            rec = make_map(cat, 20000 + 2 * j)          # even seeds = train
            Hs, Pr, Cm, Tt = collect(policy, rec, 6, view_size, 300, device)
            X.append(Hs); TT.append(Tt); CAT.append(np.full(len(Hs), ci))
    X, CAT, TT = map(np.concatenate, (X, CAT, TT))
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000).fit(X[::4], CAT[::4])
    U = np.zeros((NBINS, X.shape[1]), np.float32)
    MU = np.zeros((2, NBINS, X.shape[1]), np.float32)
    for b in range(NBINS):
        mb = np.array([tbin(t) for t in TT]) == b
        for ci in range(2):
            m = mb & (CAT == ci)
            MU[ci, b] = X[m].mean(0) if m.sum() >= 20 else (MU[ci, b - 1] if b else 0)
        u = MU[1, b] - MU[0, b]
        U[b] = u / (np.linalg.norm(u) + 1e-9)
    print("[setup] probe + time-binned class means ready", flush=True)

    # ── interventions on held-out maps (odd seeds) ─────────────────────────
    conds = ["baseline", "half", "swap"]
    results = {}
    curves = {}
    for ci, cat in enumerate(CATS):
        for cond in conds:
            allP, allC, allT, allF, allR = [], [], [], [], []
            for j in range(6):
                rec = make_map(cat, 20001 + 2 * j)      # odd seeds = held out
                if cond == "baseline":
                    Uc, RHO = None, None
                else:
                    frac = 0.5 if cond == "half" else 1.0
                    src, tgt = ci, 1 - ci
                    Uc = np.stack([(MU[tgt, b] - MU[src, b])
                                   / (np.linalg.norm(MU[tgt, b] - MU[src, b]) + 1e-9)
                                   for b in range(NBINS)]).astype(np.float32)
                    RHO = np.array([(1 - frac) * (MU[src, b] @ Uc[b])
                                    + frac * (MU[tgt, b] @ Uc[b])
                                    for b in range(NBINS)], np.float32)
                F, A, comm, ct, fc, rc = clamped_rollout(
                    policy, rec, 16, view_size, 300, device, U=Uc, RHO=RHO)
                T_, N_ = A.shape
                P = clf.predict_proba(F.reshape(-1, F.shape[-1]))[:, ci].reshape(T_, N_)
                P[~A] = np.nan
                allP.append(P[:200])
                allC.append(comm); allT.append(ct); allF.append(fc); allR.append(rc)
            Pm = np.full((len(allP), 200, 16), np.nan)
            for k, P in enumerate(allP):
                Pm[k, :P.shape[0]] = P
            curves[(cat, cond)] = np.nanmean(Pm, axis=(0, 2))
            ct = np.concatenate(allT); fc = np.concatenate(allF)
            rc = np.concatenate(allR)
            ct_in = ct[(ct >= 0) & (ct < CLAMP_T)]
            results[(cat, cond)] = dict(
                commit_in_window=float(((ct >= 0) & (ct < CLAMP_T)).mean()),
                commit_t_med=float(np.median(ct[ct >= 0])) if (ct >= 0).any() else -1,
                build=float((fc == 1).mean()), mine=float((fc == 2).mean()),
                none=float((fc == 0).mean()), success=float(rc.mean()))
            r = results[(cat, cond)]
            print(f"[{cat:5s} | {cond:8s}] commit<{CLAMP_T}: {r['commit_in_window']:.2f}  "
                  f"med t_commit: {r['commit_t_med']:.0f}  "
                  f"build/mine/none: {r['build']:.2f}/{r['mine']:.2f}/{r['none']:.2f}  "
                  f"success: {r['success']:.2f}", flush=True)

    # ── figure ──────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ls = {"baseline": "-", "half": "--", "swap": ":"}
    for ai, cat in enumerate(CATS):
        for cond in conds:
            axs[ai].plot(curves[(cat, cond)], ls[cond], lw=2, c=CAT_COL[cat],
                         label=f"{cond}")
        axs[ai].axvspan(0, CLAMP_T, color="#8e44ad", alpha=0.08)
        axs[ai].text(2, 0.05, "clamp window", fontsize=8, color="#8e44ad")
        axs[ai].axhline(0.5, ls=":", c="#999", lw=0.8)
        axs[ai].set_ylim(-0.02, 1.02)
        axs[ai].set_title(f"{cat} maps: P(true category | h) under belief clamps")
        axs[ai].set_xlabel("timestep"); axs[ai].set_ylabel("P(true cat)")
        axs[ai].legend(fontsize=8)
    fig.tight_layout()
    out = Path("outputs/report_geometry/bt_half_belief.png")
    fig.savefig(out, dpi=145)
    print(f"[half_belief_bt] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
