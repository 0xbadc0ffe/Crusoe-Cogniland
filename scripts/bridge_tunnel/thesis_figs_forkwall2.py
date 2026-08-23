#!/usr/bin/env python3
"""Second BT fork_wall experiment batch (no-aux agent) — figures for the PDF.

Subcommands (default checkpoint = the no-aux final):
  curves      (A) return + (C) terminated / door|terminated, no-aux only
  skills      3x2 matrix: mean+-std tunnels mined / bridges built per category
  ood2        OOD averaged across categories: in-dist, small, large, 3 rotations
  beliefform  P(true cat | h_t) mean+-std over many maps, one panel per category
  manifold    3D PCA of GRU states + mean hidden trajectory t0..t80 per category
  probes2     linear probe battery with std bars (5 episode splits) + y-position
  steer       probe-calibrated belief patching at episode end (dose = probe P)
  planning2   planning probe accuracy mean+-std + custom toy-map documentation

  PYTHONPATH=src python scripts/bridge_tunnel/thesis_figs_forkwall2.py <figs...>
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402
from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from thesis_figs_forkwall import (  # noqa: E402
    mk, MK_KW, _policy, collect_labeled, _draw_map, toy_map, OUT)

OUT2 = Path("outputs/thesis_forkwall2")
NOAUX = "outputs/ppo_checkpoints/forkwall_noaux/forkwall_noaux/final.pt"
CAT_COL = {"balanced": "#8a8a8a", "lakes": "#3b6fb6", "rocky": "#b5651d"}
GREEN = "#1b9e77"


def _smooth(x, k=9):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    pad = np.concatenate([x[:1].repeat(k // 2), x, x[-1:].repeat(k // 2)])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


# ─────────────────────────── 1. training curves ───────────────────────────

def fig_curves(ckpt=None):
    rows = [json.loads(l) for l in
            open("outputs/ppo_checkpoints/forkwall_noaux/forkwall_noaux/metrics.jsonl")]
    rows = [r for r in rows if "return/mean" in r]
    step = np.array([r["step"] for r in rows]) / 1e6
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 3.8))
    axs[0].plot(step, _smooth([r["return/mean"] for r in rows]), c=GREEN, lw=2)
    axs[0].set_xlabel("environment steps (M)"); axs[0].set_ylabel("mean episode return")
    axs[0].set_title("(A) return")
    axs[1].plot(step, _smooth([r["success/terminated"] for r in rows]), c=GREEN, lw=2,
                label="terminated (either door)")
    axs[1].plot(step, _smooth([r["success/door_given_terminated"] for r in rows]),
                c=GREEN, lw=2, ls="--", label="correct door | terminated")
    axs[1].set_ylim(-0.02, 1.02); axs[1].set_xlabel("environment steps (M)")
    axs[1].set_title("(C) completion vs decision quality"); axs[1].legend(fontsize=9)
    fig.suptitle("fork_wall PPO+GRU training (no auxiliary loss)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT2.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT2 / "f1_curves.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f1_curves.png'}")


# ─────────────────────────── 2. skills matrix ───────────────────────────

def fig_skills(ckpt=NOAUX):
    from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered
    policy, cargs, view_size, device = _policy(ckpt)
    stats = {}
    for cat in CATEGORIES:
        mines, builds = [], []
        for j in range(10):
            out = batched_rollout_steered(policy, mk(10_000 + j, cat), 12, view_size,
                                          600, device, commit=False)
            mines += out["n_mines"].tolist(); builds += out["n_builds"].tolist()
        stats[cat] = (np.mean(mines), np.std(mines), np.mean(builds), np.std(builds))
        print(f"[skills] {cat:9s} mines {np.mean(mines):.2f}±{np.std(mines):.2f} "
              f"builds {np.mean(builds):.2f}±{np.std(builds):.2f}", flush=True)
    rows = ["rocky", "balanced", "lakes"]
    M = np.array([[stats[c][0], stats[c][2]] for c in rows])   # mean mines, builds
    S = np.array([[stats[c][1], stats[c][3]] for c in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["tunnels mined", "bridges built"], fontsize=11)
    ax.set_yticks(range(3)); ax.set_yticklabels(rows, fontsize=11)
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.1f} ± {S[i, j]:.1f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if M[i, j] < M.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="mean per episode")
    ax.set_title("Skill use per episode (mean ± std, 120 stochastic episodes/category)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT2 / "f2_skills.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f2_skills.png'}")


# ─────────────────────────── 3. OOD incl. rotations ───────────────────────────

def _rot_record(rec, k):
    """Rotate a MapRecord by k*90 degrees CCW (numpy rot90 convention)."""
    H, W = rec.terrain.shape

    def tr(pos):
        r, c = pos
        if k == 1:      # CCW: (r,c) -> (W-1-c, r)
            return (W - 1 - c, r)
        if k == 2:
            return (H - 1 - r, W - 1 - c)
        if k == 3:      # CW: (r,c) -> (c, H-1-r)
            return (c, H - 1 - r)
        return (r, c)

    return dataclasses.replace(
        rec, terrain=np.ascontiguousarray(np.rot90(rec.terrain, k)),
        spawn=tr(rec.spawn), target=tr(rec.target),
        goal_cells=[tr(p) for p in rec.goal_cells],
        top_goal_cells=[tr(p) for p in rec.top_goal_cells],
        bottom_goal_cells=[tr(p) for p in rec.bottom_goal_cells],
        passage_cells=[tr(p) for p in rec.passage_cells],
        wall_col=None)


OOD2 = {
    "in-dist":        (dict(), 0),
    "small (24×48)":  (dict(size=24, width=48), 0),
    "large (40×96)":  (dict(size=40, width=96), 0),
    "top→down":       (dict(), 3),          # CW: east becomes south
    "bottom→up":      (dict(), 1),          # CCW: east becomes north
    "right→left":     (dict(), 2),          # 180°
}


def fig_ood2(ckpt=NOAUX):
    from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered
    policy, cargs, view_size, device = _policy(ckpt)
    res = {}
    for name, (over, k) in OOD2.items():
        per_cat = []
        for cat in CATEGORIES:
            succ = []
            for j in range(5):
                rec = generate_commit_map(seed=11_000 + j, category=cat,
                                          **{**MK_KW, **over})
                if k:
                    rec = _rot_record(rec, k)
                Hh, Ww = rec.terrain.shape
                out = batched_rollout_steered(policy, rec, 10, view_size, 800,
                                              device, commit=False)
                succ += out["success"].tolist()
            per_cat.append(float(np.mean(succ)))
        res[name] = (float(np.mean(per_cat)), float(np.std(per_cat)))
        print(f"[ood2] {name:15s} succ={res[name][0]:.2f}±{res[name][1]:.2f} "
              f"(per-cat {np.round(per_cat, 2)})", flush=True)
    names = list(OOD2)
    x = np.arange(len(names))
    m = [res[n][0] for n in names]; s = [res[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.bar(x, m, yerr=s, capsize=4, color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9, rotation=12)
    ax.set_ylim(0, 1.05); ax.set_ylabel("success (correct door)")
    ax.axhline(1.0, ls=":", c="#ccc")
    ax.set_title("Generalisation to out-of-distribution maps "
                 "(mean ± std across the 3 categories; 150 episodes/bar)")
    fig.tight_layout()
    fig.savefig(OUT2 / "f3_ood2.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f3_ood2.png'}")
    (OUT2 / "ood2.json").write_text(json.dumps(res, indent=2))


# ───────────────────── shared: gather + category probe ─────────────────────

def _gather2(policy, view_size, device, seeds, n_traj=6):
    eps = []
    for cat in CATEGORIES:
        for s in seeds:
            eps += collect_labeled(policy, mk(s, cat), n_traj, view_size, 600, device)
    return eps


def _cat_probe(eps, train_mask):
    from sklearn.linear_model import LogisticRegression
    H = np.concatenate([e["h"] for e in eps])
    CAT = np.concatenate([np.full(len(e["h"]), CATEGORIES.index(e["category"]))
                          for e in eps])
    EP = np.concatenate([np.full(len(e["h"]), i) for i, e in enumerate(eps)])
    tr = train_mask[EP]
    sub = np.random.default_rng(0).permutation(int(tr.sum()))[:40_000]
    clf = LogisticRegression(max_iter=2000).fit(H[tr][sub], CAT[tr][sub])
    return clf


# ─────────────────────────── 4. belief formation ───────────────────────────

def fig_beliefform(ckpt=NOAUX):
    policy, cargs, view_size, device = _policy(ckpt)
    eps = _gather2(policy, view_size, device, range(10_000, 10_010), n_traj=6)
    tr_mask = np.arange(len(eps)) % 2 == 0
    clf = _cat_probe(eps, tr_mask)
    TMAX = 120
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 3.9), sharey=True)
    for ax, cat in zip(axs, CATEGORIES):
        ci = CATEGORIES.index(cat)
        curves = np.full((sum(1 for i, e in enumerate(eps)
                              if not tr_mask[i] and e["category"] == cat), TMAX), np.nan)
        r = 0
        for i, e in enumerate(eps):
            if tr_mask[i] or e["category"] != cat:
                continue
            P = clf.predict_proba(e["h"])[:, ci][:TMAX]
            curves[r, :len(P)] = P
            r += 1
        m = np.nanmean(curves, 0); s = np.nanstd(curves, 0)
        tt = np.arange(TMAX)
        ax.plot(tt, m, c=CAT_COL[cat], lw=2.2)
        ax.fill_between(tt, m - s, m + s, color=CAT_COL[cat], alpha=0.22)
        ax.axhline(1 / 3, ls=":", c="#999", lw=0.9)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("timestep")
        ax.set_title(f"{cat}  (n={r} held-out episodes)")
    axs[0].set_ylabel("probe P(true category | $h_t$)")
    fig.suptitle("Belief formation: mean ± std across held-out maps and episodes",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT2 / "f4_beliefform.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f4_beliefform.png'}")


# ─────────────────────────── 5. 3D belief manifold ───────────────────────────

def fig_manifold(ckpt=NOAUX):
    policy, cargs, view_size, device = _policy(ckpt)
    eps = _gather2(policy, view_size, device, range(10_000, 10_008), n_traj=5)
    H = np.concatenate([e["h"] for e in eps])
    CAT = np.concatenate([np.full(len(e["h"]), CATEGORIES.index(e["category"]))
                          for e in eps])
    rng = np.random.default_rng(0)
    mu0 = H.mean(0)
    _, S_, Vt = np.linalg.svd(H[rng.permutation(len(H))[:8000]] - mu0,
                              full_matrices=False)
    P3 = Vt[:3]
    ev = (S_ ** 2 / (S_ ** 2).sum())[:3]
    # mean hidden trajectory per category, t = 0..80
    TMAX = 80
    mean_tr = {}
    for cat in CATEGORIES:
        ci = CATEGORIES.index(cat)
        acc = np.full((TMAX, H.shape[1]), np.nan)
        for t in range(TMAX):
            hs = [e["h"][t] for e in eps if e["category"] == cat and len(e["h"]) > t]
            if len(hs) >= 5:
                acc[t] = np.mean(hs, 0)
        mean_tr[cat] = (acc - mu0) @ P3.T
    idx = rng.permutation(len(H))[:5000]
    co = (H[idx] - mu0) @ P3.T
    fig = plt.figure(figsize=(13.2, 5.2))
    for pi, (elev, azim) in enumerate([(18, -60), (24, 30)]):
        ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
        for cat in CATEGORIES:
            m = CAT[idx] == CATEGORIES.index(cat)
            ax.scatter(co[m, 0], co[m, 1], co[m, 2], s=4, lw=0, alpha=0.16,
                       color=CAT_COL[cat])
        for cat in CATEGORIES:
            tr = mean_tr[cat]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=CAT_COL[cat], lw=3.2,
                    label=f"{cat} mean $h_t$, t=0→80")
            ok = ~np.isnan(tr[:, 0])
            t0 = np.where(ok)[0][0]; t1 = np.where(ok)[0][-1]
            ax.scatter(*tr[t0], color=CAT_COL[cat], s=55, marker="o", edgecolor="k")
            ax.scatter(*tr[t1], color=CAT_COL[cat], s=85, marker="*", edgecolor="k")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.view_init(elev=elev, azim=azim)
        if pi == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Belief manifold: 3D PCA of GRU states (EV {ev[0]:.0%}/{ev[1]:.0%}/"
                 f"{ev[2]:.0%}) with the mean hidden-state trajectory t=0→80 "
                 "(●=start, ★=t80)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT2 / "f5_manifold.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f5_manifold.png'}")


# ─────────────────────────── 6. probe battery + std ───────────────────────────

def fig_probes2(ckpt=NOAUX):
    from sklearn.linear_model import LogisticRegression
    policy, cargs, view_size, device = _policy(ckpt)
    eps = _gather2(policy, view_size, device, range(10_000, 10_008), n_traj=6)
    H = np.concatenate([e["h"] for e in eps])
    ACT = np.concatenate([e["act"] for e in eps])
    AHEAD = np.concatenate([e["ahead"] for e in eps])
    CAT = np.concatenate([np.full(len(e["h"]), CATEGORIES.index(e["category"]))
                          for e in eps])
    DOOR = np.concatenate([np.full(len(e["h"]), 1 if e["door"] == "top" else 0)
                           for e in eps])
    XP = np.concatenate([e["pos"][:, 1] for e in eps])
    YP = np.concatenate([e["pos"][:, 0] for e in eps])
    EP = np.concatenate([np.full(len(e["h"]), i) for i, e in enumerate(eps)])
    probes = {
        "map category (3)": CAT,
        "next action (6)": ACT,
        "terrain ahead (4)": np.searchsorted([0.5, 1.5, 2.5], np.clip(AHEAD, 0, 3)),
        "final door (2)": DOOR,
        "x-position (8)": np.minimum(XP // 8, 7).astype(int),
        "y-position (8)": np.minimum(YP // 4, 7).astype(int),
    }
    rng = np.random.default_rng(0)
    n_ep = len(eps)
    accs = {k: [] for k in probes}; bases = {k: [] for k in probes}
    for split in range(5):
        perm = rng.permutation(n_ep)
        tr_ep = np.zeros(n_ep, bool); tr_ep[perm[:n_ep // 2]] = True
        tr = tr_ep[EP]; te = ~tr
        sub = rng.permutation(int(tr.sum()))[:35_000]
        for name, y in probes.items():
            if len(set(y[tr][sub])) < 2:
                continue
            clf = LogisticRegression(max_iter=1500).fit(H[tr][sub], y[tr][sub])
            accs[name].append(clf.score(H[te][::4], y[te][::4]))
            bases[name].append(max(np.bincount(y[te]) / y[te].size))
        print(f"[probes2] split {split} done", flush=True)
    names = list(probes)
    m = [np.mean(accs[n]) for n in names]; s = [np.std(accs[n]) for n in names]
    bm = [np.mean(bases[n]) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ax.bar(x - 0.18, m, width=0.36, yerr=s, capsize=4, color=GREEN,
           label="probe (held-out episodes)")
    ax.bar(x + 0.18, bm, width=0.36, color="#bbbbbb", label="majority/chance")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=15)
    ax.set_ylim(0, 1.02); ax.legend(fontsize=9)
    ax.set_title("Linear probes on the GRU state (mean ± std over 5 episode splits)")
    fig.tight_layout()
    fig.savefig(OUT2 / "f6_probes2.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f6_probes2.png'}")
    (OUT2 / "probes2.json").write_text(json.dumps(
        {n: [float(np.mean(accs[n])), float(np.std(accs[n])), float(np.mean(bases[n]))]
         for n in names}, indent=2))


# ──────────────────── 7. probe-calibrated belief steering ────────────────────

def fig_steer(ckpt=NOAUX):
    """Patch h so the binary lakes/rocky probe reads target P(rocky)=p, sustained
    from the approach to the wall until termination; measure door choice vs p."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv
    from eval_bridge_tunnel_forkwall import _door_of
    policy, cargs, view_size, device = _policy(ckpt)

    # binary probe rocky-vs-lakes on the GRU state (all timesteps, held-out maps)
    eps = []
    for cat in ("lakes", "rocky"):
        for s in range(20_000, 20_006):
            eps += collect_labeled(policy, mk(s, cat), 6, view_size, 600, device)
    H = np.concatenate([e["h"] for e in eps])
    Y = np.concatenate([np.full(len(e["h"]), 1 if e["category"] == "rocky" else 0)
                        for e in eps])
    clf = LogisticRegression(max_iter=2000).fit(H[::2], Y[::2])
    acc = clf.score(H[1::2], Y[1::2])
    w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)
    b = float(clf.intercept_[0])
    wn2 = float(w @ w)
    print(f"[steer] binary probe acc {acc:.3f}", flush=True)

    P_TGT = np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95])
    N_TRAJ, N_MAPS = 12, 5

    @torch.no_grad()
    def run(cat, p_tgt):
        L = float(np.log(p_tgt / (1 - p_tgt)))
        doors = []
        for j in range(N_MAPS):
            rec = mk(10_000 + j, cat)
            wall = rec.wall_col
            Hh, Ww = rec.terrain.shape
            envs = [BridgeTunnelCommitEnv(map_record=rec, size=Hh, width=Ww,
                                          view_size=view_size, max_steps=600,
                                          commit=False) for _ in range(N_TRAJ)]
            obs = [e.reset()[0] for e in envs]
            h = torch.zeros(1, N_TRAJ, policy.gru_hidden, device=device)
            active = np.ones(N_TRAJ, bool)
            fpos = [None] * N_TRAJ
            for t in range(600):
                mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
                sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
                _, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                           torch.zeros(1, N_TRAJ, device=device), h)
                # patch: set probe logit to L for envs near/past the wall approach
                cols = np.array([e._pos[1] for e in envs])
                win = torch.from_numpy((cols >= wall - 6) & active).to(device)
                logit = h[0] @ w + b
                delta = (L - logit) / wn2
                h[0] = torch.where(win[:, None], h[0] + delta[:, None] * w[None], h[0])
                logits, _ = policy._heads(h.squeeze(0))
                acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
                for i, e in enumerate(envs):
                    if not active[i]:
                        continue
                    o, r, term, trunc, info = e.step(int(acts[i]))
                    obs[i] = o
                    if term:
                        fpos[i] = e._pos; active[i] = False
                    elif trunc:
                        active[i] = False
                if not active.any():
                    break
            doors += [_door_of(rec, p) for p in fpos]
        top = np.mean([d == "top" for d in doors])
        none = np.mean([d not in ("top", "bottom") for d in doors])
        return float(top), float(none)

    res = {}
    for cat in ("lakes", "rocky"):
        tops, nones = [], []
        for p in P_TGT:
            tp, nn = run(cat, p)
            tops.append(tp); nones.append(nn)
            print(f"[steer] {cat} p_tgt(rocky)={p:.2f} -> P(top)={tp:.2f} none={nn:.2f}",
                  flush=True)
        res[cat] = (tops, nones)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="ideal (door follows probe)")
    for cat, mkr in (("lakes", "o"), ("rocky", "s")):
        ax.plot(P_TGT, res[cat][0], "-" + mkr, ms=6, lw=2, color=CAT_COL[cat],
                label=f"{cat} maps")
    ax.set_xlabel("patched probe probability  P(rocky | $h$)")
    ax.set_ylabel("fraction choosing TOP (rocky) door")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.set_title("Probe-calibrated belief patching at the decision\n"
                 "(patch sets the probe read-out to p, sustained from wall−6 to end)")
    fig.tight_layout()
    fig.savefig(OUT2 / "f7_steer.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f7_steer.png'}")
    (OUT2 / "steer.json").write_text(json.dumps(
        {c: {"p_tgt": P_TGT.tolist(), "p_top": res[c][0], "p_none": res[c][1]}
         for c in res}, indent=2))


# ──────────────────── 8. planning probe + toy-map docs ────────────────────

def fig_planning2(ckpt=NOAUX):
    from sklearn.linear_model import LogisticRegression
    policy, cargs, view_size, device = _policy(ckpt)
    X_OBS = 24
    eps_all = []
    for obstacle in ("water", "rock"):
        for hh in (5, 8, 11):
            for seed in range(20):
                rec = toy_map(30_000 + seed, obstacle, hh)
                es = collect_labeled(policy, rec, 4, view_size, 600, device)
                for e in es:
                    e["obstacle"] = obstacle; e["hh"] = hh
                eps_all += es
    labelled = [e for e in eps_all if e["success"]]
    ys = np.array([1 if (e["used_build"] or e["used_mine"]) else 0 for e in labelled])
    print(f"[plan2] {len(labelled)} successful eps; P(skill)={ys.mean():.2f}", flush=True)
    dists = list(range(20, 1, -1))
    rng = np.random.default_rng(0)
    ACC = np.full((5, len(dists)), np.nan)
    for split in range(5):
        tr_mask = rng.permutation(len(labelled)) % 2 == 0
        for di, d in enumerate(dists):
            X, Y, G = [], [], []
            for i, e in enumerate(labelled):
                xc = e["pos"][:, 1]
                hit = np.where(xc >= X_OBS - 1)[0]
                tstar = hit[0] if len(hit) else len(xc)
                sel = np.where((X_OBS - xc == d) & (np.arange(len(xc)) < tstar))[0]
                if len(sel):
                    X.append(e["h"][sel[-1]]); Y.append(ys[i]); G.append(i)
            X = np.asarray(X); Y = np.asarray(Y); G = np.asarray(G)
            trm = tr_mask[G]
            if trm.sum() < 30 or (~trm).sum() < 30 or len(set(Y[trm])) < 2:
                continue
            clf = LogisticRegression(max_iter=1500, class_weight="balanced").fit(
                X[trm], Y[trm])
            ACC[split, di] = clf.score(X[~trm], Y[~trm])
    base = max(ys.mean(), 1 - ys.mean())
    view_r = (view_size - 1) // 2
    m = np.nanmean(ACC, 0); s = np.nanstd(ACC, 0)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(dists, m, "-o", ms=5, c="#8e44ad")
    ax.fill_between(dists, m - s, m + s, color="#8e44ad", alpha=0.22)
    ax.axhline(base, ls="--", c="#999", label=f"majority ({base:.2f})")
    ax.axvline(view_r, ls="-.", c="#d62728", lw=1.4,
               label=f"obstacle enters view (d={view_r})")
    ax.invert_xaxis()
    ax.set_xlabel("distance to obstacle (columns before contact)")
    ax.set_ylabel("probe accuracy (mean ± std, 5 splits)")
    ax.set_ylim(0.35, 1.02); ax.legend(fontsize=9)
    ax.set_title("Eventual crossing strategy decoded from $h$ before the obstacle")
    fig.tight_layout()
    fig.savefig(OUT2 / "f8a_planning.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f8a_planning.png'}")

    # toy-map documentation: 6 families x 2 example trajectories
    fig, axs = plt.subplots(3, 2, figsize=(13.2, 6.6))
    fams = [(o, hh) for o in ("water", "rock") for hh in (5, 8, 11)]
    for ax, (obstacle, hh) in zip(axs.T.flat, fams):
        rec = toy_map(30_001, obstacle, hh)
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        shown = {0: 0, 1: 0}
        for i, e in enumerate(labelled):
            if e["obstacle"] != obstacle or e["hh"] != hh:
                continue
            sthis = ys[i]
            if shown[sthis] >= 1:
                continue
            col = "#e377c2" if sthis else "#17becf"
            ax.plot(e["pos"][:, 1], e["pos"][:, 0], color=col, lw=2.0, alpha=0.9)
            shown[sthis] += 1
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{obstacle} band, half-height {hh}"
                     f"  (cyan=around, pink=skill)", fontsize=9)
    fig.suptitle("Toy maps: flat grass corridor + one obstacle band centred on the "
                 "spawn row (band = column 24–26, height = spawn±hh); wall & doors kept",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT2 / "f8b_toymaps.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT2/'f8b_toymaps.png'}")


FIGS = {"curves": fig_curves, "skills": fig_skills, "ood2": fig_ood2,
        "beliefform": fig_beliefform, "manifold": fig_manifold,
        "probes2": fig_probes2, "steer": fig_steer, "planning2": fig_planning2}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=list(FIGS))
    ap.add_argument("--checkpoint", default=NOAUX)
    a = ap.parse_args()
    for name in a.figs:
        FIGS[name](a.checkpoint) if name != "curves" else fig_curves()

# ──────────── 7b. class-mean clamp, door-vs-INDUCED-probe curve ────────────

def fig_steer2(ckpt=NOAUX):
    """Whole-episode t-binned class-mean clamp at fractions f (the intervention
    that works); for each f measure the probe P(rocky) it actually induces at
    the decision and the door chosen -> door probability vs probe probability."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv
    from eval_bridge_tunnel_forkwall import _door_of
    policy, cargs, view_size, device = _policy(ckpt)
    TBIN, NBINS = 10, 13

    # training rollouts (calibration seeds): probe + t-binned class means
    eps = []
    for cat in ("lakes", "rocky"):
        for s in range(20_000, 20_006):
            eps += collect_labeled(policy, mk(s, cat), 6, view_size, 600, device)
    H = np.concatenate([e["h"] for e in eps])
    Y = np.concatenate([np.full(len(e["h"]), 1 if e["category"] == "rocky" else 0)
                        for e in eps])
    TT = np.concatenate([np.arange(len(e["h"])) for e in eps])
    clf = LogisticRegression(max_iter=2000).fit(H[::2], Y[::2])
    MU = np.zeros((2, NBINS, H.shape[1]), np.float32)
    for b in range(NBINS):
        mb = np.minimum(TT // TBIN, NBINS - 1) == b
        for ci in range(2):
            m = mb & (Y == ci)
            MU[ci, b] = H[m].mean(0) if m.sum() >= 20 else MU[ci, b - 1]
    print("[steer2] probe + class means ready", flush=True)

    N_TRAJ, N_MAPS = 12, 5
    FR = np.linspace(0, 1, 9)

    @torch.no_grad()
    def run(cat, f):
        src = 1 if cat == "rocky" else 0
        tgt = 1 - src
        Uc = MU[tgt] - MU[src]
        Uc = Uc / (np.linalg.norm(Uc, axis=1, keepdims=True) + 1e-9)
        RHO = np.array([(1 - f) * (MU[src, b] @ Uc[b]) + f * (MU[tgt, b] @ Uc[b])
                        for b in range(NBINS)], np.float32)
        Ut = torch.from_numpy(Uc.astype(np.float32)).to(device)
        Rt = torch.from_numpy(RHO).to(device)
        doors, probeP = [], []
        for j in range(N_MAPS):
            rec = mk(10_000 + j, cat)
            wall = rec.wall_col
            Hh, Ww = rec.terrain.shape
            envs = [BridgeTunnelCommitEnv(map_record=rec, size=Hh, width=Ww,
                                          view_size=view_size, max_steps=600,
                                          commit=False) for _ in range(N_TRAJ)]
            obs = [e.reset()[0] for e in envs]
            h = torch.zeros(1, N_TRAJ, policy.gru_hidden, device=device)
            active = np.ones(N_TRAJ, bool)
            fpos = [None] * N_TRAJ
            pP = [[] for _ in range(N_TRAJ)]
            for t in range(600):
                mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
                sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
                _, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                           torch.zeros(1, N_TRAJ, device=device), h)
                b = min(t // TBIN, NBINS - 1)
                u = Ut[b]
                win = torch.from_numpy(active).to(device)
                proj = h[0] @ u
                h[0] = torch.where(win[:, None],
                                   h[0] + (Rt[b] - proj)[:, None] * u[None], h[0])
                # record induced probe P near the decision
                cols = np.array([e._pos[1] for e in envs])
                hp = h[0].cpu().numpy()
                pr = clf.predict_proba(hp)[:, 1]
                for i in range(N_TRAJ):
                    if active[i] and cols[i] >= wall - 8:
                        pP[i].append(pr[i])
                logits, _ = policy._heads(h.squeeze(0))
                acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
                for i, e in enumerate(envs):
                    if not active[i]:
                        continue
                    o, r, term, trunc, info = e.step(int(acts[i]))
                    obs[i] = o
                    if term:
                        fpos[i] = e._pos; active[i] = False
                    elif trunc:
                        active[i] = False
                if not active.any():
                    break
            doors += [_door_of(rec, p) for p in fpos]
            probeP += [np.mean(x) if len(x) else np.nan for x in pP]
        return (float(np.mean([d == "top" for d in doors])),
                float(np.nanmean(probeP)))

    res = {}
    for cat in ("lakes", "rocky"):
        pts = []
        for f in FR:
            top, pp = run(cat, f)
            pts.append((pp, top))
            print(f"[steer2] {cat} f={f:.2f} probeP(rocky)={pp:.2f} P(top)={top:.2f}",
                  flush=True)
        res[cat] = pts
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="door follows probe (ideal)")
    for cat, mkr in (("lakes", "o"), ("rocky", "s")):
        pts = np.array(res[cat])
        o = np.argsort(pts[:, 0])
        ax.plot(pts[o, 0], pts[o, 1], "-" + mkr, ms=6, lw=2, color=CAT_COL[cat],
                label=f"{cat} maps (clamp f swept 0→1)")
    ax.set_xlabel("measured probe probability P(rocky | clamped $h$) at the decision")
    ax.set_ylabel("fraction choosing TOP (rocky) door")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.set_title("Belief clamp (class-mean axis, whole episode):\n"
                 "door choice vs the probe probability the clamp induces")
    fig.tight_layout()
    fig.savefig(OUT2 / "f7b_steer2.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f7b_steer2.png'}")
    (OUT2 / "steer2.json").write_text(json.dumps(
        {c: [[float(a), float(bb)] for a, bb in res[c]] for c in res}, indent=2))


FIGS["steer2"] = fig_steer2


# ──────────── 9. map→belief and map→door matrices (end of episode) ────────────

def fig_matrices(ckpt=NOAUX):
    """(a) mean end-of-episode probe belief per true category (3x3);
    (b) door reached per true category (top / neither / bottom)."""
    from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered
    from eval_bridge_tunnel_forkwall import _door_of
    policy, cargs, view_size, device = _policy(ckpt)
    ROWS = ["rocky", "balanced", "lakes"]

    # 3-class probe trained on calibration seeds (disjoint from eval seeds)
    eps_cal = _gather2(policy, view_size, device, range(20_000, 20_006), n_traj=6)
    clf = _cat_probe(eps_cal, np.ones(len(eps_cal), bool))

    # eval episodes: final-step belief + door
    B = np.zeros((3, 3)); Bn = np.zeros(3)
    D = np.zeros((3, 3))                              # cols: top, neither, bottom
    for ri, cat in enumerate(ROWS):
        finals = []
        doors = []
        for s in range(10_000, 10_010):
            eps = collect_labeled(policy, mk(s, cat), 6, view_size, 600, device)
            for e in eps:
                finals.append(clf.predict_proba(e["h"][-1:])[0])
                doors.append(e["door"])
        P = np.array(finals)                          # (N,3) in CATEGORIES order
        # reorder columns to ROWS ordering (rocky, balanced, lakes)
        order = [CATEGORIES.index(c) for c in ROWS]
        B[ri] = P[:, order].mean(0); Bn[ri] = len(P)
        D[ri, 0] = np.mean([d == "top" for d in doors])
        D[ri, 2] = np.mean([d == "bottom" for d in doors])
        D[ri, 1] = 1.0 - D[ri, 0] - D[ri, 2]
        print(f"[matrices] {cat:9s} belief={np.round(B[ri],2)} "
              f"door(top/neither/bottom)={np.round(D[ri],2)}", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6))
    im0 = axs[0].imshow(B, cmap="viridis", vmin=0, vmax=1)
    axs[0].set_xticks(range(3))
    axs[0].set_xticklabels([f"P({c})" for c in ROWS], fontsize=10)
    axs[0].set_yticks(range(3)); axs[0].set_yticklabels(ROWS, fontsize=10)
    axs[0].set_ylabel("true map category")
    for i in range(3):
        for j in range(3):
            axs[0].text(j, i, f"{B[i, j]:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if B[i, j] < 0.6 else "black")
    axs[0].set_title("map → belief at end of episode\n(mean probe P over "
                     f"{int(Bn[0])} episodes/row)")
    plt.colorbar(im0, ax=axs[0], fraction=0.045)
    im1 = axs[1].imshow(D, cmap="viridis", vmin=0, vmax=1)
    axs[1].set_xticks(range(3))
    axs[1].set_xticklabels(["top door", "neither", "bottom door"], fontsize=10)
    axs[1].set_yticks(range(3)); axs[1].set_yticklabels(ROWS, fontsize=10)
    for i in range(3):
        for j in range(3):
            axs[1].text(j, i, f"{D[i, j]:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if D[i, j] < 0.6 else "black")
    axs[1].set_title("map → door reached\n(correct: rocky→top, lakes→bottom, "
                     "balanced→either)")
    plt.colorbar(im1, ax=axs[1], fraction=0.045)
    fig.tight_layout()
    fig.savefig(OUT2 / "f9_matrices.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT2/'f9_matrices.png'}")


FIGS["matrices"] = fig_matrices
