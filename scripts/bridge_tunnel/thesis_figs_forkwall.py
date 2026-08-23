#!/usr/bin/env python3
"""Thesis-chapter figures for the BT fork_wall environment (sections 1-7).

Subcommands (each writes into outputs/thesis_forkwall/):
  env       §1 annotated environment figure + egocentric observation
  dataset   §2 grid of dataset maps across the three categories

  python scripts/bridge_tunnel/thesis_figs_forkwall.py env dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402
from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402

OUT = Path("outputs/thesis_forkwall")
MK_KW = dict(size=32, width=64, tree_frac=0.03, goal_half=0, fork_wall=True,
             passage_half=1, wall_margin=1)
CAT_DOOR = {"lakes": "bottom", "rocky": "top", "balanced": "either"}


def mk(seed, cat):
    return generate_commit_map(seed=seed, category=cat, **MK_KW)


def _draw_map(ax, rec, mark_doors=True, mark_spawn=True):
    ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
    if mark_doors:
        top_ok = rec.correct_target in ("top", "either")
        bot_ok = rec.correct_target in ("bottom", "either")
        for cells, ok in ((rec.top_goal_cells, top_ok), (rec.bottom_goal_cells, bot_ok)):
            for r, c in cells:
                ax.scatter([c], [r], c=("lime" if ok else "red"), s=42, marker="s",
                           edgecolors="k", lw=0.8, zorder=5)
    if mark_spawn:
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=34, marker="o",
                   edgecolors="k", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])


def fig_env():
    """§1: annotated fork_wall map + the agent's egocentric observation."""
    rec = mk(7, "lakes")
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.06)

    ax = fig.add_subplot(gs[0])
    _draw_map(ax, rec)
    # annotations
    H, W = rec.terrain.shape
    ann = dict(color="white", fontsize=11, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65))
    ax.annotate("spawn", (rec.spawn[1] + 1.2, rec.spawn[0] - 1.2), **ann)
    ax.annotate("fork wall", xy=(rec.wall_col, 3.0), xytext=(rec.wall_col - 14, 2.4),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.6), **ann)
    pr, pc = rec.passage_cells[len(rec.passage_cells) // 2]
    ax.annotate("passage", (pc - 9.5, pr + 0.4), **ann)
    tr, tc = rec.top_goal_cells[0]
    br, bc = rec.bottom_goal_cells[0]
    ax.annotate("top door\n(rocky)", (tc - 8.5, tr + 1.6), **ann)
    ax.annotate("bottom door\n(lakes)", (bc - 10.5, br - 1.0), **ann)
    # example obstacle labels
    wy, wx = np.argwhere(rec.terrain == T.WATER)[len(np.argwhere(rec.terrain == T.WATER)) // 3]
    ax.annotate("water (build bridge)", (wx - 6, wy - 0.6), color="white", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="#1f4f8f", alpha=0.85))
    rocks = np.argwhere(rec.terrain == T.ROCK)
    if len(rocks):
        ry, rx = rocks[len(rocks) // 2]
        ax.annotate("rock (mine tunnel)", (rx - 6, ry - 0.6), color="white", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#6b4c2a", alpha=0.9))
    ax.set_title(f"fork_wall map (category = {rec.category}; correct door = "
                 f"{rec.correct_target}, marked lime; decoy red)", fontsize=11)

    # egocentric observation at a mid-map position
    env = BridgeTunnelCommitEnv(map_record=rec, size=H, width=W, view_size=21,
                                max_steps=800, commit=False)
    obs, _ = env.reset()
    for _ in range(14):                      # walk right a bit for a richer view
        obs, *_ = env.step(3)
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(T.TILE_COLORS[obs["minimap"]], interpolation="nearest")
    v = obs["minimap"].shape[0]
    ax2.scatter([v // 2], [v // 2], color="white", s=60, marker="o", edgecolors="k", zorder=5)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"agent observation: {v}×{v} egocentric crop\n"
                  f"+ scalars [compass, facing, step] (agent = white dot)", fontsize=10)
    fig.suptitle("The bridge_tunnel fork_wall environment", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_env.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT/'fig1_env.png'}")


def fig_dataset():
    """§2: grid of dataset maps, 3 categories x 5 seeds."""
    seeds = [1, 2, 3, 4, 5]
    fig, axs = plt.subplots(3, len(seeds), figsize=(3.1 * len(seeds), 5.6))
    for ri, cat in enumerate(CATEGORIES):
        for ci, s in enumerate(seeds):
            rec = mk(s, cat)
            _draw_map(axs[ri, ci], rec)
            axs[ri, ci].set_title(f"seed {s}", fontsize=8)
        axs[ri, 0].set_ylabel(f"{cat}\n(door: {CAT_DOOR[cat]})", fontsize=10)
        for ci in range(len(seeds)):
            axs[ri, ci].set_yticks([])
    fig.suptitle("Dataset: procedurally generated maps by category "
                 "(white=spawn, lime=correct door, red=decoy)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig2_dataset.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT/'fig2_dataset.png'}")


# ───────────────────────── §4 rollouts / OOD / regimes ─────────────────────────

def _policy(ckpt):
    import torch
    from eval_bridge_tunnel_forkwall import _load_policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cargs, view_size, env_size, env_width = _load_policy(Path(ckpt), device)
    return policy, cargs, view_size, device


def fig_rollouts(ckpt, tag=""):
    """§4a: stochastic rollout grids on held-out maps (30 rollouts/map)."""
    from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered
    from eval_bridge_tunnel_commit_ppo import _draw_commit_path
    policy, cargs, view_size, device = _policy(ckpt)
    fig, axs = plt.subplots(3, 3, figsize=(14.5, 7.2))
    for ri, cat in enumerate(CATEGORIES):
        for ci, seed in enumerate([10_000, 10_001, 10_002]):
            rec = mk(seed, cat)
            out = batched_rollout_steered(policy, rec, 30, view_size, 600, device,
                                          commit=False)
            ax = axs[ri, ci]
            ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            for i, tr in enumerate(out["trajs"]):
                _draw_commit_path(ax, tr, out["commits"][i], out["success"][i])
            _draw_map(ax, rec, mark_spawn=True)
            s = out["success"].mean(); w = (out["reached_any"] & ~out["success"]).mean()
            ax.set_title(f"{cat} s{seed}  succ {s:.0%} wrong {w:.0%}", fontsize=8.5)
    fig.suptitle("Stochastic rollouts on held-out maps (30/map)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fn = f"fig4a_rollouts{tag.replace(' ', '_')}.png"
    fig.savefig(OUT / fn, dpi=135, bbox_inches="tight")
    print(f"wrote {OUT/fn}")


OOD_SETS = {
    "in-dist":       dict(),
    "large (40×96)": dict(size=40, width=96),
    "dense trees":   dict(tree_frac=0.12),
    "narrow passage": dict(passage_half=0),
}


def fig_ood(ckpt, tag=""):
    """§4b: success on OOD map variants, per category."""
    from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered
    policy, cargs, view_size, device = _policy(ckpt)
    res = {}
    for name, over in OOD_SETS.items():
        kw = {**MK_KW, **over}
        res[name] = {}
        for cat in CATEGORIES:
            succ, wrong, none = [], [], []
            for j in range(6):
                rec = generate_commit_map(seed=11_000 + j, category=cat, **kw)
                out = batched_rollout_steered(policy, rec, 10, view_size, 800, device,
                                              commit=False)
                succ += out["success"].tolist()
                wrong += (out["reached_any"] & ~out["success"]).tolist()
                none += (~out["reached_any"]).tolist()
            res[name][cat] = (float(np.mean(succ)), float(np.mean(wrong)),
                              float(np.mean(none)))
            print(f"[ood] {name:15s} {cat:9s} succ={res[name][cat][0]:.2f} "
                  f"wrong={res[name][cat][1]:.2f} timeout={res[name][cat][2]:.2f}",
                  flush=True)
    names = list(OOD_SETS)
    x = np.arange(len(names))
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, cat in zip(axs, CATEGORIES):
        s = [res[n][cat][0] for n in names]
        w = [res[n][cat][1] for n in names]
        t = [res[n][cat][2] for n in names]
        ax.bar(x, s, color="#2ca02c", label="success")
        ax.bar(x, w, bottom=s, color="#d62728", label="wrong door")
        ax.bar(x, t, bottom=np.array(s) + np.array(w), color="#bbbbbb", label="timeout")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=12)
        ax.set_ylim(0, 1.0); ax.set_title(cat)
    axs[0].set_ylabel("fraction of episodes"); axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Generalisation to out-of-distribution maps",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fn = f"fig4b_ood{tag.replace(' ', '_')}.png"
    fig.savefig(OUT / fn, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT/fn}")
    (OUT / "ood_results.json").write_text(__import__("json").dumps(res, indent=2))


def fig_regimes(ckpt, tag=""):
    """§4c: agent observations at build / mine / avoid moments."""
    import torch
    policy, cargs, view_size, device = _policy(ckpt)
    frames = {"build": [], "mine": [], "avoid": []}
    torch.manual_seed(0)
    for cat, seed in [("lakes", 10_003), ("rocky", 10_004), ("balanced", 10_005),
                      ("lakes", 10_006), ("rocky", 10_007)]:
        rec = mk(seed, cat)
        env = BridgeTunnelCommitEnv(map_record=rec, size=rec.terrain.shape[0],
                                    width=rec.terrain.shape[1], view_size=view_size,
                                    max_steps=600, commit=False)
        obs, _ = env.reset()
        h = torch.zeros(1, 1, policy.gru_hidden, device=device)
        skill_step = {"build": [], "mine": []}
        traj = []
        with torch.no_grad():
            for t in range(600):
                mm = torch.from_numpy(obs["minimap"][None, None]).to(device)
                sc = torch.from_numpy(obs["scalars"][None, None]).to(device)
                g, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                           torch.zeros(1, 1, device=device), h)
                logits, _ = policy._heads(g.squeeze(0))
                a = int(torch.distributions.Categorical(logits=logits).sample())
                prev_mm = obs["minimap"].copy()
                pr, pc = env._pos
                obs, r, term, trunc, info = env.step(a)
                from eval_bridge_tunnel_commit_ppo import _FACE_DELTA
                dr, dc = _FACE_DELTA[info["facing"]]
                fr, fc = pr + dr, pc + dc                 # world cell ahead of the agent
                Hh, Ww = rec.terrain.shape
                fwd = rec.terrain[fr, fc] if (0 <= fr < Hh and 0 <= fc < Ww) else T.OOB
                traj.append((prev_mm, a, bool(info["placed"]), bool(info["mined"]), fwd))
                if term or trunc:
                    break
        for i, (mmf, a, placed, mined, fwd) in enumerate(traj):
            if placed and len(frames["build"]) < 3:
                frames["build"].append((mmf, cat))
            if mined and len(frames["mine"]) < 3:
                frames["mine"].append((mmf, cat))
            # avoid: obstacle directly ahead, no skill in the next 12 steps
            if (fwd in (T.WATER, T.ROCK) and not placed and not mined
                    and not any(p or m for _, _, p, m, _ in traj[i:i + 12])
                    and len(frames["avoid"]) < 3):
                frames["avoid"].append((mmf, cat))
    fig, axs = plt.subplots(3, 3, figsize=(9.6, 10.0))
    for ri, regime in enumerate(["build", "mine", "avoid"]):
        for ci in range(3):
            ax = axs[ri, ci]
            if ci < len(frames[regime]):
                mmf, cat = frames[regime][ci]
                ax.imshow(T.TILE_COLORS[mmf], interpolation="nearest")
                v = mmf.shape[0]
                ax.scatter([v // 2], [v // 2], color="white", s=46, marker="o",
                           edgecolors="k", zorder=5)
                ax.set_title(f"{regime} ({cat})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Agent observations by behavioural regime\n"
                 "(egocentric view at the step the regime is expressed; agent = white dot)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fn = f"fig4c_regimes{tag.replace(' ', '_')}.png"
    fig.savefig(OUT / fn, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT/fn}")


# ───────────────────── §5-7 labeled collector + probes ─────────────────────

ACTION_NAMES = ["up", "down", "left", "right", "build", "mine"]


def collect_labeled(policy, rec, n_traj, view_size, max_steps, device,
                    fork_wall=True):
    """Stochastic rollouts on one map capturing per-step GRU features + labels.

    Returns list of episode dicts: h (T,H), act (T,), pos (T,2), ahead (T,),
    success, door, used_build, used_mine, category."""
    import torch
    from eval_bridge_tunnel_commit_ppo import _FACE_DELTA
    from eval_bridge_tunnel_forkwall import _door_of
    Hh, Ww = rec.terrain.shape
    episodes = []
    for k in range(n_traj):
        env = BridgeTunnelCommitEnv(map_record=rec, size=Hh, width=Ww,
                                    view_size=view_size, max_steps=max_steps,
                                    commit=False, fork_wall=fork_wall)
        obs, _ = env.reset()
        h = torch.zeros(1, 1, policy.gru_hidden, device=device)
        feats, acts, poss, ahead = [], [], [], []
        nb = nm = 0
        succ = False; fpos = None
        with torch.no_grad():
            for t in range(max_steps):
                mm = torch.from_numpy(obs["minimap"][None, None]).to(device)
                sc = torch.from_numpy(obs["scalars"][None, None]).to(device)
                g, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                           torch.zeros(1, 1, device=device), h)
                logits, _ = policy._heads(g.squeeze(0))
                a = int(torch.distributions.Categorical(logits=logits).sample())
                pr, pc = env._pos
                obs, r, term, trunc, info = env.step(a)
                dr, dc = _FACE_DELTA[info["facing"]]
                fr, fc = pr + dr, pc + dc
                fwd = rec.terrain[fr, fc] if (0 <= fr < Hh and 0 <= fc < Ww) else T.OOB
                feats.append(g.squeeze(0).squeeze(0).cpu().numpy())
                acts.append(a); poss.append((pr, pc)); ahead.append(int(fwd))
                nb += int(info["placed"]); nm += int(info["mined"])
                if term:
                    succ = bool(info["reached_target"]); fpos = env._pos; break
                if trunc:
                    break
        episodes.append(dict(
            h=np.asarray(feats, np.float32), act=np.asarray(acts),
            pos=np.asarray(poss), ahead=np.asarray(ahead),
            success=succ, door=(_door_of(rec, fpos) if fpos else "none"),
            used_build=nb > 0, used_mine=nm > 0, category=rec.category))
    return episodes


def _gather(policy, view_size, device, cats=CATEGORIES, seeds=range(10_000, 10_008),
            n_traj=6, max_steps=600):
    eps = []
    for cat in cats:
        for s in seeds:
            eps += collect_labeled(policy, mk(s, cat), n_traj, view_size, 600, device)
    return eps


def fig_hidden(ckpt, tag=""):
    """§5: activations by next action + probe battery."""
    from sklearn.linear_model import LogisticRegression
    policy, cargs, view_size, device = _policy(ckpt)
    eps = _gather(policy, view_size, device)
    H = np.concatenate([e["h"] for e in eps])
    ACT = np.concatenate([e["act"] for e in eps])
    AHEAD = np.concatenate([e["ahead"] for e in eps])
    CAT = np.concatenate([np.full(len(e["h"]), CATEGORIES.index(e["category"]))
                          for e in eps])
    DOOR = np.concatenate([np.full(len(e["h"]), 1 if e["door"] == "top" else 0)
                           for e in eps])
    XPOS = np.concatenate([e["pos"][:, 1] for e in eps])
    EP = np.concatenate([np.full(len(e["h"]), i) for i, e in enumerate(eps)])
    rng = np.random.default_rng(0)

    # (A) PCA colored by next action — class-balanced subsample so the rare
    # build/mine states are visible under the dominant movement actions
    per = []
    for a in range(6):
        ia = np.where(ACT == a)[0]
        per.append(rng.permutation(ia)[:800])
    idx = np.concatenate(per)
    mu0 = H[idx].mean(0)
    Hc = H[idx] - mu0
    _, S_, Vt = np.linalg.svd(Hc, full_matrices=False)
    co = Hc @ Vt[:2].T
    # (B) class-mean activation heatmap by next action (top-40 selective units)
    mus = np.stack([H[ACT == a].mean(0) for a in range(6)])
    sel = mus.std(0)
    top = np.argsort(-sel)[:40]
    # (C) probe battery, split by episode (train even eps, test odd)
    tr = EP % 2 == 0; te = ~tr
    probes = {
        "map category (3)": (CAT, 1 / 3),
        "next action (6)": (ACT, max(np.bincount(ACT[te]) / ACT[te].size)),
        "terrain ahead (4)": (np.searchsorted([0.5, 1.5, 2.5],
                                              np.clip(AHEAD, 0, 3)), None),
        "final door (2)": (DOOR, max(np.mean(DOOR[te]), 1 - np.mean(DOOR[te]))),
        "x-position (8)": (np.minimum(XPOS // 8, 7).astype(int), 1 / 8),
    }
    accs, bases, names = [], [], []
    sub = rng.permutation(int(tr.sum()))[:40_000]
    for name, (y, base) in probes.items():
        if len(set(y[tr][sub])) < 2:          # degenerate label (e.g. constant door)
            base = max(np.bincount(y[te]) / y[te].size)
            accs.append(base); bases.append(base); names.append(name + " [const]")
            print(f"[probe] {name:20s} DEGENERATE: one class only", flush=True)
            continue
        clf = LogisticRegression(max_iter=2000).fit(H[tr][sub], y[tr][sub])
        acc = clf.score(H[te][::3], y[te][::3])
        if base is None:
            base = max(np.bincount(y[te]) / y[te].size)
        accs.append(acc); bases.append(base); names.append(name)
        print(f"[probe] {name:20s} acc={acc:.3f} (chance/majority {base:.3f})", flush=True)

    fig = plt.figure(figsize=(16.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 1.25], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    cols = plt.get_cmap("tab10")
    for a in range(6):
        m = ACT[idx] == a
        ax.scatter(co[m, 0], co[m, 1], s=6, lw=0, alpha=.5, color=cols(a),
                   label=ACTION_NAMES[a])
    ax.legend(fontsize=8, markerscale=2); ax.set_aspect("equal")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("(A) GRU state PCA, colored by NEXT action")
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(mus[:, top], aspect="auto", cmap="RdBu_r",
                    vmin=-np.abs(mus[:, top]).max(), vmax=np.abs(mus[:, top]).max())
    ax2.set_yticks(range(6)); ax2.set_yticklabels(ACTION_NAMES, fontsize=9)
    ax2.set_xlabel("GRU unit (top-40 action-selective)")
    plt.colorbar(im, ax=ax2, fraction=0.035, label="mean activation")
    ax2.set_title("(B) mean activation by next action")
    ax3 = fig.add_subplot(gs[2])
    x = np.arange(len(names))
    ax3.bar(x - 0.18, accs, width=0.36, color="#2ca02c", label="probe (held-out eps)")
    ax3.bar(x + 0.18, bases, width=0.36, color="#bbbbbb", label="majority/chance")
    ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=7.5, rotation=18)
    ax3.set_ylim(0, 1.02); ax3.legend(fontsize=8)
    ax3.set_title("(C) linear probes on the GRU state")
    fig.suptitle("What the recurrent state encodes", fontsize=13, fontweight="bold")
    fig.savefig(OUT / f"fig5_hidden{tag.replace(' ', '_')}.png", dpi=140,
                bbox_inches="tight")
    print(f"wrote {OUT}/fig5_hidden{tag.replace(' ', '_')}.png")
    return dict(zip(names, zip(accs, bases)))


def fig_belieftraj(ckpt, tag=""):
    """§6: example trajectories + probe-belief over time."""
    from sklearn.linear_model import LogisticRegression
    policy, cargs, view_size, device = _policy(ckpt)
    # train a category probe on collector data (uniform for aux & no-aux)
    eps = _gather(policy, view_size, device, seeds=range(10_000, 10_006), n_traj=5)
    H = np.concatenate([e["h"] for e in eps])
    CAT = np.concatenate([np.full(len(e["h"]), CATEGORIES.index(e["category"]))
                          for e in eps])
    clf = LogisticRegression(max_iter=2000).fit(
        H[::3][:40_000], CAT[::3][:40_000])
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 6.2),
                            gridspec_kw=dict(height_ratios=[1.5, 1.0]))
    for ci, cat in enumerate(CATEGORIES):
        rec = mk(10_020 + ci, cat)
        ep = collect_labeled(policy, rec, 1, view_size, 600, device)[0]
        ax = axs[0, ci]
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        P = ep["pos"]
        tcol = plt.get_cmap("viridis")(np.linspace(0, 1, len(P)))
        ax.scatter(P[:, 1], P[:, 0], c=tcol, s=7, lw=0, zorder=4)
        _draw_map(ax, rec)
        ax.set_title(f"{cat}: trajectory (dark→light = time)  "
                     f"door={ep['door']} succ={ep['success']}", fontsize=9)
        bel = clf.predict_proba(ep["h"])
        ax2 = axs[1, ci]
        for k, c2 in enumerate(CATEGORIES):
            ax2.plot(bel[:, k], lw=1.8,
                     color={"balanced": "#8a8a8a", "lakes": "#3b6fb6",
                            "rocky": "#b5651d"}[c2], label=c2)
        ax2.axhline(1 / 3, ls=":", c="#999", lw=0.8)
        ax2.set_ylim(-0.02, 1.02); ax2.set_xlabel("timestep")
        ax2.set_title(f"probe belief P(category | h) — true: {cat}", fontsize=9)
        if ci == 0:
            ax2.set_ylabel("P(category)"); ax2.legend(fontsize=7)
    fig.suptitle("Trajectories and belief formation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"fig6_belieftraj{tag.replace(' ', '_')}.png", dpi=140,
                bbox_inches="tight")
    print(f"wrote {OUT}/fig6_belieftraj{tag.replace(' ', '_')}.png")


# ─────────────────────────── §7 planning probe ───────────────────────────

def toy_map(seed, obstacle, hh, x_obs=24, wobs=3):
    """Grass corridor + ONE obstacle band centred on the spawn row; keeps the
    original wall/doors. obstacle: 'water'|'rock'; hh: band half-height (rows
    covered = spawn_row ± hh) — small hh makes going around cheap, large hh
    makes the skill crossing increasingly attractive."""
    rec = mk(seed, "balanced")
    terr = rec.terrain
    Hh, Ww = terr.shape
    keep = rec.wall_col - 1                    # keep wall + door strip untouched
    terr[:, 1:keep] = T.GRASS
    tile = T.WATER if obstacle == "water" else T.ROCK
    r0 = max(1, rec.spawn[0] - hh)
    r1 = min(Hh - 1, rec.spawn[0] + hh + 1)
    terr[r0:r1, x_obs:x_obs + wobs] = tile
    return rec


def fig_planning(ckpt, tag=""):
    """§7: does h predict the eventual crossing strategy BEFORE the obstacle?"""
    from sklearn.linear_model import LogisticRegression
    policy, cargs, view_size, device = _policy(ckpt)
    X_OBS = 24
    eps_all = []
    for obstacle in ("water", "rock"):
        for hh in (5, 8, 11):
            for seed in range(20):
                rec = toy_map(30_000 + seed, obstacle, hh)
                eps = collect_labeled(policy, rec, 4, view_size, 600, device)
                for e in eps:
                    e["obstacle"] = obstacle; e["gap"] = hh
                eps_all += eps
    # label: strategy actually expressed at the obstacle
    def strategy(e):
        if e["used_build"] or e["used_mine"]:
            return 1                                       # cross via skill
        return 0                                           # go around
    labelled = [e for e in eps_all if e["success"]]
    ys = np.array([strategy(e) for e in labelled])
    print(f"[plan] {len(labelled)} successful eps; P(skill)={ys.mean():.2f}", flush=True)
    # probe at distance d = X_OBS - x before FIRST reaching the obstacle column
    dists = list(range(20, 1, -1))
    accs, ns = [], []
    tr_mask = np.arange(len(labelled)) % 2 == 0
    for d in dists:
        X, Y, G = [], [], []
        for i, e in enumerate(labelled):
            xcols = e["pos"][:, 1]
            before = np.where(xcols < X_OBS)[0]
            hit = np.where(xcols >= X_OBS - 1)[0]
            tstar = hit[0] if len(hit) else len(xcols)
            sel = before[(X_OBS - xcols[before] == d) & (before < tstar)]
            if len(sel):
                X.append(e["h"][sel[-1]]); Y.append(ys[i]); G.append(i)
        X = np.asarray(X); Y = np.asarray(Y); G = np.asarray(G)
        trm = tr_mask[G]
        if trm.sum() < 30 or (~trm).sum() < 30 or len(set(Y[trm])) < 2:
            accs.append(np.nan); ns.append(len(Y)); continue
        clf = LogisticRegression(max_iter=2000,
                                 class_weight="balanced").fit(X[trm], Y[trm])
        accs.append(clf.score(X[~trm], Y[~trm])); ns.append(len(Y))
    base = max(ys.mean(), 1 - ys.mean())
    view_r = (view_size - 1) // 2
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.3))
    axs[0].plot(dists, accs, "-o", ms=5, c="#8e44ad")
    axs[0].axhline(base, ls="--", c="#999", label=f"majority ({base:.2f})")
    axs[0].axvline(view_r, ls="-.", c="#d62728", lw=1.4,
                   label=f"obstacle enters view (d={view_r})")
    axs[0].axhline(1.0, ls=":", c="#ddd")
    axs[0].invert_xaxis()
    axs[0].set_xlabel("distance to obstacle (columns before contact)")
    axs[0].set_ylabel("probe accuracy (class-balanced): eventual strategy")
    axs[0].set_ylim(0.35, 1.02); axs[0].legend(fontsize=9)
    axs[0].set_title("(A) strategy (around vs skill) decoded from h\n"
                     "as the agent approaches the obstacle")
    # (B) one toy map with example trajectories of each strategy
    rec = toy_map(30_001, "water", 8)
    axs[1].imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
    shown = {0: 0, 1: 0}
    for e in labelled:
        if e["obstacle"] != "water" or e["gap"] != 8:
            continue
        s = strategy(e)
        if shown[s] >= 2:
            continue
        col = "#e377c2" if s else "#17becf"
        axs[1].plot(e["pos"][:, 1], e["pos"][:, 0], color=col, lw=1.8, alpha=0.85)
        shown[s] += 1
    axs[1].set_xticks([]); axs[1].set_yticks([])
    axs[1].set_title("(B) toy map: around (cyan) vs bridge (pink)")
    fig.suptitle("Does the agent know its crossing plan in advance?",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / f"fig7_planning{tag.replace(' ', '_')}.png", dpi=140,
                bbox_inches="tight")
    print(f"wrote {OUT}/fig7_planning{tag.replace(' ', '_')}.png")


FIGS = {"env": fig_env, "dataset": fig_dataset}
CKPT_FIGS = {"rollouts": fig_rollouts, "ood": fig_ood, "regimes": fig_regimes,
             "hidden": fig_hidden, "belieftraj": fig_belieftraj,
             "planning": fig_planning}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["env", "dataset"])
    ap.add_argument("--checkpoint",
                    default="released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for name in a.figs:
        if name in FIGS:
            FIGS[name]()
        else:
            CKPT_FIGS[name](a.checkpoint, a.tag)
