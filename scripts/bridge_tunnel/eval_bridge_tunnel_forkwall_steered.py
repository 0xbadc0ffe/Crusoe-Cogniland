#!/usr/bin/env python3
"""Steered evaluation of a fork_wall (no-commit) bridge_tunnel PPO agent.

fork_wall task: the corridor terrain reveals the map archetype (lakes / rocky /
balanced), then a wall with a passage leads to TWO doors — only the door
matching the archetype pays the reach bonus (lakes→bottom, rocky→top,
balanced→either). There is NO commitment mechanic: build/mine are freely
repeatable. The archetype belief therefore has a direct behavioral consumer —
the door CHOICE — unlike the commit env, where the belief readout proved to be
a passenger (fully clampable with zero behavioral effect).

This script ports the steering pipeline of
``eval_bridge_tunnel_commit_ppo_steered.py`` (same strategies, same steering
functions, imported from there) to this setting and asks: when steering warps
the belief — directly (``class-mean`` / ``belief-head`` axis push,
``belief-clamp``) or as a side effect of clamping the build/mine behavior
(``suppress-skill`` / ``substitute-skill``) — does the agent start picking the
WRONG door? A performance drop here is the "steering is dangerous when it
works by changing the belief" result: the intervention aimed at a skill leaks
into every downstream decision that consumes the warped belief.

The strategy set adds ``skill-mean`` on top of the commit-env ones: a linear
push along the unit build→mine BEHAVIOR axis (mean hidden state at executed
mine events − mean at executed build events, collected from unsteered
rollouts on ``--skill-cats``; default balanced maps only, so the contrast is
not confounded by the map category). Cosines against the category class-mean
axis and the belief-head axis are printed — the entanglement measurement.
Since there is no commitment, steering effectiveness is tracked by the
EXECUTED-skill matrix (episodes classified neither/build/mine/both) and mean
executed build/mine counts per episode.

Outputs (``--out-prefix`` +):
  ``_choice_matrix.png``   baseline vs steered door-choice matrices
                           (rows=category, cols=top/bottom/no door; the
                           correct door is outlined in green)
  ``_skill_matrix.png``    baseline vs steered EXECUTED-skill matrices — the
                           no-commit analogue of the commit matrix
  ``_belief.png``          per-category belief-scalar time series
  ``_actionprob.png``      per-category π(build)/π(mine) time series
  ``_traj.png``            steered trajectory grid (green=correct door,
                           red=decoy)
  ``_dose.png``            success & wrong-door vs belief shift dose–response
                           (>1 alpha only)
  ``_results.json``        all numeric results
  ``_steer_dir_<strategy>.npy``  the (H,) direction (axis strategies only)

``--alphas`` entries are one number or a ``lakes:rocky`` pair, as in the
commit-env script.

    python scripts/bridge_tunnel/eval_bridge_tunnel_forkwall_steered.py \\
        --checkpoint released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt \\
        --strategy suppress-skill --alphas 0.25,0.5,1.0
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel import generate_commit_map, tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel.mapgen import CATEGORIES  # noqa: E402
from eval_bridge_tunnel_commit_ppo import _FACE_DELTA, _draw_commit_path  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy, _door_of  # noqa: E402
from eval_bridge_tunnel_commit_ppo_steered import (  # noqa: E402
    plot_action_probs, plot_belief_traces)
# all steering logic lives in the shared library
from cogniland.bridge_tunnel.steering import (  # noqa: E402
    A_BUILD, A_MINE, ACTION_NAMES, BELIEF2I, BELIEF_TARGET, PROHIBIT_ACTION,
    PUSH_ACTION, STEER_SIGN, alpha_label, cat_alpha, class_mean_direction,
    cosine, head_direction, make_bt_steerer, parse_alpha_token)

DOORS = ["top", "bottom", "none"]
DOOR2I = {d: i for i, d in enumerate(DOORS)}
CORRECT_DOOR = {"lakes": "bottom", "rocky": "top", "balanced": "either"}
# no-commit analogue of the commit matrix: classify each episode by which
# skills it actually EXECUTED (successful water→wood builds / rock→grass mines)
SKILL_CLASSES = ["neither", "build", "mine", "both"]


def _skill_class(n_builds: int, n_mines: int) -> int:
    if n_builds > 0 and n_mines > 0:
        return 3
    if n_mines > 0:
        return 2
    if n_builds > 0:
        return 1
    return 0


# ───────────────────────────── rollout ─────────────────────────────

@torch.no_grad()
def batched_rollout_steered(policy, rec, n_traj, view_size, max_steps, device,
                            commit=None, steer_fn=None, collect_hidden=False,
                            collect_skill_hidden=False):
    """``n_traj`` stochastic rollouts on one fixed fork_wall map in lockstep,
    with the same hidden-state steering hook and belief/action/progress
    recording as the commit-env script. ``commit=False`` for no-commit agents.

    Outcome per episode: success (CORRECT door), reached_any (either door),
    door ("top"/"bottom"/"none"), executed/attempted build & mine counts.
    ``collect_skill_hidden`` additionally stacks the hidden states that
    produced each EXECUTED build / mine (for the skill-mean direction).
    """
    H, W = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=H, width=W, view_size=view_size,
                                  max_steps=max_steps, commit=commit)
            for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    done = torch.zeros(n_traj, device=device)
    active = np.ones(n_traj, dtype=bool)
    success = np.zeros(n_traj, dtype=bool)        # reached the CORRECT door
    reached_any = np.zeros(n_traj, dtype=bool)    # reached EITHER door
    final_pos = [None] * n_traj
    n_builds = np.zeros(n_traj, dtype=np.int64)   # executed water→wood builds
    n_mines = np.zeros(n_traj, dtype=np.int64)    # executed rock→grass mines
    att_build = np.zeros(n_traj, dtype=np.int64)  # sampled build actions
    att_mine = np.zeros(n_traj, dtype=np.int64)   # sampled mine actions
    build_h, mine_h = [], []
    trajs = [[tuple(e._pos)] for e in envs]
    commits = [[0] for _ in envs]
    mine_pts, bridge_pts = [], []
    belief_probs = np.full((n_traj, max_steps, 3), np.nan, dtype=np.float32)
    action_probs = np.full((n_traj, max_steps, policy.actor.out_features),
                           np.nan, dtype=np.float32)
    progress_tr = np.full((n_traj, max_steps), np.nan, dtype=np.float32)
    hiddens = []
    # spatial progress = fraction of the spawn→target distance covered
    # (rec.target is the top-door centre; both doors share the last column,
    # so this measures progress toward the gate for either choice)
    spawn_xy = np.asarray(rec.spawn, dtype=np.float64)
    target_xy = np.asarray(rec.target, dtype=np.float64)
    total_dist = max(float(np.linalg.norm(target_xy - spawn_xy)), 1e-6)

    for t in range(max_steps):
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        pos = np.asarray([e._pos for e in envs], dtype=np.float64)
        progress = np.clip(
            1.0 - np.linalg.norm(pos - target_xy, axis=1) / total_dist, 0.0, 1.0
        ).astype(np.float32)
        progress_tr[active, t] = progress[active]
        _, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        is_logit_mask = steer_fn is not None and getattr(steer_fn, "kind", "hidden") == "logits"
        if steer_fn is not None and not is_logit_mask:
            h = steer_fn(h, t, {"progress": progress})
        x = h.squeeze(0)                       # (B, H): steered state feeds ALL heads
        logits, _ = policy._heads(x)
        if is_logit_mask:                      # action-mask: post-head, never touches x/belief
            logits = steer_fn(logits, t, {"progress": progress})
        action_probs[active, t] = torch.softmax(logits, dim=-1).cpu().numpy()[active]
        if policy.belief is not None:
            bp = torch.softmax(policy.belief(x), dim=-1).cpu().numpy()
            belief_probs[active, t] = bp[active]
        if collect_hidden and active.any():
            hiddens.append(x.cpu().numpy()[active])
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        att_build += (acts == A_BUILD) & active
        att_mine += (acts == A_MINE) & active
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, r, term, trunc, info = e.step(int(acts[i]))
            obs[i] = o
            trajs[i].append(tuple(e._pos))
            commits[i].append(int(info["commit"]))
            if info["mined"] or info["placed"]:
                dr, dc = _FACE_DELTA[info["facing"]]
                cell = (e._pos[0] + dr, e._pos[1] + dc)
                (mine_pts if info["mined"] else bridge_pts).append(cell)
                if info["mined"]:
                    n_mines[i] += 1
                else:
                    n_builds[i] += 1
                if collect_skill_hidden:
                    (mine_h if info["mined"] else build_h).append(x[i].cpu().numpy())
            if term:
                success[i] = bool(info["reached_target"])
                reached_any[i] = bool(info["reached_any_target"])
                final_pos[i] = e._pos
                active[i] = False
            elif trunc:
                active[i] = False
        done = torch.zeros(n_traj, device=device)
        if not active.any():
            break
    doors = [_door_of(rec, p) for p in final_pos]
    doors = [d if d in DOORS else "none" for d in doors]
    return dict(trajs=trajs, commits=commits, success=success,
                reached_any=reached_any, doors=doors,
                n_builds=n_builds, n_mines=n_mines,
                att_build=att_build, att_mine=att_mine,
                mine_pts=mine_pts, bridge_pts=bridge_pts,
                belief_probs=belief_probs, action_probs=action_probs,
                progress=progress_tr,
                hiddens=(np.concatenate(hiddens) if hiddens else None),
                build_hiddens=(np.stack(build_h) if build_h else None),
                mine_hiddens=(np.stack(mine_h) if mine_h else None))


def direction_class_mean_fw(policy, *, map_factory, view_size, device, commit,
                            n_maps, n_traj, seed_start, max_steps):
    """Unit lakes→rocky axis from class-mean GRU hiddens of unsteered fork_wall
    rollouts (the fork_wall analogue of the commit script's class-mean)."""
    means = {}
    for cat in ("lakes", "rocky"):
        chunks = []
        for j in range(n_maps):
            rec = map_factory(seed_start + j, cat)
            out = batched_rollout_steered(policy, rec, n_traj, view_size,
                                          max_steps, device, commit=commit,
                                          collect_hidden=True)
            chunks.append(out["hiddens"])
        means[cat] = np.concatenate(chunks).mean(axis=0)
    return class_mean_direction(means["rocky"], means["lakes"], device=device)


def direction_skill_mean_fw(policy, *, map_factory, view_size, device, commit,
                            n_maps, n_traj, seed_start, max_steps,
                            skill_cats=("balanced",)):
    """Unit build→mine BEHAVIOR axis: contrast of mean GRU hidden states at the
    steps that EXECUTED a mine vs a build, from unsteered rollouts.

    This is the 'realistic' steering handle: a direction defined purely by a
    secondary behavior, with the belief never referenced. By default the
    events are collected on BALANCED maps only — both skills occur there under
    a fixed category prior, so the contrast isolates the behavioral axis. With
    ``skill_cats=('balanced','lakes','rocky')`` the contrast is confounded:
    build events come mostly from lakes maps and mine events from rocky maps,
    so the 'behavior' axis largely re-derives the category/belief axis.
    """
    chunks = {"build": [], "mine": []}
    for cat in skill_cats:
        for j in range(n_maps):
            rec = map_factory(seed_start + j, cat)
            out = batched_rollout_steered(policy, rec, n_traj, view_size,
                                          max_steps, device, commit=commit,
                                          collect_skill_hidden=True)
            if out["build_hiddens"] is not None:
                chunks["build"].append(out["build_hiddens"])
            if out["mine_hiddens"] is not None:
                chunks["mine"].append(out["mine_hiddens"])
    n_b = sum(len(c) for c in chunks["build"])
    n_m = sum(len(c) for c in chunks["mine"])
    if n_b < 20 or n_m < 20:
        sys.exit(f"skill-mean: too few skill events on {skill_cats} calibration "
                 f"maps (build={n_b}, mine={n_m}) — raise --calib-maps/--calib-traj "
                 f"or widen --skill-cats")
    print(f"[skill-mean] {n_b} build / {n_m} mine events from {skill_cats} maps")
    return class_mean_direction(np.concatenate(chunks["mine"]).mean(axis=0),
                                np.concatenate(chunks["build"]).mean(axis=0),
                                device=device)


# ───────────────────────────── evaluation ─────────────────────────────

def run_eval(policy, view_size, device, *, map_factory, commit,
             steer_for_category, n_maps, n_traj, seed_start, max_steps):
    """Door-choice matrix + belief/action-prob traces for one steering setting."""
    counts = np.zeros((3, 3), dtype=np.float64)           # [category, door]
    skill_counts = np.zeros((3, 4), dtype=np.float64)     # [category, SKILL_CLASSES]
    succ = {c: [] for c in CATEGORIES}
    wrong = {c: [] for c in CATEGORIES}
    skills = {c: {"n_builds": [], "n_mines": [], "att_build": [], "att_mine": []}
              for c in CATEGORIES}
    beliefs = {c: [] for c in CATEGORIES}
    aprobs = {c: [] for c in CATEGORIES}
    progs = {c: [] for c in CATEGORIES}
    for ci, cat in enumerate(CATEGORIES):
        steer_fn = steer_for_category(cat)
        for j in range(n_maps):
            rec = map_factory(seed_start + j, cat)
            out = batched_rollout_steered(policy, rec, n_traj, view_size,
                                          max_steps, device, commit=commit,
                                          steer_fn=steer_fn)
            for d in out["doors"]:
                counts[ci, DOOR2I[d]] += 1
            for nb, nm in zip(out["n_builds"], out["n_mines"]):
                skill_counts[ci, _skill_class(int(nb), int(nm))] += 1
            succ[cat].extend(out["success"].tolist())
            wrong[cat].extend((out["reached_any"] & ~out["success"]).tolist())
            for k in skills[cat]:
                skills[cat][k].extend(out[k].tolist())
            beliefs[cat].append(out["belief_probs"])
            aprobs[cat].append(out["action_probs"])
            progs[cat].append(out["progress"])
    matrix = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    skill_matrix = skill_counts / skill_counts.sum(axis=1, keepdims=True).clip(min=1)
    beliefs = {c: np.concatenate(v) for c, v in beliefs.items()}
    aprobs = {c: np.concatenate(v) for c, v in aprobs.items()}
    progs = {c: np.concatenate(v) for c, v in progs.items()}
    return matrix, skill_matrix, succ, wrong, skills, beliefs, aprobs, progs


def summarize(matrix, skill_matrix, succ, wrong, skills, beliefs, aprobs, progs,
              late_from=0.5):
    """Per-category scalar metrics (door choice + skill usage + belief +
    skill-action π)."""
    res = {}
    for ci, cat in enumerate(CATEGORIES):
        bp = beliefs[cat]                                  # (N, T, 3)
        valid_a = np.isfinite(aprobs[cat][..., 0])
        ap = aprobs[cat][valid_a]
        late = aprobs[cat][valid_a & (progs[cat] >= late_from)]
        scalar = bp[..., BELIEF2I["lakes"]] - bp[..., BELIEF2I["rocky"]]
        valid = np.isfinite(bp[..., 0])
        finals = [scalar[i, np.where(valid[i])[0][-1]] for i in range(len(scalar))
                  if valid[i].any()]
        am = bp[valid].argmax(axis=-1)
        s = float(np.mean(succ[cat])) if succ[cat] else 0.0
        w = float(np.mean(wrong[cat])) if wrong[cat] else 0.0
        res[cat] = {
            "door_fracs": {d: float(matrix[ci, k]) for k, d in enumerate(DOORS)},
            "correct_door": CORRECT_DOOR[cat],
            "success": s, "wrong_door": w, "timeout": max(0.0, 1.0 - s - w),
            "skill_usage_fracs": {n: float(skill_matrix[ci, k])
                                  for k, n in enumerate(SKILL_CLASSES)},
            "mean_builds": float(np.mean(skills[cat]["n_builds"])),
            "mean_mines": float(np.mean(skills[cat]["n_mines"])),
            "mean_build_attempts": float(np.mean(skills[cat]["att_build"])),
            "mean_mine_attempts": float(np.mean(skills[cat]["att_mine"])),
            "belief_scalar_mean": float(np.nanmean(scalar)),
            "belief_scalar_final": float(np.mean(finals)) if finals else float("nan"),
            "belief_argmax_fracs": {c: float((am == BELIEF2I[c]).mean())
                                    for c in CATEGORIES},
            "pi_action_mean": {"build": float(ap[:, A_BUILD].mean()),
                               "mine": float(ap[:, A_MINE].mean())},
            "pi_action_mean_late": (
                {"build": float(late[:, A_BUILD].mean()),
                 "mine": float(late[:, A_MINE].mean())} if len(late) else None),
        }
    return res


# ───────────────────────────── plotting ─────────────────────────────

def plot_choice_matrix_pair(mats, summaries_list, titles, suptitle, out_path):
    """Baseline vs steered door-choice matrices. The correct door cell(s) per
    row are outlined in green (balanced: both doors count)."""
    fig, axes = plt.subplots(1, len(mats), figsize=(5.6 * len(mats), 4.4))
    fig.subplots_adjust(wspace=0.45)
    axes = np.atleast_1d(axes)
    for ax, m, sm, ttl in zip(axes, mats, summaries_list, titles):
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{d} door" if d != "none" else "no door" for d in DOORS],
                           fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"{c}\n(succ {sm[c]['success']:.0%} / "
                            f"wrong {sm[c]['wrong_door']:.0%})"
                            for c in CATEGORIES], fontsize=9)
        ax.set_xlabel("door reached", fontsize=10)
        for i, cat in enumerate(CATEGORIES):
            ok_doors = (("top", "bottom") if CORRECT_DOOR[cat] == "either"
                        else (CORRECT_DOOR[cat],))
            for j in range(3):
                v = m[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if v < 0.6 else "black", fontweight="bold")
                if DOORS[j] in ok_doors:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="lime", lw=2.2))
        ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel("map category (belief)", fontsize=10)
    fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02, label="fraction of episodes")
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_skill_matrix_pair(mats, summaries_list, titles, suptitle, out_path):
    """Baseline vs steered EXECUTED-skill matrices — the no-commit analogue of
    the commit matrix: each episode classified by the skills it actually used."""
    fig, axes = plt.subplots(1, len(mats), figsize=(6.2 * len(mats), 4.4))
    fig.subplots_adjust(wspace=0.45)
    axes = np.atleast_1d(axes)
    for ax, m, sm, ttl in zip(axes, mats, summaries_list, titles):
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(SKILL_CLASSES)))
        ax.set_xticklabels(SKILL_CLASSES, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels(
            [f"{c}\n(b {sm[c]['mean_builds']:.1f} / m {sm[c]['mean_mines']:.1f} per ep)"
             for c in CATEGORIES], fontsize=9)
        ax.set_xlabel("skills EXECUTED in the episode", fontsize=10)
        for i in range(3):
            for j in range(len(SKILL_CLASSES)):
                v = m[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if v < 0.6 else "black", fontweight="bold")
        ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel("map category (belief)", fontsize=10)
    fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02, label="fraction of episodes")
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_dose_response_fw(all_summaries, alpha_keys, strategy, out_path):
    """Choice performance vs belief shift as alpha grows. Each category is
    plotted against ITS OWN alpha (lakes:rocky pairs supported)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for cat, color in (("lakes", "#1f77b4"), ("rocky", "#8c564b")):
        xs = [cat_alpha(k, cat) for k in alpha_keys]
        s = [all_summaries[k][cat]["success"] for k in alpha_keys]
        w = [all_summaries[k][cat]["wrong_door"] for k in alpha_keys]
        bel = [-STEER_SIGN[cat] * all_summaries[k][cat]["belief_scalar_mean"]
               for k in alpha_keys]
        axes[0].plot(xs, s, "o-", color=color, label=f"{cat} success")
        axes[0].plot(xs, w, "o--", color=color, label=f"{cat} wrong door")
        axes[1].plot(xs, bel, "o-", color=color, label=f"{cat}")
    axes[0].set_ylabel("fraction of episodes"); axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("mean belief toward steer target\n(sign-aligned scalar)")
    axes[1].set_ylim(-1.05, 1.05); axes[1].axhline(0, color="gray", lw=0.7, ls=":")
    for ax in axes:
        ax.set_xlabel("steering strength α (per category)"); ax.legend(fontsize=8)
    fig.suptitle(f"dose–response: door choice vs belief  ·  {strategy}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def plot_grid_steered(policy, view_size, device, *, map_factory, commit,
                      steer_for_category, n_seeds, n_traj, seed_start,
                      max_steps, title, out_path):
    fig, axes = plt.subplots(len(CATEGORIES), n_seeds,
                             figsize=(max(n_seeds * 3.0, 4.5), len(CATEGORIES) * 2.0))
    axes = np.asarray(axes).reshape(len(CATEGORIES), n_seeds)
    for ci, cat in enumerate(CATEGORIES):
        steer_fn = steer_for_category(cat)
        for sj in range(n_seeds):
            rec = map_factory(seed_start + sj, cat)
            out = batched_rollout_steered(policy, rec, n_traj, view_size,
                                          max_steps, device, commit=commit,
                                          steer_fn=steer_fn)
            ax = axes[ci, sj]
            ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            for i, tr in enumerate(out["trajs"]):
                _draw_commit_path(ax, tr, out["commits"][i], out["success"][i])
            if out["mine_pts"]:
                m = np.array(out["mine_pts"]); ax.scatter(m[:, 1], m[:, 0], color="yellow", s=6, alpha=0.18, zorder=3, linewidths=0)
            if out["bridge_pts"]:
                b = np.array(out["bridge_pts"]); ax.scatter(b[:, 1], b[:, 0], color="red", s=6, alpha=0.18, zorder=3, linewidths=0)
            top_ok = rec.correct_target in ("top", "either")
            bot_ok = rec.correct_target in ("bottom", "either")
            for cells, ok in ((rec.top_goal_cells, top_ok), (rec.bottom_goal_cells, bot_ok)):
                if cells:
                    ys = [r for r, c in cells]; xs = [c for r, c in cells]
                    ax.scatter(xs, ys, c=("lime" if ok else "red"), s=26, marker="s",
                               edgecolors="k", zorder=5)
            ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=22, marker="o", edgecolors="k", zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            succ = out["success"].mean()
            wrong = (out["reached_any"] & ~out["success"]).mean()
            ax.set_title(f"{cat} s{seed_start+sj}  succ {succ:.0%} wrong {wrong:.0%}",
                         fontsize=7)
    fig.suptitle(title, fontsize=12, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


# ───────────────────────────── main ─────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="fork_wall PPO checkpoint WITH the auxiliary belief head "
                        "(e.g. released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt)")
    p.add_argument("--out-prefix", type=Path,
                   default=Path("outputs/bridge_tunnel_forkwall/ppo_steered"))
    p.add_argument("--strategy",
                   choices=["belief-head", "class-mean", "skill-mean",
                            "suppress-skill", "substitute-skill", "belief-clamp", "action-mask"],
                   default="suppress-skill")
    p.add_argument("--skill-cats", default="balanced",
                   help="skill-mean: comma list of categories whose skill events "
                        "define the build→mine axis. 'balanced' (default) "
                        "isolates the behavior axis; 'balanced,lakes,rocky' "
                        "gives the category-confounded ablation")
    p.add_argument("--alphas", default="0.25,0.5,1.0",
                   help="comma-separated steering strengths; one number or a "
                        "lakes:rocky pair per entry; baseline α=0 always runs; "
                        "the first entry is used for the headline figures")
    p.add_argument("--steer-from", type=int, default=0)
    p.add_argument("--steer-to", type=int, default=10**9)
    p.add_argument("--clamp-iters", type=int, default=10)
    p.add_argument("--clamp-target", type=float, default=0.01)
    p.add_argument("--push-beta", type=float, default=0.0)
    p.add_argument("--sub-floor", type=float, default=0.05)
    p.add_argument("--sub-from-progress", type=float, default=0.5)
    p.add_argument("--sub-target", choices=["skill", "movement", "movement-entropy"],
                   default="skill",
                   help="substitute-skill: what the floor is applied to — "
                        "'skill' (default) floors π(sub-optimal skill); "
                        "'movement' floors the SUMMED π(up)+π(down)+π(left)+"
                        "π(right) (usually redundant with clamp_target — "
                        "suppressing the optimal skill already frees this "
                        "mass by conservation); 'movement-entropy' instead "
                        "floors the NORMALIZED ENTROPY (0-1) of the movement "
                        "sub-distribution, actively resisting collapse onto "
                        "one direction rather than just adding more mass — "
                        "the actual lever for 'stop the agent oscillating/"
                        "getting stuck at the blocked obstacle'")
    p.add_argument("--sub-stuck-gate", action="store_true",
                   help="substitute-skill: additionally gate the floor term "
                        "on StuckDetector (no ctx['progress'] improvement over "
                        "the last --sub-stuck-window steps) so it only fires "
                        "once a trajectory is actually deadlocked, instead of "
                        "on every step past --sub-from-progress — avoids "
                        "disrupting trajectories that are already succeeding")
    p.add_argument("--sub-stuck-window", type=int, default=20,
                   help="sub-stuck-gate: steps of no progress before 'stuck'")
    p.add_argument("--sub-stuck-eps", type=float, default=1e-3,
                   help="sub-stuck-gate: progress improvement below this "
                        "over the window still counts as 'stuck'")
    p.add_argument("--belief-floor", type=float, default=0.75)
    p.add_argument("--steer-balanced", choices=["rocky", "lakes"], default=None,
                   help="also steer balanced maps (default: unsteered control row), "
                        "TOWARD the named archetype: 'rocky' suppresses build / pushes "
                        "mine / clamps belief->rocky, 'lakes' does the mirror. Uses the "
                        "lakes alpha from --alphas. Run both and compare — steering "
                        "balanced only one way bakes in an arbitrary asymmetry")
    p.add_argument("--matrix-maps", type=int, default=20, help="held-out maps/category")
    p.add_argument("--matrix-traj", type=int, default=16, help="stochastic rollouts/map")
    p.add_argument("--grid-seeds", type=int, default=4)
    p.add_argument("--grid-traj", type=int, default=120)
    p.add_argument("--calib-maps", type=int, default=8)
    p.add_argument("--calib-traj", type=int, default=8)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--calib-seed-start", type=int, default=20_000)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--passage-half", type=int, default=None,
                   help="default: the checkpoint's as-trained value")
    p.add_argument("--wall-margin", type=int, default=None,
                   help="default: the checkpoint's as-trained value")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.sample_seed)
    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    if policy.belief is None:
        sys.exit("checkpoint has no belief head — this experiment needs a belief "
                 "readout to measure; use the fork_wall aux-belief agent.")
    tag = args.checkpoint.stem
    alphas = [parse_alpha_token(tok) for tok in args.alphas.split(",")]
    commit = False if cargs.get("no_commit", False) else None
    passage_half = (args.passage_half if args.passage_half is not None
                    else cargs.get("passage_half", 1))
    wall_margin = (args.wall_margin if args.wall_margin is not None
                   else cargs.get("wall_margin", 1))
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None

    def map_factory(seed, category):
        return generate_commit_map(size=env_size, width=env_width, seed=seed,
                                   category=category,
                                   tree_frac=cargs.get("tree_frac", 0.03),
                                   goal_half=gh, fork_wall=True,
                                   passage_half=passage_half, wall_margin=wall_margin)

    if args.strategy in ("class-mean", "belief-head", "skill-mean"):
        calib = dict(map_factory=map_factory, view_size=view_size, device=device,
                     commit=commit, n_maps=args.calib_maps, n_traj=args.calib_traj,
                     seed_start=args.calib_seed_start, max_steps=args.max_steps)
        if args.strategy == "class-mean":
            direction, raw_norm = direction_class_mean_fw(policy, **calib)
        elif args.strategy == "skill-mean":
            direction, raw_norm = direction_skill_mean_fw(
                policy, **calib,
                skill_cats=tuple(args.skill_cats.split(",")))
        else:
            direction, raw_norm = head_direction(
                policy.belief, BELIEF2I["rocky"], BELIEF2I["lakes"])
        dir_path = Path(str(args.out_prefix) + f"_steer_dir_{args.strategy}.npy")
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(dir_path, direction.cpu().numpy())
        axis_desc = ("build→mine BEHAVIOR" if args.strategy == "skill-mean"
                     else "lakes→rocky")
        print(f"[{args.strategy}] unit {axis_desc} direction saved to {dir_path} "
              f"(raw ‖Δ‖ before normalization: {raw_norm:.3f})")
        if args.strategy == "skill-mean":
            # entanglement diagnostics: how much of the 'pure behavior' axis is
            # secretly the category/belief axis?
            cat_dir, _ = direction_class_mean_fw(policy, **calib)
            bel_dir, _ = head_direction(policy.belief, BELIEF2I["rocky"],
                                        BELIEF2I["lakes"])
            print(f"[skill-mean] cos(skill axis, category class-mean axis) = "
                  f"{cosine(direction, cat_dir):+.3f}   "
                  f"cos(skill axis, belief-head axis) = "
                  f"{cosine(direction, bel_dir):+.3f}   "
                  f"(build→mine vs lakes→rocky: positive = entangled)")
        if args.strategy == "skill-mean":
            desc = {"balanced": "control, unsteered",
                    "lakes": "pushed toward mine (skill axis)",
                    "rocky": "pushed toward build (skill axis)"}
        else:
            desc = {"balanced": "control, unsteered",
                    "lakes": "steered toward rocky", "rocky": "steered toward lakes"}
    elif args.strategy == "belief-clamp":
        direction, raw_norm = None, None
        print(f"[belief-clamp] gradient-based direct belief steering: clamp "
              f"P(rocky) ≥ {args.belief_floor:g} on lakes maps / "
              f"P(lakes) ≥ {args.belief_floor:g} on rocky maps "
              f"(≤{args.clamp_iters} × α unit steps per env step)")
        desc = {"balanced": "control, unsteered",
                "lakes": "belief clamped to rocky",
                "rocky": "belief clamped to lakes"}
    elif args.strategy == "action-mask":
        direction, raw_norm = None, None
        print(f"[action-mask] hard-masks both build and mine logits to -inf "
              f"(renormalizing over the 4 movement actions) — no magnitude, "
              f"never touches h, so belief cannot move as a side effect")
        desc = {"balanced": "control, unsteered",
                "lakes": "build+mine masked", "rocky": "build+mine masked"}
    else:
        direction, raw_norm = None, None
        print(f"[{args.strategy}] state-dependent behavior clamp: suppress "
              f"π(build) on lakes / π(mine) on rocky below {args.clamp_target:g} "
              f"(≤{args.clamp_iters} × α unit steps per env step, "
              f"push-beta={args.push_beta:g})")
        if args.strategy == "substitute-skill":
            floor_desc = {
                "skill": "π(sub-optimal skill)",
                "movement": "π(movement) [up+down+left+right]",
                "movement-entropy": "H(movement)/log(4) [normalized entropy]",
            }[args.sub_target]
            print(f"    + substitution floor: {floor_desc} kept above "
                  f"{args.sub_floor:g} once {args.sub_from_progress:.0%} of the "
                  f"spawn→target distance is covered")
            desc = {
                "skill": {"balanced": "control, unsteered",
                         "lakes": "build→mine substitution",
                         "rocky": "mine→build substitution"},
                "movement": {"balanced": "control, unsteered",
                            "lakes": "build suppressed + movement floor",
                            "rocky": "mine suppressed + movement floor"},
                "movement-entropy": {"balanced": "control, unsteered",
                                    "lakes": "build suppressed + movement entropy floor",
                                    "rocky": "mine suppressed + movement entropy floor"},
            }[args.sub_target]
        else:
            desc = {"balanced": "control, unsteered",
                    "lakes": "π(build) suppressed", "rocky": "π(mine) suppressed"}

    if args.steer_balanced:
        # balanced now uses the SAME (lakes) convention/alpha — see cat_alpha
        # and make_bt_steerer's steer_balanced docstring
        desc["balanced"] = f"steered toward {args.steer_balanced} (balanced terrain)"

    def steer_factory(alpha_pair):
        def for_category(cat):
            return make_bt_steerer(
                args.strategy, cat, policy, direction, cat_alpha(alpha_pair, cat),
                steer_from=args.steer_from, steer_to=args.steer_to,
                clamp_iters=args.clamp_iters, clamp_target=args.clamp_target,
                push_beta=args.push_beta, sub_floor=args.sub_floor,
                sub_from_progress=args.sub_from_progress,
                sub_target=args.sub_target, sub_stuck_gate=args.sub_stuck_gate,
                sub_stuck_window=args.sub_stuck_window,
                sub_stuck_eps=args.sub_stuck_eps,
                belief_floor=args.belief_floor,
                steer_balanced=(args.steer_balanced or False))
        return for_category

    all_alphas = [(0.0, 0.0)] + alphas
    matrices, skill_matrices, summaries = {}, {}, {}
    beliefs, aprobs = {}, {}
    for a in all_alphas:
        m, skm, s, w, sk, b, ap, pg = run_eval(
            policy, view_size, device, map_factory=map_factory, commit=commit,
            steer_for_category=steer_factory(a), n_maps=args.matrix_maps,
            n_traj=args.matrix_traj, seed_start=args.eval_seed_start,
            max_steps=args.max_steps)
        matrices[a], skill_matrices[a], beliefs[a], aprobs[a] = m, skm, b, ap
        summaries[a] = summarize(m, skm, s, w, sk, b, ap, pg,
                                 late_from=args.sub_from_progress)
        print(f"\nα={alpha_label(a)} door matrix (rows=category, cols={DOORS}) "
              f"+ skill usage (cols={SKILL_CLASSES}):")
        for i, c in enumerate(CATEGORIES):
            sm = summaries[a][c]
            print(f"  {c:9s} {m[i].round(2)} | {skm[i].round(2)}  "
                  f"succ={sm['success']:.2%} wrong={sm['wrong_door']:.2%}  "
                  f"exec b/m per ep={sm['mean_builds']:.1f}/{sm['mean_mines']:.1f}  "
                  f"belief_scalar mean={sm['belief_scalar_mean']:+.3f} "
                  f"final={sm['belief_scalar_final']:+.3f}  "
                  f"argmax={{{', '.join(f'{k}: {v:.0%}' for k, v in sm['belief_argmax_fracs'].items())}}}  "
                  f"π(build)={sm['pi_action_mean']['build']:.4f} "
                  f"π(mine)={sm['pi_action_mean']['mine']:.4f}")

    a1 = alphas[0]
    base_key = (0.0, 0.0)
    print(f"\n── steering effect on the CHOICE (baseline → α={alpha_label(a1)}) ──")
    for cat in ("lakes", "rocky"):
        b0, b1 = summaries[base_key][cat], summaries[a1][cat]
        print(f"  {cat:6s} (α={cat_alpha(a1, cat):g}): "
              f"success {b0['success']:.0%} → {b1['success']:.0%}   "
              f"wrong door {b0['wrong_door']:.0%} → {b1['wrong_door']:.0%}   "
              f"belief scalar {b0['belief_scalar_mean']:+.3f} → "
              f"{b1['belief_scalar_mean']:+.3f}")
        print(f"          executed skills/ep: build {b0['mean_builds']:.1f} → "
              f"{b1['mean_builds']:.1f}   mine {b0['mean_mines']:.1f} → "
              f"{b1['mean_mines']:.1f}")
        if args.strategy in ("suppress-skill", "substitute-skill"):
            pname = ACTION_NAMES[PROHIBIT_ACTION[cat]]
            print(f"          clamp bite: mean π({pname}) "
                  f"{b0['pi_action_mean'][pname]:.4f} → {b1['pi_action_mean'][pname]:.4f}")
        if args.strategy == "belief-clamp":
            tname = CATEGORIES[BELIEF_TARGET[cat]]
            print(f"          belief bite: argmax({tname}) "
                  f"{b0['belief_argmax_fracs'][tname]:.0%} → "
                  f"{b1['belief_argmax_fracs'][tname]:.0%}")
        if args.strategy == "substitute-skill":
            sname = ACTION_NAMES[PUSH_ACTION[cat]]
            l0, l1 = b0["pi_action_mean_late"], b1["pi_action_mean_late"]
            if l0 and l1:
                print(f"          floor bite: mean π({sname}) past "
                      f"{args.sub_from_progress:.0%} progress "
                      f"{l0[sname]:.4f} → {l1[sname]:.4f}")

    plot_choice_matrix_pair(
        [matrices[base_key], matrices[a1]], [summaries[base_key], summaries[a1]],
        ["baseline (α=0)", f"steered α={alpha_label(a1)} ({args.strategy})"],
        f"PPO+GRU fork_wall door choice under hidden-state steering · {tag}",
        Path(str(args.out_prefix) + "_choice_matrix.png"))
    plot_skill_matrix_pair(
        [skill_matrices[base_key], skill_matrices[a1]],
        [summaries[base_key], summaries[a1]],
        ["baseline (α=0)", f"steered α={alpha_label(a1)} ({args.strategy})"],
        f"PPO+GRU fork_wall EXECUTED skills under hidden-state steering · {tag}",
        Path(str(args.out_prefix) + "_skill_matrix.png"))
    plot_belief_traces(beliefs[base_key], beliefs[a1], a1, args.strategy, desc,
                       args.steer_from, args.steer_to,
                       Path(str(args.out_prefix) + "_belief.png"))
    plot_action_probs(aprobs[base_key], aprobs[a1], a1, args.strategy, desc,
                      Path(str(args.out_prefix) + "_actionprob.png"))
    if len(alphas) > 1:
        plot_dose_response_fw(summaries, all_alphas, args.strategy,
                              Path(str(args.out_prefix) + "_dose.png"))
    plot_grid_steered(
        policy, view_size, device, map_factory=map_factory, commit=commit,
        steer_for_category=steer_factory(a1),
        n_seeds=args.grid_seeds, n_traj=args.grid_traj,
        seed_start=args.eval_seed_start, max_steps=args.max_steps,
        title=(f"STEERED (α={alpha_label(a1)}, {args.strategy}) fork_wall · {tag} · "
               f"{args.grid_traj} rollouts/map · green=correct door / red=decoy · "
               f"dots: build=red mine=yellow"),
        out_path=Path(str(args.out_prefix) + "_traj.png"))

    results = {
        "checkpoint": str(args.checkpoint), "strategy": args.strategy,
        "direction_raw_norm": raw_norm, "steer_from": args.steer_from,
        "steer_to": args.steer_to, "clamp_iters": args.clamp_iters,
        "clamp_target": args.clamp_target, "push_beta": args.push_beta,
        "sub_floor": args.sub_floor, "sub_from_progress": args.sub_from_progress,
        "sub_target": args.sub_target, "sub_stuck_gate": args.sub_stuck_gate,
        "sub_stuck_window": args.sub_stuck_window,
        "sub_stuck_eps": args.sub_stuck_eps,
        "belief_floor": args.belief_floor, "passage_half": passage_half,
        "wall_margin": wall_margin, "skill_cats": args.skill_cats,
        "runs": {alpha_label(a): summaries[a] for a in all_alphas},
    }
    res_path = Path(str(args.out_prefix) + "_results.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(results, indent=2))
    print(f"saved {res_path}")


if __name__ == "__main__":
    main()
