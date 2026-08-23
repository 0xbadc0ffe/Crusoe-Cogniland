#!/usr/bin/env python3
"""'Boxer' / concept-erasure steering on fork_wall: instead of PUSHING along the
build->mine axis (which leaks into belief), PROJECT the activation onto the
manifold where a linear skill classifier reads 'neither' — erase the skill
direction, letting everything else (belief) move freely (INLP/LEACE style).

Two variants:
  A  erase the RAW skill axis w_skill (mine-mean − build-mean).           -> still
     overlaps belief (cos~0.27), so expect some leak.
  B  erase only the BELIEF-ORTHOGONAL part w_perp = w_skill −
     (w_skill·u_bel)u_bel.  Belief axis untouched by construction.

Erasure = clamp the coordinate along the axis to the population-mean projection
(flatten the direction), applied every step. We measure, per category:
  skill neutralization (executed mine/build, π), belief preservation (scalar +
  argmax of the TRUE category), and outcome (success / wrong-door / no-door).

  python scripts/bridge_tunnel/forkwall_skill_erase.py \
      --checkpoint released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt
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
from cogniland.bridge_tunnel.mapgen import CATEGORIES  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from eval_bridge_tunnel_forkwall_steered import (  # noqa: E402
    batched_rollout_steered, direction_skill_mean_fw, direction_class_mean_fw)
from eval_bridge_tunnel_commit_ppo_steered import BELIEF2I, A_BUILD, A_MINE  # noqa: E402

CATS = ["rocky", "lakes", "balanced"]


def make_erase_fn(w_unit, c):
    wt = w_unit
    cc = float(c)

    def fn(h, t, info):
        x = h.squeeze(0)                      # (B, H)
        proj = x @ wt                         # (B,)
        x = x - (proj - cc).unsqueeze(-1) * wt.unsqueeze(0)
        return x.unsqueeze(0)
    return fn


def agg(policy, mk, cat, steer_fn, view_size, device, commit, n_maps, n_traj,
        seed0, max_steps):
    bidx = BELIEF2I[cat]; li, ri = BELIEF2I["lakes"], BELIEF2I["rocky"]
    pm, pb, em, eb, sc, arg, succ, wrong, none = ([] for _ in range(9))
    for j in range(n_maps):
        rec = mk(seed0 + j, cat)
        out = batched_rollout_steered(policy, rec, n_traj, view_size, max_steps,
                                      device, commit=commit, steer_fn=steer_fn)
        ap = out["action_probs"]; bp = out["belief_probs"]
        va = np.isfinite(ap[..., 0])
        pm.append(ap[..., A_MINE][va]); pb.append(ap[..., A_BUILD][va])
        em += out["n_mines"].tolist(); eb += out["n_builds"].tolist()
        sc.append((bp[..., li] - bp[..., ri])[np.isfinite(bp[..., 0])])
        for i in range(bp.shape[0]):
            v = np.where(np.isfinite(bp[i, :, 0]))[0]
            if len(v):
                arg.append(int(bp[i, v[-1]].argmax()) == bidx)
        succ += out["success"].tolist()
        wrong += (out["reached_any"] & ~out["success"]).tolist()
        none += (~out["reached_any"]).tolist()
    return dict(pi_mine=float(np.concatenate(pm).mean()),
                pi_build=float(np.concatenate(pb).mean()),
                exec_mine=float(np.mean(em)), exec_build=float(np.mean(eb)),
                belief_scalar=float(np.concatenate(sc).mean()),
                belief_true=float(np.mean(arg)),
                success=float(np.mean(succ)), wrong=float(np.mean(wrong)),
                none=float(np.mean(none)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt"))
    ap.add_argument("--maps", type=int, default=10)
    ap.add_argument("--traj", type=int, default=12)
    ap.add_argument("--calib-maps", type=int, default=8)
    ap.add_argument("--calib-traj", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--calib-seed", type=int, default=20_000)
    ap.add_argument("--out", default="outputs/bridge_tunnel_forkwall/skill_erase.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    policy, cargs, view_size, env_size, env_width = _load_policy(a.checkpoint, device)
    commit = False if cargs.get("no_commit", False) else None
    ph = cargs.get("passage_half", 1); wm = cargs.get("wall_margin", 1)
    gh = cargs.get("goal_half", 0); gh = gh if (gh is not None and gh >= 0) else None

    def mk(seed, cat):
        return generate_commit_map(size=env_size, width=env_width, seed=seed, category=cat,
                                   tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                   fork_wall=True, passage_half=ph, wall_margin=wm)

    calib = dict(map_factory=mk, view_size=view_size, device=device, commit=commit,
                 n_maps=a.calib_maps, n_traj=a.calib_traj, seed_start=a.calib_seed,
                 max_steps=a.max_steps)
    w_skill, _ = direction_skill_mean_fw(policy, **calib, skill_cats=("balanced",))
    u_bel, _ = direction_class_mean_fw(policy, **calib)
    cos = float(w_skill @ u_bel)
    w_perp = w_skill - (w_skill @ u_bel) * u_bel
    w_perp = w_perp / w_perp.norm()
    print(f"[erase] cos(skill, belief)={cos:+.3f}  "
          f"‖perp‖/‖skill‖={float((w_skill - (w_skill@u_bel)*u_bel).norm()):.3f} "
          f"(fraction of skill axis orthogonal to belief)")

    # neutral targets = population-mean projection over baseline states
    Hs = []
    for cat in CATS:
        for j in range(3):
            out = batched_rollout_steered(policy, mk(a.calib_seed + 100 + j, cat), 8,
                                          view_size, a.max_steps, device, commit=commit,
                                          collect_hidden=True)
            if out["hiddens"] is not None:
                Hs.append(out["hiddens"])
    Hs = torch.from_numpy(np.concatenate(Hs)).to(device)
    c_skill = float((Hs @ w_skill).mean()); c_perp = float((Hs @ w_perp).mean())

    conds = {"baseline": None,
             "erase-raw (A)": make_erase_fn(w_skill, c_skill),
             "erase-⊥belief (B)": make_erase_fn(w_perp, c_perp)}

    res = {}
    for name, fn in conds.items():
        res[name] = {}
        for cat in CATS:
            res[name][cat] = agg(policy, mk, cat, fn, view_size, device, commit,
                                 a.maps, a.traj, a.eval_seed, a.max_steps)
        print(f"\n[{name}]")
        for cat in CATS:
            r = res[name][cat]
            print(f"  {cat:9s} succ={r['success']:.0%} wrong={r['wrong']:.0%} "
                  f"none={r['none']:.0%} | exec mine={r['exec_mine']:.1f} "
                  f"build={r['exec_build']:.1f} | π mine={r['pi_mine']:.3f} "
                  f"build={r['pi_build']:.3f} | belief(true)={r['belief_true']:.0%} "
                  f"scalar={r['belief_scalar']:+.2f}")

    # ── figure: rocky & lakes rows; skill / belief / outcome columns ─────────
    names = list(conds); x = np.arange(len(names))
    col = ["#888", "#d62728", "#2ca02c"]
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 6.6))
    for ri, cat in enumerate(["rocky", "lakes"]):
        skill_key = "exec_mine" if cat == "rocky" else "exec_build"
        sk = [res[n][cat][skill_key] for n in names]
        bt = [res[n][cat]["belief_true"] for n in names]
        sc = [res[n][cat]["success"] for n in names]
        wr = [res[n][cat]["wrong"] for n in names]
        axs[ri, 0].bar(x, sk, color=col)
        axs[ri, 0].set_title(f"{cat}: executed {skill_key.split('_')[1]} / ep (behavior)")
        axs[ri, 0].set_ylabel("count/ep")
        axs[ri, 1].bar(x, bt, color=col)
        axs[ri, 1].axhline(1/3, ls=":", c="#999"); axs[ri, 1].set_ylim(0, 1)
        axs[ri, 1].set_title(f"{cat}: P(argmax = TRUE cat) (belief kept?)")
        axs[ri, 2].bar(x - 0.18, sc, width=0.36, color="#2ca02c", label="success")
        axs[ri, 2].bar(x + 0.18, wr, width=0.36, color="#d62728", label="wrong door")
        axs[ri, 2].set_ylim(0, 1); axs[ri, 2].set_title(f"{cat}: outcome")
        axs[ri, 2].legend(fontsize=8)
        for k in range(3):
            axs[ri, k].set_xticks(x); axs[ri, k].set_xticklabels(names, fontsize=8, rotation=12)
    fig.suptitle(f"fork_wall 'boxer' skill erasure — can we suppress the skill WITHOUT "
                 f"the belief leak?  (cos(skill,belief)={cos:+.2f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
