#!/usr/bin/env python3
"""Curate a set of BALANCED fork_wall maps that make a clean steering substrate.

Two properties are wanted, and generic balanced seeds have neither:

  (1) the baseline agent's belief is STABLE — it ends the episode calling the
      map "balanced" on nearly every rollout (flip rate < --max-flip). Maps
      whose realized terrain is lopsided enough that the agent honestly reads
      them as lakes/rocky are excluded, so any belief movement under steering
      is attributable to the intervention rather than to the map.

  (2) the set is DOOR-BALANCED — pooled over maps the baseline picks top about
      half the time, built from maps that are individually *decisive* (each
      leans strongly to one door). On balanced terrain either door pays, so the
      agent is free to have a per-map habit; a set whose habits cancel means a
      steering-induced door shift can't be an artifact of a global prior.

Screens --candidates seeds, then greedily assembles the largest door-balanced
subset from the decisive survivors.

    python scripts/mechinterp/select_balanced_maps.py \
        --checkpoint outputs/ppo_checkpoints/ppo_gru_forkwall_noaux_seed1/final.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bridge_tunnel"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.mapgen import generate_commit_map  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered  # noqa: E402
from cogniland.bridge_tunnel.steering import BELIEF2I  # noqa: E402
from train_belief_probe import load_belief_probe  # noqa: E402


def screen(policy, rec, n_traj, view_size, max_steps, device, commit):
    o = batched_rollout_steered(policy, rec, n_traj, view_size, max_steps, device,
                                commit=commit, steer_fn=None)
    bp = o["belief_probs"]
    valid = np.isfinite(bp[..., 0])
    finals = []
    for i in range(bp.shape[0]):
        v = valid[i]
        if v.any():
            finals.append(int(bp[i, int(np.where(v)[0][-1])].argmax()))
    finals = np.asarray(finals)
    doors = list(o["doors"])
    n = max(len(doors), 1)
    return {
        "flip_rate": float((finals != BELIEF2I["balanced"]).mean()) if len(finals) else 1.0,
        "top_frac": float(sum(d == "top" for d in doors) / n),
        "bottom_frac": float(sum(d == "bottom" for d in doors) / n),
        "none_frac": float(sum(d == "none" for d in doors) / n),
        "success": float(np.mean(o["success"])),
        "mean_builds": float(np.mean(o["n_builds"])),
        "mean_mines": float(np.mean(o["n_mines"])),
    }


def pick_balanced_subset(cands, target=0.5, tol=0.02, decisive=0.7):
    """LARGEST subset of decisive maps whose pooled top-fraction is approx target.

    Equal counts of each side generally cannot hit 0.5: the bottom-leaning maps
    sit near 0.06 rather than 0.0, so a symmetric pick lands around 0.53. Search
    (n_top, n_bot) independently and maximise subset size subject to the
    tolerance -- the two sides end up slightly unequal, which is the point.
    """
    tops = sorted([c for c in cands if c["top_frac"] >= decisive],
                  key=lambda c: -c["top_frac"])
    bots = sorted([c for c in cands if c["top_frac"] <= 1 - decisive],
                  key=lambda c: c["top_frac"])
    if not tops or not bots:
        return [], float("nan")
    ct = np.cumsum([c["top_frac"] for c in tops])
    cb = np.cumsum([c["top_frac"] for c in bots])
    best, best_key = None, None
    for nt in range(1, len(tops) + 1):
        for nb in range(1, len(bots) + 1):
            pooled = (ct[nt - 1] + cb[nb - 1]) / (nt + nb)
            if abs(pooled - target) > tol:
                continue
            key = (nt + nb, -abs(pooled - target))   # more maps, then tighter
            if best_key is None or key > best_key:
                best_key, best = key, (tops[:nt] + bots[:nb], float(pooled))
    if best is None:
        n = min(len(tops), len(bots))
        sel = tops[:n] + bots[:n]
        return sel, float(np.mean([c["top_frac"] for c in sel]))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=REPO / "outputs/ppo_checkpoints/ppo_gru_forkwall_noaux_seed1/final.pt")
    p.add_argument("--probe", type=Path, default=None)
    p.add_argument("--candidates", type=int, default=220, help="balanced seeds to screen")
    p.add_argument("--seed-start", type=int, default=200_000,
                   help="disjoint from every other eval/calibration range in the project")
    p.add_argument("--traj", type=int, default=16, help="screening rollouts per map")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--max-flip", type=float, default=0.05,
                   help="reject maps whose baseline belief flips more often than this")
    p.add_argument("--decisive", type=float, default=0.7,
                   help="a map counts as top-leaning at >= this, bottom-leaning at <= 1 - this")
    p.add_argument("--max-none", type=float, default=0.05,
                   help="reject maps the baseline fails to solve (timeouts)")
    p.add_argument("--out", type=Path,
                   default=REPO / "data/bridge_tunnel/forkwall_balanced_clean.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    if getattr(policy, "belief", None) is None:
        lin, pmeta = load_belief_probe(
            args.probe or args.checkpoint.parent / "belief_probe.pt", device)
        policy.belief = lin
        print(f"attached belief probe (balanced acc {pmeta['balanced_accuracy']:.3f})")
    commit = False if cargs.get("no_commit", False) else None
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    torch.manual_seed(0)

    cands = []
    for j in range(args.candidates):
        seed = args.seed_start + j
        rec = generate_commit_map(size=env_size, width=env_width, seed=seed,
                                  category="balanced", tree_frac=cargs.get("tree_frac", 0.03),
                                  goal_half=gh, fork_wall=True,
                                  passage_half=cargs.get("passage_half", 1),
                                  wall_margin=cargs.get("wall_margin", 1))
        st = screen(policy, rec, args.traj, view_size, args.max_steps, device, commit)
        st["seed"] = int(seed)
        st["n_water"] = int((rec.terrain == T.WATER).sum())
        st["n_rock"] = int((rec.terrain == T.ROCK).sum())
        st["rock_minus_water"] = st["n_rock"] - st["n_water"]
        cands.append(st)
        if (j + 1) % 25 == 0:
            print(f"  screened {j+1}/{args.candidates}", flush=True)

    ok = [c for c in cands
          if c["flip_rate"] <= args.max_flip and c["none_frac"] <= args.max_none]
    dec = [c for c in ok if c["top_frac"] >= args.decisive or c["top_frac"] <= 1 - args.decisive]
    print(f"\nscreened {len(cands)} · stable belief & solvable: {len(ok)} · "
          f"also door-decisive: {len(dec)}")

    sel, pooled = pick_balanced_subset(dec)
    if not sel:
        raise SystemExit("no door-balanced subset found — loosen --decisive or "
                         "raise --candidates")
    sel = sorted(sel, key=lambda c: -c["top_frac"])
    n_top = sum(c["top_frac"] >= 0.5 for c in sel)

    print(f"\nselected {len(sel)} maps · pooled top-door {pooled:.1%} "
          f"({n_top} top-leaning / {len(sel)-n_top} bottom-leaning)")
    print(f"  mean belief-flip {np.mean([c['flip_rate'] for c in sel]):.2%}  "
          f"max {max(c['flip_rate'] for c in sel):.2%}")
    print(f"  mean success {np.mean([c['success'] for c in sel]):.1%}")
    print(f"\n{'seed':>8s} {'top%':>6s} {'flip%':>6s} {'succ%':>6s} "
          f"{'water':>6s} {'rock':>5s} {'rock-water':>11s} {'builds':>7s} {'mines':>6s}")
    for c in sel:
        print(f"{c['seed']:>8d} {c['top_frac']:>6.0%} {c['flip_rate']:>6.0%} "
              f"{c['success']:>6.0%} {c['n_water']:>6d} {c['n_rock']:>5d} "
              f"{c['rock_minus_water']:>+11d} {c['mean_builds']:>7.1f} {c['mean_mines']:>6.1f}")

    rmw = np.array([c["rock_minus_water"] for c in sel], dtype=float)
    tf = np.array([c["top_frac"] for c in sel], dtype=float)
    if rmw.std() > 0 and tf.std() > 0:
        print(f"\ncorr(rock−water, top-door lean) = {np.corrcoef(rmw, tf)[0,1]:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "criteria": {"max_flip": args.max_flip, "decisive": args.decisive,
                     "max_none": args.max_none, "screen_traj": args.traj,
                     "max_steps": args.max_steps},
        "screened": len(cands), "n_stable": len(ok), "n_decisive": len(dec),
        "pooled_top_frac": pooled, "n_selected": len(sel),
        "seeds": [c["seed"] for c in sel],
        "selected": sel, "all_candidates": cands,
    }, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
