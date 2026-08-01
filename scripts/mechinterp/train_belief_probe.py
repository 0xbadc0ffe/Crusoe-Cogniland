#!/usr/bin/env python3
"""Fit a map-category (belief) linear probe on a checkpoint's GRU state and
save it in the shape of a normal ``nn.Linear(H, 3)`` head.

Why: the no-aux agents were never supervised on the map category, so they have
no ``policy.belief``. Every downstream tool — ``steering.py``'s belief-clamp and
``head_direction``, the eval scripts' ``belief_probs`` recording — expects a
linear head. Fitting a probe externally and folding the standardiser into the
weights gives those tools exactly what they expect, while keeping the honest
distinction: this readout is *decoded from* the agent, not *trained into* it.

Methodology matches ``scripts/mechinterp/analysis/probes.py``: multinomial
logistic regression on gru_h with a GROUPED train/test split over map id, so a
probe cannot cheat by memorising individual maps.

    python scripts/mechinterp/train_belief_probe.py \
        --checkpoint outputs/ppo_checkpoints/ppo_gru_forkwall_noaux_seed1/final.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bridge_tunnel"))

from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402
from cogniland.bridge_tunnel.steering import BELIEF2I  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def fold_into_linear(pipe, H, device, scale_floor_frac=0.02):
    """(StandardScaler + LogisticRegression) -> nn.Linear on RAW activations.

    Near-dead GRU units have ~zero variance, so 1/scale_ explodes and a handful
    of units would dominate the raw-space direction even though their
    standardised coefficient is tiny. Floor the scale relative to the population
    median before folding.
    """
    sc, clf = pipe.named_steps["s"], pipe.named_steps["c"]
    scale = np.maximum(sc.scale_, scale_floor_frac * np.median(sc.scale_))
    n_floored = int((scale > sc.scale_ + 1e-12).sum())
    W = clf.coef_ / scale[None, :]
    b = clf.intercept_ - W @ sc.mean_
    if W.shape[0] == 1:                      # binary -> expand to 2 rows
        W = np.vstack([-W[0], W[0]]); b = np.array([-b[0], b[0]])
    lin = nn.Linear(H, W.shape[0]).to(device)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
        lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
    lin.eval()
    return lin, n_floored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="default: <checkpoint dir>/belief_probe.pt")
    p.add_argument("--maps", type=int, default=24, help="maps per category")
    p.add_argument("--traj", type=int, default=6, help="rollouts per map")
    p.add_argument("--seed-start", type=int, default=50_000)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fork_wall = bool(ck["args"].get("fork_wall", False))
    if fork_wall:
        from eval_bridge_tunnel_forkwall import _load_policy
        from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered
    else:
        from eval_bridge_tunnel_commit_ppo import _load_policy
        from eval_bridge_tunnel_commit_ppo_steered import batched_rollout_steered

    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    if getattr(policy, "belief", None) is not None:
        print("NOTE: this checkpoint already HAS a trained belief head; fitting a "
              "probe anyway for comparison, but downstream you probably want the "
              "real head.")
    commit = False if cargs.get("no_commit", False) else None
    gh = cargs.get("goal_half", 0 if fork_wall else 1)
    gh = gh if (gh is not None and gh >= 0) else None

    def make_map(seed, cat):
        kw = dict(size=env_size, width=env_width, seed=seed, category=cat,
                  tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh)
        if fork_wall:
            kw.update(fork_wall=True, passage_half=cargs.get("passage_half", 1),
                      wall_margin=cargs.get("wall_margin", 1))
        return generate_commit_map(**kw)

    X, y, groups = [], [], []
    for ci, cat in enumerate(CATEGORIES):
        for j in range(args.maps):
            seed = args.seed_start + ci * 1000 + j
            rec = make_map(seed, cat)
            kw = dict(steer_fn=None, collect_hidden=True)
            if fork_wall:
                kw["commit"] = commit
            o = batched_rollout_steered(policy, rec, args.traj, view_size,
                                        args.max_steps, device, **kw)
            h = o["hiddens"]
            if h is None or len(h) == 0:
                continue
            X.append(h); y.extend([ci] * len(h)); groups.extend([f"{cat}_{seed}"] * len(h))
        print(f"  collected {cat}", flush=True)

    X = np.concatenate(X).astype(np.float64)
    y = np.asarray(y); groups = np.asarray(groups)
    print(f"dataset: {X.shape[0]} steps x {X.shape[1]} dims over "
          f"{len(set(groups))} maps")

    tr, te = next(GroupShuffleSplit(1, test_size=args.test_frac,
                                    random_state=0).split(X, y, groups))
    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(C=args.C, max_iter=3000))])
    pipe.fit(X[tr], y[tr])
    pred = pipe.predict(X[te])
    acc = float((pred == y[te]).mean())
    bacc = float(balanced_accuracy_score(y[te], pred))
    cm = confusion_matrix(y[te], pred, labels=[0, 1, 2])
    print(f"held-out (map-grouped) accuracy {acc:.3f}  balanced {bacc:.3f}  (chance 0.333)")
    print("confusion (rows=true, cols=pred; balanced/lakes/rocky):")
    print(cm)

    lin, n_floored = fold_into_linear(pipe, X.shape[1], device)
    # sanity: folded head must reproduce the sklearn pipeline's predictions
    with torch.no_grad():
        got = lin(torch.tensor(X[te][:2000], dtype=torch.float32, device=device))
        got = got.argmax(-1).cpu().numpy()
    agree = float((got == pred[:2000]).mean())
    print(f"folded nn.Linear agrees with sklearn on {agree:.1%} of test rows "
          f"({n_floored}/{X.shape[1]} dims scale-floored)")

    out = args.out or args.checkpoint.parent / "belief_probe.pt"
    torch.save({"state_dict": lin.state_dict(),
                "in_features": int(X.shape[1]), "out_features": int(lin.out_features),
                "classes": list(CATEGORIES), "belief2i": BELIEF2I,
                "accuracy": acc, "balanced_accuracy": bacc,
                "confusion": cm.tolist(), "n_scale_floored": n_floored,
                "sklearn_agreement": agree,
                "source_checkpoint": str(args.checkpoint),
                "fit": {"maps": args.maps, "traj": args.traj,
                        "seed_start": args.seed_start, "max_steps": args.max_steps,
                        "test_frac": args.test_frac, "C": args.C,
                        "split": "GroupShuffleSplit over map id"}}, out)
    print(f"saved {out}")


def load_belief_probe(path, device):
    """Load a saved probe as an ``nn.Linear`` ready to assign to policy.belief."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    lin = nn.Linear(d["in_features"], d["out_features"])
    lin.load_state_dict(d["state_dict"])
    return lin.to(device).eval(), d


if __name__ == "__main__":
    main()
