#!/usr/bin/env python3
"""Reproduce & STEER a dataset trajectory — standalone (this folder + a checkpoint).

NO cogniland / repo needed: uses the bundle's maps.npz + labels + manifest, plus
the sibling env_min.py / policy_min.py. Reconstructs an exact episode and (with
--inject) adds a steering vector to the GRU hidden ``gru_h`` over a step range, so
you can test belief/skill steering (e.g. suppressing 'mine').

    # verify exact reproduction of a stored trajectory
    python steer.py --checkpoint agent.pt --map-id 30 --traj-seed 11000000000

    # steer: add alpha*vec to gru_h on steps [a,b)
    python steer.py --checkpoint agent.pt --map-id 30 --traj-seed 11000000000 \
        --inject mine_dir.npy --alpha -6 --rows 20:80
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from env_min import MiniBridgeTunnelEnv
from policy_min import PPOGRUPolicy

HERE = Path(__file__).resolve().parent
_ANAMES_BT = ["up", "down", "left", "right", "place", "mine"]
_ANAMES_BTC = ["up", "down", "left", "right", "build", "mine"]


def _labels(d):
    import pandas as pd
    p = d / "labels.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(d / "labels.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=HERE, help="bundle dir (default: next to this script)")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--map-id", type=int, required=True)
    ap.add_argument("--traj-seed", type=int, required=True)
    ap.add_argument("--inject", type=Path, default=None, help=".npy vector (H,) added to gru_h")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rows", default=None, help="step range a:b for injection (default all)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    d = args.dataset
    man = json.loads((d / "manifest.json").read_text())
    variant = "btc" if man.get("is_commit") else "bt"
    view = man["view_size"]
    max_steps = int(man.get("max_steps", 800))
    maps = np.load(d / "maps.npz", allow_pickle=True)
    terrain = maps["terrain"][args.map_id]; spawn = maps["spawn"][args.map_id]
    n_scalars = 7 if variant == "btc" else 5
    anames = _ANAMES_BTC if variant == "btc" else _ANAMES_BT

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pol = PPOGRUPolicy.from_checkpoint(ckpt, view, n_scalars, device=args.device)

    vec = None
    if args.inject is not None:
        vec = torch.from_numpy(np.load(args.inject).astype(np.float32)).view(1, 1, -1)
    a0, b0 = (0, 10 ** 9)
    if args.rows:
        a0, b0 = (int(x) for x in args.rows.split(":"))

    env = MiniBridgeTunnelEnv(terrain, spawn, variant=variant, view_size=view, max_steps=max_steps)
    torch.manual_seed(args.traj_seed)                      # the ONLY randomness
    obs = env.reset()
    h = torch.zeros(1, 1, pol.gru_hidden, device=args.device)
    acts, commits, gru = [], [], []
    with torch.no_grad():
        for t in range(max_steps):
            o = {k: torch.from_numpy(np.asarray(v)[None]).to(args.device) for k, v in obs.items()}
            inj = (args.alpha * vec) if (vec is not None and a0 <= t < b0) else None
            logits, _, h = pol.step(o, h, inject=inj)
            gru.append(h.squeeze().cpu().numpy())
            a = int(torch.distributions.Categorical(logits=logits).sample()[0])
            acts.append(a); commits.append(env.commit)
            obs, reached, done = env.step(a)
            if done:
                break
    name = {0: "none", 1: "build", 2: "mine"}
    print(f"variant={variant} steps={len(acts)} reached={reached} "
          f"final_commit={name[commits[-1]]}")
    print("actions:", [anames[a] for a in acts[:40]] + (["..."] if len(acts) > 40 else []))

    if vec is None:
        # verify byte-for-byte reproduction against the stored labels
        try:
            lab = _labels(d)
            sub = lab[(lab.map_id == args.map_id) & (lab.traj_seed == args.traj_seed)].sort_values("t")
            stored = sub["action"].tolist()
            ok = stored == acts[:len(stored)] and len(stored) == len(acts)
            print(f"reproduction vs dataset: {'EXACT MATCH ✓' if ok else 'MISMATCH ✗'} ({len(stored)} steps)")
        except Exception as e:  # noqa: BLE001
            print(f"(could not verify against labels: {e})")
    else:
        print(f"(steered: +{args.alpha}*{args.inject.name} on gru_h steps [{a0}:{b0}))")


if __name__ == "__main__":
    main()
