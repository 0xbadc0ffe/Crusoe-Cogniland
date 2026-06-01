#!/usr/bin/env python3
"""Build a mechanistic-interpretability activation dataset for a bridge_tunnel PPO+GRU
agent: probe BELIEF (obstacle ahead, type/size, cost-to-go) and STRATEGY
(avoid / bridge / tunnel) subspaces, and supply matched contrast sets for
difference-of-means steering (e.g. suppressing tunnelling).

Recipe: hold 1-2 maps FIXED, sample many stochastic rollouts, and log per step:
  * activations           — gru_h (128, the recurrent belief carrier) + enc_embed (256)
  * the FULL observation  — egocentric minimap (V,V int8) + scalars (5) so any
                            frame can be re-rendered offline
  * policy outputs        — action, action_probs (6), value
  * BELIEF labels (env truth) — ctg_to_goal, compass, obstacle-ahead type/dist,
                            in_obstacle, nearest lake/mountain size
  * STRATEGY labels       — per-step segment {free, approach, avoid, bridge,
                            tunnel} via the strict in->out crossing rule, plus a
                            per-decision table (one row per obstacle the traj
                            interacted with).

REPRODUCIBILITY: each trajectory is fully determined by its ``traj_seed`` (the
map is fixed; the only randomness is action sampling, seeded per trajectory).
Re-running ``torch.manual_seed(traj_seed)`` + reset on the same map on the same
device reproduces the EXACT trajectory, observations and sampled actions.

    python scripts/build_activation_dataset.py \\
        --checkpoint models/bridge_tunnel/natural_centergoal3_onehot.pt \\
        --maps 10000,10001 --n-traj 500 --out-dir data/mechinterp/ppo_onehot

Outputs (under --out-dir):
  activations.h5     gru_h, enc_embed, minimap, scalars, action_probs, value  (row-aligned)
  labels.parquet     one row per step (row_id PK) — keys + belief + strategy
  decisions.parquet  one row per (traj, obstacle) decision — steering contrast sets
  manifest.json      agent, sites/dims, maps (seeds+kwargs), config, reproduction recipe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as cc_label, distance_transform_cdt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel import generate_bridge_tunnel_map, tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from train_ppo_bridge_tunnel import PPOGRUPolicy  # noqa: E402

_FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
_FACE_NAME = {0: "up", 1: "down", 2: "left", 3: "right"}
_ACTION_NAME = ["up", "down", "left", "right", "place", "mine"]
_OBSTACLE = {T.WATER: "water", T.ROCK: "rock", T.TREE: "tree"}
_WALKABLE = (T.GRASS, T.WOOD, T.TARGET, T.SAND, T.DIRT)


# ───────────────────────── per-map precompute ─────────────────────────

def _map_geometry(rec, min_body=4):
    """Precompute everything we need to label belief/strategy on this map."""
    orig = np.asarray(rec.terrain)
    H, W = orig.shape
    target = tuple(rec.target)
    ctg = BridgeTunnelEnv._compute_ctg(orig, target).astype(np.float32)

    geo = {"orig": orig, "H": H, "W": W, "target": target, "ctg": ctg, "bodies": {}}
    for tile, name in ((T.WATER, "water"), (T.ROCK, "rock")):
        lbl, n = cc_label(orig == tile)
        size_map = np.zeros((H, W), np.int32)
        bodies = {}
        for bid in range(1, n + 1):
            mask = lbl == bid
            sz = int(mask.sum())
            size_map[mask] = sz
            if sz >= min_body:
                # taxicab distance field to this body (0 on the body)
                dist = distance_transform_cdt(~mask, metric="taxicab").astype(np.int32)
                bodies[bid] = {"size": sz, "cells": mask, "dist": dist}
        geo[f"{name}_lbl"] = lbl
        geo[f"{name}_size"] = size_map
        geo["bodies"][name] = bodies
    return geo


def _belief_row(geo, pos, facing, max_scan=14):
    """Ground-truth belief labels at (pos, facing) from the static map."""
    orig, H, W, ctg, target = geo["orig"], geo["H"], geo["W"], geo["ctg"], geo["target"]
    r, c = pos
    cur = int(orig[r, c])
    # scan ahead in the facing direction for the first obstacle tile
    dr, dc = _FACE_DELTA[facing]
    ahead_type, ahead_dist = "none", -1
    for d in range(1, max_scan + 1):
        rr, cc = r + dr * d, c + dc * d
        if not (0 <= rr < H and 0 <= cc < W):
            break
        t = int(orig[rr, cc])
        if t in _OBSTACLE:
            ahead_type, ahead_dist = _OBSTACLE[t], d
            break
        if t not in _WALKABLE:           # e.g. an unexpected wall
            break
    return {
        "cur_tile": cur,
        "in_obstacle": cur in (T.WATER, T.ROCK),
        "ctg_to_goal": float(ctg[r, c]),
        "compass_dr": float(np.sign(target[0] - r)),
        "compass_dc": float(np.sign(target[1] - c)),
        "obst_ahead_type": ahead_type,
        "obst_ahead_dist": ahead_dist,
        "lake_size_here": int(geo["water_size"][r, c]),
        "mtn_size_here": int(geo["rock_size"][r, c]),
    }


# ───────────────────── strict crossing / strategy ─────────────────────

def _crossings(path, orig, lbl, tile, min_cross):
    """Maximal in->out runs of consecutive path cells whose ORIGINAL terrain is
    ``tile`` (>= min_cross distinct cells, distinct entry/exit land). Returns
    list of {body_id, n_cells, start_idx, end_idx} (path indices)."""
    out, n, i = [], len(path), 0
    while i < n:
        if orig[path[i]] == tile:
            j = i
            while j < n and orig[path[j]] == tile:
                j += 1
            distinct = list(dict.fromkeys(path[i:j]))
            entry = path[i - 1] if i > 0 else None
            exit_ = path[j] if j < n else None
            if len(distinct) >= min_cross and entry is not None and exit_ is not None and entry != exit_:
                out.append({"body_id": int(lbl[path[i]]), "n_cells": len(distinct),
                            "start_idx": i, "end_idx": j - 1})
            i = j
        else:
            i += 1
    return out


def _strategy(path, geo, map_id, traj_id, traj_seed, min_cross, approach_window, near_radius):
    """Per-step segment labels + per-decision rows for one trajectory."""
    orig = geo["orig"]
    n = len(path)
    seg = ["free"] * n
    decision_id = [""] * n
    decisions = []
    crossed = {"water": set(), "rock": set()}

    # 1) crossings → bridge / tunnel (+ approach window)
    for tile, name, choice in ((T.WATER, "water", "bridge"), (T.ROCK, "rock", "tunnel")):
        for cr in _crossings(path, orig, geo[f"{name}_lbl"], tile, min_cross):
            bid = cr["body_id"]; crossed[name].add(bid)
            did = f"{map_id}:{traj_id}:{name}:{bid}"
            a0 = max(0, cr["start_idx"] - approach_window)
            for k in range(a0, cr["start_idx"]):
                if seg[k] == "free":
                    seg[k], decision_id[k] = "approach", did
            for k in range(cr["start_idx"], cr["end_idx"] + 1):
                seg[k], decision_id[k] = choice, did
            decisions.append({
                "decision_id": did, "map_id": map_id, "traj_id": traj_id,
                "traj_seed": traj_seed, "obstacle_type": name,
                "body_size": geo["bodies"][name].get(bid, {}).get("size", cr["n_cells"]),
                "choice": choice, "crossing_cells": cr["n_cells"],
                "decision_step": cr["start_idx"],
            })

    # 2) approached-but-not-crossed bodies → avoid (closest-approach step)
    path_arr = np.array(path)
    for name in ("water", "rock"):
        for bid, b in geo["bodies"][name].items():
            if bid in crossed[name]:
                continue
            d = b["dist"][path_arr[:, 0], path_arr[:, 1]]
            if d.min() > near_radius:
                continue
            did = f"{map_id}:{traj_id}:{name}:{bid}"
            near = np.where(d <= near_radius)[0]
            for k in near:
                if seg[k] == "free":
                    seg[k], decision_id[k] = "avoid", did
            decisions.append({
                "decision_id": did, "map_id": map_id, "traj_id": traj_id,
                "traj_seed": traj_seed, "obstacle_type": name, "body_size": b["size"],
                "choice": "avoid", "crossing_cells": 0,
                "decision_step": int(d.argmin()),
            })
    return seg, decision_id, decisions


# ───────────────────────────── rollout ────────────────────────────────

@torch.no_grad()
def _rollout(policy, rec, geo, map_id, traj_id, traj_seed, view, max_steps, device,
             min_cross, approach_window, near_radius):
    """One reproducible stochastic rollout; returns per-step records + decisions."""
    torch.manual_seed(traj_seed)              # <- the ONLY randomness (action sampling)
    env = BridgeTunnelEnv(map_record=rec, size=rec.terrain.shape[0], width=rec.terrain.shape[1],
                      view_size=view, max_steps=max_steps)
    obs = env.reset()[0]
    h = torch.zeros(1, 1, policy.gru_hidden, device=device)
    done = torch.zeros(1, device=device)

    # capture enc_embed (256) via a forward hook on the embed MLP
    cap = {}
    hk = policy.embed.register_forward_hook(lambda m, i, o: cap.__setitem__("embed", o.detach()))

    rows, path = [], []
    reached = False
    try:
        for t in range(max_steps):
            mm = torch.from_numpy(obs["minimap"])[None, None].to(device)
            sc = torch.from_numpy(obs["scalars"])[None, None].to(device)
            gru_out, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
            logits, value = policy._heads(gru_out.squeeze(0))
            probs = torch.softmax(logits, dim=-1)[0]
            a = int(torch.distributions.Categorical(logits=logits).sample()[0])
            pos = (int(env._pos[0]), int(env._pos[1])); facing = int(env._facing)
            path.append(pos)
            rec_row = {
                "map_id": map_id, "map_seed": geo["map_seed"], "traj_id": traj_id,
                "traj_seed": traj_seed, "t": t, "pos_r": pos[0], "pos_c": pos[1],
                "facing": facing, "facing_name": _FACE_NAME[facing],
                "action": a, "action_name": _ACTION_NAME[a], "value": float(value[0]),
                # activations + full obs are stacked into arrays after the loop:
                "_gru_h": gru_out.squeeze().cpu().numpy().astype(np.float16),
                "_enc_embed": cap["embed"].squeeze().cpu().numpy().astype(np.float16),
                "_minimap": obs["minimap"].astype(np.int8),
                "_scalars": obs["scalars"].astype(np.float16),
                "_probs": probs.cpu().numpy().astype(np.float16),
            }
            rec_row.update(_belief_row(geo, pos, facing))
            rows.append(rec_row)
            obs, r, term, trunc, info = env.step(a)
            if term:
                reached = True; break
            if trunc:
                break
    finally:
        hk.remove()

    seg, did, decisions = _strategy(path, geo, map_id, traj_id, traj_seed,
                                    min_cross, approach_window, near_radius)
    ep_len = len(rows)
    for k, row in enumerate(rows):
        row["segment"] = seg[k]; row["decision_id"] = did[k]
        row["reached"] = reached; row["ep_len"] = ep_len
    return rows, decisions


# ───────────────────────────────── main ───────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("models/bridge_tunnel/natural_centergoal3_onehot.pt"))
    p.add_argument("--maps", default="10000,10001", help="comma-sep map seeds (held fixed)")
    p.add_argument("--n-traj", type=int, default=500, help="stochastic rollouts per map")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--min-cross", type=int, default=2)
    p.add_argument("--approach-window", type=int, default=8)
    p.add_argument("--near-radius", type=int, default=3)
    p.add_argument("--device", default="cpu", help="cpu = exactly reproducible across machines")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    env_size = cargs.get("env_size", 32); env_width = cargs.get("env_width") or env_size
    view = cargs.get("view_size", 21); device = torch.device(args.device)
    natkw = dict(size=env_size, width=env_width, orientation=cargs.get("orientation", "natural"),
                 water_frac=cargs.get("water_frac", 0.14), rock_frac=cargs.get("rock_frac", 0.14),
                 tree_frac=cargs.get("tree_frac", 0.03),
                 goal_half=(cargs["goal_half"] if cargs.get("goal_half", -1) is not None
                            and cargs.get("goal_half", -1) >= 0 else None))

    # rebuild policy (handles onehot / embed checkpoints)
    sd = ckpt["policy"]; obs_enc = cargs.get("obs_encoding", "embed")
    if "tile_embed.weight" in sd:
        n_tiles = int(sd["tile_embed.weight"].shape[0])
    else:
        n_tiles = int(sd["cnn.0.weight"].shape[1]) - 2; obs_enc = "onehot"
    n_act = int(sd["actor.weight"].shape[0])
    dummy = BridgeTunnelEnv(size=env_size, width=env_width, view_size=view); dummy.reset()
    policy = PPOGRUPolicy(dummy.observation_space, num_actions=n_act,
                          gru_hidden=cargs.get("gru_hidden", 128),
                          embed_dim=cargs.get("embed_dim", 256),
                          num_tile_classes=n_tiles, obs_encoding=obs_enc).to(device)
    policy.load_state_dict(sd); policy.eval()

    map_seeds = [int(s) for s in args.maps.split(",")]
    out_dir = args.out_dir or Path("data/mechinterp") / args.checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, all_dec = [], []
    for map_id, ms in enumerate(map_seeds):
        rec = generate_bridge_tunnel_map(seed=ms, **natkw)
        geo = _map_geometry(rec); geo["map_seed"] = ms
        print(f"[map {map_id}] seed {ms}: water bodies={len(geo['bodies']['water'])} "
              f"rock bodies={len(geo['bodies']['rock'])} — {args.n_traj} rollouts...", flush=True)
        for traj_id in range(args.n_traj):
            traj_seed = ms * 100_000 + traj_id          # unique + reproducible
            rows, dec = _rollout(policy, rec, geo, map_id, traj_id, traj_seed, view,
                                 args.max_steps, device, args.min_cross,
                                 args.approach_window, args.near_radius)
            all_rows.extend(rows); all_dec.extend(dec)

    N = len(all_rows)
    for i, row in enumerate(all_rows):
        row["row_id"] = i

    # ── activations + full obs → activations.h5 (or .npz fallback) ──
    arrays = {
        "row_id": np.arange(N, dtype=np.int64),
        "gru_h": np.stack([r.pop("_gru_h") for r in all_rows]),
        "enc_embed": np.stack([r.pop("_enc_embed") for r in all_rows]),
        "minimap": np.stack([r.pop("_minimap") for r in all_rows]),
        "scalars": np.stack([r.pop("_scalars") for r in all_rows]),
        "action_probs": np.stack([r.pop("_probs") for r in all_rows]),
    }
    try:
        import h5py
        with h5py.File(out_dir / "activations.h5", "w") as f:
            for k, v in arrays.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        act_path = out_dir / "activations.h5"
    except Exception as e:   # noqa: BLE001
        print(f"  [warn] h5py unavailable ({e!r}); writing activations.npz")
        np.savez_compressed(out_dir / "activations.npz", **arrays)
        act_path = out_dir / "activations.npz"

    # ── labels + decisions → parquet (or csv fallback) ──
    import pandas as pd
    labels = pd.DataFrame(all_rows)
    decisions = pd.DataFrame(all_dec)
    def _save(df, stem):
        try:
            df.to_parquet(out_dir / f"{stem}.parquet"); return f"{stem}.parquet"
        except Exception as e:   # noqa: BLE001
            print(f"  [warn] parquet failed ({e!r}); writing {stem}.csv")
            df.to_csv(out_dir / f"{stem}.csv", index=False); return f"{stem}.csv"
    lp, dp = _save(labels, "labels"), _save(decisions, "decisions")

    manifest = {
        "checkpoint": str(args.checkpoint), "obs_encoding": obs_enc, "n_tiles": n_tiles,
        "view_size": view, "natural_kwargs": natkw, "map_seeds": map_seeds,
        "n_traj_per_map": args.n_traj, "n_rows": N, "n_decisions": len(all_dec),
        "activation_sites": {"gru_h": 128, "enc_embed": 256},
        "obs_stored": {"minimap": [view, view], "scalars": 5},
        "dtype": "float16 (activations/scalars/probs), int8 (minimap)",
        "reproduce": "torch.manual_seed(traj_seed); BridgeTunnelEnv(map_record=generate_bridge_tunnel_map("
                     "seed=map_seed, **natural_kwargs)).reset(); then sample actions in order "
                     f"on device={args.device}.",
        "files": {"activations": act_path.name, "labels": lp, "decisions": dp},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # ── summary ──
    print(f"\n=== dataset @ {out_dir} ===")
    print(f"rows={N}  decisions={len(all_dec)}")
    if len(decisions):
        print("decisions by choice:\n" + decisions["choice"].value_counts().to_string())
    if N:
        print("segment counts:\n" + labels["segment"].value_counts().to_string())
        print(f"success rate: {labels.groupby('traj_seed')['reached'].first().mean():.1%}")
    sz = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"on-disk: {sz:.1f} MB  ({act_path.name}, {lp}, {dp}, manifest.json)")


if __name__ == "__main__":
    main()
