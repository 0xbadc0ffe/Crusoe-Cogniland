#!/usr/bin/env python3
"""Build a mechanistic-interpretability activation dataset for a PPO+GRU agent on
``bridge_tunnel`` or ``bridge_tunnel_commit``.

We probe BELIEF (obstacle/category ahead, cost-to-go) and SKILL/OPTION subspaces.
For ``bridge_tunnel_commit`` the skill is an explicit, observable **commitment**
(none/build/mine) and the map carries a **category** label (balanced/lakes/rocky)
-- both logged as ground truth. For ``bridge_tunnel`` there is no commitment: the
skill is the soft per-step strategy (avoid/bridge/tunnel) recovered with the strict
in->out crossing rule, to be investigated further downstream.

The output bundle is **self-contained** -- usable for analysis/decoding with NO
repository access (it ships the raw maps + the tile palette + a standalone
decoder). Reproducing or steering the agent additionally needs the repo (see
``REPRODUCE.md`` written into the bundle).

Per step we log:
  * activations           gru_h (128, recurrent belief carrier) + enc_embed (256)
  * the FULL observation  egocentric minimap (V,V int8) + scalars (so any frame
                          can be re-rendered offline -- any local feature is then
                          derivable; we do NOT precompute local densities/sizes)
  * policy outputs        action, action_probs, value
  * belief labels         ctg_to_goal, compass, obstacle-ahead type/dist, in_obstacle
  * commit labels (commit env only) commit_state, committed_now, commit_step,
                          time_since_commit, final_commit, correct_commit, category
  * strategy labels       per-step segment {free,approach,avoid,bridge,tunnel} +
                          a per-decision contrast table (steering contrast sets)

REPRODUCIBILITY: each trajectory is fully determined by its ``traj_seed`` (the map
is fixed; the only randomness is action sampling). ``torch.manual_seed(traj_seed)``
+ reset on the same map on ``--device cpu`` reproduces the EXACT obs + actions.

    # bridge_tunnel_commit (class-balanced maps)
    python scripts/mechinterp/build_activation_dataset.py --env bridge_tunnel_commit \\
        --checkpoint released_models/bridge_tunnel_commit/ppo_commit_onehot.pt \\
        --maps-per-category 30 --n-traj 60 --out-dir activation_datasets/btc_ppo

    # bridge_tunnel
    python scripts/mechinterp/build_activation_dataset.py --env bridge_tunnel \\
        --checkpoint released_models/bridge_tunnel/natural_centergoal3_onehot.pt \\
        --n-maps 90 --n-traj 60 --out-dir activation_datasets/bt_ppo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as cc_label, distance_transform_cdt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
_FACE_NAME = {0: "up", 1: "down", 2: "left", 3: "right"}
_CATS = ("balanced", "lakes", "rocky")


# ─────────────────────── env-specific wiring ──────────────────────────

def _env_cfg(env_name: str):
    """Return a dict of env-specific callables/constants (lazy imports)."""
    if env_name == "bridge_tunnel":
        from cogniland.bridge_tunnel import generate_bridge_tunnel_map, tiles as T
        from cogniland.bridge_tunnel.env import BridgeTunnelEnv
        from cogniland.bridge_tunnel.policy import PPOGRUPolicy
        ctg_fn = lambda terr, tgt: BridgeTunnelEnv._compute_ctg(terr, tgt).astype(np.float32)
        return dict(
            T=T, Env=BridgeTunnelEnv, Policy=PPOGRUPolicy, ctg_fn=ctg_fn,
            is_commit=False,
            action_names=["up", "down", "left", "right", "place", "mine"],
            gen=lambda seed, cat, kw: generate_bridge_tunnel_map(seed=seed, **kw),
        )
    elif env_name == "bridge_tunnel_commit":
        from cogniland.bridge_tunnel_commit import generate_commit_map, tiles as T
        from cogniland.bridge_tunnel_commit.env import BridgeTunnelCommitEnv
        from cogniland.bridge_tunnel.policy import PPOGRUPolicy
        # commit-aware ctg: index 0 = 'none' field (both obstacles crossable)
        ctg_fn = lambda terr, tgt: BridgeTunnelCommitEnv._compute_all_ctg(terr, tgt)[0].astype(np.float32)
        return dict(
            T=T, Env=BridgeTunnelCommitEnv, Policy=PPOGRUPolicy, ctg_fn=ctg_fn,
            is_commit=True,
            action_names=["up", "down", "left", "right", "build", "mine"],
            gen=lambda seed, cat, kw: generate_commit_map(seed=seed, category=cat, **kw),
        )
    raise SystemExit(f"unknown --env {env_name!r}")


# ───────────────────────── per-map precompute ─────────────────────────

def _map_geometry(rec, T, ctg_fn, min_body=4):
    orig = np.asarray(rec.terrain)
    H, W = orig.shape
    target = tuple(rec.target)
    ctg = ctg_fn(orig, target)
    geo = {"orig": orig, "H": H, "W": W, "target": target, "ctg": ctg, "bodies": {},
           "walkable": (T.GRASS, T.WOOD, T.TARGET, T.SAND, T.DIRT),
           "obstacle": {T.WATER: "water", T.ROCK: "rock", T.TREE: "tree"}}
    for tile, name in ((T.WATER, "water"), (T.ROCK, "rock")):
        lbl, n = cc_label(orig == tile)
        bodies = {}
        for bid in range(1, n + 1):
            mask = lbl == bid
            sz = int(mask.sum())
            if sz >= min_body:
                dist = distance_transform_cdt(~mask, metric="taxicab").astype(np.int32)
                bodies[bid] = {"size": sz, "cells": mask, "dist": dist}
        geo[f"{name}_lbl"] = lbl
        geo["bodies"][name] = bodies
    return geo


def _belief_row(geo, pos, facing, T, max_scan=14):
    """Ground-truth belief labels (NOT derivable cheaply from obs alone)."""
    orig, H, W, ctg, target = geo["orig"], geo["H"], geo["W"], geo["ctg"], geo["target"]
    walk, obst = geo["walkable"], geo["obstacle"]
    r, c = pos
    cur = int(orig[r, c])
    dr, dc = _FACE_DELTA[facing]
    ahead_type, ahead_dist = "none", -1
    for d in range(1, max_scan + 1):
        rr, cc = r + dr * d, c + dc * d
        if not (0 <= rr < H and 0 <= cc < W):
            break
        t = int(orig[rr, cc])
        if t in obst:
            ahead_type, ahead_dist = obst[t], d
            break
        if t not in walk:
            break
    return {
        "cur_tile": cur,
        "in_obstacle": cur in (T.WATER, T.ROCK),
        "ctg_to_goal": float(ctg[r, c]),
        "compass_dr": float(np.sign(target[0] - r)),
        "compass_dc": float(np.sign(target[1] - c)),
        "obst_ahead_type": ahead_type,
        "obst_ahead_dist": ahead_dist,
    }


# ───────────────────── strict crossing / strategy ─────────────────────

def _crossings(path, orig, lbl, tile, min_cross):
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


def _strategy(path, geo, T, map_id, traj_id, traj_seed, min_cross, approach_window, near_radius):
    orig = geo["orig"]
    n = len(path)
    seg = ["free"] * n
    decision_id = [""] * n
    decisions = []
    crossed = {"water": set(), "rock": set()}
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
                "choice": choice, "crossing_cells": cr["n_cells"], "decision_step": cr["start_idx"],
            })
    path_arr = np.array(path)
    for name in ("water", "rock"):
        for bid, b in geo["bodies"][name].items():
            if bid in crossed[name]:
                continue
            d = b["dist"][path_arr[:, 0], path_arr[:, 1]]
            if d.min() > near_radius:
                continue
            did = f"{map_id}:{traj_id}:{name}:{bid}"
            for k in np.where(d <= near_radius)[0]:
                if seg[k] == "free":
                    seg[k], decision_id[k] = "avoid", did
            decisions.append({
                "decision_id": did, "map_id": map_id, "traj_id": traj_id,
                "traj_seed": traj_seed, "obstacle_type": name, "body_size": b["size"],
                "choice": "avoid", "crossing_cells": 0, "decision_step": int(d.argmin()),
            })
    return seg, decision_id, decisions


# ───────────────────────────── rollout ────────────────────────────────

@torch.no_grad()
def _rollout(policy, env_make, geo, cfg, map_id, map_seed, category, traj_id, traj_seed,
             max_steps, device, min_cross, approach_window, near_radius):
    T = cfg["T"]; is_commit = cfg["is_commit"]; anames = cfg["action_names"]
    torch.manual_seed(traj_seed)              # the ONLY randomness (action sampling)
    env = env_make()
    obs = env.reset()[0]
    h = torch.zeros(1, 1, policy.gru_hidden, device=device)
    done = torch.zeros(1, device=device)
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
            commit_pre = int(getattr(env, "_commit", 0))
            path.append(pos)
            row = {
                "map_id": map_id, "map_seed": map_seed, "traj_id": traj_id,
                "traj_seed": traj_seed, "t": t, "pos_r": pos[0], "pos_c": pos[1],
                "facing": facing, "facing_name": _FACE_NAME[facing],
                "action": a, "action_name": anames[a], "value": float(value[0]),
                "_gru_h": gru_out.squeeze().cpu().numpy().astype(np.float16),
                "_enc_embed": cap["embed"].squeeze().cpu().numpy().astype(np.float16),
                "_minimap": obs["minimap"].astype(np.int8),
                "_scalars": obs["scalars"].astype(np.float16),
                "_probs": probs.cpu().numpy().astype(np.float16),
            }
            row.update(_belief_row(geo, pos, facing, T))
            if is_commit:
                row["category"] = category
                row["commit_state"] = ("none", "build", "mine")[commit_pre]
            obs, r, term, trunc, info = env.step(a)
            if is_commit:
                row["committed_now"] = bool(info.get("committed_now", False))
            rows.append(row)
            if term:
                reached = True; break
            if trunc:
                break
    finally:
        hk.remove()

    seg, did, decisions = _strategy(path, geo, T, map_id, traj_id, traj_seed,
                                    min_cross, approach_window, near_radius)
    ep_len = len(rows)
    # commit post-processing (commit env)
    commit_step = -1
    if cfg["is_commit"]:
        for k, row in enumerate(rows):
            if row.get("committed_now"):
                commit_step = k; break
        final_commit = rows[-1]["commit_state"] if rows else "none"
        dom = {"lakes": "build", "rocky": "mine"}.get(category)  # balanced -> either
    for k, row in enumerate(rows):
        row["segment"] = seg[k]; row["decision_id"] = did[k]
        row["reached"] = reached; row["ep_len"] = ep_len
        if cfg["is_commit"]:
            row["commit_step"] = commit_step
            row["time_since_commit"] = (k - commit_step) if commit_step >= 0 and k >= commit_step else -1
            row["final_commit"] = final_commit
            cs = row["commit_state"]
            row["correct_commit"] = (cs == dom) if (dom is not None and cs != "none") else (cs != "none")
    return rows, decisions


# ───────────────────────────────── main ───────────────────────────────

def _build_map_list(cfg, args):
    """Return list of dicts: {map_id, map_seed, category, rec}."""
    natkw = args._natkw
    out = []
    if cfg["is_commit"]:
        cats = [c.strip() for c in args.categories.split(",")]
        mid = 0
        # distinct seed block per category so map_seed (hence traj_seed) is GLOBALLY
        # unique across categories — else traj_seed collides between balanced/lakes/rocky.
        for ci, cat in enumerate(cats):
            base = args.seed_start + ci * 100_000
            for i in range(args.maps_per_category):
                seed = base + i
                rec = cfg["gen"](seed, cat, natkw)
                out.append({"map_id": mid, "map_seed": seed, "category": cat, "rec": rec}); mid += 1
    else:
        if args.maps:
            seeds = [int(s) for s in args.maps.split(",")]
        else:
            seeds = list(range(args.seed_start, args.seed_start + args.n_maps))
        for mid, seed in enumerate(seeds):
            rec = cfg["gen"](seed, None, natkw)
            out.append({"map_id": mid, "map_seed": seed, "category": None, "rec": rec})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=("bridge_tunnel", "bridge_tunnel_commit"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--maps", default=None, help="bridge_tunnel: comma-sep map seeds (overrides --n-maps)")
    p.add_argument("--n-maps", type=int, default=90, help="bridge_tunnel: number of maps from --seed-start")
    p.add_argument("--maps-per-category", type=int, default=30, help="commit env: maps per category")
    p.add_argument("--categories", default="balanced,lakes,rocky", help="commit env categories")
    p.add_argument("--seed-start", type=int, default=10_000, help="held-out map seeds")
    p.add_argument("--n-traj", type=int, default=60, help="stochastic rollouts per map")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--min-cross", type=int, default=2)
    p.add_argument("--approach-window", type=int, default=8)
    p.add_argument("--near-radius", type=int, default=3)
    p.add_argument("--device", default="cpu", help="cpu = exactly reproducible across machines")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    cfg = _env_cfg(args.env)
    T = cfg["T"]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    env_size = cargs.get("env_size", 32); env_width = cargs.get("env_width") or env_size
    view = cargs.get("view_size", 21); device = torch.device(args.device)
    gh = cargs.get("goal_half", 1)
    goal_half = gh if (gh is not None and gh >= 0) else None
    if args.env == "bridge_tunnel":
        args._natkw = dict(size=env_size, width=env_width, orientation=cargs.get("orientation", "natural"),
                           water_frac=cargs.get("water_frac", 0.14), rock_frac=cargs.get("rock_frac", 0.14),
                           tree_frac=cargs.get("tree_frac", 0.03), goal_half=goal_half)
    else:
        args._natkw = dict(size=env_size, width=env_width, tree_frac=cargs.get("tree_frac", 0.03),
                           goal_half=goal_half)

    sd = ckpt["policy"]; obs_enc = cargs.get("obs_encoding", "embed")
    if "tile_embed.weight" in sd:
        n_tiles = int(sd["tile_embed.weight"].shape[0])
    else:
        n_tiles = int(sd["cnn.0.weight"].shape[1]) - 2; obs_enc = "onehot"
    n_act = int(sd["actor.weight"].shape[0])
    dummy = cfg["Env"](size=env_size, width=env_width, view_size=view); dummy.reset()
    policy = cfg["Policy"](dummy.observation_space, num_actions=n_act,
                           gru_hidden=cargs.get("gru_hidden", 128),
                           embed_dim=cargs.get("embed_dim", 256),
                           num_tile_classes=n_tiles, obs_encoding=obs_enc).to(device)
    policy.load_state_dict(sd); policy.eval()
    n_scalars = int(dummy.observation_space["scalars"].shape[0])
    agent_sha = hashlib.sha1(Path(args.checkpoint).read_bytes()).hexdigest()[:10]

    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    maps = _build_map_list(cfg, args)

    all_rows, all_dec = [], []
    for m in maps:
        rec = m["rec"]
        geo = _map_geometry(rec, T, cfg["ctg_fn"])
        env_make = (lambda rec=rec: cfg["Env"](map_record=rec, size=rec.terrain.shape[0],
                    width=rec.terrain.shape[1], view_size=view, max_steps=args.max_steps))
        tag = m["category"] or f"seed{m['map_seed']}"
        print(f"[map {m['map_id']}] {tag} seed {m['map_seed']}: "
              f"water={len(geo['bodies']['water'])} rock={len(geo['bodies']['rock'])} bodies — "
              f"{args.n_traj} rollouts...", flush=True)
        for traj_id in range(args.n_traj):
            traj_seed = m["map_seed"] * 100_000 + traj_id
            rows, dec = _rollout(policy, env_make, geo, cfg, m["map_id"], m["map_seed"],
                                 m["category"], traj_id, traj_seed, args.max_steps, device,
                                 args.min_cross, args.approach_window, args.near_radius)
            all_rows.extend(rows); all_dec.extend(dec)

    N = len(all_rows)
    for i, row in enumerate(all_rows):
        row["row_id"] = i

    # ── activations + full obs → activations.h5 ──
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
        act_name = "activations.h5"
    except Exception as e:   # noqa: BLE001
        print(f"  [warn] h5py unavailable ({e!r}); writing activations.npz")
        np.savez_compressed(out_dir / "activations.npz", **arrays); act_name = "activations.npz"

    # ── raw maps → maps.npz (self-contained rendering, no repo needed) ──
    Hm, Wm = maps[0]["rec"].terrain.shape
    terr = np.stack([np.asarray(m["rec"].terrain, np.int8) for m in maps])
    spawn = np.array([m["rec"].spawn for m in maps], np.int32)
    target = np.array([m["rec"].target for m in maps], np.int32)
    goal_mask = np.stack([(np.asarray(m["rec"].terrain) == T.TARGET) for m in maps])
    map_seed_arr = np.array([m["map_seed"] for m in maps], np.int64)
    map_extra = {}
    if cfg["is_commit"]:
        map_extra["category"] = np.array([m["category"] for m in maps])
    np.savez_compressed(out_dir / "maps.npz", terrain=terr, spawn=spawn, target=target,
                        goal_mask=goal_mask, map_seed=map_seed_arr, **map_extra)

    # ── labels + decisions → parquet (csv fallback) ──
    import pandas as pd
    labels = pd.DataFrame(all_rows); decisions = pd.DataFrame(all_dec)

    def _save(df, stem):
        try:
            df.to_parquet(out_dir / f"{stem}.parquet"); return f"{stem}.parquet"
        except Exception as e:   # noqa: BLE001
            print(f"  [warn] parquet failed ({e!r}); writing {stem}.csv")
            df.to_csv(out_dir / f"{stem}.csv", index=False); return f"{stem}.csv"
    lp, dp = _save(labels, "labels"), _save(decisions, "decisions")

    # ── manifest (schema + palette + recipe), decoder, REPRODUCE.md ──
    manifest = {
        "env": args.env, "checkpoint": str(args.checkpoint), "agent_sha": agent_sha,
        "obs_encoding": obs_enc, "n_tiles": int(T.NUM_TILES), "view_size": view,
        "n_scalars": n_scalars, "n_actions": n_act, "action_names": cfg["action_names"],
        "natural_kwargs": args._natkw, "n_maps": len(maps), "n_traj_per_map": args.n_traj,
        "n_rows": N, "n_decisions": len(all_dec),
        "activation_sites": {"gru_h": int(policy.gru_hidden), "enc_embed": int(cargs.get("embed_dim", 256))},
        "obs_stored": {"minimap": [view, view, "int8"], "scalars": [n_scalars, "float16"]},
        "tile_names": {int(k): v for k, v in T.TILE_NAMES.items()},
        "tile_colors": np.asarray(T.TILE_COLORS).tolist(),
        "is_commit": cfg["is_commit"],
        "traj_seed_formula": "map_seed*100000 + traj_id",
        "reproduce": (f"torch.manual_seed(traj_seed); env=<{args.env} Env>(map_record="
                      f"gen(map_seed[, category]), view_size={view}, max_steps={args.max_steps}); "
                      f"env.reset(); sample Categorical(actor(gru(...))) in order on device=cpu."),
        "files": {"activations": act_name, "labels": lp, "decisions": dp, "maps": "maps.npz"},
        "column_dictionary": {
            "row_id": "global index aligned with activations.h5",
            "map_id": "index into maps.npz", "map_seed": "seed that generated the map",
            "traj_id": "trajectory index on this map", "traj_seed": "action-sampling seed",
            "t": "timestep", "pos_r/pos_c": "agent cell", "facing": "0/1/2/3=U/D/L/R",
            "action/action_name": "sampled action", "value": "critic",
            "reached/ep_len": "episode outcome/length",
            "ctg_to_goal": "Dijkstra cost-to-go (none-field for commit)",
            "compass_dr/dc": "sign of direction to goal",
            "obst_ahead_type/dist": "nearest obstacle in facing ray", "in_obstacle": "on water/rock",
            "segment": "free/approach/avoid/bridge/tunnel (strict in->out)", "decision_id": "obstacle decision key",
            "category": "(commit) map category = belief label",
            "commit_state": "(commit) none/build/mine at time t", "committed_now": "(commit) committed this step",
            "commit_step/time_since_commit/final_commit/correct_commit": "(commit) commitment dynamics",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # ship the standalone decoder + reproduction doc with the data
    dec_src = _ROOT / "scripts" / "mechinterp" / "decode_dataset.py"
    if dec_src.exists():
        shutil.copy(dec_src, out_dir / "decode_dataset.py")
    _write_reproduce_md(out_dir, manifest, args)

    # ── summary ──
    print(f"\n=== {args.env} dataset @ {out_dir} ===")
    print(f"maps={len(maps)}  rows={N}  decisions={len(all_dec)}")
    if N and cfg["is_commit"]:
        fc = labels.groupby(['map_id', 'traj_id']).first()   # one row per episode
        print("final_commit x category (episode counts):")
        print(fc.groupby(['category', 'final_commit']).size().to_string())
    if len(decisions):
        print("decisions by choice:\n" + decisions["choice"].value_counts().to_string())
    if N:
        print(f"success: {labels.groupby(['map_id','traj_id'])['reached'].first().mean():.1%}")
    sz = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file()) / 1e6
    print(f"on-disk: {sz:.1f} MB")


def _write_reproduce_md(out_dir, manifest, args):
    md = f"""# Reproducing the `{manifest['env']}` PPO activation dataset

This bundle is **self-contained for analysis**: `activations.h5`, `labels.*`,
`decisions.*`, `maps.npz`, `manifest.json` need **no repository access**. Use the
shipped `decode_dataset.py` to render frames / trajectories / videos.

## Files
| file | contents |
|---|---|
| `activations.h5` | per-row `gru_h`({manifest['activation_sites']['gru_h']}), `enc_embed`({manifest['activation_sites']['enc_embed']}), full obs `minimap`{manifest['obs_stored']['minimap'][:2]} int8 + `scalars`({manifest['n_scalars']}) f16, `action_probs`, `row_id` |
| `labels.*` | one row per timestep (PK `row_id`) — see `manifest.column_dictionary` |
| `decisions.*` | one row per (trajectory, obstacle) — steering contrast sets |
| `maps.npz` | raw `terrain` (N,H,W int8), `spawn`, `target`, `goal_mask`{', `category`' if manifest['is_commit'] else ''}, `map_seed` |
| `manifest.json` | schema, tile palette, agent sha, reproduction recipe |

## Decode (no repo needed)
```bash
python decode_dataset.py --row 12345                       # one labeled frame
python decode_dataset.py --traj <map_id> <traj_seed> --video out.mp4
python decode_dataset.py --traj <map_id> <traj_seed> --plot path.png
```
Every rendered frame is captioned with its `row_id`, `t`, action and (commit) state.

## Exact trajectory reproduction (needs the repo, for re-rollout / steering)
A trajectory is fully determined by `(map_seed, traj_seed)`; `traj_seed = {manifest['traj_seed_formula']}`.
The map is `gen(seed=map_seed{', category=category' if manifest['is_commit'] else ''})` with
`natural_kwargs = {manifest['natural_kwargs']}`. Then:

```
torch.manual_seed(traj_seed)            # the ONLY randomness = action sampling
env = <{manifest['env']} Env>(map_record=map, view_size={manifest['view_size']}, max_steps={args.max_steps})
obs = env.reset()[0]
# step: a ~ Categorical(actor(GRU(encode(obs), h)));  device = cpu
```
Run on **cpu** for bit-exact cross-machine reproduction. Use
`scripts/mechinterp/replay_trajectory.py --env {manifest['env']} --checkpoint <ckpt> --map-seed M --traj-seed S`
to replay (and assert it matches the stored obs/actions), and `--inject vec.npy --alpha A --rows a:b`
to add a steering vector to `gru_h` over a row range and log the perturbed behaviour.

Agent: `{manifest['checkpoint']}` (sha `{manifest['agent_sha']}`), obs_encoding `{manifest['obs_encoding']}`.
"""
    (out_dir / "REPRODUCE.md").write_text(md)


if __name__ == "__main__":
    main()
