#!/usr/bin/env python3
"""Activation dataset for belief analysis — one per agent, same 1 200 maps.

Every held-out map is played exactly once by each agent, and every timestep is
recorded: the activations the actor consumed, where the agent was, what it did,
and how much evidence it had seen by then. This is the substrate for probing,
for the evidence-integration analysis, and for steering — a steered re-run on
the same (map, seed) reproduces the baseline trajectory exactly, so the two can
be diffed step by step.

Outputs, in activation_datasets/cogniland_belief/:

  <agent>_<feat>.npy    (N, D) per feature; float16 for continuous, int8 for
                        discrete latents (class index per slot, lossless)
  <agent>_steps.csv     N rows: ep, map_id, category, t, row, col, facing,
                        action, reward, ret, water_seen, rock_seen, water_now,
                        rock_now, phase, col_rel_wall
  <agent>_episodes.csv  1 200 rows: outcome, door, seed, tool counts, totals
  <agent>_manifest.json shapes, dtypes, checkpoint, seeds, git sha

Features per agent:
  ppo      h        (128)   GRU hidden, the whole carried state
  dreamer  deter    (3072)  RSSM deterministic path, carried across time
           stoch_idx (32)   discrete latent, class index per slot
  storm    h        (512)   transformer output over the rolling context
           stoch_idx (32)   discrete latent, class index per slot

  PYTHONPATH=src python scripts/mechinterp/build_belief_dataset.py --agent ppo
  PYTHONPATH=src:r2dreamer_model ... --agent dreamer
  (from STORM_model/) PYTHONPATH=.:..:../src python ../scripts/... --agent storm
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

STEP_COLS = ["ep", "map_id", "category", "t", "row", "col", "facing", "action",
             "reward", "ret", "water_seen", "rock_seen", "water_now", "rock_now",
             "phase", "col_rel_wall"]
EP_COLS = ["ep", "map_id", "category", "correct_target", "seed", "steps", "success",
           "door", "final_row", "final_col", "ret", "builds", "mines",
           "water_seen", "rock_seen", "n_steps_recorded"]


# exact command to rebuild each agent's half of the dataset, including the
# environment it must run in -- these are three mutually incompatible envs
REPRO = {
    "ppo": ("conda activate crusoe && "
            "PYTHONPATH=src python scripts/mechinterp/build_belief_dataset.py --agent ppo"),
    "dreamer": ("conda activate r2dreamer && PYTHONPATH=src:r2dreamer_model "
                "python scripts/mechinterp/build_belief_dataset.py --agent dreamer"),
    "storm": ("cd STORM_model && source .venv/bin/activate && PYTHONPATH=.:..:../src "
              "python ../scripts/mechinterp/build_belief_dataset.py --agent storm"),
}


def sha256(path, cap=None):
    """Hash a file; `cap` bytes is enough to fingerprint a big checkpoint."""
    import hashlib
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if cap and n >= cap:
                break
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["ppo", "dreamer", "storm"])
    ap.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    ap.add_argument("--out", default=str(REPO / "activation_datasets/cogniland_belief"))
    ap.add_argument("--limit", type=int, default=0, help="first N maps only (smoke test)")
    ap.add_argument("--base-seed", type=int, default=1000,
                    help="episode seed = base_seed + map_id, recorded per episode")
    ap.add_argument("--ppo-ckpt", default=str(REPO / "final_models/ppo/ppo_plain_noaux.pt"))
    ap.add_argument("--storm-bundle", default=str(REPO / "final_models/storm"))
    ap.add_argument("--storm-step", type=int, default=624489)
    ap.add_argument("--dreamer-ckpt",
                    default=str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"))
    ap.add_argument("--dreamer-size", default="size25M")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from cogniland.bridge_tunnel import tiles as T
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS, make_dreamer, make_ppo, make_storm

    if a.agent == "ppo":
        act, reset = make_ppo(a.ppo_ckpt, sampled=True)
        ckpt = a.ppo_ckpt
    elif a.agent == "storm":
        act, reset = make_storm(a.storm_bundle, a.storm_step, sampled=True)
        ckpt = f"{a.storm_bundle}@{a.storm_step}"
    else:
        act, reset = make_dreamer(a.dreamer_ckpt, a.device, a.dreamer_size, sampled=True)
        ckpt = a.dreamer_ckpt
    get_feats = getattr(act, "get_features", None)
    if get_feats is None:
        sys.exit(f"{a.agent}: adapter exposes no get_features()")

    with open(a.maps, "rb") as f:
        pool = pickle.load(f)
    if a.limit:
        pool = pool[: a.limit]
    half = FORKWALL_KWARGS["view_size"] // 2
    A_BUILD, A_MINE = 4, 5

    feat_bufs: dict[str, list] = {}
    step_rows, ep_rows = [], []
    t0 = time.time()

    for ep, rec in enumerate(pool):
        seed = a.base_seed + ep
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            pass

        if hasattr(act, "set_seed"):
            act.set_seed(seed)      # agents with their own PRNG (STORM)
        env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
        obs, _ = env.reset()
        reset()
        H, W = rec.terrain.shape
        wall = rec.wall_col
        mem_lo = max(0, wall - 16)
        pass_col = rec.passage_cells[0][1] if rec.passage_cells else wall
        seen = np.zeros((H, W), bool)
        n_water = n_rock = n_build = n_mine = 0
        ret = 0.0
        n_rec = 0

        for t in range(FORKWALL_KWARGS["max_steps"]):
            r, c = env._pos
            r0, r1 = max(0, r - half), min(H, r + half + 1)
            c0, c1 = max(0, c - half), min(W, c + half + 1)
            terr = np.asarray(env._terrain)
            win_now = terr[r0:r1, c0:c1]
            water_now = int((win_now == T.WATER).sum())
            rock_now = int((win_now == T.ROCK).sum())
            fresh = ~seen[r0:r1, c0:c1]
            if fresh.any():
                tiles = win_now[fresh]                 # value at FIRST sight
                n_water += int((tiles == T.WATER).sum())
                n_rock += int((tiles == T.ROCK).sum())
                seen[r0:r1, c0:c1] = True

            action = act(obs, False)                   # act, then read what it used
            f = get_feats()
            for k, v in f.items():
                feat_bufs.setdefault(k, []).append(v)

            phase = ("evidence" if c < mem_lo else
                     "corridor" if c < pass_col else "past_wall")
            obs, rw, term, trunc, info = env.step(action)
            ret += float(rw)
            if action == A_BUILD and info.get("placed"):
                n_build += 1
            if action == A_MINE and info.get("mined"):
                n_mine += 1

            step_rows.append([ep, ep, rec.category, t, r, c, env._facing, int(action),
                              round(float(rw), 5), round(ret, 5), n_water, n_rock,
                              water_now, rock_now, phase, int(c - wall)])
            n_rec += 1
            if term or trunc:
                break

        fr, fc = env._pos
        top = {p[0] for p in rec.top_goal_cells}
        bot = {p[0] for p in rec.bottom_goal_cells}
        door = "top" if fr in top else "bottom" if fr in bot else "none"
        ep_rows.append([ep, ep, rec.category, rec.correct_target, seed, n_rec,
                        int(env._pos in (env._correct_cells or set())), door,
                        fr, fc, round(ret, 5), n_build, n_mine,
                        n_water, n_rock, n_rec])
        if (ep + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {ep+1}/{len(pool)} episodes, {len(step_rows)} steps, "
                  f"{el:.0f}s ({el/(ep+1):.2f}s/ep)", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck_path = Path(ckpt.split("@")[0])
    manifest = {"agent": a.agent, "checkpoint": ckpt, "git": git_sha(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "python": sys.version.split()[0],
                "reproduce": REPRO[a.agent] + (f" --limit {a.limit}" if a.limit else ""),
                "checkpoint_sha256_head": (sha256(ck_path, cap=1 << 24)
                                           if ck_path.is_file() else None),
                "replay_one": ("python scripts/mechinterp/replay_episode.py "
                               f"--agent {a.agent} --map-id <id>"),
                "maps": str(a.maps), "episodes": len(ep_rows),
                "steps": len(step_rows), "base_seed": a.base_seed,
                "seed_rule": "base_seed + map_id", "sampled": True,
                "env_kwargs": {k: v for k, v in FORKWALL_KWARGS.items()
                               if isinstance(v, (int, float, str, bool))},
                "features": {}}
    for k, buf in feat_bufs.items():
        arr = np.stack(buf)
        np.save(out / f"{a.agent}_{k}.npy", arr)
        manifest["features"][k] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                                   "mb": round(arr.nbytes / 1e6, 1)}
        print(f"  wrote {a.agent}_{k}.npy  {arr.shape} {arr.dtype} "
              f"({arr.nbytes/1e6:.0f} MB)")
        del arr

    for name, cols, rows in (("steps", STEP_COLS, step_rows),
                             ("episodes", EP_COLS, ep_rows)):
        with open(out / f"{a.agent}_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols); w.writerows(rows)
        print(f"  wrote {a.agent}_{name}.csv  ({len(rows)} rows)")

    # fingerprint every artefact so silent corruption or drift is detectable
    manifest["files"] = {}
    for f in sorted(out.glob(f"{a.agent}_*")):
        if f.name.endswith("manifest.json"):
            continue
        manifest["files"][f.name] = {"bytes": f.stat().st_size,
                                     "sha256": sha256(f)}
    (out / f"{a.agent}_manifest.json").write_text(json.dumps(manifest, indent=1))
    ok = sum(r[6] for r in ep_rows)
    print(f"\n{a.agent}: {len(ep_rows)} episodes, {len(step_rows)} steps, "
          f"success {ok/len(ep_rows):.4f}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
