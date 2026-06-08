#!/usr/bin/env python3
"""Find + render imagined rollouts where DreamerV3 imagines a specific event:
  bridge  — a build action turns the WATER cell ahead into WOOD (in imagination)
  tunnel  — a mine  action turns the ROCK cell ahead into GRASS
  reach   — the TARGET tile reaches the agent's own cell (imagined goal arrival)

Sweeps eval seeds, runs warmup→open-loop imagination (reusing the viz module),
detects the event in the DECODED imagined frames, and writes the first/best
matching rollout to a video. Pure imagination = world model + actor, no env.

    python scripts/bridge_tunnel/imagine_find_events.py --event bridge \
        --checkpoint <ckpt> --category lakes --warmup 10 --horizon 35
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "bridge_tunnel"))

import viz_dreamer_bridge_tunnel_imagine as VZ   # noqa: E402
from cogniland.bridge_tunnel.jax import BridgeTunnelJaxEnv   # noqa: E402

GRASS, WATER, ROCK, WOOD, TARGET = 0, 1, 2, 3, 4
_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}      # up/down/left/right


def _facings(frames):
    out, cur = [], 3
    for f in frames:
        if f["action"] < 4:
            cur = f["action"]
        out.append(cur)
    return out


def detect(frames, V):
    """Return dict of (event -> step index of first occurrence, else None)."""
    c = V // 2
    fac = _facings(frames)
    idxs = [i for i, f in enumerate(frames) if f["phase"] == "open-loop"]
    ev = {"bridge": None, "tunnel": None, "reach": None}
    for k in range(len(idxs)):
        i = idxs[k]
        f = frames[i]
        if f["tiles"][c, c] == TARGET and ev["reach"] is None:
            ev["reach"] = i
        if k + 1 < len(idxs):
            nf = frames[idxs[k + 1]]
            dr, dc = _DELTA[fac[i]]
            ar, ac = c + dr, c + dc
            if 0 <= ar < V and 0 <= ac < V:
                now, nxt = int(f["tiles"][ar, ac]), int(nf["tiles"][ar, ac])
                if f["action"] == 4 and now != WOOD and nxt == WOOD and ev["bridge"] is None:
                    ev["bridge"] = i
                if f["action"] == 5 and now == ROCK and nxt == GRASS and ev["tunnel"] is None:
                    ev["tunnel"] = i
    return ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True, choices=("bridge", "tunnel", "reach"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--category", default="lakes", choices=("balanced", "lakes", "rocky"))
    p.add_argument("--seeds", type=int, default=40, help="how many seeds to sweep")
    p.add_argument("--seed-start", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--horizon", type=int, default=35)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--sprite-px", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/videos/imagine_events"))
    args = p.parse_args()

    ck = args.checkpoint.resolve()
    cfg = json.loads((ck.parent.parent / "config.json").read_text())
    VZ._DECODER_MODE = cfg.get("decoder", "categorical")
    if cfg.get("env_id") == "bridge_tunnel_commit":
        VZ.ACTION_NAMES = ["up", "down", "left", "right", "build", "mine"]
    pay = ocp.PyTreeCheckpointer().restore(str(ck))
    wm = jax.tree_util.tree_map(jnp.asarray, pay["wm_params"])
    acp = jax.tree_util.tree_map(jnp.asarray, pay["ac_params"])
    models = VZ._build_model(cfg)
    env = BridgeTunnelJaxEnv()
    sprites = VZ._load_sprite_imgs(args.sprite_px)
    palette = VZ._NATURAL_PALETTE
    V = cfg["view_size"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(0)

    print(f"searching {args.seeds} seeds for an imagined '{args.event}' "
          f"(variant={cfg.get('env_id')}, cat={args.category}, warmup={args.warmup}, horizon={args.horizon})")
    for j in range(args.seeds):
        seed = args.seed_start + j
        params, rec = VZ._single_map_params(seed, cfg, args.category)
        key, sub = jax.random.split(key)
        frames = VZ.rollout_and_imagine(models, wm, acp, env, params, cfg,
                                        args.warmup, args.horizon, sub, palette)
        ev = detect(frames, V)
        imag = [f for f in frames if f["phase"] == "open-loop"]
        nb = sum(f["action"] == 4 for f in imag); nm = sum(f["action"] == 5 for f in imag)
        print(f"  seed {seed}: imagined build={nb} mine={nm}  "
              f"events bridge={ev['bridge']} tunnel={ev['tunnel']} reach={ev['reach']}", flush=True)
        hit = ev[args.event]
        if hit is not None:
            facs = _facings(frames)
            rgb = [VZ._frame_to_rgb(f, 12, sprites, args.sprite_px, fc)
                   for f, fc in zip(frames, facs)]
            base = args.out_dir / f"imagine_{args.event}_seed{seed}"
            path, fmt = VZ._write_video(rgb, base, args.fps)
            try:
                VZ._write_strip(frames, args.out_dir / f"imagine_{args.event}_strip_seed{seed}",
                                3, sprites, args.sprite_px, facs)
            except Exception as e:  # noqa: BLE001
                print(f"  (strip skipped: {e})")
            print(f"  FOUND '{args.event}' at imagined step {hit} on seed {seed} → wrote {path}")
            return
        if (j + 1) % 10 == 0:
            print(f"  ...{j+1} seeds, no '{args.event}' yet", flush=True)
    print(f"  no '{args.event}' found in {args.seeds} seeds — try more seeds / longer horizon")


if __name__ == "__main__":
    main()
