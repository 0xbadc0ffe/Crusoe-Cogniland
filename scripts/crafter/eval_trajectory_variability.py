"""Measure the behavioural variability of a stochastic policy.

Samples ``N`` trajectories (default 100) from the *same* stochastic policy
on a *fixed* map + start — so the only source of variation is the policy's
own action sampling (plus env slip) — and reports, per eval map:

  * **state-occupancy entropy** — how widely the rollouts spread over cells
    (+ the across-trajectory Jensen–Shannon divergence), and
  * **number of modes** — how many distinct macro-trajectories (path
    bundles) the path distribution contains.

Works for both checkpoint kinds (auto-detected):

  * **Dreamer** — an orbax checkpoint *directory*
    (``runs/<id>/checkpoints/step_<N>``). Needs ``--maps-path`` + the run's
    ``config.json`` (auto-found next to the checkpoint).
  * **PPO** — a ``.pt`` file from ``train_ppo_gru.py``. Eval maps are
    generated with ``generate_map`` (biome from ``--map-type``).

Outputs ``<out-dir>/variability.json`` and one cluster-coloured
``overlay_map<k>.png`` per eval map so you can *see* the modes.

Examples
--------
    # Dreamer
    python scripts/crafter/eval_trajectory_variability.py \
        --checkpoint runs/<id>/checkpoints/step_1000000 \
        --maps-path data/crafter_in_cogniland/train_256.pkl \
        --n-maps 4 --n-trajectories 100

    # PPO
    python scripts/crafter/eval_trajectory_variability.py \
        --checkpoint runs/ppo_<id>/policy.pt \
        --map-type rocky --n-maps 4 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cogniland.nav.tiles import TILE_COLORS
from cogniland.trajectory_variability import (
    count_modes,
    occupancy_entropy,
    render_overlay,
    summarize,
)


# ───────────────────────────── Dreamer ───────────────────────────────────


def _rollout_dreamer(checkpoint: Path, maps_path: str, config: Path,
                     map_indices, n_traj, max_steps, seed):
    """Yield (terrain, target, start, trajectories, grid_shape) per map."""
    import jax
    import jax.numpy as jnp

    # reuse the frozen-model loader from the viz script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "viz_dreamer_trajectory", str(ROOT / "scripts" / "viz_dreamer_trajectory.py")
    )
    viz = importlib.util.module_from_spec(spec)
    # register before exec so flax's dataclass transform can resolve the
    # module of the nn.Module subclasses defined inside it
    sys.modules["viz_dreamer_trajectory"] = viz
    spec.loader.exec_module(viz)

    from cogniland.crafter_in_cogniland import (
        EnvParams, load_map_arrays, constants as C, build_obs,
    )
    from cogniland.crafter_in_cogniland import dynamics as dyn
    from cogniland.crafter_in_cogniland.state import EnvState

    cfg = json.loads(Path(config).read_text())
    pd = viz.load_frozen(checkpoint, cfg)
    arrays = load_map_arrays(maps_path)
    params = EnvParams.from_map_arrays(
        **arrays, max_steps=cfg["max_steps"], view_size=cfg["view_size"],
    )

    encoder, rssm, actor = pd["encoder"], pd["rssm"], pd["actor"]
    wm, ac_p = pd["wm_params"], pd["ac_params"]

    # Infer the obs scalar count this checkpoint was trained with: the
    # encoder's first kernel has shape (view²+n_scalars, hidden). Current
    # code uses 4 scalars; older checkpoints used 5 (a legacy active_obj/2
    # channel at index 2). Reconstruct whichever this checkpoint expects so
    # the tool is robust across the obs-schema change.
    in_dim = int(wm["encoder"]["params"]["Dense_0"]["kernel"].shape[0])
    n_scalars = in_dim - cfg["view_size"] * cfg["view_size"]

    def initial_state(map_idx):
        sr = params.spawn[map_idx, 0]
        sc = params.spawn[map_idx, 1]
        return EnvState(
            map_idx=jnp.int32(map_idx), agent_r=sr, agent_c=sc,
            facing=jnp.int32(1), active_object=jnp.int32(C.OBJ_NONE),
            step_count=jnp.int32(0), last_ctg=params.ctg_none[map_idx, sr, sc],
        )

    def step_fn(carry, key):
        state, rstate, last_a, is_first = carry
        obs = build_obs(state, params)
        mm = obs["minimap"].astype(jnp.float32)[None] / float(C.NUM_TERRAIN_TILES)
        sca = obs["scalars"].astype(jnp.float32)            # (4,)
        if n_scalars == 5:
            # legacy ordering: [compass_r, compass_c, active_obj/2, build, step]
            ao = state.active_object.astype(jnp.float32) / 2.0
            sca = jnp.concatenate([sca[:2], ao[None], sca[2:]])
        flat = jnp.concatenate([mm.reshape(1, -1), sca[None]], axis=-1)
        k_enc, k_pol, k_step = jax.random.split(key, 3)
        embed = encoder.apply(wm["encoder"], flat)
        _, post = rssm.apply(wm["rssm"], rstate, last_a, embed, is_first,
                             rngs={"stoch": k_enc})
        logits = actor.apply(ac_p["actor"], post.features())   # (1, A)
        a = jax.random.categorical(k_pol, logits[0])           # stochastic
        a_oh = jax.nn.one_hot(a, C.NUM_ACTIONS)[None]
        new_state, _r, done, _info = dyn.step(k_step, state, a, params)
        return ((new_state, post, a_oh, jnp.zeros((1,), bool)),
                (new_state.agent_r, new_state.agent_c, done))

    def run_one(key, map_idx):
        carry0 = (initial_state(map_idx), rssm.initial_state((1,)),
                  jnp.zeros((1, C.NUM_ACTIONS)), jnp.ones((1,), bool))
        keys = jax.random.split(key, max_steps)
        _, (rs, cs, dones) = jax.lax.scan(step_fn, carry0, keys)
        return rs, cs, dones

    run_batch = jax.jit(jax.vmap(run_one, in_axes=(0, None)), static_argnums=())

    for mi in map_indices:
        keys = jax.random.split(jax.random.PRNGKey(seed + mi), n_traj)
        rs, cs, dones = run_batch(keys, mi)
        rs, cs, dones = np.array(rs), np.array(cs), np.array(dones)
        sr = int(params.spawn[mi, 0]); sc = int(params.spawn[mi, 1])
        trajs = _assemble(rs, cs, dones, (sr, sc))
        terrain = np.asarray(params.terrain[mi])
        target = (int(params.target[mi, 0]), int(params.target[mi, 1]))
        yield terrain, target, (sr, sc), trajs, terrain.shape


# ─────────────────────────────── PPO ─────────────────────────────────────


def _rollout_ppo(checkpoint: Path, map_type, n_maps, n_traj, max_steps,
                 seed, device):
    import torch
    from cogniland.inference import PPOAgent
    from cogniland.nav import CognilandNavEnv, generate_map

    agent = PPOAgent.load(checkpoint, device=device)
    a = agent.ckpt_args
    size = int(a.get("env_size", 64))
    mt = map_type or a.get("map_type", "random")

    for mi in range(n_maps):
        rec = generate_map(size=size, map_type=mt, seed=1000 + mi, max_retries=400)
        env = CognilandNavEnv(
            size=size, map_type=mt, view_size=a.get("view_size", 21),
            tile_px=a.get("tile_px", 8), obs_mode=a.get("obs_mode", "symbolic"),
            max_steps=a.get("max_steps", 1000), map_record=rec,
        )
        trajs = []
        for i in range(n_traj):
            torch.manual_seed(seed * 100003 + mi * 9973 + i)
            obs, _ = env.reset()
            hidden = agent.initial_hidden(1)
            pos = [tuple(int(x) for x in env._pos)]
            done = False
            for _ in range(max_steps):
                act, hidden = agent.act(obs, hidden, done=False, greedy=False)
                obs, _r, term, trunc, info = env.step(act)
                pos.append(tuple(int(x) for x in info["position"]))
                if bool(term or trunc):
                    break
            trajs.append(np.array(pos, dtype=int))
        env.close()
        terrain = rec.terrain
        target = (int(rec.target[0]), int(rec.target[1]))
        start = (int(rec.spawn[0]), int(rec.spawn[1]))
        yield terrain, target, start, trajs, terrain.shape


# ───────────────────────────── shared ────────────────────────────────────


def _assemble(rs, cs, dones, start):
    """Per trajectory: prepend start, truncate at the first done (inclusive)."""
    out = []
    N, T = rs.shape
    for i in range(N):
        if dones[i].any():
            k = int(np.argmax(dones[i]))            # first True
        else:
            k = T - 1
        path = [start] + [(int(rs[i, j]), int(cs[i, j])) for j in range(k + 1)]
        out.append(np.array(path, dtype=int))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--algo", choices=["auto", "dreamer", "ppo"], default="auto")
    ap.add_argument("--n-trajectories", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--n-maps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps-path", default=None, help="(dreamer) pkl of map arrays")
    ap.add_argument("--config", type=Path, default=None, help="(dreamer) run config.json")
    ap.add_argument("--map-type", default=None, help="(ppo) eval biome")
    ap.add_argument("--device", default="cpu", help="(ppo) torch device")
    ap.add_argument("--out-dir", type=Path, default=None)
    # mode-clustering knobs
    ap.add_argument("--mode-dist-frac", type=float, default=0.12)
    ap.add_argument("--mode-min-frac", type=float, default=0.05)
    args = ap.parse_args()

    algo = args.algo
    if algo == "auto":
        algo = "dreamer" if args.checkpoint.is_dir() else "ppo"

    out_dir = args.out_dir or (
        args.checkpoint.parent if algo == "ppo" else args.checkpoint.parents[1]
    ) / "variability"
    out_dir.mkdir(parents=True, exist_ok=True)

    if algo == "dreamer":
        config = args.config or (args.checkpoint.parents[1] / "config.json")
        if args.maps_path is None:
            raise SystemExit("--maps-path is required for a Dreamer checkpoint")
        gen = _rollout_dreamer(
            args.checkpoint, args.maps_path, config,
            list(range(args.n_maps)), args.n_trajectories, args.max_steps, args.seed,
        )
    else:
        gen = _rollout_ppo(
            args.checkpoint, args.map_type, args.n_maps,
            args.n_trajectories, args.max_steps, args.seed, args.device,
        )

    per_map = []
    for k, (terrain, target, start, trajs, grid) in enumerate(gen):
        occ = occupancy_entropy(trajs, grid)
        modes = count_modes(trajs, grid, dist_frac=args.mode_dist_frac,
                            min_cluster_frac=args.mode_min_frac)
        reached = sum(int(tuple(t[-1]) == target) for t in trajs) / len(trajs)
        rec = {
            "map": k,
            "reach_rate": reached,
            "n_modes": modes["n_modes"],
            "n_clusters_total": modes["n_clusters_total"],
            "occupancy_entropy_nats": occ["occupancy_entropy_nats"],
            "occupancy_entropy_norm": occ["occupancy_entropy_norm"],
            "across_traj_jsd_nats": occ["across_traj_jsd_nats"],
            "across_traj_jsd_norm": occ["across_traj_jsd_norm"],
            "n_distinct_cells": occ["n_distinct_cells"],
            "cluster_sizes": modes["cluster_sizes"],
        }
        per_map.append(rec)
        render_overlay(
            terrain, target, trajs, modes["labels"], out_dir / f"overlay_map{k}.png",
            TILE_COLORS,
            title=(f"map {k}: {modes['n_modes']} modes, "
                   f"H_occ={occ['occupancy_entropy_nats']:.2f} nats, "
                   f"reach={reached:.0%}"),
        )
        print(f"map {k}: modes={modes['n_modes']} "
              f"(clusters {modes['cluster_sizes']})  "
              f"H_occ={occ['occupancy_entropy_nats']:.3f}  "
              f"JSD={occ['across_traj_jsd_nats']:.3f}  reach={reached:.0%}")

    summary = summarize(per_map)
    payload = {
        "checkpoint": str(args.checkpoint),
        "algo": algo,
        "n_trajectories": args.n_trajectories,
        "max_steps": args.max_steps,
        "fixed_start": True,
        "summary": summary,
        "per_map": per_map,
    }
    (out_dir / "variability.json").write_text(json.dumps(payload, indent=2))
    print("\n── summary (mean across maps) ──")
    print(f"  n_modes:          {summary.get('n_modes/mean', float('nan')):.2f}"
          f" ± {summary.get('n_modes/std', 0):.2f}")
    print(f"  occupancy entropy:{summary.get('occupancy_entropy_nats/mean', float('nan')):.3f}"
          f" ± {summary.get('occupancy_entropy_nats/std', 0):.3f} nats")
    print(f"  across-traj JSD:  {summary.get('across_traj_jsd_nats/mean', float('nan')):.3f} nats")
    print(f"\nwrote {out_dir}/variability.json and {len(per_map)} overlay PNGs")


if __name__ == "__main__":
    main()
