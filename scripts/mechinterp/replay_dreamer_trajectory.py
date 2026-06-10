#!/usr/bin/env python3
"""Reproduce (and optionally steer) a single DreamerV3 dataset trajectory —
the Dreamer sibling of ``replay_trajectory.py`` (PPO).

The builder's randomness is ``jax.random.PRNGKey(0)`` split once per map (in
map_id order), with all ``n_traj_per_map`` rollouts vmapped under that key.
Replaying (map_id, traj_id) re-runs that map's vmapped rollout with the
identical key stream and slices trajectory ``traj_id``.

JAX PRNG keys are bit-stable across runs, but the network FLOAT math is not
(XLA autotuning) — a near-tie categorical sample can flip mid-episode. So the
default mode is **forced**: the logged actions are teacher-forced for the
replayed trajectory while every RNG key is consumed identically, which makes the
episode (positions, slips, commit) reproduce EXACTLY; activations recompute on
the true obs path. ``--mode free`` lets the policy sample (used for steering:
compare free α=0 vs α≠0 under the same keys).

    # plain replay (verified against the logged episode)
    python scripts/mechinterp/replay_dreamer_trajectory.py \
        --dataset activation_datasets/btc_dreamer --map-id 0 --traj-id 3

    # steered replay: add alpha*vec to the RSSM deter over steps [a,b)
    python scripts/mechinterp/replay_dreamer_trajectory.py \
        --dataset activation_datasets/btc_dreamer --map-id 0 --traj-id 3 \
        --inject lakes_minus_rocky.npy --alpha 4 --site actor --steps 0:40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="activation_datasets/<name>")
    p.add_argument("--map-id", type=int, required=True)
    p.add_argument("--traj-id", type=int, required=True)
    p.add_argument("--inject", type=Path, default=None,
                   help=".npy steering vector in deter space (deter_dim,)")
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--site", choices=("actor", "recur"), default="actor",
                   help="actor: add to the deter the actor reads; recur: add to the carried deter")
    p.add_argument("--steps", default=None, help="a:b step range for injection (default: all)")
    p.add_argument("--mode", choices=("forced", "free"), default=None,
                   help="default: forced for plain replay, free when steering")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()
    steering = args.inject is not None and args.alpha != 0.0
    mode = args.mode or ("free" if steering else "forced")

    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    import pandas as pd

    from mechinterp.build_dreamer_activation_dataset import _build_model
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from cogniland.bridge_tunnel.jax import BridgeTunnelJaxEnv, EnvParams
    from cogniland.bridge_tunnel.mapgen import CATEGORIES

    man = json.loads((args.dataset / "manifest.json").read_text())
    ck = (_ROOT / man["checkpoint"]).resolve()
    cfg = json.loads((ck.parent.parent / "config.json").read_text())
    payload = ocp.PyTreeCheckpointer().restore(str(ck))
    wm_params, ac_params = payload["wm_params"], payload["ac_params"]
    models = _build_model(cfg)
    env = BridgeTunnelJaxEnv()
    n_traj, max_steps = man["n_traj_per_map"], man["max_steps"]
    is_commit = man["is_commit"]
    anames = man["action_names"]

    # ---- env params for THIS map, rebuilt from the bundle's stored arrays ----
    z = np.load(args.dataset / "maps.npz", allow_pickle=True)
    mi = args.map_id
    terrain, target = z["terrain"][mi], z["target"][mi]
    if is_commit:
        ctg = BridgeTunnelEnv._compute_all_ctg(terrain, tuple(target))[None]
        cat = np.array([CATEGORIES.index(str(z["category"][mi]))], np.int32)
    else:
        ctg = BridgeTunnelEnv._compute_ctg(terrain, tuple(target)).astype(np.float32)[None]
        cat = None
    params = EnvParams.from_map_arrays(
        terrain=terrain[None], spawn=z["spawn"][mi][None], target=target[None],
        goal_mask=z["goal_mask"][mi][None], ctg=ctg, category=cat,
        max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        slack_penalty=cfg["slack_penalty"], reach_bonus=cfg["reach_bonus"],
        shaping_coef=cfg["shaping_coef"], build_cost=cfg["build_cost"], gamma=cfg["gamma"])

    # ---- the builder's exact key sequence: PRNGKey(0) split once per map ----
    key = jax.random.PRNGKey(0)
    for _ in range(args.map_id + 1):
        key, sub = jax.random.split(key)

    # ---- logged episode (verification target + teacher-forcing source) ----
    lab = pd.read_parquet(args.dataset / "labels.parquet",
                          filters=[("map_id", "==", args.map_id),
                                   ("traj_id", "==", args.traj_id)]).sort_values("t")
    logged_actions = lab["action"].to_numpy()
    i = args.traj_id

    forced = None
    if mode == "forced":
        forced = np.full((max_steps,), -1, np.int32)
        forced[:len(logged_actions)] = logged_actions

    ivec, a_lo, a_hi = None, 0, max_steps
    if steering:
        ivec = jnp.asarray(np.load(args.inject), jnp.float32) * args.alpha
        if args.steps is not None:
            a_lo, a_hi = map(int, args.steps.split(":"))

    rollout = _make_replay_rollout(models, wm_params, ac_params, env,
                                   ivec, args.site, a_lo, a_hi, forced, i)
    outs = rollout(params, n_traj, max_steps, sub)
    outs_np = {k: np.asarray(v) for k, v in outs.items()}

    already = outs_np["already_done"][:, i]
    reached_now = outs_np["reached_now"][:, i]
    valid = ~already
    end = int(np.argmax(reached_now)) if reached_now.any() else int(valid.sum()) - 1
    L = max(end, 0) + 1
    acts = outs_np["action"][:L, i]
    commit_final = int(outs_np["commit_after"][end, i]) if is_commit else 0
    reached = bool(reached_now.any())
    cname = ("none", "build", "mine")[commit_final] if is_commit else "-"
    print(f"map_id={args.map_id} traj_id={i}  mode={mode}  len={L}  reached={reached}  commit={cname}")
    print("actions:", " ".join(anames[a][0] for a in acts))

    if mode == "forced" and not args.no_verify:
        pos = np.stack([outs_np["pos_r"][:L, i], outs_np["pos_c"][:L, i]], 1)
        lpos = lab[["pos_r", "pos_c"]].to_numpy()
        ok = (len(logged_actions) == L and (logged_actions == acts).all()
              and (lpos == pos).all()
              and bool(lab["reached"].iloc[0]) == reached)
        if is_commit:
            ok = ok and lab["final_commit"].iloc[-1] == cname
        print(f"verify vs bundle: {'MATCH' if ok else 'MISMATCH'} "
              f"(logged len={len(logged_actions)}, replay len={L}; positions "
              f"{'identical' if (len(lpos) == len(pos) and (lpos == pos).all()) else 'DIFFER'})")
        sys.exit(0 if ok else 1)


def _make_replay_rollout(models, wm_params, ac_params, env, ivec, site, a_lo, a_hi,
                         forced, traj):
    """The builder's rollout (identical key consumption) + two replay hooks:
    * ``forced``  — (max_steps,) int32 logged actions (-1 = free) teacher-forced
                    onto trajectory ``traj`` (exact episode reproduction);
    * ``ivec``    — optional alpha*vec added to the RSSM deter on steps
                    [a_lo, a_hi): site=actor (read-only) or recur (carried).
    Mirrors build_dreamer_activation_dataset._make_rollout step-for-step."""
    import jax
    import jax.numpy as jnp
    import purejaxwm.dreamerv3.behavior as ac
    from cogniland.bridge_tunnel.jax import constants as C
    from mechinterp.build_dreamer_activation_dataset import _flatten_obs

    encoder, rssm, actor, _ = models
    forced_j = None if forced is None else jnp.asarray(forced)

    def rollout(params, n_traj, max_steps, key):
        key, kr = jax.random.split(key)
        obs0, state0 = jax.vmap(env.reset_env, in_axes=(0, None))(
            jax.random.split(kr, n_traj), params)
        rssm_state = rssm.initial_state((n_traj,))
        last_action = jnp.zeros((n_traj, C.NUM_ACTIONS))
        last_is_first = jnp.ones((n_traj,), dtype=bool)
        done = jnp.zeros((n_traj,), dtype=bool)

        def step(carry, t):
            state, obs, rssm_state, last_action, last_is_first, done, key = carry
            am = jnp.where(last_is_first[..., None], 0.0, last_action)
            flat = _flatten_obs(obs)
            key, s_stoch, s_pol, s_step = jax.random.split(key, 4)
            embed = encoder.apply(wm_params["encoder"], flat)
            _, post = rssm.apply(wm_params["rssm"], rssm_state, am, embed, last_is_first,
                                 rngs={"stoch": s_stoch})
            if ivec is not None:
                on = ((t >= a_lo) & (t < a_hi)).astype(jnp.float32)
                bumped = post._replace(deter=post.deter + on * ivec[None, :])
                if site == "recur":
                    post = bumped
                feat = bumped.features()
            else:
                feat = post.features()
            logits = ac.unimix_logits(actor.apply(ac_params["actor"], feat))
            probs = jax.nn.softmax(logits, axis=-1)
            a = jax.random.categorical(s_pol, logits)
            if forced_j is not None:                 # teacher-force the replayed traj
                fa = forced_j[t]
                a = a.at[traj].set(jnp.where(fa >= 0, fa, a[traj]))
            nobs, nstate, _, done_next, info = jax.vmap(
                env.step_env, in_axes=(0, 0, 0, None))(
                jax.random.split(s_step, n_traj), state, a, params)

            def _sel(nx, pv):
                m = done.reshape(done.shape + (1,) * (nx.ndim - 1))
                return jnp.where(m, pv, nx)
            nstate = jax.tree_util.tree_map(_sel, nstate, state)
            nobs = jax.tree_util.tree_map(_sel, nobs, obs)
            out = {
                "action": a.astype(jnp.int32), "probs": probs,
                "deter": post.deter,
                "pos_r": state.agent_r, "pos_c": state.agent_c,
                "commit": state.commit, "commit_after": nstate.commit,
                "reached_now": info["reached_target"] & (~done),
                "already_done": done,
            }
            carry = (nstate, nobs, post, jax.nn.one_hot(a, C.NUM_ACTIONS),
                     jnp.zeros((n_traj,), bool), done | done_next, key)
            return carry, out

        carry = (state0, obs0, rssm_state, last_action, last_is_first, done, key)
        _, outs = jax.lax.scan(step, carry, jnp.arange(max_steps))
        return outs

    return rollout


if __name__ == "__main__":
    main()
