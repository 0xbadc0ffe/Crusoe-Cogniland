#!/usr/bin/env python3
"""Uniform success eval for all four agents × {stochastic, argmax} action modes.

Rolls each agent out on held-out maps (seeds 10000+) using the REAL envs (PyTorch
for PPO, JAX for Dreamer — both include the stochastic slip), under stochastic
(sample) and argmax (greedy) action selection, and prints a success table.

    python scripts/bridge_tunnel/eval_success_table.py --n-traj 16 --maps 12
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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "bridge_tunnel"))

CATS = ["balanced", "lakes", "rocky"]


# ───────────────────────────── PPO (torch) ────────────────────────────────
def ppo_eval(variant, ckpt, recs, n_traj, max_steps, mode, device="cpu"):
    import torch
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv, BridgeTunnelCommitEnv
    from cogniland.bridge_tunnel.policy import PPOGRUPolicy
    EnvCls = BridgeTunnelCommitEnv if variant == "btc" else BridgeTunnelEnv
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cargs = ck["args"]; sd = ck["policy"]
    view = cargs.get("view_size", 21)
    obs_enc = "onehot" if "tile_embed.weight" not in sd else cargs.get("obs_encoding", "embed")
    n_tiles = int(sd["tile_embed.weight"].shape[0]) if "tile_embed.weight" in sd \
        else int(sd["cnn.0.weight"].shape[1]) - 2
    dummy = EnvCls(size=cargs.get("env_size", 32), width=cargs.get("env_width", 64), view_size=view)
    dummy.reset()
    pol = PPOGRUPolicy(dummy.observation_space, num_actions=int(sd["actor.weight"].shape[0]),
                       gru_hidden=cargs.get("gru_hidden", 128), embed_dim=cargs.get("embed_dim", 256),
                       num_tile_classes=n_tiles, obs_encoding=obs_enc).to(device)
    pol.load_state_dict(sd); pol.eval()

    out = {}
    with torch.no_grad():
        for rec, cat in recs:
            H, W = rec.terrain.shape
            envs = [EnvCls(map_record=rec, size=H, width=W, view_size=view, max_steps=max_steps)
                    for _ in range(n_traj)]
            obs = [e.reset()[0] for e in envs]
            h = torch.zeros(1, n_traj, pol.gru_hidden, device=device)
            done = torch.zeros(n_traj, device=device)
            active = np.ones(n_traj, bool); reached = np.zeros(n_traj, bool)
            for _ in range(max_steps):
                mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
                sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
                gru, h = pol._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
                logits, _ = pol._heads(gru.squeeze(0))
                acts = (logits.argmax(-1) if mode == "argmax"
                        else torch.distributions.Categorical(logits=logits).sample()).cpu().numpy()
                if not active.any():
                    break
                for i, e in enumerate(envs):
                    if not active[i]:
                        continue
                    o, r, term, trunc, info = e.step(int(acts[i]))
                    obs[i] = o
                    if term:
                        reached[i] = True; active[i] = False
                    elif trunc:
                        active[i] = False
            out.setdefault(cat, []).extend(reached.tolist())
    return out


# ──────────────────────────── Dreamer (jax) ───────────────────────────────
def dreamer_eval(variant, ckpt, recs, n_traj, max_steps, mode):
    import jax, jax.numpy as jnp
    import orbax.checkpoint as ocp
    import eval_bridge_tunnel_commit_dreamer as ev
    import purejaxwm.dreamerv3.behavior as ac
    from cogniland.bridge_tunnel.jax import BridgeTunnelJaxEnv, constants as C

    ck = Path(ckpt).resolve()
    cfg = json.loads((ck.parent.parent / "config.json").read_text())
    ev._DECODER_MODE = cfg.get("decoder", "categorical")
    pay = ocp.PyTreeCheckpointer().restore(str(ck))
    wm = jax.tree_util.tree_map(jnp.asarray, pay["wm_params"])
    acp = jax.tree_util.tree_map(jnp.asarray, pay["ac_params"])
    encoder, rssm, actor = ev._build_model(cfg)
    env = BridgeTunnelJaxEnv()

    def rollout(params, key):
        key, kr = jax.random.split(key)
        obs0, st0 = jax.vmap(env.reset_env, in_axes=(0, None))(jax.random.split(kr, n_traj), params)
        rs = rssm.initial_state((n_traj,)); la = jnp.zeros((n_traj, C.NUM_ACTIONS))
        lif = jnp.ones((n_traj,), bool); done = jnp.zeros((n_traj,), bool)

        def step(carry, _):
            st, obs, rs, la, lif, done, key = carry
            am = jnp.where(lif[..., None], 0.0, la)
            key, s1, s2, s3 = jax.random.split(key, 4)
            embed = encoder.apply(wm["encoder"], ev._flatten_obs(obs))
            _, post = rssm.apply(wm["rssm"], rs, am, embed, lif, rngs={"stoch": s1})
            logits = ac.unimix_logits(actor.apply(acp["actor"], post.features()))
            a = jnp.argmax(logits, -1) if mode == "argmax" else jax.random.categorical(s2, logits)
            nobs, nst, _, dn, info = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(
                jax.random.split(s3, n_traj), st, a, params)

            def sel(nx, pv):
                m = done.reshape(done.shape + (1,) * (nx.ndim - 1)); return jnp.where(m, pv, nx)
            nst = jax.tree_util.tree_map(sel, nst, st); nobs = jax.tree_util.tree_map(sel, nobs, obs)
            return (nst, nobs, post, jax.nn.one_hot(a, C.NUM_ACTIONS), jnp.zeros((n_traj,), bool),
                    done | dn, key), (info["reached_target"] & (~done))
        _, reached = jax.lax.scan(step, (st0, obs0, rs, la, lif, done, key), None, length=max_steps)
        return np.asarray(reached.any(0))

    key = jax.random.PRNGKey(0); out = {}
    for rec, cat in recs:
        key, sub = jax.random.split(key)
        params = ev._params_for(rec, cfg)
        out.setdefault(cat, []).extend(rollout(params, sub).tolist())
    return out


# ─────────────────────────────── maps ─────────────────────────────────────
def make_recs(variant, ckpt_cfg, n_maps, seed_start=10000):
    from cogniland.bridge_tunnel import generate_commit_map, generate_bridge_tunnel_map
    size = ckpt_cfg.get("env_size") or ckpt_cfg.get("map_size", 32)
    width = ckpt_cfg.get("env_width") or ckpt_cfg.get("map_width", 64)
    gh = ckpt_cfg.get("goal_half", 1)
    gh = gh if (gh is not None and gh >= 0) else None
    recs = []
    if variant == "btc":
        for ci, c in enumerate(CATS):
            for j in range(n_maps):
                recs.append((generate_commit_map(size=size, width=width, seed=seed_start + ci * 100000 + j,
                                                  category=c, tree_frac=0.03, goal_half=gh), c))
    else:
        for j in range(n_maps):
            recs.append((generate_bridge_tunnel_map(size=size, width=width, seed=seed_start + j,
                                                    tree_frac=0.03, goal_half=gh), "all"))
    return recs


def _cfg_of(ckpt, algo):
    import torch, json
    if algo == "ppo":
        return torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    return json.loads((Path(ckpt).resolve().parent.parent / "config.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj", type=int, default=16)
    p.add_argument("--maps", type=int, default=12, help="maps per category (btc) or total (bt)")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--out", default="outputs/analysis/success_table.csv")
    args = p.parse_args()

    MODELS = [
        ("bt_ppo", "ppo", "bt", "released_models/bridge_tunnel/ppo_gru.pt"),
        ("btc_ppo", "ppo", "btc", "released_models/bridge_tunnel_commit/ppo_gru_commit.pt"),
        ("bt_dreamer", "dreamer", "bt", "outputs/dreamer_runs/dreamer_bt_25M_behavior/checkpoints/step_1500000"),
        ("btc_dreamer", "dreamer", "btc", "released_models/bridge_tunnel_commit/dreamerv3_commit/checkpoints/step_6000000"),
    ]
    rows = []
    for name, algo, variant, ckpt in MODELS:
        cfg = _cfg_of(ckpt, algo)
        recs = make_recs(variant, cfg, args.maps)
        for mode in ["stochastic", "argmax"]:
            fn = ppo_eval if algo == "ppo" else dreamer_eval
            per = fn(variant, ckpt, recs, args.n_traj, args.max_steps, mode)
            allr = [x for v in per.values() for x in v]
            row = {"dataset": name, "algo": algo, "variant": variant, "mode": mode,
                   "success": float(np.mean(allr)), "n_episodes": len(allr)}
            if variant == "btc":
                for c in CATS:
                    row[f"succ_{c}"] = float(np.mean(per.get(c, [0])))
            rows.append(row)
            extra = "  ".join(f"{c}={row.get(f'succ_{c}',float('nan')):.2f}" for c in CATS) if variant == "btc" else ""
            print(f"{name:12s} {mode:10s} success={row['success']:.3f} (n={row['n_episodes']})  {extra}", flush=True)

    import pandas as pd
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")
    # pretty pivot
    df = pd.DataFrame(rows)
    piv = df.pivot(index="dataset", columns="mode", values="success")
    print("\n=== SUCCESS (overall) ===\n" + piv.round(3).to_string())


if __name__ == "__main__":
    main()
