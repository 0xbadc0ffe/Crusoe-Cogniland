#!/usr/bin/env python3
"""Probe-gated projected suppression of tunnel/bridge skills — DreamerV3 (BT).

Same intervention as suppress_skill_ppo, applied to the RSSM deter inside the
JAX rollout: when the lookahead probe fires (p > tau_on), the closed-form
minimal projected step pushes the margin to logit(tau_off); the suppressed
deter is CARRIED. p <= tau_on → untouched.

    python -m scripts.mechinterp.analysis.suppress_skill_dreamer --n-maps 12 --n-traj 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from mechinterp.analysis.bundle import ActivationBundle  # noqa: E402
from mechinterp.analysis.suppress_skill_ppo import crossings, MODES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="activation_datasets/bt_dreamer")
    ap.add_argument("--probes", default="outputs/suppression/probes_bt_dreamer.npz")
    ap.add_argument("--n-maps", type=int, default=12)
    ap.add_argument("--n-traj", type=int, default=20)
    ap.add_argument("--tau-on", type=float, default=0.6)
    ap.add_argument("--tau-off", type=float, default=0.2)
    ap.add_argument("--probe-kind", choices=("sub", "full"), default="sub")
    ap.add_argument("--overshoot", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help=">0: also push along -alpha*unit(grad of skill-action logit wrt deter)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--leaf", action="store_true",
                    help="foliation leaf-transport suppression")
    ap.add_argument("--inlp", action="store_true",
                    help="leaf transport over the FULL INLP-exhausted transverse bundle")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/suppression"))
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    import pandas as pd
    import purejaxwm.dreamerv3.behavior as ac
    from cogniland.bridge_tunnel import tiles as T
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from cogniland.bridge_tunnel.jax import BridgeTunnelJaxEnv, EnvParams, constants as C
    from mechinterp.build_dreamer_activation_dataset import _build_model, _flatten_obs

    b = ActivationBundle(args.dataset)
    man = b.manifest
    ckpt = (_ROOT / man["checkpoint"]).resolve()
    cfg = json.loads((ckpt.parent.parent / "config.json").read_text())
    payload = ocp.PyTreeCheckpointer().restore(str(ckpt))
    wm_params, ac_params = payload["wm_params"], payload["ac_params"]
    encoder, rssm, actor, _ = _build_model(cfg)
    env = BridgeTunnelJaxEnv()
    max_steps = man["max_steps"]; n_traj = args.n_traj

    npz = np.load(args.probes)
    pre = "w_sub_" if args.probe_kind == "sub" else "w_"
    bre = "b_sub_" if args.probe_kind == "sub" else "b_"
    A_SKILL = {"tunnel": 5, "bridge": 4}             # mine, place
    CLS = {"free": 0, "avoid": 1, "bridge": 2, "tunnel": 3}

    def make_leaf_sup(skills):
        fol = np.load(Path(args.probes).parent / f"foliation_{Path(args.dataset).name}.npz")
        Wb = jnp.asarray(fol["W"], jnp.float32)            # boxer (4, D)
        bb = jnp.asarray(fol["b"], jnp.float32)
        if args.inlp:                                      # per-skill INLP bundle
            parts = [(CLS[s], jnp.asarray(fol[f"U_{s}"], jnp.float32),
                      jnp.asarray(fol[f"muav_{s}"], jnp.float32)) for s in skills]
        else:                                              # 3-dim LDA transverse
            parts = [(CLS[s], jnp.asarray(fol["B"], jnp.float32),
                      jnp.asarray(fol["mu"][CLS["avoid"]], jnp.float32)) for s in skills]

        def sup(deter, feat_rest):
            p = jax.nn.softmax(deter @ Wb.T + bb, axis=-1)
            trig = jnp.zeros(deter.shape[0]); dn = jnp.zeros(deter.shape[0])
            for c, U, mu in parts:
                onf = (p[:, c] > args.tau_on).astype(deter.dtype)
                d = (mu[None, :] - deter @ U.T) @ U        # transverse pin
                deter = deter + onf[:, None] * d
                trig = trig + onf; dn = dn + onf * jnp.linalg.norm(d, axis=-1)
            return deter, trig, dn
        return sup

    def make_sup(skills, actor_apply=None, stoch_of=None):
        parts = [(sk, jnp.asarray(npz[f"{pre}{sk}"], jnp.float32),
                  float(npz[f"{bre}{sk}"]),
                  float(npz[f"{pre}{sk}"].astype(np.float64) @
                        npz[f"{pre}{sk}"].astype(np.float64))) for sk in skills]
        l_on = float(np.log(args.tau_on / (1 - args.tau_on)))
        l_off = float(np.log(args.tau_off / (1 - args.tau_off)))

        def sup(deter, feat_rest):                   # (n,D),(n,R) -> (n,D),(n,),(n,)
            import jax
            trig = jnp.zeros(deter.shape[0]); dn = jnp.zeros(deter.shape[0])
            for sk, w, bb, ww in parts:
                logit = deter @ w + bb
                on = (logit > l_on).astype(deter.dtype)
                d = -args.overshoot * ((logit - l_off) / ww)[:, None] * w[None, :]
                if args.alpha > 0 and actor_apply is not None:
                    # per-step PGD: grad of the skill-action logit wrt deter,
                    # through the (nonlinear) actor MLP — state-dependent direction
                    def skill_logit(det1, rest1):
                        lg = actor_apply(jnp.concatenate([det1, rest1])[None])[0]
                        return lg[A_SKILL[sk]]
                    g = jax.vmap(jax.grad(skill_logit), in_axes=(0, 0))(deter, feat_rest)
                    gu = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
                    d = d - args.alpha * gu
                deter = deter + on[:, None] * d
                trig = trig + on; dn = dn + on * jnp.linalg.norm(d, axis=-1)
            return deter, trig, dn
        return sup

    def make_rollout(sup):
        def rollout(params, key):
            key, kr = jax.random.split(key)
            obs0, state0 = jax.vmap(env.reset_env, in_axes=(0, None))(
                jax.random.split(kr, n_traj), params)
            rssm_state = rssm.initial_state((n_traj,))
            last_action = jnp.zeros((n_traj, C.NUM_ACTIONS))
            last_is_first = jnp.ones((n_traj,), dtype=bool)
            done = jnp.zeros((n_traj,), dtype=bool)

            def step(carry, _):
                state, obs, rssm_state, last_action, last_is_first, done, key = carry
                am = jnp.where(last_is_first[..., None], 0.0, last_action)
                flat = _flatten_obs(obs)
                key, s_stoch, s_pol, s_step = jax.random.split(key, 4)
                embed = encoder.apply(wm_params["encoder"], flat)
                _, post = rssm.apply(wm_params["rssm"], rssm_state, am, embed,
                                     last_is_first, rngs={"stoch": s_stoch})
                trig = jnp.zeros(n_traj); dn = jnp.zeros(n_traj)
                if sup is not None:
                    rest = post.features()[:, post.deter.shape[-1]:]
                    det2, trig, dn = sup(post.deter, rest)
                    post = post._replace(deter=det2)
                logits = ac.unimix_logits(actor.apply(ac_params["actor"], post.features()))
                a = jax.random.categorical(s_pol, logits)
                nobs, nstate, _, done_next, info = jax.vmap(
                    env.step_env, in_axes=(0, 0, 0, None))(
                    jax.random.split(s_step, n_traj), state, a, params)

                def _sel(nx, pv):
                    msk = done.reshape(done.shape + (1,) * (nx.ndim - 1))
                    return jnp.where(msk, pv, nx)
                nstate = jax.tree_util.tree_map(_sel, nstate, state)
                nobs = jax.tree_util.tree_map(_sel, nobs, obs)
                out = {"pos_r": state.agent_r, "pos_c": state.agent_c,
                       "reached_now": info["reached_target"] & (~done),
                       "already_done": done, "trig": trig, "dn": dn}
                carry = (nstate, nobs, post, jax.nn.one_hot(a, C.NUM_ACTIONS),
                         jnp.zeros((n_traj,), bool), done | done_next, key)
                return carry, out

            carry = (state0, obs0, rssm_state, last_action, last_is_first, done, key)
            carry, outs = jax.lax.scan(step, carry, None, length=max_steps)
            outs["final_terrain"] = carry[0].terrain
            return outs
        return rollout

    # held-out maps: same split rule as the probe trainer
    m = b.maps
    rng = np.random.default_rng(0)
    u = np.arange(len(m["map_seed"]))
    tr_maps = set(rng.choice(u, int(0.7 * len(u)), replace=False).tolist())
    ev_maps = [i for i in u if i not in tr_maps][:args.n_maps]
    print(f"eval maps: {ev_maps}", flush=True)

    def params_for(mi):
        terrain, target = m["terrain"][mi], m["target"][mi]
        ctg = BridgeTunnelEnv._compute_ctg(terrain, tuple(target)).astype(np.float32)[None]
        return EnvParams.from_map_arrays(
            terrain=terrain[None], spawn=m["spawn"][mi][None], target=target[None],
            goal_mask=m["goal_mask"][mi][None], ctg=ctg,
            max_steps=cfg["max_steps"], view_size=cfg["view_size"],
            slack_penalty=cfg["slack_penalty"], reach_bonus=cfg["reach_bonus"],
            shaping_coef=cfg["shaping_coef"], build_cost=cfg["build_cost"],
            gamma=cfg["gamma"])

    actor_apply = lambda f: ac.unimix_logits(actor.apply(ac_params["actor"], f))  # noqa: E731
    mk = make_leaf_sup if args.leaf else (lambda sks: make_sup(sks, actor_apply))
    rollouts = {mode: jax.jit(make_rollout(
        None if mode == "none" else mk(
            ["tunnel", "bridge"] if mode == "both" else [mode])))
        for mode in MODES}

    rows, paths = [], {}
    import jax.random as jr
    for mode in MODES:
        for mi in ev_maps:
            outs = rollouts[mode](params_for(mi), jr.PRNGKey(int(m["map_seed"][mi])))
            o = {k: np.asarray(v) for k, v in outs.items()}
            terr0 = m["terrain"][mi]
            for i in range(n_traj):
                already = o["already_done"][:, i]; rn = o["reached_now"][:, i]
                L = (int(np.argmax(rn)) if rn.any() else int((~already).sum()) - 1) + 1
                path = list(zip(o["pos_r"][:L, i].tolist(), o["pos_c"][:L, i].tolist()))
                cr = crossings(path, terr0, T)
                ft = o["final_terrain"][i]
                mined_cells = np.argwhere((terr0 == T.ROCK) & (ft != T.ROCK))
                placed_cells = np.argwhere((terr0 == T.WATER) & (ft != T.WATER))
                rows.append(dict(mode=mode, map_id=int(mi), traj=i,
                                 reached=bool(rn.any()), steps=L,
                                 tunnel_cross=cr["tunnel"], bridge_cross=cr["bridge"],
                                 mined_rocks=len(mined_cells), placed_water=len(placed_cells),
                                 n_interventions=int(o["trig"][:L, i].sum()),
                                 mean_dnorm=float(o["dn"][:L, i].sum() / max(o["trig"][:L, i].sum(), 1))))
                paths[f"{mode}/{mi}/{i}"] = np.asarray(path, np.int16)
                paths[f"{mode}/{mi}/{i}/mined"] = mined_cells.astype(np.int16)
                paths[f"{mode}/{mi}/{i}/placed"] = placed_cells.astype(np.int16)
        df = pd.DataFrame([r for r in rows if r["mode"] == mode])
        print(f"[{mode:6s}] reach={df.reached.mean():.2f} "
              f"tunnel_cross/ep={df.tunnel_cross.mean():.2f} "
              f"bridge_cross/ep={df.bridge_cross.mean():.2f} "
              f"mined={df.mined_rocks.mean():.1f} placed={df.placed_water.mean():.1f} "
              f"interventions/ep={df.n_interventions.mean():.1f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    pd.DataFrame(rows).to_csv(args.out_dir / f"dreamer_metrics{tag}.csv", index=False)
    np.savez_compressed(args.out_dir / f"dreamer_rollouts{tag}.npz",
                        **paths, eval_maps=np.asarray(ev_maps))
    print("wrote", args.out_dir / f"dreamer_metrics{tag}.csv", "and rollouts npz")


if __name__ == "__main__":
    main()
