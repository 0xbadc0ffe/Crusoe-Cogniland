#!/usr/bin/env python3
"""Act three, part B1 for STORM: the transformer forecasting its own tool use.

Same claim as the Dreamer half -- a world model does not need a trained probe
to answer "will I mine soon", it can imagine -- but through a transformer world
model, whose imagination must be primed with the real context window (storm2's
IMAGINATION section: rollouts inherit real history, never a 1-token call).

At each step of a replayed episode we take the agent's live (z, a) window,
imagine M continuations of K steps under the frozen actor exactly as storm2's
imag_step does, and record the fraction whose imagined actions contain MINE
(resp. BUILD). Calibration = AUC of that forecast against what the episode
actually did in the next K steps.

STATUS: DOES NOT PASS ITS SANITY CHECK -- do not report numbers from this file.
At K=1 the forecast collapses to the actor's own probability of taking the tool
action next, so its AUC against the realised next action should be high; it
measures 0.55 (chance), both when step 0 is seeded from the prior and when it
is seeded from the true posterior. Leading suspect: action selection samples z
from the posterior (`_sample_z(post_logits, r_z)`) while `features()` returns
the posterior MODE, so the feature vector given to the imagined actor is not
the one the real actor consumed. Fix that, re-run --K 1, and only trust the
K=15 numbers once the K=1 AUC is high.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:STORM_model:scripts/mechinterp:scripts/figures \
    STORM_model/.venv/bin/python scripts/mechinterp/behavior_steering/act3_storm.py --n-calib 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/mechinterp", "scripts/figures"):
    sys.path.insert(0, str(REPO / p))

OUT = REPO / "outputs/behavior_steering/act3"
A_BUILD, A_MINE = 4, 5


def build_forecaster(agent):
    """-> fn(state, rng, M, K) -> (P_mine, P_build): storm2's imagination at
    inference. Mirrors `_loss_fn`'s imag_step -- the live (z, a) window is the
    context, the transformer is re-run over context+imagined tokens each step,
    and h at the last written position drives the prior, the actor and the
    heads.

    The buffer has a FIXED length (W + K + 1) and the context length is passed
    as a traced value, so the scan compiles once instead of once per step
    (ctx_len grows every step early in an episode)."""
    import functools

    import jax
    import jax.numpy as jnp

    W = agent.env_context

    @functools.partial(jax.jit, static_argnums=(5, 6))
    def _roll(params, wm_state, rng, ctx_len, z0, M, K):
        wm_params, pol = params.wm, params.policy
        # the transformer's position table is exactly W long, so the buffer
        # stays at W: the last K slots hold the imagined tokens and the context
        # keeps its most recent (W - K) tokens (causal attention, so the recent
        # ones are the ones that matter). Episodes here rarely fill the window,
        # in which case nothing is dropped at all.
        C = W - K
        c_eff = jnp.minimum(ctx_len, C)
        start = ctx_len - c_eff
        z_ctx = jax.lax.dynamic_slice(wm_state.z_ctx, (0, start, 0),
                                      (1, C, agent.stoch_flat))
        a_ctx = jax.lax.dynamic_slice(wm_state.a_ctx, (0, start), (1, C))
        bz = jnp.zeros((M, W, agent.stoch_flat))
        ba = jnp.zeros((M, W), dtype=jnp.int32)
        bz = bz.at[:, :C].set(jnp.repeat(z_ctx, M, axis=0))
        ba = ba.at[:, :C].set(jnp.repeat(a_ctx, M, axis=0))
        causal = jnp.tril(jnp.ones((W, W), dtype=bool))
        pos = jnp.arange(W)

        def step(carry, inp):
            bz, ba = carry
            i, rng_i = inp
            r1, r2 = jax.random.split(rng_i)
            # a key is valid if it is a real context token or one imagined so far
            n_valid = (c_eff + i).reshape(())
            mask = (causal & (pos[None, :] < n_valid))[None]
            feats = agent._transformer_fwd(wm_params, bz, ba, mask)
            h = jnp.take_along_axis(
                feats, jnp.broadcast_to((n_valid - 1).reshape(1, 1, 1),
                                        (M, 1, feats.shape[-1])), axis=1)[:, 0]
            # the CURRENT step is not imagined: the agent has really observed
            # it, so use the true posterior z_t. Only future steps come from
            # the prior. (Seeding step 0 from the prior instead makes the
            # forecast disagree with the actor's own next-action distribution,
            # which the K=1 diagnostic catches.)
            _, z_prior = agent._sample_z(agent._prior_logits(wm_params, h), r1)
            zf = jnp.where(i == 0, jnp.broadcast_to(z0, z_prior.shape), z_prior)
            feat = jnp.concatenate([zf, h], axis=-1)
            a_idx = agent.policy.apply_actor(pol.actor, feat,
                                             training=True).sample(seed=r2)
            a_idx = jnp.reshape(a_idx, (M,)).astype(jnp.int32)
            bz = jax.lax.dynamic_update_slice(bz, zf[:, None, :], (0, n_valid, 0))
            ba = jax.lax.dynamic_update_slice(ba, a_idx[:, None], (0, n_valid))
            return (bz, ba), a_idx

        rngs = jax.random.split(rng, K)
        _, acts = jax.lax.scan(step, (bz, ba), (jnp.arange(K), rngs))
        acts = acts.T                                          # [M, K]
        return (jnp.mean(jnp.any(acts == A_MINE, axis=1)),
                jnp.mean(jnp.any(acts == A_BUILD, axis=1)))

    def forecast(state, rng, z0, M=8, K=15):
        wm_state = state.runtime.wm_state
        cl = wm_state.ctx_len.reshape(-1)[0]
        if int(cl) < 1:
            return float("nan"), float("nan")
        pm, pb = _roll(state.train_state.params, wm_state, rng, cl,
                       jnp.asarray(z0).reshape(1, -1), M, K)
        return float(pm), float(pb)

    return forecast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=8)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--K", type=int, default=15)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import jax
    import replay_episode as RE
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS

    act, reset = RE._get_agent("storm", "cuda")
    agent = act.agent
    forecast = build_forecaster(agent)
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    BEL = np.load(REPO / "outputs/belief_report/steer_axis_storm.npz")

    P = {"mine": [], "build": []}
    Y = {"mine": [], "build": []}
    rows = []
    todo = ([("rocky", int(m)) for m in BEL["rocky"][:a.n_calib]] +
            [("lakes", int(m)) for m in BEL["lakes"][:a.n_calib]])
    key = jax.random.PRNGKey(0)
    for cat, mid in todo:
        seed = int(RE.episode_meta("storm", mid)["seed"])
        np.random.seed(seed)
        act.set_hook(None); act.set_logit_bias(None); act.set_seed(seed)
        env = BridgeTunnelEnv(seed=0, map_record=pool[mid], **FORKWALL_KWARGS)
        obs, _ = env.reset(); reset()
        acts, fc = [], []
        for t in range(FORKWALL_KWARGS["max_steps"]):
            aa = act(obs, False)
            st, obs_in, isf = act.last[0]  # the state the action was taken from
            z0, _, _ = agent.features(st, obs_in, is_first=isf)
            key, sub = jax.random.split(key)
            fc.append(forecast(st, sub, z0, a.M, a.K))
            acts.append(int(aa))
            obs, _, term, trunc, _ = env.step(aa)
            if term or trunc:
                break
        acts = np.array(acts)
        for tool, tid, j in (("mine", A_MINE, 0), ("build", A_BUILD, 1)):
            ev = (acts == tid).astype(int)
            for t in range(len(acts)):
                hi = min(t + 1 + a.K, len(acts))
                Y[tool].append(int(ev[t + 1:hi].sum() > 0))
                P[tool].append(fc[t][j])
        rows.append(dict(cat=cat, map_id=mid, steps=len(acts),
                         mines=int((acts == A_MINE).sum()),
                         builds=int((acts == A_BUILD).sum())))
        print(f"calib {cat:6s} map {mid:4d} steps {len(acts):3d} "
              f"mines {rows[-1]['mines']:3d} builds {rows[-1]['builds']:3d}",
              flush=True)

    def auc_of(y, p):
        """Rank-based AUC, so this script does not need sklearn (STORM's venv
        has no scikit-learn and installing into it is out of scope)."""
        order = np.argsort(p, kind="mergesort")
        r = np.empty(len(p), float)
        sp = p[order]
        i = 0
        rank = np.arange(1, len(p) + 1, dtype=float)
        while i < len(sp):                      # average ranks within ties
            j = i
            while j + 1 < len(sp) and sp[j + 1] == sp[i]:
                j += 1
            rank[i:j + 1] = rank[i:j + 1].mean()
            i = j + 1
        ry = np.empty(len(p), float)
        ry[order] = rank
        n1 = float(y.sum()); n0 = float(len(y) - n1)
        if n1 == 0 or n0 == 0:
            return None
        return float((ry[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    out = dict(K=a.K, M=a.M, n_episodes=len(rows), rows=rows)
    for tool in ("mine", "build"):
        y = np.array(Y[tool]); p = np.array(P[tool])
        ok = np.isfinite(p)
        y, p = y[ok], p[ok]
        out[tool] = dict(n=int(len(y)), base_rate=round(float(y.mean()), 4),
                         auc=(round(auc_of(y, p), 4)
                              if 0 < y.mean() < 1 else None),
                         corr=(round(float(np.corrcoef(y, p)[0, 1]), 4)
                               if y.std() > 0 and p.std() > 0 else None))
        print(tool, out[tool], flush=True)
    (OUT / "storm_calib.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
