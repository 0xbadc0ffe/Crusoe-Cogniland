#!/usr/bin/env python3
"""Render a warmup→imagination video, optionally FORCING a tool action during the
imagined phase, to test/visualize whether the world model imagines a tunnel
(mine: rock→grass), a bridge (build: water→wood), or reaching the target.

--force {build,mine}  forces that action every imagined step (after warming up
                      until the agent genuinely faces the matching obstacle);
                      omit --force to imagine open-loop under the actor (for reach).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import numpy as np, jax, jax.numpy as jnp
import orbax.checkpoint as ocp
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts" / "bridge_tunnel"))
import viz_dreamer_bridge_tunnel_imagine as VZ
import purejaxwm.dreamerv3.behavior as ac
from purejaxwm.dreamerv3.distributions import TwoHotDist
from cogniland.bridge_tunnel.jax import BridgeTunnelJaxEnv, constants as C

DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
OBST = {"build": 1, "mine": 2}        # build faces water(1), mine faces rock(2)
AIDX = {"build": 4, "mine": 5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--category", default="rocky")
    p.add_argument("--seed", type=int, default=10001)
    p.add_argument("--force", choices=("build", "mine"), default=None)
    p.add_argument("--warmup", type=int, default=60, help="max real steps")
    p.add_argument("--horizon", type=int, default=18)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--sprite-px", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    ck = a.checkpoint.resolve()
    cfg = json.loads((ck.parent.parent / "config.json").read_text())
    VZ._DECODER_MODE = cfg.get("decoder", "categorical")
    VZ.ACTION_NAMES = (["up", "down", "left", "right", "build", "mine"]
                       if cfg.get("env_id") == "bridge_tunnel_commit"
                       else ["up", "down", "left", "right", "place", "mine"])
    pay = ocp.PyTreeCheckpointer().restore(str(ck))
    wm = jax.tree_util.tree_map(jnp.asarray, pay["wm_params"])
    acp = jax.tree_util.tree_map(jnp.asarray, pay["ac_params"])
    enc, dec, rssm, actor, rew = VZ._build_model(cfg)
    env = BridgeTunnelJaxEnv(); V = cfg["view_size"]
    params, rec = VZ._single_map_params(a.seed, cfg, a.category); terr = np.asarray(rec.terrain)
    decode = lambda feat: VZ._decode_to_tiles(np.asarray(dec.apply(wm["decoder"], feat)).reshape(-1), V)

    k = jax.random.PRNGKey(0); k, kr = jax.random.split(k)
    obs, st = env.reset_env(kr, params); obs = jax.tree_util.tree_map(lambda x: x[None], obs)
    rs = rssm.initial_state((1,)); la = jnp.zeros((1, C.NUM_ACTIONS)); first = jnp.ones((1,), bool)
    frames = []
    want_obst = OBST.get(a.force); started = a.force is None      # if no force, imagine immediately after fixed warmup

    for t in range(a.warmup):
        k, ks = jax.random.split(k)
        embed = enc.apply(wm["encoder"], VZ._flatten_obs(obs))
        _, post = rssm.apply(wm["rssm"], rs, la, embed, first, rngs={"stoch": ks})
        feat = post.features()
        act = int(jnp.argmax(ac.unimix_logits(actor.apply(acp["actor"], feat))))
        frames.append({"phase": "warmup", "tiles": decode(feat), "action": act, "reward": 0.0})
        dr, dc = DELTA[int(st.facing)]; fr, fc = int(st.agent_r) + dr, int(st.agent_c) + dc
        ahead = int(terr[fr, fc]) if (0 <= fr < terr.shape[0] and 0 <= fc < terr.shape[1]) else -1
        if a.force is not None and ahead == want_obst and int(st.commit) == 0:
            started = True; break                                # warmup ends facing the obstacle
        k, kstep = jax.random.split(k)
        obs2, st, _, done, _ = env.step_env(kstep, st, act, params)
        obs = jax.tree_util.tree_map(lambda x: x[None], obs2)
        la = jax.nn.one_hot(jnp.array([act]), C.NUM_ACTIONS); first = jnp.zeros((1,), bool); rs = post
        if bool(done):
            break

    rs_i = post
    for s in range(a.horizon):
        k, ki = jax.random.split(k)
        if a.force is not None:
            act = AIDX[a.force]
        else:
            act = int(jnp.argmax(ac.unimix_logits(actor.apply(acp["actor"], rs_i.features()))))
        prior = rssm.apply(wm["rssm"], rs_i, jax.nn.one_hot(jnp.array([act]), C.NUM_ACTIONS),
                           None, None, False, rngs={"stoch": ki})
        feat = prior.features()
        r = float(TwoHotDist(rew.apply(wm["reward"], feat)).mean()[0])
        frames.append({"phase": "open-loop", "tiles": decode(feat), "action": act, "reward": r})
        rs_i = prior

    facings, cur = [], 3
    for f in frames:
        if f["action"] < 4:
            cur = f["action"]
        facings.append(cur)
    sprites = VZ._load_sprite_imgs(a.sprite_px)
    rgb = [VZ._frame_to_rgb(f, 12, sprites, a.sprite_px, fc) for f, fc in zip(frames, facings)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    path, fmt = VZ._write_video(rgb, a.out.with_suffix(""), a.fps)
    print(f"wrote {path}  ({len([f for f in frames if f['phase']=='warmup'])} real + "
          f"{len([f for f in frames if f['phase']=='open-loop'])} imagined frames)")


if __name__ == "__main__":
    main()
