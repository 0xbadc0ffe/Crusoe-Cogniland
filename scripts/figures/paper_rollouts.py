#!/usr/bin/env python3
"""Roll out the three released agents on shared held-out maps.

Writes, per (agent, map):
  * an mp4 of the episode (upscaled tile render, trajectory trail, HUD)
  * a summary row in rollouts.json (success, steps, return, door taken)
and a combined trajectory figure per agent.

Agents live in three different environments, so this script is designed to be
called once per agent with the matching interpreter:

  # PPO   (conda crusoe)
  PYTHONPATH=src python scripts/figures/paper_rollouts.py --agent ppo
  # Dreamer (conda r2dreamer)
  PYTHONPATH=src:r2dreamer_model python scripts/figures/paper_rollouts.py --agent dreamer
  # STORM  (STORM_model/.venv, run from STORM_model/)
  PYTHONPATH=..:../src python scripts/figures/paper_rollouts.py --agent storm

Map selection is deterministic (--map-ids), so all three agents play the SAME
maps and the videos are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from cogniland.bridge_tunnel.tiles import TILE_COLORS  # noqa: E402

FORKWALL_KWARGS = dict(
    variant="btc", commit=False, fork_wall=True,
    categories=("balanced", "lakes", "rocky"),
    passage_half=1, wall_margin=1, mem_gap=16, shaping_gamma=1.0,
    size=32, width=64, view_size=21, max_steps=800,
    orientation="natural", tree_frac=0.03, goal_half=0,
    slack_penalty=-0.01, shaping_coef=0.015, reach_bonus=3.0,
    build_cost=0.0, commit_cost=0.05, illegal_penalty=0.02,
    gamma=0.99,
)
SCALE = 10          # px per tile
TRAIL = 26          # trail length in frames


# ── rendering ────────────────────────────────────────────────────────────

def render_frame(env, rec, traj, step, reward_sum, agent_name):
    """RGB frame: terrain + doors + fading trail + agent + HUD strip."""
    img = TILE_COLORS[env._terrain].copy()
    # doors
    for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
        good = rec.correct_target in ("either", name)
        for (r, c) in cells:
            img[r, c] = (34, 197, 94) if good else (239, 68, 68)
    # trail (older = dimmer)
    for i, (r, c) in enumerate(traj[-TRAIL:]):
        f = (i + 1) / min(len(traj), TRAIL)
        img[r, c] = (np.array(img[r, c]) * (1 - .55 * f) +
                     np.array((255, 255, 255)) * (.55 * f)).astype(np.uint8)
    r, c = env._pos
    img[r, c] = (255, 255, 255)
    big = np.kron(img, np.ones((SCALE, SCALE, 1), dtype=np.uint8))

    hud = np.zeros((26, big.shape[1], 3), dtype=np.uint8) + 17
    try:                                        # optional text overlay
        from PIL import Image, ImageDraw
        pil = Image.fromarray(hud)
        d = ImageDraw.Draw(pil)
        d.text((6, 3), f"{agent_name}   cat={rec.category}   "
                       f"step={step:3d}   return={reward_sum:+.2f}",
               fill=(235, 235, 235))
        d.text((6, 14), f"rewarded door: {rec.correct_target}", fill=(150, 200, 150))
        hud = np.asarray(pil)
    except Exception:
        pass
    return np.concatenate([big, hud], axis=0)


# ── agent adapters: each returns act(obs, done) -> int ───────────────────

def make_ppo(ckpt_path, sampled=True):
    import torch
    from cogniland.bridge_tunnel.policy import PPOGRUPolicy
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    probe = BridgeTunnelEnv(seed=0, **FORKWALL_KWARGS); probe.reset()
    policy = PPOGRUPolicy.from_checkpoint(ckpt, probe.observation_space)
    h = torch.zeros(1, 1, policy.gru_hidden)
    h_prev = [h]

    def reset():
        nonlocal h
        h = torch.zeros(1, 1, policy.gru_hidden)
        h_prev[0] = h

    hook = [None]        # optional intervention on the GRU state, see set_hook
    step = [0]

    def set_hook(fn):
        """fn(h_np, t, info) -> h_np, applied to the GRU state each step BEFORE
        the action is chosen, so the intervention actually affects behaviour."""
        hook[0] = fn
        step[0] = 0

    def act(obs, done):
        nonlocal h
        with torch.no_grad():
            t = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
            if hook[0] is None:
                a, _, _, _, h = policy.get_action_and_value(t, h, torch.zeros(1))
                if not sampled:                   # greedy: argmax of the logits
                    obs_seq = {k: v.unsqueeze(0) for k, v in t.items()}
                    gru_out, _ = policy._gru_forward(obs_seq, torch.zeros(1, 1), h_prev[0])
                    logits, _ = policy._heads(gru_out.squeeze(0))
                    a = logits.argmax(-1)
            else:
                # split the forward pass so the edit lands between the GRU and
                # the heads, and is carried forward by the recurrence
                obs_seq = {k: v.unsqueeze(0) for k, v in t.items()}
                _, h_new = policy._gru_forward(obs_seq, torch.zeros(1, 1), h)
                edited = hook[0](h_new.numpy().reshape(-1).astype(np.float32),
                                 step[0], {})
                h = torch.as_tensor(np.asarray(edited, dtype=np.float32)).reshape(h_new.shape)
                logits, _ = policy._heads(h.squeeze(0))
                a = (torch.distributions.Categorical(logits=logits).sample()
                     if sampled else logits.argmax(-1))
        step[0] += 1
        h_prev[0] = h
        return int(a.item())
    act.set_hook = set_hook

    # expose the carried state so evidence-integration analyses can read it
    act.get_state = lambda: h.detach().cpu().numpy().reshape(-1)
    # uniform accessor used by the activation-dataset builder
    act.get_features = lambda: {"h": h.detach().cpu().numpy().reshape(-1).astype(np.float16)}
    return act, reset


def make_storm(bundle, step, env_context=128, sampled=True):
    """STORM (storm2): rolling (z,a) context window, sampled actions."""
    from cl.config import setup_environment
    setup_environment()
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    import orbax.checkpoint as ocp
    from cl.agents import load_agent
    from cl.trainer.utils import RNGManager

    bundle = Path(bundle)
    cfg = OmegaConf.merge(OmegaConf.load(bundle / "run_config.yaml"), OmegaConf.create({
        "seed": 0,
        "agent": {"model": {"env_context": env_context}},
        "env": {"num_parallel_envs": 1, "num_parallel_envs_eval": 1},
    }))
    agent = load_agent(cfg)
    state = agent.init(RNGManager(seed=0).get_key())
    cands = sorted(bundle.glob("checkpoint*step_*"))
    if step is not None:
        cands = [c for c in cands if c.name.endswith(f"{step:08d}")]
    ck = ocp.StandardCheckpointer().restore(cands[-1].resolve())
    state = agent.state_from_checkpoint(ck, state.runtime)
    rng = jax.random.PRNGKey(0)
    prev = jnp.zeros((1, agent.action_space))
    first = True
    _last = [None]          # obs + is_first of the step about to be taken

    def set_seed(sd):
        """Reseed STORM's PRNG for the next episode.

        Two traps here. STORM samples through its own JAX key, which numpy and
        torch seeding do not touch; and `agent.act` reads that key from
        `runtime.rng`, ignoring the one passed to `select_action`. So the key
        that actually matters lives in the agent state and must be reset there,
        otherwise it advances across episodes and no episode can be replayed on
        its own."""
        nonlocal state, rng
        rng = jax.random.PRNGKey(int(sd))
        state = state.replace(
            runtime=state.runtime.replace(rng=jax.random.PRNGKey(int(sd))))

    def reset():
        nonlocal state, prev, first
        state = state.replace(runtime=state.runtime.replace(
            wm_state=agent.initial_wm_state(1)))
        prev = jnp.zeros((1, agent.action_space))
        first = True

    def act(obs, done):
        nonlocal state, prev, rng, first
        view = int(np.asarray(obs["minimap"]).shape[0])
        oh = np.zeros((view, view, 9), dtype=np.float32)
        rr, cc = np.indices((view, view))
        oh[rr, cc, np.asarray(obs["minimap"], dtype=np.int64)] = 1.0
        vec = np.concatenate([oh.reshape(-1),
                              np.asarray(obs["scalars"], dtype=np.float32)])
        rng, ar = jax.random.split(rng)
        obs_in = {"vector": jnp.asarray(vec)[None]}
        # capture the state BEFORE acting: select_action appends z_t to the
        # rolling context, so afterwards the transformer would see one token too
        # many and h would not be the vector the actor actually consumed.
        _last[0] = (state, obs_in, jnp.asarray([first]))
        a, state = agent.select_action(
            state, obs_in, ar,
            is_first=jnp.asarray([first]), prev_action=prev, training=sampled)
        prev = jax.nn.one_hot(a, agent.action_space)
        first = False
        return int(a[0])

    def _feats():
        """z and h exactly as the actor consumed them for the last action."""
        st_before, obs_in, isf = _last[0]
        z, h, _ = agent.features(st_before, obs_in, is_first=isf)
        z = np.asarray(z).reshape(agent.stoch_dim, agent.classes)
        return {"h": np.asarray(h).reshape(-1).astype(np.float16),
                "stoch_idx": z.argmax(-1).astype(np.int8)}
    act.get_features = _feats
    act.set_seed = set_seed
    return act, reset


def make_dreamer(ckpt_path, device="cuda", model_size="size25M", sampled=False):
    """R2-Dreamer, loaded exactly as scripts/bridge_tunnel/eval_forkwall_fixed.py does
    (hydra config + strict=False state dict); deterministic (eval=True) actions."""
    import gymnasium as gym
    import torch
    from hydra import compose, initialize_config_dir
    from tensordict import TensorDict
    sys.path.insert(0, str(REPO / "r2dreamer_model"))
    from dreamer import Dreamer

    cfg_dir = str((REPO / "r2dreamer_model/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="configs", overrides=[
            "env=bridge_tunnel_forkwall", "env.task=bridgetunnel_forkwall",
            f"model={model_size}", "model.rep_loss=dreamer",
            f"device={device}", "model.compile=False"])
    view, n_tiles, n_scalars = 21, 9, 5
    vd = view * view * n_tiles + n_scalars
    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(-np.inf, np.inf, (vd,), np.float32),
        "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "is_first": gym.spaces.Box(0, 1, (), bool),
        "is_last": gym.spaces.Box(0, 1, (), bool),
        "is_terminal": gym.spaces.Box(0, 1, (), bool)})

    class _OH(gym.spaces.Box):
        discrete = True

    agent = Dreamer(cfg.model, obs_space, _OH(0, 1, (6,), np.float32)).to(device)
    agent.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=False)["agent_state_dict"],
                          strict=False)
    agent.eval()
    st = [agent.get_initial_state(1)]
    first = [True]

    def reset():
        st[0] = agent.get_initial_state(1)
        first[0] = True

    def act(obs, done):
        oh = np.zeros((view, view, n_tiles), dtype=np.float32)
        rr, cc = np.indices((view, view))
        oh[rr, cc, np.asarray(obs["minimap"], dtype=np.int64)] = 1.0
        vec = np.concatenate([oh.reshape(-1),
                              np.asarray(obs["scalars"], dtype=np.float32)])
        trans = TensorDict({
            "vector": torch.as_tensor(vec, device=device, dtype=torch.float32)[None],
            "is_first": torch.tensor([first[0]], device=device)}, batch_size=(1,))
        with torch.no_grad():
            a, st[0] = agent.act(trans, st[0], eval=not sampled)
        first[0] = False
        return int(a.argmax(-1))

    # the RSSM deterministic path is the only part carried across time
    act.get_state = lambda: st[0]["deter"].detach().cpu().numpy().reshape(-1)

    def _feats():
        d = st[0]["deter"].detach().cpu().numpy().reshape(-1).astype(np.float16)
        z = st[0]["stoch"].detach().cpu().numpy()          # (1, S, K) one-hot
        # store the class index per slot: 32 int8 instead of 768 float16, lossless
        return {"deter": d, "stoch_idx": z.reshape(-1, z.shape[-1]).argmax(-1).astype(np.int8)}
    act.get_features = _feats
    return act, reset


# ── rollout driver ───────────────────────────────────────────────────────

def rollout(act, reset, rec, agent_name, out_mp4, fps=18, max_steps=800):
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    reset()
    frames, traj, ret = [], [env._pos], 0.0
    done = False
    for t in range(max_steps):
        frames.append(render_frame(env, rec, traj, t, ret, agent_name))
        a = act(obs, done)
        obs, r, term, trunc, info = env.step(a)
        ret += float(r)
        traj.append(env._pos)
        if term or trunc:
            done = True
            for _ in range(int(fps * 0.9)):          # hold the final frame
                frames.append(render_frame(env, rec, traj, t + 1, ret, agent_name))
            break
    success = env._pos in (env._correct_cells or set())
    if out_mp4:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out_mp4, frames, fps=fps, codec="libx264",
                         output_params=["-pix_fmt", "yuv420p", "-crf", "28"],
                         macro_block_size=1)
    return dict(agent=agent_name, category=rec.category, success=bool(success),
                steps=len(traj) - 1, ret=round(ret, 3),
                timeout=bool(len(traj) - 1 >= max_steps),
                traj=[(int(r), int(c)) for r, c in traj])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["ppo", "dreamer", "storm"])
    p.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    p.add_argument("--map-ids", default="", help="comma-separated indices into the pool")
    p.add_argument("--per-cat", type=int, default=1, help="maps per category if no ids")
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    p.add_argument("--ppo-ckpt", default=str(REPO / "final_models/ppo/ppo_plain.pt"))
    p.add_argument("--storm-bundle", default=str(REPO / "final_models/storm"))
    p.add_argument("--storm-step", type=int, default=624489)
    p.add_argument("--dreamer-ckpt", default=str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"))
    p.add_argument("--dreamer-size", default="size25M")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(args.maps, "rb") as f:
        pool = pickle.load(f)

    if args.map_ids:
        ids = [int(x) for x in args.map_ids.split(",")]
    else:                                    # first N of each category, deterministic
        ids, seen = [], {c: 0 for c in ("lakes", "balanced", "rocky")}
        for i, r in enumerate(pool):
            if r.category in seen and seen[r.category] < args.per_cat:
                ids.append(i); seen[r.category] += 1
            if all(v >= args.per_cat for v in seen.values()):
                break
    print("map ids:", ids)

    if args.agent == "ppo":
        act, reset = make_ppo(args.ppo_ckpt)
    elif args.agent == "storm":
        act, reset = make_storm(args.storm_bundle, args.storm_step)
    else:
        act, reset = make_dreamer(args.dreamer_ckpt, args.device, args.dreamer_size)

    rows = []
    for i in ids:
        rec = pool[i]
        mp4 = out / "videos" / f"{args.agent}_map{i}_{rec.category}.mp4"
        row = rollout(act, reset, rec, args.agent.upper(), mp4)
        row["map_id"] = i
        rows.append(row)
        print(f"map {i:5d} {rec.category:9s} success={row['success']!s:5s} "
              f"steps={row['steps']:3d} return={row['ret']:+.2f}  -> {mp4.name}")

    jf = out / f"rollouts_{args.agent}.json"
    jf.write_text(json.dumps(rows, indent=1))
    print("wrote", jf)


if __name__ == "__main__":
    main()
