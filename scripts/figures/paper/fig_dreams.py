#!/usr/bin/env python3
"""Dreamed futures: what the world models *imagine* will happen.

Both DreamerV3 and STORM plan inside a learned model. This script exposes that
model directly: it feeds a few real steps as context, then lets the model roll
forward on its own -- no further observations -- decoding each imagined latent
back into a 21x21 observation and rendering it with the Crafter tiles.

Output per agent:
  fig_dream_<agent>.png   filmstrip: real context | ground truth vs dream
  dreams_<agent>.json     per-step tile agreement between dream and reality

The comparison is the point: the top row is what really happened, the bottom row
is what the model believed would happen having last seen the context frame.

  PYTHONPATH=src:r2dreamer_model python scripts/figures/paper/fig_dreams.py --agent dreamer
  (from STORM_model/) PYTHONPATH=.:..:../src python ../scripts/figures/paper/fig_dreams.py --agent storm
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygame

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import text as TXT  # noqa: E402

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from paper_rollouts import FORKWALL_KWARGS  # noqa: E402
from fig_task import obs_rgb  # noqa: E402
from paper_rollouts_textured import load_sprites  # noqa: E402

VIEW, NTILE, NSCAL = 21, 9, 5
VEC = VIEW * VIEW * NTILE + NSCAL


def to_vec(raw):
    oh = np.zeros((VIEW, VIEW, NTILE), dtype=np.float32)
    rr, cc = np.indices((VIEW, VIEW))
    oh[rr, cc, np.asarray(raw["minimap"], dtype=np.int64)] = 1.0
    return np.concatenate([oh.reshape(-1), np.asarray(raw["scalars"], np.float32)])


def vec_to_tiles(vec):
    """Decoded observation vector -> (21,21) tile ids by argmax over channels."""
    grid = np.asarray(vec[: VIEW * VIEW * NTILE], dtype=np.float32)
    return grid.reshape(VIEW, VIEW, NTILE).argmax(-1).astype(np.int64)


# ── DreamerV3 ────────────────────────────────────────────────────────────

def dream_dreamer(rec, ckpt, device, size, context, horizon):
    import gymnasium as gym
    import torch
    from hydra import compose, initialize_config_dir
    from tensordict import TensorDict
    sys.path.insert(0, str(REPO / "r2dreamer_model"))
    from dreamer import Dreamer

    with initialize_config_dir(version_base=None,
                               config_dir=str((REPO / "r2dreamer_model/configs").resolve())):
        cfg = compose(config_name="configs", overrides=[
            "env=bridge_tunnel_forkwall", "env.task=bridgetunnel_forkwall",
            f"model={size}", "model.rep_loss=dreamer", f"device={device}",
            "model.compile=False"])
    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(-np.inf, np.inf, (VEC,), np.float32),
        "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "is_first": gym.spaces.Box(0, 1, (), bool),
        "is_last": gym.spaces.Box(0, 1, (), bool),
        "is_terminal": gym.spaces.Box(0, 1, (), bool)})

    class _OH(gym.spaces.Box):
        discrete = True

    agent = Dreamer(cfg.model, obs_space, _OH(0, 1, (6,), np.float32)).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["agent_state_dict"], strict=False)
    agent.eval()

    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    raw, _ = env.reset()
    st = agent.get_initial_state(1)
    real, acts = [], []
    with torch.no_grad():
        for t in range(context):                       # burn in on real frames
            real.append(np.asarray(raw["minimap"]).copy())
            trans = TensorDict({
                "vector": torch.as_tensor(to_vec(raw), device=device,
                                          dtype=torch.float32)[None],
                "is_first": torch.tensor([t == 0], device=device)}, batch_size=(1,))
            a, st = agent.act(trans, st, eval=True)
            acts.append(a)
            raw, *_ = env.step(int(a.argmax(-1)))

        # from here the model is on its own: prior rollout, policy-chosen actions
        stoch, deter = st["stoch"], st["deter"]
        dream, truth = [], []
        for h in range(horizon):
            feat = agent._frozen_rssm.get_feat(stoch, deter)
            a = agent._frozen_actor(feat).mode
            stoch, deter = agent._frozen_rssm.img_step(stoch, deter, a)
            feat = agent._frozen_rssm.get_feat(stoch, deter)
            dec = agent.decoder(stoch, deter)["vector"]
            v = getattr(dec, "mode", None)
            v = v() if callable(v) else (v if v is not None else dec.mean)
            v = np.asarray(v.detach().cpu().numpy()).reshape(-1)
            dream.append(vec_to_tiles(v))
            truth.append(np.asarray(raw["minimap"]).copy())
            raw, _, term, trunc, _ = env.step(int(a.argmax(-1)))
            if term or trunc:
                break
    return real, truth, dream


# ── STORM ────────────────────────────────────────────────────────────────

def dream_storm(rec, bundle, step, context, horizon, env_context=128):
    from cl.config import setup_environment
    setup_environment()
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from omegaconf import OmegaConf
    from cl.agents import load_agent
    from cl.trainer.utils import RNGManager

    bundle = Path(bundle)
    cfg = OmegaConf.merge(OmegaConf.load(bundle / "run_config.yaml"), OmegaConf.create({
        "seed": 0, "agent": {"model": {"env_context": env_context}},
        "env": {"num_parallel_envs": 1, "num_parallel_envs_eval": 1}}))
    agent = load_agent(cfg)
    state = agent.init(RNGManager(seed=0).get_key())
    cands = sorted(bundle.glob("checkpoint*step_*"))
    if step is not None:
        cands = [c for c in cands if c.name.endswith(f"{step:08d}")]
    ck = ocp.StandardCheckpointer().restore(cands[-1].resolve())
    state = agent.state_from_checkpoint(ck, state.runtime)
    wm = state.train_state.params.wm

    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    raw, _ = env.reset()
    rng = jax.random.PRNGKey(0)
    prev = jnp.zeros((1, agent.action_space))
    first = True
    real, z_hist, a_hist = [], [], []

    for t in range(context):                            # real context
        real.append(np.asarray(raw["minimap"]).copy())
        vec = jnp.asarray(to_vec(raw))[None]
        emb = agent._encode(wm, {"vector": vec})
        _, zf = agent._sample_z(agent._post_logits(wm, emb), None)
        z_hist.append(zf[0])
        rng, ar = jax.random.split(rng)
        a, state = agent.select_action(state, {"vector": vec}, ar,
                                       is_first=jnp.asarray([first]),
                                       prev_action=prev, training=False)
        prev = jax.nn.one_hot(a, agent.action_space)
        first = False
        a_hist.append(int(a[0]))
        raw, *_ = env.step(int(a[0]))

    # imagination: extend the token sequence with the model's own predictions
    zs = jnp.stack(z_hist)[None]                        # (1, C, S)
    as_ = jnp.asarray(a_hist, dtype=jnp.int32)[None]    # (1, C)
    dream, truth = [], []
    for h in range(horizon):
        L = zs.shape[1]
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None]
        feats = agent._transformer_fwd(wm, zs, as_, mask)
        hstate = feats[:, -1]
        rng, r1, r2 = jax.random.split(rng, 3)
        _, zf = agent._sample_z(agent._prior_logits(wm, hstate), r1)
        feat = jnp.concatenate([zf, hstate], axis=-1)
        a_dist = agent.policy.apply_actor(state.train_state.params.policy.actor,
                                          feat, training=False)
        a_idx = a_dist.mode()
        recon = agent.decoder.apply(wm.decoder, {
            "deter": zf, "stoch": zf.reshape(1, agent.stoch_dim, agent.classes)})
        v = np.asarray(recon["vector"].mean).reshape(-1)
        dream.append(vec_to_tiles(v))
        truth.append(np.asarray(raw["minimap"]).copy())
        zs = jnp.concatenate([zs, zf[:, None]], axis=1)
        as_ = jnp.concatenate([as_, jnp.asarray(a_idx, jnp.int32).reshape(1, 1)], axis=1)
        raw, _, term, trunc, _ = env.step(int(a_idx[0]))
        if term or trunc:
            break
    return real, truth, dream


# ── figure ───────────────────────────────────────────────────────────────

def make_fig(agent, real, truth, dream, out, context, stride=3):
    pygame.init(); pygame.display.set_mode((1, 1))
    sprites = load_sprites(8)
    idx = list(range(0, len(dream), stride))[:6]
    ncol = 1 + len(idx)

    rc = {"figure.dpi": 140, "savefig.dpi": 140, "font.size": 8.5, "axes.titlesize": 8.5}
    agree = [float((t == d).mean()) for t, d in zip(truth, dream)]
    with plt.rc_context(rc):
        fig, ax = plt.subplots(2, ncol, figsize=(1.72 * ncol, 4.0))
        ax[0, 0].imshow(obs_rgb(real[-1], 3, sprites, 8), interpolation="nearest")
        ax[0, 0].set_title(TXT.FIG_DREAMS["context"].format(n=context), loc="left", fontsize=8)
        ax[1, 0].axis("off")
        ax[1, 0].text(.5, .5, TXT.FIG_DREAMS["cut"],
                      ha="center", va="center", fontsize=8, color="#6d7a70",
                      transform=ax[1, 0].transAxes)
        for k, i in enumerate(idx):
            ax[0, k + 1].imshow(obs_rgb(truth[i], 3, sprites, 8), interpolation="nearest")
            ax[0, k + 1].set_title(TXT.FIG_DREAMS["step"].format(i=i + 1), loc="left", fontsize=8)
            ax[1, k + 1].imshow(obs_rgb(dream[i], 3, sprites, 8), interpolation="nearest")
            ax[1, k + 1].set_xlabel(TXT.FIG_DREAMS["agreement"].format(pct=agree[i] * 100), fontsize=7.5,
                                    color="#6d7a70")
        for a in ax.flat:
            a.set_xticks([]); a.set_yticks([])
        ax[0, 0].set_ylabel(TXT.FIG_DREAMS["row_real"], fontsize=9)
        ax[1, 1].set_ylabel(TXT.FIG_DREAMS["row_dream"], fontsize=9)
        fig.suptitle(TXT.FIG_DREAMS["title"].format(AGENT=agent.upper()),
                     y=1.0, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .96])
        fig.savefig(out / f"fig_dream_{agent}.png", bbox_inches="tight")
        plt.close(fig)
    return agree


def make_video(agent, real, truth, dream, agree, out, context, fps=4, tp=16):
    """A rollout that turns into a dream.

    The clip opens as an ordinary episode: both panes show the same real
    observation, step by step. At the cut the right pane stops receiving
    observations and runs on the model's own dynamics, so the two panes start
    identical and visibly drift apart. That reads far better than a row of
    stills, which cannot show the moment the divergence begins.
    """
    import imageio.v2 as imageio
    pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    sprites = load_sprites(tp)
    V = real[-1].shape[0]
    pane, pad, hud = V * tp, 12, 50
    W, H = pad * 3 + pane * 2, hud + pane + 30
    fnt = pygame.font.SysFont("dejavusans", 15)
    fnt_s = pygame.font.SysFont("dejavusans", 11)
    surf = pygame.Surface((W, H))
    INK, DIM = (232, 236, 226), (140, 152, 138)
    LIVE, DREAM = (110, 190, 130), (232, 170, 90)

    def frame(img_l, img_r, title, sub, lab_r, col_r, flash=False):
        surf.fill((14, 17, 14))
        surf.blit(fnt.render(title, True, INK), (pad, 8))
        surf.blit(fnt_s.render(sub, True, DIM), (pad, 30))
        for x, im, col in ((pad, img_l, LIVE), (pad * 2 + pane, img_r, col_r)):
            s2 = pygame.surfarray.make_surface(np.transpose(im, (1, 0, 2)))
            surf.blit(s2, (x, hud))
            pygame.draw.rect(surf, col, (x - 2, hud - 2, pane + 4, pane + 4),
                             3 if flash else 1)
        surf.blit(fnt_s.render("reality", True, LIVE), (pad, hud + pane + 7))
        surf.blit(fnt_s.render(lab_r, True, col_r), (pad * 2 + pane, hud + pane + 7))
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy()

    frames = []
    n = len(real)
    # phase 1 -- an ordinary rollout, both panes fed the same observations
    for t, obs in enumerate(real):
        img = obs_rgb(obs, 3, sprites, tp)
        frames.append(frame(img, img,
                            f"{agent.upper()} — real rollout",
                            f"step {t + 1} of {n} · the model is still observing",
                            "model input", LIVE))
    # the cut -- hold on the last real frame and say what is about to happen
    img = obs_rgb(real[-1], 3, sprites, tp)
    for k in range(fps * 2):
        frames.append(frame(img, img, f"{agent.upper()} — observations stop here",
                            "from now the right pane runs on the model's own dynamics",
                            "imagination begins", DREAM, flash=(k // 2) % 2 == 0))
    # phase 2 -- reality continues; the right pane is imagined
    for i in range(len(dream)):
        frames.append(frame(
            obs_rgb(truth[i], 3, sprites, tp), obs_rgb(dream[i], 3, sprites, tp),
            f"{agent.upper()} — dreaming, +{i + 1}",
            f"no observations for {i + 1} step{'s' if i else ''} · "
            f"tile agreement {agree[i] * 100:.0f}%",
            "dream", DREAM))
    frames += [frames[-1]] * (fps * 2)

    mp4 = out / "videos_textured" / f"dream_{agent}.mp4"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(mp4, frames, fps=fps, codec="libx264",
                     output_params=["-pix_fmt", "yuv420p", "-crf", "24"],
                     macro_block_size=1)
    print(f"  wrote {mp4.name}  ({n} real + {len(dream)} imagined steps)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["dreamer", "storm"])
    p.add_argument("--map-id", type=int, default=16)
    p.add_argument("--context", type=int, default=24)
    p.add_argument("--horizon", type=int, default=18)
    p.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    p.add_argument("--storm-bundle", default=str(REPO / "final_models/storm"))
    p.add_argument("--storm-step", type=int, default=624489)
    p.add_argument("--dreamer-ckpt", default=str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"))
    p.add_argument("--dreamer-size", default="size25M")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(a.maps, "rb") as f:
        rec = pickle.load(f)[a.map_id]

    if a.agent == "dreamer":
        real, truth, dream = dream_dreamer(rec, a.dreamer_ckpt, a.device,
                                           a.dreamer_size, a.context, a.horizon)
    else:
        real, truth, dream = dream_storm(rec, a.storm_bundle, a.storm_step,
                                         a.context, a.horizon)
    agree = make_fig(a.agent, real, truth, dream, out, a.context)
    make_video(a.agent, real, truth, dream, agree, out, a.context)
    (out / f"dreams_{a.agent}.json").write_text(json.dumps(
        {"map_id": a.map_id, "context": a.context, "agreement": agree}))
    print(f"{a.agent}: {len(dream)} imagined steps, tile agreement "
          f"{agree[0]*100:.0f}% -> {agree[-1]*100:.0f}%")


if __name__ == "__main__":
    main()
