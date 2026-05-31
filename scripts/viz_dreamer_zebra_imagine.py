#!/usr/bin/env python3
"""Render VIDEOS of DreamerV3's IMAGINED (latent-imagination / "dream") rollouts
for the zebra_nav agent.

An "imagined trajectory" warms up the RSSM posterior on a few REAL observations
(the agent actually stepping the env), then rolls the world model FORWARD IN
LATENT SPACE under the actor — no env interaction — decoding each imagined
latent back to a reconstructed egocentric minimap. The decoded sequence is
played as a video so you can eyeball world-model quality.

It reconstructs the world model + actor + decoder + reward-head EXACTLY as
``dreamerv3_zebra_nav.py`` built them (config from ``<run>/config.json``),
restores the orbax PyTree checkpoint (params only), and:

  1. resets the real env on an eval map and steps the actor for ``--warmup``
     REAL steps, carrying the posterior RSSM state + collecting decoded real obs;
  2. from that latent state, IMAGINES ``--horizon`` steps open-loop:
        action = argmax actor(feat)  →  RSSM imagine-step (prior only, embed=None)
        →  next latent  →  decode(feat) → reconstructed obs ; predicted reward
        via the reward head's TwoHotDist mean;
  3. renders an animation (one panel) that runs the REAL decoded frames then the
     IMAGINED decoded frames, each titled REAL/IMAGINED + step + action + reward.

Outputs per example (j = 0..n-1, eval map seed = eval_seed_start + j):
  <out-dir>/imagine_seed<seed>.mp4   (falls back to .gif if mp4 unavailable)
  <out-dir>/imagine_strip_seed<seed>.png   (filmstrip of every Nth frame)

    python scripts/viz_dreamer_zebra_imagine.py \\
        --checkpoint runs/dreamer_natural_wholewall/checkpoints/step_1000000 \\
        --n-examples 4 --warmup 10 --horizon 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# GPU courtesy: don't grab the whole device (must be set before importing jax).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))   # so `purejaxwm` resolves

from cogniland.zebra_nav import generate_zebra_map, tiles as T  # noqa: E402
from cogniland.zebra_nav_jax import (  # noqa: E402
    EnvParams,
    ZebraNavJaxEnv,
    constants as C,
    records_to_arrays,
)

import flax.linen as nn  # noqa: E402
import purejaxwm.dreamerv3.behavior as ac  # noqa: E402
from purejaxwm.dreamerv3.world_model import MLPHead, RSSM  # noqa: E402
from purejaxwm.dreamerv3.distributions import TwoHotDist  # noqa: E402
from purejaxwm.commons import resolve_dtype  # noqa: E402

# natural-maps task kwargs (matches the current default env: 3-cell centre door).
NATURAL_KWARGS = dict(
    size=32, width=64, orientation="natural",
    water_frac=0.14, rock_frac=0.14, tree_frac=0.03, goal_half=1,
)
SCALAR_DIM = 5
ACTION_NAMES = ["up", "down", "left", "right", "place", "mine"]

# ── Crafter-sprite rendering (mirrors scripts/play_zebra.py) ───────────
_SPRITE_DIR = _ROOT / "src/cogniland/assets/sprites"
_BASE = {T.GRASS: "grass", T.WATER: "water", T.ROCK: "stone", T.WOOD: "path",
         T.TREE: "tree", T.SAND: "sand", T.DIRT: "path", T.TARGET: "grass"}
_OVERLAY = {T.TARGET: "flag"}
# facing ids F_UP/F_DOWN/F_LEFT/F_RIGHT = 0/1/2/3 == move-action ids
_FACE_SPRITE = {0: "player-up", 1: "player-down", 2: "player-left", 3: "player-right"}
_BG = (18, 22, 30)


def _load_sprite_imgs(tp: int) -> dict:
    """Load the Crafter PNG sprites as PIL RGBA images scaled to ``tp`` px."""
    from PIL import Image
    names = ["grass", "water", "stone", "sand", "tree", "lava", "path", "flag",
             "diamond", "player", "player-up", "player-down", "player-left", "player-right"]
    return {n: Image.open(_SPRITE_DIR / f"{n}.png").convert("RGBA").resize(
        (tp, tp), Image.NEAREST) for n in names}


def _tiles_to_sprite_rgb(tiles: np.ndarray, sprites: dict, tp: int, facing: int) -> np.ndarray:
    """Composite a (V,V) decoded tile-id grid into a Crafter-sprite RGB image,
    with the player sprite drawn (facing) at the egocentric centre."""
    from PIL import Image
    V = tiles.shape[0]
    canvas = Image.new("RGB", (V * tp, V * tp), _BG)
    for r in range(V):
        for c in range(V):
            t = int(tiles[r, c])
            if t == T.OOB:                       # off-map padding → leave dark
                continue
            base = sprites[_BASE.get(t, "grass")]
            canvas.paste(base, (c * tp, r * tp), base)
            if t in _OVERLAY:
                ov = sprites[_OVERLAY[t]]
                canvas.paste(ov, (c * tp, r * tp), ov)
    pl = sprites[_FACE_SPRITE.get(facing, "player")]
    cen = V // 2
    canvas.paste(pl, (cen * tp, cen * tp), pl)
    return np.asarray(canvas)


def _frame_base_rgb(frame: dict, scale: int, sprites, sprite_px: int, facing: int) -> np.ndarray:
    """Base image for a frame: Crafter sprites if ``sprites`` given, else the
    flat tile-colour minimap upscaled by ``scale``."""
    if sprites is not None:
        return _tiles_to_sprite_rgb(frame["tiles"], sprites, sprite_px, facing)
    rgb = T.TILE_COLORS[frame["tiles"]]
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


# Exact copies of dreamerv3_zebra_nav.{ZebraEncoder,ZebraDecoder} so restored
# params bind without importing the trainer.
class ZebraEncoder(nn.Module):
    hidden: int
    num_layers: int
    embed_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden, use_bias=False,
                         dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = jax.nn.silu(x)
        x = nn.Dense(self.embed_dim, use_bias=False,
                     dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return jax.nn.silu(x)


class ZebraDecoder(nn.Module):
    hidden: int
    num_layers: int
    out_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden, use_bias=False,
                         dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = jax.nn.silu(x)
        x = nn.Dense(self.out_dim, use_bias=True,
                     dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x.astype(jnp.float32)


def _flatten_obs(obs: dict) -> jax.Array:
    """Matches FlattenObsWrapper._flatten."""
    mm = obs["minimap"].astype(jnp.float32) / float(C.NUM_TILES)
    return jnp.concatenate([
        mm.reshape(*mm.shape[:-2], -1),
        obs["scalars"].astype(jnp.float32),
    ], axis=-1)


# With the natural-only 9-tile vocab, EVERY id (0..8) is a real tile, so the
# palette is just the full id range and palette-snapping is effectively a no-op
# (kept so the --palette natural flag still works without changing the decode
# path). The obsidian/cue phantom-tile problem is gone with those tiles removed.
_NATURAL_PALETTE = np.arange(C.NUM_TILES, dtype=np.int64)


def _decode_to_tiles(flat_pred: np.ndarray, view: int, palette=None) -> np.ndarray:
    """Turn a decoded flat obs vector (V*V + 5,) into a (V,V) tile-id grid.

    The decoder regresses the *normalised* minimap (tile_id / NUM_TILES); undo the
    scaling. With ``palette`` given, snap each cell's continuous prediction to the
    nearest tile id that genuinely occurs (kills phantom obsidian/cue tiles);
    otherwise round to nearest of all NUM_TILES ids."""
    raw = np.asarray(flat_pred[: view * view]).reshape(view, view) * float(C.NUM_TILES)
    if palette is not None:
        idx = np.argmin(np.abs(raw[..., None] - palette.astype(np.float64)), axis=-1)
        return palette[idx]
    return np.clip(np.rint(raw).astype(np.int64), 0, C.NUM_TILES - 1)


def _build_model(cfg: dict):
    compute_dtype = resolve_dtype(cfg.get("compute_dtype", "float32"))
    param_dtype = jnp.float32
    flat_dim = cfg["view_size"] * cfg["view_size"] + SCALAR_DIM
    encoder = ZebraEncoder(
        hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
        embed_dim=cfg["wm_hidden"], dtype=compute_dtype, param_dtype=param_dtype,
    )
    decoder = ZebraDecoder(
        hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"], out_dim=flat_dim,
        dtype=compute_dtype, param_dtype=param_dtype,
    )
    rssm = RSSM(
        deter_dim=cfg["deter"], stoch_size=cfg["stoch"], classes=cfg["classes"],
        hidden=cfg["wm_hidden"], unimix=cfg["unimix"], blocks=cfg["blocks"],
        dtype=compute_dtype, param_dtype=param_dtype,
    )
    actor_head = MLPHead(
        hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"],
        out_dim=C.NUM_ACTIONS, outscale=0.01,
        dtype=compute_dtype, param_dtype=param_dtype,
    )
    reward_head = MLPHead(
        hidden=cfg["wm_hidden"], num_layers=1, out_dim=cfg["num_reward_bins"],
        outscale=0.0, dtype=compute_dtype, param_dtype=param_dtype,
    )
    return encoder, decoder, rssm, actor_head, reward_head


def _single_map_params(seed: int, cfg: dict) -> tuple[EnvParams, object]:
    rec = generate_zebra_map(seed=seed, **NATURAL_KWARGS)
    arrays = records_to_arrays([rec])
    params = EnvParams.from_map_arrays(
        **arrays,
        max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        slack_penalty=cfg["slack_penalty"], reach_bonus=cfg["reach_bonus"],
        shaping_coef=cfg["shaping_coef"], build_cost=cfg["build_cost"],
        gamma=cfg["gamma"],
    )
    return params, rec


def rollout_and_imagine(models, wm_params, ac_params, env, params, cfg,
                        warmup: int, horizon: int, key, palette=None):
    """Warm the RSSM on `warmup` real steps then imagine `horizon` steps.

    Returns a list of frame dicts:
      {phase: 'REAL'|'IMAGINED', tiles: (V,V) int, action: int, reward: float}
    `reward` is the env reward for REAL frames and the reward-head mean for
    IMAGINED frames.  Plus the real terrain rec for context (unused here).
    """
    encoder, decoder, rssm, actor_head, reward_head = models
    view = cfg["view_size"]
    A = C.NUM_ACTIONS

    @jax.jit
    def warmup_step(rssm_state, last_action_oh, is_first, flat, k):
        ks, kp = jax.random.split(k)
        embed = encoder.apply(wm_params["encoder"], flat)
        _, post = rssm.apply(
            wm_params["rssm"], rssm_state, last_action_oh, embed, is_first,
            rngs={"stoch": ks},
        )
        feat = post.features()
        logits = ac.unimix_logits(actor_head.apply(ac_params["actor"], feat))
        action_idx = jnp.argmax(logits, axis=-1)   # deterministic (greedy)
        action_oh = jax.nn.one_hot(action_idx, A)
        rec_pred = decoder.apply(wm_params["decoder"], feat)
        return post, action_idx, action_oh, rec_pred

    @jax.jit
    def imagine_step(rssm_state, action_oh, k):
        # pure imagination: prior-only RSSM step (embed=None), training=False so
        # the latent is the categorical *mode* (deterministic dream).
        prior = rssm.apply(
            wm_params["rssm"], rssm_state, action_oh, None, None, False,
            rngs={"stoch": k},
        )
        feat = prior.features()
        logits = ac.unimix_logits(actor_head.apply(ac_params["actor"], feat))
        action_idx = jnp.argmax(logits, axis=-1)
        next_oh = jax.nn.one_hot(action_idx, A)
        rec_pred = decoder.apply(wm_params["decoder"], feat)
        rew = TwoHotDist(reward_head.apply(wm_params["reward"], feat)).mean()
        return prior, action_idx, next_oh, rec_pred, rew

    frames = []
    key, kr = jax.random.split(key)
    obs, state = env.reset_env(kr, params)
    obs = jax.tree_util.tree_map(lambda x: x[None], obs)        # add batch dim
    rssm_state = rssm.initial_state((1,))
    last_action_oh = jnp.zeros((1, A))
    is_first = jnp.ones((1,), dtype=bool)

    # One unified rollout. For the first ``warmup`` steps the RSSM is CLOSED-LOOP
    # (posterior conditioned on the real obs); after that it is OPEN-LOOP (prior
    # only, the model no longer sees the env). In BOTH phases the chosen action is
    # applied to the real env, so ``gt_tiles`` is always the true egocentric obs at
    # that step and ``tiles`` is the model's decode (a reconstruction during warmup,
    # a pure prediction during the open-loop phase).
    for t in range(warmup + horizon):
        gt_mm = np.asarray(obs["minimap"][0]).astype(np.int64)   # ground-truth obs
        key, k = jax.random.split(key)
        if t < warmup:
            flat = _flatten_obs(obs)
            rssm_state, action_idx, action_oh, rec_pred = warmup_step(
                rssm_state, last_action_oh, is_first, flat, k)
            phase, rew = "warmup", None
        else:
            rssm_state, action_idx, action_oh, rec_pred, rew_arr = imagine_step(
                rssm_state, last_action_oh, k)
            phase, rew = "open-loop", float(rew_arr[0])
        frames.append({
            "phase": phase,
            "tiles": _decode_to_tiles(np.asarray(rec_pred[0]), view, palette),
            "gt_tiles": gt_mm,
            "action": int(action_idx[0]), "reward": rew,
        })
        key, ks = jax.random.split(key)
        nobs, state, reward, done, info = env.step_env(ks, state, action_idx[0], params)
        obs = jax.tree_util.tree_map(lambda x: x[None], nobs)
        last_action_oh = action_oh
        is_first = jnp.zeros((1,), dtype=bool)
    return frames


def _frame_to_rgb(frame: dict, scale: int, sprites=None, sprite_px: int = 16,
                  facing: int = 1) -> np.ndarray:
    """Render one frame dict → uint8 RGB array (Crafter sprites if ``sprites``
    given, else flat tile colours), with a coloured banner (green=warmup/closed-
    loop, red=open-loop imagination) and a burnt-in title."""
    big = _frame_base_rgb(frame, scale, sprites, sprite_px, facing)

    # draw onto a matplotlib canvas for the title text + phase banner.
    H, W = big.shape[:2]
    fig = plt.figure(figsize=(W / 100 + 0.2, H / 100 + 0.9), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.85])
    ax.imshow(big, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    banner = "tab:green" if frame["phase"] == "warmup" else "tab:red"
    for spine in ax.spines.values():
        spine.set_edgecolor(banner)
        spine.set_linewidth(6)
    rtxt = "" if frame["reward"] is None else f"   r={frame['reward']:+.3f}"
    title = f"{frame['phase']}   a={ACTION_NAMES[frame['action']]}{rtxt}"
    fig.text(0.5, 0.93, title, ha="center", va="center", fontsize=11,
             color=banner, fontweight="bold")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _write_video(frames_rgb, base: Path, fps: int) -> tuple[Path, str]:
    """Try mp4 (ffmpeg) first, fall back to gif. Returns (path, fmt)."""
    mp4 = base.with_suffix(".mp4")
    try:
        with imageio.get_writer(mp4, fps=fps, codec="libx264",
                                quality=8, macro_block_size=None) as w:
            for fr in frames_rgb:
                w.append_data(fr)
        return mp4, "mp4"
    except Exception as e:   # noqa: BLE001
        print(f"  [warn] mp4 writer failed ({e!r}); falling back to gif")
        gif = base.with_suffix(".gif")
        imageio.mimsave(gif, frames_rgb, fps=fps)
        return gif, "gif"


def _tiles_rgb(tiles, sprites, sprite_px, facing):
    """One tile grid → RGB (Crafter sprites if given, else flat tile colours)."""
    if sprites is not None:
        return _tiles_to_sprite_rgb(tiles, sprites, sprite_px, facing)
    return T.TILE_COLORS[tiles]


def _write_strip(frames, base: Path, every: int, sprites=None, sprite_px: int = 16,
                 facings=None) -> Path:
    """Two-row filmstrip: TOP = the model's decoded egocentric view (a
    reconstruction during warmup, an open-loop prediction after), BOTTOM = the
    GROUND-TRUTH egocentric obs from the env at the same step. Green columns are
    closed-loop warmup, red columns are open-loop imagination."""
    sel = [(i, f) for i, f in enumerate(frames) if i % every == 0]
    if (len(frames) - 1) % every != 0:        # always include the last frame
        sel.append((len(frames) - 1, frames[-1]))
    n = len(sel)
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.7), squeeze=False)
    for col, (i, f) in enumerate(sel):
        fc = facings[i] if facings is not None else 1
        c = "green" if f["phase"] == "warmup" else "red"
        for row, key in ((0, "tiles"), (1, "gt_tiles")):
            ax = axes[row][col]
            ax.imshow(_tiles_rgb(f[key], sprites, sprite_px, fc), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(c); s.set_linewidth(2)
        axes[0][col].set_title(f"t{i} {f['phase']}\n{ACTION_NAMES[f['action']]}",
                               fontsize=7, color=c)
    axes[0][0].set_ylabel("decoded\n(model)", fontsize=9)
    axes[1][0].set_ylabel("ground truth\n(env)", fontsize=9)
    fig.tight_layout()
    out = base.with_suffix(".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("runs/dreamer_natural_wholewall/checkpoints/step_1000000"))
    p.add_argument("--n-examples", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--scale", type=int, default=12)
    p.add_argument("--strip-every", type=int, default=4)
    p.add_argument("--render", choices=("sprites", "tiles"), default="sprites",
                   help="sprites = decode to Crafter PNG sprites (default); tiles = flat tile colours")
    p.add_argument("--sprite-px", type=int, default=16, help="px per tile when --render sprites")
    p.add_argument("--palette", choices=("natural", "all"), default="natural",
                   help="natural = snap decoded scalar to ids that actually occur "
                        "(no phantom obsidian/cue tiles); all = round to any of NUM_TILES")
    p.add_argument("--out-dir", type=Path, default=Path("videos/dreamer_imagine"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sprites = _load_sprite_imgs(args.sprite_px) if args.render == "sprites" else None
    palette = _NATURAL_PALETTE if args.palette == "natural" else None

    ckpt_dir = args.checkpoint.resolve()
    cfg_path = ckpt_dir.parent.parent / "config.json"
    cfg = json.loads(cfg_path.read_text())
    print(f"[load] config     {cfg_path}")
    print(f"[load] checkpoint {ckpt_dir}")

    payload = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    wm_params = jax.tree_util.tree_map(jnp.asarray, payload["wm_params"])
    ac_params = jax.tree_util.tree_map(jnp.asarray, payload["ac_params"])

    models = _build_model(cfg)
    env = ZebraNavJaxEnv()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)

    for j in range(args.n_examples):
        seed = args.eval_seed_start + j
        params, rec = _single_map_params(seed, cfg)
        key, sub = jax.random.split(key)
        print(f"[example {j}] seed {seed}: warmup={args.warmup} "
              f"horizon={args.horizon} ...", flush=True)
        frames = rollout_and_imagine(
            models, wm_params, ac_params, env, params, cfg,
            args.warmup, args.horizon, sub, palette,
        )
        imag = [f for f in frames if f["phase"] == "open-loop"]
        acts = "".join(ACTION_NAMES[f["action"]][0] for f in imag)
        print(f"  open-loop actions: {acts}")
        if imag:
            rs = [f["reward"] for f in imag]
            print(f"  open-loop reward (head mean) sum={sum(rs):+.3f}  max={max(rs):+.3f}")

        # facing per frame: carry last move direction (default right → goal).
        facings, cur = [], 3
        for f in frames:
            if f["action"] < 4:
                cur = f["action"]
            facings.append(cur)

        frames_rgb = [_frame_to_rgb(f, args.scale, sprites, args.sprite_px, fc)
                      for f, fc in zip(frames, facings)]
        vid_base = args.out_dir / f"imagine_seed{seed}"
        vid_path, fmt = _write_video(frames_rgb, vid_base, args.fps)
        strip_path = _write_strip(
            frames, args.out_dir / f"imagine_strip_seed{seed}", args.strip_every,
            sprites, args.sprite_px, facings,
        )
        print(f"  wrote {vid_path}  ({fmt})")
        print(f"  wrote {strip_path}")

    print("done.")


if __name__ == "__main__":
    main()
