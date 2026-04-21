#!/usr/bin/env python3
"""Cogniland demo — play an RGB map as a human.

Uses the RGB dataset produced by scripts/generate_dataset.py.
Each map is a 128x128x3 image; the game logic reads terrain classes from
the accompanying terrain_idx grid and applies the tuned TileEffects drains.

Controls:
    WASD / arrows — move
    F — forage (forest → wood, berry → HP)
    C — craft a tool (costs 100 wood, one tool only)
    R — reset map
    ESC — back to menu

Usage:
    python scripts/generate_dataset.py --preview
    python demo.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_maps as gt
from tune_tile_effects import TileEffects, drain_for


# ── Config ──────────────────────────────────────────────────────────────────

VAL_PATH = Path("data/maps/val.pt")
MAP_SIZE = gt.CROP_SIZE  # 128

# Folder that ships with the demo, containing one subfolder per selectable
# model. Each subfolder must mirror ``results/<run_id>/checkpoints/`` layout:
#   demo_artifacts/<model_name>/run_config.yaml
#   demo_artifacts/<model_name>/cogniland-v0/best/   (orbax dir)
# Drop in new trained checkpoints here and they will show up in the UI on
# next launch — no code changes needed.
DEMO_ARTIFACTS_PATH = PROJECT_ROOT / "demo_artifacts"

WINDOW_W, WINDOW_H = 1200, 780
MAP_DISPLAY = 512
PANEL_X = MAP_DISPLAY + 60
PANEL_W = WINDOW_W - PANEL_X - 20

# Terrain-dependent visibility radii (in tiles) — kept in sync with
# configs/env/cogniland.yaml::env.terrain_vis_radius.
TERRAIN_VIS_RADIUS: dict[str, int] = {
    "ocean":      22,
    "deep_water": 18,
    "water":      14,
    "beach":      12,
    "sandy":      12,
    "grassland":  12,
    "forest":     10,
    "rocky":      18,
    "mountains":  22,
}
DEFAULT_VIS_RADIUS = 7
MINIMAP_RADIUS = 22  # max ray for minimap patch extraction
MINIMAP_DISPLAY = 340  # pixel size of minimap on screen
CLEAR_TOLERANCE = 0.15  # height diff above agent that blocks vision

COLORS = {
    "bg":       (22, 22, 30),
    "panel":    (32, 32, 42),
    "fg":       (215, 215, 220),
    "dim":      (120, 120, 130),
    "white":    (255, 255, 255),
    "player":   (255,  60,  60),
    "target":   ( 60, 255,  80),
    "hp_full":  ( 70, 210, 110),
    "hp_mid":   (240, 190,  60),
    "hp_low":   (240,  80,  70),
    "wood":     (200, 140,  60),
    "berry":    gt.BERRY_COLOR,
    "accent":   ( 90, 160, 255),
    "highlight":(255, 200,  60),
    "craft_bg": ( 40,  40,  55),
    "forage":   (120, 200,  80),
}

ACTIONS = {
    pygame.K_UP:    (-1, 0),  pygame.K_w: (-1, 0),
    pygame.K_DOWN:  ( 1, 0),  pygame.K_s: ( 1, 0),
    pygame.K_LEFT:  ( 0,-1),  pygame.K_a: ( 0,-1),
    pygame.K_RIGHT: ( 0, 1),  pygame.K_d: ( 0, 1),
}

CRAFTABLE_TOOLS = ["raft", "rope", "shoes"]

# Mirrors cogniland.envs.env: 0-3 move, 4 forage, 5-7 craft raft/rope/shoes.
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
}
ACTION_LABELS = ["up", "down", "left", "right",
                 "forage", "craft raft", "craft rope", "craft shoes"]

# Inferno-inspired color ramp for trajectory visit counts:
# 1 visit = bright red/orange, more visits = darker
INFERNO_RAMP = [
    (220,  50,  30),   # 1 visit  — bright red
    (180,  30,  20),   # 2 visits
    (130,  20,  15),   # 3 visits
    ( 90,  12,  10),   # 4 visits
    ( 55,   5,   5),   # 5+ visits — very dark
]

TOOL_SYMBOLS = {"raft": "R", "rope": "P", "shoes": "S"}


# ── Dataset ─────────────────────────────────────────────────────────────────

def load_val_dataset():
    if not VAL_PATH.exists():
        return None
    return torch.load(str(VAL_PATH), map_location="cpu", weights_only=False)


def _sample_spawn_target(tidx: np.ndarray, seed: int,
                         min_manhattan: int = 60) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pick two distinct land cells at least `min_manhattan` apart."""
    water_idx = gt.TERRAIN_NAMES.index("water")
    land = np.argwhere(tidx > water_idx)
    rng = random.Random(seed)
    if len(land) < 2:
        return (MAP_SIZE // 2, MAP_SIZE // 2), (MAP_SIZE // 2, MAP_SIZE // 2)
    for _ in range(500):
        i, j = rng.randrange(len(land)), rng.randrange(len(land))
        s = tuple(int(x) for x in land[i])
        t = tuple(int(x) for x in land[j])
        if abs(s[0] - t[0]) + abs(s[1] - t[1]) >= min_manhattan:
            return s, t
    return tuple(int(x) for x in land[0]), tuple(int(x) for x in land[-1])


# ── Minimap occlusion (simplified from cogniland/env/core.py) ──────────────

def _bresenham_ray(y0: int, x0: int, y1: int, x1: int) -> list[tuple[int, int]]:
    """Bresenham line from (y0,x0) to (y1,x1)."""
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    ray = []
    while True:
        ray.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return ray


def compute_occlusion_mask(heightmap: np.ndarray, center_r: int, center_c: int,
                           vis_radius: int) -> np.ndarray:
    """Compute a (2*MINIMAP_RADIUS+1)^2 visibility mask with raycasting.

    Returns a bool array where True = visible.
    """
    R = MINIMAP_RADIUS
    diameter = 2 * R + 1
    visible = np.zeros((diameter, diameter), dtype=bool)
    visible[R, R] = True  # center always visible

    H, W = heightmap.shape
    center_h = heightmap[center_r, center_c] if (0 <= center_r < H and 0 <= center_c < W) else 0.0

    # Cast rays to perimeter of the patch
    perimeter = []
    for i in range(diameter):
        perimeter.append((0, i))
        perimeter.append((diameter - 1, i))
    for i in range(1, diameter - 1):
        perimeter.append((i, 0))
        perimeter.append((i, diameter - 1))

    for py, px in perimeter:
        ray = _bresenham_ray(R, R, py, px)
        blocked = False
        for ry, rx in ray[1:]:  # skip center
            dist = math.sqrt((ry - R) ** 2 + (rx - R) ** 2)
            if dist > vis_radius:
                break
            wr = center_r + (ry - R)
            wc = center_c + (rx - R)
            if not (0 <= wr < H and 0 <= wc < W):
                break
            if blocked:
                continue  # already occluded along this ray
            visible[ry, rx] = True
            cell_h = heightmap[wr, wc]
            if cell_h >= center_h + CLEAR_TOLERANCE:
                blocked = True

    return visible


def render_minimap(rgb: np.ndarray, heightmap: np.ndarray,
                   pos: tuple[int, int], target: tuple[int, int],
                   terrain_name: str) -> pygame.Surface:
    """Render a minimap surface with occlusion. Unseen pixels are opaque black."""
    R = MINIMAP_RADIUS
    diameter = 2 * R + 1
    vis_r = TERRAIN_VIS_RADIUS.get(terrain_name, DEFAULT_VIS_RADIUS)

    vis_mask = compute_occlusion_mask(heightmap, pos[0], pos[1], vis_r)

    # Build RGB patch — black by default (unseen = opaque)
    patch = np.zeros((diameter, diameter, 3), dtype=np.uint8)
    H, W = rgb.shape[:2]
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            wr, wc = pos[0] + dy, pos[1] + dx
            py, px = dy + R, dx + R
            dist = math.sqrt(dy * dy + dx * dx)
            if dist > vis_r + 0.5:
                continue  # outside vision radius → black
            if 0 <= wr < H and 0 <= wc < W and vis_mask[py, px]:
                patch[py, px] = rgb[wr, wc]
            # else: stays black (out of bounds or occluded)

    # Draw target marker if visible
    ty, tx = target[0] - pos[0] + R, target[1] - pos[1] + R
    if 0 <= ty < diameter and 0 <= tx < diameter and vis_mask[ty, tx]:
        for d in range(-1, 2):
            for oy, ox in [(d, 0), (0, d)]:
                ny, nx = ty + oy, tx + ox
                if 0 <= ny < diameter and 0 <= nx < diameter:
                    patch[ny, nx] = COLORS["target"]

    # Player dot at center
    patch[R, R] = COLORS["player"]

    surf = pygame.Surface((diameter, diameter))
    pygame.surfarray.blit_array(surf, patch.transpose(1, 0, 2))
    scaled = pygame.transform.scale(surf, (MINIMAP_DISPLAY, MINIMAP_DISPLAY))

    # Compass arrow from center of minimap pointing toward target
    center_px = MINIMAP_DISPLAY // 2
    draw_compass_arrow(scaled, center_px, center_px, pos, target,
                       length=min(28, MINIMAP_DISPLAY // 6))
    return scaled


# ── Game state ──────────────────────────────────────────────────────────────

class CognilandGame:
    def __init__(self, rgb: np.ndarray, heightmap: np.ndarray,
                 tidx: np.ndarray, berry_mask: np.ndarray,
                 biome: str, seed: int):
        self.biome = biome
        self.seed = seed
        self.rgb = rgb.copy()
        self.heightmap = heightmap.astype(np.float32)
        self.tidx = tidx.astype(np.int32)
        self.berry_mask = berry_mask.copy()
        self.effects = TileEffects()

        self.spawn, self.target = _sample_spawn_target(self.tidx, seed)
        self.pos = self.spawn
        self.hp: float = float(self.effects.init_hp)
        self.wood: int = 0
        self.tool: str | None = None  # at most one crafted tool
        self.consec_grass = 0
        self.steps = 0
        self.path: list[tuple[int, int]] = [self.spawn]
        self.hp_history: list[float] = [self.hp]
        self.game_over = False
        self.won = False
        self.last_drain: int | None = None
        self.last_terrain: str = self._terrain_name(*self.pos)
        self.last_forage_msg: str = ""
        self.forage_msg_timer: int = 0
        self.crafting = False
        self.craft_step: int | None = None  # step index where tool was crafted
        self.craft_tool_name: str | None = None  # which tool was crafted

    def _terrain_name(self, r: int, c: int) -> str:
        idx = int(self.tidx[r, c])
        if idx < 0:
            return "deadly"
        return gt.TERRAIN_NAMES[idx]

    @property
    def tools(self) -> frozenset[str]:
        if self.tool is None:
            return frozenset()
        return frozenset({self.tool})

    def forage(self):
        """Explicit forage action on current tile. Uses one step."""
        if self.game_over:
            return
        r, c = self.pos
        terrain = self._terrain_name(r, c)

        foraged = False
        berry_forage = False
        if self.berry_mask[r, c]:
            # Berries are permanent tiles — heal every time, no drain
            self.hp = min(float(self.effects.hp_max), self.hp + self.effects.berry_heal)
            self.last_forage_msg = f"+{self.effects.berry_heal} HP (berry)"
            self.last_drain = 0
            self.forage_msg_timer = 90
            foraged = True
            berry_forage = True
        elif terrain == "forest":
            self.wood = min(self.wood + self.effects.forest_wood, self.effects.wood_max)
            self.last_forage_msg = f"+{self.effects.forest_wood} wood"
            self.forage_msg_timer = 90
            foraged = True
        else:
            self.last_forage_msg = "nothing to forage"
            self.forage_msg_timer = 60

        if foraged:
            self.consec_grass = self.consec_grass + 1 if terrain == "grassland" else 0
            if not berry_forage:
                # Forest foraging costs the tile's drain
                drain = drain_for(terrain, self.tools, self.consec_grass, self.effects)
                self.hp -= drain
                self.last_drain = drain
            self.steps += 1
            self.path.append(self.pos)  # same position = forage in place
            self.hp_history.append(self.hp)
            if self.hp <= 0:
                self.hp = 0.0
                self.game_over = True
                self.won = False

    def craft(self, tool_name: str) -> bool:
        """Try to craft a tool. Returns True on success."""
        if self.tool is not None:
            return False
        if self.wood < self.effects.craft_cost:
            return False
        if tool_name not in CRAFTABLE_TOOLS:
            return False
        self.wood -= self.effects.craft_cost
        self.tool = tool_name
        self.craft_step = len(self.path) - 1  # index in path where craft happened
        self.craft_tool_name = tool_name
        return True

    def step(self, dr: int, dc: int):
        if self.game_over:
            return
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if not (0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE):
            return

        idx = int(self.tidx[nr, nc])
        if idx < 0:
            self.pos = (nr, nc)
            self.hp = 0.0
            self.last_terrain = "deadly"
            self.last_drain = int(self.effects.init_hp)
            self.path.append(self.pos)
            self.hp_history.append(self.hp)
            self.steps += 1
            self.game_over = True
            self.won = False
            return

        terrain = gt.TERRAIN_NAMES[idx]
        self.consec_grass = self.consec_grass + 1 if terrain == "grassland" else 0
        drain = drain_for(terrain, self.tools, self.consec_grass, self.effects)

        self.hp -= drain
        self.pos = (nr, nc)
        self.steps += 1
        self.path.append(self.pos)
        self.hp_history.append(self.hp)
        self.last_drain = drain
        self.last_terrain = terrain

        if self.hp <= 0:
            self.hp = 0.0
            self.game_over = True
            self.won = False
        elif self.pos == self.target:
            self.game_over = True
            self.won = True


# ── AI agent playback ───────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Bundle of metadata for one selectable model in the demo."""
    name: str                       # human-readable (folder name)
    path: Path                      # path passed to AIAgentPlayer
    step: int | None = None         # training step at checkpoint
    eval_success: float | None = None
    hidden_size: int | None = None
    lstm_size: int | None = None
    agent_name: str | None = None

    def summary(self) -> str:
        """One-line metadata summary (for the picker UI)."""
        parts: list[str] = []
        if self.eval_success is not None:
            parts.append(f"{self.eval_success * 100:.1f}% success")
        if self.step is not None:
            parts.append(f"step {self.step}")
        if self.hidden_size is not None and self.lstm_size is not None:
            parts.append(f"hidden={self.hidden_size} lstm={self.lstm_size}")
        if self.agent_name and self.agent_name != "ppo_rnn":
            parts.append(self.agent_name)
        return "  ·  ".join(parts) if parts else "(no metadata)"


def scan_demo_artifacts(root: Path = DEMO_ARTIFACTS_PATH) -> list[ModelInfo]:
    """Return every valid model folder inside ``root``.

    A folder is considered valid if it contains a ``run_config.yaml`` and a
    subdir that has a ``best/`` (or ``last/``) orbax checkpoint. Metadata
    (step, eval success, net sizes) is pulled opportunistically from the
    ``custom_metadata.json`` written next to each checkpoint.
    """
    models: list[ModelInfo] = []
    if not root.exists():
        return models
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "run_config.yaml").exists():
            continue
        env_ckpt_dir: Path | None = None
        for sub in entry.iterdir():
            if sub.is_dir() and ((sub / "best").exists() or (sub / "last").exists()):
                env_ckpt_dir = sub
                break
        if env_ckpt_dir is None:
            continue
        ckpt = env_ckpt_dir / "best"
        if not ckpt.exists():
            ckpt = env_ckpt_dir / "last"
        info = ModelInfo(name=entry.name, path=entry)
        meta_path = ckpt / "custom_metadata.json"
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                info.step = m.get("step")
                metrics = m.get("metrics") or {}
                info.eval_success = metrics.get("eval_success")
                agent_cfg = (m.get("config") or {}).get("agent") or {}
                info.agent_name = agent_cfg.get("name")
                info.hidden_size = agent_cfg.get("hidden_size")
                info.lstm_size = agent_cfg.get("lstm_size")
            except (json.JSONDecodeError, OSError):
                pass
        models.append(info)
    # Rank by eval success desc (best on top); unknown success sinks.
    models.sort(key=lambda m: (m.eval_success is None, -(m.eval_success or 0.0), m.name))
    return models


class AIAgentPlayer:
    """Loads a trained ppo_rnn checkpoint and drives a CognilandGame.

    Model-size-agnostic: the policy network is rebuilt from ``run_config.yaml``
    saved alongside the checkpoint, so ``hidden_size`` / ``lstm_size`` /
    ``task_embedding_dim`` are read from disk and do not need to match the
    current defaults in ``configs/agent/ppo_rnn.yaml``.

    ``checkpoint_dir`` may point at:
      * a run root (``results/<run_id>/``)
      * its ``checkpoints/`` subdir
      * the env-specific subdir (``checkpoints/<env_id>/``)
      * or a concrete ``best/`` / ``last/`` / ``step_XXXXXXXX/`` orbax dir.

    The task embedding is a fixed-size one-hot of length ``TASK_EMBEDDING_DIM``
    (see ``cogniland.envs.tasks``), independent of the saved config.
    """

    def __init__(self, checkpoint_dir: str | Path, prefer: str = "best"):
        import jax
        import jax.numpy as jnp
        from omegaconf import OmegaConf

        from cogniland.agents.ppo_rnn import make_ppo_rnn
        from cogniland.agents.state import RuntimeState
        from cogniland.trainer.checkpoint import CheckpointManager
        from cogniland.envs.env import (
            _build_circular_masks,
            _compute_minimap_batch,
            MINIMAP_RADIUS,
            MINIMAP_DIAMETER,
            TERRAIN_NAMES,
            DEFAULT_TERRAIN_VIS,
        )

        self._jax = jax
        self._jnp = jnp
        self._compute_minimap_batch = _compute_minimap_batch
        self._minimap_radius = MINIMAP_RADIUS
        self._minimap_diameter = MINIMAP_DIAMETER
        self._terrain_names = TERRAIN_NAMES

        base_dir, env_ckpt_dir = self._resolve_checkpoint_layout(Path(checkpoint_dir))
        self.checkpoint_root = base_dir

        config_path = base_dir / "run_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"run_config.yaml not found next to checkpoints: {config_path}"
            )
        self.config = OmegaConf.load(config_path)

        # Build the network at the *saved* sizes. We bypass registry.load_agent()
        # because it eagerly instantiates a full CognilandEnv (loading all map
        # datasets); for inference we can pass a hand-rolled obs_space.
        obs_space = {
            "minimap": (6, MINIMAP_DIAMETER, MINIMAP_DIAMETER),
            "scalars": (6,),
        }
        self.agent = make_ppo_rnn(self.config, obs_space, 8)

        # Orbax checkpointer keyed at env_ckpt_dir. ``prefer`` selects best/last.
        manager = CheckpointManager(
            checkpoint_dir=str(env_ckpt_dir),
            keep_last=0,
            save_best=True,
        )
        load_best = (prefer == "best") and (env_ckpt_dir / "best").exists()
        try:
            state_dict, _saved_cfg, self.metadata = manager.load(load_best=load_best)
        except FileNotFoundError:
            # Fall back to whichever checkpoint exists.
            load_best = not load_best
            state_dict, _saved_cfg, self.metadata = manager.load(load_best=load_best)
        self._which = "best" if load_best else "latest"

        runtime = RuntimeState(
            buffer_state=None,
            wm_state=None,
            step=jnp.array(0),
            train_steps=jnp.array(0),
            rng=jax.random.PRNGKey(0),
            current_num_actions=jnp.array(8),
            current_task_id=jnp.array(0, dtype=jnp.int32),
        )
        self._base_state = self.agent.state_from_checkpoint(state_dict, runtime)

        # Visibility precomputation (mirrors CognilandEnv.__init__).
        terrain_vis = OmegaConf.to_container(
            self.config.env.get("terrain_vis_radius", None), resolve=True
        ) if hasattr(self.config.env, "terrain_vis_radius") else None
        if not terrain_vis:
            terrain_vis = dict(DEFAULT_TERRAIN_VIS)
        self._vis_per_terrain = np.array(
            [terrain_vis.get(name, 7) for name in TERRAIN_NAMES], dtype=np.int32,
        )
        circular = _build_circular_masks(MINIMAP_RADIUS)
        max_r = max(circular.keys())
        stack = np.zeros((max_r + 1, MINIMAP_DIAMETER, MINIMAP_DIAMETER), dtype=bool)
        for r, m in circular.items():
            stack[r] = m
        self._disk_stack = stack

        from cogniland.envs.tasks import TASK_EMBEDDING_DIM
        self._task_embedding_dim = TASK_EMBEDDING_DIM
        self.deterministic = True
        self.task_id = 0

        self.reset()

    # -- public API ---------------------------------------------------------

    def reset(self) -> None:
        """Reset the LSTM carry and episode-first flag for a new rollout."""
        jnp = self._jnp
        task_emb = jnp.zeros((1, self._task_embedding_dim), dtype=jnp.float32)
        task_emb = task_emb.at[0, self.task_id % self._task_embedding_dim].set(1.0)
        self._task_emb_jax = task_emb
        # Start from the freshly-restored base state with no carry cache. The
        # carry is rebuilt (zeros) inside select_action on first call.
        self.state = self._base_state
        self._is_first = True
        self.last_action: int | None = None

    def act(self, game: "CognilandGame",
            visibility_lut_map: np.ndarray) -> int:
        """Return an action index 0..7 given the current game state.

        ``visibility_lut_map`` is the bit-packed LUT for the map being played
        (shape ``[H, W, 254]`` uint8) — it comes straight from the dataset.
        """
        obs = self._build_obs(game, visibility_lut_map)
        rng = self._jax.random.PRNGKey(game.steps + 1_000 * (self.task_id + 1))
        # select_action reads the task embedding and LSTM carry off the state
        # via ``object.__setattr__`` attributes. Because ``state.replace(...)``
        # inside select_action creates a fresh dataclass instance, any non-
        # field attribute we set on ``self.state`` would be dropped after the
        # first call. We therefore re-attach the task embedding every step;
        # the carry IS propagated because select_action attaches it to the
        # returned state (the one we reuse next tick).
        object.__setattr__(self.state, "_task_emb_cache", self._task_emb_jax)
        action_arr, new_state = self.agent.select_action(
            self.state, obs, rng,
            is_first=np.array([self._is_first], dtype=bool),
            prev_action=None,
            training=not self.deterministic,
        )
        self.state = new_state
        self._is_first = False
        action = int(action_arr[0])
        self.last_action = action
        return action

    def apply_action(self, game: "CognilandGame", action: int) -> None:
        """Translate a model action index into the matching CognilandGame op."""
        if game.game_over:
            return
        if 0 <= action <= 3:
            dr, dc = ACTION_DELTAS[action]
            game.step(dr, dc)
        elif action == 4:
            game.forage()
        elif 5 <= action <= 7:
            tool = CRAFTABLE_TOOLS[action - 5]
            # Match env semantics: if craft fails (no wood / already crafted),
            # it's a no-op step — the episode still progresses but the agent
            # is not punished for trying. We mirror that by doing nothing so
            # the "step" counter does not advance here; this keeps the demo
            # visually tidy but is lenient vs training.
            game.craft(tool)
        # action codes outside 0..7 are ignored.

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _resolve_checkpoint_layout(path: Path) -> tuple[Path, Path]:
        """Given any ``results/<run>/...`` path, return ``(config_dir, env_ckpt_dir)``.

        ``config_dir`` is where ``run_config.yaml`` lives; ``env_ckpt_dir`` is
        the orbax manager's root (the dir containing ``best/`` and ``last/``).
        The real layout is ``results/<run>/checkpoints/run_config.yaml`` with
        ``results/<run>/checkpoints/<env_id>/{best,last,step_XXX}/`` below it,
        so we try several common starting points in order.
        """
        path = path.resolve()
        # Candidate base dirs (where run_config.yaml might live), in priority
        # order: the given path, then its ``checkpoints/`` child, then each
        # parent walking up to cover ``.../best/`` style paths.
        candidates: list[Path] = [path, path / "checkpoints"]
        parent = path
        for _ in range(5):
            if parent.parent == parent:
                break
            parent = parent.parent
            candidates.append(parent)

        base_dir: Path | None = None
        for c in candidates:
            if (c / "run_config.yaml").exists():
                base_dir = c
                break
        if base_dir is None:
            raise FileNotFoundError(
                f"Could not locate run_config.yaml from {path} "
                f"(tried {', '.join(str(c) for c in candidates)})"
            )
        for candidate in base_dir.iterdir():
            if candidate.is_dir() and (
                (candidate / "best").exists() or (candidate / "last").exists()
            ):
                return base_dir, candidate
        raise FileNotFoundError(
            f"No env-specific checkpoint dir (with best/ or last/) under {base_dir}"
        )

    def _build_obs(self, game: "CognilandGame",
                   visibility_lut_map: np.ndarray) -> dict:
        """Build a single-env obs dict matching CognilandEnv._get_obs."""
        jnp = self._jnp
        rgb = game.rgb[None]
        heightmap = game.heightmap[None].astype(np.float32)
        tidx = game.tidx[None].astype(np.int8)
        berry_mask = game.berry_mask[None].astype(bool)
        vis_lut = visibility_lut_map[None]

        map_idx = np.zeros(1, dtype=np.int32)
        pos_r = np.array([game.pos[0]], dtype=np.int32)
        pos_c = np.array([game.pos[1]], dtype=np.int32)
        yes_r = np.array([game.target[0]], dtype=np.int32)
        yes_c = np.array([game.target[1]], dtype=np.int32)
        # The env places a paired NO target to test answer selection; the demo
        # game has only a single (YES) target, so we park NO far off-map so
        # it never lands inside any minimap patch.
        no_r = np.array([-10_000], dtype=np.int32)
        no_c = np.array([-10_000], dtype=np.int32)

        minimap = self._compute_minimap_batch(
            rgb, heightmap, tidx, berry_mask,
            map_idx, pos_r, pos_c, yes_r, yes_c, no_r, no_c,
            self._vis_per_terrain,
            vis_lut_packed=vis_lut,
            disk_stack=self._disk_stack,
            occlude=True,
        )

        # Scalars: compass + tile class + hp + wood + tool (matches env).
        dr = float(yes_r[0] - pos_r[0])
        dc = float(yes_c[0] - pos_c[0])
        dist = max(math.hypot(dr, dc), 1e-6)
        tile = int(game.tidx[game.pos[0], game.pos[1]])
        tile_cls = max(tile, 0)
        if game.berry_mask[game.pos[0], game.pos[1]]:
            tile_cls = 9
        tool_id = {None: 0, "raft": 1, "rope": 2, "shoes": 3}[game.tool]

        scalars = np.array([[
            dc / dist, dr / dist,
            tile_cls / 9.0,
            game.hp / float(game.effects.hp_max),
            game.wood / float(game.effects.wood_max),
            tool_id / 3.0,
        ]], dtype=np.float32)

        return {
            "minimap": jnp.asarray(minimap),
            "scalars": jnp.asarray(scalars),
        }


# ── Rendering helpers ───────────────────────────────────────────────────────

def rgb_to_surface(rgb: np.ndarray, display: int) -> pygame.Surface:
    surf = pygame.Surface(rgb.shape[:2])
    pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (display, display))


def draw_star(screen, cx, cy, r_outer, r_inner=None, color=(255, 215, 0),
              outline=(0, 0, 0)):
    if r_inner is None:
        r_inner = r_outer * 0.38
    pts = []
    for i in range(10):
        r = r_outer if i % 2 == 0 else r_inner
        a = math.pi * i / 5 - math.pi / 2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, outline, pts, 1)


def hp_color(hp: float, hp_max: float):
    r = hp / max(hp_max, 1)
    if r > 0.6: return COLORS["hp_full"]
    if r > 0.3: return COLORS["hp_mid"]
    return COLORS["hp_low"]


def _visit_color(count: int) -> tuple[int, int, int]:
    """Map visit count to inferno-style color: bright red (1) → dark (5+)."""
    idx = min(count - 1, len(INFERNO_RAMP) - 1)
    return INFERNO_RAMP[idx]


def draw_compass_arrow(surface: pygame.Surface, cx: int, cy: int,
                       pos: tuple[int, int], target: tuple[int, int],
                       length: int = 28):
    """Draw a yellow compass arrow on a surface starting from (cx, cy)."""
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    dist = math.sqrt(dr * dr + dc * dc)
    if dist < 1.0:
        return
    color = (255, 215, 0)  # gold/yellow
    angle = math.atan2(dr, dc)
    tip_x = cx + int(length * math.cos(angle))
    tip_y = cy + int(length * math.sin(angle))

    pygame.draw.line(surface, color, (cx, cy), (tip_x, tip_y), 2)

    head_len = 7
    wing_angle = 0.45
    lx = tip_x - int(head_len * math.cos(angle - wing_angle))
    ly = tip_y - int(head_len * math.sin(angle - wing_angle))
    rx = tip_x - int(head_len * math.cos(angle + wing_angle))
    ry = tip_y - int(head_len * math.sin(angle + wing_angle))
    pygame.draw.polygon(surface, color, [(tip_x, tip_y), (lx, ly), (rx, ry)])


# ── Screens ─────────────────────────────────────────────────────────────────

def screen_main_menu(screen, clock, val_ok: bool,
                     ai_info: str = f"Drop models into {DEMO_ARTIFACTS_PATH.name}/",
                     ai_ready: bool = False):
    ft = pygame.font.Font(None, 72)
    fm = pygame.font.Font(None, 34)
    fs = pygame.font.Font(None, 22)
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:                              return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:                       return None
                if ev.key == pygame.K_h and val_ok:                 return "human"
                if ev.key == pygame.K_a:                            return "agent"
        screen.fill(COLORS["bg"])
        title = ft.render("Cogniland", True, COLORS["accent"])
        screen.blit(title, title.get_rect(center=(WINDOW_W // 2, 160)))
        screen.blit(fm.render("Choose a mode", True, COLORS["fg"]),
                    (WINDOW_W // 2 - 110, 250))

        options = [
            ("H", "Human", "Play on a val map" if val_ok
                           else "Val dataset missing — run generate_dataset.py",
             val_ok),
            ("A", "AI Agent", ai_info, ai_ready and val_ok),
        ]
        y = 340
        for key, label, desc, enabled in options:
            kcol = COLORS["accent"] if enabled else COLORS["dim"]
            lcol = COLORS["white"] if enabled else COLORS["dim"]
            screen.blit(fm.render(f"[{key}]", True, kcol), (WINDOW_W // 2 - 180, y))
            screen.blit(fm.render(f"  {label}", True, lcol), (WINDOW_W // 2 - 130, y))
            screen.blit(fs.render(desc, True, COLORS["dim"]), (WINDOW_W // 2 - 130, y + 32))
            y += 90

        screen.blit(fs.render("ESC — Quit", True, COLORS["dim"]),
                    (WINDOW_W // 2 - 40, WINDOW_H - 50))
        pygame.display.flip()
        clock.tick(30)


def screen_pick_map(screen, clock, dataset):
    rgbs = dataset["rgb"].numpy()
    biomes = dataset["biomes"]
    seeds = dataset["seeds"]
    N = rgbs.shape[0]

    thumbs = [rgb_to_surface(rgbs[i], 140) for i in range(N)]

    COLS, THUMB, PAD, TOP = 4, 140, 18, 90
    grid_w = COLS * THUMB + (COLS - 1) * PAD
    grid_x = (WINDOW_W - grid_w) // 2

    def rect(i):
        col, row = i % COLS, i // COLS
        return pygame.Rect(grid_x + col * (THUMB + PAD),
                           TOP + row * (THUMB + PAD + 26), THUMB, THUMB)

    fm = pygame.font.Font(None, 42)
    fs = pygame.font.Font(None, 20)
    sel = 0
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:                  return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:           return None
                if ev.key == pygame.K_RETURN:           return sel
                if ev.key == pygame.K_RIGHT: sel = (sel + 1) % N
                if ev.key == pygame.K_LEFT:  sel = (sel - 1) % N
                if ev.key == pygame.K_DOWN:  sel = min(sel + COLS, N - 1)
                if ev.key == pygame.K_UP:    sel = max(sel - COLS, 0)
            if ev.type == pygame.MOUSEMOTION:
                for i in range(N):
                    if rect(i).collidepoint(ev.pos):
                        sel = i
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for i in range(N):
                    if rect(i).collidepoint(ev.pos):
                        return i

        screen.fill(COLORS["bg"])
        screen.blit(fm.render("Select Map", True, COLORS["accent"]),
                    (WINDOW_W // 2 - 100, 28))
        for i in range(N):
            r = rect(i)
            screen.blit(thumbs[i], r.topleft)
            label = f"{biomes[i][:4]} s{seeds[i]}"
            lbl = fs.render(label, True, COLORS["fg"])
            screen.blit(lbl, (r.x + THUMB // 2 - lbl.get_width() // 2, r.y + THUMB + 4))
            color = COLORS["highlight"] if i == sel else COLORS["dim"]
            pygame.draw.rect(screen, color, r, 3 if i == sel else 1)
        hint = fs.render("Arrows / mouse  •  Enter / click to select  •  ESC to go back",
                         True, COLORS["dim"])
        screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 36))
        pygame.display.flip()
        clock.tick(30)


def draw_craft_menu(screen, game: CognilandGame, fm, fs):
    """Overlay craft menu in the center of the screen."""
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    box_w, box_h = 360, 260
    bx = (WINDOW_W - box_w) // 2
    by = (WINDOW_H - box_h) // 2
    pygame.draw.rect(screen, COLORS["craft_bg"], (bx, by, box_w, box_h))
    pygame.draw.rect(screen, COLORS["accent"], (bx, by, box_w, box_h), 2)

    title = fm.render("Craft a Tool", True, COLORS["accent"])
    screen.blit(title, title.get_rect(center=(WINDOW_W // 2, by + 30)))

    cost_txt = fs.render(f"Cost: {game.effects.craft_cost} wood  (you have {game.wood})",
                         True, COLORS["wood"])
    screen.blit(cost_txt, cost_txt.get_rect(center=(WINDOW_W // 2, by + 60)))

    can_craft = game.wood >= game.effects.craft_cost and game.tool is None
    y = by + 95
    for i, tool in enumerate(CRAFTABLE_TOOLS):
        key = str(i + 1)
        col = COLORS["fg"] if can_craft else COLORS["dim"]
        desc = {"raft": "cheaper water/ocean crossing",
                "rope": "cheaper rocky/mountain crossing",
                "shoes": "fast grassland after 5 consecutive steps"}[tool]
        screen.blit(fm.render(f"[{key}] {tool.capitalize()}", True, col), (bx + 30, y))
        screen.blit(fs.render(desc, True, COLORS["dim"]), (bx + 30, y + 28))
        y += 52

    if game.tool is not None:
        warn = fs.render("Already crafted a tool!", True, COLORS["hp_low"])
        screen.blit(warn, warn.get_rect(center=(WINDOW_W // 2, by + box_h - 30)))
    elif game.wood < game.effects.craft_cost:
        warn = fs.render(f"Need {game.effects.craft_cost} wood to craft", True, COLORS["hp_mid"])
        screen.blit(warn, warn.get_rect(center=(WINDOW_W // 2, by + box_h - 30)))

    esc_txt = fs.render("ESC / C — close", True, COLORS["dim"])
    screen.blit(esc_txt, esc_txt.get_rect(center=(WINDOW_W // 2, by + box_h - 10)))


def draw_game(screen, game: CognilandGame, fs, fm, fl, ft):
    screen.fill(COLORS["bg"])

    # ── Map ────────────────────────────────────────────────────────────────
    map_surf = rgb_to_surface(game.rgb, MAP_DISPLAY)
    MAP_X, MAP_Y = 20, 60
    screen.blit(map_surf, (MAP_X, MAP_Y))
    pygame.draw.rect(screen, COLORS["white"], (MAP_X, MAP_Y, MAP_DISPLAY, MAP_DISPLAY), 1)

    scale = MAP_DISPLAY / MAP_SIZE

    def w2s(r, c):
        return int(c * scale + scale / 2) + MAP_X, int(r * scale + scale / 2) + MAP_Y

    # Trajectory with inferno gradient based on visit count
    if len(game.path) >= 2:
        # Count visits per cell up to each step for segment coloring
        visit_counts: Counter[tuple[int, int]] = Counter()
        for i in range(len(game.path)):
            cell = game.path[i]
            visit_counts[cell] += 1
            if i > 0:
                # Color this segment by the visit count of the destination cell
                p0 = w2s(*game.path[i - 1])
                p1 = w2s(*cell)
                col = _visit_color(visit_counts[cell])
                pygame.draw.line(screen, col, p0, p1, 1)

    # Craft marker on trajectory
    if game.craft_step is not None and game.craft_step < len(game.path):
        cx, cy = w2s(*game.path[game.craft_step])
        sym = TOOL_SYMBOLS.get(game.craft_tool_name, "?")
        # Draw a small diamond + label
        pygame.draw.polygon(screen, COLORS["accent"],
                            [(cx, cy - 6), (cx + 5, cy), (cx, cy + 6), (cx - 5, cy)])
        pygame.draw.polygon(screen, COLORS["white"],
                            [(cx, cy - 6), (cx + 5, cy), (cx, cy + 6), (cx - 5, cy)], 1)
        lbl = fs.render(sym, True, COLORS["white"])
        screen.blit(lbl, (cx + 7, cy - 8))

    # Spawn, target, player
    sx, sy = w2s(*game.spawn)
    pygame.draw.circle(screen, (200, 200, 200), (sx, sy), 6, 1)
    draw_star(screen, *w2s(*game.target), r_outer=7, r_inner=3,
              color=(255, 215, 0), outline=(0, 0, 0))
    px, py = w2s(*game.pos)
    pygame.draw.circle(screen, COLORS["player"], (px, py), 7)
    pygame.draw.circle(screen, COLORS["white"],  (px, py), 7, 1)

    # Title
    title = fl.render(f"{game.biome.upper()}  seed={game.seed}", True, COLORS["accent"])
    screen.blit(title, (MAP_X, 20))

    # ── Right panel ─────────────────────────────────────────────────────────
    pygame.draw.rect(screen, COLORS["panel"],
                     (PANEL_X - 10, 50, PANEL_W + 20, WINDOW_H - 70))
    y = 70

    # ── Minimap (agent's view) ──────────────────────────────────────────────
    minimap_surf = render_minimap(
        game.rgb, game.heightmap,
        game.pos, game.target, game.last_terrain,
    )
    mm_x = PANEL_X + (PANEL_W - MINIMAP_DISPLAY) // 2
    mm_y = y
    screen.blit(minimap_surf, (mm_x, mm_y))
    pygame.draw.rect(screen, COLORS["dim"], (mm_x, mm_y, MINIMAP_DISPLAY, MINIMAP_DISPLAY), 1)
    screen.blit(fs.render("Agent view (occlusion)", True, COLORS["fg"]),
                (mm_x, mm_y - 16))
    y = mm_y + MINIMAP_DISPLAY + 10

    # Wood & tool
    screen.blit(fm.render(f"Wood: {game.wood}", True, COLORS["wood"]), (PANEL_X, y))
    if game.tool:
        tool_txt = fm.render(f"  [{game.tool.upper()}]", True, COLORS["accent"])
        screen.blit(tool_txt, (PANEL_X + 120, y))
    y += 28

    # Current step info
    cur = game.last_terrain
    drain_str = "-" if game.last_drain is None else f"-{game.last_drain}"
    for lbl, val, col in [
        ("Terrain", cur, COLORS["fg"]),
        ("Drain", drain_str, COLORS["fg"]),
        ("Steps", str(game.steps), COLORS["fg"]),
        ("Pos", f"({game.pos[0]}, {game.pos[1]})", COLORS["fg"]),
    ]:
        screen.blit(fs.render(lbl, True, COLORS["dim"]), (PANEL_X, y))
        screen.blit(fs.render(val, True, col), (PANEL_X + 70, y))
        y += 20
    y += 4

    # Forage message
    if game.forage_msg_timer > 0:
        fcol = COLORS["forage"] if "+" in game.last_forage_msg else COLORS["dim"]
        screen.blit(fs.render(game.last_forage_msg, True, fcol), (PANEL_X, y))
        game.forage_msg_timer -= 1
    y += 20

    # ── HP plot ─────────────────────────────────────────────────────────────
    screen.blit(fs.render(f"HP: {int(game.hp)}/{game.effects.hp_max}", True,
                          hp_color(game.hp, game.effects.hp_max)), (PANEL_X, y))
    y += 16
    PW, PH = PANEL_W - 10, 80
    plot = pygame.Surface((PW, PH))
    plot.fill((15, 15, 22))
    pygame.draw.rect(plot, COLORS["dim"], (0, 0, PW, PH), 1)
    hist = game.hp_history
    n = max(len(hist) - 1, 1)
    def _x(i): return int(i / n * (PW - 2)) + 1
    def _y(h): return PH - 1 - int(min(max(h, 0), game.effects.hp_max) / game.effects.hp_max * (PH - 2))
    if len(hist) >= 2:
        pts = [(_x(i), _y(h)) for i, h in enumerate(hist)]
        pygame.draw.lines(plot, COLORS["hp_full"], False, pts, 2)
    screen.blit(plot, (PANEL_X, y))
    y += PH + 8

    # Controls
    screen.blit(fs.render("Controls", True, COLORS["accent"]), (PANEL_X, y))
    y += 16
    for line in ["WASD / arrows - move",
                 "F - forage (berry=HP, forest=wood)",
                 f"C - craft tool ({game.effects.craft_cost} wood)",
                 "R - reset  |  ESC - menu"]:
        screen.blit(fs.render(line, True, COLORS["dim"]), (PANEL_X, y))
        y += 16

    # Game over overlay
    if game.game_over:
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        msg = "TARGET REACHED!" if game.won else "YOU DIED"
        mcol = COLORS["hp_full"] if game.won else COLORS["hp_low"]
        surf = ft.render(msg, True, mcol)
        screen.blit(surf, surf.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 40)))
        sub = fm.render(f"Steps: {game.steps}   HP: {int(game.hp)}   Wood: {game.wood}",
                        True, COLORS["white"])
        screen.blit(sub, sub.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 10)))
        tool_msg = f"Tool: {game.tool or 'none'}"
        sub2 = fs.render(tool_msg, True, COLORS["dim"])
        screen.blit(sub2, sub2.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 40)))
        hint = fs.render("R = new game   |   ESC = menu", True, COLORS["dim"])
        screen.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 70)))


def screen_play(screen, clock, dataset, idx: int):
    rgb = dataset["rgb"][idx].numpy()
    heightmap = dataset["heightmap"][idx].numpy()
    tidx = dataset["terrain_idx"][idx].numpy()
    mask = dataset["berry_mask"][idx].numpy()
    biome = dataset["biomes"][idx]
    seed = int(dataset["seeds"][idx])

    def make_game():
        return CognilandGame(rgb, heightmap, tidx, mask, biome, seed)

    game = make_game()
    fs = pygame.font.Font(None, 22)
    fm = pygame.font.Font(None, 30)
    fl = pygame.font.Font(None, 38)
    ft = pygame.font.Font(None, 64)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                # Craft menu handling
                if game.crafting:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_c):
                        game.crafting = False
                        continue
                    if ev.key == pygame.K_1:
                        game.craft("raft")
                        game.crafting = False
                        continue
                    if ev.key == pygame.K_2:
                        game.craft("rope")
                        game.crafting = False
                        continue
                    if ev.key == pygame.K_3:
                        game.craft("shoes")
                        game.crafting = False
                        continue
                    continue  # swallow other keys while crafting

                if ev.key == pygame.K_ESCAPE:
                    return "menu"
                if ev.key == pygame.K_r:
                    game = make_game()
                    continue
                if ev.key == pygame.K_c and not game.game_over:
                    game.crafting = True
                    continue
                if ev.key == pygame.K_f and not game.game_over:
                    game.forage()
                    continue
                if ev.key in ACTIONS and not game.game_over:
                    dr, dc = ACTIONS[ev.key]
                    game.step(dr, dc)

        draw_game(screen, game, fs, fm, fl, ft)
        if game.crafting:
            draw_craft_menu(screen, game, fm, fs)
        pygame.display.flip()
        clock.tick(60)


def screen_agent_stub(screen, clock, lines: list[str] | None = None):
    """Fallback screen shown when no models are available (or loading failed)."""
    fm = pygame.font.Font(None, 36)
    fs = pygame.font.Font(None, 22)
    lines = lines or [
        "No trained models found.",
        f"Drop a run's checkpoints/ folder into  {DEMO_ARTIFACTS_PATH.name}/<name>/",
        "(each folder needs run_config.yaml + <env_id>/best/).",
    ]
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:              return "quit"
            if ev.type == pygame.KEYDOWN:           return "menu"
        screen.fill(COLORS["bg"])
        title = fm.render("AI Agent", True, COLORS["accent"])
        screen.blit(title, title.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 80)))
        y = WINDOW_H // 2 - 20
        for line in lines:
            surf = fs.render(line, True, COLORS["fg"])
            screen.blit(surf, surf.get_rect(center=(WINDOW_W // 2, y)))
            y += 28
        hint = fs.render("Press any key to go back.", True, COLORS["dim"])
        screen.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H - 70)))
        pygame.display.flip()
        clock.tick(30)


def screen_pick_model(screen, clock, models: list[ModelInfo]) -> int | None:
    """Vertical list picker for AI models. Returns the selected index, or None on ESC."""
    fh = pygame.font.Font(None, 56)
    fm = pygame.font.Font(None, 30)
    fs = pygame.font.Font(None, 22)
    fx = pygame.font.Font(None, 18)

    N = len(models)
    sel = 0

    row_h = 78
    pad = 40
    list_y = 110
    list_w = min(720, WINDOW_W - 2 * pad)
    list_x = (WINDOW_W - list_w) // 2

    def row_rect(i: int) -> pygame.Rect:
        return pygame.Rect(list_x, list_y + i * (row_h + 10), list_w, row_h)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return None
                if ev.key == pygame.K_RETURN:
                    return sel
                if ev.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % N
                if ev.key in (pygame.K_UP, pygame.K_w):
                    sel = (sel - 1) % N
            if ev.type == pygame.MOUSEMOTION:
                for i in range(N):
                    if row_rect(i).collidepoint(ev.pos):
                        sel = i
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for i in range(N):
                    if row_rect(i).collidepoint(ev.pos):
                        return i

        screen.fill(COLORS["bg"])
        title = fh.render("Select Model", True, COLORS["accent"])
        screen.blit(title, title.get_rect(center=(WINDOW_W // 2, 55)))

        for i, mi in enumerate(models):
            rect = row_rect(i)
            is_sel = (i == sel)
            bg = COLORS["craft_bg"] if is_sel else COLORS["panel"]
            edge = COLORS["highlight"] if is_sel else COLORS["dim"]
            pygame.draw.rect(screen, bg, rect)
            pygame.draw.rect(screen, edge, rect, 2 if is_sel else 1)

            name_col = COLORS["white"] if is_sel else COLORS["fg"]
            screen.blit(fm.render(mi.name, True, name_col),
                        (rect.x + 20, rect.y + 10))
            screen.blit(fs.render(mi.summary(), True, COLORS["dim"]),
                        (rect.x + 20, rect.y + 42))
            # Right-justified path hint.
            try:
                path_str = str(mi.path.relative_to(PROJECT_ROOT))
            except ValueError:
                path_str = str(mi.path)
            path_surf = fx.render(path_str, True, COLORS["dim"])
            screen.blit(path_surf,
                        (rect.right - path_surf.get_width() - 16, rect.y + 12))

        hint = fs.render(
            "Up/Down/Enter  •  click to pick  •  ESC to go back",
            True, COLORS["dim"],
        )
        screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 36))
        pygame.display.flip()
        clock.tick(30)


def _load_model_with_splash(screen, mi: ModelInfo) -> tuple[AIAgentPlayer | None, str | None]:
    """Show a loading splash and load the selected model. Returns (player, error)."""
    screen.fill(COLORS["bg"])
    font = pygame.font.Font(None, 34)
    subfont = pygame.font.Font(None, 22)
    msg = font.render(f"Loading  {mi.name} ...", True, COLORS["accent"])
    screen.blit(msg, msg.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 10)))
    sub = subfont.render("(JAX warm-up may take several seconds on first launch)",
                         True, COLORS["dim"])
    screen.blit(sub, sub.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 26)))
    pygame.display.flip()

    try:
        return AIAgentPlayer(mi.path), None
    except Exception as e:  # noqa: BLE001 — surface any load failure to UI.
        err = f"{type(e).__name__}: {e}"
        print(f"[demo] Failed to load model '{mi.name}': {err}", file=sys.stderr)
        return None, err


def _draw_ai_overlay(screen, fm, fs, player: AIAgentPlayer,
                     paused: bool, step_hz: float):
    """Small status strip under the map: AI info + playback state."""
    strip_h = 32
    strip = pygame.Surface((MAP_DISPLAY, strip_h), pygame.SRCALPHA)
    strip.fill((0, 0, 0, 140))
    screen.blit(strip, (20, 60 + MAP_DISPLAY + 2))

    label = "PAUSED" if paused else f"{step_hz:.1f} steps/s"
    mode = "greedy" if player.deterministic else "sampled"
    act_name = ACTION_LABELS[player.last_action] if player.last_action is not None else "—"
    text = (f"AI  [{label}]   task={player.task_id}   policy={mode}   "
            f"last={act_name}   ({player._which} ckpt)")
    screen.blit(fs.render(text, True, COLORS["white"]),
                (28, 60 + MAP_DISPLAY + 8))


def screen_ai_play(screen, clock, dataset, idx: int, player: AIAgentPlayer):
    """AI-controlled playback on the selected map.

    Controls:
      SPACE     pause / resume
      N         single-step while paused
      R         reset (same map, re-sample spawn/target)
      [  /  ]   slower / faster (1-30 actions per second)
      D         toggle deterministic (greedy) vs sampled policy
      T         cycle task id (0..task_embedding_dim-1)
      ESC       back to menu
    """
    rgb = dataset["rgb"][idx].numpy()
    heightmap = dataset["heightmap"][idx].numpy()
    tidx = dataset["terrain_idx"][idx].numpy()
    mask = dataset["berry_mask"][idx].numpy()
    biome = dataset["biomes"][idx]
    seed = int(dataset["seeds"][idx])

    vis_lut = dataset.get("visibility_lut", None)
    if vis_lut is None:
        return "no_vis_lut"
    vis_lut_map = (vis_lut[idx].numpy()
                   if hasattr(vis_lut[idx], "numpy") else np.asarray(vis_lut[idx]))
    vis_lut_map = vis_lut_map.astype(np.uint8)

    def make_game():
        return CognilandGame(rgb, heightmap, tidx, mask, biome, seed)

    game = make_game()
    player.reset()

    fs = pygame.font.Font(None, 22)
    fm = pygame.font.Font(None, 30)
    fl = pygame.font.Font(None, 38)
    ft = pygame.font.Font(None, 64)

    paused = False
    step_hz = 8.0
    acc = 0.0
    single_step = False

    while True:
        dt = clock.tick(60) / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return "menu"
                if ev.key == pygame.K_r:
                    game = make_game()
                    player.reset()
                    acc = 0.0
                    continue
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                    continue
                if ev.key == pygame.K_n and paused:
                    single_step = True
                    continue
                if ev.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    step_hz = max(1.0, step_hz / 1.5)
                    continue
                if ev.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS):
                    step_hz = min(30.0, step_hz * 1.5)
                    continue
                if ev.key == pygame.K_d:
                    player.deterministic = not player.deterministic
                    continue
                if ev.key == pygame.K_t:
                    player.task_id = (player.task_id + 1) % player._task_embedding_dim
                    player.reset()
                    game = make_game()
                    acc = 0.0
                    continue

        # Step the agent.
        if not game.game_over:
            if paused and single_step:
                action = player.act(game, vis_lut_map)
                player.apply_action(game, action)
                single_step = False
            elif not paused:
                acc += dt * step_hz
                while acc >= 1.0 and not game.game_over:
                    acc -= 1.0
                    action = player.act(game, vis_lut_map)
                    player.apply_action(game, action)

        draw_game(screen, game, fs, fm, fl, ft)
        _draw_ai_overlay(screen, fm, fs, player, paused, step_hz)
        pygame.display.flip()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Cogniland")
    clock = pygame.time.Clock()

    dataset = load_val_dataset()
    val_ok = dataset is not None

    # Scan the shipped artifacts folder for selectable models.
    models = scan_demo_artifacts()
    if models:
        ai_info = (f"{len(models)} model{'s' if len(models) != 1 else ''} available "
                   f"in {DEMO_ARTIFACTS_PATH.name}/")
    else:
        ai_info = f"No models in {DEMO_ARTIFACTS_PATH.name}/"

    while True:
        mode = screen_main_menu(
            screen, clock, val_ok,
            ai_info=ai_info,
            ai_ready=bool(models) and val_ok,
        )
        if mode is None:
            break

        if mode == "agent":
            if not models or not val_ok:
                lines: list[str] = []
                if not models:
                    lines += [
                        "No trained models found.",
                        f"Drop a checkpoint folder into {DEMO_ARTIFACTS_PATH.name}/<name>/",
                        "(each folder needs run_config.yaml + <env_id>/best/).",
                    ]
                if not val_ok:
                    if lines:
                        lines.append("")
                    lines.append("val dataset missing — run: python scripts/generate_dataset.py")
                if screen_agent_stub(screen, clock, lines) == "quit":
                    break
                continue

            # Model picker → load → map picker → play loop.
            while True:
                model_idx = screen_pick_model(screen, clock, models)
                if model_idx is None:
                    break
                mi = models[model_idx]
                player, err = _load_model_with_splash(screen, mi)
                if player is None:
                    if screen_agent_stub(
                        screen, clock,
                        [f"Failed to load  {mi.name}", "",
                         err or "(unknown error — see terminal)"],
                    ) == "quit":
                        pygame.quit(); sys.exit()
                    continue

                while True:
                    idx = screen_pick_map(screen, clock, dataset)
                    if idx is None:
                        break
                    result = screen_ai_play(screen, clock, dataset, idx, player)
                    if result == "quit":
                        pygame.quit(); sys.exit()
                    if result == "no_vis_lut":
                        stub = screen_agent_stub(
                            screen, clock,
                            ["val dataset missing visibility_lut.",
                             "Rerun: python scripts/generate_dataset.py"],
                        )
                        if stub == "quit":
                            pygame.quit(); sys.exit()
                        break
            continue

        if mode == "human":
            while True:
                idx = screen_pick_map(screen, clock, dataset)
                if idx is None:
                    break
                result = screen_play(screen, clock, dataset, idx)
                if result == "quit":
                    pygame.quit(); sys.exit()

    pygame.quit()


if __name__ == "__main__":
    main()
