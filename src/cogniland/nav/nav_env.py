"""Gymnasium environment — partially-observable navigation with a
permanent build choice.

The agent always sees an RGB local crop centred on itself and a single
binary scalar telling it whether *any* object has been built. The agent
**does not see which object** was built — that secret is its own to
remember. The build decision is committed once: subsequent build actions
are noops but still incur the build cost.
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import skills as sk
from .mapgen import MapRecord, generate_map
from .renderer import SpriteSheet
from .tiles import NUM_TILES, TARGET, WATER, ROCK

MapType = Literal["lake", "rocky", "balanced", "random"]

_MOVE_DELTAS = {
    0: (-1, 0),   # up
    1: (+1, 0),   # down
    2: (0, -1),   # left
    3: (0, +1),   # right
}
_FACING_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}
BUILD_ACTION = 4
NUM_ACTIONS = 5  # up/down/left/right/build


class CognilandNavEnv(gym.Env):
    """Single-agent navigation env with a one-shot build commitment.

    Parameters
    ----------
    size:
        Map side length (32 / 64 / 96 / 128).
    map_type:
        ``"lake"``, ``"rocky"`` or ``"random"`` (default).
    view_size:
        Side length of the partial-observation window in tiles. Must be odd.
    tile_px:
        Render resolution per tile (default 16; matches the Crafter PNG).
    seed:
        Default seed for ``reset(seed=None)``.
    include_semantic:
        If True, observations include a `(view_size, view_size)` tile-id
        crop in addition to the RGB image.
    render_mode:
        ``None`` / ``"human"`` / ``"rgb_array"``. Human mode opens a
        pygame window on first ``render()`` call.
    max_steps:
        Episode truncation length. Default is ``4 * size``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        size: int = 64,
        map_type: MapType = "random",
        view_size: int = 21,
        tile_px: int = 16,
        seed: int = 0,
        obs_mode: str = "symbolic",
        include_semantic: bool | None = None,  # legacy alias
        render_mode: str | None = None,
        max_steps: int | None = None,
        map_record: MapRecord | None = None,
    ) -> None:
        super().__init__()
        if view_size % 2 == 0 or view_size < 3:
            raise ValueError(f"view_size={view_size} must be odd and ≥ 3")
        if obs_mode not in ("symbolic", "rgb", "both"):
            raise ValueError(f"obs_mode={obs_mode!r} must be 'symbolic', 'rgb', or 'both'")
        self.size = int(size)
        self.map_type: MapType = map_type
        self.view_size = int(view_size)
        self.tile_px = int(tile_px)
        self.obs_mode = obs_mode
        # Legacy: include_semantic=True forces semantic into the obs even in rgb mode
        if include_semantic is True and obs_mode == "rgb":
            self.obs_mode = "both"
        self.render_mode = render_mode
        self.max_steps = int(max_steps) if max_steps is not None else 1000
        self._seed = int(seed)
        self._fixed_record = map_record

        self._rng = np.random.default_rng(self._seed)
        self._sprites: SpriteSheet | None = None  # lazy
        self._window = None  # pygame Surface (lazy)

        self.action_space = spaces.Dict(
            {
                "move": spaces.Discrete(NUM_ACTIONS),
                "build_scalar": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        obs_dict: dict[str, spaces.Space] = {
            "skill_active": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }
        if self.obs_mode in ("symbolic", "both"):
            obs_dict["semantic"] = spaces.Box(
                low=0, high=NUM_TILES, shape=(self.view_size, self.view_size), dtype=np.int8
            )
        if self.obs_mode in ("rgb", "both"):
            img_shape = (3, self.view_size * self.tile_px, self.view_size * self.tile_px)
            obs_dict["image"] = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_dict)
        self.include_semantic = self.obs_mode in ("symbolic", "both")

        # state — filled by reset
        self._record: MapRecord | None = None
        self._pos: tuple[int, int] = (0, 0)
        self._target: tuple[int, int] = (0, 0)
        self._active_object: int = sk.NONE
        self._last_build_scalar: float = float("nan")
        self._step_count: int = 0
        self._episode_return: float = 0.0
        self._facing: int = 1  # down by default

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
        if self._fixed_record is not None:
            self._record = self._fixed_record
        else:
            map_seed = int(self._rng.integers(0, 2**31))
            self._record = generate_map(
                size=self.size,
                map_type=self.map_type,
                seed=map_seed,
            )
        rec = self._record
        # Fill ctg arrays if not already present (e.g. injected fixed_record).
        self._ensure_ctg()
        self._pos = (int(rec.spawn[0]), int(rec.spawn[1]))
        self._target = (int(rec.target[0]), int(rec.target[1]))
        self._active_object = sk.NONE
        self._last_build_scalar = float("nan")
        self._step_count = 0
        self._episode_return = 0.0
        self._facing = 1

        obs = self._make_obs()
        info = self._make_info(
            reward=0.0,
            collision=False,
            invalid_build=False,
            reached=False,
            slipped=False,
        )
        return obs, info

    def _ensure_ctg(self) -> None:
        rec = self._record
        assert rec is not None
        if rec.ctg_none is None or rec.ctg_raft is None or rec.ctg_harness is None:
            from .mapgen import cost_to_go_unit
            tgt = (int(rec.target[0]), int(rec.target[1]))
            rec.ctg_none = cost_to_go_unit(rec.terrain, tgt, sk.NONE).astype(np.float32)
            rec.ctg_raft = cost_to_go_unit(rec.terrain, tgt, sk.RAFT).astype(np.float32)
            rec.ctg_harness = cost_to_go_unit(rec.terrain, tgt, sk.HARNESS).astype(np.float32)

    def _ctg(self, obj: int) -> np.ndarray:
        rec = self._record
        assert rec is not None
        if obj == sk.RAFT:
            return rec.ctg_raft  # type: ignore[return-value]
        if obj == sk.HARNESS:
            return rec.ctg_harness  # type: ignore[return-value]
        return rec.ctg_none  # type: ignore[return-value]

    @staticmethod
    def _delta_ctg(ctg: np.ndarray, old_pos, new_pos) -> float:
        """``ctg_old − ctg_new`` (positive = move toward goal)."""
        c0 = float(ctg[old_pos])
        c1 = float(ctg[new_pos])
        if not (np.isfinite(c0) and np.isfinite(c1)):
            return 0.0
        return c0 - c1

    def step(self, action):  # type: ignore[override]
        if self._record is None:
            raise RuntimeError("step() called before reset()")
        rec = self._record

        move = self._extract_move(action)
        build_scalar_arr = self._extract_build_scalar(action)
        build_scalar = float(np.clip(build_scalar_arr[0], -1.0, 1.0))

        # Every action pays the flat slack penalty up-front.
        reward = float(sk.SLACK_PENALTY)
        collision = False
        invalid_build = False
        reached = False
        slipped = False
        terminated = False

        if move == BUILD_ACTION:
            # Build is a pure slack-only action — no shaping (unit-cost ctg
            # is identical across skills now that water/rock are universally
            # walkable). The benefit of the right skill shows up later via
            # the slip mechanic.
            if self._active_object == sk.NONE:
                self._active_object = sk.object_from_scalar(build_scalar)
                self._last_build_scalar = build_scalar
            else:
                invalid_build = True
        elif move in _MOVE_DELTAS:
            dr, dc = _MOVE_DELTAS[move]
            self._facing = move
            nr, nc = self._pos[0] + dr, self._pos[1] + dc
            in_bounds = 0 <= nr < self.size and 0 <= nc < self.size
            if not in_bounds:
                collision = True
                # stay still → only the slack penalty applies
            else:
                tile = int(rec.terrain[nr, nc])
                if not sk.walkable(self._active_object, tile):
                    collision = True
                else:
                    if self._rng.random() < sk.slip_chance(self._active_object, tile):
                        slipped = True
                    else:
                        old_pos = self._pos
                        self._pos = (nr, nc)
                        delta = self._delta_ctg(
                            self._ctg(self._active_object), old_pos, self._pos
                        )
                        reward += sk.SHAPING_COEF * delta
                        if tile == TARGET:
                            reward += sk.REACH_BONUS
                            reached = True
                            terminated = True
        else:
            raise ValueError(f"invalid move action: {move!r}")

        self._step_count += 1
        self._episode_return += reward
        truncated = (not terminated) and self._step_count >= self.max_steps

        obs = self._make_obs()
        info = self._make_info(
            reward=reward,
            collision=collision,
            invalid_build=invalid_build,
            reached=reached,
            slipped=slipped,
        )
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------ rendering

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_full_array()
        if self.render_mode == "human":
            self._render_human()
            return None
        return None

    def _render_full_array(self) -> np.ndarray:
        if self._record is None:
            raise RuntimeError("render() called before reset()")
        if self._sprites is None:
            self._sprites = SpriteSheet(tile_px=self.tile_px)
        rec = self._record
        half = self.view_size // 2
        view_rect = (self._pos[0] - half, self._pos[1] - half, self.view_size, self.view_size)
        return self._sprites.render_full(
            rec.terrain,
            self._pos,
            self._target,
            view_rect=view_rect,
            agent_facing=_FACING_NAMES[self._facing],
        )

    def _render_human(self) -> None:
        import pygame  # local — pygame is optional for headless callers

        if self._window is None:
            pygame.init()
            img = self._render_full_array()
            self._window = pygame.display.set_mode((img.shape[1], img.shape[0]))
            pygame.display.set_caption("Cogniland Nav")
        else:
            img = self._render_full_array()
        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        self._window.blit(surface, (0, 0))
        pygame.display.flip()

    def close(self):
        if self._window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._window = None

    # -------------------------------------------------------------- helpers

    def _extract_move(self, action: Any) -> int:
        if isinstance(action, dict):
            return int(action["move"])
        if isinstance(action, (tuple, list)) and len(action) == 2:
            return int(action[0])
        return int(action)  # fall back: treat as raw move

    def _extract_build_scalar(self, action: Any) -> np.ndarray:
        if isinstance(action, dict):
            return np.asarray(action.get("build_scalar", [0.0]), dtype=np.float32).reshape(-1)
        if isinstance(action, (tuple, list)) and len(action) == 2:
            return np.asarray(action[1], dtype=np.float32).reshape(-1)
        return np.zeros((1,), dtype=np.float32)

    def _make_obs(self) -> dict[str, np.ndarray]:
        assert self._record is not None
        skill_active = np.array(
            [1.0 if self._active_object != sk.NONE else 0.0], dtype=np.float32
        )
        obs: dict[str, np.ndarray] = {"skill_active": skill_active}
        if self.obs_mode in ("symbolic", "both"):
            obs["semantic"] = self._semantic_crop()
        if self.obs_mode in ("rgb", "both"):
            if self._sprites is None:
                self._sprites = SpriteSheet(tile_px=self.tile_px)
            obs["image"] = self._sprites.render_observation(
                self._record.terrain,
                self._pos,
                view_size=self.view_size,
                agent_facing=_FACING_NAMES[self._facing],
            )
        return obs

    def _semantic_crop(self) -> np.ndarray:
        assert self._record is not None
        half = self.view_size // 2
        out = np.zeros((self.view_size, self.view_size), dtype=np.int8)
        r0, c0 = self._pos[0] - half, self._pos[1] - half
        for vr in range(self.view_size):
            mr = r0 + vr
            for vc in range(self.view_size):
                mc = c0 + vc
                if 0 <= mr < self.size and 0 <= mc < self.size:
                    out[vr, vc] = int(self._record.terrain[mr, mc])
                else:
                    out[vr, vc] = 6  # OOB tile id (see tiles.py)
        return out

    def _make_info(
        self,
        reward: float,
        collision: bool,
        invalid_build: bool,
        reached: bool,
        slipped: bool = False,
    ) -> dict[str, Any]:
        assert self._record is not None
        rec = self._record
        cur_tile = int(rec.terrain[self._pos[0], self._pos[1]])
        return {
            "position": self._pos,
            "target": self._target,
            "active_object": sk.OBJECT_NAMES[self._active_object],
            "skill_active": 1 if self._active_object != sk.NONE else 0,
            "last_build_scalar": self._last_build_scalar,
            "map_type": rec.map_type,
            "correct_object": sk.OBJECT_NAMES[rec.correct_object],
            "step": self._step_count,
            "episode_return": self._episode_return,
            "current_tile": cur_tile,
            "collision": collision,
            "invalid_build": invalid_build,
            "reached_target": reached,
            "slipped": slipped,
            "no_skill_oracle_cost": rec.no_skill_cost,
            "raft_oracle_cost": rec.raft_cost,
            "harness_oracle_cost": rec.harness_cost,
        }
