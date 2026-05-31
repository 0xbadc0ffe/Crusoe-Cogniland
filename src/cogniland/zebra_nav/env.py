"""Zebra-stripe POMDP env.

Action space
------------
``Discrete(6)``:
    0 = move UP     (also faces up)
    1 = move DOWN
    2 = move LEFT
    3 = move RIGHT
    4 = PLACE      (if the cell in the facing direction is WATER → WOOD)
    5 = MINE       (if the cell in the facing direction is ROCK  → GRASS)

A move action always updates ``facing`` to that direction, even if the move is
blocked by an impassable tile. PLACE and MINE never change the agent's
position; they only modify the tile in front of it. Obsidian is inviolable.

Observation
-----------
Egocentric crop of size ``view_size × view_size`` of tile ids (int8) centred on
the agent, plus a scalar vector ``[face_one_hot (4), step/max]``. Cells outside
the world get the ``OOB`` tile id.

Reward
------
``-0.01`` per step (slack), ``+1.0`` on the step that reaches the target, plus
PBRS shaping ``shaping_coef · (γ·φ(s') − φ(s))`` with potential ``φ = −ctg``,
where ``ctg`` is the BFS cost-to-go to the target over all non-obsidian cells
(water/rock counted as unit-cost, since they can be bridged / mined). Because
the thick side has more cells to cross, its cost-to-go is higher — so the
shaping itself nudges the agent toward the thinner side, on top of the slack
paid for the extra PLACE / MINE actions.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .mapgen import MapRecord, generate_zebra_map
from .tiles import (
    GRASS, NUM_TILES, OOB, ROCK, TARGET, TILE_COLORS, TREE, WATER, WOOD,
    is_walkable,
)

# action ids
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_PLACE, A_MINE = range(6)
NUM_ACTIONS = 6

# facing ids (same order as the move actions)
F_UP, F_DOWN, F_LEFT, F_RIGHT = range(4)
_FACE_DELTA = {
    F_UP:    (-1, 0),
    F_DOWN:  (+1, 0),
    F_LEFT:  (0, -1),
    F_RIGHT: (0, +1),
}
_MOVE_TO_FACING = {A_UP: F_UP, A_DOWN: F_DOWN, A_LEFT: F_LEFT, A_RIGHT: F_RIGHT}


class ZebraNavEnv(gym.Env):
    """Crafter-style 32×32 zebra-stripe navigation env.

    A new map is generated on every reset unless ``map_record=`` is supplied
    (useful for fixed-map evaluation).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        size: int = 32,
        view_size: int = 11,
        max_steps: int = 1000,        # generous timeout: success = reaching ability, not speed
        n_stripes: int = 4,
        thick_half: int = 3,
        thin_half: int = 1,
        obsidian_half: int = 1,
        window_h: int = 3,
        orientation: str = "natural",    # only "natural" supported (stripes retired)
        width: int | None = None,        # map width; height = size (default square)
        water_frac: float = 0.14,        # natural-only: water coverage
        rock_frac: float = 0.14,         # natural-only: rock coverage
        tree_frac: float = 0.03,         # natural-only: impassable tree coverage
        goal_half: int | None = None,    # natural: None ⇒ whole right wall is goal; N ⇒ central door
        seed: int = 0,
        map_record: MapRecord | None = None,
        slack_penalty: float = -0.01,
        reach_bonus: float = 1.0,
        shaping_coef: float = 0.01,
        build_cost: float = 0.02,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        if view_size % 2 == 0 or view_size < 3:
            raise ValueError(f"view_size={view_size} must be odd and >= 3")
        self.size = int(size)
        self.height = int(size)
        self.width = int(width) if width is not None else int(size)
        self.view_size = int(view_size)
        self.max_steps = int(max_steps)
        self.n_stripes = int(n_stripes)
        self.thick_half = int(thick_half)
        self.thin_half = int(thin_half)
        self.obsidian_half = int(obsidian_half)
        self.window_h = int(window_h)
        if orientation != "natural":
            raise ValueError(
                f"orientation must be 'natural' (stripes retired), got {orientation!r}")
        self.orientation = orientation
        self.water_frac = float(water_frac)
        self.rock_frac = float(rock_frac)
        self.tree_frac = float(tree_frac)
        self.goal_half = goal_half
        self.slack_penalty = float(slack_penalty)
        self.reach_bonus = float(reach_bonus)
        self.shaping_coef = float(shaping_coef)
        self.build_cost = float(build_cost)   # extra penalty per successful PLACE/MINE
        self.gamma = float(gamma)
        self._seed = int(seed)
        self._fixed_record = map_record
        self._rng = np.random.default_rng(self._seed)

        # spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "minimap": spaces.Box(low=0, high=NUM_TILES - 1,
                                  shape=(self.view_size, self.view_size),
                                  dtype=np.int8),
            "scalars": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        })

        # state — populated by reset()
        self._record: MapRecord | None = None
        self._terrain: np.ndarray | None = None
        self._ctg: np.ndarray | None = None       # BFS cost-to-go potential field
        self._pos: tuple[int, int] = (0, 0)
        self._facing: int = F_RIGHT
        self._step_count: int = 0
        self._episode_return: float = 0.0
        self._traj: list[tuple[int, int]] = []     # positions this episode

    # ── lifecycle ────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
        if self._fixed_record is not None:
            self._record = self._fixed_record
        else:
            sub = int(self._rng.integers(0, 2 ** 31))
            self._record = generate_zebra_map(
                size=self.height, width=self.width, seed=sub,
                orientation=self.orientation, water_frac=self.water_frac,
                rock_frac=self.rock_frac, tree_frac=self.tree_frac,
                goal_half=self.goal_half,
            )
        self._terrain = self._record.terrain.copy()       # mutable per-episode
        self._ctg = self._compute_ctg(self._terrain, self._record.target)
        self._pos = tuple(self._record.spawn)
        self._facing = F_RIGHT
        self._step_count = 0
        self._episode_return = 0.0
        self._traj = [self._pos]
        return self._make_obs(), self._make_info(reward=0.0, mined=False,
                                                 placed=False, blocked=False,
                                                 reached=False)

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._terrain is None:
            raise RuntimeError("step() called before reset()")
        action = int(action)
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"action {action} out of range [0,{NUM_ACTIONS})")

        reward = self.slack_penalty
        mined = placed = blocked = reached = False
        ctg_prev = float(self._ctg[self._pos]) if self._ctg is not None else 0.0

        if action in _MOVE_TO_FACING:                # move (and face)
            self._facing = _MOVE_TO_FACING[action]
            dr, dc = _FACE_DELTA[self._facing]
            nr, nc = self._pos[0] + dr, self._pos[1] + dc
            if 0 <= nr < self.height and 0 <= nc < self.width \
                    and is_walkable(int(self._terrain[nr, nc])):
                self._pos = (nr, nc)
                if self._terrain[nr, nc] == TARGET:
                    reward += self.reach_bonus
                    reached = True
            else:
                blocked = True
        elif action == A_PLACE:                       # bridge water → wood
            dr, dc = _FACE_DELTA[self._facing]
            fr, fc = self._pos[0] + dr, self._pos[1] + dc
            if 0 <= fr < self.height and 0 <= fc < self.width \
                    and self._terrain[fr, fc] == WATER:
                self._terrain[fr, fc] = WOOD
                placed = True
                reward -= self.build_cost
        elif action == A_MINE:                        # mine rock → grass
            dr, dc = _FACE_DELTA[self._facing]
            fr, fc = self._pos[0] + dr, self._pos[1] + dc
            if 0 <= fr < self.height and 0 <= fc < self.width \
                    and self._terrain[fr, fc] == ROCK:
                self._terrain[fr, fc] = GRASS
                mined = True
                reward -= self.build_cost

        # PBRS shaping: γ·φ(s') − φ(s) with φ = −ctg (cost-to-go to target).
        # Crossing the thick side visits more cells → higher ctg → the shaping
        # itself rewards committing to the thinner side.
        if self.shaping_coef != 0.0 and self._ctg is not None:
            ctg_curr = float(self._ctg[self._pos])
            reward += self.shaping_coef * (ctg_prev - self.gamma * ctg_curr)

        self._step_count += 1
        self._episode_return += reward
        self._traj.append(self._pos)
        terminated = reached
        truncated = (not terminated) and (self._step_count >= self.max_steps)
        info = self._make_info(reward=reward, mined=mined, placed=placed,
                               blocked=blocked, reached=reached)
        if terminated or truncated:
            n_ok, n_tot = self._thin_side_accuracy()
            info["thin_correct"] = n_ok
            info["thin_total"] = n_tot
        return self._make_obs(), float(reward), terminated, truncated, info

    def _thin_side_accuracy(self) -> tuple[int, int]:
        """Retired stripe metric — natural maps have no discrete thin/thick
        choice, so this is always ``(0, 0)``. Kept for caller compatibility."""
        return 0, 0

    # ── cost-to-go potential ─────────────────────────────────────────────

    @staticmethod
    def _compute_ctg(terrain: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        """**Min-action** cost-to-go to the goal (Dijkstra). Entering a cell costs
        1 action on walkable land (grass/wood/target/sand/dirt) and 2 on
        WATER/ROCK (one PLACE/MINE *plus* the move), so the potential reflects
        the true cost of crossing an obstacle vs walking around it. TREE is
        impassable. Seeded from *every* TARGET cell (the whole goal wall / door
        for natural maps). Used as the PBRS potential
        ``φ = −ctg`` — so a policy maximising shaped return minimises episode
        length, and the cross-vs-detour choice falls out naturally."""
        import heapq
        H, W = terrain.shape
        INF = H * W * 4
        dist = np.full((H, W), INF, dtype=np.int32)
        seeds = list(map(tuple, np.argwhere(terrain == TARGET)))
        if not seeds:
            seeds = [target]
        pq = []
        for (tr, tc) in seeds:
            dist[tr, tc] = 0
            heapq.heappush(pq, (0, tr, tc))
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist[r, c]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                t = terrain[nr, nc]
                if t == TREE:
                    continue
                step = 2 if (t == WATER or t == ROCK) else 1
                nd = d + step
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
        return dist

    # ── observation ──────────────────────────────────────────────────────

    def _make_obs(self) -> dict[str, np.ndarray]:
        assert self._terrain is not None
        crop = self._egocentric_crop()
        face_oh = np.zeros(4, dtype=np.float32)
        face_oh[self._facing] = 1.0
        step_norm = np.float32(self._step_count / max(1, self.max_steps))
        scalars = np.concatenate([face_oh, np.array([step_norm], np.float32)])
        return {"minimap": crop, "scalars": scalars}

    def _egocentric_crop(self) -> np.ndarray:
        assert self._terrain is not None
        V = self.view_size
        half = V // 2
        ar, ac = self._pos
        out = np.full((V, V), OOB, dtype=np.int8)
        r0, c0 = ar - half, ac - half
        rs = max(0, -r0); re = V - max(0, (r0 + V) - self.height)
        cs = max(0, -c0); ce = V - max(0, (c0 + V) - self.width)
        if rs < re and cs < ce:
            out[rs:re, cs:ce] = self._terrain[r0 + rs:r0 + re, c0 + cs:c0 + ce]
        return out

    # ── info / render ────────────────────────────────────────────────────

    def _make_info(self, reward: float, mined: bool, placed: bool,
                   blocked: bool, reached: bool) -> dict[str, Any]:
        assert self._terrain is not None
        cur = int(self._terrain[self._pos[0], self._pos[1]])
        return {
            "position": self._pos,
            "target": self._record.target if self._record else (0, 0),
            "facing": self._facing,
            "step": self._step_count,
            "episode_return": self._episode_return,
            "current_tile": cur,
            "reward": float(reward),
            "mined": mined,
            "placed": placed,
            "blocked": blocked,
            "reached_target": reached,
        }

    def render(self) -> np.ndarray:
        """Return the full map as an (H, W, 3) uint8 RGB image with the agent
        drawn as a small marker."""
        assert self._terrain is not None
        img = TILE_COLORS[self._terrain].copy()
        ar, ac = self._pos
        # darken the agent cell and tint by facing
        face_col = np.array([255, 255, 255], dtype=np.uint8)
        img[ar, ac] = face_col
        return img


__all__ = ["ZebraNavEnv", "NUM_ACTIONS", "A_UP", "A_DOWN", "A_LEFT", "A_RIGHT",
           "A_PLACE", "A_MINE", "F_UP", "F_DOWN", "F_LEFT", "F_RIGHT"]
