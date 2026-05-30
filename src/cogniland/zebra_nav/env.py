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
    GRASS, NUM_TILES, OBSIDIAN, OOB, ROCK, TARGET, TILE_COLORS, TREE, WATER, WOOD,
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

# relative ("turn/forward") action set: 5 actions. Turns are FREE (0 reward) so
# the agent can curve without penalty; only FORWARD / PLACE / MINE cost reward.
R_TURN_LEFT, R_TURN_RIGHT, R_FORWARD, R_PLACE, R_MINE = range(5)
_CW = (F_UP, F_RIGHT, F_DOWN, F_LEFT)            # clockwise facing cycle
_CW_INDEX = {f: i for i, f in enumerate(_CW)}


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
        orientation: str = "diagonal",   # "diagonal" | "vertical" | "natural" | "mixed"
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
        action_mode: str = "absolute",   # "absolute" (4 moves) | "relative" (turn/forward)
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
        if orientation not in ("diagonal", "vertical", "natural", "mixed"):
            raise ValueError(
                f"orientation must be diagonal|vertical|natural|mixed, got {orientation!r}")
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
        if action_mode not in ("absolute", "relative"):
            raise ValueError(f"action_mode must be absolute|relative, got {action_mode!r}")
        self.action_mode = action_mode
        self.n_actions = 6 if action_mode == "absolute" else 5
        self._seed = int(seed)
        self._fixed_record = map_record
        self._rng = np.random.default_rng(self._seed)

        # spaces
        self.action_space = spaces.Discrete(self.n_actions)
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
            if self.orientation == "mixed":
                orient = "vertical" if self._rng.random() < 0.5 else "diagonal"
            else:
                orient = self.orientation
            self._record = generate_zebra_map(
                size=self.height, width=self.width, seed=sub, n_stripes=self.n_stripes,
                thick_half=self.thick_half, thin_half=self.thin_half,
                obsidian_half=self.obsidian_half, window_h=self.window_h,
                orientation=orient, water_frac=self.water_frac, rock_frac=self.rock_frac,
                tree_frac=self.tree_frac, goal_half=self.goal_half,
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
        if self.action_mode == "relative":
            return self._step_relative(int(action))
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

    def _step_relative(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Turn/forward action set (5 actions). TURN_LEFT/RIGHT only rotate the
        facing and cost **nothing** (0 reward, no shaping) — so the agent can
        curve freely; only FORWARD / PLACE / MINE pay the slack (and PLACE/MINE
        the build cost). Turns still advance the step counter (so the episode
        still times out). This is meant to elicit curved paths through obstacles."""
        if not (0 <= action < 5):
            raise ValueError(f"action {action} out of range [0,5)")
        mined = placed = blocked = reached = False
        reward = 0.0

        if action in (R_TURN_LEFT, R_TURN_RIGHT):       # free rotation, 0 reward
            i = _CW_INDEX[self._facing]
            self._facing = _CW[(i + (1 if action == R_TURN_RIGHT else -1)) % 4]
        else:
            reward = self.slack_penalty
            ctg_prev = float(self._ctg[self._pos]) if self._ctg is not None else 0.0
            dr, dc = _FACE_DELTA[self._facing]
            fr, fc = self._pos[0] + dr, self._pos[1] + dc
            in_bounds = 0 <= fr < self.height and 0 <= fc < self.width
            if action == R_FORWARD:
                if in_bounds and is_walkable(int(self._terrain[fr, fc])):
                    self._pos = (fr, fc)
                    if self._terrain[fr, fc] == TARGET:
                        reward += self.reach_bonus
                        reached = True
                else:
                    blocked = True
            elif action == R_PLACE:
                if in_bounds and self._terrain[fr, fc] == WATER:
                    self._terrain[fr, fc] = WOOD
                    placed = True
                    reward -= self.build_cost
            elif action == R_MINE:
                if in_bounds and self._terrain[fr, fc] == ROCK:
                    self._terrain[fr, fc] = GRASS
                    mined = True
                    reward -= self.build_cost
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
        """Per-episode count of stripe crossings that took the *thinner* side.
        Denominator is ``n_stripes`` (missed stripes count as wrong), so the
        metric couples cue-following with actually solving the task.

        Which window the agent used is read off the trajectory: for diagonal
        walls the crossing happens at ``t = r-c = C`` and the side is water if
        ``s = r+c < S_mid`` (else rock); for vertical walls it happens at
        ``c = C`` and the side is water if ``r < R_mid`` (else rock)."""
        rec = self._record
        if rec is None or rec.orientation == "natural" or not rec.stripe_centers:
            return 0, 0          # natural maps have no discrete thin/thick choice
        vertical = rec.orientation == "vertical"
        mid = float(self.height // 2) if vertical else (self.height + self.width - 2) / 2.0
        n_ok = 0
        for k, C in enumerate(rec.stripe_centers):
            # trajectory point closest to this wall's centre line
            best_d, best_v = 1e9, None
            for (r, c) in self._traj:
                d = abs(c - C) if vertical else abs((r - c) - C)
                if d < best_d:
                    best_d, best_v = d, (r if vertical else r + c)
            if best_v is None or best_d > 1:        # never crossed this stripe
                continue
            side = "water" if best_v < mid else "rock"
            if side == rec.stripe_thinner[k]:
                n_ok += 1
        return n_ok, len(rec.stripe_centers)

    # ── cost-to-go potential ─────────────────────────────────────────────

    @staticmethod
    def _compute_ctg(terrain: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        """**Min-action** cost-to-go to the goal (Dijkstra). Entering a cell costs
        1 action on walkable land (grass/wood/target/cue) and 2 on WATER/ROCK
        (one PLACE/MINE *plus* the move), so the potential reflects the true cost
        of crossing an obstacle vs walking around it. OBSIDIAN and TREE are
        impassable. Seeded from *every* TARGET cell (one cell for stripe maps,
        the whole goal wall for natural maps). Used as the PBRS potential
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
                if t == OBSIDIAN or t == TREE:
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
           "A_PLACE", "A_MINE", "F_UP", "F_DOWN", "F_LEFT", "F_RIGHT",
           "R_TURN_LEFT", "R_TURN_RIGHT", "R_FORWARD", "R_PLACE", "R_MINE"]
