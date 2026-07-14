"""bridge_tunnel — one POMDP navigation env with two variants.

``variant="bt"`` (base): cross water by **placing** a wood bridge or rock by
**mining**, or detour; both tools are always available. Obs scalars are
``[facing one-hot(4), step/max]`` (5).

``variant="btc"`` (commit): the agent must **commit** to a single crossing tool.
Commitment is *implicit* — the **first successful** BUILD locks it to building, the
first successful MINE to mining; afterwards the opposite tool is a no-op with a
small penalty. Maps come in three labelled categories (balanced/lakes/rocky). Obs
scalars add the two commitment flags → ``[facing(4), step/max, commit_build,
commit_mine]`` (7), and the PBRS cost-to-go is commitment-aware (3 fields).

Action space is ``Discrete(6)`` in both: 0–3 move, 4 = BUILD/PLACE (water→wood),
5 = MINE (rock→grass). Reaching a TARGET cell wins.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .ctg import commit_ctg_stack, compute_ctg
from .mapgen import MapRecord, generate_map
from .tiles import (
    GRASS, NUM_TILES, OOB, ROCK, TARGET, TILE_COLORS, WATER, WOOD, is_walkable,
)

# action ids
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_BUILD, A_MINE = range(6)
A_PLACE = A_BUILD                 # bt alias (the bridge action)
NUM_ACTIONS = 6

# commitment slot states (btc)
COMMIT_NONE, COMMIT_BUILD, COMMIT_MINE = range(3)

# facing ids (same order as the move actions)
F_UP, F_DOWN, F_LEFT, F_RIGHT = range(4)
_FACE_DELTA = {F_UP: (-1, 0), F_DOWN: (+1, 0), F_LEFT: (0, -1), F_RIGHT: (0, +1)}
_MOVE_TO_FACING = {A_UP: F_UP, A_DOWN: F_DOWN, A_LEFT: F_LEFT, A_RIGHT: F_RIGHT}

VARIANTS = ("bt", "btc")


class BridgeTunnelEnv(gym.Env):
    """Parametric bridge_tunnel env; ``variant`` selects base (bt) or commit (btc)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        variant: str = "bt",
        size: int = 32,
        view_size: int = 11,
        max_steps: int = 1000,
        orientation: str = "natural",
        width: int | None = 64,
        water_frac: float = 0.14,          # bt: obstacle coverage (btc: set by category)
        rock_frac: float = 0.14,
        categories: tuple[str, ...] = ("balanced", "lakes", "rocky"),  # btc
        tree_frac: float = 0.03,
        goal_half: int | None = 1,
        fork_wall: bool = False,           # split-decision gate: wall+passage, then top/bottom doors
        passage_half: int = 1,             # fork_wall: passage is 2*passage_half+1 cells
        wall_margin: int = 1,              # fork_wall: wall is this many cells from the right edge
        seed: int = 0,
        map_record: MapRecord | None = None,
        slack_penalty: float = -0.01,
        reach_bonus: float = 1.0,
        shaping_coef: float = 0.01,
        build_cost: float = 0.05,
        commit_cost: float = 0.05,         # btc: one-time cost on the committing build/mine
        illegal_penalty: float = 0.02,     # btc: using the locked opposite tool
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if view_size % 2 == 0 or view_size < 3:
            raise ValueError(f"view_size={view_size} must be odd and >= 3")
        if orientation != "natural":
            raise ValueError(f"orientation must be 'natural', got {orientation!r}")
        self.variant = variant
        self.commit_enabled = (variant == "btc")
        self.size = self.height = int(size)
        self.width = int(width) if width is not None else int(size)
        self.view_size = int(view_size)
        self.max_steps = int(max_steps)
        self.orientation = orientation
        self.water_frac = float(water_frac)
        self.rock_frac = float(rock_frac)
        self.categories = tuple(categories)
        self.tree_frac = float(tree_frac)
        self.goal_half = goal_half
        self.fork_wall = bool(fork_wall)
        self.passage_half = int(passage_half)
        self.wall_margin = int(wall_margin)
        self.slack_penalty = float(slack_penalty)
        self.reach_bonus = float(reach_bonus)
        self.shaping_coef = float(shaping_coef)
        self.build_cost = float(build_cost)
        self.commit_cost = float(commit_cost)
        self.illegal_penalty = float(illegal_penalty)
        self.gamma = float(gamma)
        self._seed = int(seed)
        self._fixed_record = map_record
        self._rng = np.random.default_rng(self._seed)

        self.n_scalars = 7 if self.commit_enabled else 5
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "minimap": spaces.Box(low=0, high=NUM_TILES - 1,
                                  shape=(self.view_size, self.view_size), dtype=np.int8),
            "scalars": spaces.Box(low=-1.0, high=1.0, shape=(self.n_scalars,), dtype=np.float32),
        })

        self._record: MapRecord | None = None
        self._terrain: np.ndarray | None = None
        self._ctg: np.ndarray | None = None      # bt: (H,W); btc: (3,H,W)
        self._correct_cells: set[tuple[int, int]] | None = None  # fork_wall: the rewarded door(s)
        self._pos: tuple[int, int] = (0, 0)
        self._facing: int = F_RIGHT
        self._commit: int = COMMIT_NONE
        self._step_count: int = 0
        self._episode_return: float = 0.0
        self._traj: list[tuple[int, int]] = []

    # ── cost-to-go (classmethods kept for the JAX map oracle + parity) ────
    @classmethod
    def _compute_ctg(cls, terrain, target, seeds=None):
        """bt potential: both obstacles crossable, uncapped (matches the JAX oracle)."""
        return compute_ctg(terrain, target, water_cross=True, rock_cross=True, cap=None, seeds=seeds)

    @classmethod
    def _compute_all_ctg(cls, terrain, target, seeds=None):
        """btc potential: (3,H,W) commit-indexed [none,build,mine], capped at 2(H+W)."""
        return commit_ctg_stack(terrain, target, seeds=seeds)

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
        if self._fixed_record is not None:
            self._record = self._fixed_record
        else:
            sub = int(self._rng.integers(0, 2 ** 31))
            cat = (self.categories[int(self._rng.integers(0, len(self.categories)))]
                   if self.commit_enabled else None)
            self._record = generate_map(
                variant=self.variant, seed=sub, size=self.height, width=self.width,
                category=cat, water_frac=self.water_frac, rock_frac=self.rock_frac,
                tree_frac=self.tree_frac, goal_half=self.goal_half,
                fork_wall=self.fork_wall, passage_half=self.passage_half,
                wall_margin=self.wall_margin)
        self._terrain = self._record.terrain.copy()
        tgt = self._record.target
        rec = self._record
        if rec.correct_target == "top":
            self._correct_cells = set(rec.top_goal_cells)
        elif rec.correct_target == "bottom":
            self._correct_cells = set(rec.bottom_goal_cells)
        elif rec.correct_target == "either":
            self._correct_cells = set(rec.top_goal_cells) | set(rec.bottom_goal_cells)
        else:
            self._correct_cells = None
        seeds = list(self._correct_cells) if self._correct_cells is not None else None
        self._ctg = self._compute_all_ctg(self._terrain, tgt, seeds=seeds) if self.commit_enabled \
            else self._compute_ctg(self._terrain, tgt, seeds=seeds)
        self._pos = tuple(self._record.spawn)
        self._facing = F_RIGHT
        self._commit = COMMIT_NONE
        self._step_count = 0
        self._episode_return = 0.0
        self._traj = [self._pos]
        return self._make_obs(), self._make_info(0.0, False, False, False, False, False)

    def step(self, action: int):
        if self._terrain is None:
            raise RuntimeError("step() called before reset()")
        action = int(action)
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"action {action} out of range [0,{NUM_ACTIONS})")

        reward = self.slack_penalty
        mined = placed = blocked = reached_any = success = committed_now = False
        commit_prev = self._commit
        ctg_prev = self._ctg_at(commit_prev, self._pos)

        if action in _MOVE_TO_FACING:
            self._facing = _MOVE_TO_FACING[action]
            dr, dc = _FACE_DELTA[self._facing]
            nr, nc = self._pos[0] + dr, self._pos[1] + dc
            if 0 <= nr < self.height and 0 <= nc < self.width \
                    and is_walkable(int(self._terrain[nr, nc])):
                self._pos = (nr, nc)
                if self._terrain[nr, nc] == TARGET:
                    reached_any = True
                    # fork_wall maps: only the door matching the map's belief
                    # (category) pays the reach bonus / counts as success; the
                    # decoy door still ends the episode, just with no bonus.
                    success = (self._correct_cells is None
                              or (nr, nc) in self._correct_cells)
                    if success:
                        reward += self.reach_bonus
            else:
                blocked = True
        elif action == A_BUILD:
            placed, reward, committed_now = self._do_tool(
                tile=WATER, become=WOOD, slot=COMMIT_BUILD, locked=COMMIT_MINE, reward=reward)
        elif action == A_MINE:
            mined, reward, committed_now = self._do_tool(
                tile=ROCK, become=GRASS, slot=COMMIT_MINE, locked=COMMIT_BUILD, reward=reward)

        if self.shaping_coef != 0.0 and self._ctg is not None:
            ctg_curr = self._ctg_at(self._commit, self._pos)
            reward += self.shaping_coef * (ctg_prev - self.gamma * ctg_curr)

        self._step_count += 1
        self._episode_return += reward
        self._traj.append(self._pos)
        terminated = reached_any
        truncated = (not terminated) and (self._step_count >= self.max_steps)
        info = self._make_info(reward, mined, placed, blocked, success, committed_now)
        info["reached_any_target"] = reached_any
        if terminated or truncated:
            info["thin_correct"], info["thin_total"] = 0, 0   # retired metric (bt compat)
        return self._make_obs(), float(reward), terminated, truncated, info

    def _do_tool(self, *, tile, become, slot, locked, reward):
        """Shared BUILD/MINE handler. bt: always act. btc: act unless locked to the
        opposite tool; the first successful act commits (pays commit_cost)."""
        did = committed = False
        if self.commit_enabled and self._commit == locked:
            return did, reward - self.illegal_penalty, committed       # prohibited
        dr, dc = _FACE_DELTA[self._facing]
        fr, fc = self._pos[0] + dr, self._pos[1] + dc
        if 0 <= fr < self.height and 0 <= fc < self.width and self._terrain[fr, fc] == tile:
            self._terrain[fr, fc] = become
            did = True
            reward -= self.build_cost
            if self.commit_enabled and self._commit == COMMIT_NONE:
                self._commit = slot
                committed = True
                reward -= self.commit_cost
        return did, reward, committed

    def _ctg_at(self, commit, pos):
        if self._ctg is None:
            return 0.0
        return float(self._ctg[commit][pos] if self.commit_enabled else self._ctg[pos])

    def _thin_side_accuracy(self):
        return 0, 0

    # ── observation / info / render ───────────────────────────────────────
    def _make_obs(self):
        crop = self._egocentric_crop()
        face_oh = np.zeros(4, dtype=np.float32)
        face_oh[self._facing] = 1.0
        step_norm = np.float32(self._step_count / max(1, self.max_steps))
        if self.commit_enabled:
            scalars = np.concatenate([face_oh, np.array(
                [step_norm, np.float32(self._commit == COMMIT_BUILD),
                 np.float32(self._commit == COMMIT_MINE)], np.float32)])
        else:
            scalars = np.concatenate([face_oh, np.array([step_norm], np.float32)])
        return {"minimap": crop, "scalars": scalars}

    def _egocentric_crop(self):
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

    def _make_info(self, reward, mined, placed, blocked, success, committed_now):
        cur = int(self._terrain[self._pos[0], self._pos[1]])
        return {
            "position": self._pos,
            "target": self._record.target if self._record else (0, 0),
            "facing": self._facing,
            "commit": self._commit,                       # 0 none / 1 build / 2 mine (bt: always 0)
            "committed_now": committed_now,
            "category": self._record.category if self._record else None,
            "step": self._step_count,
            "episode_return": self._episode_return,
            "current_tile": cur,
            "reward": float(reward),
            "mined": mined, "placed": placed, "blocked": blocked,
            # fork_wall: True only if the door reached matches the map's belief
            # (category); non-fork maps: True whenever any target is touched.
            "reached_target": success,
        }

    def render(self):
        img = TILE_COLORS[self._terrain].copy()
        ar, ac = self._pos
        img[ar, ac] = np.array([255, 255, 255], dtype=np.uint8)
        return img


class BridgeTunnelCommitEnv(BridgeTunnelEnv):
    """Back-compat thin subclass defaulting to ``variant='btc'``."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("variant", "btc")
        super().__init__(*args, **kwargs)


__all__ = ["BridgeTunnelEnv", "BridgeTunnelCommitEnv", "NUM_ACTIONS", "VARIANTS",
           "A_UP", "A_DOWN", "A_LEFT", "A_RIGHT", "A_BUILD", "A_MINE", "A_PLACE",
           "COMMIT_NONE", "COMMIT_BUILD", "COMMIT_MINE",
           "F_UP", "F_DOWN", "F_LEFT", "F_RIGHT"]
