"""bridge_tunnel_commit — POMDP navigation with a one-shot build/mine commitment.

Same natural terrain as ``cogniland.bridge_tunnel`` (lakes / mountains / trees;
spawn centre-left, goal a 3-cell door on the right wall), but the agent must
**commit** to a single crossing tool before it can use it, and the commitment is
irreversible for the rest of the episode.

Action space
------------
``Discrete(8)``:
    0 = move UP      (also faces up)
    1 = move DOWN
    2 = move LEFT
    3 = move RIGHT
    4 = BUILD        (water → wood) — **no-op unless committed to build**
    5 = MINE         (rock  → grass) — **no-op unless committed to mine**
    6 = COMMIT_BUILD (first use unlocks BUILD; no-op once any commit is made)
    7 = COMMIT_MINE  (first use unlocks MINE;  no-op once any commit is made)

Commitment is a single slot ``commit ∈ {none, build, mine}``. It starts ``none``
(neither BUILD nor MINE does anything). ``COMMIT_BUILD`` / ``COMMIT_MINE`` set it
the first time they are used and are no-ops thereafter — once set it can never
change, and the *other* tool stays permanently locked. So before committing the
agent can only move + commit; after committing it can move + use the one tool it
locked into. The commitment is exposed in the observation scalars.

Observation
-----------
Egocentric ``view_size × view_size`` tile-id crop (int8, OOB-padded) plus a
``(7,)`` scalar vector ``[facing one-hot (4), step/max, commit==build, commit==mine]``.

Reward
------
``slack_penalty`` per step, ``+reach_bonus`` on reaching the target, ``−build_cost``
per successful BUILD/MINE, plus PBRS shaping ``shaping_coef · (φ_prev − γ·φ_curr)``
with ``φ = −ctg``. The cost-to-go ``ctg`` is **commitment-aware**: before
committing, both water and rock count as crossable (unit-ish cost); once
committed, only the committed obstacle is crossable and the other is treated as
an impassable wall. Three static ctg fields (none / build / mine) are
precomputed at reset and indexed by the current commitment, so committing wrong
on a one-sided (lakes / rocky) map raises the cost-to-go (or makes the goal
unreachable) — the agent is taught to read the terrain and commit correctly.
"""
from __future__ import annotations

import heapq
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .mapgen import MapRecord, generate_commit_map
from .tiles import (
    GRASS, NUM_TILES, OOB, ROCK, TARGET, TILE_COLORS, TREE, WATER, WOOD,
    is_walkable,
)

# action ids
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_BUILD, A_MINE, A_COMMIT_BUILD, A_COMMIT_MINE = range(8)
A_PLACE = A_BUILD                # alias (base env calls the bridge action PLACE)
NUM_ACTIONS = 8

# commitment slot states
COMMIT_NONE, COMMIT_BUILD, COMMIT_MINE = range(3)

# facing ids (same order as the move actions)
F_UP, F_DOWN, F_LEFT, F_RIGHT = range(4)
_FACE_DELTA = {
    F_UP:    (-1, 0),
    F_DOWN:  (+1, 0),
    F_LEFT:  (0, -1),
    F_RIGHT: (0, +1),
}
_MOVE_TO_FACING = {A_UP: F_UP, A_DOWN: F_DOWN, A_LEFT: F_LEFT, A_RIGHT: F_RIGHT}

N_SCALARS = 7   # facing one-hot (4) + step/max (1) + commit_build (1) + commit_mine (1)


class BridgeTunnelCommitEnv(gym.Env):
    """Natural-terrain navigation with an irreversible build/mine commitment.

    A new map is generated on every reset unless ``map_record=`` is supplied
    (fixed-map evaluation). With ``map_record=None`` a random category is drawn
    each reset (uniform over ``categories``)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        size: int = 32,
        view_size: int = 11,
        max_steps: int = 1000,
        orientation: str = "natural",
        width: int | None = 64,
        categories: tuple[str, ...] = ("balanced", "lakes", "rocky"),
        tree_frac: float = 0.03,
        goal_half: int | None = 1,
        seed: int = 0,
        map_record: MapRecord | None = None,
        slack_penalty: float = -0.01,
        reach_bonus: float = 1.0,
        shaping_coef: float = 0.01,
        build_cost: float = 0.05,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        if view_size % 2 == 0 or view_size < 3:
            raise ValueError(f"view_size={view_size} must be odd and >= 3")
        if orientation != "natural":
            raise ValueError(
                f"orientation must be 'natural' (stripes retired), got {orientation!r}")
        self.size = int(size)
        self.height = int(size)
        self.width = int(width) if width is not None else int(size)
        self.view_size = int(view_size)
        self.max_steps = int(max_steps)
        self.orientation = orientation
        self.categories = tuple(categories)
        self.tree_frac = float(tree_frac)
        self.goal_half = goal_half
        self.slack_penalty = float(slack_penalty)
        self.reach_bonus = float(reach_bonus)
        self.shaping_coef = float(shaping_coef)
        self.build_cost = float(build_cost)
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
            "scalars": spaces.Box(low=-1.0, high=1.0, shape=(N_SCALARS,), dtype=np.float32),
        })

        # state — populated by reset()
        self._record: MapRecord | None = None
        self._terrain: np.ndarray | None = None
        self._ctg: np.ndarray | None = None       # (3, H, W) commit-indexed potential
        self._pos: tuple[int, int] = (0, 0)
        self._facing: int = F_RIGHT
        self._commit: int = COMMIT_NONE
        self._step_count: int = 0
        self._episode_return: float = 0.0
        self._traj: list[tuple[int, int]] = []

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
            cat = self.categories[int(self._rng.integers(0, len(self.categories)))]
            self._record = generate_commit_map(
                size=self.height, width=self.width, seed=sub, category=cat,
                tree_frac=self.tree_frac, goal_half=self.goal_half,
            )
        self._terrain = self._record.terrain.copy()
        self._ctg = self._compute_all_ctg(self._terrain, self._record.target)
        self._pos = tuple(self._record.spawn)
        self._facing = F_RIGHT
        self._commit = COMMIT_NONE
        self._step_count = 0
        self._episode_return = 0.0
        self._traj = [self._pos]
        return self._make_obs(), self._make_info(reward=0.0, mined=False,
                                                 placed=False, blocked=False,
                                                 reached=False, committed_now=False)

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._terrain is None:
            raise RuntimeError("step() called before reset()")
        action = int(action)
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"action {action} out of range [0,{NUM_ACTIONS})")

        reward = self.slack_penalty
        mined = placed = blocked = reached = committed_now = False
        commit_prev = self._commit
        ctg_prev = float(self._ctg[commit_prev][self._pos]) if self._ctg is not None else 0.0

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
        elif action == A_BUILD:                       # bridge water → wood (if committed)
            if self._commit == COMMIT_BUILD:
                dr, dc = _FACE_DELTA[self._facing]
                fr, fc = self._pos[0] + dr, self._pos[1] + dc
                if 0 <= fr < self.height and 0 <= fc < self.width \
                        and self._terrain[fr, fc] == WATER:
                    self._terrain[fr, fc] = WOOD
                    placed = True
                    reward -= self.build_cost
        elif action == A_MINE:                        # mine rock → grass (if committed)
            if self._commit == COMMIT_MINE:
                dr, dc = _FACE_DELTA[self._facing]
                fr, fc = self._pos[0] + dr, self._pos[1] + dc
                if 0 <= fr < self.height and 0 <= fc < self.width \
                        and self._terrain[fr, fc] == ROCK:
                    self._terrain[fr, fc] = GRASS
                    mined = True
                    reward -= self.build_cost
        elif action == A_COMMIT_BUILD:                # lock into building (once)
            if self._commit == COMMIT_NONE:
                self._commit = COMMIT_BUILD
                committed_now = True
        elif action == A_COMMIT_MINE:                 # lock into mining (once)
            if self._commit == COMMIT_NONE:
                self._commit = COMMIT_MINE
                committed_now = True

        # PBRS shaping with a commitment-aware potential φ = −ctg[commit].
        # Read φ_prev with the pre-action commitment and φ_curr with the
        # post-action commitment (so the commit step itself is shaped by the
        # change in reachable cost-to-go).
        if self.shaping_coef != 0.0 and self._ctg is not None:
            ctg_curr = float(self._ctg[self._commit][self._pos])
            reward += self.shaping_coef * (ctg_prev - self.gamma * ctg_curr)

        self._step_count += 1
        self._episode_return += reward
        self._traj.append(self._pos)
        terminated = reached
        truncated = (not terminated) and (self._step_count >= self.max_steps)
        info = self._make_info(reward=reward, mined=mined, placed=placed,
                               blocked=blocked, reached=reached,
                               committed_now=committed_now)
        return self._make_obs(), float(reward), terminated, truncated, info

    def _thin_side_accuracy(self) -> tuple[int, int]:
        """Retired stripe metric — kept for caller compatibility."""
        return 0, 0

    # ── cost-to-go potential (commitment-aware) ──────────────────────────

    @staticmethod
    def _compute_ctg(terrain: np.ndarray, target: tuple[int, int],
                     water_cross: bool, rock_cross: bool) -> np.ndarray:
        """Min-action cost-to-go to the goal (Dijkstra) under a given crossing
        capability. Entering walkable land costs 1; entering a *crossable*
        obstacle (water if ``water_cross``, rock if ``rock_cross``) costs 2 (the
        build/mine + the move). TREE — and any obstacle that is **not** crossable
        under the current commitment — is an impassable wall. Seeded from every
        TARGET cell.

        Unlike the base env, cells unreachable under the current commitment are
        common (the non-crossable obstacle walls off whole regions). The raw
        ``INF = H·W·4`` sentinel would blow up the PBRS term ``(1−γ)·ctg`` on
        those cells (e.g. ``0.01·8192 ≈ 82`` per step), dwarfing the reach bonus,
        so the distance field is **capped at ``2·(H+W)``** — comfortably above the
        largest real reachable distance (~130 on a 32×64 map) yet bounded, so the
        shaping stays well-scaled and a wrong commitment raises the cost-to-go
        without producing reward spikes."""
        H, W = terrain.shape
        INF = H * W * 4
        CAP = 2 * (H + W)
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
                if t == WATER:
                    if not water_cross:
                        continue
                    step = 2
                elif t == ROCK:
                    if not rock_cross:
                        continue
                    step = 2
                else:
                    step = 1
                nd = d + step
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
        return np.minimum(dist, CAP)

    @classmethod
    def _compute_all_ctg(cls, terrain: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        """Stack the three commitment-indexed ctg fields: [none, build, mine].

        * none  — both water and rock crossable (pre-commitment potential).
        * build — only water crossable (committed to building; rock = wall).
        * mine  — only rock crossable  (committed to mining;  water = wall).
        """
        none = cls._compute_ctg(terrain, target, water_cross=True, rock_cross=True)
        build = cls._compute_ctg(terrain, target, water_cross=True, rock_cross=False)
        mine = cls._compute_ctg(terrain, target, water_cross=False, rock_cross=True)
        return np.stack([none, build, mine], axis=0).astype(np.float32)

    # ── observation ──────────────────────────────────────────────────────

    def _make_obs(self) -> dict[str, np.ndarray]:
        assert self._terrain is not None
        crop = self._egocentric_crop()
        face_oh = np.zeros(4, dtype=np.float32)
        face_oh[self._facing] = 1.0
        step_norm = np.float32(self._step_count / max(1, self.max_steps))
        commit_build = np.float32(self._commit == COMMIT_BUILD)
        commit_mine = np.float32(self._commit == COMMIT_MINE)
        scalars = np.concatenate([
            face_oh,
            np.array([step_norm, commit_build, commit_mine], np.float32),
        ])
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
                   blocked: bool, reached: bool, committed_now: bool) -> dict[str, Any]:
        assert self._terrain is not None
        cur = int(self._terrain[self._pos[0], self._pos[1]])
        return {
            "position": self._pos,
            "target": self._record.target if self._record else (0, 0),
            "facing": self._facing,
            "commit": self._commit,                      # 0 none / 1 build / 2 mine
            "committed_now": committed_now,
            "category": self._record.category if self._record else "balanced",
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
        assert self._terrain is not None
        img = TILE_COLORS[self._terrain].copy()
        ar, ac = self._pos
        img[ar, ac] = np.array([255, 255, 255], dtype=np.uint8)
        return img


__all__ = ["BridgeTunnelCommitEnv", "NUM_ACTIONS", "N_SCALARS",
           "A_UP", "A_DOWN", "A_LEFT", "A_RIGHT", "A_BUILD", "A_MINE", "A_PLACE",
           "A_COMMIT_BUILD", "A_COMMIT_MINE",
           "COMMIT_NONE", "COMMIT_BUILD", "COMMIT_MINE",
           "F_UP", "F_DOWN", "F_LEFT", "F_RIGHT"]
