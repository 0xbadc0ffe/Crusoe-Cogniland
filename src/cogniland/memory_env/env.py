"""MemoryEnv — a Farama-MiniGrid memory task for DreamerV3 latent analysis.

MemoryEnv is a substrate for **analyzing how DreamerV3 integrates and maintains
task information in its latents over time**. It is built directly on the Farama
**MiniGrid** library (``minigrid.minigrid_env.MiniGridEnv`` — the triangle
agent, turn/forward grid physics and partial-observability visibility model),
patterned after MiniGrid's own ``MemoryEnv`` but with two additions:

* a **custom oriented Key cue** (a normal upright key vs. a vertically flipped
  "upside-down" key), and
* an **extra mid-corridor branch** the agent must commit to *before* the final
  doors are visible.

The episode is a strict temporal pipeline::

    nothing -> see cue -> remember -> choose corridor branch in advance ->
    maintain memory -> choose visible door.

A single coloured, oriented **key cue** carries two independent task variables:

* **shape** = key orientation: an upright key = ``down`` cue, an upside-down
  key = ``up`` cue. Shape selects the **branch** taken at the mid fork
  (up -> upper branch, down -> lower branch) *before the doors are visible*;
* **colour** = key colour (green / blue). Colour selects the **final door**
  (green cue -> green door, blue cue -> blue door).

Crucially the two pieces of information are needed at **different times**:

* **shape** must be recalled BEFORE the branch (the fork is reached while the
  cue is long out of view, and the doors are not yet visible);
* **colour** must be recalled LATER, at the doors, after passing through a
  branch and a visually identical reconnecting corridor.

This temporal/role separation is the whole point. When probing or steering a
DreamerV3 agent trained here, steering a *branch* behaviour should ideally move
only the shape/branch belief. If steering the branch ALSO flips the final
colour choice, that is evidence the latent **entangles** the two task variables
(or hallucinates task context) rather than keeping them factorised.

The two task variables can be sampled **factorised** (all four cues uniformly,
shape ⟂ colour) or **entangled** (only ``green_up`` / ``blue_down``, so shape and
colour are perfectly correlated and a single direction can predict both).

Observations are egocentric **RGB pixels** (``uint8 (H, W, 3)``), produced by
``minigrid.wrappers.RGBImgPartialObsWrapper`` and therefore DreamerV3
compatible. Partial observability (``see_through_walls=False``) is what makes
this a *memory* task: the cue is only visible while the agent is in the start
room and the doors only near the end. **No privileged labels appear in the
observation** — phase, cue identity, branch, etc. live only in the ``info``
dict so they can be used to align latents to phases offline.

MiniGrid's symbolic tile encoding cannot represent key *orientation*, which is
exactly why the observation must be RGB pixels: the up/down distinction only
exists in the rendered cue.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

import gymnasium as gym
from minigrid.core.constants import COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Key, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_circle, point_in_rect
from minigrid.wrappers import RGBImgPartialObsWrapper

# --------------------------------------------------------------------------- #
# Actions — native MiniGrid action space (Discrete(7)).
#   0 = turn left, 1 = turn right, 2 = forward, 3 = pickup, 4 = drop,
#   5 = toggle, 6 = done.  We navigate purely with {left, right, forward}.
# --------------------------------------------------------------------------- #
A_LEFT_TURN, A_RIGHT_TURN, A_FORWARD = 0, 1, 2
NUM_ACTIONS = 7

# Direction ids in MiniGrid: 0=east(+x) 1=south(+y) 2=west(-x) 3=north(-y).
DIR_EAST, DIR_SOUTH, DIR_WEST, DIR_NORTH = 0, 1, 2, 3

# Phase names, in temporal order.
PHASES = (
    "blank",
    "cue",
    "pre_branch_memory",
    "branch_choice",
    "post_branch_memory",
    "door_visible",
    "terminal",
)

# Cue catalogue.
CUE_TYPES = ("green_up", "blue_up", "green_down", "blue_down")
_CUE_SHAPE = {"green_up": "up", "blue_up": "up", "green_down": "down", "blue_down": "down"}
_CUE_COLOR = {"green_up": "green", "blue_up": "blue", "green_down": "green", "blue_down": "blue"}

# Exact RGB triples MiniGrid uses for red/blue (so tests can match cue/door
# pixels exactly). These come from minigrid.core.constants.COLORS.
_COL_GREEN = np.array(COLORS["green"], np.uint8)
_COL_BLUE = np.array(COLORS["blue"], np.uint8)


# --------------------------------------------------------------------------- #
# Custom world object: an orientable Key cue.
# --------------------------------------------------------------------------- #
class OrientedKey(Key):
    """A :class:`minigrid.core.world_object.Key` that can render upside-down.

    ``orientation="down"`` renders the stock MiniGrid key (ring at top, teeth
    at bottom — a normal upright key). ``orientation="up"`` renders the key
    flipped vertically (ring at bottom), which reads as an "upside-down" key.

    The colour is the standard MiniGrid key colour. Orientation is *not*
    representable in MiniGrid's symbolic tile encoding, so it only shows up in
    the RGB render — which is exactly why the observation must be pixels.
    """

    def __init__(self, color: str = "blue", orientation: str = "down"):
        super().__init__(color)
        assert orientation in ("up", "down")
        self.orientation = orientation

    def can_pickup(self):
        # The cue is a *visual* signal, not a collectible: it must stay on the
        # grid for the whole episode. If it were pickupable, the MiniGrid
        # `pickup` action would empty its cell and `agent_sees(cue_pos)` would
        # assert (world_cell is None).
        return False

    def encode(self):  # keep symbolic encoding identical to a plain Key
        return super().encode()

    def render(self, img):
        c = COLORS[self.color]

        if self.orientation == "down":
            # Stock MiniGrid Key.render (upright key).
            fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)
            fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
            fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)
            fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
            fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))
        else:
            # Same key with every y replaced by (1 - y): a vertical mirror, so
            # the ring sits at the bottom (an upside-down key).
            fill_coords(img, point_in_rect(0.50, 0.63, 1 - 0.88, 1 - 0.31), c)
            fill_coords(img, point_in_rect(0.38, 0.50, 1 - 0.66, 1 - 0.59), c)
            fill_coords(img, point_in_rect(0.38, 0.50, 1 - 0.88, 1 - 0.81), c)
            fill_coords(img, point_in_circle(cx=0.56, cy=1 - 0.28, r=0.190), c)
            fill_coords(img, point_in_circle(cx=0.56, cy=1 - 0.28, r=0.064), (0, 0, 0))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class MemoryEnvConfig:
    """Configuration + defaults for :class:`MemoryEnv`.

    The length parameters drive the MiniGrid grid geometry (see
    ``_gen_grid``): the start room, the pre-branch hallway, the mid-corridor
    branch, the reconnect hallway and the final door split. Defaults give clean
    temporal separation of the seven phases for latent alignment while keeping
    the task learnable.
    """

    pre_cue_steps: int = 1          # initial blank corridor before the cue (was 3)
    cue_visible_steps: int = 2      # kept for schedule/back-compat (informational)
    pre_branch_corridor_len: int = 5   # cue room -> fork hallway (was 8)
    branch_len: int = 4
    post_branch_corridor_len: int = 5  # fork -> door hallway (was 8; eased)
    door_visible_distance: int = 3
    max_steps: int = 200            # MiniGrid turn/forward needs more steps than 1-D
    view_size: int = 5             # egocentric crop (cells); MUST be odd (==agent_view_size); was 7
    center_wall_thickness: int = 3  # walled rows between the up/down branches (was 1); must be odd
    cell_px: int = 8               # pixels per cell -> obs is (view*cell, view*cell, 3)
    cue_distribution: str = "factorized"   # "factorized" | "entangled" | "custom"
    custom_cues: Sequence[str] | None = None       # for cue_distribution="custom"
    custom_weights: Sequence[float] | None = None
    forced_branch: str | None = None       # None | "up" | "down"
    suppress_down_action: bool = False
    suppress_up_action: bool = False
    # rewards: +0.5 for taking the shape-correct branch, +0.5 for the
    # colour-correct door (so a fully-correct episode totals +1.0; the branch
    # and door rewards are split so shape and colour are BOTH actively used).
    success_reward: float = 0.5    # colour-correct final door
    wrong_door_reward: float = 0.0
    step_penalty: float = 0.0
    branch_bonus: float = 0.5      # shape-correct corridor branch
    # Make the SHAPE→branch decision consequential (else the branches reconnect
    # to the same doors and the branch is a reward-optional side-bonus the agent
    # learns to ignore). wrong_branch_penalty is added on entering the wrong
    # branch; wrong_branch_terminates ends the episode there (classic T-maze:
    # wrong arm = failure), forcing the agent to use shape memory.
    wrong_branch_penalty: float = 0.0
    wrong_branch_terminates: bool = False
    # Conjunctive reward (no termination): award the colour-door reward ONLY if
    # the shape-correct branch was also taken. Then the reward-maximising policy
    # MUST use both shape (branch) and colour (door) — there is no max-reward
    # policy that uses only one bit. Episodes still run full length.
    success_requires_branch: bool = False
    # Dense progress shaping: PBRS on a potential = horizontal column reached
    # (capped at the door column). Reward += shaping_coef * (phi_t - phi_{t-1}),
    # which is farming-proof (net-zero for oscillation) and leaks NO task info
    # (it is shape/colour/branch-agnostic — it only rewards getting to the
    # doors). At 0.01 over the (now ~23-col) corridor the full-traversal shaping
    # totals ~0.23 — a guide that stays well under the +1.0 task reward.
    shaping_coef: float = 0.01

    def __post_init__(self) -> None:
        if self.view_size % 2 == 0 or self.view_size < 3:
            raise ValueError(f"view_size={self.view_size} must be odd and >= 3")
        if self.cue_distribution not in ("factorized", "entangled", "custom"):
            raise ValueError(f"bad cue_distribution={self.cue_distribution!r}")
        if self.forced_branch not in (None, "up", "down"):
            raise ValueError(f"bad forced_branch={self.forced_branch!r}")
        if self.branch_len < 2:
            raise ValueError("branch_len must be >= 2")
        if self.center_wall_thickness < 1 or self.center_wall_thickness % 2 == 0:
            raise ValueError(
                f"center_wall_thickness={self.center_wall_thickness} must be odd and >= 1")


# --------------------------------------------------------------------------- #
# The MiniGrid environment
# --------------------------------------------------------------------------- #
class _MemoryMiniGridEnv(MiniGridEnv):
    """Bare MiniGrid env: the triangle agent on the branch corridor.

    This class implements the grid, the partial-obs visibility and the success
    / failure terminations. The public :class:`MemoryEnv` wraps it with
    ``RGBImgPartialObsWrapper`` and the cogniland contract (info dict, phases,
    helpers). Most users want :class:`MemoryEnv`.
    """

    def __init__(self, cfg: MemoryEnvConfig, render_mode: str | None = None):
        self.cfg = cfg
        # --- geometry (column layout along the middle row) ---------------- #
        # x=0 left wall. Start room occupies cols 1..room_w. The cue sits at
        # the back of the start room (col 1). Then a pre-branch hallway, the
        # branch zone, the reconnect hallway, then the final vertical door
        # split.
        self._room_w = 4
        self._pre_len = max(cfg.pre_branch_corridor_len, 2)
        self._branch_len = cfg.branch_len
        self._post_len = max(cfg.post_branch_corridor_len, 2)
        # Initial empty corridor: the agent spawns here with NO cue in sight and
        # walks a few steps (the "blank" baseline phase) before the cue comes
        # into view / it reaches the start room. Sized so the cue is out of the
        # partial view at spawn (distance occlusion) for ~pre_cue_steps steps.
        self._precue_len = max(cfg.pre_cue_steps, 1) + (cfg.view_size - 1)

        # column anchors (computed left -> right)
        self._x_precue_start = 1
        self._x_precue_end = self._x_precue_start + self._precue_len - 1
        self._x_room_start = self._x_precue_end + 1
        self._x_room_end = self._x_room_start + self._room_w - 1     # last room col
        self._x_pre_start = self._x_room_end + 1
        self._x_pre_end = self._x_pre_start + self._pre_len - 1
        self._x_branch_start = self._x_pre_end + 1
        self._x_branch_end = self._x_branch_start + self._branch_len - 1
        self._x_post_start = self._x_branch_end + 1
        self._x_post_end = self._x_post_start + self._post_len - 1
        self._x_doorcol = self._x_post_end + 1     # vertical door split column
        # +1 door column, +1 closing wall column, +1 right border wall
        width = self._x_doorcol + 3

        # Vertical layout. The branch zone's central wall is
        # `center_wall_thickness` rows thick, so the up/down BRANCH corridors sit
        # `bgap = (t+1)//2` rows off the middle row. The start room stays a
        # compact 3-row box (my∓1). The doors sit just outside the branches but
        # NEVER farther from the middle than the agent can see — otherwise a
        # small view could never read the door colours on approach — so the door
        # offset is capped at the view's vertical half-reach. The cue stays
        # visible from the middle corridor and the doors stay occluded until the
        # door column enters the POV, independent of the wall thickness.
        #   t=1,view=7 -> bgap=1, doors at my∓2, height=7 (original)
        #   t=3,view=5 -> bgap=2, doors at my∓2, height=7
        t = self.cfg.center_wall_thickness
        self._bgap = (t + 1) // 2                        # branch rows = my ∓ bgap
        door_off = min(self._bgap + 1, (self.cfg.view_size - 1) // 2)
        outer = max(self._bgap, door_off)               # outermost feature offset
        self._my = outer + 1                            # middle (through) row
        self._row_up = self._my - self._bgap            # upper BRANCH row
        self._row_lo = self._my + self._bgap            # lower BRANCH row
        self._row_room_up = self._my - 1                # start-room upper row (cue)
        self._row_room_lo = self._my + 1                # start-room lower row (cue)
        self._row_door_top = self._my - door_off
        self._row_door_bot = self._my + door_off
        height = self._my + outer + 2                   # +bottom border wall

        mission_space = MissionSpace(mission_func=lambda: "go to the matching door")
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            see_through_walls=False,
            max_steps=cfg.max_steps,
            agent_view_size=cfg.view_size,
            render_mode=render_mode,
        )

        # episode state (set in _gen_grid)
        self.cue_type = "green_up"
        self.door_pos_green = "top"       # "top" | "bottom"
        self.door_pos_blue = "bottom"
        self.taken_branch: str | None = None
        self.selected_door: str | None = None
        self._pending_branch_intent: str | None = None

    # -- cue sampling (uses the gym np_random seeded by reset) ------------- #
    def _sample_cue(self) -> str:
        c = self.cfg
        if c.cue_distribution == "factorized":
            pool, weights = list(CUE_TYPES), None
        elif c.cue_distribution == "entangled":
            pool, weights = ["green_up", "blue_down"], None
        else:
            if not c.custom_cues:
                raise ValueError("cue_distribution='custom' requires custom_cues")
            pool = list(c.custom_cues)
            for ct in pool:
                if ct not in CUE_TYPES:
                    raise ValueError(f"unknown cue {ct!r}")
            weights = None
            if c.custom_weights is not None:
                w = np.asarray(c.custom_weights, float)
                weights = w / w.sum()
        if weights is None:
            idx = int(self.np_random.integers(0, len(pool)))
        else:
            idx = int(self.np_random.choice(len(pool), p=weights))
        return pool[idx]

    @property
    def cue_shape(self) -> str:
        return _CUE_SHAPE[self.cue_type]

    @property
    def cue_color(self) -> str:
        return _CUE_COLOR[self.cue_type]

    @property
    def correct_branch(self) -> str:
        return self.cue_shape

    @property
    def target_door_color(self) -> str:
        return self.cue_color

    # -- grid generation --------------------------------------------------- #
    def _gen_grid(self, width, height):
        c = self.cfg
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        my, ru, rl = self._my, self._row_up, self._row_lo

        # Reset episode state.
        self.cue_type = self._sample_cue()
        self.taken_branch = None
        self.selected_door = None
        self._pending_branch_intent = None
        self._forced_branch = c.forced_branch
        # door positions randomized (top/bottom for the vertical split)
        if self.np_random.integers(0, 2) == 0:
            self.door_pos_green, self.door_pos_blue = "top", "bottom"
        else:
            self.door_pos_green, self.door_pos_blue = "bottom", "top"

        rdt, rdb = self._row_door_top, self._row_door_bot

        # Fill the whole interior with walls; we then carve the corridor.
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                self.grid.set(x, y, Wall())

        def carve(x, y):
            self.grid.set(x, y, None)

        # ---- Start room: a 3-row open box (rows my∓1) over cols
        # [x_room_start .. x_room_end]. The cue key sits on a non-middle room row
        # so it falls in the agent's forward POV during the cue phase. The room
        # rows are my∓1 regardless of the branch central-wall thickness.
        rru, rrl = self._row_room_up, self._row_room_lo
        for x in range(self._x_room_start, self._x_room_end + 1):
            for y in (rru, my, rrl):
                carve(x, y)
        # Wall the room's right edge except the middle-row doorway, so the cue
        # is occluded once the agent steps into the hallway.
        rx = self._x_room_end
        self.grid.set(rx, rru, Wall())
        self.grid.set(rx, rrl, Wall())
        carve(rx, my)

        # Cue position: random over the non-corridor cells of the 3x3 start
        # room — the 3 open columns [room_start .. room_end-1] (room_end's
        # upper/lower cells are the occluding wall) x the 2 non-middle rows
        # {my-1, my+1}. The middle row (my) is the through-corridor and excluded.
        cue_col = int(self.np_random.integers(self._x_room_start, self._x_room_end))
        cue_row = rru if self.np_random.integers(0, 2) == 0 else rrl
        self._cue_pos = (cue_col, cue_row)
        self.grid.set(
            *self._cue_pos,
            OrientedKey(self.cue_color, orientation=self.cue_shape),
        )

        # ---- Initial empty corridor (blank-baseline phase): a middle-row
        # corridor west of the start room. The agent spawns at its far-west end
        # with the cue out of view, then walks east into the room.
        for x in range(self._x_precue_start, self._x_room_start):
            carve(x, my)

        # ---- Pre-branch hallway: middle-row corridor only.
        for x in range(self._x_pre_start, self._x_pre_end + 1):
            carve(x, my)

        # ---- Branch zone: the central rows are walled (a `center_wall_thickness`
        # -thick wall), the corridor splits into an upper (ru) and a lower (rl)
        # path that both run the branch length and reconnect. The agent must
        # commit to one before the doors show.
        bs, be = self._x_branch_start, self._x_branch_end
        for x in range(bs, be + 1):
            carve(x, ru)
            carve(x, rl)
            # the rows between ru and rl stay Wall -> the thick central wall
        # Junction: carve a vertical slot at the last pre-branch column so the
        # agent can turn from the middle corridor up to ru or down to rl. The
        # slot spans ru..rl, so it works for any central-wall thickness.
        jx = self._x_pre_end
        for y in range(ru, rl + 1):
            carve(jx, y)

        # ---- Reconnect + post hallway: carve the reconnect slot (ru..rl) so the
        # branches funnel back to the middle row, then a middle-row corridor to
        # the door split.
        rxp = self._x_post_start
        for y in range(ru, rl + 1):
            carve(rxp, y)
        for x in range(self._x_post_start, self._x_post_end + 1):
            carve(x, my)

        # ---- Final door split (MiniGrid MemoryEnv style): a vertical corridor
        # at x_doorcol whose ends hold the two coloured doors, set 2 rows off
        # the middle (rows rdt / rdb). The column x_doorcol-1 is walled except
        # the middle doorway and x_doorcol+1 is fully walled, so the doors are
        # OCCLUDED until the agent reaches (x_doorcol, my).
        dx = self._x_doorcol
        for y in range(rdt, rdb + 1):     # open the vertical door corridor
            carve(dx, y)
        # closing wall column to the right of the door corridor
        for y in range(1, height - 1):
            self.grid.set(dx + 1, y, Wall())
        # the approach: only the middle doorway connects the post hallway to the
        # door corridor (x_doorcol-1 is already wall on rows != my; carve mid).
        carve(self._x_post_end, my)

        top_color = "green" if self.door_pos_green == "top" else "blue"
        bot_color = "green" if self.door_pos_green == "bottom" else "blue"
        self.grid.set(dx, rdt, _ColoredDoor(top_color))
        self.grid.set(dx, rdb, _ColoredDoor(bot_color))
        self._door_top_pos = (dx, rdt)
        self._door_bot_pos = (dx, rdb)
        self._door_color_at = {(dx, rdt): top_color, (dx, rdb): bot_color}

        # ---- Agent: start at the back of the room on the middle row, facing
        # east so the cue key (one cell ahead, on the upper room row) is in its
        # point of view during the cue phase.
        self.agent_pos = (self._x_precue_start, my)
        self.agent_dir = DIR_EAST
        self._prev_phi = self._progress_phi()

        self.mission = "go to the matching door"

    # -- phase derivation from agent x-region ------------------------------ #
    @property
    def _door_visible_x(self) -> int:
        """First column from which the final doors become visible.

        The doors sit at the top/bottom of the vertical door corridor; their
        cells flood into view once the door column's middle cell enters the
        forward POV, i.e. roughly ``view_size - 2`` cells before the door
        column. ``door_visible_distance`` clamps how early the phase may begin.
        """
        return self._x_doorcol - (self.cfg.view_size - 1)

    def current_phase(self) -> str:
        if getattr(self, "_episode_done", False):
            return "terminal"
        ax = int(self.agent_pos[0])
        if ax >= self._door_visible_x:
            return "door_visible"
        if ax >= self._x_post_start:
            return "post_branch_memory"
        # branch junction (last pre-branch column) + branch zone
        if ax >= self._x_pre_end:
            return "branch_choice"
        # hallway past the start room -> cue occluded by the room's right wall
        if ax >= self._x_pre_start:
            return "pre_branch_memory"
        # precue corridor + start room: cue phase only while the cue is in view
        if self._sees_cue():
            return "cue"
        # cue not visible: blank in the approach corridor, pre_branch once the
        # agent is inside/past the room (cue occluded behind it)
        if ax >= self._x_room_start:
            return "pre_branch_memory"
        return "blank"

    def _progress_phi(self) -> float:
        """PBRS potential: horizontal column reached, capped at the door column
        (so progress saturates once the agent is at the doors). Branch/colour-
        agnostic — purely 'how far toward the doors'."""
        return float(min(int(self.agent_pos[0]), self._x_doorcol))

    def _sees_cue(self) -> bool:
        """Whether the cue cell is within the agent's POV, robust to the cell
        being empty (MiniGrid's ``agent_sees`` asserts ``world_cell is not
        None``; the cue is non-pickupable so it shouldn't empty, but stay safe)."""
        if self.grid.get(*self._cue_pos) is None:
            return False
        return self.agent_sees(*self._cue_pos)

    # -- branch interventions --------------------------------------------- #
    def _branch_dir(self, branch: str) -> int:
        """Facing direction that enters the given branch row from the middle."""
        return DIR_NORTH if branch == "up" else DIR_SOUTH

    def _intervened_branch(self, action: int) -> str | None:
        """Return the branch the interventions force the agent into, if any.

        Called only at the junction cell ``(x_pre_end, my)``. ``forced_branch``
        always wins (auto-route). Otherwise ``suppress_up/down`` redirects an
        attempt to enter the suppressed branch into the allowed one.
        """
        c = self.cfg
        if self._forced_branch is not None:
            return self._forced_branch

        # What branch is the agent trying to enter this step? Use the facing
        # direction (north -> up, south -> down) rather than the next-cell row,
        # so it is robust to a thick central wall (the branch row may be several
        # cells from the middle corridor).
        attempt = None
        if action == A_FORWARD:
            dy = int(self.dir_vec[1])
            if dy < 0:
                attempt = "up"
            elif dy > 0:
                attempt = "down"
        # (turn actions alone don't move into a row; nothing to redirect yet)

        if attempt == "up" and c.suppress_up_action:
            return "down" if not c.suppress_down_action else None
        if attempt == "down" and c.suppress_down_action:
            return "up" if not c.suppress_up_action else None
        return None

    # -- step override: branch tracking, suppression, forced branch, doors -- #
    def step(self, action):
        c = self.cfg
        action = int(action)
        my = self._my

        at_junction = (
            int(self.agent_pos[0]) == self._x_pre_end
            and int(self.agent_pos[1]) == my
            and self.taken_branch is None
        )

        # --- branch interventions: auto-route at the junction. ----------- #
        forced_to: str | None = None
        if at_junction:
            forced_to = self._intervened_branch(action)
            if forced_to is not None:
                # Teleport the agent into the first cell of the forced branch
                # row, facing east, and mark the branch taken. This makes the
                # intervention robust to any action the policy chooses.
                row = self._row_up if forced_to == "up" else self._row_lo
                self.agent_pos = (self._x_branch_start, row)
                self.agent_dir = DIR_EAST
                self.taken_branch = forced_to
                self.step_count += 1
                reward = float(c.step_penalty)
                if forced_to == self.correct_branch:
                    reward += c.branch_bonus
                phi = self._progress_phi()
                reward += c.shaping_coef * (phi - self._prev_phi)
                self._prev_phi = phi
                truncated = self.step_count >= self.max_steps
                self._episode_done = bool(truncated)
                obs = self.gen_obs()
                return obs, reward, False, truncated, {}

            # suppress without a redirect target (both suppressed, or a turn):
            # block any forward that would step into a suppressed row.
            if action == A_FORWARD:
                fy = int((self.agent_pos + self.dir_vec)[1])
                if (c.suppress_up_action and fy == self._row_up) or (
                    c.suppress_down_action and fy == self._row_lo
                ):
                    action = 6  # 'done' == no-op

        obs, reward, terminated, truncated, info = super().step(action)
        reward = float(c.step_penalty)

        ax = int(self.agent_pos[0])
        ay = int(self.agent_pos[1])

        # --- record which branch row the agent entered. ------------------ #
        if self.taken_branch is None and self._x_branch_start <= ax <= self._x_branch_end:
            if ay == self._row_up:
                self.taken_branch = "up"
            elif ay == self._row_lo:
                self.taken_branch = "down"
            if self.taken_branch is not None:
                if self.taken_branch == self.correct_branch:
                    reward += c.branch_bonus
                else:
                    # shape-wrong branch: penalise and (optionally) end the episode
                    reward += c.wrong_branch_penalty
                    if c.wrong_branch_terminates:
                        terminated = True

        # --- door termination: stepping onto a coloured door cell ends it.
        cell = (ax, ay)
        if cell in self._door_color_at:
            self.selected_door = self._door_color_at[cell]
            terminated = True
            door_ok = self.selected_door == self.target_door_color
            # conjunctive: the door reward requires the shape-correct branch too,
            # so maximising reward forces BOTH shape and colour memory.
            branch_ok = (not c.success_requires_branch) or (
                self.taken_branch == self.correct_branch)
            reward += c.success_reward if (door_ok and branch_ok) else c.wrong_door_reward

        # dense PBRS progress shaping (farming-proof, task-info-agnostic)
        phi = self._progress_phi()
        reward += c.shaping_coef * (phi - self._prev_phi)
        self._prev_phi = phi

        self._episode_done = bool(terminated or truncated)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self._episode_done = False
        return super().reset(seed=seed, options=options)


class _ColoredDoor(WorldObj):
    """A walkable coloured target cell (a "door") at the final split.

    Implemented as an overlappable solid-colour cell (like Goal) so the agent
    *steps onto* it to select it, matching MiniGrid MemoryEnv's success/failure
    semantics. Rendered as a solid block of its colour so red vs blue is
    unambiguous in pixels.
    """

    def __init__(self, color: str = "green"):
        super().__init__("goal", color)   # reuse the 'goal' type for encoding

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 1.0), COLORS[self.color])


# --------------------------------------------------------------------------- #
# Public env: MiniGrid + RGB partial obs + cogniland contract
# --------------------------------------------------------------------------- #
class MemoryEnv:
    """Pixel-observation MiniGrid memory env with the cogniland contract.

    Wraps :class:`_MemoryMiniGridEnv` with ``RGBImgPartialObsWrapper`` so the
    observation is an egocentric **RGB uint8 image** (the agent's point of
    view), and re-exposes the cogniland Gym-style API + ``info`` contract::

        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)

    The agent is the native MiniGrid triangle; the action space is MiniGrid's
    ``Discrete(7)`` (turn-left / turn-right / forward / pickup / drop / toggle /
    done). Navigation uses {left, right, forward}. See the module docstring for
    the task rationale.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: MemoryEnvConfig | None = None, **overrides: Any) -> None:
        if config is None:
            config = MemoryEnvConfig(**overrides)
        elif overrides:
            config = MemoryEnvConfig(**{**config.__dict__, **overrides})
        self.cfg = config

        base = _MemoryMiniGridEnv(config, render_mode="rgb_array")
        self._mg = base
        self._env = RGBImgPartialObsWrapper(base, tile_size=config.cell_px)

        # observation/action spaces (image only — drop the dict wrapper for the
        # cogniland contract; obs returned by reset/step is the raw RGB array).
        px = config.view_size * config.cell_px
        self._obs_shape = (px, px, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self._obs_shape, dtype=np.uint8
        )
        self.action_space = base.action_space          # Discrete(7)

        self._global_step = 0
        self._done = False

    # -- pass-throughs to the underlying minigrid env --------------------- #
    @property
    def cue_type(self) -> str:
        return self._mg.cue_type

    @property
    def cue_shape(self) -> str:
        return self._mg.cue_shape

    @property
    def cue_color(self) -> str:
        return self._mg.cue_color

    @property
    def correct_branch(self) -> str:
        return self._mg.correct_branch

    @property
    def target_door_color(self) -> str:
        return self._mg.target_door_color

    # -- phase schedule helpers (region-based) ---------------------------- #
    def phase_ranges(self) -> dict[str, tuple[int, int]]:
        """Approximate ``{phase: (start_step, end_step_exclusive)}`` for docs.

        Phases here are derived from the agent's *position*, not a fixed clock,
        so this returns the canonical per-phase budget (in oracle steps) as a
        contiguous schedule for alignment/documentation purposes.
        """
        c = self.cfg
        lens = {
            "blank": max(c.pre_cue_steps, 1),
            "cue": max(c.cue_visible_steps, 1),
            "pre_branch_memory": max(c.pre_branch_corridor_len, 1),
            "branch_choice": max(c.branch_len, 1),
            "post_branch_memory": max(c.post_branch_corridor_len, 1),
            "door_visible": max(c.door_visible_distance, 1),
        }
        out: dict[str, tuple[int, int]] = {}
        acc = 0
        for name in PHASES[:-1]:
            out[name] = (acc, acc + lens[name])
            acc += lens[name]
        out["terminal"] = (acc, acc + 1)
        return out

    # -- lifecycle --------------------------------------------------------- #
    def reset(self, seed: int | None = None, options: dict | None = None):
        obs_dict, _ = self._env.reset(seed=seed, options=options)
        self._global_step = 0
        self._done = False
        obs = self._extract_obs(obs_dict)
        info = self._make_info(reward=0.0)
        return obs, info

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        obs_dict, reward, terminated, truncated, _ = self._env.step(int(action))
        self._global_step += 1
        self._done = bool(terminated or truncated)
        obs = self._extract_obs(obs_dict)
        info = self._make_info(reward=float(reward))
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._mg.get_frame(tile_size=self.cfg.cell_px, agent_pov=True)

    def _extract_obs(self, obs_dict) -> np.ndarray:
        img = obs_dict["image"] if isinstance(obs_dict, dict) else obs_dict
        img = np.asarray(img, dtype=np.uint8).copy()
        # The agent is the classic MiniGrid red triangle. Red is NOT a task
        # colour here (cue + doors are green & blue), so no recolour is needed.
        return img

    # -- info contract ----------------------------------------------------- #
    def _make_info(self, reward: float) -> dict[str, Any]:
        c = self.cfg
        mg = self._mg
        phase = "terminal" if self._done else mg.current_phase()

        taken = mg.taken_branch
        branch_correct: bool | None = None
        if taken is not None:
            branch_correct = taken == mg.correct_branch

        selected = mg.selected_door
        success = selected is not None and selected == mg.target_door_color
        wrong_door = selected is not None and selected != mg.target_door_color

        return {
            # private handle so oracle_action(info) can read the agent pose;
            # underscore-prefixed -> not part of the labelled task contract.
            "_mg": mg,
            "env_name": "memory",
            "phase": phase,
            "phase_step": int(self._global_step),
            "global_step": int(self._global_step),
            "cue_shape": mg.cue_shape,
            "cue_color": mg.cue_color,
            "cue_type": mg.cue_type,
            "correct_branch": mg.correct_branch,
            "taken_branch": taken,
            "branch_correct": branch_correct,
            "target_door_color": mg.target_door_color,
            "selected_door_color": selected,
            "door_position_green": mg.door_pos_green,
            "door_position_blue": mg.door_pos_blue,
            "success": bool(success),
            "wrong_door": bool(wrong_door),
            "forced_branch": c.forced_branch,
            "suppress_down_action": c.suppress_down_action,
            "suppress_up_action": c.suppress_up_action,
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_memory_env(config: MemoryEnvConfig | None = None, **overrides: Any) -> MemoryEnv:
    """Build a :class:`MemoryEnv` (MiniGrid + RGB partial-obs wrapper)."""
    return MemoryEnv(config, **overrides)


# --------------------------------------------------------------------------- #
# Oracle policy — navigates the MiniGrid corridor with turn/forward actions.
# --------------------------------------------------------------------------- #
def oracle_action(info: dict[str, Any], env: "MemoryEnv | None" = None) -> int:
    """Optimal scripted action.

    Uses the privileged labels (which a learned policy cannot see) plus the
    agent's current pose to drive the MiniGrid triangle through the corridor,
    pick the cue-shape-correct branch, traverse it, and reach the
    cue-colour-correct final door.

    ``env`` is optional; when omitted, the most-recently-reset ``MemoryEnv``
    passed to :func:`evaluate` / :func:`record_trajectory` is used via the
    bound minigrid pose. For convenience the env is read off
    ``info['_env']`` if present.
    """
    mg = info.get("_mg")
    if mg is None and env is not None:
        mg = env._mg
    assert mg is not None, "oracle_action needs the bound minigrid env (info['_mg'])"

    ax, ay = int(mg.agent_pos[0]), int(mg.agent_pos[1])
    adir = int(mg.agent_dir)
    my = mg._my
    ru, rl = mg._row_up, mg._row_lo
    rdt, rdb = mg._row_door_top, mg._row_door_bot

    def face(target_dir: int) -> int | None:
        """Return a turn action to rotate toward target_dir, or None if facing."""
        if adir == target_dir:
            return None
        # turn right if that gets closer; minigrid dirs are cyclic
        if (adir + 1) % 4 == target_dir:
            return A_RIGHT_TURN
        return A_LEFT_TURN

    phase = info["phase"]

    # Decide the desired row to be on as we move right.
    # - In the branch zone, be on the shape-correct row.
    # - At the door column, be on the colour-correct door row.
    want_branch = info["correct_branch"]            # "up"/"down"
    branch_row = ru if want_branch == "up" else rl

    # target door row:
    want_color = info["target_door_color"]
    door_top = info["door_position_green"] if want_color == "green" else info["door_position_blue"]
    # door_top is "top"/"bottom" giving the side of the wanted colour
    door_row = rdt if door_top == "top" else rdb

    dx_branch_start = mg._x_branch_start
    dx_branch_end = mg._x_branch_end
    dx_post_start = mg._x_post_start
    dx_doorcol = mg._x_doorcol
    jx = mg._x_pre_end

    # ---- Unified row-follower: pick the desired row for the current x-region,
    # move vertically onto it (the junction / reconnect / door columns carve a
    # full vertical slot), then head east. Robust to any central-wall thickness.
    if ax < jx:                         # approach: middle corridor
        desired, allow_east = my, True
    elif ax == jx:                      # junction: rise/drop to the branch row
        desired, allow_east = branch_row, True
    elif ax <= dx_branch_end:           # inside the branch
        desired, allow_east = branch_row, True
    elif ax == dx_post_start:           # reconnect: funnel back to the middle
        desired, allow_east = my, True
    elif ax < dx_doorcol:               # post hallway: middle corridor
        desired, allow_east = my, True
    else:                               # at the door column: go to the door row
        desired, allow_east = door_row, False

    if ay != desired:
        tgt = DIR_NORTH if desired < ay else DIR_SOUTH
        a = face(tgt)
        return a if a is not None else A_FORWARD
    if allow_east:
        a = face(DIR_EAST)
        return a if a is not None else A_FORWARD
    return A_FORWARD


# --------------------------------------------------------------------------- #
# Evaluation helper
# --------------------------------------------------------------------------- #
def evaluate(
    env: MemoryEnv,
    policy: Callable[[np.ndarray, dict], int] | None = None,
    n_episodes: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Run ``policy`` (or the oracle) for ``n_episodes`` and report aggregates.

    ``policy`` takes ``(obs, info)`` and returns an action; if ``None`` the
    oracle policy is used. Reports success / branch / door choice broken down by
    cue type, plus the cross-feature confusion rates the analysis cares about.
    """
    from collections import defaultdict

    def _bind(info):
        info = dict(info)
        info["_mg"] = env._mg
        return info

    if policy is None:
        policy = lambda obs, info: oracle_action(info)  # noqa: E731

    by_cue = defaultdict(lambda: {"n": 0, "success": 0,
                                  "branch_up": 0, "branch_down": 0,
                                  "door_green": 0, "door_blue": 0})
    blue_down_supp = {"n": 0, "green": 0}
    green_up_supp = {"n": 0, "blue": 0}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        cue = info["cue_type"]
        terminated = truncated = False
        last_info = info
        while not (terminated or truncated):
            a = policy(obs, _bind(info))
            obs, _, terminated, truncated, info = env.step(a)
            last_info = info
        rec = by_cue[cue]
        rec["n"] += 1
        rec["success"] += int(last_info["success"])
        tb = last_info["taken_branch"]
        if tb == "up":
            rec["branch_up"] += 1
        elif tb == "down":
            rec["branch_down"] += 1
        sd = last_info["selected_door_color"]
        if sd == "green":
            rec["door_green"] += 1
        elif sd == "blue":
            rec["door_blue"] += 1

        if cue == "blue_down" and env.cfg.suppress_down_action:
            blue_down_supp["n"] += 1
            blue_down_supp["green"] += int(sd == "green")
        if cue == "green_up" and env.cfg.suppress_up_action:
            green_up_supp["n"] += 1
            green_up_supp["blue"] += int(sd == "blue")

    def _rate(num: int, den: int) -> float:
        return num / den if den else float("nan")

    success_by_cue = {k: _rate(v["success"], v["n"]) for k, v in by_cue.items()}
    branch_by_cue = {
        k: {"up": _rate(v["branch_up"], v["n"]), "down": _rate(v["branch_down"], v["n"])}
        for k, v in by_cue.items()
    }
    door_by_cue = {
        k: {"green": _rate(v["door_green"], v["n"]), "blue": _rate(v["door_blue"], v["n"])}
        for k, v in by_cue.items()
    }

    blue_cues = {"blue_up", "blue_down"}
    green_cues = {"green_up", "green_down"}
    rd_on_blue = sum(by_cue[c]["door_green"] for c in blue_cues if c in by_cue)
    n_blue = sum(by_cue[c]["n"] for c in blue_cues if c in by_cue)
    bd_on_red = sum(by_cue[c]["door_blue"] for c in green_cues if c in by_cue)
    n_red = sum(by_cue[c]["n"] for c in green_cues if c in by_cue)

    return {
        "n_episodes": n_episodes,
        "success_by_cue": success_by_cue,
        "branch_by_cue": branch_by_cue,
        "door_by_cue": door_by_cue,
        "green_door_rate_on_blue_cues": _rate(rd_on_blue, n_blue),
        "blue_door_rate_on_green_cues": _rate(bd_on_red, n_red),
        "green_door_rate_on_blue_down_suppress_down": _rate(blue_down_supp["green"], blue_down_supp["n"]),
        "blue_door_rate_on_green_up_suppress_up": _rate(green_up_supp["blue"], green_up_supp["n"]),
        "counts_by_cue": {k: dict(v) for k, v in by_cue.items()},
    }


# --------------------------------------------------------------------------- #
# Trajectory logger
# --------------------------------------------------------------------------- #
def record_trajectory(
    env: MemoryEnv,
    policy: Callable[[np.ndarray, dict], int] | None = None,
    seed: int | None = None,
    as_arrays: bool = False,
) -> list[dict[str, Any]] | dict[str, np.ndarray]:
    """Roll out one episode and record per-timestep task-aligned fields.

    Each record holds ``observation, action, reward, done`` plus the phase /
    cue / branch / door labels needed to align DreamerV3 latents to task phases.
    Returns a list of per-step dicts, or (``as_arrays=True``) a dict of arrays.
    """
    def _bind(info):
        info = dict(info)
        info["_mg"] = env._mg
        return info

    if policy is None:
        policy = lambda obs, info: oracle_action(info)  # noqa: E731

    obs, info = env.reset(seed=seed)
    records: list[dict[str, Any]] = []
    terminated = truncated = False
    prev_obs, prev_info = obs, info
    while not (terminated or truncated):
        action = policy(prev_obs, _bind(prev_info))
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        records.append({
            "observation": prev_obs,
            "action": int(action),
            "reward": float(reward),
            "done": bool(terminated or truncated),
            "phase": prev_info["phase"],
            "phase_step": prev_info["phase_step"],
            "cue_shape": prev_info["cue_shape"],
            "cue_color": prev_info["cue_color"],
            "cue_type": prev_info["cue_type"],
            "correct_branch": prev_info["correct_branch"],
            "taken_branch": next_info["taken_branch"],
            "target_door_color": prev_info["target_door_color"],
            "selected_door_color": next_info["selected_door_color"],
            "success": next_info["success"],
        })
        prev_obs, prev_info = next_obs, next_info

    if not as_arrays:
        return records

    keys = records[0].keys()
    out: dict[str, np.ndarray] = {}
    for k in keys:
        vals = [r[k] for r in records]
        if k == "observation":
            out[k] = np.stack(vals).astype(np.uint8)
        else:
            out[k] = np.array(vals, dtype=object if any(v is None for v in vals) else None)
    return out
