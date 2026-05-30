"""Reward, walkability, and slip mechanics for Cogniland navigation.

Reward design (PBRS)
--------------------
Every env step pays the SAME flat slack penalty plus, on a *successful*
move, a potential-based shaping term proportional to the change in cost-to-
go (positive when getting closer, negative when getting farther, 0 when
the position didn't change). A sparse reach bonus anchors the optimum.

* ``SLACK_PENALTY = -0.005`` — paid on *every* action (move, build, slip,
  collision). Makes idling strictly worse than productive moves.
* ``SHAPING_COEF * (ctg_old - ctg_new)`` — main driver. With ``ctg``
  measured in cells (unit cost), this is ``±SHAPING_COEF`` for a
  successful step and ``0`` for stay-still (slip, collision, build).
* ``REACH_BONUS = +1.0`` — sparse, paid on the step that lands on TARGET.

Per-action arithmetic:

  successful move toward target  : -0.005 + 0.01   = +0.005
  successful move sideways       : -0.005 +  0    = -0.005
  successful move away from goal : -0.005 - 0.01   = -0.015
  slip (stay still)              : -0.005
  collision (stay still)         : -0.005
  build (always)                 : -0.005
  reach target (toward step)     : -0.005 + 0.01 + 1.0 = +1.005

Walkability + slip
------------------
Every terrain except lava / out-of-map is *walkable* — including water and
rock. The skill mechanic is purely a slip-rate modifier; each skill owns two
terrains (RAFT → water + sand, HARNESS → dirt + rock):

  ``slip_chance(*, water)`` = 0.75, RAFT → 0
  ``slip_chance(*, rock)``  = 0.75, HARNESS → 0
  ``slip_chance(*, sand)``  = 0.30, RAFT → 0
  ``slip_chance(*, dirt)``  = 0.30, HARNESS → 0
  ``slip_chance(*, tree)``  = 0.75 always; grass/target = 0

Slip never blocks: the agent can wade across the wrong terrain, it just takes
more attempts per cell — so the env is always solvable, but the matching skill
is far more efficient.
"""

from __future__ import annotations

import math

import numpy as np

from .tiles import DIRT, GRASS, LAVA, OOB, ROCK, SAND, TARGET, TREE, WATER

# Active object id (the agent's "item")
NONE = 0
RAFT = 1
HARNESS = 2
NUM_OBJECTS = 3

OBJECT_NAMES = {NONE: "none", RAFT: "raft", HARNESS: "harness"}

# ── Reward constants ──────────────────────────────────────────────────────
SLACK_PENALTY = -0.02     # flat per-action cost (was -0.005; raised so
                          # length-of-path dominates the return, pushing
                          # the policy to keep tightening the route)
SHAPING_COEF = +0.01      # × Δctg (positive = closer)
REACH_BONUS = +1.0        # sparse terminal reward for reaching the target
CLIP_NEG_SHAPING = False  # if True, clip Δctg at 0 in the env's PBRS shaping
                          # so backward steps (Δctg < 0) get only the flat
                          # slack penalty, never the asymmetric -SHAPING*1
                          # penalty. Removes the "always move toward top-right"
                          # bias while keeping uniform per-step time pressure.

# Legacy names kept for any external callers — the env uses SLACK_PENALTY now.
STEP_COST = SLACK_PENALTY
BUILD_COST = SLACK_PENALTY
COLLISION_PENALTY = 0.0   # already folded into the flat slack

# ── Slip mechanic ─────────────────────────────────────────────────────────
# Each skill specialises in two terrains: RAFT → {water, sand}, HARNESS →
# {dirt, rock}. The matching skill drops slip to 0 on its two terrains; the
# wrong/no skill slips at the terrain's base rate.
SLIP_PROB_DEFAULT = 0.75   # major barriers (water, rock, tree) without the
                           # matching item.
SLIP_PROB_MINOR = 0.30     # bare-handed sand/dirt — fixed apron tax for the
                           # no-skill agent (skill-active sand/dirt rises to
                           # SLIP_PROB_LAND_WITH_SKILL).
SLIP_PROB_LAND_WITH_SKILL = 0.50
                           # Land weight tax (2026-05-28, lowered from 0.75):
                           # grass / sand / dirt all slip at 50 %% whenever ANY
                           # skill is committed. Carrying a raft/harness still
                           # makes land notably slippery, but the gap to a
                           # truly impassable barrier (0.75) is reserved for
                           # water/rock/tree.
SLIP_PROB_GRASS_NOSKILL = 0.0   # grass slip while NO skill is committed
                           # (bare-handed). Default 0 — swept in the
                           # grass-slip experiment to probe how baseline
                           # ground friction shapes the bare-handed policy.
SLIP_PROB_GRASS = SLIP_PROB_LAND_WITH_SKILL  # deprecated alias (pre-2026-05-28)
SLIP_WEIGHT_LAND = SLIP_PROB_MINOR   # deprecated alias (old "weight tax" name)

# ── Sentinel ──────────────────────────────────────────────────────────────
BLOCKED = math.inf

# Old per-(skill, tile) cost names — alias to SLACK_PENALTY for any caller.
COST_NONE_LAND = SLACK_PENALTY
COST_RAFT_LAND = SLACK_PENALTY
COST_HARNESS_LAND = SLACK_PENALTY
COST_RAFT_WATER = SLACK_PENALTY
COST_HARNESS_ROCK = SLACK_PENALTY


def walkable(obj: int, tile: int) -> bool:
    """True if ``tile`` can be stepped onto (regardless of object).

    Water, rock, and trees are all walkable for everyone — the skill just
    reduces slip on water/rock. Trees stay 90 %% slip for everyone.
    Lava and off-map cells are always blocked.
    """
    return tile in (GRASS, DIRT, SAND, TARGET, WATER, ROCK, TREE)


def slip_chance(obj: int, tile: int) -> float:
    """Probability that a move onto ``tile`` slips (stays in place).

    RAFT zeroes water; HARNESS zeroes rock; trees always slip 75 %%.
    **Hard-land weight tax (2026-05-28):** when ANY skill is committed, grass
    / sand / dirt all slip at ``SLIP_PROB_LAND_WITH_SKILL`` (75 %%). With no
    skill: sand/dirt slip at ``SLIP_PROB_MINOR`` (30 %%) and grass slips at
    ``SLIP_PROB_GRASS_NOSKILL`` (default 0 %%, sweep knob). The target tile
    never slips.
    """
    if tile == WATER:
        return 0.0 if obj == RAFT else SLIP_PROB_DEFAULT
    if tile == ROCK:
        return 0.0 if obj == HARNESS else SLIP_PROB_DEFAULT
    if tile == SAND or tile == DIRT:
        return SLIP_PROB_LAND_WITH_SKILL if obj != NONE else SLIP_PROB_MINOR
    if tile == TREE:
        return SLIP_PROB_DEFAULT
    if tile == GRASS:
        return SLIP_PROB_LAND_WITH_SKILL if obj != NONE else SLIP_PROB_GRASS_NOSKILL
    return 0.0  # target — the goal tile never slips


def object_from_scalar(s: float) -> int:
    """tanh-style scalar → object. ≥ 0 = raft, < 0 = harness (deterministic tie)."""
    return RAFT if s >= 0.0 else HARNESS


# ────────────────────────────────────────────── Dijkstra cost surfaces ──


_WALKABLE_TILES = (GRASS, DIRT, SAND, TARGET, WATER, ROCK, TREE)


def _walk_mask(terrain: np.ndarray) -> np.ndarray:
    mask = np.zeros(terrain.shape, dtype=bool)
    for t in _WALKABLE_TILES:
        mask |= terrain == t
    return mask


def unit_cost_grid(obj: int, terrain: np.ndarray) -> np.ndarray:
    """``1`` per walkable cell, ``inf`` otherwise. Identical for all objects."""
    out = np.full(terrain.shape, BLOCKED, dtype=np.float32)
    out[_walk_mask(terrain)] = 1.0
    return out


def expected_attempts_grid(obj: int, terrain: np.ndarray) -> np.ndarray:
    """``1 / (1 − slip_chance(obj, tile))`` per walkable cell, ``inf`` else."""
    out = np.full(terrain.shape, BLOCKED, dtype=np.float32)
    for t in _WALKABLE_TILES:
        out[terrain == t] = 1.0 / (1.0 - slip_chance(obj, t))
    return out


# ── Backwards-compat shims ────────────────────────────────────────────────
def cost_grid(obj: int, terrain: np.ndarray) -> np.ndarray:
    """Legacy alias for callers that imported the old name."""
    out = np.full(terrain.shape, BLOCKED, dtype=np.float32)
    out[np.isfinite(unit_cost_grid(obj, terrain))] = -SLACK_PENALTY
    return out


def step_cost(obj: int, tile: int) -> float:  # pragma: no cover - legacy
    if not walkable(obj, tile):
        return BLOCKED
    return SLACK_PENALTY
