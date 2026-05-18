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
Every terrain except trees / lava / out-of-map is *walkable* — including
water and rock. The skill mechanic is purely a slip-rate modifier:

  default ``slip_chance(NONE, water) = 0.90``    (with raft → ``0``)
  default ``slip_chance(NONE, rock)  = 0.90``    (with harness → ``0``)
  slip_chance on land               = ``0`` for all objects

A 90 %% slip means the agent can in principle wade across a lake without
a raft — it just takes ~10 attempts per cell — so the env is always
solvable, but building the correct skill is far more efficient.
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
REACH_BONUS = +0.0        # disabled — PBRS shaping is the sole positive
                          # signal; eliminates the value-fn cliff at the
                          # target and removes the "any-route-wins" plateau

# Legacy names kept for any external callers — the env uses SLACK_PENALTY now.
STEP_COST = SLACK_PENALTY
BUILD_COST = SLACK_PENALTY
COLLISION_PENALTY = 0.0   # already folded into the flat slack

# ── Slip mechanic ─────────────────────────────────────────────────────────
SLIP_PROB_DEFAULT = 0.75   # on water/rock without the matching item.
                           # Lowered from 0.90 so that going *around* a
                           # small lake/rocky patch (4× attempts/cell
                           # without skill, was 10×) competes with
                           # committing to raft/harness — drives the
                           # policy to use no-skill when shape allows.
SLIP_WEIGHT_LAND = 0.30    # carrying any item slips this often on plain land
                           # — the "weight" tax that makes the wrong skill
                           # strictly worse than carrying nothing.

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
    """Probability that a move onto ``tile`` slips (stays in place)."""
    if tile == WATER:
        return 0.0 if obj == RAFT else SLIP_PROB_DEFAULT
    if tile == ROCK:
        return 0.0 if obj == HARNESS else SLIP_PROB_DEFAULT
    if tile == TREE:
        return SLIP_PROB_DEFAULT
    # Plain land — slips only if the agent is carrying something (weight tax).
    return SLIP_WEIGHT_LAND if obj != NONE else 0.0


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
    walk = _walk_mask(terrain)
    # Base land cost: 1 if no item, slightly more if carrying one (weight tax).
    land_cost = 1.0 if obj == NONE else 1.0 / (1.0 - SLIP_WEIGHT_LAND)
    out[walk] = land_cost
    big = 1.0 / (1.0 - SLIP_PROB_DEFAULT)
    if obj != RAFT:
        out[terrain == WATER] = big
    else:
        out[terrain == WATER] = 1.0  # raft: water no-slip
    if obj != HARNESS:
        out[terrain == ROCK] = big
    else:
        out[terrain == ROCK] = 1.0  # harness: rock no-slip
    out[terrain == TREE] = big  # trees always slip
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
