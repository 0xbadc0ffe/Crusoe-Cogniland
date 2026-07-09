"""Deterministic geometry builder for the JAX MemoryEnv.

MemoryEnv's grid is fully determined by the config (unlike bridge_tunnel there is
no procedural map dataset); only the cue type, the two door positions and the
cue cell are randomised per episode. This module computes the column/row anchors
and the static base terrain (walls + carved corridors, WITHOUT the cue or door
tiles, which are overlaid per-episode). It mirrors `_MemoryMiniGridEnv.__init__`
and `_gen_grid` in cogniland.memory_env.env exactly.
"""
from __future__ import annotations

import numpy as np

from . import constants as C

ROOM_W = 4


def build_geometry(*, pre_cue_steps=1, pre_branch_corridor_len=5, branch_len=4,
                   post_branch_corridor_len=5, view_size=5,
                   center_wall_thickness=3) -> dict:
    """Return all anchors + the static base terrain (H, W) int8.

    Returns a dict with the integer anchors (ints, become static EnvParams
    fields) and ``base_terrain`` (numpy int8 array): EMPTY corridors carved out
    of a WALL-filled interior; cue/door tiles are NOT included here.
    """
    if view_size % 2 == 0 or view_size < 3:
        raise ValueError(f"view_size={view_size} must be odd and >= 3")
    if center_wall_thickness < 1 or center_wall_thickness % 2 == 0:
        raise ValueError(f"center_wall_thickness={center_wall_thickness} must be odd >= 1")

    pre_len = max(pre_branch_corridor_len, 2)
    post_len = max(post_branch_corridor_len, 2)
    precue_len = max(pre_cue_steps, 1) + (view_size - 1)

    x_precue_start = 1
    x_precue_end = x_precue_start + precue_len - 1
    x_room_start = x_precue_end + 1
    x_room_end = x_room_start + ROOM_W - 1
    x_pre_start = x_room_end + 1
    x_pre_end = x_pre_start + pre_len - 1
    x_branch_start = x_pre_end + 1
    x_branch_end = x_branch_start + branch_len - 1
    x_post_start = x_branch_end + 1
    x_post_end = x_post_start + post_len - 1
    x_doorcol = x_post_end + 1
    width = x_doorcol + 3

    bgap = (center_wall_thickness + 1) // 2
    door_off = min(bgap + 1, (view_size - 1) // 2)
    outer = max(bgap, door_off)
    my = outer + 1
    row_up = my - bgap
    row_lo = my + bgap
    row_room_up = my - 1
    row_room_lo = my + 1
    row_door_top = my - door_off
    row_door_bot = my + door_off
    height = my + outer + 2

    # ── carve base terrain (WALL-filled interior, then carve EMPTY corridors) ──
    t = np.full((height, width), C.WALL, dtype=np.int8)  # full of walls
    t[0, :] = C.WALL; t[-1, :] = C.WALL; t[:, 0] = C.WALL; t[:, -1] = C.WALL

    def carve(x, y):
        t[y, x] = C.EMPTY

    # start room: 3 rows (room_up, my, room_lo) x [room_start..room_end]
    for x in range(x_room_start, x_room_end + 1):
        for y in (row_room_up, my, row_room_lo):
            carve(x, y)
    # wall the room's right edge except the middle doorway
    t[row_room_up, x_room_end] = C.WALL
    t[row_room_lo, x_room_end] = C.WALL
    carve(x_room_end, my)
    # initial empty corridor west of the room (middle row)
    for x in range(x_precue_start, x_room_start):
        carve(x, my)
    # pre-branch hallway (middle row)
    for x in range(x_pre_start, x_pre_end + 1):
        carve(x, my)
    # branch zone: upper + lower branch rows; rows between stay WALL (thick wall)
    for x in range(x_branch_start, x_branch_end + 1):
        carve(x, row_up)
        carve(x, row_lo)
    # junction vertical slot at last pre-branch column (row_up..row_lo)
    for y in range(row_up, row_lo + 1):
        carve(x_pre_end, y)
    # reconnect vertical slot at first post column
    for y in range(row_up, row_lo + 1):
        carve(x_post_start, y)
    # post hallway (middle row)
    for x in range(x_post_start, x_post_end + 1):
        carve(x, my)
    # final door corridor (vertical, door_top..door_bot) + approach doorway
    for y in range(row_door_top, row_door_bot + 1):
        carve(x_doorcol, y)
    carve(x_post_end, my)
    # closing wall column right of the door corridor (already WALL from fill)

    return dict(
        base_terrain=t,
        width=int(width), height=int(height),
        my=int(my), row_up=int(row_up), row_lo=int(row_lo),
        row_room_up=int(row_room_up), row_room_lo=int(row_room_lo),
        row_door_top=int(row_door_top), row_door_bot=int(row_door_bot),
        x_precue_start=int(x_precue_start), x_precue_end=int(x_precue_end),
        x_room_start=int(x_room_start), x_room_end=int(x_room_end),
        x_pre_start=int(x_pre_start), x_pre_end=int(x_pre_end),
        x_branch_start=int(x_branch_start), x_branch_end=int(x_branch_end),
        x_post_start=int(x_post_start), x_post_end=int(x_post_end),
        x_doorcol=int(x_doorcol),
    )
