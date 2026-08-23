#!/usr/bin/env python
"""MiniGrid-style tile textures for MemoryEnv visualizations.

Pure-numpy rasterizer (supersampled + mean-pooled) drawing the classic MiniGrid
look: black floor with grid lines, grey walls, framed doors with a handle, and
four CUSTOM cue tiles (colored tile + white arrow pointing up/down). Marker
doors get a closed texture (ids 9/10) and an open variant (render-only ids
11/12: thin door leaf against the cell edge, MiniGrid open-door style).

The agent-view minimap renders EXACTLY the observation tile ids (an opened
marker shows floor — that is what the agent sees); the high-level view may
substitute the open-door textures via ``OPEN_TEX_ID``.
"""
from __future__ import annotations

import numpy as np

from cogniland.memory_env.jax import constants as C

PX = 24      # rendered pixels per cell
_SUB = 4     # supersampling factor

# MiniGrid palette
_GREEN = (0, 200, 0)
_BLUE = (60, 110, 255)
_PURPLE = (112, 39, 195)
_YELLOW = (212, 176, 0)
_GREY = (100, 100, 100)
_FLOOR = (12, 12, 12)
_LINE = (60, 60, 60)


def _blank():
    n = PX * _SUB
    img = np.zeros((n, n, 3), np.float32)
    img[:] = np.asarray(_FLOOR, np.float32) / 255.0
    return img


def _fill(img, pred, color):
    n = img.shape[0]
    ys, xs = np.mgrid[0:n, 0:n]
    xf, yf = (xs + 0.5) / n, (ys + 0.5) / n
    img[pred(xf, yf)] = np.asarray(color, np.float32) / 255.0


def _rect(xa, xb, ya, yb):
    return lambda x, y: (x >= xa) & (x < xb) & (y >= ya) & (y < yb)


def _circle(cx, cy, r):
    return lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _tri(p1, p2, p3):
    def pred(x, y):
        def side(a, b):
            return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
        d1, d2, d3 = side(p1, p2), side(p2, p3), side(p3, p1)
        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(neg & pos)
    return pred


def _down(img):
    return img.reshape(PX, _SUB, PX, _SUB, 3).mean((1, 3))


def _grid_lines(img):
    _fill(img, _rect(0.0, 0.031, 0.0, 1.0), _LINE)
    _fill(img, _rect(0.0, 1.0, 0.0, 0.031), _LINE)


def _floor():
    img = _blank()
    _grid_lines(img)
    return _down(img)


def _wall():
    img = _blank()
    _fill(img, _rect(0, 1, 0, 1), _GREY)
    return _down(img)


def _oob():
    img = _blank()
    _fill(img, _rect(0, 1, 0, 1), (45, 45, 45))
    return _down(img)


def _cue(color, up):
    """Custom cue tile: colored tile with a white arrow pointing up or down."""
    img = _blank()
    _grid_lines(img)
    _fill(img, _rect(0.08, 0.92, 0.08, 0.92), color)
    w = (255, 255, 255)
    if up:
        _fill(img, _rect(0.42, 0.58, 0.42, 0.82), w)                     # shaft
        _fill(img, _tri((0.50, 0.14), (0.24, 0.46), (0.76, 0.46)), w)     # head
    else:
        _fill(img, _rect(0.42, 0.58, 0.18, 0.58), w)
        _fill(img, _tri((0.50, 0.86), (0.24, 0.54), (0.76, 0.54)), w)
    return _down(img)


def _goal_door(color):
    """Final colored door (goal-like walkable cell): solid color, thin frame."""
    img = _blank()
    _fill(img, _rect(0.0, 1.0, 0.0, 1.0), tuple(int(c * 0.55) for c in color))
    _fill(img, _rect(0.06, 0.94, 0.06, 0.94), color)
    return _down(img)


def _door_closed(color):
    """MiniGrid closed door: colored frame, inset panels, round handle."""
    img = _blank()
    _fill(img, _rect(0.00, 1.00, 0.00, 1.00), color)
    _fill(img, _rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
    _fill(img, _rect(0.08, 0.92, 0.08, 0.92), color)
    _fill(img, _rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))
    _fill(img, _circle(0.75, 0.50, 0.08), color)
    return _down(img)


def _door_open(color):
    """MiniGrid open door: thin leaf against the cell edge, floor visible."""
    img = _blank()
    _grid_lines(img)
    _fill(img, _rect(0.88, 1.00, 0.00, 1.00), color)
    _fill(img, _rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
    return _down(img)


# texture atlas: env tile ids 0..NUM_TILES-1, then render-only open-door ids
OPEN_A_ID = C.NUM_TILES          # 11: opened top marker (render-only)
OPEN_B_ID = C.NUM_TILES + 1      # 12: opened bottom marker (render-only)
OPEN_TEX_ID = {C.MARK_A: OPEN_A_ID, C.MARK_B: OPEN_B_ID}

TILE_TEX = np.stack([
    _floor(),                    # 0 EMPTY
    _wall(),                     # 1 WALL
    _cue(_GREEN, up=True),       # 2 CUE_GREEN_UP
    _cue(_BLUE, up=True),        # 3 CUE_BLUE_UP
    _cue(_GREEN, up=False),      # 4 CUE_GREEN_DOWN
    _cue(_BLUE, up=False),       # 5 CUE_BLUE_DOWN
    _goal_door(_GREEN),          # 6 DOOR_GREEN
    _goal_door(_BLUE),           # 7 DOOR_BLUE
    _oob(),                      # 8 OOB
    _door_closed(_PURPLE),       # 9 MARK_A closed
    _door_closed(_YELLOW),       # 10 MARK_B closed
    _door_open(_PURPLE),         # 11 MARK_A open (render-only)
    _door_open(_YELLOW),         # 12 MARK_B open (render-only)
])


def render_grid(ids):
    """(H, W) int tile ids -> (H*PX, W*PX, 3) float RGB image.

    Show with ``imshow(img, extent=(-0.5, W-0.5, H-0.5, -0.5))`` so overlays
    keep using cell coordinates.
    """
    ids = np.asarray(ids, np.int32)
    tiles = TILE_TEX[ids]                        # (H, W, PX, PX, 3)
    H, W = ids.shape
    return tiles.transpose(0, 2, 1, 3, 4).reshape(H * PX, W * PX, 3)


def agent_triangle(x, y, d, size=1.0):
    """Matplotlib Polygon vertices for the MiniGrid red agent triangle at cell
    (x, y) facing d (0=E, 1=S, 2=W, 3=N; +y down)."""
    base = np.array([[-0.32, -0.32], [0.42, 0.0], [-0.32, 0.32]]) * size
    th = d * np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (base @ R.T) + np.array([x, y])
