"""Minimal, self-contained bridge_tunnel env — numpy only, NO cogniland import.

Reconstructs an episode from a stored terrain grid (``maps.npz``) so a colleague
with just the dataset folder + a checkpoint can reproduce and steer trajectories.
It reproduces the training env's **observation + dynamics** bit-for-bit (move /
build / mine / commit gating / egocentric crop / scalars); it omits reward and the
cost-to-go (irrelevant to the policy's behaviour, which depends only on the obs).

variant "bt": place(water→wood)/mine(rock→grass) always active, 5 scalars.
variant "btc": first successful build/mine commits irreversibly; the opposite tool
is then a no-op; 7 scalars (+commit_build, commit_mine).
"""
from __future__ import annotations
import numpy as np

GRASS, WATER, ROCK, WOOD, TARGET, OOB, TREE, SAND, DIRT = range(9)
_WALK = (GRASS, WOOD, TARGET, SAND, DIRT)
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_BUILD, A_MINE = range(6)
F_UP, F_DOWN, F_LEFT, F_RIGHT = range(4)
_DELTA = {F_UP: (-1, 0), F_DOWN: (1, 0), F_LEFT: (0, -1), F_RIGHT: (0, 1)}
COMMIT_NONE, COMMIT_BUILD, COMMIT_MINE = range(3)


class MiniBridgeTunnelEnv:
    def __init__(self, terrain, spawn, *, variant="bt", view_size=21, max_steps=800):
        self.variant = variant
        self.commit_enabled = (variant == "btc")
        self.terrain0 = np.asarray(terrain, np.int8)
        self.H, self.W = self.terrain0.shape
        self.spawn = tuple(int(x) for x in spawn)
        self.view = int(view_size)
        self.max_steps = int(max_steps)
        self.reset()

    def reset(self):
        self.terrain = self.terrain0.copy()
        self.pos = self.spawn
        self.facing = F_RIGHT
        self.commit = COMMIT_NONE
        self.t = 0
        return self._obs()

    def step(self, a):
        a = int(a)
        reached = False
        if a < 4:
            self.facing = a
            dr, dc = _DELTA[self.facing]
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and int(self.terrain[nr, nc]) in _WALK:
                self.pos = (nr, nc)
                if self.terrain[nr, nc] == TARGET:
                    reached = True
        elif a == A_BUILD:
            self._tool(WATER, WOOD, COMMIT_BUILD, COMMIT_MINE)
        elif a == A_MINE:
            self._tool(ROCK, GRASS, COMMIT_MINE, COMMIT_BUILD)
        self.t += 1
        done = reached or self.t >= self.max_steps
        return self._obs(), reached, done

    def _tool(self, tile, become, slot, locked):
        if self.commit_enabled and self.commit == locked:
            return                                   # locked opposite → no-op
        dr, dc = _DELTA[self.facing]
        fr, fc = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= fr < self.H and 0 <= fc < self.W and self.terrain[fr, fc] == tile:
            self.terrain[fr, fc] = become
            if self.commit_enabled and self.commit == COMMIT_NONE:
                self.commit = slot

    def _obs(self):
        V, half = self.view, self.view // 2
        ar, ac = self.pos
        crop = np.full((V, V), OOB, np.int8)
        r0, c0 = ar - half, ac - half
        rs = max(0, -r0); re = V - max(0, (r0 + V) - self.H)
        cs = max(0, -c0); ce = V - max(0, (c0 + V) - self.W)
        if rs < re and cs < ce:
            crop[rs:re, cs:ce] = self.terrain[r0 + rs:r0 + re, c0 + cs:c0 + ce]
        face = np.zeros(4, np.float32); face[self.facing] = 1.0
        step = np.float32(self.t / max(1, self.max_steps))
        if self.commit_enabled:
            scalars = np.concatenate([face, np.array(
                [step, np.float32(self.commit == COMMIT_BUILD),
                 np.float32(self.commit == COMMIT_MINE)], np.float32)])
        else:
            scalars = np.concatenate([face, np.array([step], np.float32)])
        return {"minimap": crop, "scalars": scalars}
