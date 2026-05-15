"""Sprite-based renderer for Cogniland navigation.

The sprite cache pulls Crafter PNGs from the package's ``assets/sprites/``
directory (copied at install time) and scales each one once with PIL
``NEAREST`` to the requested ``tile_px``. Two renderers operate on the same
sprite cache:

* :meth:`SpriteSheet.render_observation` returns the agent's CHW RGB local
  crop centred on its current cell, with cells outside the map rendered
  black.
* :meth:`SpriteSheet.render_full` returns an HxWx3 array of the entire map
  with optional agent + target overlays and a highlighted view rectangle
  — used by the playable demo and by ``render_mode="human"``.

No pygame dependency in this module — everything is pure numpy/PIL so it
imports cleanly in headless environments. The demo wraps a pygame ``Surface``
around the numpy output.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from . import skills as sk
from .tiles import (
    DIRT,
    GRASS,
    LAVA,
    OOB,
    ROCK,
    SAND,
    TARGET,
    TILE_COLORS,
    TREE,
    WATER,
)

_TERRAIN_SPRITES: dict[int, str] = {
    GRASS: "grass.png",
    DIRT: "path.png",
    SAND: "sand.png",
    WATER: "water.png",
    ROCK: "stone.png",
    TARGET: "grass.png",  # target is land; the diamond is drawn on top
    TREE: "grass.png",    # tree sits on a grass base; tree overlay added later
    LAVA: "lava.png",
}

_OBJECT_SPRITES: dict[int, str] = {
    sk.NONE: "player.png",
    sk.RAFT: "player.png",
    sk.HARNESS: "player.png",
}


class SpriteSheet:
    """Pre-scaled sprite cache + render helpers."""

    def __init__(self, tile_px: int = 16) -> None:
        self.tile_px = int(tile_px)
        if self.tile_px < 4:
            raise ValueError(f"tile_px={tile_px} must be ≥ 4")
        self._sprites: dict[str, np.ndarray] = {}
        self._sprite_alpha: dict[str, np.ndarray] = {}
        self._load_sprites()

    # -- asset loading ---------------------------------------------------

    def _asset_dir(self) -> Path:
        # importlib.resources keeps the path inside the installed wheel; for an
        # editable install this resolves to the repo's src/ tree.
        with resources.as_file(resources.files("cogniland.assets") / "sprites") as p:
            return Path(p)

    def _load_one(self, fname: str) -> tuple[np.ndarray, np.ndarray]:
        path = self._asset_dir() / fname
        img = Image.open(path).convert("RGBA")
        img = img.resize((self.tile_px, self.tile_px), resample=Image.NEAREST)
        arr = np.array(img, dtype=np.uint8)
        rgb = arr[..., :3]
        alpha = arr[..., 3].astype(np.float32) / 255.0
        return rgb, alpha

    def _load_sprites(self) -> None:
        for fname in (
            "grass.png",
            "path.png",
            "sand.png",
            "water.png",
            "stone.png",
            "diamond.png",
            "tree.png",
            "lava.png",
            "player.png",
            "player-up.png",
            "player-down.png",
            "player-left.png",
            "player-right.png",
        ):
            rgb, alpha = self._load_one(fname)
            self._sprites[fname] = rgb
            self._sprite_alpha[fname] = alpha

    # -- low-level draw --------------------------------------------------

    def _blit(self, canvas: np.ndarray, fname: str, r: int, c: int) -> None:
        """Blit sprite onto canvas at pixel offset (r*tile_px, c*tile_px)."""
        t = self.tile_px
        y, x = r * t, c * t
        H, W, _ = canvas.shape
        if y + t > H or x + t > W or y < 0 or x < 0:
            return
        rgb = self._sprites[fname]
        a = self._sprite_alpha[fname][..., None]
        canvas[y : y + t, x : x + t] = (rgb * a + canvas[y : y + t, x : x + t] * (1 - a)).astype(np.uint8)

    def _fill_color(self, canvas: np.ndarray, color: tuple[int, int, int], r: int, c: int) -> None:
        t = self.tile_px
        y, x = r * t, c * t
        canvas[y : y + t, x : x + t] = np.array(color, dtype=np.uint8)

    # -- public renders --------------------------------------------------

    def render_full(
        self,
        terrain: np.ndarray,
        agent_pos: tuple[int, int],
        target_pos: tuple[int, int],
        view_rect: tuple[int, int, int, int] | None = None,
        agent_facing: str = "down",
    ) -> np.ndarray:
        """Render the entire map.

        ``view_rect`` is ``(r0, c0, view_h, view_w)`` in tile coordinates; if
        provided, a 1-px highlight rectangle is drawn around it.
        Returns ``uint8 [size*tile_px, size*tile_px, 3]``.
        """
        H, W = terrain.shape
        canvas = np.zeros((H * self.tile_px, W * self.tile_px, 3), dtype=np.uint8)
        for r in range(H):
            row = terrain[r]
            for c in range(W):
                tile = int(row[c])
                fname = _TERRAIN_SPRITES.get(tile)
                if fname is not None:
                    self._blit(canvas, fname, r, c)
                else:
                    self._fill_color(canvas, (0, 0, 0), r, c)
                if tile == TREE:
                    self._blit(canvas, "tree.png", r, c)
        # target overlay
        tr, tc = target_pos
        self._blit(canvas, "diamond.png", tr, tc)
        # agent overlay
        ar, ac = agent_pos
        face_map = {
            "up": "player-up.png",
            "down": "player-down.png",
            "left": "player-left.png",
            "right": "player-right.png",
        }
        self._blit(canvas, face_map.get(agent_facing, "player.png"), ar, ac)
        # view rect highlight
        if view_rect is not None:
            r0, c0, vh, vw = view_rect
            y0 = max(0, r0 * self.tile_px)
            x0 = max(0, c0 * self.tile_px)
            y1 = min(canvas.shape[0] - 1, (r0 + vh) * self.tile_px - 1)
            x1 = min(canvas.shape[1] - 1, (c0 + vw) * self.tile_px - 1)
            color = np.array([255, 255, 0], dtype=np.uint8)
            canvas[y0, x0 : x1 + 1] = color
            canvas[y1, x0 : x1 + 1] = color
            canvas[y0 : y1 + 1, x0] = color
            canvas[y0 : y1 + 1, x1] = color
        return canvas

    def render_observation(
        self,
        terrain: np.ndarray,
        agent_pos: tuple[int, int],
        view_size: int,
        agent_facing: str = "down",
    ) -> np.ndarray:
        """Render the agent's local view as a CHW uint8 RGB array.

        ``view_size`` is the side length in tiles (odd; agent centred).
        Cells outside the map render black (OOB).
        Returns shape ``(3, view_size*tile_px, view_size*tile_px)``.
        """
        if view_size % 2 == 0 or view_size < 3:
            raise ValueError(f"view_size={view_size} must be odd and ≥ 3")
        ar, ac = agent_pos
        H, W = terrain.shape
        half = view_size // 2
        r0, c0 = ar - half, ac - half
        canvas = np.zeros((view_size * self.tile_px, view_size * self.tile_px, 3), dtype=np.uint8)
        for vr in range(view_size):
            mr = r0 + vr
            for vc in range(view_size):
                mc = c0 + vc
                if 0 <= mr < H and 0 <= mc < W:
                    tile = int(terrain[mr, mc])
                    fname = _TERRAIN_SPRITES.get(tile)
                    if fname is not None:
                        self._blit(canvas, fname, vr, vc)
                    else:
                        self._fill_color(canvas, (0, 0, 0), vr, vc)
                    if tile == TREE:
                        self._blit(canvas, "tree.png", vr, vc)
                # outside-map → already black from np.zeros
        # Target overlay if the target tile falls inside the view.
        for vr in range(view_size):
            mr = r0 + vr
            if not (0 <= mr < H):
                continue
            for vc in range(view_size):
                mc = c0 + vc
                if 0 <= mc < W and int(terrain[mr, mc]) == TARGET:
                    self._blit(canvas, "diamond.png", vr, vc)
        # agent overlay at view centre
        face_map = {
            "up": "player-up.png",
            "down": "player-down.png",
            "left": "player-left.png",
            "right": "player-right.png",
        }
        self._blit(canvas, face_map.get(agent_facing, "player.png"), view_size // 2, view_size // 2)
        # convert to CHW
        return canvas.transpose(2, 0, 1).copy()


# ---------------------------------------------------- fallback flat renderer

def render_color_grid(terrain: np.ndarray, cell_px: int = 4) -> np.ndarray:
    """Fast no-sprite renderer used by inspection scripts; one colour per tile."""
    H, W = terrain.shape
    colours = TILE_COLORS[np.clip(terrain, 0, len(TILE_COLORS) - 1)]
    out = np.kron(colours, np.ones((cell_px, cell_px, 1), dtype=np.uint8))
    return out.astype(np.uint8)
