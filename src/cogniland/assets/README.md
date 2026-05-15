# Cogniland Nav — assets

The PNGs in `sprites/` are copies of Crafter sprites used as visual tiles in the
`cogniland.nav` environment. Each PNG is 16×16, indexed-color, nearest-neighbor
scaled at render time.

| File | Use |
|------|-----|
| `grass.png`       | GRASS tile |
| `path.png`        | DIRT tile |
| `sand.png`        | SAND tile |
| `water.png`       | WATER tile |
| `stone.png`       | ROCK tile |
| `diamond.png`     | TARGET overlay |
| `player.png`      | agent overlay (default) |
| `player-up.png`   | agent overlay facing up |
| `player-down.png` | agent overlay facing down |
| `player-left.png` | agent overlay facing left |
| `player-right.png`| agent overlay facing right |

## Attribution

These sprites are taken from
[Crafter](https://github.com/danijar/crafter) by Danijar Hafner, released under
the MIT License. See `crafter-main/LICENSE` in this repository for the
original license text. The Cogniland environment imports no code from Crafter;
only the static image assets are reused.
