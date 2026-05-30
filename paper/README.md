# Crafter-in-Cogniland — schematic write-up

`crafter_in_cogniland.tex` is a self-contained, Overleaf-compatible document
(stock packages only, compiles with pdfLaTeX). Upload the whole `paper/`
folder to Overleaf and build `crafter_in_cogniland.tex`.

## Regenerate the figures

Run from the repo root in the `crusoe` conda env:

```bash
conda run -n crusoe python paper/gen_maps.py          # tile icons + size×biome + seed grids
conda run -n crusoe python paper/gen_trajectories.py  # 100-rollout glow grid (uses the 4 PPO checkpoints)
```

Outputs land in `paper/figures/` (and `paper/figures/tiles/`).

- `gen_maps.py` — composites sprite icons for the tile table and renders the
  map-size×biome and 64×64 seed-variation grids.
- `gen_trajectories.py` — loads `checkpoints/ppo_gru_size{32,64,96,128}_*_final.pt`,
  fixes one map per (biome, size), runs 100 batched stochastic rollouts from the
  same spawn/target, and overlays them as thin progress-coloured trajectories on
  a darkened map.
