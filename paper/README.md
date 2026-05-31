# Paper write-ups

Two self-contained, Overleaf-compatible documents (stock packages only, compile
with pdfLaTeX). Upload the whole `paper/` folder to Overleaf and build either.

- **`zebra_nav.tex`** — the **current** project: the `zebra_nav` POMDP (bridge /
  mine / detour strategy choice), the natural-maps task + reward, the trained PPO
  and DreamerV3 agents, rollout examples, DreamerV3 imagination, and the
  mechanistic-interpretability plan (probe + steer belief/strategy subspaces;
  suppress tunneling in activation space). Figures in `figures/zebra_nav/`.
- **`crafter_in_cogniland.tex`** — the original build-commitment (raft vs harness)
  navigation env with slip mechanics. Figures in `figures/`.

## Regenerate the `zebra_nav` figures

From the repo root in the `crusoe` conda env, then `cd paper && latexmk -pdf zebra_nav.tex`:

```bash
python scripts/make_zebra_val_maps.py --orientation natural --n 16
cp data/zebra_nav/val_maps_preview.png paper/figures/zebra_nav/maps.png
python scripts/zebra_traj_grid.py        --checkpoint models/zebra_nav/natural_centergoal3.pt --out paper/figures/zebra_nav/ppo_traj.png
python scripts/viz_dreamer_zebra_traj.py --checkpoint runs/dreamer_natural_v2/checkpoints/step_1000000 --out paper/figures/zebra_nav/dreamer_traj.png
python scripts/zebra_strategy_examples.py --checkpoint models/zebra_nav/natural_centergoal3.pt --out paper/figures/zebra_nav/strategy_examples.png
python scripts/viz_dreamer_zebra_imagine.py --checkpoint runs/dreamer_natural_v2/checkpoints/step_1000000 --render sprites
cp videos/dreamer_imagine/imagine_strip_seed10001.png paper/figures/zebra_nav/dreamer_imagine_strip.png
```

---

## Crafter-in-Cogniland — schematic write-up

`crafter_in_cogniland.tex` is a self-contained, Overleaf-compatible document
(stock packages only, compiles with pdfLaTeX).

## Regenerate its figures

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
