# Paper write-ups

Two self-contained, Overleaf-compatible documents (stock packages only, compile
with pdfLaTeX). Upload the whole `paper/` folder to Overleaf and build either.

- **`bridge_tunnel.tex`** — the **current** project: the `bridge_tunnel` POMDP (bridge /
  mine / detour strategy choice), the natural-maps task + reward, the trained PPO
  and DreamerV3 agents, rollout examples, DreamerV3 imagination, and the
  mechanistic-interpretability plan (probe + steer belief/strategy subspaces;
  suppress tunneling in activation space). Figures in `figures/bridge_tunnel/`.
- **`crafter_in_cogniland.tex`** — the original build-commitment (raft vs harness)
  navigation env with slip mechanics. Figures in `figures/`.

## Regenerate the `bridge_tunnel` figures

From the repo root (venv activated, `pip install -e .`), then `cd paper && latexmk -pdf bridge_tunnel.tex`:

```bash
python scripts/bridge_tunnel/make_bridge_tunnel_val_maps.py --orientation natural --n 16
cp data/bridge_tunnel/val_maps_preview.png paper/figures/bridge_tunnel/maps.png
python scripts/bridge_tunnel/bridge_tunnel_traj_grid.py        --checkpoint released_models/bridge_tunnel/natural_centergoal3.pt --out paper/figures/bridge_tunnel/ppo_traj.png
python scripts/bridge_tunnel/viz_dreamer_bridge_tunnel_traj.py --checkpoint runs/dreamer_natural_v2/checkpoints/step_1000000 --out paper/figures/bridge_tunnel/dreamer_traj.png
python scripts/bridge_tunnel/bridge_tunnel_strategy_examples.py --checkpoint released_models/bridge_tunnel/natural_centergoal3.pt --out paper/figures/bridge_tunnel/strategy_examples.png
python scripts/bridge_tunnel/viz_dreamer_bridge_tunnel_imagine.py --checkpoint runs/dreamer_natural_v2/checkpoints/step_1000000 --render sprites
cp videos/dreamer_imagine/imagine_strip_seed10001.png paper/figures/bridge_tunnel/dreamer_imagine_strip.png
```

---

## Crafter-in-Cogniland — schematic write-up

`crafter_in_cogniland.tex` is a self-contained, Overleaf-compatible document
(stock packages only, compiles with pdfLaTeX).

Its figures are committed under `paper/figures/` (and `paper/figures/tiles/`).
The original generator scripts (`gen_maps.py`, `gen_trajectories.py`) were removed
when the repo was refocused on `bridge_tunnel`; re-export from git history if needed.
