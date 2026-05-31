# zebra_nav — released agents

Frozen PPO+GRU policies for the `cogniland.zebra_nav` env. Each `*.pt` is a dict
`{"policy": state_dict, "args": {...}, "iteration", "global_step"}`; the matching
`*.yaml` is the **exact, reproducible** trainer config (hyperparameters + seed)
that produced it.

> **Vocabulary (2026-05-31):** zebra_nav is **natural-only** with a 9-tile
> vocabulary (`GRASS WATER ROCK WOOD TARGET OOB TREE SAND DIRT`, `NUM_TILES=9`).
> The obsidian + cue tiles and the diagonal/vertical **stripe orientations are
> retired** (they caused phantom lava/diamond decoder artifacts). The old
> `diagonal_cuefollower` / `vertical_cuefollower` and the pre-remap `natural_agent`
> were deleted (invalid under the new vocab).

| file | map type | what it does | success | episode len |
|------|----------|--------------|---------|-------------|
| `natural_centergoal3.pt` | natural (32×64, lakes/mountains, edge forests) | routes through the **central corridor** to a 3-cell centre door, crossing obstacles (mine/bridge) and detouring around the larger ones | 100% | ~95 |

Recipe: a **3-cell central goal door** (`goal_half=1`) + **tree forests biased
heavily to the top & bottom walls** funnel the agent through the obstacle-filled
middle; **no entropy annealing** + entropy 0.045 keeps the stochastic policy
spread out → a mix of avoid / bridge / tunnel.

## See it play (pygame demo)

```bash
# defaults: AI plays the released agent on the curated validation maps
python scripts/play_zebra.py
```
A start **menu** lets you pick Human/AI and the map (a validation map, or Random).
In-game keys: arrows move, `B` build, `M` mine, `A` toggle AI, `Space` single AI
step, `+/-` speed, `R` new map, `Esc` back to menu. The demo reads the map size
from the checkpoint, so no extra flags are needed for AI play.

## Reproduce

```bash
python scripts/train_ppo_zebra.py --config models/zebra_nav/natural_centergoal3.yaml \
    --run-name natural_repro --wandb-mode disabled
```
2M-step budget. The diversity comes from the goal/forest geometry + high entropy
with no annealing.

## Evaluate

```bash
python scripts/eval_zebra_agent.py  --checkpoint models/zebra_nav/natural_centergoal3.pt --n-maps 8
python scripts/zebra_traj_grid.py   --checkpoint models/zebra_nav/natural_centergoal3.pt --n-maps 6 --n-traj 200
```

A DreamerV3 agent on the same task lives under `runs/dreamer_natural_*/` (not
committed); render its dreams with `scripts/viz_dreamer_zebra_imagine.py`.

See `docs/zebra_nav.md` for the full project guide.
