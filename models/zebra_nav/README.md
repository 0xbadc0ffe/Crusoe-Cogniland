# zebra_nav — released agents

Frozen PPO+GRU policies for the `cogniland.zebra_nav` env. Each `*.pt` is a dict
`{"policy": state_dict, "args": {...}, "iteration", "global_step"}`; the matching
`*.yaml` is the **exact, reproducible** trainer config (hyperparameters + seed)
that produced it.

> **Vocabulary change (2026-05-31):** zebra_nav is now **natural-only** with a
> 9-tile vocabulary (`GRASS WATER ROCK WOOD TARGET OOB TREE SAND DIRT`). The
> obsidian wall + cue tiles and the diagonal/vertical **stripe orientations are
> retired** (they caused phantom lava/diamond decoder artifacts and a tile-id
> remap). The old `diagonal_cuefollower` / `vertical_cuefollower` agents were
> deleted — they are invalid under the new vocab. **The `natural_*` checkpoints
> below are also stale** (trained against the old 12-tile vocab / tile ids) and
> are being retrained; expect them to be replaced.

| file | map type | what it does | success | episode len |
|------|----------|--------------|---------|-------------|
| `natural_agent.pt` | natural (32×64, lakes/mountains/trees) | **diverse** routes through the middle to the goal door, crossing some obstacles (mine/bridge) and going around others | 100% (stale vocab) | ~95 |

`natural_agent` is the diverse one: a **central goal door** keeps it traversing
the centre (no edge-hugging), and **no entropy annealing** + entropy 0.045 keeps
the stochastic policy spread out → many distinct paths. Trees now cluster heavily
along the top & bottom walls so wall-hugging to the door is blocked by forest.

## See it play (pygame demo)

```bash
# defaults: AI plays natural_agent on the curated validation maps
python scripts/play_zebra.py
```
A start **menu** lets you pick Human/AI and the map (a validation map, or Random).
In-game keys: arrows move, `B` build, `M` mine, `A` toggle AI, `Space` single AI
step, `+/-` speed, `R` new map, `Esc` back to menu. The demo reads the map size
from the checkpoint, so no extra flags are needed for AI play.

## Reproduce

```bash
python scripts/train_ppo_zebra.py --config models/zebra_nav/natural_agent.yaml \
    --run-name natural_repro --wandb-mode disabled
```
All use the agreed 2M-step budget. The natural agent's diversity comes from high
entropy + no annealing.

## Evaluate

```bash
python scripts/eval_zebra_agent.py  --checkpoint models/zebra_nav/natural_agent.pt --n-maps 8
python scripts/zebra_traj_grid.py   --checkpoint models/zebra_nav/natural_agent.pt --n-maps 6 --n-traj 200
```

See `docs/zebra_nav.md` for the full project guide.
