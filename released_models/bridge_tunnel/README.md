# bridge_tunnel — released agents

Frozen PPO+GRU policies for the `cogniland.bridge_tunnel` env. Each `*.pt` is a dict
`{"policy": state_dict, "args": {...}, "iteration", "global_step"}`; the matching
`*.yaml` is the **exact, reproducible** trainer config (hyperparameters + seed)
that produced it.

> **Vocabulary (2026-05-31):** bridge_tunnel is **natural-only** with a 9-tile
> vocabulary (`GRASS WATER ROCK WOOD TARGET OOB TREE SAND DIRT`, `NUM_TILES=9`).
> The obsidian + cue tiles and the diagonal/vertical **stripe orientations are
> retired** (they caused phantom lava/diamond decoder artifacts). The old
> `diagonal_cuefollower` / `vertical_cuefollower` and the pre-remap `natural_agent`
> were deleted (invalid under the new vocab).

All agents share the env (natural 32×64, 3-cell centre door `goal_half=1`, edge
forests). The **`_onehot` / categorical** agents additionally share the same
**categorical one-hot observation** (`V×V×9`) — a fair PPO-vs-DreamerV3 comparison
where only the algorithm differs (see `paper/bridge_tunnel.tex`).

| agent | algo / obs | success (held-out grid) | notes |
|------|-----------|------|------|
| `natural_centergoal3.pt` | PPO+GRU, tile-embed | 100% | original released agent |
| `natural_centergoal3_onehot.pt` | PPO+GRU, **one-hot** | 100% | fair-comparison PPO (`obs_encoding: onehot`) |
| `dreamer_natural_categorical/` | DreamerV3 25M, **categorical** | 85% | fair-comparison Dreamer, 1M steps (orbax, **git-LFS**) |

Recipe (all): **3-cell central goal door** + **tree forests biased to the top &
bottom walls** funnel the agent through the obstacle-filled middle; **no entropy
annealing** + entropy 0.045 → a mix of avoid / bridge / tunnel. DreamerV3 logs
~96% during training but ~85% in clean per-episode eval, and is far less
sample-efficient than PPO here (PPO ~100% by 0.2M; Dreamer ~96% by 1M).

The DreamerV3 checkpoint is an orbax PyTree dir (`config.json` + `checkpoints/
step_1000000/`), stored via **git-LFS** (~74 MB). PPO `*.pt` are plain dicts
(`{"policy", "args", ...}`) with a matching reproducible `*.yaml`.

## See it play (pygame demo)

```bash
# defaults: AI plays the released agent on the curated validation maps
python scripts/bridge_tunnel/play_bridge_tunnel.py
```
A start **menu** lets you pick Human/AI and the map (a validation map, or Random).
In-game keys: arrows move, `B` build, `M` mine, `A` toggle AI, `Space` single AI
step, `+/-` speed, `R` new map, `Esc` back to menu. The demo reads the map size
from the checkpoint, so no extra flags are needed for AI play.

## Reproduce

```bash
python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config released_models/bridge_tunnel/natural_centergoal3.yaml \
    --run-name natural_repro --wandb-mode disabled
```
2M-step budget. The diversity comes from the goal/forest geometry + high entropy
with no annealing.

## Evaluate

```bash
python scripts/bridge_tunnel/eval_bridge_tunnel_agent.py  --checkpoint released_models/bridge_tunnel/natural_centergoal3.pt --n-maps 8
python scripts/bridge_tunnel/bridge_tunnel_traj_grid.py   --checkpoint released_models/bridge_tunnel/natural_centergoal3.pt --n-maps 6 --n-traj 200
```

Evaluate / visualise the **DreamerV3 categorical** agent (auto-detects the
categorical encoder from its `config.json`):

```bash
python scripts/bridge_tunnel/viz_dreamer_bridge_tunnel_traj.py    --checkpoint released_models/bridge_tunnel/dreamer_natural_categorical/checkpoints/step_1000000
python scripts/bridge_tunnel/viz_dreamer_bridge_tunnel_imagine.py --checkpoint released_models/bridge_tunnel/dreamer_natural_categorical/checkpoints/step_1000000
```
(`git lfs pull` to fetch its weights after cloning.)

See `docs/bridge_tunnel.md` for the full project guide.
