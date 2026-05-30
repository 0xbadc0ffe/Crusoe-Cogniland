# zebra_nav — project guide

A small, self-contained Crafter-style **navigation POMDP** with a build/mine
decision, plus PPO+GRU training, evaluation, a pygame demo, and released agents.
This doc is the map of *what is where*.

> For the **whole-repo** map (all clusters, dependency diagrams, and a
> keep/legacy/remove inventory), see [`codebase_map.md`](codebase_map.md).

```
src/cogniland/zebra_nav/        the environment package (pure numpy + gymnasium)
  tiles.py                      tile ids, colours, walkability (grass/water/rock/
                                obsidian/wood/cues/target/tree/sand/dirt)
  mapgen.py                     procedural map generation (3 orientations) + MapRecord
  env.py                        ZebraNavEnv: obs, actions, reward, cost-to-go shaping
  _solver.py                    BFS reference solver (used as a solvability test)
  __init__.py                   exports ZebraNavEnv, generate_zebra_map, tiles

scripts/
  train_ppo_zebra.py            PPO+GRU trainer (single file; W&B optional)
  eval_zebra_agent.py           deterministic eval grid + success / thin-side
  zebra_traj_grid.py            overlay N stochastic rollouts (path=blue, mine=yellow, bridge=red)
  play_zebra.py                 pygame demo — human or AI, mining/bridge animations
  make_zebra_val_maps.py        curate the fixed validation/demo map set
  zebra_sweep.yaml              W&B sweep: diagonal/vertical cue-following
  zebra_natural_sweep.yaml      W&B sweep: natural maps
  launch_zebra_sweep.sh         launch N parallel sweep agents (default ~9 runs)

tests/test_zebra_nav.py         env + mapgen contract tests (run: pytest tests/test_zebra_nav.py)

models/zebra_nav/               RELEASED agents (weights + reproducible *.yaml configs) — see its README
data/zebra_nav/                 val_maps.pkl (the curated demo == validation maps) + preview
```

## The task

Navigate from a spawn to a goal across obstacles you can **bridge** (PLACE turns
WATER→WOOD) or **mine** (MINE turns ROCK→GRASS), or walk around. Observation is
an egocentric `view_size × view_size` crop of tile ids + a scalar vector
(`facing` one-hot + step fraction) — so it's partially observed.

Three map **orientations** (`generate_zebra_map(orientation=...)`, env
`--orientation`):

- **diagonal** — BL→TR; diagonal obsidian walls, each with a WATER and a ROCK
  crossing window; a cue says which side is **thinner**. Goal = read cue, cross thin.
- **vertical** — midL→midR (32×64); full-height walls, water-top/rock-bottom with
  an obsidian divider; same cue→thin-side task.
- **natural** — midL→right wall (32×64); open procedural terrain: lakes (bridge),
  mountains/ridges (mine), impassable **tree** patches (walk around), with cosmetic
  sand/dirt fringes. Goal = a **central door** on the right wall. Behaviour
  (cross-vs-detour) emerges from minimising episode length.

Actions: `Discrete(6)` — up / down / left / right (a move also sets `facing`) +
PLACE (bridge water in front) + MINE (mine rock in front).

## Reward

`slack` per action + `reach_bonus` on the goal + PBRS shaping with potential
`φ = −ctg`, where `ctg` is a **min-action** cost-to-go (entering water/rock costs
2 = build+move, grass 1, obsidian/tree impassable). With `build_cost=0` this makes
the agent minimise total actions; `build_cost>0` makes crossings costlier so it
prefers detours. The cue-following stripe agents add a per-build cost to favour
the thin side.

## Quickstart

```bash
conda activate crusoe && pip install -e .          # one-time
pytest tests/test_zebra_nav.py                     # sanity

# play a released agent on the validation maps
python scripts/play_zebra.py --checkpoint models/zebra_nav/natural_agent.pt \
    --maps data/zebra_nav/val_maps.pkl

# train from scratch (reproduce the natural agent)
python scripts/train_ppo_zebra.py --config models/zebra_nav/natural_agent.yaml \
    --run-name natural_repro --wandb-mode disabled

# regenerate the validation/demo maps
python scripts/make_zebra_val_maps.py --orientation natural --n 16
```

Training writes scratch checkpoints to `checkpoints/` (git-ignored). The curated,
committed agents live in `models/zebra_nav/`.
