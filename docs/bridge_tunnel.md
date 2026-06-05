# bridge_tunnel — project guide

A small, self-contained Crafter-style **navigation POMDP** with a build/mine
decision, plus PPO+GRU training, evaluation, a pygame demo, and released agents.
This doc is the map of *what is where*.

> For the **whole-repo** map (all clusters, dependency diagrams, and a
> keep/legacy/remove inventory), see [`codebase_map.md`](codebase_map.md).

> **Natural-only (2026-05-31).** The env is now a single **natural** orientation
> with a contiguous **9-tile** vocabulary
> (`GRASS WATER ROCK WOOD TARGET OOB TREE SAND DIRT`). The obsidian wall + cue
> tiles and the diagonal/vertical **stripe orientations are retired** — they
> caused phantom lava/diamond decoder artifacts. **TREE is the only inviolable
> tile.** Old stripe agents are deleted and the `natural_*` checkpoints are
> stale under the new ids (retraining in progress).

```
src/cogniland/bridge_tunnel/        the environment package (pure numpy + gymnasium)
  tiles.py                      9 tile ids, colours, walkability
                                (grass/water/rock/wood/target/oob/tree/sand/dirt)
  mapgen.py                     procedural natural-map generation + MapRecord
  env.py                        BridgeTunnelEnv: obs, actions, reward, cost-to-go shaping
  _solver.py                    BFS reference solver (used as a solvability test)
  __init__.py                   exports BridgeTunnelEnv, generate_bridge_tunnel_map, tiles
src/cogniland/bridge_tunnel_jax/    pure-JAX parity port (DreamerV3); mirrors tiles/dynamics

scripts/
  train_ppo_bridge_tunnel.py            PPO+GRU trainer (single file; W&B optional)
  eval_bridge_tunnel_agent.py           deterministic eval grid + success (thin-side retired → 0)
  bridge_tunnel_traj_grid.py            overlay N stochastic rollouts (path=blue, mine=yellow, bridge=red)
  play_bridge_tunnel.py                 pygame demo — human or AI, mining/bridge animations
  make_bridge_tunnel_val_maps.py        curate the fixed validation/demo map set (natural)
  bridge_tunnel_natural_sweep.yaml      W&B sweep: natural maps
  bridge_tunnel_sweep.yaml              DEPRECATED stripe sweep (will error — kept for provenance)
  launch_bridge_tunnel_sweep.sh         launch N parallel sweep agents (default ~9 runs)

tests/test_bridge_tunnel.py         env + mapgen contract tests (run: pytest tests/test_bridge_tunnel.py)

released_models/bridge_tunnel/               RELEASED agents (weights + reproducible *.yaml configs) — see its README
data/bridge_tunnel/                 val_maps.pkl (the curated demo == validation maps) + preview
```

## The task

Navigate from a spawn to a goal across obstacles you can **bridge** (PLACE turns
WATER→WOOD) or **mine** (MINE turns ROCK→GRASS), or walk around. Observation is
an egocentric `view_size × view_size` crop of tile ids + a scalar vector
(`facing` one-hot + step fraction) — so it's partially observed.

One map type, **natural** (`generate_bridge_tunnel_map(orientation="natural")`, env
`--orientation natural`; any other value raises):

- **natural** — midL→right wall (32×64); open procedural terrain: lakes (bridge),
  mountains/ridges (mine), impassable **tree** patches (walk around), with cosmetic
  sand/dirt fringes. Trees cluster heavily along the **top & bottom walls** so naive
  wall-hugging to the door is blocked by forest. Goal = a **central door** on the
  right wall (`goal_half=1` ⇒ 3-cell door; `None` ⇒ whole wall). Behaviour
  (cross-vs-detour) emerges from minimising episode length.

Actions: `Discrete(6)` — up / down / left / right (a move also sets `facing`) +
PLACE (bridge water in front) + MINE (mine rock in front).

## Reward

`slack` per action + `reach_bonus` on the goal + PBRS shaping with potential
`φ = −ctg`, where `ctg` is a **min-action** cost-to-go (entering water/rock costs
2 = build+move, grass 1, tree impassable). With `build_cost=0` this makes the
agent minimise total actions; `build_cost>0` makes crossings costlier so it
prefers detours.

## Quickstart

```bash
conda activate crusoe && pip install -e .          # one-time
pytest tests/test_bridge_tunnel.py                     # sanity

# play a released agent on the validation maps
python scripts/bridge_tunnel/play_bridge_tunnel.py --checkpoint released_models/bridge_tunnel/natural_agent.pt

# train from scratch (reproduce the natural agent)
python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config released_models/bridge_tunnel/natural_agent.yaml \
    --run-name natural_repro --wandb-mode disabled

# regenerate the validation/demo maps
python scripts/bridge_tunnel/make_bridge_tunnel_val_maps.py --orientation natural --n 16
```

Training writes scratch checkpoints to `checkpoints/` (git-ignored). The curated,
committed agents live in `released_models/bridge_tunnel/`.
