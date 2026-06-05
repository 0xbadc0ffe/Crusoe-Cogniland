# Codebase map — what is where

Navigation guide for the repo after the unification + reorg.

## Top-level layout

```
src/cogniland/          the library
  bridge_tunnel/        the env. ONE package, two variants (variant="bt"|"btc")
  assets/sprites/       Crafter PNG sprites used by the pygame demo / imagination viz
purejaxwm/              vendored DreamerV3 (RSSM, behavior, TwoHot, LaProp, RetNorm)
scripts/                grouped by purpose (see below)
configs/bridge_tunnel/  experiment configs + REGISTRY.md
tests/                  env contract + JAX↔PyTorch parity + algorithm-lib tests
released_models/        frozen released agents (+ as-trained .yaml). git-LFS for orbax
data/                   procedural map datasets (val maps; *_jax train sets, regenerable)
activation_datasets/    mech-interp bundles (self-contained; shipped to colleagues)  [gitignored]
outputs/                ALL generated artifacts: ppo_checkpoints/ dreamer_runs/ logs/
                        rollouts/ previews/ videos/  [gitignored]
paper/ docs/            write-up + guides
wandb/                  W&B local logs  [gitignored]
```

## bridge_tunnel — the active env (one package, two variants)

`BridgeTunnelEnv(variant="bt"|"btc")` — `bt` = base (place/mine always active);
`btc` = implicit build/mine commitment (first successful build/mine locks the
skill) + 3 labelled map categories (balanced/lakes/rocky). Both are `Discrete(6)`.

```
src/cogniland/bridge_tunnel/
  tiles.py      9-tile vocab + palette/walkability
  ctg.py        min-action cost-to-go (PBRS potential); commit-aware 3-field stack for btc
  mapgen.py     generate_map(variant=...): bt single-map | btc categories + winnability + make_split
  env.py        BridgeTunnelEnv(variant=...)  (+ BridgeTunnelCommitEnv alias)
  policy.py     PPOGRUPolicy (single source of truth; from_checkpoint helper)
  _solver.py    BFS reference solver (both variants; used by tests)
  jax/          pure-JAX port (Gymnax-style) — EnvParams.commit is a STATIC flag
    constants state dynamics render env maps
```
The PyTorch and JAX envs are proven **bit-for-bit equivalent** for both variants
(`tests/test_bridge_tunnel*parity.py`).

## scripts/ (grouped)

| dir | contents |
|---|---|
| `scripts/bridge_tunnel/` | `train_ppo_bridge_tunnel.py` / `dreamerv3_bridge_tunnel.py` (both `--variant bt|btc`), `eval_bridge_tunnel_agent.py`, `eval_bridge_tunnel_commit_{ppo,dreamer}.py`, `bridge_tunnel_traj_grid.py`, `bridge_tunnel_strategy_examples.py`, `viz_dreamer_bridge_tunnel_{traj,imagine}.py`, `play_bridge_tunnel.py`, `make_bridge_tunnel_val_maps.py`, sweeps |
| `scripts/mechinterp/` | `build_activation_dataset.py` (`--env bridge_tunnel[_commit]`), `decode_dataset.py` (standalone), `replay_trajectory.py` (replay + gru_h steering), `steering_kit/` (shipped into bundles) |
| `scripts/figures/` | architecture / training-curve drawing scripts |

## Where do I start

| I want to… | command |
|---|---|
| train PPO | `scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config configs/bridge_tunnel/btc_ppo_onehot.yaml` |
| train DreamerV3 | `scripts/bridge_tunnel/dreamerv3_bridge_tunnel.py --variant btc --size 25M --decoder categorical` |
| see an agent play | `scripts/bridge_tunnel/play_bridge_tunnel.py` |
| commit matrix + grids | `scripts/bridge_tunnel/eval_bridge_tunnel_commit_ppo.py --checkpoint …` |
| build an activation dataset | `scripts/mechinterp/build_activation_dataset.py --env bridge_tunnel_commit --checkpoint …` |
| decode a dataset frame/traj | `python activation_datasets/<name>/decode_dataset.py --row N` (no repo needed) |
| change env rules | `src/cogniland/bridge_tunnel/env.py` (+ keep `jax/` in parity — run the parity tests) |
| released agents + how to reproduce | `configs/bridge_tunnel/REGISTRY.md` |

See also `docs/bridge_tunnel.md` (task guide) and `CLAUDE.md` (architecture + invariants).
