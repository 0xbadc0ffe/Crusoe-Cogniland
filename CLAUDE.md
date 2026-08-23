# Crusoe-Cogniland — Architecture & Developer Guide

One POMDP memory task (`bridge_tunnel` fork_wall), three agents trained on the
identical env + reward + fixed 6k-map dataset — **PPO+GRU** (PyTorch,
model-free), **DreamerV3** (`r2dreamer_model`, RSSM world model), **STORM**
(`STORM_model`, transformer world model) — built as a substrate for mechanistic
interpretability (belief/skill probing + steering). The curated checkpoints and
per-agent reproduction commands live in `final_models/` (read its
`ENVIRONMENT.md` first); earlier released agents in
`configs/bridge_tunnel/REGISTRY.md`.

## Layout

```
final_models/                   ★ the three checkpoints + docs + repro commands
src/cogniland/
  bridge_tunnel/                THE env: ONE package, two variants
    tiles.py ctg.py mapgen.py   variant="bt" (base) | "btc" (commitment + map
    env.py policy.py _solver.py  categories). fork_wall task = btc maps,
    map_pool.py                  commit=False, fork_wall=True. Discrete(6).
    jax/                        pure-JAX port (Gymnax-style), bit-identical
  memory_env/                   MiniGrid MemoryEnv fork (T-maze; kept secondary)
  assets/sprites/               rendering sprites
purejaxwm/                      in-tree DreamerV3 lib (RSSM, TwoHot, LaProp, …)
r2dreamer_model/                Dreamer pipeline (conda env `r2dreamer`)
STORM_model/                    STORM pipeline (own .venv; agent `storm2`)
scripts/bridge_tunnel/          PPO train/eval/viz + slurm/ launchers
scripts/memory_env/             memory_env training + analysis
scripts/mechinterp/             activation datasets, probing, steering kits
configs/bridge_tunnel/          PPO/experiment configs + REGISTRY.md
released_models/                earlier released agents (git-LFS)
data/bridge_tunnel/forkwall6k/  the shared fixed dataset (train/test)
tests/                          env contract + JAX↔PyTorch parity + purejaxwm
```

## The fork_wall task (all three agents, byte-identical)

Map 32×64, category visible early (lakes/rocky/balanced terrain), then a
pure-grass memory corridor (`mem_gap=16`), a wall with a 3-cell passage, and a
top/bottom door pair. Only the category-matching door is rewarded
(rocky→top, lakes→bottom, balanced→either). Reward: `-0.01` slack/step,
`+0.015·(ctg_prev − ctg_curr)` PBRS toward the correct door (`shaping_gamma=1`),
`+3.0` at the correct door; wrong door ends the episode unrewarded.
Obs `{minimap: (21,21) int8, scalars: (5,)}`; `max_steps=800`; γ=0.99.
Canonical kwargs: `configs/bridge_tunnel/btc_ppo_forkwall_plain_solved.yaml`,
mirrored byte-for-byte in `r2dreamer_model/envs/bridge_tunnel.py` and
`STORM_model/cl/environments/bridge_tunnel.py` (`FORKWALL_KWARGS`).

## How to run

```bash
pip install -e .                              # env + PPO + purejaxwm deps
pytest tests/                                 # 81 tests: contract + parity + units

python scripts/bridge_tunnel/make_forkwall_dataset.py   # regen forkwall6k (deterministic)

# PPO      (conda `crusoe`)      final_models/ppo/README.md
# Dreamer  (conda `r2dreamer`)   final_models/dreamer/README.md
# STORM    (STORM_model/.venv)   final_models/storm/README.md
```

## Evaluation convention (important)

Report the **TRUE door metric** — final cell ∈ correct-door set — never
`return > 0`: fast wrong-door episodes collect more PBRS than slack, so the
return proxy inflates success by ~6–13pp. PPO and STORM evaluate with sampled
actions (greedy STORM deadlocks into timeouts), Dreamer deterministically.
STORM evaluator: `STORM_model/scripts/true_eval_w.py --sampled --env-context 128`.

## Hard-won training lore (per agent)

* All agents face a **constant-door local optimum** (~67%: balanced + one
  category). Escape is seed-dependent and door-binding is metastable —
  archive checkpoints and select best on held-out data.
* **PPO**: plain reward + default entropy collapses to constant-door; the
  escape recipe is high starting entropy 0.15 **annealed** to 0 (2/3 seeds).
* **Dreamer/STORM**: the ~75-step evidence→door dependency must fit the
  training window: `batch_length ≥ 128` (r2dreamer has no replay_context;
  STORM's transformer attends only within the window). STORM additionally
  needs act-time context `env_context=128` and entropy 0.01 (0.03 entrenches
  the basin).
* STORM's `storm2` agent replaced a memoryless first implementation (1-token
  transformer calls, actor saw only z_t) — do not resurrect that pattern.

## Design invariants

- The env never imports torch; the PyTorch and pure-JAX bridge_tunnel envs are
  proven bit-for-bit equivalent (`tests/test_bridge_tunnel*parity.py`) — if you
  change env rules, keep `jax/` in parity and run those tests.
- The three pipelines never import each other; they share only
  `src/cogniland/bridge_tunnel` and `data/bridge_tunnel/forkwall6k`.
- Map generation is numpy + deterministic by seed; all models train from the
  same pickled `forkwall6k/train.pkl` pool and are tested on `test.pkl`.
- Checkpoints in `final_models/` are load-only artifacts (no optimizer state).

## Mech-interp workflow

`scripts/mechinterp/` builds activation datasets and steering kits from the
released agents (see `docs/codebase_map.md` and REGISTRY.md). Interp assets
(`activation_datasets/`, `artifacts/`, `outputs/*report*`) are kept as-is.

## Tests

```bash
pytest tests/                  # env contract + parity + purejaxwm units
```
