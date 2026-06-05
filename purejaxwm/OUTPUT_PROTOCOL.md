# Output protocol (v0)

This document pins the on-disk contract for experiment outputs, and is deliberately small so that cross-algorithm comparability stays cheap. DESIGN.md Principle 7: "standardize outputs, not internal control flow." Everything else in the repository can churn; this should not.

## Run directory layout

Every training script writes to a single run directory with this structure:

```
runs/<run_id>/
  config.yaml           # the fully resolved Hydra config
  metrics.jsonl         # one flat record per (step, metric) logged
  summary.json          # final aggregated metrics (final return, loss components, wall time)
  checkpoints/          # orbax pytree — final params only in v0
  hlo_dump.txt          # compiled HLO of `train`, for Principle 8 inspection
```

`<run_id>` is a short human-readable slug: `<algo>_<env>_<timestamp>_<shortsha>`.

## `metrics.jsonl` schema

One JSON object per line. Fields are flat; no nested structures. The fixed schema is:

```json
{
  "step": 12345,
  "metric_name": "loss/rec",
  "value": 1.234,
  "seed": 0,
  "env": "craftax_classic_pixels",
  "algo": "dreamerv3",
  "run_id": "dreamerv3_craftax_classic_pixels_20260418_abcd12",
  "wall_time": 1745000000.123
}
```

- `step`: int, env-step count at time of emission
- `metric_name`: string, slash-separated namespace (e.g. `loss/rec`, `loss/dyn`, `return/mean`, `return/std`)
- `value`: float, a single scalar — never an array
- `seed`: int, the seed for this logical run (see multi-seed note)
- `env`: string, normalized environment id
- `algo`: string, algorithm id
- `run_id`: string, stable per run directory
- `wall_time`: float, unix seconds at emission

Scripts that need to emit a vector metric (e.g. histogram-like data) must decompose it into multiple rows with distinct `metric_name` suffixes.

## Multi-seed convention

When a script uses `jax.vmap(train, in_axes=(0, None))` over seed keys (the expected pattern per DESIGN Principle 8), each seed still emits its own logical `metrics.jsonl` with its `seed` field set. All seeds in one vmapped call share a `run_id`. The recommended pattern:

```python
stacked_metrics = jax.vmap(train, in_axes=(0, None))(seeds, cfg)
# stacked_metrics.field[seed_idx, step_idx, ...]
for seed_idx, seed in enumerate(seeds_as_ints):
    write_metrics_jsonl(
        run_dir / f"seed_{seed_idx}" / "metrics.jsonl",
        stacked_metrics,
        seed_idx, seed,
    )
```

WandB logging happens **outside** the vmap, after `train(...)` returns, one stream per seed. `io_callback` inside the scan is discouraged in v0 because it fires per-vmapped-shard and interacts poorly with WandB step counters.

## Hyperparameter sweeps

Sweeps happen at the SLURM-array level: one vmapped-multi-seed run per HP configuration. Do not vmap over HPs inside `train` — that conflates seed variance with HP variance and makes per-HP curves unrecoverable.

## `summary.json`

A single JSON object at the end of training:

```json
{
  "run_id": "...",
  "algo": "dreamerv3",
  "env": "craftax_classic_pixels",
  "num_seeds": 3,
  "total_env_steps": 1000000,
  "wall_time_s": 3600.0,
  "final": {
    "return/mean": 5.2,
    "return/std": 1.1,
    "loss/rec": 0.15,
    "loss/dyn": 2.3
  }
}
```

## Checkpoints

`checkpoints/` holds one Orbax pytree checkpoint, named by training step (e.g. `step_1000000/`). Only the final checkpoint is saved in v0. Contents are algorithm-specific (DreamerV3 saves `train_state.params` + `slow_critic_params` + `normalizers`; PPO-RNN saves `train_state.params` only).

Resumption from checkpoints is **not** a v0 feature.

## What is explicitly not part of the protocol

- Internal staging artifacts (e.g. IRIS tokenizer pretrain outputs) — private to each algorithm folder.
- Algorithm-specific checkpoint pytree layouts beyond the top-level structure.
- Any log format other than `metrics.jsonl` (no TensorBoard event files, no pickle dumps).

Benchmark-specific reductions (AUC, normalized return, transfer metrics) are computed by offline reducers that consume these run directories — not by the training script.
