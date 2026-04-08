# Known Issues and Cleanup Targets

Bugs, inconsistencies, and architectural debt identified in the codebase. Organized by priority.

---

## Eval / test logging bugs

### 1. `final_resources` reads from post-reset state

**Location:** `runner.py:263`

```python
final_resources = eval_env.state.resources  # BUG: reads AFTER auto-reset
```

For episodes that finished, the auto-reset in `wrappers.py:86` has already overwritten `eval_env.state` with the new episode's initial state. So `final_resources` reflects `init_resources` (100.0) for finished episodes, not the actual resources at termination.

**Contrast:** `final_hp` is captured correctly at `runner.py:236` via the `new_finalized` check *before* auto-reset clears the state. `final_resources` needs the same treatment.

**Impact:** `final_resources_mean` and related metrics in WandB are inflated for finished episodes.

### 2. Val eval env is not re-seeded between evals

**Location:** `ppo.py:298-305`

The val eval env is created once with `reset(seed=eval_seed)` and cached as `self.eval_env`. When `_run_eval()` runs at each eval interval, `EvalRunner.run()` calls `eval_env.reset()` again — but this time *without* a seed, so spawn/target positions are random.

This means each val eval runs on different spawn/target pairs (though the same set of maps, since `world_maps` are fixed). Val metrics across training are not directly comparable — variance comes from both policy improvement and spawn/target randomness.

**Fix options:** Either re-seed at each eval, or accept that val is noisy and rely on test (which does get a fresh seeded env each time at `ppo.py:617`).

### 3. Recurrent eval doesn't reset hidden state on episode auto-reset

**Location:** `recurrent_ppo.py:658-664`

The closure-based recurrent policy carries `h_det[0]` across all steps. But `EvalRunner.run()` never signals to the policy that an episode just reset (the runner doesn't know about hidden state). Since eval runs episodes to completion without auto-reset (each episode runs independently), this is actually fine for the standard eval path — episodes don't auto-reset mid-evaluation because all episodes start fresh.

However, if `n_episodes > eval_env.num_envs` and the runner were to reuse envs (it currently doesn't), the hidden state would bleed across episodes. Currently not a bug, but fragile.

### 4. Test eval seed double-offset

**Location:** `ppo.py:622`

```python
test_env.reset(seed=self._eval_seed + 1000)
```

`self._eval_seed` is already `base_seed + eval_seed_offset` (default +1000). So test uses `seed + 2000`. This is not a bug per se (test maps are still held-out), but the naming `eval_seed_offset` is misleading — the actual test offset from the base seed is `2 * eval_seed_offset`.

---

## Pipeline architecture debt

### 5. Duplicated training loop code across 3 model files

**Locations:** `ppo.py:224`, `recurrent_ppo.py:310`, `drc.py` (similar)

The following logic is copy-pasted in all three files:
- Curriculum setup and stage transitions (~20 lines)
- MapDataset loading (~15 lines)
- Eval env caching (~15 lines)
- Checkpoint directory setup and best-model tracking (~20 lines)
- `_run_eval()` orchestration (~90 lines)
- `_run_behavioral_eval()` (~70 lines)
- End-of-training finalization (test eval, artifact upload, summary) (~30 lines)

**Total:** ~260 lines duplicated 3x. A change to the eval pipeline (e.g., fixing the `final_resources` bug) must be applied in 3 places.

**Refactor target:** Extract a `Trainer` base class or a `TrainingLoop` helper that owns curriculum, checkpointing, eval orchestration. Each model file then only defines the network architecture and the algorithm-specific update step.

### 6. Hardcoded terrain names in logging.py

**Location:** `logging.py:13`

```python
TERRAIN_NAMES = ["ocean", "deep_water", "water", "beach", "sandy", "grassland", "forest", "rocky", "mountains"]
```

This list is hardcoded and must match the terrain order in `default.yaml`. If someone changes the terrain list in YAML, the WandB terrain distribution chart silently mislabels terrains.

Similarly, `_TERRAIN_ORDER` at `logging.py:233` hardcodes terrain colors for the chart.

Both should read from `CompiledTerrainData.terrain_names` instead.

### 7. `_make_run_name` hardcodes "sweep_reward"

**Location:** `logging.py:53-56`

```python
def _make_run_name(cfg) -> str:
    model = cfg.models.name
    env_mode = "sweep_reward"     # hardcoded artifact from a previous sweep
    return f"{model}_{env_mode}"
```

Every run gets named `ppo_sweep_reward` regardless of context. Same issue in `_make_group_name` at `logging.py:59`.

---

## Sweep / experiment pipeline

### 8. SLURM sweep is a manual array job with hardcoded grid

**Location:** `scripts/slurm/sweep_slurm.sh`

The sweep grid is defined as a bash array: `LP_VALUES=(0.02 0.05 0.08 0.15 0.30 1.00)`. Adding a parameter dimension or changing the grid requires editing the shell script. No early stopping, no Bayesian optimization, no automatic analysis.

### 9. Local grid search has no failure recovery

**Location:** `scripts/run_grid_search.py`

Uses `subprocess.Popen` to spawn training processes. If a process crashes, the script notes it and moves on, but there's no retry logic, no partial result collection, and no way to resume a failed grid search.

### 10. No wandb sweeps integration

The codebase is set up for Hydra CLI overrides, which maps cleanly onto wandb sweeps. A sweep config YAML + `wandb agent` would replace both the SLURM array script and the local grid search script. See the WandB sweeps migration section below.

---

## WandB Sweeps migration plan

The codebase is already compatible with wandb sweeps — `train.py` accepts all hyperparams as Hydra CLI overrides. The migration requires:

### 1. Create a sweep config YAML

Example for a reward parameter sweep:

```yaml
# configs/sweeps/reward_sweep.yaml
program: train.py
method: bayes           # or grid, random
metric:
  name: val_det/env/success_rate
  goal: maximize
parameters:
  env.reward.lambda_p:
    distribution: log_uniform_values
    min: 0.01
    max: 1.0
  env.reward.lambda_t:
    values: [30, 60, 100]
  env.reward.lambda_d:
    distribution: uniform
    min: 0.05
    max: 0.5
command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}
  - logging.wandb.mode=online
```

### 2. Launch the sweep

```bash
# Create sweep (returns sweep ID)
wandb sweep configs/sweeps/reward_sweep.yaml --project cogniland --entity crusoe

# Launch agents (local)
wandb agent crusoe/cogniland/SWEEP_ID

# Launch on SLURM: replace the array job with N identical jobs, each running:
wandb agent crusoe/cogniland/SWEEP_ID
```

### 3. Benefits over current approach

| Feature | SLURM array / grid search | wandb sweeps |
|---------|--------------------------|--------------|
| Early stopping | No | Yes (via `early_terminate`) |
| Bayesian optimization | No | Yes (`method: bayes`) |
| Grid definition | Hardcoded in bash/python | Declarative YAML |
| Failure recovery | None | Agent retries automatically |
| Coordination | Manual | wandb server coordinates |
| Analysis | Manual or `run_grid_search.py` report | Built-in parallel coordinates, importance plots |

### 4. Hydra + wandb sweep compatibility

Hydra uses `key=value` syntax for overrides. wandb sweeps pass `--key value` by default. The `command:` section with `${args_no_hyphens}` converts wandb's format to Hydra-compatible positional args.

One caveat: Hydra's `--multirun` is not needed — wandb handles the outer loop. Each wandb agent invocation is a single Hydra run.

### 5. SLURM integration

Create a generic SLURM script that runs `wandb agent`:

```bash
#!/bin/bash
#SBATCH --job-name=wandb_agent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-N

wandb agent crusoe/cogniland/$SWEEP_ID
```

Each array task pulls the next set of hyperparameters from the wandb server. No hardcoded grids needed.
