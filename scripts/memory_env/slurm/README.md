# MemoryEnv → R2-Dreamer trainings on SLURM

Trains 3 DreamerV3 (R2-Dreamer) models on the cogniland `MemoryEnv`, one per cue
distribution, then evaluates all three on the **same held-out 4-cue test set** and
plots **average reward per cue**. Designed to expose the entanglement failure mode:
a model trained on correlated cues should mis-pick the door on off-distribution cues.

| array idx | task          | cues seen                                   | role          |
|-----------|---------------|---------------------------------------------|---------------|
| 0         | `memory_2cue` | green_up, blue_down                         | entangled     |
| 1         | `memory_3cue` | green_up, green_down, blue_down             | partial       |
| 2         | `memory_4cue` | green_up, blue_up, green_down, blue_down    | factorized    |

Each model: **size25M**, **10M env steps**, **16 parallel envs**, progress-shaped
MemoryEnv (dense PBRS so the 33-cell corridor is learnable; shaping is
shape/colour/branch-agnostic and leaks no task info).

## 0. Get the repo on the cluster
`git clone`/`rsync` this repo to the cluster. The `external/r2dreamer/` code and
`src/cogniland/memory_env/` are both needed; nothing else from cogniland is.

## 1. One-time env setup (login or interactive GPU node)
```bash
bash scripts/memory_env/slurm/setup_env.sh        # creates conda env `r2dreamer`
```
Creates a Python 3.11 conda env, `pip install -e external/r2dreamer` + `minigrid`
+ `matplotlib`, and sanity-checks that MemoryEnv imports and produces a (56,56,3)
observation. Override the env name with `CONDA_ENV=...`.

## 2. Submit (training array + dependent eval)
```bash
bash scripts/memory_env/slurm/submit.sh
```
Submits the 3-task array, then an eval job chained with `--dependency=afterok` so
it runs once all three finish. Pass cluster flags via `SBATCH_FLAGS`:
```bash
SBATCH_FLAGS="--partition=gpu --account=MYACCT" bash scripts/memory_env/slurm/submit.sh
```

## Cluster-specific knobs (edit the `#SBATCH` lines or override on the CLI)
- **GPU**: `--gres=gpu:1` (some sites use `--gpus=1` / a specific type like
  `--gres=gpu:a100:1`).
- **partition / account**: add `--partition=...`, `--account=...`.
- **time**: default `24:00:00` per model — bump if your throughput is lower.
- **cpus-per-task**: default `18` (16 parallel envs + headroom). MemoryEnv steps
  on CPU workers, so give it cores.
- **mem**: default `64G`.

Override-via-env knobs in `train_memory.sbatch`: `REPO`, `CONDA_ENV`, `STEPS`,
`MODEL`, `SEED`.

## 3. Outputs
- Checkpoints: `external/r2dreamer/runs/memory_{2,3,4}cue/latest.pt`
  (saved at end of training; key `agent_state_dict`).
- TensorBoard / metric logs: under each `runs/memory_*` dir.
- Final plot: **`outputs/report/memoryenv_reward_per_cue.png`** — grouped bars,
  avg reward per cue for the 3 models on the shared test set.

After the eval job finishes, copy that PNG back for analysis.

## Notes
- The held-out test set uses seeds ≥ 1,000,000; training seeds stay below that
  (disjoint), so test episodes are never seen in training.
- Doors are 50/50 left/right; the cue spawns uniformly over the 6 non-corridor
  cells of the 3×3 start room — both randomised per episode.
- To change the step budget cluster-wide: edit `steps:` in
  `external/r2dreamer/configs/env/memory.yaml` (currently `10e6`) or pass
  `STEPS=...` to the sbatch job.
- Single-GPU sequential alternative (no SLURM):
  `bash scripts/memory_env/launch_r2dreamer.sh`.
