# MemoryEnv — PPO agents, activation datasets, steering experiments

A T-maze memory task (cue **direction** → branch, cue **color** → door, doors
randomized 50/50) used as a mechanistic-interpretability substrate. The pure-JAX
env lives in `src/cogniland/memory_env/jax/` (bit-parity with the MiniGrid env,
see `tests/test_memory_env_jax_parity.py`). Full write-up: the LaTeX report in
`docs/memory_env_report.tex` (figures regenerate into `outputs/report_figs/`).

**Released models** (frozen, git-LFS): `released_models/memory_env/ppo_{2,3,4}cue/`
— orbax checkpoint (`checkpoints/step_25000000`, params only), the as-trained
`config.json`, and the exact `activations.npz` used by all probing/steering
analyses. All three solve **all their training cues** (success = 1.00, n=96,
random doors). PPO+GRU, 733,476 parameters.

## Reproduce: training

One RTX-3090, ~10 min per model (~70k steps/s). The three released models were
trained with different settings (each the survivor of its own tuning arc — see
report §7):

```bash
# 2cue  (trained on green_up, blue_down)     [= released ppo_2cue]
python scripts/memory_env/train_ppo_memory.py --cue 2cue --total-timesteps 25000000 \
  --seed 1 --tag vs2 --set num_envs=256 ent_coef=0.03            # penalty -1.0 (default)

# 3cue  (green_up, green_down, blue_down)    [= released ppo_3cue]
python scripts/memory_env/train_ppo_memory.py --cue 3cue --total-timesteps 25000000 \
  --seed 0 --tag vs --set num_envs=256                            # ent 0.01, penalty -1.0 (defaults)

# 4cue  (all four cues)                      [= released ppo_4cue]
python scripts/memory_env/train_ppo_memory.py --cue 4cue --total-timesteps 25000000 \
  --seed 1 --tag vs4 --set num_envs=256 ent_coef=0.03 wrong_branch_penalty=0.0
```

SLURM: `sbatch --array=0-2 --export=ALL,SRC_DIR=...,TAG=...,STEPS=25000000,SEED=... \
scripts/memory_env/slurm/ppo_memory.sbatch` (EXTRA_SET carries the `--set` overrides).
Note: PPO here is seed-sensitive — some (cue-set, recipe) combinations collapse
to a never-terminating shaping optimum; the positive-only reward (penalty 0) +
ent 0.03 was required for 4cue.

## Reproduce: evaluation and analyses

```bash
RD=released_models/memory_env/ppo_4cue           # or a fresh outputs/ppo_runs/<run>

# per-cue success + direction breakdown (greedy, n=96, random doors)
python scripts/memory_env/diag_ppo.py --run-dir $RD --n 96

# activation dataset (512 greedy episodes, all 4 cues, GRU hidden + obs embedding)
python scripts/memory_env/build_ppo_activations.py --run-dir $RD --n 512 --tmax 60

# steering experiments (report §6): tables / static figure / video
python scripts/memory_env/steer2_ppo.py quant --run-dir $RD --n 96
python scripts/memory_env/steer2_ppo.py fig   --run-dir $RD --out steer2_4cue.png
python scripts/memory_env/steer2_ppo.py video --run-dir $RD --out steer2_4cue.mp4

# activation report (PCA/UMAP/avg, HTML) and belief-geometry report
python scripts/memory_env/report_ppo_activations.py --npz $RD/activations.npz --out report.html
python scripts/memory_env/report_belief_geometry.py
```

Everything downstream of training is deterministic given the checkpoint
(fixed PRNG seeds inside the scripts; `activations.npz` regenerates identically
— the released copies are provided so probe fits match the report exactly).

GPU note: the released orbax checkpoints were saved on GPU; restoring on a
CPU-only host can fail with a sharding error — run analyses on a GPU node.

## Script map

| script | role |
|---|---|
| `train_ppo_memory.py` | single-file JAX recurrent PPO (PPO+GRU) trainer |
| `diag_ppo.py` / `diag_jax.py` | per-cue eval for PPO / Dreamer checkpoints |
| `build_ppo_activations.py` | activation dataset (feat = GRU hidden, obs_embed) |
| `steer2_ppo.py` | steering suite: behavior-steer via activations, transient memory swap |
| `steer_ppo.py` | v1 steering (single-axis add/clamp) — superseded by steer2 |
| `probe_jax.py` | per-timestep direction/color probes for Dreamer checkpoints |
| `report_ppo_activations.py`, `report_belief_geometry.py`, `plot_belief_behavior.py` | analysis reports/figures |
| `fig_trajectories_ppo.py`, `video_rollout.py`, `video_steering.py`, `viz_*` | trajectory figures & belief-overlay videos |
| `dreamerv3_memory.py` | pure-JAX DreamerV3 baseline (solves direction, never the door) |

Labels note: the cue *direction* is stored under the legacy key `shape` inside
`activations.npz`; all documentation calls it direction.
