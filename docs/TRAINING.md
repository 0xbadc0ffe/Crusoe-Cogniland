# Training Pipeline

How a training run works end-to-end, with emphasis on WandB logging, the eval pipeline, and the experiment workflow.

Cross-references use `file.py:line`.

---

## Training loop structure

All three models (PPO, RecurrentPPO, DRC) follow the same high-level loop. Shown for PPO (`ppo.py:224`):

```
model.train(cfg)
  ├── setup
  │   ├── set_reproducibility(seed)
  │   ├── load MapDataset (train/val/test splits)
  │   ├── build train env: BatchedIslandEnv(env_config, num_envs=320, world_maps=train_maps)
  │   ├── build optimizer: Adam(lr=5e-4, eps=1e-5)
  │   ├── WandBLogger(cfg) — inits wandb run with slim config
  │   ├── cache eval env: BatchedIslandEnv(env_config, num_envs=16, world_maps=val_maps)
  │   ├── cache EvalRunner
  │   └── curriculum: start at EXTRA_EASY
  │
  ├── for update in 1..num_updates:                          ppo.py:324
  │   ├── LR annealing: lr = (1 - update/num_updates) * initial_lr
  │   ├── _collect_rollout(env, model, obs, rollout_steps)   ppo.py:139
  │   │   └── for rollout_steps: get_action_and_value → env.step → buffer.add
  │   │       captures episode_stats for completed episodes
  │   ├── global_step += num_envs * rollout_steps
  │   ├── curriculum stage transition check                   ppo.py:337
  │   ├── log_rollout_stats(logger, episode_stats, step=update)   logging.py:24
  │   ├── _compute_gae(buffer, next_value, gamma, lambda)    ppo.py:163
  │   ├── _ppo_update(optimizer, flat_data, advantages, returns)  ppo.py:522
  │   │   └── for epochs: for minibatches: clipped PPO loss + critic MSE - entropy bonus
  │   ├── logger.log(train_metrics, step=update)
  │   ├── if update % eval_every == 0:
  │   │   ├── _run_eval(cfg, logger, update, split="val")    ppo.py:601
  │   │   ├── save ckpt_last.pt
  │   │   └── if det_sr > best: save ckpt_best.pt
  │   └── if update % ckpt_every == 0: save ckpt_{update}.pt
  │
  └── finalize
      ├── upload best checkpoint to WandB (if store_last_ckpt=true)
      ├── _run_eval(cfg, logger, global_step, split="test")  ppo.py:416
      ├── logger.log_final_test_summary(test_metrics)
      ├── _run_behavioral_eval(logger, global_step)           ppo.py:438
      └── logger.finish()
```

---

## WandB logging flow

### Initialization (`logging.py:148`)

`WandBLogger.__init__()`:
1. `wandb.init(project, entity, name, group, mode, config=slim_config, tags=[model_name])`
2. Archives full Hydra config as `_full_config` JSON string in `run.config`
3. Pre-initializes summary keys for test metrics (so they appear as empty columns from the start)

**Run naming:** `_make_run_name()` returns `"{model}_{env_mode}"`. `_make_group_name()` appends LR. Both are overridable via `logging.wandb.name` / `logging.wandb.group`.

**Config stored in WandB** (`logging.py:67`): Slim dict with prefixed keys:
- `reward/lambda_p`, `reward/lambda_s`, etc.
- `ppo/lr`, `ppo/clip_range`, `ppo/epochs`, etc.
- `gae/gamma`, `gae/lambda`
- `rollout/parallel_envs`, `rollout/moves_per_update`, `rollout/total_moves`
- `curriculum/switch_steps`, `curriculum/easy_radius`
- `model/name`, `model/hidden_dim`, `model/cnn_channels`
- `env/map_size`, `env/max_steps`, `env/seed`

### Per-update logging

| What | WandB key pattern | Logged by |
|------|------------------|-----------|
| PPO losses | `train/model/policy_loss`, `value_loss`, `entropy`, `clipfrac`, `approx_kl`, `explained_variance` | `_ppo_update` return dict |
| LR + throughput | `train/model/learning_rate`, `train/sps` | Training loop |
| Episode stats (noisy) | `train/env/episode_return_mean`, `episode_length_mean`, `success_rate` | `log_rollout_stats()` |

### Eval logging

`_run_eval()` runs both det and stoch policies, then:

1. **Scalar metrics** → `logger.log(metrics, step=update)` — keys like `val_det/env/success_rate`, `val_det/env/directness_mean`
2. **Trajectory images** → `logger.log_trajectory_images(...)` — det only, first 3 episodes
3. **Terrain distribution** → `logger.log_terrain_scalars(...)` — val only, growing WandB Table
4. **Per-episode table** → `logger.log_eval_table(...)` — both det and stoch

### End-of-training logging

1. **Test eval** → same as val eval but on held-out test maps (fresh env with `seed + 2000`)
2. **`log_final_test_summary()`** → pushes all test metrics to `run.summary` + logs `test/summary_table`
3. **Model artifact** → `log_model_artifact()` uploads best checkpoint (if `store_last_ckpt=true`)
4. **Behavioral eval** → runs on 9 custom maps → `test/behavioral/{name}/...` + trajectory images

---

## Eval pipeline internals

### `_run_eval()` flow (`ppo.py:601`)

```
_run_eval(cfg, logger, global_step, split)
  ├── if split=="test": build fresh BatchedIslandEnv with test_maps, seed+1000
  │   else: reuse cached val eval_env
  ├── det_result = runner.run(get_deterministic_action, n_eps, "det", split)
  ├── sto_result = runner.run(get_action_and_value()[0], n_eps, "stoch", split)
  ├── scalar_metrics = summarizer.scalar_metrics(det_result) | scalar_metrics(sto_result)
  ├── log trajectory images (det only, first max_images episodes)
  ├── log terrain distribution (val only)
  └── log per-episode tables (both modes)
```

### `EvalRunner.run()` internals (`runner.py:64`)

The runner steps N parallel episodes to completion, tracking:
- Per-step: HP, resources, terrain visits, visibility counts, risk drawdowns
- Per-episode: return, length, outcome, trajectory, observed mask, terrain cost

Key implementation details:
- **Pre-step snapshots** (`runner.py:170`): Position and cost are captured *before* `env.step()` because auto-reset overwrites state for done envs.
- **Finalization** (`runner.py:228`): When an episode finishes, final state is captured from the pre-step snapshot, not from the post-step state (which may already be reset).
- **Visibility accumulation** (`runner.py:216`): `vis_counts[N, H, W]` tracks per-cell observation counts. Seeded with initial spawn visibility. Just-finished episodes are excluded from vis accumulation (their minimap is already reset).

### Recurrent model eval difference

`RecurrentPPOAgent._run_eval()` (`recurrent_ppo.py:631`) wraps the policy functions in closures that carry the RNN hidden state across steps:

```python
h_det = [model.init_hidden(n_eps, device)]
def det_policy(obs):
    act, h_new = model.get_deterministic_action(obs, h_det[0])
    h_det[0] = h_new
    return act
```

---

## Checkpointing

**What's saved** (`utils.py:36`): `model_state_dict`, `optimizer_state_dict`, `torch_rng_state`, `np_rng_state`, `step`.

**Checkpoint strategy:**
- `ckpt_best.pt` — saved whenever val det success rate improves
- `ckpt_last.pt` — saved at every eval interval
- `ckpt_{update}.pt` — periodic (if `checkpoint_every_n_updates > 0`)
- All saved to `artifacts/{wandb_run_id}/`

**Resume:** `python train.py resume=artifacts/abc123/ckpt_last.pt` — restores model, optimizer, RNG state, and resumes from the correct update number.

---

## Curriculum

Three stages (`types.py:16`), activated when `curriculum_switch_steps > 0`:

```
EXTRA_EASY                              EASY                              NORMAL
(spawn/target within 25-cell radius)    (within 50-cell radius)           (any land cell)
(compass noise: 5 deg)                  (compass noise: 30 deg)           (compass noise: 60 deg)
     │                                       │                                 │
     └── at global_step >= 6M ──────────────►└── at global_step >= 16M ──────►│
```

Both the train env and eval env curriculum stages are set together. The eval env is always initialized once and reused (for val); test env is built fresh.

---

## Experiment workflow (current)

### Local single run

```bash
python train.py models=ppo_1m logging.wandb.mode=online
```

### SLURM single run

```bash
sbatch scripts/slurm/train_slurm.sh
```

Uses `models=ppo_1m`, 48-hour walltime, 1 GPU, 8 CPUs.

### SLURM sweep (current approach)

`scripts/slurm/sweep_slurm.sh` — SLURM array job with hardcoded parameter grid. Example: 1D sweep over `lambda_p` with 6 values = 6 array tasks.

### Local grid search

```bash
python scripts/run_grid_search.py --workers 2
```

Spawns subprocesses for each grid combination. Creates a WandB report with scatter plots at the end.

### Limitations of current approach

See `docs/KNOWN_ISSUES.md` for details on:
- Manual SLURM array grids (no early stopping, no Bayesian optimization)
- Subprocess-based local grid search (no failure recovery)
- Eval pipeline bugs
