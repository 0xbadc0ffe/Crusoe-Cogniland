# Architecture Guide

Cross-references use `file.py:line` format for jumping between docs and code.

---

## High-level flow

```
train.py                                          Entry point (@hydra.main)
  ├── EnvConfig.from_hydra(cfg)                   types.py:302  — parses YAML → frozen dataclass
  ├── build_model(cfg, env_config, device)         models/__init__.py:6  — dispatches on cfg.models.name
  └── model.train(cfg)                             ppo.py:224 / recurrent_ppo.py:310 / drc.py
        ├── BatchedIslandEnv                       wrappers.py:11  — auto-reset wrapper
        │     └── Islands                          islands.py:128  — map pool + pure-function step
        │           └── core.env_step()            core.py:24  — one batched step (pure function)
        ├── WandBLogger(cfg)                       logging.py:148
        ├── training loop: collect → GAE → update → log → eval → checkpoint
        └── EvalRunner → CognilandSummarizer       eval/runner.py:51, eval/summarizer.py:21
```

---

## Module map

### `cogniland/env/` — Environment

| File | What it owns | Key symbols |
|------|-------------|-------------|
| `constants.py` | Action definitions only (5 actions). Terrain data is now config-driven. | `ACTIONS`, `ACTION_DELTAS`, `NUM_ACTIONS` |
| `types.py` | All immutable types. `EnvState` (NamedTuple), `EnvConfig` (frozen dataclass), `CompiledTerrainData` (tensor lookup tables built once from terrain list). Sub-configs: `MapGenConfig`, `AgentConfig`, `MinimapConfig`, `RewardConfig`. | `EnvState:23`, `EnvConfig:193`, `CompiledTerrainData:72` |
| `core.py` | **Pure functions** for one step. No classes, no self, no mutation. Contains `env_step()`, `compute_reward()`, movement, terrain effects, and the full minimap computation with Bresenham raycasting. | `env_step:24`, `compute_reward:121`, `compute_minimap_batch:407`, `compute_occlusion_mask_batch:343` |
| `islands.py` | `Islands` — owns a pool of `world_maps`, delegates to `core.env_step()`. Handles procedural generation, spawn/target sampling (curriculum-constrained), forward + reverse Dijkstra precomputation at reset. | `Islands:128`, `generate_island:42` |
| `pathfinding.py` | Scipy Dijkstra on grid graphs. Two graph types: move-cost (for eval metrics) and reward cost-to-go (with resource drain + raft penalty, for progress signal). `ThreadPoolExecutor` parallelism (scipy releases GIL). | `batch_dijkstra_from_sources:161`, `batch_reverse_dijkstra:211` |
| `wrappers.py` | `BatchedIslandEnv` — training wrapper. Auto-resets done envs, builds obs dict `{"scalars": [B,5], "minimap": [B,3,D,D]}`, tracks episode rewards/lengths. | `BatchedIslandEnv:11` |
| `dataset.py` | `MapDataset` — three disjoint sets (train/val/test) of pre-generated heightmaps as CPU float32 tensors. Loaded from per-split `.pt` files. | `MapDataset:13` |
| `custom_maps.py` | 9 hand-crafted behavioral test maps (occlusion, resource mgmt, routing). Each returns `(canvas, spawn_rc, target_rc)`. | `_GENERATORS:392` |

**One step, traced through the call stack:**

```
BatchedIslandEnv.step(action)                            wrappers.py:58
  └─ Islands.step(state, action, target_pos)             islands.py:344
       └─ core.env_step(state, action, per_env_maps, target_pos, config, compiled, ctg_maps)
            ├─ apply_movement()                          core.py:167
            ├─ compass update (unit dir + noise)         core.py:48
            ├─ compute_terrain_levels()                  core.py:179
            ├─ compute_minimap_batch()                   core.py:407
            ├─ apply_movement_costs()                    core.py:212
            ├─ apply_terrain_effects()                   core.py:229
            │    ├─ per-terrain resource drain
            │    ├─ forage action on forest (HP-first heal, then resources)
            │    └─ land-to-water boat cost
            ├─ cost-to-go lookup from Dijkstra maps      core.py:84
            └─ compute_reward(ctg_delta, cost, ...)      core.py:121
```

### `cogniland/models/` — Self-contained agents

Each file = architecture + training loop + eval orchestration. To swap models, change `models=ppo` to `models=recurrent_ppo`.

| File | Agent class | Network class | Algorithm | Params |
|------|------------|---------------|-----------|--------|
| `ppo.py` | `PPOAgent` | `ActorCritic` | PPO + GAE, flattened minibatch shuffle | ~250K (ppo.yaml) or ~1.04M (ppo_1m.yaml) |
| `recurrent_ppo.py` | `RecurrentPPOAgent` | `RecurrentActorCritic` | PPO + truncated BPTT, sequence-chunk updates | ~249K |
| `drc.py` | `DRCAgent` | `DRCActorCritic` | IMPALA V-trace, un-normalised advantages | variable (DRC(D,N)) |

**`ActorCritic` architecture — small PPO** (`ppo.py:36`, `ppo.yaml`):

```
minimap [B,3,45,45]
  → Conv2d(3→16, 3x3) → ReLU → MaxPool2d(2)           → [B,16,22,22]
  → Conv2d(16→32, 3x3) → ReLU
  → Conv2d(32→32, 3x3) → ReLU → AdaptiveMaxPool2d(4)  → [B,32,4,4] → Flatten → [B,512]

scalars [B,5]
  → Linear(5→64) → ReLU                                → [B,64]

concat [B,576]
  → Linear(576→256) → ReLU → Linear(256→256) → ReLU   → [B,256]
  → actor: Linear(256→5)   (orthogonal init, std=0.01)
  → critic: Linear(256→1)  (orthogonal init, std=1.0)

~250K parameters
```

**PPO 1M architecture** (`ppo.yaml` with `ppo_1m.yaml` overrides):

```
                    minimap [B, 3, 45, 45]                          scalars [B, 5]
                           │                                              │
              Conv2d(3→32, 3x3, pad=1)                          Linear(5→128)
                     ReLU                                            ReLU
                 MaxPool2d(2)             ← 45→22                    │
              Conv2d(32→64, 3x3, pad=1)                              │
                     ReLU                                            │
              Conv2d(64→64, 3x3, pad=1)                              │
                     ReLU                                            │
              AdaptiveMaxPool2d(5)        ← 22→5                     │
                   Flatten                                           │
                  [B, 1600]                                      [B, 128]
                       │                                             │
                       └──────────── concat ─────────────────────────┘
                                      │
                                  [B, 1728]
                                      │
                              Linear(1728→448)
                                    ReLU
                              Linear(448→448)
                                    ReLU
                                  [B, 448]
                                 ╱        ╲
                    Linear(448→5)          Linear(448→1)
                    actor logits           critic value
                       │                       │
                  Categorical(logits)     V(obs) scalar
                       │
                   action ∈ {0..4}

  ~1.04M parameters
  ppo.yaml variant: cnn_channels=32, cnn_out_spatial=4, hidden_dim=256, scalar_hidden=64 → ~250K params
```

**Recurrent PPO architecture** (`recurrent_ppo.py:44`):

```
                    minimap [B, 3, 45, 45]                          scalars [B, 5]
                           │                                              │
            ┌──────────────┴──────────────┐                     ┌─────────┴─────────┐
            │  Same CNN as PPO (3 conv    │                     │ Linear(5→64)      │
            │  layers + AdaptiveMaxPool)  │                     │ ReLU              │
            └──────────────┬──────────────┘                     └─────────┬─────────┘
                       [B, 512]                                       [B, 64]
                           │                                              │
                           └──────────── concat ──────────────────────────┘
                                           │
                                       [B, 576]
                                           │
                                   Linear(576→256)                    "trunk"
                                         ReLU                       (same as PPO)
                                   Linear(256→256)
                                         ReLU
                                       [B, 256]
                                           │
            ═══════════════════════════════╪═══════════════════════════════════
                                           │         ▲
                                           ▼         │
                                    ┌──────────────────────┐
                                    │   RNNCell(256 → 64)  │◄──── h_{t-1} [B, 64]
                                    │   (Elman, tanh)      │
                                    └──────────┬───────────┘
                                               │
                                          h_t [B, 64]
                                               │
            ═══════════════════════════════════╪═══════════════════════════════════
                                               │
                              ┌─────── skip connection ───────┐
                              │                               │
                        trunk output                     rnn output
                          [B, 256]                        [B, 64]
                              │                               │
                              └──────── concat ───────────────┘
                                          │
                                      [B, 320]
                                      ╱        ╲
                         Linear(320→5)          Linear(320→1)
                         actor logits           critic value

  ~249K parameters (rnn_hidden_dim=64, cnn_channels=32, hidden_dim=256)
```

### How the RNN works

The RNN is a vanilla Elman cell (`nn.RNNCell`, `recurrent_ppo.py:92`):

```
h_t = tanh(W_ih @ feat_t + W_hh @ h_{t-1} + bias)
```

where `feat_t` is the trunk output (256-dim) and `h_{t-1}` is the previous hidden state (64-dim). The hidden state is a single vector — no cell state, no gating — making it straightforward to visualise and probe.

**What the skip connection does:** The actor and critic heads receive `[trunk_feat, h_t]` (concat of 256 + 64 = 320). This means the heads always have direct access to the current observation (`trunk_feat`) in addition to the temporal summary (`h_t`). The RNN doesn't need to re-encode the current observation — it only needs to carry *what's missing from the current frame*: memory of past terrain, resource trends, previously-visible cells that are now occluded.

**Why vanilla RNN over LSTM/GRU:** Chosen for interpretability. The hidden state `h_t` is a single 64-dim vector with tanh activation (values in [-1, 1]). Every unit can be directly plotted over time, correlated with environment features, and probed with linear classifiers. LSTMs have twice the state (hidden + cell), and gating makes individual unit dynamics harder to interpret.

#### During training (`recurrent_ppo.py:196, 495`)

**Rollout collection** — hidden state carried across timesteps:

```
h = model.init_hidden(num_envs, device)     # [B, 64] zeros
for step in rollout:
    action, log_prob, _, value, h_new = model(obs, h)    # full forward pass
    obs, reward, done, info = env.step(action)
    buffer.add(obs, action, log_prob, reward, done, value, h)   # store h BEFORE update
    h_new[done] = 0.0     # reset hidden for auto-reset envs    recurrent_ppo.py:213
    h = h_new
```

Key detail: `h` is stored in the buffer *before* the RNN update, so `buffer.hiddens[t]` = the hidden state that was input to the RNN at step t.

**PPO update** — sequence-chunk truncated BPTT (`recurrent_ppo.py:495`):

Standard PPO shuffles individual transitions. Recurrent PPO can't do that — shuffling destroys temporal order, which the RNN needs. Instead:

1. Reshape the rollout `[T, B]` into chunks of `seq_len` steps (default 16)
2. For each epoch, shuffle *env indices* (not timesteps) → pick a minibatch of envs
3. For each chunk × env minibatch: re-run the RNN forward sequentially through the chunk, starting from the stored `h` at the chunk's first timestep
4. Compute PPO losses on the chunk outputs, backprop through the seq_len-step unrolled RNN

```
for epoch in range(4):
    env_perm = shuffle(range(B))
    for mb_envs in chunks(env_perm, minibatch_envs=64):
        for chunk_idx in range(T // seq_len):
            h = buffer.hiddens[chunk_start, mb_envs].detach()    # initial h for this chunk
            for t in range(seq_len):                              # sequential forward
                _, lp, ent, val, h = model(obs_t, h, action_t)
                h[done_t] = 0.0                                   # reset on episode boundary
            # PPO loss on this chunk's outputs, then backward()
```

The `.detach()` on the initial hidden state means gradients don't flow *between* chunks — this is the "truncated" in truncated BPTT. The RNN learns from at most `seq_len` steps of context per gradient computation.

#### During inference / eval (`recurrent_ppo.py:148, 658`)

**Deterministic eval** — hidden state carried via closure:

```python
h = model.init_hidden(n_episodes, device)      # [N, 64] zeros

def policy(obs):
    action, h_new = model.get_deterministic_action(obs, h)
    h = h_new                                    # carry across steps
    return action
```

The hidden state accumulates information over the entire episode (up to 1000 steps). This gives the RNN a much longer effective context than the 16-step training chunks. The training learns *what to store* in 16-step windows; at inference the stored representations are useful over much longer horizons.

**Stateless fallback** — for compatibility with the non-recurrent `EvalRunner` interface, `RecurrentPPOAgent` provides stateless wrappers (`recurrent_ppo.py:286`) that create fresh zero hidden state each call. These are only used when the caller doesn't maintain state (e.g., the standalone `eval.py` script).

**`DRCActorCritic`** (`drc.py:88`): 2x Conv2d(4x4) encoder (no nonlinearity) → D stacked ConvLSTM cells × N think steps with pool-and-inject → AdaptiveMaxPool2d readout → trunk → heads.

**Design choice — duplicated training loops:** Each model file copy-pastes curriculum logic, checkpointing, eval orchestration, and behavioral eval. This means changes to the pipeline (e.g., logging format, curriculum stages) must be applied in 3 places. See `docs/KNOWN_ISSUES.md`.

### `cogniland/eval/` — Evaluation pipeline

Three layers, no WandB dependency in the first two:

```
EvalRunner.run(policy_fn, n_episodes, ...)     runner.py:64   → EvalResult
CognilandSummarizer.scalar_metrics(result)     summarizer.py:24  → dict[str, float]
WandBLogger.log(metrics, step)                 logging.py:188    → wandb
```

| File | Responsibility |
|------|---------------|
| `runner.py` | Runs N parallel episodes. Tracks per-step accumulators (HP, resources, terrain visits, visibility counts, risk drawdowns). Computes derived metrics via `metrics.py`. Returns `EvalResult` with list of `EpisodeResult`. |
| `metrics.py` | Pure functions: `compute_directness`, `compute_risk_exposure`, `compute_danger_fraction`, `compute_exploration`, `compute_terrain_visit_fractions`. Each: tensors in → `[N]` result out. |
| `summarizer.py` | Aggregates `EvalResult` → flat dict with `{prefix}/success_rate`, `{prefix}/{metric}_mean/_std/_max/_min`. Also `terrain_pcts()` and `eval_table_rows()`. |

### `cogniland/logging.py` — WandB

See `docs/TRAINING.md` for the full logging flow and metric namespaces.

### `cogniland/utils.py`

| Function | Purpose |
|----------|---------|
| `set_reproducibility(seed)` | Pins torch/numpy/Python random + CUDA seeds + CuDNN deterministic. |
| `save_checkpoint` / `load_checkpoint` | Model weights + optimizer state + RNG state. Supports `resume=path`. |
| `render_trajectory(...)` | Matplotlib figure: terrain map + path with visit-count coloring (red→black) + fog-of-war. |

---

## Config structure

```
configs/
├── main.yaml              Top-level: device, resume, logging (wandb project/entity/mode), eval settings
├── env/
│   └── default.yaml       Map generation, agent params, minimap, max_steps, terrain list, reward coefficients
└── models/
    ├── ppo.yaml           Small PPO (~250K): cnn_channels=32, hidden_dim=256
    ├── ppo_1m.yaml        Large PPO (~1.04M): cnn_channels=64, hidden_dim=448
    └── recurrent_ppo.yaml RNN PPO (~249K): rnn_hidden_dim=64, seq_len=16
```

`main.yaml` defaults to `env: default` and `models: ppo_1m`. Override from CLI:

```bash
python train.py models=ppo env.reward.lambda_p=0.1 logging.wandb.mode=disabled
```

---

## Design decisions

| Decision | Rationale |
|----------|-----------|
| **Immutable state** (`NamedTuple`) | `state._replace()` for updates. Valid JAX pytree. No accidental mutation. |
| **Pure functions in `core.py`** | Swap `torch.*` → `jnp.*` for JIT compilation. Testable in isolation. |
| **Config-driven terrains** | Terrain data (thresholds, costs, rates, vis, colors, tags) lives in YAML. `CompiledTerrainData` builds tensor LUTs once. Engine code never reads global constants for terrain. |
| **Level Replay** | Each env in the batch can use a different map. Maps re-sampled on episode reset. |
| **Dual Dijkstra at reset** | Forward: optimal traversal time (time-efficiency at success). Reverse: cost-to-go maps (per-step progress signal). |

---

## Performance profile

| Operation | Where | Cost | Notes |
|-----------|-------|------|-------|
| **Dijkstra at reset** | `islands.py:257,270` | ~50ms per map (250x250) | Dominant cost per episode reset. Parallelised via ThreadPoolExecutor. |
| **Minimap + occlusion** | `core.py:407,343` | Per-step, GPU | Bresenham rays cached via `lru_cache`. `scatter_reduce` for visibility. Most tensor-heavy per-step op. |
| **Map generation** | `islands.py:42` | ~5-10s per map, CPU | Nested Python loops. Runs once at init or loaded from dataset. |
| **RNN BPTT** | `recurrent_ppo.py:495` | Per-update | Sequential forward through seq_len=16 chunks. Slower than feedforward PPO updates. |
| **DRC think steps** | `drc.py:231` | Per-step | D×N ConvLSTM passes per environment step. Significant per-step overhead. |
