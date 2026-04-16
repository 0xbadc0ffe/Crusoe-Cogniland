# Refactor Guide — Multi-Task RL Framework (PPO-RNN + Dreamer + STORM)

This is a step-by-step plan for restructuring a multi-task RL codebase to
match the layered design of this reference repo. Target agents: **PPO-RNN**,
**DreamerV3**, and **STORM**.

## Multi-task setup

There is a **single environment** shared by 7 tasks. Tasks differ only in
reward:

```
reward(task_i) = reward_base + reward_i      # for i in 0..6
```

The agent is told which task to perform via a **task embedding** concatenated
to its observation (or otherwise injected as input). During training, the
agent sees all 7 tasks (e.g. by sampling / cycling through tasks across
episodes or rollouts). There is no sequential curriculum — all tasks are
trained concurrently, so there is no forgetting / forward-transfer problem.

**Implications for the framework:**

- The env wrapper must: accept a task index, return the task-specific reward,
  and expose the task embedding as part of the observation.
- The agent network takes `(obs, task_embedding)` as input.
- Training samples tasks (e.g. round-robin per episode, or random per
  rollout).
- Evaluation runs all 7 tasks separately and logs per-task + aggregate
  metrics.
- W&B metrics: `train/<metric>` (aggregate across tasks during training),
  `eval/task_{i}/<metric>` (per-task eval), `eval/aggregate/<metric>`.

The `dreamer.py`, `storm.py`, and their supporting modules (`commons/`,
`world_models/`, `policy/`) are provided in the `reference_files/` folder
at the root of this repo. That folder contains all files copied from the
reference codebase, pre-organized by destination. The new repo's `Agent`
dataclass, `AgentState`/`RuntimeState`, and registry must be kept
structurally compatible so those files drop in with only minor edits.

---

## 0. What you keep, drop, and add vs. the reference

**Drop entirely:**
- `GlobalMetricsTracker` (FT/BT/forgetting, AUC learning curves, baselines).
- `configs/baselines/`, `scripts/aggregate_baselines.py`.
- The sequential `for env_idx, env_name in env_manager.envs_dict` training
  loop in `Trainer.run`.
- Per-task `set_environment` / `on_environment_end` hooks on the agent.
- `eval/ft_auc/*` metric prefixes.

**Keep (and copy):**
- Layered OmegaConf config (env YAML + agent YAML + sweep YAML).
- Functional `Agent` dataclass + decorator registry + auto-discovery.
- `Trainer` as the orchestrator (init agent, run training, periodic eval,
  checkpoint, log).
- `RunLogger` with `wandb.define_metric` step-metric registration and
  config artifact upload.
- `scripts/run_sweep.py` parallel sweep launcher.
- Orbax checkpointing.

**Copy from `reference_files/`** (provided in this repo):

All files below are in the `reference_files/` folder, pre-organized to
mirror the target structure. Copy them into `src/<pkg>/` and fix imports.

- `reference_files/agents/state.py` → `src/<pkg>/agents/state.py`
- `reference_files/agents/agent.py` → use as reference for `src/<pkg>/agents/agent.py`
- `reference_files/agents/utils.py` → `src/<pkg>/agents/utils.py`
- `reference_files/agents/dreamer.py` → `src/<pkg>/agents/dreamer.py`
- `reference_files/agents/storm.py` → `src/<pkg>/agents/storm.py`
- `reference_files/agents/commons/` → `src/<pkg>/agents/commons/` (entire dir)
- `reference_files/agents/policy/` → `src/<pkg>/agents/policy/` (entire dir)
- `reference_files/agents/world_models/` → `src/<pkg>/agents/world_models/` (entire dir)
- `reference_files/trainer/checkpoint.py` → `src/<pkg>/trainer/checkpoint.py`
- `reference_files/trainer/utils.py` → use as reference for `RNGManager`
- `reference_files/scripts/run_sweep.py` → `scripts/run_sweep.py`
- `reference_files/configs/agent/dreamerv3.yaml` → `configs/agent/dreamerv3.yaml`
- `reference_files/configs/agent/storm.yaml` → `configs/agent/storm.yaml`
- `reference_files/configs/agent/ppo.yaml` → use as reference for `configs/agent/ppo_rnn.yaml`

**Add new (not in reference):**
- `TaskSampler` — decides which task each parallel env runs next episode.
- `MultiTaskEnvWrapper` — wraps the base env: injects task index → modifies
  reward, appends task embedding to observation.
- `MetricsTracker` — new, simpler than the reference's
  `TaskMetricsTracker` + `GlobalMetricsTracker`, but tracks per-task deques
  during eval.

---

## 1. Target directory layout

```
your_repo/
├── pyproject.toml
├── configs/
│   ├── env/
│   │   └── <env_name>.yaml
│   ├── agent/
│   │   ├── ppo_rnn.yaml
│   │   ├── dreamerv3.yaml
│   │   └── storm.yaml
│   └── sweeps/
│       └── ppo_rnn_seeds.yaml
├── scripts/
│   ├── train.py
│   ├── run_sweep.py
│   └── launch_sweep.sh
├── src/<pkg>/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── env.py                   # setup_environment() — XLA/JAX env vars
│   │   └── utils.py                 # load_config(), configure_sweep_config()
│   ├── shared/
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── registry.py              # make_env(env_id, config)
│   │   ├── multitask_wrapper.py     # MultiTaskEnvWrapper — reward routing + task embedding
│   │   └── task_sampler.py          # TaskSampler — round-robin / random task selection
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent.py                 # Agent dataclass
│   │   ├── state.py                 # AgentState/RuntimeState (copied from reference)
│   │   ├── registry.py              # @register_agent + load_agent
│   │   ├── utils.py                 # RatioTracker, sg, etc. (copied)
│   │   ├── ppo_rnn.py               # your agent
│   │   ├── dreamer.py               # copied + minor edits
│   │   ├── storm.py                 # copied + minor edits
│   │   ├── commons/                 # copied wholesale
│   │   ├── policy/                  # copied wholesale
│   │   └── world_models/            # copied wholesale
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── tracker.py               # MetricsTracker (train + per-task eval)
│   └── trainer/
│       ├── __init__.py
│       ├── trainer.py
│       ├── run_logger.py            # RunLogger (W&B)
│       ├── checkpoint.py            # CheckpointCallback (orbax)
│       └── utils.py                 # RNGManager
└── notes/
```

---

## 2. Configuration

### `configs/env/<env_name>.yaml`

```yaml
experiment_name: default
results_path: /path/to/results
offline: false

entity: your-wandb-entity
project: your-wandb-project

seed: 42

env_id: YourEnv-v0                 # single underlying env
num_tasks: 7                       # number of tasks sharing this env
task_embedding_dim: 8              # dimension of learned or fixed task embedding
task_sampling: round_robin         # "round_robin" | "uniform_random"

env:
  num_parallel_envs: 32
  num_parallel_envs_eval: 4
  # ...env-specific options

trainer:
  num_train_frames: 5_000_000
  num_eval_frames: 20_000          # per task during evaluation
  eval_interval_frames: 500_000
  log_interval_episodes: 50

metrics_tracker:
  moving_avg_window_size: 100
```

Notes:
- `num_tasks` is read by the env wrapper and the trainer.
- `task_embedding_dim` is read by the agent to size its input layer.
- `task_sampling` controls how the trainer assigns tasks to parallel envs each
  episode.

### `configs/agent/ppo_rnn.yaml`

```yaml
agent:
  name: ppo_rnn
  type: ppo_rnn

  lr: 2.5e-4
  anneal_lr: true
  num_steps: 128
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.1
  entropy_coef: 0.01
  value_coef: 0.5
  clip_grad: 0.5
  normalize_advantages: true
  ppo_epochs: 4
  num_minibatches: 4

  hidden_size: 256
  head_num_layers: 1
  activation: relu

  use_rnn: true
  lstm_size: 256

  checkpoint:
    enabled: true
    interval: 1000
    keep_last: 3
    save_best: true
    checkpoint_dir: checkpoints
```

Agent configs remain task-agnostic. The network's input dimension is
computed at init time from `obs_space + task_embedding_dim`.

### Loader (`src/<pkg>/config/utils.py`)

Same as reference — merge env YAML + agent YAML, agent wins on conflicts:

```python
import os
from pathlib import Path
from omegaconf import OmegaConf

def load_config(agent_config_path: str, env_config_path: str) -> OmegaConf:
    env_cfg   = OmegaConf.load(env_config_path)
    agent_cfg = OmegaConf.load(agent_config_path)
    cfg = OmegaConf.merge(env_cfg, agent_cfg)

    cfg.name = f"{Path(env_config_path).stem}_{Path(agent_config_path).stem}"
    cfg.pid  = os.getpid()
    return cfg

def configure_sweep_config(base_config: OmegaConf, sweep_config_dict: dict) -> OmegaConf:
    dotlist = [f"{k}={v}" for k, v in sweep_config_dict.items()]
    return OmegaConf.merge(base_config, OmegaConf.from_dotlist(dotlist))
```

### XLA setup (`src/<pkg>/config/env.py`)

```python
def setup_environment():
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
```

**Must be called before any `import jax`**.

---

## 3. Multi-task environment layer

This is the only structurally new piece vs. the reference. It sits between
the raw environment and the agent.

### `src/<pkg>/envs/task_sampler.py`

```python
import jax
import jax.numpy as jnp

class TaskSampler:
    """Assigns a task index to each parallel env at episode reset."""

    def __init__(self, num_tasks: int, num_envs: int, mode: str = "round_robin"):
        self.num_tasks = num_tasks
        self.num_envs  = num_envs
        self.mode      = mode
        self._counter  = 0

    def sample(self, rng=None) -> jnp.ndarray:
        """Returns (num_envs,) int array of task indices."""
        if self.mode == "round_robin":
            tasks = jnp.array([(self._counter + i) % self.num_tasks
                               for i in range(self.num_envs)])
            self._counter += self.num_envs
            return tasks
        elif self.mode == "uniform_random":
            return jax.random.randint(rng, (self.num_envs,), 0, self.num_tasks)
        else:
            raise ValueError(f"Unknown task_sampling mode: {self.mode}")

    def fixed(self, task_id: int) -> jnp.ndarray:
        """All envs run the same task (used during eval)."""
        return jnp.full((self.num_envs,), task_id, dtype=jnp.int32)
```

### `src/<pkg>/envs/multitask_wrapper.py`

This is a thin wrapper around your base environment. It does two things:

1. **Modifies reward**: `reward = reward_base + reward_fn(task_id)`.
2. **Appends task embedding to observation**: the agent sees
   `concat(obs, task_emb)`.

The task embedding can be a one-hot vector, a learned embedding (stored in
agent params), or a fixed random projection. Start with one-hot — it's the
simplest and most debuggable.

```python
import jax.numpy as jnp

class MultiTaskEnvWrapper:
    """Wraps a base env to add task-conditioned reward and task embedding."""

    def __init__(self, base_env, num_tasks: int, task_embedding_dim: int):
        self.base_env = base_env
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim

        # Fixed task embeddings (one-hot or random projection)
        # One-hot: task_embedding_dim must == num_tasks
        # You can replace this with a learned embedding in the agent instead
        self.task_embeddings = jnp.eye(num_tasks)  # (num_tasks, num_tasks)

        # Current task assignment per env, shape (num_envs,)
        self.current_tasks = None

    def set_tasks(self, task_ids: jnp.ndarray):
        """Set which task each parallel env is running."""
        self.current_tasks = task_ids

    def observation_space(self):
        """Returns obs_shape with task embedding appended."""
        base_shape = self.base_env.observation_space()
        # If obs is (H, W, C) image: task embedding is tiled spatially
        # If obs is (D,) vector: task embedding is concatenated
        # Decide your convention here and document it.
        return base_shape  # agent handles concat internally

    def get_task_embeddings(self, task_ids: jnp.ndarray) -> jnp.ndarray:
        """Look up embeddings for given task ids. Shape: (num_envs, emb_dim)."""
        return self.task_embeddings[task_ids]

    # Delegate everything else to base_env, modifying reward on step:
    # step() returns (obs, reward, done, info)
    # reward = base_reward + task_reward_fn(task_id, info)
    # You implement task_reward_fn based on your domain.
```

**Design decision:** whether the task embedding is concatenated to the
observation inside the env wrapper or inside the agent's network forward pass
is up to you. Keeping it in the agent is more flexible (lets you experiment
with learned embeddings, FiLM conditioning, etc.), but means the agent must
know about `num_tasks` and `task_embedding_dim`. Keeping it in the wrapper
is simpler but locks you into a fixed embedding scheme.

**Recommendation:** have the env wrapper expose `get_task_embeddings(task_ids)`
and let the agent's network do the concat/conditioning. The trainer passes
`current_task_ids` to the agent alongside the observation.

---

## 4. The functional Agent pattern

### `src/<pkg>/agents/agent.py`

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import jax

PRNGKey = jax.random.PRNGKey

@dataclass
class Agent:
    """Container for agent functions. NOT a PyTree — just a namespace.

    Kept structurally compatible with the reference repo so that
    dreamer.py and storm.py can be copied in with minor edits.
    """
    # Required
    init: Callable[[PRNGKey], Any]
    train: Callable        # (state, env, rng, num_frames, **kw) -> (state, metrics)
    evaluate: Callable     # (state, env, rng, num_frames, **kw) -> metrics

    # Optional with defaults (kept for reference-repo compat)
    reset: Callable = lambda s, rng: s
    set_environment: Callable = lambda s, **kw: s
    on_environment_end: Callable = lambda s, **kw: s

    # Optional agent-specific extras
    select_action: Optional[Callable] = None
    state_from_checkpoint: Optional[Callable] = None
    get_latents: Optional[Callable] = None

    # Metadata
    obs_space: Any = None
    action_space: Any = None
```

### `src/<pkg>/agents/state.py`

**Copy `reference_files/agents/state.py` verbatim.** Keep `current_task_id`
— the trainer will use it to tell agents which task is active, and the
replay buffer can tag transitions by task.

Only edit: remove `current_num_actions` if all 7 tasks share the same action
space (they should, since it's the same env). If they do differ, keep it.

### `src/<pkg>/agents/ppo_rnn.py` — multi-task aware

The agent factory receives `config.num_tasks` and `config.task_embedding_dim`
from the merged config and sizes its network accordingly.

```python
@register_agent("ppo_rnn")
def make_ppo_rnn(config, obs_space, act_space) -> Agent:
    num_tasks = config.num_tasks
    task_emb_dim = config.task_embedding_dim

    # Build network: input is concat(obs_features, task_embedding)
    # The encoder output size is hidden_size; task embedding adds task_emb_dim
    # So the LSTM / policy heads see (hidden_size + task_emb_dim)
    # OR: use a learned embedding table inside the network
    ...

    def train(state, env, rng, num_train_frames, progress_bar=None,
              checkpoint_callback=None, task_ids=None):
        # task_ids: (num_envs,) int array — which task each env is running
        # Look up task embeddings and concat to observations in the forward pass
        # Everything else is standard PPO
        ...
        return new_state, {
            "episode_info": episode_info,
            # any scalar → auto-logged by trainer:
            "policy_loss": ...,
            "value_loss": ...,
            "entropy": ...,
        }

    def evaluate(state, env, rng, num_eval_frames, progress_bar=None,
                 task_ids=None):
        # Same: task_ids tells the agent which task embedding to use
        ...
        return {"episode_info": ...}

    return Agent(init=init, train=train, evaluate=evaluate,
                 obs_space=obs_space, action_space=act_space)
```

**Key contract:** the trainer passes `task_ids` as a kwarg to `train()` and
`evaluate()`. The agent uses it to look up task embeddings. The episode_info
arrays are the same shape as before — the trainer handles per-task
bookkeeping externally.

### Edits for copied `dreamer.py` / `storm.py`

The files are in `reference_files/agents/dreamer.py` and
`reference_files/agents/storm.py`. After copying them to
`src/<pkg>/agents/`, apply these edits:

1. **Fix imports** — `from cl.` → `from <pkg>.` (use the sed command below).
2. **Delete `set_environment` / `on_environment_end`** definitions and their
   entries in the returned `Agent(...)` kwargs.
3. **Delete `visualize_buffer_task_distribution` calls.**
4. **Add `task_ids` kwarg** to `train()` and `evaluate()`.
5. **Inject task embedding into the observation** before feeding to the
   world model / encoder. The cleanest place is right after
   `normalize_image(extract_image_obs(...))` — concat the embedding there.
6. **Store per-env `task_id` in replay buffer transitions** (replace the
   scalar `current_task_id` with the per-env array).

### Copied support modules

Copy `reference_files/agents/{commons,policy,world_models,utils.py}`
into `src/<pkg>/agents/`. Then fix imports:

```bash
find src/<pkg>/agents/{commons,policy,world_models} \
  src/<pkg>/agents/{dreamer,storm,utils,state}.py \
  -type f -name "*.py" \
  -exec sed -i '' 's|from cl\.|from <pkg>.|g; s|import cl\.|import <pkg>.|g' {} +
```

---

## 5. Metrics — training aggregate + per-task eval

### `src/<pkg>/metrics/tracker.py`

The tracker is used in two modes:

1. **Train mode** — a single tracker aggregating across all tasks (the agent
   sees mixed tasks during training; we track aggregate reward/success/length
   with rolling windows).
2. **Eval mode** — one tracker per task, so the trainer can report per-task
   performance.

```python
from collections import deque
from enum import Enum
import time
import numpy as np
from omegaconf import OmegaConf

class Mode(Enum):
    TRAIN = "train"
    EVAL  = "eval"

class MetricsTracker:
    def __init__(self, config: OmegaConf, num_parallel_envs: int, mode: str):
        self.config = config
        self.mode = Mode(mode)
        self.num_parallel_envs = num_parallel_envs
        self.window_size = config.metrics_tracker.moving_avg_window_size

        self.metrics_base = ["frame", "episode", "fps", "reward", "success", "length"]
        self.metric_functions = {
            "moving_avg_reward":       lambda: float(np.mean(self.episode_reward_history)),
            "moving_avg_success_rate": lambda: float(np.mean(self.episode_success_history)),
            "moving_avg_length":       lambda: float(np.mean(self.episode_length_history)),
        }

    @property
    def step_metric(self) -> str:
        return "train_episode" if self.mode == Mode.TRAIN else "eval_set"

    @property
    def metric_prefix(self) -> str:
        return self.mode.value

    def get_metric_names(self) -> list[str]:
        if self.mode == Mode.TRAIN:
            return self.metrics_base + list(self.metric_functions.keys())
        return [f"avg_{n}" for n in self.metrics_base]

    def initialize(self):
        self.env_total_frames = 0
        self.env_total_episodes = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.episode_reward_history  = deque([0.0]*self.window_size, maxlen=self.window_size)
        self.episode_success_history = deque([0.0]*self.window_size, maxlen=self.window_size)
        self.episode_length_history  = deque([0]  *self.window_size, maxlen=self.window_size)
```

The trainer creates **one** train-mode tracker and **7** eval-mode trackers
(one per task). See §7.

---

## 6. RunLogger — all W&B integration in one place

### `src/<pkg>/trainer/run_logger.py`

```python
import os
from omegaconf import OmegaConf
import wandb
from <pkg>.metrics.tracker import MetricsTracker
from <pkg>.shared import setup_logger

logger = setup_logger(__name__)

class RunLogger:
    def __init__(self, config: OmegaConf):
        self.config = config
        self.wandb_run = self._init_wandb_run(config)
        self.run_name = self.wandb_run.name
        self.run_id   = self.wandb_run.id
        self.results_dir = os.path.join(config.results_path, self.run_id)
        os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    def _init_wandb_run(config):
        run = wandb.init(
            entity=config.entity,
            project=config.project,
            config=OmegaConf.to_container(config, resolve=True),
            mode="offline" if config.offline else "online",
        )
        run.name = "_".join([
            config.name, config.agent.name, config.experiment_name, run.id
        ])
        artifact = wandb.Artifact(name="config", type="config")
        path = "config.yaml"
        with open(path, "w") as f:
            f.write(OmegaConf.to_yaml(config))
        artifact.add_file(path)
        run.log_artifact(artifact)
        try:
            os.remove(path)
        except OSError:
            pass
        return run

    def register_metrics(self, tracker: MetricsTracker, prefix_override: str = None):
        prefix = prefix_override or tracker.metric_prefix
        for name in tracker.get_metric_names():
            full = f"{prefix}/{name}"
            self.wandb_run.define_metric(full, step_metric=tracker.step_metric)
```

### Step metrics

| Metric prefix             | Step metric    | What it tracks                       |
| ------------------------- | -------------- | ------------------------------------ |
| `train/*`                 | `train_steps`  | Aggregate training metrics           |
| `eval/task_{i}/*`         | `train_frames` | Per-task eval (i = 0..6)             |
| `eval/aggregate/*`        | `train_frames` | Mean across all 7 tasks              |

Registered once in `Trainer.__init__`:

```python
self.run_logger.wandb_run.define_metric("eval/*", step_metric="train_frames")
```

---

## 7. Trainer — multi-task lifecycle

This is the main departure from the reference. The trainer runs **one**
training loop (not one per task) and handles task assignment.

### `src/<pkg>/trainer/trainer.py`

```python
import time
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm
from tabulate import tabulate

from <pkg>.agents.agent import Agent
from <pkg>.envs.registry import make_env
from <pkg>.envs.task_sampler import TaskSampler
from <pkg>.metrics.tracker import MetricsTracker
from <pkg>.trainer.run_logger import RunLogger
from <pkg>.trainer.checkpoint import CheckpointCallback
from <pkg>.trainer.utils import RNGManager
from <pkg>.shared import setup_logger

logger = setup_logger(__name__)


class Trainer:
    def __init__(self, config: OmegaConf, agent: Agent):
        self.config = config
        self.agent  = agent
        self.num_tasks = config.num_tasks

        self.num_train_frames     = config.trainer.num_train_frames
        self.num_eval_frames      = config.trainer.num_eval_frames
        self.eval_interval_frames = config.trainer.get("eval_interval_frames", None)

        # W&B
        self.run_logger = RunLogger(config)
        config.results_dir = self.run_logger.results_dir
        self.run_logger.wandb_run.define_metric("eval/*", step_metric="train_frames")

        # RNG
        self.rng_manager = RNGManager(seed=config.seed)

        # Environments
        self.train_env = make_env(config.env_id, config, train=True)
        self.eval_env  = make_env(config.env_id, config, train=False)

        # Task sampler
        self.task_sampler = TaskSampler(
            num_tasks=self.num_tasks,
            num_envs=config.env.num_parallel_envs,
            mode=config.get("task_sampling", "round_robin"),
        )

        # Agent
        self.agent_state = self.agent.init(self.rng_manager.get_key())

        # Metrics: one train tracker (aggregate), 7 eval trackers (per-task)
        self.train_metrics = MetricsTracker(config, config.env.num_parallel_envs, "train")
        self.train_metrics.initialize()
        self.run_logger.register_metrics(self.train_metrics)

        num_eval_envs = config.env.get("num_parallel_envs_eval", config.env.num_parallel_envs)
        self.eval_trackers = {}
        for task_id in range(self.num_tasks):
            t = MetricsTracker(config, num_eval_envs, "eval")
            self.eval_trackers[task_id] = t
            self.run_logger.register_metrics(t, prefix_override=f"eval/task_{task_id}")

        self.eval_set = 0

        # Checkpoint
        if config.agent.get("checkpoint", {}).get("enabled", False):
            self.checkpoint_callback = CheckpointCallback(
                agent=self.agent, config=config,
                results_dir=self.run_logger.results_dir,
                wandb_run=self.run_logger.wandb_run,
            )
        else:
            self.checkpoint_callback = None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        logger.info("=== Multi-task training start (%d tasks) ===", self.num_tasks)
        total_trained = 0
        pbar = tqdm(total=self.num_train_frames, desc="train")

        if self.eval_interval_frames is not None:
            self._run_evaluation(global_train_frames=0)

        while total_trained < self.num_train_frames:
            remaining = self.num_train_frames - total_trained
            seg = min(self.eval_interval_frames or remaining, remaining)

            # Sample task assignments for this training segment
            rng = self.rng_manager.get_key()
            task_rng, train_rng = jax.random.split(rng)
            task_ids = self.task_sampler.sample(rng=task_rng)

            t0 = time.time()
            self.agent_state, metrics = self.agent.train(
                self.agent_state, self.train_env, train_rng, seg,
                progress_bar=pbar,
                checkpoint_callback=self.checkpoint_callback,
                task_ids=task_ids,
            )
            fps = seg / max(time.time() - t0, 1e-9)

            self._log_training_metrics(metrics, total_trained, pbar, fps)
            total_trained += seg

            if self.eval_interval_frames and total_trained < self.num_train_frames:
                self.rng_manager.checkpoint()
                self._run_evaluation(global_train_frames=total_trained)
                self.rng_manager.restore()

        if self.eval_interval_frames is not None:
            self._run_evaluation(global_train_frames=total_trained)

        pbar.close()
        logger.info("=== Training done ===")

    # ------------------------------------------------------------------ #
    # Training metrics (aggregate across tasks)
    # ------------------------------------------------------------------ #
    def _log_training_metrics(self, metrics: dict, total_trained: int, pbar, fps: float):
        """Log aggregate training metrics. Task identity is not tracked here."""
        episode_info = metrics.get("episode_info")
        if episode_info is None:
            self._log_agent_metrics(metrics, total_trained)
            return

        returns = jnp.array(episode_info["returned_episode_returns"]).reshape(-1)
        lengths = jnp.array(episode_info["returned_episode_lengths"]).reshape(-1)
        done    = jnp.array(episode_info["returned_episode"]).reshape(-1)

        if not bool(done.any()):
            self._log_agent_metrics(metrics, total_trained)
            return

        returns_np = np.array(returns[done])
        lengths_np = np.array(lengths[done])
        successes_np = (returns_np > 0).astype(np.int32)

        for i in range(len(returns_np)):
            r, l, s = float(returns_np[i]), int(lengths_np[i]), int(successes_np[i])
            self.train_metrics.episode_reward_history.append(r)
            self.train_metrics.episode_length_history.append(l)
            self.train_metrics.episode_success_history.append(s)
            self.train_metrics.env_total_episodes += 1

            ma_r = float(np.mean(self.train_metrics.episode_reward_history))
            ma_s = float(np.mean(self.train_metrics.episode_success_history))
            ma_l = float(np.mean(self.train_metrics.episode_length_history))

            self.run_logger.wandb_run.log({
                "train/reward": r,
                "train/success": s,
                "train/length": l,
                "train/moving_avg_reward":       ma_r,
                "train/moving_avg_success_rate": ma_s,
                "train/moving_avg_length":       ma_l,
                "train/fps":     fps,
                "train/frame":   total_trained,
                "train/episode": self.train_metrics.env_total_episodes,
                "train_steps":   total_trained,
                "train_episode": self.train_metrics.env_total_episodes,
            })

        pbar.set_postfix(ep=self.train_metrics.env_total_episodes,
                         ma_r=f"{ma_r:.2f}", fps=f"{fps:.0f}")
        self._log_agent_metrics(metrics, total_trained)

    def _log_agent_metrics(self, metrics: dict, train_steps: int):
        extras = {f"train/{k}": v for k, v in metrics.items()
                  if k != "episode_info" and isinstance(v, (int, float))}
        if extras:
            extras["train_steps"] = train_steps
            self.run_logger.wandb_run.log(extras)

    # ------------------------------------------------------------------ #
    # Evaluation — runs all 7 tasks separately
    # ------------------------------------------------------------------ #
    def _run_evaluation(self, global_train_frames: int):
        logger.info("=== Eval set %d (all %d tasks) ===", self.eval_set, self.num_tasks)

        all_task_metrics = {}

        for task_id in range(self.num_tasks):
            tracker = self.eval_trackers[task_id]
            tracker.initialize()

            # All eval envs run the same task
            task_ids = self.task_sampler.fixed(task_id)

            pbar = tqdm(total=self.num_eval_frames,
                        desc=f"eval task {task_id}", leave=False)

            rng = self.rng_manager.get_key()
            agent_metrics = self.agent.evaluate(
                self.agent_state, self.eval_env, rng,
                self.num_eval_frames, progress_bar=pbar,
                task_ids=task_ids,
            )
            pbar.close()

            # Process episode info
            episode_info = agent_metrics.get("episode_info")
            if episode_info is not None:
                returns = jnp.array(episode_info["returned_episode_returns"]).reshape(-1)
                lengths = jnp.array(episode_info["returned_episode_lengths"]).reshape(-1)
                done    = jnp.array(episode_info["returned_episode"]).reshape(-1)
                r = returns[done]; l = lengths[done]
                tracker.episode_reward_history.extend(r.tolist())
                tracker.episode_length_history.extend(l.tolist())
                tracker.episode_success_history.extend(
                    (r > 0).astype(jnp.int32).tolist()
                )
                tracker.env_total_episodes += int(done.sum())

            agg = {
                "avg_reward":  float(np.mean(tracker.episode_reward_history)),
                "avg_success": float(np.mean(tracker.episode_success_history)),
                "avg_length":  float(np.mean(tracker.episode_length_history)),
                "episodes":    tracker.env_total_episodes,
            }
            all_task_metrics[task_id] = agg

            # Log per-task eval
            self.run_logger.wandb_run.log({
                f"eval/task_{task_id}/avg_reward":  agg["avg_reward"],
                f"eval/task_{task_id}/avg_success": agg["avg_success"],
                f"eval/task_{task_id}/avg_length":  agg["avg_length"],
                f"eval/task_{task_id}/episodes":    agg["episodes"],
                "train_frames": global_train_frames,
            })

        # Log aggregate across all tasks
        avg_reward  = np.mean([m["avg_reward"]  for m in all_task_metrics.values()])
        avg_success = np.mean([m["avg_success"] for m in all_task_metrics.values()])
        avg_length  = np.mean([m["avg_length"]  for m in all_task_metrics.values()])

        self.run_logger.wandb_run.log({
            "eval/aggregate/avg_reward":  avg_reward,
            "eval/aggregate/avg_success": avg_success,
            "eval/aggregate/avg_length":  avg_length,
            "train_frames": global_train_frames,
        })

        # Console table
        rows = []
        for tid, m in all_task_metrics.items():
            rows.append([f"task_{tid}", f"{m['avg_reward']:.3f}",
                         f"{m['avg_success']:.3f}", m['episodes']])
        rows.append(["AGGREGATE", f"{avg_reward:.3f}", f"{avg_success:.3f}", ""])
        logger.info("\nEval set %d\n%s", self.eval_set,
                    tabulate(rows, headers=["task", "reward", "success", "episodes"],
                             tablefmt="grid"))

        # Checkpoint (use aggregate reward as the tracking metric)
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.on_validation_end(
                agent_state=self.agent_state,
                step=int(self.agent_state.runtime.train_steps),
                metrics={"eval_return": avg_reward, "eval_success": avg_success},
            )
        self.eval_set += 1
```

**What's different from the reference:**
- One training loop, not one per env. Tasks are sampled per segment.
- `task_ids` is passed as a kwarg to `agent.train()` and `agent.evaluate()`.
- Eval loops over all 7 tasks, each time setting all eval envs to the same
  task via `task_sampler.fixed(task_id)`.
- Metrics: train is aggregate; eval is 7 per-task panels + 1 aggregate panel.
- No FT/BT/forgetting computation. Just raw per-task and mean performance.

### `RNGManager` (`src/<pkg>/trainer/utils.py`)

```python
import jax

class RNGManager:
    def __init__(self, seed: int):
        self._key = jax.random.PRNGKey(seed)
        self._stack = []

    def get_key(self):
        self._key, sub = jax.random.split(self._key)
        return sub

    def checkpoint(self):
        self._stack.append(self._key)

    def restore(self):
        self._key = self._stack.pop()
```

---

## 8. Entry point

### `scripts/train.py`

```python
from argparse import ArgumentParser

from <pkg>.config import setup_environment
setup_environment()

from omegaconf import OmegaConf
import wandb
from <pkg>.config import load_config, configure_sweep_config
from <pkg>.agents import load_agent
from <pkg>.trainer import Trainer
from <pkg>.shared import setup_logger

logger = setup_logger(__name__)


def get_args():
    p = ArgumentParser()
    p.add_argument("--env-config",   required=True)
    p.add_argument("--agent-config", required=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--device",  type=int)
    p.add_argument("--sweep",   action="store_true")
    args, unknown = p.parse_known_args()
    args.set = [u.lstrip("-") for u in unknown if "=" in u]
    return args


def get_config(args):
    cfg = load_config(args.agent_config, args.env_config)
    if args.offline:             cfg.offline = True
    if args.device is not None:  cfg.device_id = args.device
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    if not args.sweep:
        return cfg
    run = wandb.init(project=cfg.project, entity=cfg.entity,
                     config=OmegaConf.to_container(cfg, resolve=True))
    return configure_sweep_config(base_config=cfg, sweep_config_dict=run.config)


def main():
    args   = get_args()
    config = get_config(args)
    agent  = load_agent(config)
    Trainer(config, agent).run()

if __name__ == "__main__":
    main()
```

### Agent registry

```python
# src/<pkg>/agents/registry.py
import importlib, pkgutil
from pathlib import Path
from typing import Callable
from omegaconf import OmegaConf
from .agent import Agent

class AgentRegistry:
    def __init__(self):
        self.agents: dict[str, Callable] = {}

    def register(self, name, factory):
        self.agents[name] = factory

    def discover(self, paths: list[tuple[str, str]]):
        infrastructure = ("__init__", "registry", "agent", "state")
        for path, package in paths:
            for item in Path(path).glob("*.py"):
                if item.stem not in infrastructure:
                    importlib.import_module(f"{package}.{item.stem}")

    def load(self, config: OmegaConf) -> Agent:
        from <pkg>.envs.registry import make_env
        env = make_env(config.env_id, config)
        obs_space = env.observation_space()
        act_space = env.action_space()

        # dreamer/storm read config.raw_obs_space for replay buffer init
        if hasattr(env, "raw_observation_space"):
            config.raw_obs_space = env.raw_observation_space()
        else:
            config.raw_obs_space = obs_space

        return self.agents[config.agent.name](config, obs_space, act_space)

AGENT_REGISTRY = AgentRegistry()

def register_agent(name: str):
    def decorator(fn):
        if name not in AGENT_REGISTRY.agents:
            AGENT_REGISTRY.register(name, fn)
        return fn
    return decorator

def load_agent(config: OmegaConf) -> Agent:
    return AGENT_REGISTRY.load(config)
```

---

## 9. Sweeps and SLURM cluster integration

The sweep system has three layers: W&B sweep YAML (defines the parameter
grid), `scripts/train.py --sweep` (a single run that talks to W&B), and a
launch mechanism (either `run_sweep.py` for local/interactive use or SLURM
job arrays for cluster use).

### How `--sweep` works with `wandb agent`

When `wandb agent <sweep_id>` runs, W&B:

1. Picks the next set of parameters from the sweep grid.
2. Sets those parameters as environment variables.
3. Invokes the `command:` from the sweep YAML.
4. That command runs `scripts/train.py --sweep`, which:
   a. Loads the base config from the YAML files specified in the command.
   b. Calls `wandb.init(...)` — this picks up the sweep's W&B run context.
   c. Reads `wandb.config` (the sweep-assigned parameters) and merges them
      back into the OmegaConf via `configure_sweep_config()`.
   d. Proceeds with training normally. The `RunLogger` does **not** call
      `wandb.init()` again — it detects that a run already exists (from the
      sweep) and reuses it.

**Critical detail for sweep compatibility in `RunLogger`:**

```python
@staticmethod
def _init_wandb_run(config):
    # When running under a sweep, wandb.init() was already called in
    # train.py::get_config(). wandb.init() is idempotent in the same
    # process — calling it again returns the existing run.
    # BUT: if you're NOT in a sweep, this is the first wandb.init() call.
    run = wandb.init(
        entity=config.entity,
        project=config.project,
        config=OmegaConf.to_container(config, resolve=True),
        mode="offline" if config.offline else "online",
    )
    # Only set name if we own the run (not a sweep-managed run)
    if run.sweep_id is None:
        run.name = "_".join([
            config.name, config.agent.name, config.experiment_name, run.id
        ])
    # ... artifact upload etc.
    return run
```

This means the same `train.py` works for:
- Direct runs: `python scripts/train.py --env-config ... --agent-config ...`
- Sweep runs: `wandb agent <sweep_id>` (which calls `train.py --sweep`)
- Offline runs: `python scripts/train.py --offline ...`

### Sweep YAML examples

**K-seed benchmark (grid over seeds):**

```yaml
# configs/sweeps/ppo_rnn_seeds.yaml
name: ppo_rnn_multitask_k10
entity: your-wandb-entity
project: ppo_rnn_multitask
method: grid

command:
  - ${env}
  - python
  - ${program}
  - --env-config
  - configs/env/<env_name>.yaml
  - --agent-config
  - configs/agent/ppo_rnn.yaml
  - --sweep

program: scripts/train.py

parameters:
  seed: {values: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]}
  trainer.num_train_frames: {value: 5000000}
```

**Hyperparameter sweep (random search):**

```yaml
# configs/sweeps/ppo_rnn_hpsearch.yaml
name: ppo_rnn_lr_sweep
entity: your-wandb-entity
project: ppo_rnn_multitask
method: random
metric:
  name: eval/aggregate/avg_reward
  goal: maximize

command:
  - ${env}
  - python
  - ${program}
  - --env-config
  - configs/env/<env_name>.yaml
  - --agent-config
  - configs/agent/ppo_rnn.yaml
  - --sweep

program: scripts/train.py

parameters:
  seed: {value: 42}
  agent.lr: {distribution: log_uniform_values, min: 1e-5, max: 1e-3}
  agent.entropy_coef: {distribution: log_uniform_values, min: 0.001, max: 0.1}
  agent.clip_eps: {values: [0.1, 0.2, 0.3]}
```

**Multi-agent sweep (compare PPO, Dreamer, STORM):**

Create one sweep YAML per agent, each targeting the same W&B project.
Or use the same YAML and override `--agent-config` via sweep parameters.

### SLURM job scripts

#### `scripts/job_sweep.slurm` — one agent per SLURM array task

```bash
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/sweep_%A_%a.log
#SBATCH --export=ALL

# Environment variables (passed via sbatch or launch_sweep.sh):
#   SWEEP_ID        - (required) entity/project/sweep_id
#   RUNS_PER_AGENT  - (optional) number of runs this agent handles (default: 1)

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not set"
    exit 1
fi

RUNS_PER_AGENT=${RUNS_PER_AGENT:-1}

# Activate environment (adjust to your cluster)
source /software/anaconda3/etc/profile.d/conda.sh
conda activate your_env

echo "============================================================"
echo "Sweep Agent — SLURM Array Task"
echo "============================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Node:            $(hostname)"
echo "Sweep ID:        $SWEEP_ID"
echo "Runs per agent:  $RUNS_PER_AGENT"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"

# Run the W&B agent
# --count N means this agent will pick up and run N jobs from the sweep queue
wandb agent --count "$RUNS_PER_AGENT" "$SWEEP_ID"

echo "Agent $SLURM_ARRAY_TASK_ID completed"
```

#### `scripts/launch_sweep.sh` — create sweep + submit SLURM array

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$(dirname "$SCRIPT_DIR")/logs"
mkdir -p "$LOGS_DIR"

usage() {
    cat << 'EOF'
Usage:
  ./scripts/launch_sweep.sh <sweep_config.yaml> [options]
  ./scripts/launch_sweep.sh --sweep-id <existing_id> [options]

Options:
  -n, --num-agents NUM    Number of SLURM array tasks (default: 10)
  -r, --runs-per NUM      Runs per agent (default: 1)
  -N, --nodes LIST        SLURM --nodelist
  -x, --exclude LIST      SLURM --exclude
  -t, --time TIME         Max time (default: 24:00:00)
  -m, --mem MEM           Memory (default: 32G)
  --dry-run               Print command without submitting

Examples:
  # K=10 seeds: create sweep from YAML, launch 10 agents each doing 1 run
  ./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_seeds.yaml -n 10 -r 1

  # HP search: 20 agents, each doing 5 runs (100 total random trials)
  ./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_hpsearch.yaml -n 20 -r 5

  # Reuse existing sweep
  ./scripts/launch_sweep.sh --sweep-id entity/project/abc123 -n 10 -r 1
EOF
}

NUM_AGENTS=10
RUNS_PER_AGENT=1
NODES=""
EXCLUDE=""
TIME="24:00:00"
MEM="32G"
DRY_RUN=false
SWEEP_ID=""
SWEEP_CONFIG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --sweep-id)     SWEEP_ID="$2";       shift 2 ;;
        -n|--num-agents) NUM_AGENTS="$2";    shift 2 ;;
        -r|--runs-per)  RUNS_PER_AGENT="$2"; shift 2 ;;
        -N|--nodes)     NODES="$2";          shift 2 ;;
        -x|--exclude)   EXCLUDE="$2";        shift 2 ;;
        -t|--time)      TIME="$2";           shift 2 ;;
        -m|--mem)       MEM="$2";            shift 2 ;;
        --dry-run)      DRY_RUN=true;        shift ;;
        -h|--help)      usage; exit 0 ;;
        -*)             echo "Unknown option: $1"; usage; exit 1 ;;
        *)              SWEEP_CONFIG="$1";   shift ;;
    esac
done

# Create sweep if no ID given
if [ -z "$SWEEP_ID" ]; then
    if [ -z "$SWEEP_CONFIG" ]; then
        echo "Error: provide a sweep YAML or --sweep-id"
        usage; exit 1
    fi
    echo "Creating W&B sweep from $SWEEP_CONFIG ..."
    SWEEP_ID=$(wandb sweep "$SWEEP_CONFIG" 2>&1 | grep -oP '(?<=wandb agent )\S+')
    echo "Created sweep: $SWEEP_ID"
fi

echo ""
echo "Sweep ID:        $SWEEP_ID"
echo "Num agents:      $NUM_AGENTS"
echo "Runs per agent:  $RUNS_PER_AGENT"
echo "Total runs:      $((NUM_AGENTS * RUNS_PER_AGENT))"
echo ""

# Build sbatch command
SBATCH_CMD="sbatch --array=0-$((NUM_AGENTS - 1)) --time=$TIME --mem=$MEM"
[ -n "$NODES" ]   && SBATCH_CMD="$SBATCH_CMD --nodelist=$NODES"
[ -n "$EXCLUDE" ] && SBATCH_CMD="$SBATCH_CMD --exclude=$EXCLUDE"
SBATCH_CMD="$SBATCH_CMD $SCRIPT_DIR/job_sweep.slurm"

echo "Command: SWEEP_ID=$SWEEP_ID RUNS_PER_AGENT=$RUNS_PER_AGENT $SBATCH_CMD"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Not submitted"
    exit 0
fi

export SWEEP_ID RUNS_PER_AGENT
$SBATCH_CMD

echo ""
echo "Submitted! Monitor with: squeue -u \$USER"
echo "Logs:    tail -f logs/sweep_*.log"
echo "Cancel:  scancel <job_id>"
```

### `scripts/run_sweep.py` — local parallel agent launcher (non-SLURM)

Available at `reference_files/scripts/run_sweep.py`. Used when you have direct GPU access
(e.g. a dev machine or interactive allocation) and want to run multiple
sweep agents in parallel without SLURM.

```bash
# Local: 5 agents, 1 run each, across 2 GPUs
python scripts/run_sweep.py <SWEEP_ID> --num-agents 5 --count 1 --gpus 0 1
```

### W&B sweep flow summary

```
[your machine / login node]
  wandb sweep configs/sweeps/X.yaml          → creates sweep, prints SWEEP_ID
  ./scripts/launch_sweep.sh ... -n 10 -r 1   → submits SLURM array of 10 tasks

[SLURM cluster, 10 nodes]
  job_sweep.slurm (array task 0)
    wandb agent --count 1 $SWEEP_ID
      → W&B picks seed=42, invokes:
        python scripts/train.py --env-config ... --agent-config ... --sweep
          → train.py calls wandb.init(), reads wandb.config (seed=42),
            merges into OmegaConf, creates Trainer, runs training
          → Trainer creates RunLogger (reuses the sweep's wandb.init()),
            logs train/*, eval/task_{0..6}/*, eval/aggregate/*

  job_sweep.slurm (array task 1)
    wandb agent --count 1 $SWEEP_ID
      → W&B picks seed=43, same flow ...

  ... (8 more in parallel)
```

---

## 10. Checkpointing

Use orbax. Save **only** `train_state` (params + optimizer), never `runtime`.
Copy `reference_files/trainer/checkpoint.py`, fix `from cl.` imports.

`CheckpointCallback` exposes:
- `on_validation_end(agent_state, step, metrics)` — saves checkpoint, rotates
  `keep_last`, optionally tracks best by `metrics["eval_return"]` (which is now
  the aggregate reward across all 7 tasks).

---

## 11. Migration order — parallelized across Claude Code agents

The refactor is designed so that multiple Claude Code agents can work on
independent modules simultaneously using git worktrees. Each agent gets its
own branch and worktree; you merge them in order at the end.

### Phase 0: Bootstrap (you do this, one agent, ~5 min)

**Create the skeleton first** — every other agent needs the directory layout
and shared interfaces to exist.

1. Create the directory layout from §1. Empty `__init__.py`s everywhere.
2. Write `src/<pkg>/config/env.py` (the `setup_environment()` function).
3. Write `src/<pkg>/config/utils.py` (`load_config`, `configure_sweep_config`).
4. Write `src/<pkg>/shared/logger.py` (`setup_logger`).
5. Write `src/<pkg>/agents/agent.py` (the `Agent` dataclass from §4).
6. Copy `reference_files/agents/state.py` → `src/<pkg>/agents/state.py`.
   Fix imports. Remove `current_num_actions` if all tasks share the same
   action space, keep `current_task_id`.
7. Write `src/<pkg>/agents/registry.py` (from §8).
8. Write `src/<pkg>/metrics/tracker.py` (from §5).
9. Create `configs/env/<env_name>.yaml` and `configs/agent/ppo_rnn.yaml`.
10. Commit to `main`. Push.

This gives every parallel agent a stable base to build on.

### Phase 1: Parallel work (spawn 5 agents simultaneously)

After Phase 0 is on `main`, launch these agents in parallel. Each works in
its own worktree/branch. **They share no files** — that's the whole point.

---

**Agent 1: `env-layer`** — Multi-task environment wrapper

Branch: `feat/env-layer`

Task: implement the multi-task environment layer from §3.

Files to create:
- `src/<pkg>/envs/registry.py` — `make_env(env_id, config)` factory
- `src/<pkg>/envs/task_sampler.py` — `TaskSampler` class
- `src/<pkg>/envs/multitask_wrapper.py` — `MultiTaskEnvWrapper` class

Acceptance criteria:
- `TaskSampler("round_robin", 7, 32).sample()` returns shape `(32,)` with
  values cycling through `0..6`.
- `TaskSampler("uniform_random", 7, 32).sample(rng)` returns random task ids.
- `task_sampler.fixed(3)` returns all-3 array.
- `MultiTaskEnvWrapper` wraps the base env, delegates `step()`/`reset()`,
  modifies reward based on task id, and exposes `get_task_embeddings()`.
- Write a small test in `tests/test_env_layer.py` that verifies the above.

Does NOT depend on: agents, trainer, metrics, W&B.

---

**Agent 2: `trainer-core`** — Trainer + RunLogger + RNGManager + entry point

Branch: `feat/trainer-core`

Task: implement the training orchestration layer from §6, §7, §8.

Files to create:
- `src/<pkg>/trainer/utils.py` — `RNGManager`
- `src/<pkg>/trainer/run_logger.py` — `RunLogger` (with sweep-aware
  `wandb.init` from §9)
- `src/<pkg>/trainer/trainer.py` — `Trainer` class (full multi-task version)
- `scripts/train.py` — entry point

The trainer should import from `<pkg>.agents.agent` (the dataclass),
`<pkg>.agents.registry` (`load_agent`), `<pkg>.envs.registry` (`make_env`),
`<pkg>.envs.task_sampler` (`TaskSampler`), `<pkg>.metrics.tracker`
(`MetricsTracker`), and `<pkg>.trainer.run_logger` (`RunLogger`). All of
these exist from Phase 0 or from Agent 1's branch (they'll be merged before
integration testing).

Use the full `Trainer` code from §7 — the multi-task training loop with
`_log_training_metrics`, `_log_agent_metrics`, `_run_evaluation`.

Acceptance criteria:
- `Trainer.__init__` creates `RunLogger`, `RNGManager`, `TaskSampler`,
  train/eval `MetricsTracker`s, and optionally `CheckpointCallback`.
- `Trainer.run()` trains for `num_train_frames` with periodic eval across
  all 7 tasks.
- `RunLogger` handles both sweep and non-sweep modes correctly (checks
  `run.sweep_id`).
- `scripts/train.py` parses args, loads config, calls `load_agent` +
  `Trainer`.

Does NOT depend on: any specific agent implementation, checkpoint.py (can
stub it), specific env internals.

---

**Agent 3: `ppo-rnn`** — PPO-RNN agent factory

Branch: `feat/ppo-rnn`

Task: wrap the existing PPO-RNN code into a factory function matching the
`Agent` dataclass interface from §4.

Files to create/modify:
- `src/<pkg>/agents/ppo_rnn.py` — `@register_agent("ppo_rnn")` factory

Requirements:
- Accept `task_ids` kwarg in `train()` and `evaluate()`.
- Look up task embeddings (from `config.num_tasks`, `config.task_embedding_dim`)
  and inject them into the network's forward pass.
- Return `episode_info` with the three standard arrays:
  `returned_episode_returns`, `returned_episode_lengths`, `returned_episode`.
- Return any scalar losses/diagnostics as extra keys in the metrics dict.
- Use `AgentState` from `state.py` with `train_state` and `runtime` substates.

Does NOT depend on: trainer, run_logger, metrics, env wrapper internals
(just the `task_ids` array shape).

---

**Agent 4: `copy-dreamer-storm`** — Copy and adapt world-model agents

Branch: `feat/world-model-agents`

Task: copy DreamerV3 and STORM from `reference_files/` in this repo, fix
imports, adapt for multi-task.

**Source files are already in this repo** under `reference_files/`. Copy
them to the target locations:

```
reference_files/agents/dreamer.py       → src/<pkg>/agents/dreamer.py
reference_files/agents/storm.py         → src/<pkg>/agents/storm.py
reference_files/agents/utils.py         → src/<pkg>/agents/utils.py
reference_files/agents/commons/         → src/<pkg>/agents/commons/    (entire dir)
reference_files/agents/policy/          → src/<pkg>/agents/policy/     (entire dir)
reference_files/agents/world_models/    → src/<pkg>/agents/world_models/ (entire dir)
reference_files/configs/agent/dreamerv3.yaml → configs/agent/dreamerv3.yaml
reference_files/configs/agent/storm.yaml     → configs/agent/storm.yaml
```

Edits to `dreamer.py` and `storm.py`:
1. Fix all `from cl.` → `from <pkg>.` imports (use sed).
2. Delete `set_environment` / `on_environment_end` definitions and their
   entries in the returned `Agent(...)` kwargs.
3. Delete `visualize_buffer_task_distribution` calls.
4. Add `task_ids` kwarg to `train()` and `evaluate()`.
5. Inject task embedding into observation before the encoder/world-model
   (right after `normalize_image(extract_image_obs(...))`).
6. Store per-env `task_id` in replay buffer transitions (replace the
   scalar `current_task_id` with the per-env array).

Edits to `commons/`, `policy/`, `world_models/`:
- Fix `from cl.` imports only. No logic changes.

Run the sed script:
```bash
find src/<pkg>/agents/{commons,policy,world_models} \
  src/<pkg>/agents/{dreamer,storm,utils,state}.py \
  -type f -name "*.py" \
  -exec sed -i '' 's|from cl\.|from <pkg>.|g; s|import cl\.|import <pkg>.|g' {} +
```

Does NOT depend on: trainer, env wrapper, metrics.

---

**Agent 5: `sweep-infra`** — Sweep configs + SLURM scripts + checkpoint

Branch: `feat/sweep-infra`

Task: create all the operational infrastructure for running on a cluster.

Source files from `reference_files/`:
```
reference_files/scripts/run_sweep.py      → scripts/run_sweep.py  (copy verbatim)
reference_files/trainer/checkpoint.py     → src/<pkg>/trainer/checkpoint.py (fix imports)
```

Files to create from the templates in §9:
- `scripts/job_sweep.slurm` — adjust conda env name, mail, GPU type
- `scripts/launch_sweep.sh`
- `configs/sweeps/ppo_rnn_seeds.yaml` — K-seed benchmark sweep
- `configs/sweeps/ppo_rnn_hpsearch.yaml` — HP search sweep
- `configs/sweeps/dreamerv3_seeds.yaml` — K-seed for dreamer
- `configs/sweeps/storm_seeds.yaml` — K-seed for storm

Acceptance criteria:
- `wandb sweep configs/sweeps/ppo_rnn_seeds.yaml` succeeds.
- `./scripts/launch_sweep.sh --dry-run configs/sweeps/ppo_rnn_seeds.yaml -n 10 -r 1`
  prints the correct sbatch command without submitting.
- `python scripts/run_sweep.py --help` shows usage.
- Checkpoint module imports cleanly.

Does NOT depend on: any agent implementation, trainer logic (just the
interfaces in `agent.py` and `state.py`).

---

### Phase 2: Merge and integrate (sequential, one agent)

After all 5 agents complete, merge branches in this order:

```
main
  ← feat/env-layer          # no conflicts (new files only)
  ← feat/sweep-infra        # no conflicts (new files only)
  ← feat/ppo-rnn            # no conflicts (new file: ppo_rnn.py)
  ← feat/world-model-agents # no conflicts (new files in agents/)
  ← feat/trainer-core       # no conflicts (new files in trainer/ + scripts/)
```

Then run the integration test:

```bash
# Smoke test with tiny config
python scripts/train.py \
  --env-config configs/env/<env_name>.yaml \
  --agent-config configs/agent/ppo_rnn.yaml \
  trainer.num_train_frames=10000 \
  trainer.eval_interval_frames=5000 \
  --offline

# Verify:
# 1. Training loop runs without error
# 2. Eval runs all 7 tasks and prints the tabulate table
# 3. Checkpoints are saved in results/{run_id}/checkpoints/
# 4. No wandb errors in offline mode
```

Then test dreamer and storm:

```bash
python scripts/train.py \
  --env-config configs/env/<env_name>.yaml \
  --agent-config configs/agent/dreamerv3.yaml \
  trainer.num_train_frames=10000 \
  --offline

python scripts/train.py \
  --env-config configs/env/<env_name>.yaml \
  --agent-config configs/agent/storm.yaml \
  trainer.num_train_frames=10000 \
  --offline
```

### Phase 3: Cleanup (one agent)

1. Delete any old code that the new structure replaces.
2. Test a real sweep: `wandb sweep configs/sweeps/ppo_rnn_seeds.yaml`,
   then `python scripts/run_sweep.py <ID> --num-agents 2 --count 1 --gpus 0`.
   Verify two runs appear in W&B with different seeds.
3. Verify SLURM: `./scripts/launch_sweep.sh --dry-run ...` looks correct.

### Agent dependency graph

```
Phase 0 (bootstrap, on main)
    │
    ├──→ Agent 1: env-layer           ──┐
    ├──→ Agent 2: trainer-core        ──┤
    ├──→ Agent 3: ppo-rnn             ──┼──→ Phase 2 (merge + integrate)
    ├──→ Agent 4: copy-dreamer-storm  ──┤       │
    └──→ Agent 5: sweep-infra         ──┘       ▼
                                          Phase 3 (cleanup)
```

All 5 agents run fully in parallel — none depends on another's output. They
only depend on Phase 0 (the skeleton + interfaces on `main`). Merge order
doesn't matter for correctness (no overlapping files), but the order above
is easiest for conflict resolution.

### Spawning the agents in Claude Code

After Phase 0 is committed and pushed, run this in your other repo:

```
Use 5 subagents in parallel with worktree isolation. Each agent gets its own
branch. Here is the refactor guide: [paste this document or reference the file]

Agent 1: "env-layer" — implement §3 (task_sampler.py, multitask_wrapper.py, registry.py)
Agent 2: "trainer-core" — implement §6 + §7 + §8 (RunLogger, Trainer, train.py)
Agent 3: "ppo-rnn" — implement §4 PPO-RNN factory (ppo_rnn.py)
Agent 4: "copy-dreamer-storm" — copy and adapt dreamer.py, storm.py, commons/, policy/, world_models/ from [reference repo path]
Agent 5: "sweep-infra" — create sweep YAMLs, SLURM scripts, copy checkpoint.py
```

Each agent should use `isolation: "worktree"` so it works on a clean copy.
When all 5 complete, merge the branches sequentially and run the integration
test.

---

## 12. Invariants to preserve

- **Trainer never imports anything agent-specific.** No
  `if config.agent.name == "ppo": ...`.
- **Agent never imports anything trainer-specific.** No `import wandb`.
- **One source of truth for metric names** — the `MetricsTracker`.
- **Step metrics registered once** in `Trainer.__init__` /
  `RunLogger.register_metrics`.
- **`AgentState` is immutable** and threaded through the trainer.
- **Resolved config uploaded as a W&B artifact every run.**
- **`results_dir = {results_path}/{wandb_run_id}`.**
- **Only `train_state` is checkpointed.**
- **Task identity flows through `task_ids` kwarg**, not through env switching
  or global state. The agent receives it; the trainer decides it.
- **Eval always runs all 7 tasks** with fixed task assignment. Training
  mixes tasks according to `task_sampling`.

---

## 13. Usage guide and codebase documentation

This section is the user-facing documentation. Put it (or a version of it) in
your repo's README or a separate USAGE.md.

### Prerequisites

- Python >= 3.11
- CUDA 12+ with a compatible GPU
- W&B account (`wandb login` completed)
- Conda environment with the project installed (`pip install -e .`)

### Quick start — single run

```bash
# 1. Set up XLA before any JAX import (handled by train.py internally)
# 2. Run training
python scripts/train.py \
  --env-config configs/env/<env_name>.yaml \
  --agent-config configs/agent/ppo_rnn.yaml

# Override any config value via dotlist notation:
python scripts/train.py \
  --env-config configs/env/<env_name>.yaml \
  --agent-config configs/agent/ppo_rnn.yaml \
  seed=123 \
  trainer.num_train_frames=100000 \
  agent.lr=1e-4

# Run offline (no W&B sync):
python scripts/train.py \
  --env-config configs/env/<env_name>.yaml \
  --agent-config configs/agent/ppo_rnn.yaml \
  --offline

# Force a specific GPU:
python scripts/train.py \
  --env-config configs/env/<env_name>.yaml \
  --agent-config configs/agent/ppo_rnn.yaml \
  --device 1
```

### Quick start — K-seed benchmark on a cluster

```bash
# Step 1: Create the W&B sweep (run from login node)
wandb sweep configs/sweeps/ppo_rnn_seeds.yaml
# Output: Created sweep with ID: your-entity/your-project/abc123

# Step 2: Submit SLURM job array (10 seeds = 10 array tasks)
./scripts/launch_sweep.sh --sweep-id your-entity/your-project/abc123 -n 10 -r 1

# Step 3: Monitor
squeue -u $USER                    # check SLURM queue
tail -f logs/sweep_<jobid>_*.log   # watch a specific agent's log
# Open W&B dashboard to see runs appearing in real time
```

### Quick start — hyperparameter search on a cluster

```bash
# Step 1: Create sweep with random search config
wandb sweep configs/sweeps/ppo_rnn_hpsearch.yaml

# Step 2: Launch 20 agents, each running 5 trials (= 100 total)
./scripts/launch_sweep.sh --sweep-id your-entity/your-project/xyz789 -n 20 -r 5

# Step 3: Check W&B for best run, then do a seed sweep with those params
```

### Quick start — local parallel runs (no SLURM)

```bash
# For dev machines with multiple GPUs
wandb sweep configs/sweeps/ppo_rnn_seeds.yaml
python scripts/run_sweep.py <SWEEP_ID> --num-agents 5 --count 1 --gpus 0 1
```

### Available agents

| Agent     | Config file                    | Description                           |
| --------- | ------------------------------ | ------------------------------------- |
| `ppo_rnn` | `configs/agent/ppo_rnn.yaml`   | PPO with LSTM backbone                |
| `dreamerv3` | `configs/agent/dreamerv3.yaml` | DreamerV3 world model agent           |
| `storm`   | `configs/agent/storm.yaml`     | STORM world model agent               |

Switch agents by changing `--agent-config`:

```bash
python scripts/train.py \
  --env-config configs/env/<env_name>.yaml \
  --agent-config configs/agent/dreamerv3.yaml
```

### Configuration reference

Configuration is layered: **env YAML** (experiment context) is loaded first,
then **agent YAML** (model hyperparameters) is merged on top (agent wins on
conflicts). CLI dotlist overrides are applied last. In sweep mode, W&B sweep
parameters override everything.

#### Priority order (highest wins):

```
W&B sweep parameters  >  CLI dotlist overrides  >  agent YAML  >  env YAML
```

#### Environment config (`configs/env/<env_name>.yaml`)

| Key                               | Type    | Description                                       |
| --------------------------------- | ------- | ------------------------------------------------- |
| `experiment_name`                 | str     | Human label for grouping runs in W&B              |
| `results_path`                    | str     | Base directory for checkpoints and local artifacts |
| `offline`                         | bool    | If true, W&B runs in offline mode                 |
| `entity`                          | str     | W&B team/user entity                              |
| `project`                         | str     | W&B project name                                  |
| `seed`                            | int     | Global random seed                                |
| `env_id`                          | str     | Environment ID (gym-style)                        |
| `num_tasks`                       | int     | Number of tasks (7)                               |
| `task_embedding_dim`              | int     | Task embedding dimension                          |
| `task_sampling`                   | str     | `"round_robin"` or `"uniform_random"`             |
| `env.num_parallel_envs`           | int     | Vectorized envs during training                   |
| `env.num_parallel_envs_eval`      | int     | Vectorized envs during eval                       |
| `trainer.num_train_frames`        | int     | Total training env frames                         |
| `trainer.num_eval_frames`         | int     | Frames per task per eval checkpoint               |
| `trainer.eval_interval_frames`    | int     | Train frames between eval checkpoints (null=off)  |
| `trainer.log_interval_episodes`   | int     | Console log frequency                             |
| `metrics_tracker.moving_avg_window_size` | int | Rolling window for moving averages          |

#### Agent config (`configs/agent/<agent>.yaml`)

Each agent config sets `agent.name` (must match `@register_agent` name) and
agent-specific hyperparameters. The `agent.checkpoint.*` block controls
checkpointing:

| Key                          | Type  | Description                                   |
| ---------------------------- | ----- | --------------------------------------------- |
| `agent.checkpoint.enabled`   | bool  | Enable periodic checkpoint saving              |
| `agent.checkpoint.interval`  | int   | Save every N training steps                    |
| `agent.checkpoint.keep_last` | int   | Keep last N checkpoints (0 = keep all)         |
| `agent.checkpoint.save_best` | bool  | Track and save best checkpoint by eval return  |

### W&B metrics reference

All metrics logged during a run:

#### Training metrics (logged per completed episode)

| W&B key                         | Description                         |
| ------------------------------- | ----------------------------------- |
| `train/reward`                  | Raw episode return (aggregate)      |
| `train/success`                 | 1 if return > 0, else 0             |
| `train/length`                  | Episode length in steps             |
| `train/moving_avg_reward`       | Rolling mean over last N episodes   |
| `train/moving_avg_success_rate` | Rolling mean success rate           |
| `train/moving_avg_length`       | Rolling mean episode length         |
| `train/fps`                     | Training frames per second          |
| `train/frame`                   | Current training frame              |
| `train/episode`                 | Cumulative episode count            |
| `train/<agent_key>`             | Any scalar the agent returns        |

Step metric: `train_steps` (total training frames).

#### Evaluation metrics (logged per eval checkpoint)

| W&B key                            | Description                       |
| ---------------------------------- | --------------------------------- |
| `eval/task_{i}/avg_reward`         | Mean reward on task i             |
| `eval/task_{i}/avg_success`        | Mean success rate on task i       |
| `eval/task_{i}/avg_length`         | Mean episode length on task i     |
| `eval/task_{i}/episodes`           | Number of eval episodes           |
| `eval/aggregate/avg_reward`        | Mean reward across all 7 tasks    |
| `eval/aggregate/avg_success`       | Mean success across all 7 tasks   |
| `eval/aggregate/avg_length`        | Mean length across all 7 tasks    |

Step metric: `train_frames` (global training frames at eval time).

#### W&B run name format

```
{env_config_stem}_{agent_config_stem}_{agent_name}_{experiment_name}_{wandb_run_id}
```

Example: `myenv_ppo_rnn_ppo_rnn_default_a1b2c3d4`

#### W&B artifacts

Every run uploads a `config` artifact (type: `config`) containing the fully
resolved OmegaConf as `config.yaml`. This is the single source of truth for
reproducing a run — it includes all sweep overrides, CLI overrides, and
merged config values.

### Directory structure

#### Source code (`src/<pkg>/`)

```
config/         Config loading and XLA setup
  env.py          setup_environment() — call before any JAX import
  utils.py        load_config(), configure_sweep_config()

shared/         Cross-cutting utilities
  logger.py       setup_logger() — consistent Python logging

envs/           Environment wrappers
  registry.py     make_env(env_id, config) — environment factory
  multitask_wrapper.py  Reward routing + task embedding lookup
  task_sampler.py       Round-robin / random task selection

agents/         Agent implementations
  agent.py        Agent dataclass (the interface)
  state.py        AgentState, RuntimeState, PolicyParams, OptState
  registry.py     @register_agent decorator + load_agent()
  utils.py        RatioTracker, sg(), get_agent_parameter_count()
  ppo_rnn.py      PPO-RNN factory
  dreamer.py      DreamerV3 factory (copied from reference)
  storm.py        STORM factory (copied from reference)
  commons/        Shared NN building blocks (distributions, networks,
                  normalizers, optimizers, replay buffer, preprocessing)
  policy/         Actor-critic heads (MLP policy, imagination rollouts)
  world_models/   World model implementations (DreamerV3, STORM)

metrics/        Metrics tracking
  tracker.py      MetricsTracker — rolling stats, metric name registry

trainer/        Training orchestration
  trainer.py      Trainer — main loop, eval, metric logging
  run_logger.py   RunLogger — W&B init, naming, artifact upload
  checkpoint.py   CheckpointCallback — orbax save/load/rotation
  utils.py        RNGManager — deterministic key splitting
```

#### Configs (`configs/`)

```
env/            One YAML per environment/experiment setup
agent/          One YAML per agent (hyperparameters)
sweeps/         One YAML per sweep (seed benchmarks, HP searches)
```

#### Scripts (`scripts/`)

```
train.py          Entry point — loads config, creates agent + trainer, runs
run_sweep.py      Local parallel W&B agent launcher (non-SLURM)
launch_sweep.sh   SLURM job array submitter (cluster use)
job_sweep.slurm   SLURM job script (one wandb agent per array task)
```

#### Outputs

```
{results_path}/{wandb_run_id}/              Per-run directory
  checkpoints/                               Orbax checkpoint files
    step_NNNN/                               Checkpoint at step N
    best/                                    Best checkpoint by eval return
logs/                                        SLURM log files
  sweep_{job_id}_{array_id}.log              One log per sweep agent
```

### Adding a new agent

1. Create `src/<pkg>/agents/my_agent.py`.
2. Define a factory function decorated with `@register_agent("my_agent")`.
3. The factory receives `(config, obs_space, act_space)` and returns an
   `Agent(init=..., train=..., evaluate=...)`.
4. `train()` must return `(new_state, metrics_dict)` where `metrics_dict`
   contains `"episode_info"` with the standard arrays, plus any scalar
   metrics you want auto-logged.
5. Create `configs/agent/my_agent.yaml` with `agent.name: my_agent`.
6. Run: `python scripts/train.py --env-config ... --agent-config configs/agent/my_agent.yaml`

No changes needed to `Trainer`, `RunLogger`, `MetricsTracker`, or any other
infrastructure file.

### Troubleshooting

**"Agent X not registered"** — the auto-discovery imports every `.py` in
`src/<pkg>/agents/`. Check that your file isn't in a subdirectory that
lacks `__init__.py`, and that the `@register_agent` decorator is present.

**Sweep runs all use the same seed** — make sure `seed` is in the sweep
YAML's `parameters:` block, not hardcoded in the env YAML when running
sweeps. The sweep override happens via `configure_sweep_config()`.

**W&B metrics have wrong x-axis** — ensure you're including the step metric
key (`train_steps` or `train_frames`) in every `wandb.log()` call. The
`define_metric()` registration only tells W&B which key to *look for*; you
must still log it.

**SLURM job fails immediately** — check `logs/sweep_*.log`. Common issues:
conda env not found (fix the `conda activate` line in `job_sweep.slurm`),
`SWEEP_ID` not exported (use `launch_sweep.sh` instead of raw `sbatch`).

**Out of GPU memory** — never pass `AgentState` (which contains the replay
buffer) into `jax.jit`. Pass only `train_state` and individual arrays. See
the docstring in `state.py::RuntimeState` for the JIT hygiene rules.
