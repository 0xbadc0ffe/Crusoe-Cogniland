# Cogniland — one memory task, three agents

Three reinforcement-learning agents — **PPO+GRU** (model-free), **DreamerV3**
(RSSM world model), and **STORM** (Transformer world model) — trained and
evaluated on the **identical** POMDP memory task, reward, and fixed 6k-map
dataset, as a substrate for mechanistic interpretability (belief probing and
steering).

**Start here → [`final_models/`](final_models/README.md)** — the three trained
checkpoints, the shared-environment proof (`ENVIRONMENT.md`), the architecture
notes (`ARCHITECTURES.md`), and one-command reproduction per agent.

## The task: `bridge_tunnel` fork_wall

A 32×64 procedural map whose terrain type (lakes / rocky / balanced) is visible
early, then hidden behind an information-free grass corridor (`mem_gap=16`).
At the far wall the agent must choose between two doors; only the door matching
the *remembered* category is rewarded. Reward: −0.01/step slack,
+0.015·Δ(cost-to-go) potential shaping toward the correct door, +3.0 at the
correct door. The wrong door ends the episode with no bonus.

Held-out results (`data/bridge_tunnel/forkwall6k/test.pkl`, chance for a
memoryless policy ≈ 67%):

| Agent | paradigm | held-out success |
|---|---|---:|
| PPO+GRU | model-free policy gradient | 98–99.5% |
| DreamerV3 25M | RSSM world model, imagination | 98.0% |
| STORM | Transformer world model, imagination | **99.3%** |

Each agent needed a different ingredient to escape the memoryless
"constant-door" local optimum: PPO an annealed high-entropy schedule, Dreamer
and STORM a training/backprop window longer than the ~75-step evidence→door
dependency (`batch_length ≥ 128`). Details in `final_models/ARCHITECTURES.md`.

## Layout

```
final_models/            ★ the three checkpoints + docs + repro commands
src/cogniland/
  bridge_tunnel/         the env (PyTorch/numpy + bit-identical pure-JAX port)
  memory_env/            MiniGrid MemoryEnv fork (T-maze; secondary task)
  assets/                sprites for rendering
purejaxwm/               in-tree DreamerV3 algorithm library (JAX)
external/r2dreamer/      DreamerV3 pipeline used for the Dreamer result
STORM_model/             STORM pipeline (JAX/Flax; agent `storm2`)
scripts/
  bridge_tunnel/         PPO training, eval, viz, slurm launchers
  memory_env/            memory_env training + analysis
  mechinterp/            activation datasets, probing, steering
  figures/               figure generation
configs/                 experiment configs (+ configs/bridge_tunnel/REGISTRY.md)
released_models/         earlier released agents (git-LFS)
data/bridge_tunnel/      fixed map datasets (forkwall6k train/test + val maps)
tests/                   env contract + JAX↔PyTorch parity + purejaxwm units
paper/  docs/            write-ups
```

## Quick start

```bash
# environment for the env + PPO + purejaxwm (deps in pyproject.toml)
python -m venv .venv && source .venv/bin/activate && pip install -e .

pytest tests/                       # env contract + parity + algo units (81 tests)

# regenerate the shared dataset (deterministic)
python scripts/bridge_tunnel/make_forkwall_dataset.py

# train each agent — exact commands in final_models/{ppo,dreamer,storm}/README.md
```

## Evaluation convention

Report the **TRUE door metric** — final cell inside the correct-door set —
never episode `return > 0`: fast wrong-door episodes collect more shaping than
slack and the return proxy counts them as successes (≈6–13pp inflation).
PPO and STORM are evaluated with sampled actions (their operating mode),
Dreamer deterministically. STORM's evaluator:
`STORM_model/scripts/true_eval_w.py --sampled --env-context 128`.
