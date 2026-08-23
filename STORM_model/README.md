# STORM pipeline — Transformer world model on `bridge_tunnel` fork_wall

JAX/Flax implementation of **STORM** (Stochastic Transformer-based wORld
Model, [Zhang et al. 2023](https://arxiv.org/abs/2310.09615)), trained on the
shared fork_wall memory task. The trained checkpoint and its numbers live in
`../final_models/storm/`.

The agent is `storm2` (`cl/agents/storm2.py`) — a faithful port of the
original [weipu-zhang/STORM](https://github.com/weipu-zhang/STORM) training
scheme:

* **training**: the causal transformer processes whole replay sequences in
  parallel (episode-segment attention mask); position *t*−1's output `h_t`
  summarizes history and conditions the prior over `z_t`, the reward and the
  continuation heads (so both can be belief-dependent);
* **features**: the actor-critic reads `concat(z_t, h_t)` — `z` from the
  current observation, `h` from attention over up to `batch_length` past
  (z, a) tokens;
* **acting**: a rolling window of the last `env_context` tokens is re-encoded
  each step (windows are tiny; no KV-cache plumbing needed);
* **imagination**: rollouts are primed with a context of real posterior
  latents before the policy takes over (original
  `ImagineContextLength`/`ImagineBatchLength` recipe).

An earlier in-repo STORM variant called the transformer on 1-token sequences
and fed the actor `z_t` only — a memoryless agent that cannot solve this task;
`storm2` replaces it.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "jax[cuda12]" flax optax chex distrax omegaconf wandb pillow \
            tabulate imageio python-dotenv einops matplotlib tqdm \
            opencv-python-headless scipy gymnasium opensimplex orbax-checkpoint
```

The bridge_tunnel env is imported from the parent repo's `src/` (added to
`sys.path` automatically by `cl/environments/bridge_tunnel.py`).

## Train (winning recipe)

```bash
source .venv/bin/activate
python -m scripts.train \
  --env-config   configs/envs/bridge_tunnel_storm2_run6.yaml \
  --agent-config configs/agents/sweep/stormH_ent001_bl128.yaml \
  --offline --device 0
# SLURM: sbatch --job-name=stormH scripts/slurm_storm2_sweep.sh \
#   configs/agents/sweep/stormH_ent001_bl128.yaml configs/envs/bridge_tunnel_storm2_run6.yaml
```

Key recipe values (vs the paper defaults): `batch_length=128` and
`transformer.max_length=128` (the ~75-step evidence→door dependency must fit
the attention window), `env_context=128` (act-time rolling window), actor
entropy `0.01` (higher values entrench the constant-door local optimum),
`train_ratio=256`, γ=0.99. Door-binding is metastable (seed lottery,
mid-training dips), so checkpoints are archived every 50k steps and the best
is selected on held-out data.

## Evaluate

Always use the **TRUE door metric** with **sampled** actions (see the root
README's evaluation convention):

```bash
python -m scripts.true_eval_w --bundle ../final_models/storm \
  --episodes 2500 --sampled --env-context 128
# or on a training run: --results-dir results/<id> [--step N]
```

## Layout

```
cl/
  agents/storm2.py               the agent (train/eval loops, loss, acting)
  agents/world_models/storm/     transformer, distribution heads, state
  agents/world_models/dreamerv3/ encoder/decoder MLPs (reused by storm2)
  agents/commons/                replay buffer, LaProp, normalizers, MLP heads
  agents/policy/                 actor-critic on imagined rollouts
  environments/bridge_tunnel.py  adapter around the parent repo's env (+ fixed map pool)
  trainer/                       generic frame-budget trainer + checkpointing
configs/
  agents/sweep/stormH_ent001_bl128.yaml   ★ winning agent config
  envs/bridge_tunnel_storm2_run6.yaml     ★ winning env/run config (6M frames)
scripts/
  train.py                       entrypoint
  true_eval_w.py                 held-out TRUE-metric evaluator (bundle or run dir)
  eval_bridge_tunnel_forkwall.py legacy evaluator (return-proxy metric; avoid for reporting)
  slurm_storm2_sweep.sh          cluster launcher
```
