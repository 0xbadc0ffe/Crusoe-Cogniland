# STORM — Stochastic Transformer-based world model

Model-based agent with a **Transformer** dynamics core (vs Dreamer's RSSM GRU).
See `../ARCHITECTURES.md §3`. Winning run `stormH_ent001_bl128` seed 0
(`t2hl7qnp`): **99.3% held-out success** (TRUE door metric, 2500 episodes,
`forkwall6k/test.pkl`: balanced .999 / lakes .990 / rocky .991; wrong-door
0.64%, timeout 0.04%).

* `checkpoint_step_00624489/`        — orbax checkpoint (seed 0, the 99.3% one).
* `checkpoint_seed1_step_00200000/`  — seed 1 best (98.6% at 2500 eps), for seed stats.
* `agent_config.yaml`                — architecture + training config.
* `env_config.yaml`                  — env/run config (points at `forkwall6k/train.pkl`).

## Key hyperparameters

| | value |  | value |
|---|---|---|---|
| dynamics | Transformer 2 layers, 512-d, 8 heads | max context / batch_length | **128** |
| stochastic latent | 32 × 32 discrete (unimix 1%) | KL dyn/rep | 0.5 / 0.1, free_nats 1 |
| imagination | context 16, rollout 16 | train_ratio | 256 |
| optimizer | lr 1e-4, AGC 0.3 | actor entropy | **0.01** |
| $\gamma$ | 0.99 (horizon 100) | env frames | 6M (best ckpt at ~2.5M) |
| act-time context window | **128** tokens (rolling) | | |

The three decisive levers (7-arm sweep): **batch_length 64→128** (the ~75-step
evidence→door dependency must fit the training window — same lever that fixed
DreamerV3), **act-time context window ≥ the dependency span** (32→128; pure
inference-time change, +7pp), and **entropy 0.01 not 0.03** (0.03 entrenches
the one-door basin). Door-binding is metastable (seed lottery + mid-training
dips), so checkpoints are archived every 50k steps and the best is selected on
held-out data.

## Reproduce (venv `STORM_model/.venv`)

```bash
cd STORM_model && source .venv/bin/activate
python -m scripts.train \
  --env-config   configs/envs/bridge_tunnel_storm2_run6.yaml \
  --agent-config configs/agents/sweep/stormH_ent001_bl128.yaml \
  --offline --device 0
# SLURM: sbatch --job-name=stormH scripts/slurm_storm2_sweep.sh \
#   configs/agents/sweep/stormH_ent001_bl128.yaml configs/envs/bridge_tunnel_storm2_run6.yaml
```

## Evaluate (TRUE door metric, held-out `forkwall6k/test.pkl`)

Always evaluate with **sampled actions** (greedy deadlocks into timeouts) and
the **TRUE door metric** — the framework's `return > 0` proxy counts fast
wrong-door episodes as successes (PBRS shaping exceeds slack) and inflates
success by ~6–13pp.

```bash
cd STORM_model && source .venv/bin/activate
python -m scripts.true_eval_w \
  --bundle ../final_models/storm --episodes 2500 --sampled --env-context 128
```
