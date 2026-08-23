# DreamerV3 — 25M, batch_length 64

Model-based world-model agent (RSSM). See `../ARCHITECTURES.md §2`. Best of the
`{12M,25M}×{batch_length 64,128}` sweep on held-out test (decisive-door 97.0%).

* `dreamer_25M_bl64.pt` — orbax/torch checkpoint (`agent_state_dict`), 25M preset.
* `config.yaml`         — the exact as-trained Hydra config.

## Key hyperparameters (from `config.yaml`)

| | value |  | value |
|---|---|---|---|
| preset | `size25M` | RSSM deterministic | 3072 |
| RSSM stochastic | 32 × 24 discrete | RSSM blocks | 8 (block GRU) |
| MLP units | 384 | rep_loss | `dreamer` (reconstruction WM) |
| **batch_length** | **64** | imag_horizon | 15 |
| train_ratio | 64 | act_entropy | 0.01 |
| optimizer | LaProp, lr 4e-5, AGC 0.3 | steps | 3.0e6 |

## Reproduce (conda env `r2dreamer`)

```bash
export BT_MAPS=$PWD/data/bridge_tunnel/forkwall6k/train.pkl     # shared dataset
python r2dreamer_model/train.py \
  env=bridge_tunnel_forkwall env.task=bridgetunnel_forkwall \
  model=size25M model.rep_loss=dreamer \
  batch_length=64 model.imag_horizon=15 model.act_entropy=0.01 \
  env.train_ratio=64 env.steps=3e6 seed=0 device=cuda:0 \
  logdir=r2dreamer_model/runs/fw_sw_25M_bl64_h15
# SLURM: sbatch --export=ALL,SRC_DIR=$HOME,MODEL=size25M,BATCH_LENGTH=64,IMAG_HORIZON=15,\
#   ACT_ENTROPY=0.01,TRAIN_RATIO=64,STEPS=3e6,SEED=0,RUN_NAME=fw_sw_25M_bl64_h15 \
#   scripts/bridge_tunnel/slurm/train_forkwall_fixed_dreamer.sbatch
```

## Evaluate (held-out per-category)

```bash
python scripts/bridge_tunnel/eval_forkwall_fixed.py \
  --checkpoint final_models/dreamer/dreamer_25M_bl64.pt \
  --maps data/bridge_tunnel/forkwall6k/test.pkl --n 150 --model-size size25M
```
