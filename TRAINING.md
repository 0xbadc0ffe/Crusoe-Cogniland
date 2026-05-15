# Cogniland Nav — Training Guide

Two trainers ship in this repo, both writing to W&B:

| Trainer | Obs space | Algorithm | Where |
|---|---|---|---|
| **PPO-RNN** | Symbolic tile-id grid (21×21 int8) + skill flag | PPO + GRU, hybrid action (Categorical move + tanh-Gaussian build scalar) | `scripts/train_ppo_gru.py` |
| **DreamerV3** | RGB image (3 × 168 × 168) + skill flag | World-model + imagined-rollout AC (RSSM with discrete latents, KL balancing, symlog reward) | `scripts/train_dreamer.py` |

DreamerV3 trains **in pixel space** (RGB sprites). PPO-RNN trains on the symbolic tile-id grid (faster, no CNN over pixels, embeds 9 tile classes via `nn.Embedding`).

---

## 0. One-time setup on the RTX 4090 box

```bash
git clone <this-repo> && cd Crusoe-Cogniland
conda env create -f environment.yml         # or:  python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install wandb opensimplex imageio imageio-ffmpeg
wandb login                                  # paste your API key
python scripts/generate_cogniland_dataset.py --help    # optional sanity check
python -m pytest tests/                      # ~5s
```

Confirm CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

---

## 1. PPO-RNN

Symbolic observation (the default) — embeds each tile id, runs a small 2D CNN over the embedded grid, then GRU → discrete-move head + tanh-Gaussian scalar head. Hybrid action: the categorical move and the continuous build_scalar are both sampled every step; the env consumes the scalar only on `build` actions.

### Recommended 4090 command

```bash
python scripts/train_ppo_gru.py \
    --total-timesteps 5_000_000 \
    --num-envs 32 --num-steps 128 \
    --env-size 64 --view-size 21 --map-type random \
    --max-steps 1000 \
    --gru-hidden 256 --embed-dim 512 \
    --learning-rate 3e-4 --ent-coef 0.01 --clip-coef 0.2 \
    --num-minibatches 4 --update-epochs 4 \
    --device cuda \
    --wandb-project cogniland-nav --run-name ppo_gru_main
```

### Quick smoke test (verify everything wires up)

```bash
python scripts/train_ppo_gru.py \
    --total-timesteps 200_000 --num-envs 8 \
    --num-steps 64 --num-minibatches 4 \
    --view-size 21 --env-size 32 \
    --wandb-mode disabled --device cuda
```

### Key knobs

| Flag | Default | Notes |
|---|---|---|
| `--env-size` | 64 | one of {32, 64, 96, 128} |
| `--map-type` | random | `lake` / `rocky` / `balanced` / `random` |
| `--view-size` | 21 | agent's partial-obs window side (odd, ≥3) |
| `--obs-mode` | symbolic | `symbolic` (default) / `rgb` / `both` |
| `--num-envs` | 16 | parallel envs |
| `--num-steps` | 128 | rollout length per env |
| `--num-minibatches` | 4 | must divide `--num-envs` |
| `--update-epochs` | 4 | PPO epochs per rollout |
| `--anneal-lr` | off | linear LR decay to 0 |
| `--target-kl` | none | enable early-stop e.g. `0.02` |

### What W&B will show

- **`charts/episode_return_mean`** — main learning curve
- **`charts/reach_rate`** — fraction of episodes reaching target
- **`charts/built_correct_frac` / `built_wrong_frac` / `built_none_frac`** — the belief/commitment signal (does the agent pick the right item per map family?)
- **`train/policy_loss`, `train/value_loss`, `train/entropy`, `train/approx_kl`, `train/clipfrac`** — PPO health
- **`train/scalar_std`** — the learned σ on the tanh-Gaussian scalar head (should shrink as the policy commits)
- **`train/sps`** — env steps/sec (RTX 4090 target: ~3-5k sps at default config)

---

## 2. DreamerV3 (pixel space)

Model components:

- **Encoder**: 3 stride-2 convs + 1×1 bottleneck → linear → 1024-dim embed (image must be `mod 8`).
- **RSSM**: deterministic GRU state (`--deter 512`) + stochastic categorical latents (`--stoch-classes 32 × --stoch-dim 32`), straight-through gradients.
- **Decoder**: linear → 3 stride-2 deconvs → RGB logits, MSE reconstruction loss on sigmoid.
- **Heads**: reward (symlog MSE), continue (BCE on `1 − done`).
- **Actor + Critic**: Categorical move + tanh-Gaussian scalar; AC trained on imagined latent rollouts of `--imagine-horizon` steps; λ-returns with a slow-EMA target critic.
- **KL**: balanced (`α = 0.8`) with free bits (`--kl-free 1.0`).

Every `--imagine-every` updates the trainer writes a video of imagined trajectories to `--imagine-dir` AND uploads it to W&B as `imagine/video`. Top row = real prefix frames; bottom row = pure model dreams.

### Recommended 4090 command

```bash
python scripts/train_dreamer.py \
    --total-env-steps 1_000_000 \
    --num-envs 4 --train-ratio 32 \
    --batch-size 16 --batch-length 64 \
    --env-size 64 --view-size 21 --tile-px 8 \
    --max-steps 1000 --map-type random \
    --imagine-every 2000 --imagine-batch 4 \
    --device cuda \
    --wandb-project cogniland-nav-dreamer --run-name dreamer_main
```

At default config the model is ~30M params; one update step on a 4090 takes ~80–100 ms, so `train-ratio 32` runs at roughly 1 env step every ~3 s of wall-time and converges in ~12-24 hours for 1 M env steps. Drop `--train-ratio` to 16 if you want faster wall-clock at the cost of sample efficiency.

### Quick smoke test

```bash
python scripts/train_dreamer.py \
    --total-env-steps 30_000 --num-envs 2 \
    --batch-size 8 --batch-length 32 \
    --train-ratio 4 --prefill 1000 \
    --env-size 32 --view-size 11 --tile-px 8 \
    --imagine-every 200 --save-every-updates 500 \
    --wandb-mode disabled --device cuda
```

### Important Dreamer knobs

| Flag | Default | Notes |
|---|---|---|
| `--train-ratio` | 32 | model updates per env step (the DV3 default) |
| `--batch-size` × `--batch-length` | 16 × 64 | each replay sample has this shape |
| `--imagine-horizon` | 15 | imagined-rollout length for AC training |
| `--world-lr` / `--actor-lr` / `--critic-lr` | 1e-4 / 3e-5 / 3e-5 | three independent optimizers |
| `--kl-alpha` / `--kl-free` | 0.8 / 1.0 | KL balancing |
| `--deter` / `--stoch-classes` × `--stoch-dim` | 512 / 32 × 32 | latent capacity |
| `--imagine-every` | 2000 | how often to log an imagined video |
| `--imagine-batch` | 4 | number of parallel rollouts in each video |
| `--save-every-updates` | 5000 | checkpoint + W&B upload cadence |

### W&B panels to watch

- **`wm/image_loss`** — should drop below 0.005 in the first 100k env steps once the encoder/decoder catch up
- **`wm/reward_loss`**, **`wm/cont_loss`**, **`wm/kl_loss`** — reward + done heads should track quickly; KL should sit at the free-bits floor most of training
- **`actor/loss`**, **`actor/entropy`**, **`critic/loss`** — AC sanity
- **`imag/return_mean`** — running mean of imagined λ-returns; correlates with true `charts/episode_return_mean`
- **`charts/episode_return_mean`** + **`charts/reach_rate`** — actual env performance
- **`imagine/video`** — auto-logged every `--imagine-every`; look for the bottom row matching the top row after ~50k env steps

---

## 3. Inspecting an imagination after training

```bash
python scripts/imagine_video.py \
    --checkpoint checkpoints/dreamer_main_upd10000.pt \
    --out imagine/manual_inspect.mp4 \
    --episodes 4 --prefix-steps 12 --horizon 64 \
    --map-type lake
```

This is the offline counterpart of the periodic video log — handy when you want to inspect a specific checkpoint with a specific `--map-type` (e.g. validate that the model dreams water correctly on lake maps).

---

## 4. Troubleshooting

- **W&B not logging videos**: install `imageio-ffmpeg`. `imageio.get_writer` falls back to PIL otherwise, which won't write mp4.
- **CUDA out of memory** in Dreamer: drop `--batch-size` or `--batch-length` first, then `--deter` / `--stoch-classes`. The encoder + decoder dominate VRAM for 168×168 images.
- **Encoder asserts "image dims must be divisible by 8"**: with default `--view-size 21 --tile-px 8` the image is 168×168 (divisible by 8). If you pick non-mod-8 combinations, round one of them.
- **PPO entropy collapses early** (`train/scalar_std → 0`, all builds in one direction): try `--ent-coef 0.03 --target-kl 0.02`.
- **Dreamer KL blowing up** (no reconstruction learned, KL grows without bound): drop `--world-lr` to 5e-5 and increase `--kl-free` to 2.0.

---

## 5. Where things live

```
scripts/
  train_ppo_gru.py             PPO-RNN trainer (symbolic obs by default)
  train_dreamer.py             DreamerV3 trainer (pixel obs)
  imagine_video.py             offline imagination viz from a checkpoint
src/cogniland/
  nav/                         env + mapgen + renderer (see README)
  nav_dreamer_video.py         the render_imagined helper (called by both
                               train_dreamer.py and scripts/imagine_video.py)
checkpoints/                   auto-created; auto-uploaded to W&B
imagine/                       auto-created; imagined videos
```
