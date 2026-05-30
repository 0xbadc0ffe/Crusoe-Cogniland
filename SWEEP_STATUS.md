# Overnight PPO sweep — status (launched 2026-05-25 ~23:10)

## TL;DR
- **Sweep:** `crusoe/crafter_in_cogniland/e0mkqeye`
- **Dashboard:** https://wandb.ai/crusoe/crafter_in_cogniland/sweeps/e0mkqeye
- **10–11 parallel agents** running on the 4090, ~15.5 GB / 24 GB GPU, 96% util.
- Training on the **new natural maps** (PPO generates them on the fly).
- Objective: **maximise `success/rolling100`**; entropy + reward shaping swept
  so you can read the performance ↔ policy-diversity trade-off off the dashboard.

## Monitor / control
```bash
# live agent logs
tail -f sweep_logs/agent_1.log
# how many trainers are alive
pgrep -fc train_ppo_gru.py
# GPU
nvidia-smi
# add more agents (only if GPU util < ~80% and VRAM has room)
bash scripts/launch_ppo_sweep.sh crusoe/crafter_in_cogniland/e0mkqeye 4
# STOP everything
bash scripts/stop_ppo_sweep.sh
```
Each agent runs trials **sequentially and forever** until stopped, so the sweep
keeps exploring all night. Best checkpoints land in `checkpoints/sweep/<run>/`.

## What's being swept (`scripts/ppo_sweep.yaml`, bayes)
| param | range | why |
|---|---|---|
| `ent-coef` | 0.002–0.05 (log) | **policy stochasticity / diversity** |
| `slack-penalty` | −0.04…−0.01 | reward: per-step cost |
| `shaping-coef` | 0.005–0.02 (log) | reward: PBRS strength |
| `learning-rate` | 1e-4…5e-4 (log) | |
| `gamma` | 0.985–0.999 | |
| `gae-lambda` | 0.90–0.97 | |
| `num-steps` | {64,128,256} | |
| `update-epochs` | {2,4,8} | |
| `vf-coef` | 0.3–1.0 | |
| `belief-coef` | 0.1–1.0 | build-belief aux loss |
| `clip-coef` | 0.10–0.30 | |

Fixed: `env-size=64`, `num-envs=16`, `num-minibatches=4`, `map-type=random`
(trains across lake/rocky/balanced → **resilient**), `total-timesteps=2,000,000`
(~1.5–2 h/run under GPU sharing), `--anneal-lr`.

## Reading "diverse stochastic policy"
A single scalar can't capture "high-performing **and** diverse", so the sweep
optimises success and **logs `loss/entropy`** (policy entropy) alongside it.
On the dashboard, make a scatter of `success/rolling100` vs `loss/entropy` and
pick a point on the high-success / high-entropy frontier. For a rigorous
diversity read on a chosen checkpoint, run the variability tool I added:
```bash
python scripts/eval_trajectory_variability.py \
  --checkpoint checkpoints/sweep/<run>/final.pt --map-type random --n-maps 4
```
(reports state-occupancy entropy + number of macro-trajectory modes).

## Caveats / honest notes
- **Shared GPU:** another job (`neurorvq_ablation_suite.py`, user `leonardo`) is
  on this card. I capped at ~10 agents (≈15.5 GB) to leave it headroom rather
  than fill all 24 GB. Bump with `launch_ppo_sweep.sh` if it frees up.
- The env is **CPU-bound numpy**, so GPU memory — not compute — is what each run
  mostly consumes; that's why ~10 runs is the sweet spot, not 40.
- **Maps changed + reward physics changed** this session (`SLIP_WEIGHT_LAND`
  0.30→0.15; natural simplex+component generator, no Dijkstra validation), so
  these runs are **not comparable to older checkpoints** — fresh baseline.
- Dreamer (JAX) was **not** swept: it needs the pickled dataset regenerated with
  the new maps first. Say the word and I'll do a Dreamer sweep too.
