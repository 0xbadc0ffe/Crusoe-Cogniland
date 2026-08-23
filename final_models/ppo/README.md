# PPO + GRU (recurrent, belief head) — plain reward

Model-free recurrent PPO. See `../ARCHITECTURES.md §1`. Trained on the **plain**
reward (identical to DreamerV3 and STORM — no wrong-door penalty, no
balanced-neutral).

* `ppo_plain.pt` — checkpoint (policy+value+GRU+belief head). Best of a
  4-config × 3-seed exploration sweep (run `ppo_pl_ent15_anneal`, seed 1).
* `config.yaml`  — the exact as-trained config (`btc_ppo_forkwall_plain_solved.yaml`).

## Held-out test (`forkwall6k/test.pkl`, stochastic policy)

| category | success | door chosen |
|---|---:|---|
| balanced | 99.3% | 62% top / 37% bottom |
| lakes | 98.7% | → bottom |
| rocky | 96.7% | → top |
| **overall / decisive** | **98.2% / 97.7%** | (chance on decisive = 50%) |

## The escape recipe (why the plain reward needs it)

With **default** entropy the plain-reward PPO collapses to a **constant-door**
policy (decisive ≈ 50%): the GRU *encodes* the category (`belief_acc ≈ 0.88`) but
the actor never learns to use it, because "always go top" already earns 2/3 of the
reward and on-policy PPO stops exploring the alternative. This is an exploration /
credit-assignment trap, not a representation failure. A 4-config × 3-seed sweep
found the fix:

| config | escaped trap |
|---|---|
| **ent 0.15 + anneal→0** | **2/3 seeds** ✅ (shipped) |
| ent 0.12 constant | 1/3 |
| ent 0.15 + anneal + belief_coef 1.0 | 0/3 (aux loss overwhelmed RL) |
| ent 0.045 + anneal | 0/3 (entropy too low) |

So: **high starting entropy (0.15) is necessary; annealing it to 0 lets the policy
explore both doors early, then commit.** It is seed-sensitive (2/3), so reproduce
with a few seeds.

## Key hyperparameters (from `config.yaml`)

| | value |  | value |
|---|---|---|---|
| recurrence | GRU, hidden 128 | obs encoder | MLP, embed 256, one-hot |
| **ent_coef** | **0.15** | **anneal_ent** | **true** (→0) |
| clip $\epsilon$ | 0.2 | GAE $\lambda$ / $\gamma$ | 0.95 / 0.99 |
| belief_coef (aux) | 0.3 | lr | 3e-4 |
| num_envs × num_steps | 32 × 128 | epochs / minibatches | 4 / 4 |

## Reproduce (conda env `crusoe`)

```bash
python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py \
  --config configs/bridge_tunnel/btc_ppo_forkwall_plain_solved.yaml \
  --maps-path data/bridge_tunnel/forkwall6k/train.pkl \
  --total-timesteps 4000000 --seed 1
# (escape is seed-sensitive; sweep seeds 0-2 and keep the one that reaches ~1.0 train succ)
# equivalently: --config btc_ppo_forkwall_plain.yaml --ent-coef 0.15 --anneal-ent
```

## Evaluate (held-out per-category)

```bash
python scripts/bridge_tunnel/eval_forkwall_ppo.py \
  --checkpoint final_models/ppo/ppo_plain.pt \
  --maps data/bridge_tunnel/forkwall6k/test.pkl --n 150
```
