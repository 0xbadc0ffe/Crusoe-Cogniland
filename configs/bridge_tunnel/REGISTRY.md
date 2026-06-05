# bridge_tunnel — experiment registry

The released agents and how to reproduce / evaluate them. One env
(`cogniland.bridge_tunnel`, `variant=bt|btc`); PPO+GRU and DreamerV3.

| experiment | variant | algo | obs | checkpoint | config / command |
|---|---|---|---|---|---|
| bt PPO (embed) | bt | PPO+GRU | tile-embed | `released_models/bridge_tunnel/natural_centergoal3.pt` | `configs/bridge_tunnel/bt_ppo_embed.yaml` |
| bt PPO (one-hot) | bt | PPO+GRU | one-hot | `released_models/bridge_tunnel/natural_centergoal3_onehot.pt` | `configs/bridge_tunnel/bt_ppo_onehot.yaml` |
| bt DreamerV3 (categorical) | bt | DreamerV3 25M | one-hot/categorical | `released_models/bridge_tunnel/dreamer_natural_categorical/` (git-LFS) | `dreamerv3_bridge_tunnel.py --variant bt --size 25M --decoder categorical` |
| btc PPO (one-hot) | btc | PPO+GRU | one-hot | `released_models/bridge_tunnel_commit/ppo_commit_onehot.pt` | `configs/bridge_tunnel/btc_ppo_onehot.yaml` |

## Reproduce
```bash
# PPO (config-driven)
python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config configs/bridge_tunnel/btc_ppo_onehot.yaml
# DreamerV3 (flags)
python scripts/bridge_tunnel/dreamerv3_bridge_tunnel.py --variant btc --size 25M --decoder categorical --total-env-steps 1_500_000
```

## Evaluate
```bash
python scripts/bridge_tunnel/eval_bridge_tunnel_agent.py       --checkpoint <bt ckpt>          # bt: traj grid + success
python scripts/bridge_tunnel/eval_bridge_tunnel_commit_ppo.py  --checkpoint <btc ckpt>         # btc: 3x3 commit matrix + grid
```

## W&B convention
project `bridge_tunnel`, `group = bt|btc`, tags `[algo, variant, obs/decoder]`.
