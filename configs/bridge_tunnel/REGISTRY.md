# bridge_tunnel — experiment registry

The released agents and how to reproduce / evaluate them. One env
(`cogniland.bridge_tunnel`, `variant=bt|btc`); PPO+GRU and DreamerV3.

| experiment | variant | algo | obs | checkpoint | config / command |
|---|---|---|---|---|---|
| bt PPO (embed) | bt | PPO+GRU | tile-embed | `released_models/bridge_tunnel/ppo_gru_embed.pt` | `configs/bridge_tunnel/bt_ppo_embed.yaml` |
| bt PPO (one-hot) | bt | PPO+GRU | one-hot | `released_models/bridge_tunnel/ppo_gru.pt` | `configs/bridge_tunnel/bt_ppo_onehot.yaml` |
| bt DreamerV3 (categorical) | bt | DreamerV3 25M | one-hot/categorical | `released_models/bridge_tunnel/dreamerv3/` (git-LFS) | `dreamerv3_bridge_tunnel.py --variant bt --size 25M --decoder categorical` |
| btc PPO (one-hot) | btc | PPO+GRU | one-hot | `released_models/bridge_tunnel_commit/ppo_gru_commit.pt` | `configs/bridge_tunnel/btc_ppo_onehot.yaml` |
| btc PPO + aux belief | btc | PPO+GRU | one-hot | `released_models/bridge_tunnel_commit/ppo_gru_commit_aux_belief.pt` | `... ppo_gru_commit_aux_belief.yaml --belief-coef 0.3` |
| btc PPO fork_wall + belief | btc | PPO+GRU | one-hot | `released_models/bridge_tunnel_commit/ppo_gru_forkwall_belief.pt` | `configs/bridge_tunnel/btc_ppo_forkwall.yaml` |
| btc PPO fork_wall (no-commit) | btc/bt-rules | PPO+GRU | one-hot | `released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt` | `configs/bridge_tunnel/btc_ppo_forkwall_nocommit.yaml` (seed 2) |
| btc DreamerV3 (categorical) | btc | DreamerV3 25M | categorical | `released_models/bridge_tunnel_commit/dreamerv3_commit/` (git-LFS) | `dreamerv3_bridge_tunnel.py --variant btc --size 25M --decoder categorical --total-env-steps 6_000_000 --set entropy_coef=0.01` |

The btc DreamerV3 was trained with raised exploration (`entropy_coef=0.01`, vs the
3e-4 paper default) so it learns the map→skill commitment like PPO. On held-out
eval maps its 3×3 commit matrix is map-type-biased — none/build/mine ≈
balanced 0.32/0.33/0.35, lakes 0.39/**0.44**/0.18, rocky 0.35/0.12/**0.53**;
success 84–96%. The env reward is identical to the btc PPO agent's.

`ppo_gru_commit_aux_belief` is the btc PPO agent with one extra head: a 3-class
auxiliary classifier of the map category trained on `gru_h` (`belief_coef=0.3`),
the only config difference from `ppo_gru_commit`. Task performance is unchanged
(99.1% reach) but map-type belief becomes far more linearly decodable from
`gru_h` (map-grouped probe 0.40→0.70). Released as a mech-interp substrate.

`ppo_gru_forkwall_belief` is the same PPO+GRU+belief agent trained on the
**fork_wall** task (`btc_ppo_forkwall.yaml`, `fork_wall: true`): the corridor
ends in a 3-cell passage through a wall, then a top/bottom door pair where only
the door matching the map category (rocky→top, lakes→bottom, balanced→either)
counts as success. This makes the category belief behaviourally load-bearing —
it must survive the passage and select the final door — as a belief-steering
substrate. Held-out: 96.8% correct-door / 0.5% wrong-door / 2.7% timeout; door
matrix rocky→top 0.99, lakes→bottom 0.99, balanced ≈0.48/0.45. Eval with
`scripts/bridge_tunnel/eval_bridge_tunnel_forkwall.py`; held-out maps in
`data/bridge_tunnel/val_maps_btc_forkwall.pkl`.

`ppo_gru_forkwall_nocommit` is the BT-rules counterpart: same fork_wall task and
belief head but commitment disabled (`no_commit: true` → bt mechanics, 5-scalar
obs). It shows the belief→door binding does not require commitment — held-out
100% success, map→door rocky→top 1.00 / lakes→bottom 1.00 / balanced→bottom 0.79,
map→belief diagonal 0.98/0.95/0.93. Training has a fixed-door basin: 2/3 reseeds
condition, the released one is seed=2. Correct-door PBRS shaping is byte-identical
to the committed variant.

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
