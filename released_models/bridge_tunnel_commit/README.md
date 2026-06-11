# bridge_tunnel_commit — released PPO agent

`ppo_gru_commit.pt` — PPO+GRU on `cogniland.bridge_tunnel_commit` (implicit
commitment: first successful build/mine locks the skill; `Discrete(6)`, one-hot
obs, view 21). 6M steps. `*.yaml` is the exact reproducible trainer config.

Held-out commit matrix (rows=category, cols=none/build/mine): when it commits it
picks the right tool — lakes→build (0.28 vs 0.06), rocky→mine (0.37 vs 0.02),
balanced split; 99–100% success (it detours the rest, ~0.6 "none").

Reproduce: `python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config released_models/bridge_tunnel_commit/ppo_gru_commit.yaml`
Evaluate:  `python scripts/bridge_tunnel/eval_bridge_tunnel_commit_ppo.py --checkpoint released_models/bridge_tunnel_commit/ppo_gru_commit.pt`
