# bridge_tunnel_commit — released PPO agent

`ppo_gru_commit.pt` — PPO+GRU on `cogniland.bridge_tunnel_commit` (implicit
commitment: first successful build/mine locks the skill; `Discrete(6)`, one-hot
obs, view 21). 6M steps. `*.yaml` is the exact reproducible trainer config.

Held-out commit matrix (rows=category, cols=none/build/mine): when it commits it
picks the right tool — lakes→build (0.28 vs 0.06), rocky→mine (0.37 vs 0.02),
balanced split; 99–100% success (it detours the rest, ~0.6 "none").

Reproduce: `python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config released_models/bridge_tunnel_commit/ppo_gru_commit.yaml`
Evaluate:  `python scripts/bridge_tunnel/eval_bridge_tunnel_commit_ppo.py --checkpoint released_models/bridge_tunnel_commit/ppo_gru_commit.pt`

## `ppo_gru_commit_aux_belief.pt` — same agent + auxiliary belief head

Identical training to `ppo_gru_commit` **plus** a 3-class auxiliary head that
classifies the map category (balanced/lakes/rocky) from the GRU state, with the
CE gradient shaping the trunk (`belief_coef: 0.3`; the only config difference —
see `ppo_gru_commit_aux_belief.yaml`). Task performance is unchanged (99.1%
reach), but the aux loss makes the latent **belief** far more linearly decodable
from `gru_h` (map-grouped probe balanced accuracy 0.40 → 0.70) and the
belief-driven crossing skills decodable several steps earlier — without any
behavioural cost. Built as a mech-interp substrate for belief/skill probing.

Reproduce: `python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config released_models/bridge_tunnel_commit/ppo_gru_commit_aux_belief.yaml --belief-coef 0.3`
Evaluate:  `python scripts/bridge_tunnel/eval_bridge_tunnel_commit_ppo.py --checkpoint released_models/bridge_tunnel_commit/ppo_gru_commit_aux_belief.pt`

## `ppo_gru_forkwall_belief.pt` — fork_wall split-decision task + aux belief head

Same PPO+GRU + aux belief head (`belief_coef: 0.3`), trained on the **fork_wall**
variant of `btc`: after the open category-revealing corridor the agent passes
through a 3-cell gap in a wall (1 cell from the right edge), then must pick one
of two single-cell doors — **top if the map is rocky, bottom if lakes, either if
balanced**. Only the door matching the map category pays the reach bonus / counts
as success; the decoy door still ends the episode with no reward. This makes the
map-category belief a *behaviourally load-bearing* variable that must be carried
through the passage to the final action — the intended substrate for belief
steering (does patching the belief flip the door choice?).

Held-out eval (16 maps/category × 32 stochastic rollouts, seeds ≥10000):
**96.8% correct-door success, 0.5% wrong-door, 2.7% timeout**; aux belief
accuracy ≈0.87. Door-choice matrix (rows=category, cols=top/neither/bottom):
rocky→top 0.99, lakes→bottom 0.99, balanced ≈0.48/0.45 split. See
`forkwall_figures/` for the door-choice matrix, training curves, and trajectory
grid.

Reproduce: `python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config released_models/bridge_tunnel_commit/ppo_gru_forkwall_belief.yaml`
Evaluate:  `python scripts/bridge_tunnel/eval_bridge_tunnel_forkwall.py --checkpoint released_models/bridge_tunnel_commit/ppo_gru_forkwall_belief.pt`

## `ppo_gru_forkwall_nocommit.pt` — fork_wall under BT rules (no commitment)

Same fork_wall task and aux belief head, but the **commitment mechanic is
disabled** (`no_commit: true`): build and mine are always available, no
lock / commit-cost, and the obs drops the two commit flags (5 scalars, bt-style).
The maps are still the labelled category maps (needed to define the correct
door); only the mechanics are bt. This is the BT-rules counterpart to
`ppo_gru_forkwall_belief`, used to show the belief→door binding does **not**
depend on commitment.

Held-out eval (16 maps/category × 24 stochastic rollouts, seeds ≥10000):
**100% success**, clean conditioning — map→door rocky→top 1.00, lakes→bottom
1.00, balanced→bottom 0.79 (either valid); map→belief diagonal 0.98/0.95/0.93.
See `forkwall_figures/nocommit_*`.

Note on training variance: the fixed-door basin is escapable but not on every
seed — **2 of 3 reseeds condition** (the third collapses to always-top), see
`forkwall_figures/nocommit_reseed_door_matrices.png`. This released checkpoint is
the seed=2 run (perfect conditioning). Correct-door PBRS shaping is identical to
the committed variant; commitment is not required, just a favourable basin.

Reproduce: `python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py --config released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.yaml`
Evaluate:  `python scripts/bridge_tunnel/eval_bridge_tunnel_forkwall.py --checkpoint released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt`
