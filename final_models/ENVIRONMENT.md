# The shared task: `bridge_tunnel` **fork_wall** (BT-rules, no commitment)

All three agents in this folder — DreamerV3, STORM, and recurrent PPO+GRU — are
trained and evaluated on **exactly the same environment, reward, and map dataset**.
This file is the canonical specification and the proof that they match.

---

## 1. What the task is

`fork_wall` is a partially observed navigation problem (a POMDP) designed so that
success **requires memory**: the information the agent needs to make the final
decision is only visible early, then disappears from view.

```
   spawn                     memory corridor            fork wall
     ●───────── terrain ─────────(grass, mem_gap=16)───────║  ┌── top door
     │  (water / rock reveal                                ║  │
     │   the hidden CATEGORY)                               ║──┤
     │                                                      ║  │
     └──────────────────────────────────────────────────── ║  └── bottom door
```

* The map has a hidden **category**, inferable only from the terrain the agent
  passes through on the left:
  * **`lakes`** — water-dominated  → the **bottom** door is correct.
  * **`rocky`** — rock-dominated   → the **top** door is correct.
  * **`balanced`** — neither dominates → **either** door is correct.
* After the terrain, a **memory corridor** of `mem_gap = 16` grass columns carries
  no category information. The agent must *remember* the category across it.
* At the far right a **fork wall** blocks the way, with a **top** and a **bottom**
  door. Choosing the door that matches the remembered category is the whole task.

Because the deciding evidence (left terrain, seen around steps ~10–45) and the
consequence (door reward, ~step 85+) are separated by the corridor, an agent that
only reacts to its current view cannot solve it — it must build and carry a
**belief** about the category. This is what makes the task a clean probe of memory
and of learned world models.

## 2. Observation and action space (identical for all three)

* **Observation** — a flat vector of length **3974**:
  * an egocentric **`view_size = 21 × 21`** crop of the minimap, one-hot over the
    **9 tile types** → `21·21·9 = 3969` values, plus
  * **5 scalars**: `[compass_row, compass_col, active_obj/2, build_active, step/max]`.
  * Out-of-map cells are padded with the `OOB` tile id. No RGB image — the world is
    symbolic, so every network here uses an **MLP encoder**, not a CNN.
* **Action** — `Discrete(6)`: `0/1/2/3 = up/down/left/right`, `4 = build_raft`,
  `5 = build_harness`. (With `commit=False` the build actions exist but are never
  needed on fork_wall; the policies learn to ignore them.)

## 3. Reward function (identical for all three)

Per step $t$ the reward is

$$
r_t \;=\; \underbrace{-0.01}_{\text{slack}}
\;+\; \underbrace{0.015\,\big(d_{t-1}-d_{t}\big)}_{\text{progress shaping (PBRS)}}
\;+\; \underbrace{3.0\cdot\mathbb{1}[\text{correct door}]}_{\text{terminal bonus}}
$$

where $d_t$ is the **cost-to-go** to the goal in grid cells (unit step cost). Key
points, all set by the canonical parameters below:

* **Slack** $-0.01$/step gently pressures the agent to finish.
* **Progress shaping** uses `shaping_gamma = 1.0`, i.e. the *pure-progress*,
  potential-difference form $d_{t-1}-d_t$ (not the discounted PBRS
  $d_{t-1}-\gamma d_t$). Pure progress is **not farmable** — pacing back and forth
  nets zero — which removes the "loitering optimum" that discounted shaping creates.
* **Terminal bonus** $+3.0$ for stepping into a **correct** door. On `balanced`
  maps *either* door is correct, so both pay $+3.0$.
* **No wrong-door penalty** (`wrong_door_penalty = 0.0`) and **no balanced-neutral**
  rule (`balanced_neutral = false`): a wrong door simply pays no bonus. This is the
  minimal, dense reward that STORM was solved with; it is the reward all three now
  share. *(An earlier iteration of the PPO used `wrong_door_penalty = 3.0` and
  `balanced_neutral = true`; that agent has been retrained on this plain reward so
  the three match — see `ppo/README.md`.)*

## 4. Canonical parameters (the single source of truth)

Every field below is **identical** across the DreamerV3 wrapper
(`external/r2dreamer/envs/bridge_tunnel.py :: FORKWALL_KWARGS`), the STORM wrapper
(`STORM_model/cl/environments/bridge_tunnel.py :: FORKWALL_KWARGS`), and the PPO
config (`final_models/ppo/config.yaml`).

| Parameter | Value | Meaning |
|---|---|---|
| `variant` | `btc` | category-labelled maps |
| `commit` / `no_commit` | commitment **disabled** | BT movement rules, no raft/harness commit |
| `fork_wall` | `true` | top/bottom door decision task |
| `categories` | `balanced, lakes, rocky` | the three hidden classes |
| `size` (height) × `width` | `32 × 64` | map dimensions |
| `view_size` | `21` | egocentric observation crop |
| `max_steps` | `800` | episode timeout |
| `mem_gap` | `16` | grass memory-corridor width before the wall |
| `passage_half`, `wall_margin` | `1`, `1` | door/passage geometry |
| `orientation` | `natural` | spawn left, goal right |
| `tree_frac` | `0.03` | inviolable tree clutter |
| `goal_half` | `0` | single-cell doors |
| `slack_penalty` | `-0.01` | per-step slack |
| `shaping_coef` | `0.015` | progress-shaping weight |
| `shaping_gamma` | `1.0` | **pure-progress** shaping (non-farmable) |
| `reach_bonus` | `3.0` | correct-door terminal bonus |
| `wrong_door_penalty` | `0.0` | **no** penalty for the wrong door |
| `balanced_neutral` | `false` | on `balanced`, **either** door pays the bonus |
| `illegal_penalty` | `0.02` | penalty for an illegal build |
| `build_cost`, `commit_cost` | `0.0`, `0.05` | (unused on fork_wall) |
| `gamma` | `0.99` | task return discount |

## 5. Map dataset (identical for all three)

* **Training** maps: `data/bridge_tunnel/forkwall6k/train.pkl` — 4 800 maps
  (1 600 per category), generated once with `mem_gap = 16` baked into the terrain.
* **Held-out test** maps: `data/bridge_tunnel/forkwall6k/test.pkl` — 1 200 maps
  (400 per category), never seen in training.
* Generated by `scripts/bridge_tunnel/make_forkwall_dataset.py` (deterministic by
  seed). All final numbers in this folder are reported on the **test** split.

## 6. How "same env and reward" is guaranteed in code

* **DreamerV3** reads the maps via `BT_MAPS=…/forkwall6k/train.pkl` and constructs
  the env from `FORKWALL_KWARGS`, which was aligned **byte-for-byte** to STORM's.
* **STORM** reads `maps_path: …/forkwall6k/train.pkl` (its `env_config.yaml`) and the
  same `FORKWALL_KWARGS`.
* **PPO** reads `--maps-path …/forkwall6k/train.pkl` and the env parameters from
  `ppo/config.yaml`, whose reward fields equal the table above.

The three wrappers all instantiate the *same* numpy `cogniland.bridge_tunnel.env.
BridgeTunnelEnv`; only the agent on top differs. See `ARCHITECTURES.md` for the
agents.
