# `final_models/` — three agents that solve the same memory task

Three trained agents on the **`bridge_tunnel` fork_wall** POMDP — a navigation task
whose only difficulty is **remembering** a hidden map category across an
information-free corridor and picking the matching door. All three are trained and
evaluated on the **identical environment, reward, and 6k-map dataset**.

* **`ENVIRONMENT.md`** — the shared task, observation/action space, reward math, and
  the proof that all three use the same env + reward. **Read this first.**
* **`ARCHITECTURES.md`** — a textbook-style explanation of all three architectures
  (PPO+GRU, DreamerV3, STORM), with the mathematics and a comparison.
* **`dreamer/ storm/ ppo/`** — each holds the trained checkpoint, its as-trained
  config, and a `README.md` with the exact command to reproduce it.

## Results (held-out test split, `forkwall6k/test.pkl`)

The decisive metric is **decisive-door success** on `lakes`+`rocky` maps, where a
memoryless constant-door policy scores 50%. `balanced` maps accept either door.

| Agent | file | overall success | decisive-door (chance 50%) |
|---|---|---:|---:|
| **DreamerV3** (25M, bl=64) | `dreamer/dreamer_25M_bl64.pt` | 98.0% | **97.0%** |
| **STORM** (transformer 2L·512, bl=128) | `storm/checkpoint_step_00624489/` | **99.3%** † | 99.1% |
| **PPO+GRU** (belief head) | `ppo/ppo_plain.pt` | 98.2% | **97.7%** |

† STORM: 2500 held-out episodes, TRUE door metric (final cell ∈ correct-door
set — NOT the `return>0` proxy, which counts fast wrong-door episodes as
successes and inflates by ~6–13pp), sampled actions, act-time context 128.
Per-category: balanced .999 / lakes .990 / rocky .991. Dreamer is evaluated
deterministically, PPO and STORM stochastically (their native operating mode).

**All three solve the identical plain reward.** DreamerV3 and STORM needed enough
*memory budget* (§4.2 of `ARCHITECTURES.md`); model-free PPO additionally needed the
right *exploration schedule* — a plain-reward PPO with default entropy collapses to a
constant-door shortcut (the belief is encoded but unused), and is rescued by **high
starting entropy (0.15) + annealing** (`ppo/config.yaml`). See `ppo/README.md`.

## Reproduce everything

```bash
# 0. one-time: build the shared 6k-map dataset (deterministic)
python scripts/bridge_tunnel/make_forkwall_dataset.py         # -> data/bridge_tunnel/forkwall6k/{train,test}.pkl

# 1. DreamerV3   (conda env: r2dreamer)      see dreamer/README.md
# 2. STORM       (venv: STORM_model/.venv)   see storm/README.md
# 3. PPO+GRU     (conda env: crusoe)         see ppo/README.md
```

Each sub-README has the single command that produced its checkpoint, plus the
held-out evaluation command.

## Provenance

* Dreamer checkpoint: run `fw_sw_25M_bl64_h15`, best of the `{12M,25M}×{bl64,bl128}`
  sweep (see `ARCHITECTURES.md §4.2`).
* STORM checkpoint: run `t2hl7qnp` (`stormH_ent001_bl128`, seed 0), step 624 489 —
  winner of a 7-arm sweep over {entropy .01/.03/.045} × {imag context} ×
  {train_ratio} × {batch_length 64/128}; the decisive levers were
  `batch_length=128`, act-time context window 128, and entropy 0.01
  (see `storm/README.md`). Seed-1 best (`checkpoint_seed1_step_00200000/`,
  98.6%) included for seed statistics.
* PPO checkpoint: trained on the **plain** reward with the escape recipe
  (`ppo/config.yaml` = `btc_ppo_forkwall_plain_solved.yaml`: `ent_coef 0.15` +
  `anneal_ent`), best of a 4-config × 3-seed sweep (`ent15_anneal` escaped the
  constant-door trap on 2/3 seeds; winner = seed 1). The earlier
  `btc_ppo_forkwall_nocommit.yaml` (wrong-door penalty) is *not* used here.
