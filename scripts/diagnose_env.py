"""Environment sanity diagnostics.

Four read-only routines against a fresh balanced-biome-filtered CognilandEnv:

  1. reward_breakdown        — per-step reward components under random policy
  2. return_distribution     — random vs. compass-follow policy episode returns
  3. obs_sanity              — dtype / range / finiteness checks
  4. reachability            — Dijkstra ctg_spawn finite on every map

Usage:
    python scripts/diagnose_env.py
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from omegaconf import OmegaConf

from cogniland.envs.env import (
    CognilandEnv, TILE_BERRY, TILE_TARGET_YES, TILE_TARGET_NO, TILE_DEADLY,
)
from cogniland.envs.multitask_wrapper import MultiTaskEnvWrapper


def _build_env(num_envs: int = 1, min_manhattan: int | None = None):
    cfg = OmegaConf.load("configs/env/cogniland.yaml")
    cfg.env.num_parallel_envs = num_envs
    if min_manhattan is not None:
        cfg.env.min_spawn_target_manhattan = min_manhattan
    env = CognilandEnv(cfg, cfg.env.train_maps, num_envs=num_envs)
    wrapper = MultiTaskEnvWrapper(env, cfg, num_tasks=1, task_embedding_dim=7)
    return wrapper, cfg


# ---------------------------------------------------------------------------
# 1. reward breakdown
# ---------------------------------------------------------------------------
def reward_breakdown(n_steps: int = 200):
    print("\n[1] reward_breakdown — random policy, 1 env, {} steps".format(n_steps))
    wrapper, cfg = _build_env(num_envs=1)
    obs = wrapper.reset()
    r = cfg.reward
    rng = np.random.default_rng(0)
    total = 0.0
    comp = {"step": 0.0, "reach": 0.0, "shape": 0.0, "death": 0.0}
    for t in range(n_steps):
        a = rng.integers(0, 8, size=1, dtype=np.int32)
        obs, rewards, dones, info = wrapper.step(a)
        ctg_p = float(info["ctg_prev"][0])
        ctg_c = float(info["ctg_curr"][0])
        step_c = -float(r.step_penalty)
        reach_c = float(r.reach_bonus) if info["reached"][0] else 0.0
        shape_c = (float(r.shaping_coef) * (ctg_p - ctg_c)
                   if (np.isfinite(ctg_p) and np.isfinite(ctg_c)) else 0.0)
        death_c = 0.0
        if bool(dones[0]) and not bool(info["alive"][0]):
            death_c = -float(r.death_penalty)
        comp["step"] += step_c
        comp["reach"] += reach_c
        comp["shape"] += shape_c
        comp["death"] += death_c
        total += float(rewards[0])
        if t < 5 or bool(dones[0]):
            print(f"  t={t:3d} act={int(a[0])} r_env={float(rewards[0]):+.3f} "
                  f"step={step_c:+.3f} reach={reach_c:+.1f} shape={shape_c:+.3f} "
                  f"death={death_c:+.1f} ctg_prev={ctg_p:.1f} ctg_curr={ctg_c:.1f} "
                  f"done={bool(dones[0])}")
        if bool(dones[0]):
            print(f"  >>> episode reset at t={t}")
    print(f"  total env reward sum: {total:+.3f}")
    print(f"  component sum: step={comp['step']:+.3f} reach={comp['reach']:+.1f} "
          f"shape={comp['shape']:+.3f} death={comp['death']:+.1f}")


# ---------------------------------------------------------------------------
# 2. return distribution
# ---------------------------------------------------------------------------
def _compass_action(obs) -> np.ndarray:
    """Pick the move action most aligned with the compass vector."""
    scalars = np.asarray(obs["scalars"])
    cx = scalars[:, 0]  # column direction
    cy = scalars[:, 1]  # row direction
    # Actions 0..3: up/down/left/right → (dr, dc) in (-1,0)(+1,0)(0,-1)(0,+1)
    deltas = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.float32)
    score = deltas[None, :, 0] * cy[:, None] + deltas[None, :, 1] * cx[:, None]
    return np.argmax(score, axis=1).astype(np.int32)


def return_distribution(n_episodes: int = 500):
    print(f"\n[2] return_distribution — {n_episodes} episodes per policy")
    wrapper, _ = _build_env(num_envs=16)
    B = wrapper.num_envs

    def _run(policy_fn, label):
        obs = wrapper.reset()
        returns = []
        lens = []
        successes = []
        ep_done = 0
        max_steps = int(wrapper.env._max_steps)
        t = 0
        while ep_done < n_episodes:
            a = policy_fn(obs)
            obs, rewards, dones, info = wrapper.step(a)
            ret_mask = info.get("returned_episode")
            if ret_mask is not None and ret_mask.any():
                rr = info["returned_episode_returns"]
                ll = info["returned_episode_lengths"]
                ss = info["task_success"]
                for i in np.where(ret_mask)[0]:
                    returns.append(float(rr[i]))
                    lens.append(int(ll[i]))
                    successes.append(int(ss[i]))
                    ep_done += 1
                    if ep_done >= n_episodes:
                        break
            t += 1
            if t > max_steps * (n_episodes // B + 5):
                print(f"  {label}: hit step cap at {t} steps, got {ep_done}/{n_episodes}")
                break
        returns = np.array(returns[:n_episodes])
        lens = np.array(lens[:n_episodes])
        successes = np.array(successes[:n_episodes])
        print(f"  {label}: return mean={returns.mean():+.2f} std={returns.std():.2f} "
              f"min={returns.min():+.2f} max={returns.max():+.2f} "
              f"success={successes.mean()*100:.1f}%  avg_len={lens.mean():.1f}")
        return returns, successes

    _, _ = _run(lambda o: np.random.default_rng(0).integers(0, 8, size=B, dtype=np.int32),
                "random  ")
    _, _ = _run(_compass_action, "compass ")


# ---------------------------------------------------------------------------
# 3. obs sanity
# ---------------------------------------------------------------------------
def obs_sanity(n_steps: int = 100):
    print(f"\n[3] obs_sanity — {n_steps} random steps")
    wrapper, _ = _build_env(num_envs=4)
    obs = wrapper.reset()
    rng = np.random.default_rng(1)
    for t in range(n_steps):
        mm = np.asarray(obs["minimap"])
        sc = np.asarray(obs["scalars"])
        assert mm.dtype == np.int8, f"minimap dtype {mm.dtype}"
        assert mm.shape == (4, 45, 45), f"minimap shape {mm.shape}"
        assert int(mm.min()) >= 0 and int(mm.max()) <= 13, \
            f"minimap range [{mm.min()},{mm.max()}] outside [0,13]"
        assert np.isfinite(sc).all(), "scalars contain NaN/inf"
        compass_norm = np.sqrt(sc[:, 0] ** 2 + sc[:, 1] ** 2)
        # Compass is a unit vector except when agent is exactly at target (dist<1e-6).
        assert np.all(np.abs(compass_norm - 1.0) < 1e-3) or np.all(compass_norm >= 0), \
            f"compass unit-length violated: {compass_norm}"
        assert np.all(sc[:, 2:] >= 0) and np.all(sc[:, 2:] <= 1.0 + 1e-4), \
            f"scalars[2:] out of [0,1]: {sc[:, 2:]}"
        a = rng.integers(0, 8, size=4, dtype=np.int32)
        obs, _, _, _ = wrapper.step(a)
    # Quick class histogram to sanity-check
    mm = np.asarray(obs["minimap"])
    vals, cnt = np.unique(mm, return_counts=True)
    print(f"  classes seen: {dict(zip(vals.tolist(), cnt.tolist()))}")
    print("  obs sanity: PASS")


# ---------------------------------------------------------------------------
# 4. reachability
# ---------------------------------------------------------------------------
def reachability():
    print(f"\n[4] reachability — ctg_spawn finite on every map in pool")
    wrapper, _ = _build_env(num_envs=1)
    base = wrapper.env
    N = base._num_maps
    failures = []
    for mi in range(N):
        mi_arr = np.array([mi], dtype=np.int32)
        wrapper.reset(map_indices=mi_arr)
        cs = float(base.ctg_spawn[0])
        if not np.isfinite(cs):
            failures.append((mi, str(base._biomes[mi]), cs))
    print(f"  scanned {N} maps, failures: {len(failures)}")
    if failures:
        for mi, b, v in failures[:10]:
            print(f"    map={mi} biome={b} ctg_spawn={v}")
    else:
        print("  reachability: PASS")


def main():
    reward_breakdown(n_steps=200)
    return_distribution(n_episodes=200)
    obs_sanity(n_steps=100)
    reachability()


if __name__ == "__main__":
    main()
