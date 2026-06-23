"""Train cue-sets + held-out test set + per-cue reward eval for MemoryEnv.

Three training regimes (cue subsets); all models are evaluated on the SAME
held-out test set covering all four cues, so off-distribution generalisation /
entanglement shows up as low reward on the unseen cues.
"""
from __future__ import annotations
import numpy as np
from cogniland.memory_env import MemoryEnv, MemoryEnvConfig, oracle_action

# training cue subsets (entangled -> partially -> fully factorized)
TRAIN_CUES = {
    "2cue": ["green_up", "blue_down"],                              # entangled
    "3cue": ["green_up", "green_down", "blue_down"],               # partial
    "4cue": ["green_up", "blue_up", "green_down", "blue_down"],    # factorized
}
ALL_CUES = TRAIN_CUES["4cue"]

# disjoint seed ranges: training draws from [0, TEST_SEED0); test is held out.
TEST_SEED0 = 1_000_000
TEST_N_PER_CUE = 128


def eval_per_cue(act_fn, *, cfg: MemoryEnvConfig | None = None,
                 n_per_cue: int = TEST_N_PER_CUE, seed0: int = TEST_SEED0):
    """Run `act_fn(obs, info)->action` over the held-out test set; return mean
    reward (and success) per cue type, doors/positions randomised per episode."""
    base = cfg or MemoryEnvConfig()
    out = {}
    for ci, cue in enumerate(ALL_CUES):
        c = MemoryEnvConfig(**{**base.__dict__, "cue_distribution": "custom",
                               "custom_cues": [cue]})
        rews, succ = [], []
        for k in range(n_per_cue):
            env = MemoryEnv(c)
            obs, info = env.reset(seed=seed0 + ci * 100000 + k)
            ep_r, done = 0.0, False
            while not done:
                a = act_fn(obs, info)
                obs, r, term, trunc, info = env.step(a)
                ep_r += r; done = term or trunc
            rews.append(ep_r); succ.append(float(info["success"]))
        out[cue] = {"avg_reward": float(np.mean(rews)),
                    "success": float(np.mean(succ)), "n": n_per_cue}
    out["overall"] = {"avg_reward": float(np.mean([v["avg_reward"] for k, v in out.items() if k in ALL_CUES])),
                      "success": float(np.mean([v["success"] for k, v in out.items() if k in ALL_CUES]))}
    return out


if __name__ == "__main__":
    # sanity: the scripted oracle should solve every cue (reward ~ +1.0)
    rep = eval_per_cue(lambda obs, info: oracle_action(info), n_per_cue=32)
    for cue in ALL_CUES + ["overall"]:
        v = rep[cue]; print(f"  {cue:11s} avg_reward={v['avg_reward']:+.3f} success={v['success']:.2f}")
