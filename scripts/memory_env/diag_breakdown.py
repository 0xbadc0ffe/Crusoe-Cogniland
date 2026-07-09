#!/usr/bin/env python
"""Per-cue SUCCESS vs BRANCH-CORRECT vs DOOR breakdown for trained MemoryEnv models.

Disambiguates the failure mode behind the 1.23/0.73 reward quantization:
  - success         = stepped on the colour-correct door (colour memory)
  - branch_correct  = took the shape-correct branch     (shape memory)
If colour memory works but the branch collapses to one direction, success will be
high everywhere while branch_correct is ~1.0 only on the agent's preferred branch
direction — i.e. "colour learned, shape ignored" (a structural local optimum, not
under-training).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "external" / "r2dreamer"))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env import MemoryEnv, MemoryEnvConfig  # noqa: E402
from datasets import ALL_CUES, TRAIN_CUES, TEST_SEED0  # noqa: E402
from eval_r2dreamer import build_act_fn  # noqa: E402

MODELS = ["2cue", "3cue", "4cue"]


def run(ckpt, model, device, n):
    act = build_act_fn(ckpt, model, device=device)
    out = {}
    for ci, cue in enumerate(ALL_CUES):
        cfg = MemoryEnvConfig(cue_distribution="custom", custom_cues=[cue])
        succ = branch = 0
        tbs, sds = [], []
        for k in range(n):
            env = MemoryEnv(cfg)
            obs, info = env.reset(seed=TEST_SEED0 + ci * 100000 + k)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(act(obs, info))
                done = term or trunc
            succ += int(info["success"])
            branch += int(bool(info["branch_correct"]))
            tbs.append(info["taken_branch"])
            sds.append(info["selected_door_color"])
        out[cue] = dict(
            success=succ / n, branch_correct=branch / n,
            in_train=cue in TRAIN_CUES[model],
            modal_branch=Counter(tbs).most_common(1)[0],
            modal_door=Counter(sds).most_common(1)[0],
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    for m in MODELS:
        ap.add_argument(f"--ckpt-{m}")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=96)
    a = ap.parse_args()
    ck = {m: getattr(a, f"ckpt_{m}") for m in MODELS}
    rep = {}
    for m in MODELS:
        if not ck[m]:
            continue
        print(f"== {m}: {ck[m]}", flush=True)
        rep[m] = run(ck[m], m, a.device, a.n)
        for cue in ALL_CUES:
            v = rep[m][cue]
            tag = "train  " if v["in_train"] else "heldout"
            print(f"  {cue:11s} [{tag}] success={v['success']:.2f} "
                  f"branch_correct={v['branch_correct']:.2f} "
                  f"branch={v['modal_branch']} door={v['modal_door']}", flush=True)
    out = _REPO / "outputs" / "report" / "memoryenv_breakdown.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
