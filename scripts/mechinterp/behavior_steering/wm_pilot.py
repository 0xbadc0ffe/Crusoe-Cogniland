#!/usr/bin/env python3
"""Map-77 dose-response pilots for the world models, persisted as JSON.

Two families on the same six figure-seeds (2000..2005):
  logit    soft actor-logit bias on the tool action (MINE on map 77)
  tooladd  sustained additive displacement along the context-matched
           v_mine axis, dose = lam * proj_sd

Baseline is the plain replay path (no steering machinery engaged).

  # dreamer (conda r2dreamer)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:r2dreamer_model:scripts/mechinterp:scripts/figures \
    python scripts/mechinterp/behavior_steering/wm_pilot.py --agent dreamer
  # storm (STORM_model/.venv, run from STORM_model/)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from replay_episode import replay  # noqa: E402

OUT = REPO / "outputs/behavior_steering"
A_MINE, MAP = 5, 77
LOGIT_BIASES = (-1.5, -3.0, -4.5, -6.0, +1.0, +2.0)
TOOL_LAMS = (-1.0, -2.0, -4.0, +2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["dreamer", "storm"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    z = np.load(OUT / a.agent / "axes.npz")
    meta = json.loads((OUT / a.agent / "axes_meta.json").read_text())
    v = z["v_mine"].astype(np.float32)
    sd = float(meta["v_mine"]["proj_sd"])

    def block(**kw):
        rows = []
        for k in range(6):
            r = replay(a.agent, MAP, seed=2000 + k, device=a.device, **kw)
            rows.append(dict(seed=2000 + k, success=bool(r["success"]),
                             steps=r["steps"], door=r["door"],
                             mines=int(sum(x == A_MINE for x in r["actions"]))))
        return rows

    out = {"map": MAP, "axis_sd": sd, "baseline": block()}
    print("baseline", [(r["mines"], r["success"]) for r in out["baseline"]],
          flush=True)
    for b in LOGIT_BIASES:
        lb = lambda t, b=b: np.array([0, 0, 0, 0, 0, b], np.float32)
        out[f"logit_{b:+.1f}"] = rows = block(logit_bias=lb)
        print(f"logit {b:+.1f}", [(r["mines"], r["success"]) for r in rows],
              flush=True)
    for lam in TOOL_LAMS:
        if a.agent == "storm":
            pl = {"v": v, "gate": (lambda t, lam=lam: (True, lam * sd, True))}
            rows = block(hook=pl)
        else:
            hk = lambda d, t, info, lam=lam: d + (lam * sd) * v
            rows = block(hook=hk)
        out[f"tooladd_{lam:+.1f}"] = rows
        print(f"tooladd {lam:+.1f}", [(r["mines"], r["success"]) for r in rows],
              flush=True)
    p = OUT / a.agent / "pilot_map77.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
