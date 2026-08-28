#!/usr/bin/env python3
"""Act two for the world models: the CAA-style tool-axis displacement at doses
around each agent's flip knee, with the belief readback recorded per episode.

Last night's framing measured these doses as failures ("0/6"); the act-two
question decomposes them: a failure that COMPLETES at the wrong door is the
decision flip the thesis claims, a timeout is the uninformative overdose. The
existing pilots say Dreamer flips at -1 sd (6/6 wrong, 0 timeouts) and STORM
starts at -4 sd (0.17) -- this grid measures the dose-response on held-out
maps, plus the readback that ties the flip to the belief coordinate.

  # dreamer (conda r2dreamer)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:r2dreamer_model:scripts/mechinterp:scripts/figures \
    python scripts/mechinterp/behavior_steering/act2_wm.py --agent dreamer --stage sham
  ... --stage grid
  # storm (STORM_model/.venv)
  ... PYTHONPATH=src:STORM_model:scripts/mechinterp:scripts/figures ... --agent storm
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

OUT = REPO / "outputs/behavior_steering/act2"
A_BUILD, A_MINE = 4, 5

# per-agent belief readback: (axis key, mus key suffix, feature key, window)
READBACK = {
    "dreamer": dict(vkey="v", mkey="", feat="deter", win=(-8, 0), prewall=True),
    "storm": dict(vkey="v_wall", mkey="_wall", feat="h", win=(0, 3), prewall=False),
}


def load_kit(agent):
    zb = np.load(REPO / f"outputs/belief_report/steer_axis_{agent}.npz")
    za = np.load(REPO / f"outputs/behavior_steering/{agent}/axes.npz")
    meta = json.loads(
        (REPO / f"outputs/behavior_steering/{agent}/axes_meta.json").read_text())
    rb = READBACK[agent]
    v_bel = zb[rb["vkey"]].astype(np.float32)
    v_bel /= np.linalg.norm(v_bel) + 1e-12
    return dict(
        v_bel=v_bel,
        mu_l=float(zb["mu_lakes" + rb["mkey"]]),
        mu_r=float(zb["mu_rocky" + rb["mkey"]]),
        test={c: [int(x) for x in zb[c]] for c in ("lakes", "rocky", "balanced")},
        v_mine=za["v_mine"].astype(np.float32),
        v_build=za["v_build"].astype(np.float32),
        sd_mine=float(meta["v_mine"]["proj_sd"]),
        sd_build=float(meta["v_build"]["proj_sd"]),
        cos_mine=float(meta["cos_to_belief"]
                       ["v_mine" + ("_vs_wall" if agent == "storm" else "")]),
        cos_build=float(meta["cos_to_belief"]
                        ["v_build" + ("_vs_wall" if agent == "storm" else "")]),
    )


def make_iv(agent, kit, tool, lam_sd):
    """Additive displacement delta = lam_sd * proj_sd along the tool axis,
    every step -- byte-for-byte the wm_campaign 'tooladd' construction."""
    v = kit["v_mine"] if tool == "mine" else kit["v_build"]
    sd = kit["sd_mine"] if tool == "mine" else kit["sd_build"]
    delta = float(lam_sd * sd)
    if agent == "storm":
        return {"v": v, "gate": lambda t: (True, delta, True)}
    def hook(d, t, info, v=v, delta=delta):
        return d + delta * v
    return hook


def belief_readback(agent, r, wall):
    rb = READBACK[agent]
    kit = belief_readback._kit
    feats = r["features"]
    if not feats:
        return float("nan")
    H = np.stack([np.asarray(f[rb["feat"]], np.float32) for f in feats])
    cols = np.array([p[1] for p in r["positions"][:len(H)]])
    crw = cols - wall
    if rb["prewall"]:
        crossed = np.where(cols >= wall)[0]
        stop = int(crossed[0]) if len(crossed) else len(H)
        crw, H2, cols2 = crw[:stop], H[:stop], cols[:stop]
    else:
        H2, cols2 = H, cols
    m = (crw[:len(H2)] >= rb["win"][0]) & (crw[:len(H2)] < rb["win"][1])
    proj = H2 @ kit["v_bel"]
    if m.any():
        return float(proj[m].mean())
    if len(H2):
        return float(proj[np.argmax(cols2)])
    return float("nan")


def run_row(agent, kit, mid, wall, dev, tool, lam_sd, cond, lbias=None):
    iv = None if lam_sd is None else make_iv(agent, kit, tool, lam_sd)
    r = replay(agent, mid, hook=iv, device=dev, logit_bias=lbias)
    acts = r["actions"]
    to = r["steps"] >= 799
    cos = None
    if lam_sd is not None:
        cos = float(np.sign(lam_sd)) * (kit["cos_mine"] if tool == "mine"
                                        else kit["cos_build"])
    return dict(cond=cond, tool=tool, map_id=mid, cat=r["category"],
                lam_sd=lam_sd,
                mines=int(sum(a == A_MINE for a in acts)),
                builds=int(sum(a == A_BUILD for a in acts)),
                steps=r["steps"], door=r["door"], success=bool(r["success"]),
                timeout=bool(to), wrong=bool((not r["success"]) and (not to)),
                proj=belief_readback(agent, r, wall), cos_delta=cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["dreamer", "storm"])
    ap.add_argument("--stage", required=True, choices=["sham", "grid", "logit"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-rocky", type=int, default=25)
    ap.add_argument("--n-lakes", type=int, default=15)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    kit = load_kit(a.agent)
    belief_readback._kit = kit
    import pickle
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))

    if a.stage == "sham":
        from act2_ppo import archive_sham
        bad, rows = 0, []
        for mid in kit["test"]["rocky"][:3]:
            base = replay(a.agent, mid, device=a.device)
            iv = make_iv(a.agent, kit, "mine", 0.0)
            sham = replay(a.agent, mid, hook=iv, device=a.device)
            same = base["actions"] == sham["actions"]
            bad += not same
            rows.append(dict(cat="rocky", map_id=int(mid), match=bool(same),
                             steps=base["steps"], door=base["door"]))
            print(f"sham map {mid}: {'MATCH' if same else 'DIFFERS'} "
                  f"({base['steps']} vs {sham['steps']} steps)", flush=True)
        archive_sham(a.agent, rows)
        print("SHAM", "PASS" if bad == 0 else f"FAIL ({bad})")
        sys.exit(1 if bad else 0)

    def log(sub, tag):
        n = len(sub)
        print(f"{tag:24s} n={n:3d} succ {np.mean([r['success'] for r in sub]):.2f} "
              f"WRONG {np.mean([r['wrong'] for r in sub]):.2f} "
              f"TO {np.mean([r['timeout'] for r in sub]):.2f} "
              f"mines {np.mean([r['mines'] for r in sub]):5.1f} "
              f"builds {np.mean([r['builds'] for r in sub]):5.1f} "
              f"proj {np.nanmean([r['proj'] for r in sub]):+7.2f}", flush=True)

    if a.stage == "logit":
        # actuator-level contrast: logit +3 on MINE flips ~0.25 of rocky maps
        # in the old grid at zero timeouts -- here rerun WITH the readback, the
        # prediction being that the belief coordinate stays INTACT (the flip
        # is a behavioural detour, not a belief corruption).
        maps = kit["test"]["rocky"][:a.n_rocky]
        walls = {mid: int(pool[mid].wall_col) for mid in maps}
        b = np.zeros(6, np.float32)
        b[A_MINE] = 3.0
        rows = [run_row(a.agent, kit, mid, walls[mid], a.device, "mine",
                        None, "logit_inc_+3.0", lbias=lambda t: b)
                for mid in maps]
        log(rows, "rocky logit inc +3.0")
        (OUT / f"{a.agent}_logit_readback.json").write_text(json.dumps(rows))
        print("wrote", OUT / f"{a.agent}_logit_readback.json", len(rows), "rows")
        return

    doses = dict(
        dreamer=dict(rocky=[-0.25, -0.5, -1.0, -2.0, 0.5, 1.0],
                     lakes=[-0.5, -1.0]),
        storm=dict(rocky=[-2.0, -3.0, -4.0, -6.0, 2.0, 4.0],
                   lakes=[-3.0, -4.0]),
    )[a.agent]
    rows = []

    for cat, tool, n in (("rocky", "mine", a.n_rocky), ("lakes", "build", a.n_lakes)):
        maps = kit["test"][cat][:n]
        walls = {mid: int(pool[mid].wall_col) for mid in maps}
        base = [run_row(a.agent, kit, mid, walls[mid], a.device, tool,
                        None, "baseline") for mid in maps]
        rows += base
        log(base, f"{cat} baseline")
        for lam in doses[cat]:
            sub = [run_row(a.agent, kit, mid, walls[mid], a.device, tool,
                           lam, f"tooladd_{lam:+.2f}") for mid in maps]
            rows += sub
            log(sub, f"{cat} tooladd {lam:+.2f}sd")
            (OUT / f"{a.agent}_tooladd_grid.json").write_text(json.dumps(rows))
    print("wrote", OUT / f"{a.agent}_tooladd_grid.json", len(rows), "rows")


if __name__ == "__main__":
    main()
