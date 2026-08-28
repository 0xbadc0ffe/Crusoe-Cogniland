#!/usr/bin/env python3
"""Behaviour dispreference by activation steering — implementation + smoke test.

Modes (--suppress): mine | build | both | none, where `none` means "go as
straight as possible": suppress BOTH tool axes and add a straightness push.

Intervention (sustained, every step): a projection SHRINK per suppressed axis
    h' = h - beta (h.v_hat) v_hat        (beta=1 removes the component,
                                          beta=2 reflects it)
plus, in `none` mode, a bounded additive push along v_straight scaled in
projection-SD units (0.5 * beta * sd_straight). Shrinks are state-dependent
and bounded, so they avoid the absolute-add overdose failure seen in the
belief campaign.

Every steered episode is compared with the seed-exact baseline replay of the
same map. The belief-leak check reads the late-corridor projection of h on the
belief axis in both runs.

  smoke grid (default):  --arm smoke   (12 test maps/category x modes x betas
                                        x {raw, belief-orthogonalised})
  single condition:      --arm one --suppress mine --beta 1.0 --variant raw --n 100
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/mechinterp"))
sys.path.insert(0, str(REPO / "scripts/mechinterp/belief_report"))
from data import load, split_maps  # noqa: E402
from replay_episode import replay  # noqa: E402

OUT = REPO / "outputs/behavior_steering"
FIG = REPO / "paper/figures/behavior_steering"
KIT = OUT / "behavior_axes.npz"
A_BUILD, A_MINE, A_RIGHT = 4, 5, 3
BETAS = [0.5, 1.0, 2.0]
MODES = ["mine", "build", "both", "none"]
SD_STRAIGHT_GAIN = 0.5      # additive push = GAIN * beta * proj_sd(v_straight)


def make_hook(sup_axes, add_axis=None, add_mag=0.0, beta=1.0):
    def hook(h, t, info):
        for v in sup_axes:
            h = h - beta * float(h @ v) * v
        if add_axis is not None:
            h = h + add_mag * add_axis
        return h
    return hook


def corridor_proj(res, wall_col, v_b):
    """Mean projection on the belief axis over steps in the late corridor."""
    ps = []
    for (r, c), f in zip(res["positions"], res["features"]):
        if wall_col - 8 <= c < wall_col:
            ps.append(float(np.asarray(f["h"], np.float32) @ v_b))
    return float(np.mean(ps)) if ps else None


def summarise(res, base, wall_col, v_b, row0):
    a = np.asarray(res["actions"])
    rows = np.asarray([p[0] for p in res["positions"]], float)
    return dict(
        mines=int((a == A_MINE).sum()), builds=int((a == A_BUILD).sum()),
        success=bool(res["success"]), door=res["door"], steps=res["steps"],
        straight=round(float(np.abs(rows - row0).mean()), 2),
        belief_proj=corridor_proj(res, wall_col, v_b),
        belief_proj_base=corridor_proj(base, wall_col, v_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="smoke", choices=["smoke", "one"])
    ap.add_argument("--suppress", choices=MODES)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--variant", default="raw", choices=["raw", "perp"])
    ap.add_argument("--n", type=int, default=12, help="test maps per category")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    kit = np.load(KIT)
    v_b = kit["v_belief"].astype(np.float32)
    sd_straight = 2.477          # train proj SD (behavior_axes_meta.json)

    def axes_for(mode, variant):
        sfx = "_perp" if variant == "perp" else ""
        sup, add, mag = [], None, 0.0
        if mode in ("mine", "both", "none"):
            sup.append(kit["v_mine" + sfx].astype(np.float32))
        if mode in ("build", "both", "none"):
            sup.append(kit["v_build" + sfx].astype(np.float32))
        if mode == "none":
            add = kit["v_straight" + sfx].astype(np.float32)
        return sup, add

    _, df = load("ppo")
    tr, te = split_maps(df)
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    percat = {}
    per = df.drop_duplicates("map_id")[["map_id", "category"]]
    for c in ("lakes", "balanced", "rocky"):
        ids = [int(i) for i in per.loc[per.category == c, "map_id"] if i in set(te)]
        percat[c] = sorted(ids)[:a.n]
    maps = [(c, m) for c in percat for m in percat[c]]

    conds = ([(m, b, v) for m in MODES for b in BETAS for v in ("raw", "perp")]
             if a.arm == "smoke" else [(a.suppress, a.beta, a.variant)])

    base_cache = {}
    rows = []
    for cat, mid in maps:
        wall_col = int(pool[mid].wall_col)
        base = replay("ppo", mid, device=a.device)
        base_cache[mid] = base
        row0 = base["positions"][0][0]
        rows.append(dict(cond="baseline", cat=cat, map_id=mid,
                         **summarise(base, base, wall_col, v_b, row0)))
    print(f"baselines done ({len(maps)} maps)", flush=True)

    for mode, beta, variant in conds:
        sup, add = axes_for(mode, variant)
        mag = SD_STRAIGHT_GAIN * beta * sd_straight if add is not None else 0.0
        for cat, mid in maps:
            wall_col = int(pool[mid].wall_col)
            base = base_cache[mid]
            row0 = base["positions"][0][0]
            res = replay("ppo", mid, device=a.device,
                         hook=make_hook(sup, add, mag, beta))
            rows.append(dict(cond=f"{mode}|b{beta}|{variant}", cat=cat,
                             map_id=mid, mode=mode, beta=beta, variant=variant,
                             **summarise(res, base, wall_col, v_b, row0)))
        print(f"{mode} beta={beta} {variant} done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / ("smoke_results.json" if a.arm == "smoke"
                 else f"steer_{a.suppress}_{a.beta}_{a.variant}.json")
    out.write_text(json.dumps(dict(n_maps=len(maps), rows=rows)))
    print("wrote", out, f"({len(rows)} episodes)")

    if a.arm == "smoke":
        figure(rows)


def agg(rows, cond):
    r = [x for x in rows if x["cond"] == cond]
    if not r:
        return None
    return dict(n=len(r),
                tools=float(np.mean([x["mines"] + x["builds"] for x in r])),
                mines=float(np.mean([x["mines"] for x in r])),
                builds=float(np.mean([x["builds"] for x in r])),
                success=float(np.mean([x["success"] for x in r])),
                steps=float(np.mean([x["steps"] for x in r])),
                straight=float(np.mean([x["straight"] for x in r])),
                dproj=float(np.mean([(x["belief_proj"] or 0) - (x["belief_proj_base"] or 0)
                                     for x in r
                                     if x["belief_proj"] is not None
                                     and x["belief_proj_base"] is not None])))


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = agg(rows, "baseline")
    with plt.rc_context({"figure.dpi": 150, "font.size": 8.5}):
        fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))
        X = np.arange(len(MODES))
        W = 0.38
        for k, (variant, col) in enumerate((("raw", "#d97706"), ("perp", "#7c3aed"))):
            tools = [agg(rows, f"{m}|b1.0|{variant}")["tools"] for m in MODES]
            succ = [agg(rows, f"{m}|b1.0|{variant}")["success"] for m in MODES]
            dpj = [agg(rows, f"{m}|b1.0|{variant}")["dproj"] for m in MODES]
            off = (k - .5) * W
            ax[0].bar(X + off, tools, W, color=col,
                      label="raw axis" if k == 0 else "belief-orthogonalised")
            ax[1].bar(X + off, succ, W, color=col)
            ax[2].bar(X + off, dpj, W, color=col)
        ax[0].axhline(base["tools"], color="#111827", ls="--", lw=1.2,
                      label=f"baseline ({base['tools']:.1f})")
        ax[1].axhline(base["success"], color="#111827", ls="--", lw=1.2,
                      label=f"baseline ({base['success']:.2f})")
        ax[2].axhline(0, color="#111827", ls="--", lw=1)
        for i in range(3):
            ax[i].set_xticks(X)
            ax[i].set_xticklabels(["−mine", "−build", "−both", "none\n(straight)"])
        ax[0].set_ylabel("tool actions / episode"); ax[0].set_ylim(0, None)
        ax[0].set_title("(a) tool use under suppression (β=1)", loc="left")
        ax[0].legend(frameon=False, fontsize=7)
        ax[1].set_ylabel("success rate"); ax[1].set_ylim(0, 1.05)
        ax[1].set_title("(b) task success", loc="left")
        ax[1].legend(frameon=False, fontsize=7)
        ax[2].set_ylabel("Δ belief projection (corridor)")
        ax[2].set_title("(c) belief leak (steered − baseline)", loc="left")
        fig.suptitle("Dispreference smoke test — 36 held-out maps, sustained "
                     "projection shrink, point estimates only", y=1.03)
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "fig_smoke_dispreference.png", bbox_inches="tight")
        print("wrote fig_smoke_dispreference.png")


if __name__ == "__main__":
    main()
