#!/usr/bin/env python3
"""The correction ablation: GradientClamp with and without `project_out`.

The standard method (`src/cogniland/bridge_tunnel/steering.py`, GradientClamp)
clamps P(tool) below a threshold by stepping the recurrent state along the
head's own gradient. The correction under test is the module's `project_out`
hook, fed the belief axis, which removes the belief component of every applied
edit.

Because the clamp iterates until the constraint is met, BOTH variants satisfy
the same P(tool) < threshold at the same threshold: the behavioural target is
matched by construction, so any difference in the door statistic is the
correction's doing and not a dose difference. Baseline rows are unsteered in
both runs and must be identical -- that identity is checked here and used as
the sham.

  PYTHONPATH=src python scripts/mechinterp/behavior_steering/act4_ablation.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/mechinterp/behavior_steering"))
from act4_analyze import perm_p, stars  # noqa: E402
ACT4 = REPO / "outputs/behavior_steering/act4"
FIG = REPO / "paper/figures/behavior_steering"
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}
CONDS = ("sup_mine", "sup_build", "sup_both")
LBL = {"sup_mine": "suppress mine", "sup_build": "suppress build",
       "sup_both": "suppress both"}
C_OFF, C_ON = "#dc2626", "#2563eb"


def load(tag):
    return json.loads((ACT4 / f"{tag}.json").read_text())


def bymap(rows):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["map_id"]][r["cond"]].append(r)
    return g


def ptop(rs):
    """P(top door) over episodes that actually ENDED at a door."""
    d = [x["door"] for x in rs
         if not x.get("timeout") and x["door"] in ("top", "bottom")]
    return (sum(1 for x in d if x == "top") / len(d)) if d else None


def tool_pct(rows, key):
    """Per-map ±% against that map's own baseline, averaged over maps whose
    baseline uses the tool (the campaign's headline convention)."""
    g = bymap(rows)
    out = {}
    for cond in CONDS:
        vals = []
        for mid, cs in g.items():
            b = np.mean([x[key] for x in cs["baseline"]])
            if b <= 0:
                continue
            s = np.mean([x[key] for x in cs[cond]])
            vals.append(100 * (s - b) / b)
        out[cond] = (float(np.mean(vals)), len(vals))
    return out


def main():
    on, off = load("test_ppo_clamp"), load("test_ppo_clamp_noorth")
    null = load("test_ppo_clamp_null")
    mk = "true_mines" if "true_mines" in on[0] else "mines"
    bk = "true_builds" if "true_builds" in on[0] else "builds"

    # --- the sham: unsteered rows must be identical in both runs ---
    def basekey(rows):
        return sorted((r["map_id"], r["steps"], r["door"], r[mk], r[bk])
                      for r in rows if r["cond"] == "baseline")
    same = basekey(on) == basekey(off)
    print(f"baseline identity (sham): {'MATCH' if same else 'DIFFERS'} "
          f"({len(basekey(on))} episodes)")

    G = {"on": bymap(on), "off": bymap(off), "null": bymap(null)}
    rng = np.random.default_rng(0)
    rows_out = {"sham_baseline_identical": bool(same)}
    print(f"\n{'condition':14s} {'variant':9s} {'target%':>8s} {'off-tgt%':>9s} "
          f"{'succ':>5s} {'TO':>5s} {'|Δdoor|':>8s} {'≥20pp':>7s} {'p vs null':>10s}")
    for cond in CONDS:
        tgt = mk if cond in ("sup_mine", "sup_both") else bk
        oth = bk if tgt == mk else mk
        for var in ("off", "on"):
            src = off if var == "off" else on
            tp = tool_pct(src, tgt)[cond]
            op = tool_pct(src, oth)[cond]
            sub = [r for r in src if r["cond"] == cond]
            succ = np.mean([r["success"] for r in sub])
            to = np.mean([r["timeout"] for r in sub])
            shifts, nulls = [], []
            for mid, cs in G[var].items():
                b, s = ptop(cs["baseline"]), ptop(cs[cond])
                nb = ptop(G["null"].get(mid, {}).get("baseline", []))
                if None in (b, s, nb):
                    continue
                shifts.append(abs(s - b)); nulls.append(abs(nb - b))
            shifts, nulls = np.array(shifts), np.array(nulls)
            d = shifts - nulls
            B = 20000
            cnt = sum(1 for _ in range(B)
                      if (d * rng.choice([-1, 1], len(d))).mean() >= d.mean())
            p = (cnt + 1) / (B + 1)
            mv = int((shifts >= 0.20).sum())
            rows_out[f"{cond}/{var}"] = dict(
                target_pct=round(tp[0], 1), n_target_maps=tp[1],
                offtarget_pct=round(op[0], 1), success=round(float(succ), 3),
                timeout=round(float(to), 3), door_abs=round(float(shifts.mean()), 3),
                null_abs=round(float(nulls.mean()), 3), moved_20pp=mv,
                n_maps=len(shifts), perm_p=round(p, 4))
            print(f"{LBL[cond]:14s} {'without' if var=='off' else 'WITH':9s} "
                  f"{tp[0]:+8.1f} {op[0]:+9.1f} {succ:5.2f} {to:5.2f} "
                  f"{shifts.mean():8.3f} {mv:3d}/{len(shifts):<3d} {p:10.4f}")

    # the paired test that carries the thesis claim: does the correction
    # reduce the decision shift on the SAME maps and seeds?
    paired = {}
    for cond in CONDS:
        off_d, on_d = [], []
        for mid in G["on"]:
            b = ptop(G["on"][mid]["baseline"])
            s_on = ptop(G["on"][mid][cond])
            s_off = ptop(G["off"].get(mid, {}).get(cond, []))
            if None in (b, s_on, s_off):
                continue
            off_d.append(abs(s_off - b)); on_d.append(abs(s_on - b))
        pv = perm_p(off_d, on_d)          # H1: without > with
        paired[cond] = dict(without=round(float(np.mean(off_d)), 3),
                            with_=round(float(np.mean(on_d)), 3),
                            reduction_pct=round(100 * (np.mean(off_d) - np.mean(on_d))
                                                / np.mean(off_d), 1),
                            perm_p=round(pv, 5), stars=stars(pv), n_maps=len(on_d))
        print(f"  paired {LBL[cond]:14s} {paired[cond]['without']:.3f} -> "
              f"{paired[cond]['with_']:.3f}  ({paired[cond]['reduction_pct']:+.0f}%) "
              f"p={pv:.4f} {paired[cond]['stars']}")
    rows_out["_paired_with_vs_without"] = paired
    (ACT4 / "ablation_projectout.json").write_text(json.dumps(rows_out, indent=1))

    # ---- figure: the correction's effect, per condition ----
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
        xs = np.arange(len(CONDS)); w = 0.36
        ax = axes[0]
        for i, (var, c, lab) in enumerate((("off", C_OFF, "without project_out"),
                                           ("on", C_ON, "with project_out"))):
            ax.bar(xs + (i - .5) * w,
                   [abs(rows_out[f'{c2}/{var}']["target_pct"]) for c2 in CONDS],
                   w, color=c, label=lab)
        ax.set_xticks(xs, [LBL[c] for c in CONDS], fontsize=7.6)
        ax.set_ylim(0)
        ax.set_ylabel("commanded tool cut (%)")
        ax.set_title("behaviour: the correction costs nothing", loc="left",
                     fontsize=9)
        ax.legend(fontsize=7.2, frameon=False)

        ax = axes[1]
        for i, (var, c) in enumerate((("off", C_OFF), ("on", C_ON))):
            vals = [rows_out[f'{c2}/{var}']["door_abs"] for c2 in CONDS]
            ax.bar(xs + (i - .5) * w, vals, w, color=c)
            for x, v, c2 in zip(xs + (i - .5) * w, vals, CONDS):
                ax.text(x, v + .004, stars(rows_out[f'{c2}/{var}']["perm_p"]),
                        ha="center", va="bottom", fontsize=6.4, color="#111827")
        # paired with-vs-without bracket above each pair
        top = max(rows_out[f'{c2}/{v}']["door_abs"]
                  for c2 in CONDS for v in ("on", "off"))
        for j, c2 in enumerate(CONDS):
            y = top * (1.13 + .07 * (j % 2))
            ax.plot([j - w / 2, j - w / 2, j + w / 2, j + w / 2],
                    [y - top * .02, y, y, y - top * .02], color="#111827", lw=.8)
            ax.text(j, y + top * .005, paired[c2]["stars"], ha="center",
                    va="bottom", fontsize=7.2, color="#111827")
        ax.set_ylim(0, top * 1.34)
        nf = np.mean([rows_out[f"{c}/on"]["null_abs"] for c in CONDS])
        ax.axhline(nf, color="#111827", lw=1.0, ls=":")
        ax.text(len(CONDS) - .5, nf, " noise floor", fontsize=7,
                va="bottom", ha="right")
        ax.set_xticks(xs, [LBL[c] for c in CONDS], fontsize=7.6)
        ax.set_ylabel("|Δ P(top door)|, paired per map")
        ax.set_title("decision: what the correction buys", loc="left", fontsize=9)

        ax = axes[2]
        for i, (var, c) in enumerate((("off", C_OFF), ("on", C_ON))):
            ax.bar(xs + (i - .5) * w,
                   [100 * rows_out[f'{c2}/{var}']["moved_20pp"] /
                    rows_out[f'{c2}/{var}']["n_maps"] for c2 in CONDS], w,
                   color=c)
        ax.set_xticks(xs, [LBL[c] for c in CONDS], fontsize=7.6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("maps whose door moved ≥20pp (%)")
        ax.set_title("how often the decision is corrupted", loc="left",
                     fontsize=9)
        fig.suptitle(
            "GradientClamp on held-out balanced maps: the standard method "
            "(red) against the same clamp with the belief-orthogonalising\n"
            "correction (blue). Both satisfy the SAME P(tool) constraint, so "
            "the behaviour is matched and only the decision differs.\n"
            "Stars on bars: paired permutation against that variant's noise "
            "floor. Stars on brackets: with vs without, paired per map.  "
            "ns p≥0.05  * <0.05  ** <0.01  *** <0.001  **** <1e-4",
            y=1.07, fontsize=10.4)
        fig.tight_layout()
        fig.savefig(FIG / "act4_ablation_projectout.png", bbox_inches="tight")
        print("\nwrote act4_ablation_projectout.png")


if __name__ == "__main__":
    main()
