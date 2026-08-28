#!/usr/bin/env python3
"""Act three analysis: tables + figures for the plan-steering campaign.

  --what ppo   the horizon-probe arms: does an edit that respects the belief
               still steer the behaviour, or is the plan inseparable from it?
  --what wm    imagination steering: does plan-level control move the tools
               WITHOUT displacing the belief (readback never crosses)?

  PYTHONPATH=src:scripts/mechinterp/belief_report python act3_analyze.py --what ppo
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ACT3 = REPO / "outputs/behavior_steering/act3"
FIG = REPO / "paper/figures/behavior_steering"
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}
C_WRONG, C_TO, C_OK = "#dc2626", "#9ca3af", "#2563eb"

BEL = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz")
MID_PPO = 0.5 * (float(BEL["mu_lakes"]) + float(BEL["mu_rocky"]))


def crossed(cat, proj, mid):
    if proj is None or not np.isfinite(proj):
        return False
    return (proj < mid) if cat == "rocky" else (proj > mid)


def agg(sub):
    f = lambda k: float(np.mean([r[k] for r in sub]))       # noqa: E731
    pr = [r["proj"] for r in sub if r.get("proj") is not None
          and np.isfinite(r["proj"])]
    return dict(n=len(sub), success=round(f("success"), 3),
                wrong=round(f("wrong"), 3), timeout=round(f("timeout"), 3),
                mines=round(f("mines"), 1), builds=round(f("builds"), 1),
                mines_raw=f("mines"), builds_raw=f("builds"),
                steps=round(f("steps"), 1),
                proj=round(float(np.mean(pr)), 2) if pr else None,
                c_mean=(round(f("c_mean"), 4) if "c_mean" in sub[0] else None),
                cos_mean=(round(f("cos_mean"), 3)
                          if "cos_mean" in sub[0] else None))


def ppo():
    rows = json.loads((ACT3 / "ppo_grid.json").read_text())
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["cat"], r["cond"])].append(r)
    table = {}
    for (cat, cond), sub in by.items():
        a = agg(sub)
        a.update(cat=cat, cond=cond,
                 crossed=round(float(np.mean(
                     [crossed(cat, r["proj"], MID_PPO) for r in sub])), 3))
        table[f"{cat}/{cond}"] = a

    # the tool column, expressed against that category's own baseline
    for cat, tool in (("rocky", "mines"), ("lakes", "builds")):
        b = table[f"{cat}/baseline"][tool + "_raw"]     # ratio from UNROUNDED means
        for k, v in table.items():
            if v["cat"] == cat:
                v["tool_ratio"] = round(v[tool + "_raw"] / b, 3) if b else None
                v["tool_key"] = tool

    (ACT3 / "ppo_summary.json").write_text(json.dumps(table, indent=1))

    order = sorted(table, key=lambda k: (table[k]["cat"], k))
    print(f"{'cat/cond':38s} {'n':>3s} {'succ':>5s} {'wrong':>5s} {'TO':>5s} "
          f"{'tool':>6s} {'ratio':>6s} {'c̄':>6s} {'proj':>7s} {'cross':>5s}")
    for k in order:
        t = table[k]
        print(f"{k:38s} {t['n']:3d} {t['success']:5.2f} {t['wrong']:5.2f} "
              f"{t['timeout']:5.2f} {t[t['tool_key']]:6.1f} "
              f"{t['tool_ratio'] if t['tool_ratio'] is not None else float('nan'):6.2f} "
              f"{t['c_mean'] if t['c_mean'] is not None else float('nan'):6.3f} "
              f"{t['proj'] if t['proj'] is not None else float('nan'):+7.2f} "
              f"{t['crossed']:5.2f}")

    # ── figure: the plain/orth comparison, per category ──
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.4))
        for col, (cat, tool) in enumerate((("rocky", "mines"), ("lakes", "builds"))):
            conds = [c for c in table if table[c]["cat"] == cat
                     and table[c]["cond"] != "baseline"
                     and not table[c]["cond"].startswith(("rand_", "lin_set"))
                     and "+" not in table[c]["cond"]]
            pairs = sorted({table[c]["cond"].replace("_orth", "") for c in conds})
            xs = np.arange(len(pairs))
            # top: failure decomposition, plain vs orth
            ax = axes[0][col]
            for off, suff, hatch in ((-0.2, "", None), (0.2, "_orth", "//")):
                W = [table.get(f"{cat}/{p}{suff}") for p in pairs]
                ax.bar(xs + off, [w["wrong"] if w else 0 for w in W], .36,
                       color=C_WRONG, hatch=hatch, edgecolor="white",
                       label="wrong door" + (" (belief-safe)" if suff else ""))
                ax.bar(xs + off, [w["timeout"] if w else 0 for w in W], .36,
                       bottom=[w["wrong"] if w else 0 for w in W], color=C_TO,
                       hatch=hatch, edgecolor="white",
                       label="timeout" + (" (belief-safe)" if suff else ""))
            b = table[f"{cat}/baseline"]
            ax.axhline(b["wrong"], color=C_WRONG, lw=.9, ls=":")
            ax.set_xticks(xs, pairs, fontsize=7, rotation=20, ha="right")
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("failure rate")
            ax.set_title(f"{cat}: solid = plain edit, hatched = belief-orthogonalised",
                         loc="left", fontsize=9)
            if col == 0:
                ax.legend(fontsize=6.6, frameon=False, ncol=2, loc="upper left")
            # bottom: the commanded behaviour, as a share of baseline
            ax = axes[1][col]
            for off, suff, hatch in ((-0.2, "", None), (0.2, "_orth", "//")):
                W = [table.get(f"{cat}/{p}{suff}") for p in pairs]
                ax.bar(xs + off, [w["tool_ratio"] if w else 0 for w in W], .36,
                       color=C_OK, hatch=hatch, edgecolor="white")
            ax.axhline(1.0, color="#111827", lw=.9, ls="--")
            ax.text(len(pairs) - .5, 1.02, "baseline tool use", fontsize=6.8,
                    ha="right", color="#111827")
            ax.set_xticks(xs, pairs, fontsize=7, rotation=20, ha="right")
            ax.set_ylim(0, max(1.35, max(
                [table[c]["tool_ratio"] or 0 for c in conds] + [1.1]) * 1.05))
            ax.set_ylabel(f"{tool} / baseline {tool}")
            ax.set_title("commanded direction: suppression means bars below 1",
                         loc="left", fontsize=9)
        fig.suptitle("PPO horizon-probe steering (Bush et al. adapted): the edits "
                     "flip the door before they cut the tools.\nRemoving the belief "
                     "component halves the flip on the fixed direction, does nothing "
                     "on the gradient, and never delivers the behaviour",
                     y=1.005, fontsize=10.4)
        fig.tight_layout()
        fig.savefig(FIG / "act3_ppo.png", bbox_inches="tight")
        print("wrote act3_ppo.png")


def wm():
    rows = json.loads((ACT3 / "dreamer_mpc_grid.json").read_text())
    cal = json.loads((ACT3 / "dreamer_calib.json").read_text())
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    mid = next((r["midpoint"] for r in rows if r.get("midpoint") is not None), None)
    table = {}
    for cond, sub in by.items():
        a = agg(sub)
        a["crossed"] = (round(float(np.mean(
            [crossed(r["cat"], r["proj"], mid) for r in sub
             if r.get("proj") is not None])), 3) if mid is not None else None)
        table[cond] = a
    (ACT3 / "wm_summary.json").write_text(json.dumps(
        dict(table=table, calibration=cal, midpoint=mid), indent=1))

    print(f"{'cond':16s} {'n':>3s} {'succ':>5s} {'wrong':>5s} {'TO':>5s} "
          f"{'mines':>6s} {'builds':>6s} {'proj':>7s} {'cross':>5s}")
    for cond in sorted(table):
        t = table[cond]
        print(f"{cond:16s} {t['n']:3d} {t['success']:5.2f} {t['wrong']:5.2f} "
              f"{t['timeout']:5.2f} {t['mines']:6.1f} {t['builds']:6.1f} "
              f"{t['proj'] if t['proj'] is not None else float('nan'):+7.2f} "
              f"{t['crossed'] if t['crossed'] is not None else float('nan'):5.2f}")

    # cond names are "plain" and "mpc_<tool>_lam<value>"
    def lam_of(c):
        return 0.0 if c == "plain" else float(c.split("lam")[-1])

    def tool_of(c):
        return "" if c == "plain" else c.split("_")[1]

    conds = ["plain"] + sorted([c for c in table if c.startswith("mpc")],
                               key=lambda c: (tool_of(c) == "both", lam_of(c)))
    lbl = lambda c: ("plain\n(λ=0)" if c == "plain"                     # noqa
                     else f"{tool_of(c)}\nλ={lam_of(c):g}")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.5))
        xs = np.arange(len(conds))
        ax = axes[0]
        w = .38
        ax.bar(xs - w / 2, [table[c]["mines"] for c in conds], w, color=C_OK,
               label="mines")
        ax.bar(xs + w / 2, [table[c]["builds"] for c in conds], w,
               color="#93c5fd", label="builds")
        ax.axhline(table["plain"]["mines"], color="#111827", lw=.9, ls="--")
        ax.text(len(conds) - .5, table["plain"]["mines"], " plain agent",
                fontsize=7, va="bottom", ha="right")
        ax.set_xticks(xs, [lbl(c) for c in conds], fontsize=7)
        ax.set_ylim(0)
        ax.set_ylabel("tool actions per episode")
        ax.set_title("commanded behaviour (watch for substitution)",
                     loc="left", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax = axes[1]
        ax.bar(xs, [table[c]["success"] for c in conds], .6, color="#16a34a")
        ax.set_xticks(xs, [lbl(c) for c in conds], fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("success (true door)")
        ax.set_title("task performance", loc="left", fontsize=9)
        ax = axes[2]
        ax.bar(xs, [table[c]["wrong"] for c in conds], .6, color=C_WRONG,
               label="wrong door")
        ax.bar(xs, [table[c]["timeout"] for c in conds], .6,
               bottom=[table[c]["wrong"] for c in conds], color=C_TO,
               label="timeout")
        ax.set_xticks(xs, [lbl(c) for c in conds], fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("failure rate")
        ax.set_title("the decision survives", loc="left", fontsize=9)
        ax.legend(fontsize=7.2, frameon=False)
        # 4th panel: the belief readback, the claim the title makes
        ax = axes[3]
        zb = np.load(REPO / "outputs/belief_report/steer_axis_dreamer.npz")
        mu_l, mu_r = float(zb["mu_lakes"]), float(zb["mu_rocky"])
        ax.bar(xs, [table[c]["proj"] for c in conds], .6, color="#374151")
        ax.axhline(mu_r, color="#b45309", lw=.9, ls="--")
        ax.axhline(mid, color="#6b7280", lw=.9, ls=":")
        for y, t, c in ((mu_r, "rocky pole", "#b45309"),
                        (mid, "midpoint = flip line", "#6b7280")):
            ax.text(-0.42, y, t, fontsize=6.8, va="bottom", ha="left", color=c,
                    bbox=dict(fc="white", ec="none", pad=.8), zorder=5)
        ax.set_ylim(0, max(mu_r, max(table[c]["proj"] for c in conds)) * 1.22)
        ax.set_xticks(xs, [lbl(c) for c in conds], fontsize=7)
        ax.set_ylabel("belief readback (h·v̂)")
        ax.set_title("the belief does not move", loc="left", fontsize=9)
        auc = cal.get("mine", {}).get("auc")
        n_maps = table["plain"]["n"]
        fig.suptitle(
            "DreamerV3 steered through its own imagination: the policy is "
            "tilted by what it foresees, no latent is edited.\n"
            f"Its tool forecast is calibrated (AUC {auc}); in the working window "
            f"the belief readback does not move.  n = {n_maps} held-out rocky maps.",
            y=1.02, fontsize=10.4)
        fig.tight_layout()
        fig.savefig(FIG / "act3_wm.png", bbox_inches="tight")
        print("wrote act3_wm.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["ppo", "wm"])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    (ppo if a.what == "ppo" else wm)()
