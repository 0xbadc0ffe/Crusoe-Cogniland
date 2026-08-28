#!/usr/bin/env python3
"""Aggregate the PPO behaviour campaign -> summary.json + the quantitative figure.

  PYTHONPATH=... python scripts/mechinterp/behavior_steering/ppo_summarize.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/behavior_steering/ppo"
FIG = REPO / "paper/figures/behavior_steering"

LABEL = {
    "baseline": "baseline",
    "donor_prefix": "suppress:\ndonor-prefix",
    "m1g_0": "suppress:\nroute-axis λ=0",
    "m2_-2": "suppress:\nlogit −2",
    "m3_sup": "suppress:\nSAE clamp",
    "m1g_1.5": "incentivize:\nroute-axis λ=1.5",
    "m1p_1": "incentivize:\nroute-axis ⊥belief",
    "m2_+2": "incentivize:\nlogit +2",
    "m2_both_-2": "suppress both:\nlogit −2",
}
ORDER = {"rocky": ["baseline", "donor_prefix", "m1g_0", "m2_-2", "m3_sup",
                   "m1g_1.5", "m1p_1", "m2_+2"],
         "lakes": ["baseline", "donor_prefix", "m1g_0", "m2_-2", "m3_sup",
                   "m1g_1.5", "m1p_1", "m2_+2"],
         "balanced": ["baseline", "donor_prefix", "m2_-2", "m2_both_-2"]}
TOOL = {"rocky": "mines", "lakes": "builds", "balanced": "mines"}


def main():
    rows = json.loads((OUT / "campaign.json").read_text())
    summary = {}
    for cat in ("rocky", "lakes", "balanced"):
        for cond in sorted({r["cond"] for r in rows if r["cat"] == cat}):
            sub = [r for r in rows if r["cat"] == cat and r["cond"] == cond]
            if not sub:
                continue
            key = TOOL[cat]
            summary[f"{cat}/{cond}"] = dict(
                n=len(sub),
                success=float(np.mean([r["success"] for r in sub])),
                mines=float(np.mean([r["mines"] for r in sub])),
                builds=float(np.mean([r["builds"] for r in sub])),
                tool=float(np.mean([r[key] for r in sub])),
                timeout=float(np.mean([r["steps"] >= 799 for r in sub])),
                steps=float(np.mean([r["steps"] for r in sub])))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    rc = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 8.5,
          "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(13.6, 6.4),
                                 gridspec_kw=dict(height_ratios=[1, 1]))
        for j, cat in enumerate(("rocky", "lakes", "balanced")):
            conds = [c for c in ORDER[cat] if f"{cat}/{c}" in summary]
            xs = np.arange(len(conds))
            tool = [summary[f"{cat}/{c}"]["tool"] for c in conds]
            succ = [summary[f"{cat}/{c}"]["success"] for c in conds]
            ns = [summary[f"{cat}/{c}"]["n"] for c in conds]
            base_tool = summary[f"{cat}/baseline"]["tool"]

            ax = axes[0, j]
            cols = ["#6b7280" if c == "baseline" else
                    "#b91c1c" if c.startswith(("donor", "m1g_0", "m2_-", "m3", "m2_both"))
                    else "#16a34a" for c in conds]
            ax.bar(xs, tool, color=cols, width=.62)
            ax.margins(y=0.14)
            ax.axhline(base_tool, color="#111827", lw=1, ls=":")
            for x, v, n in zip(xs, tool, ns):
                lbl = f"{v:.1f}" if n == ns[0] else f"{v:.1f}\n(n={n})"
                ax.annotate(lbl, (x, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=7.0)
            ax.set_xticks(xs)
            ax.set_xticklabels([LABEL.get(c, c) for c in conds], fontsize=6.4,
                               rotation=28, ha="right")
            ax.set_ylabel(f"{TOOL[cat]} / episode" if j == 0 else "")
            ax.set_title(f"{cat}  (n={ns[0]} held-out maps)", loc="left", fontsize=9.5)

            ax = axes[1, j]
            ax.bar(xs, [s * 100 for s in succ], color=cols, width=.62)
            for x, v in zip(xs, succ):
                ax.annotate(f"{v*100:.0f}", (x, v * 100), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=7.4)
            ax.set_ylim(0, 108)
            ax.set_xticks(xs)
            ax.set_xticklabels([LABEL.get(c, c) for c in conds], fontsize=6.4,
                               rotation=28, ha="right")
            ax.set_ylabel("success (%)" if j == 0 else "")
        fig.suptitle("PPO behaviour steering on held-out maps: tool use (top) and "
                     "task success (bottom), baseline grey, suppression red, "
                     "incentive green.\nBars with their own (n=..) run on the "
                     "donor-covered subset only, whose baselines differ from the "
                     "full-set dotted line.", y=1.02, fontsize=10.5)
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "ppo_beh_summary.png", bbox_inches="tight")
        print("wrote ppo_beh_summary.png")

    for k, v in summary.items():
        print(f"{k:28s} n={v['n']:3d} succ {v['success']:.2f} "
              f"tool {v['tool']:5.1f} to {v['timeout']:.2f}")


if __name__ == "__main__":
    main()
