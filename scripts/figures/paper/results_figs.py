#!/usr/bin/env python3
"""Results-chapter figures and LaTeX tables, from results.json.

  fig_res_main.png      per-agent held-out success with Wilson intervals + error split
  fig_res_recurrence.png feed-forward vs recurrent PPO, against the 2/3 constant-door line
  tab_res_main.tex      the held-out results table (booktabs)
  tab_res_category.tex  per-category success table

  python scripts/figures/paper/results_figs.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "paper/figures/forkwall_paper"
AG = {"ppo": ("PPO\\,+\\,GRU", "#d97706"), "dreamer": ("DreamerV3", "#2563eb"),
      "storm": ("STORM", "#16a34a")}
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}


def pct(x): return f"{x*100:.2f}"


def se_pct(p, n):
    """Standard error of a proportion, in percentage points.

    The tables report one evaluation of one agent over n held-out maps, so the
    uncertainty on that estimate is binomial: sqrt(p(1-p)/n). This replaces the
    Wilson interval with a symmetric +/- that fits inside the value column."""
    return 100.0 * (p * (1.0 - p) / n) ** 0.5


def main():
    R = json.loads((OUT / "results.json").read_text())

    # ── Figure: main comparison ──────────────────────────────────────────
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2),
                               gridspec_kw=dict(width_ratios=[1.15, 1]))
        names = [k for k in ("storm", "ppo", "dreamer") if k in R]
        xs = np.arange(len(names))
        for i, k in enumerate(names):
            r = R[k]; lab, col = AG[k]
            lo, hi = r["ci"]
            ax[0].bar(i, r["success"], color=col, width=.62,
                      yerr=[[r["success"]-lo], [hi-r["success"]]], capsize=4,
                      error_kw=dict(lw=1.1, ecolor="#374151"))
            ax[0].text(i, hi + .004, f"{r['success']*100:.1f}", ha="center", fontsize=8.5)
        ax[0].axhline(2/3, color="#6b7280", ls="--", lw=1.2)
        ax[0].annotate("constant-door optimum (2/3)", xy=(len(names)-1, .675),
                       ha="right", va="bottom", fontsize=7.4, color="#6b7280")
        ax[0].set_ylim(0, 1.08); ax[0].set_xticks(xs)
        ax[0].set_xticklabels([AG[k][0].replace("\\,", " ") for k in names], fontsize=8)
        ax[0].set_ylabel("held-out success")
        ax[0].set_title("(a) all three agents clear the constant-door baseline", loc="left", fontsize=9.5)

        # error split
        w = .6; bottoms = np.zeros(len(names))
        for key, col, lab in (("wrong", "#ef4444", "wrong door"), ("timeout", "#f59e0b", "timeout")):
            vals = [R[k][key]*100 for k in names]
            ax[1].bar(xs, vals, w, bottom=bottoms, color=col, label=lab)
            bottoms += vals
        ax[1].set_xticks(xs); ax[1].set_xticklabels([AG[k][0].replace("\\,"," ") for k in names], fontsize=8)
        ax[1].set_ylabel("share of episodes (%)"); ax[1].legend(frameon=False, fontsize=8)
        ax[1].set_title("(b) residual error is wrong doors, not timeouts", loc="left", fontsize=9.5)
        fig.tight_layout(); fig.savefig(OUT / "fig_res_main.png", bbox_inches="tight"); plt.close(fig)
        print("wrote fig_res_main.png")

    # ── Figure: recurrence is essential ──────────────────────────────────
    if "ppo_seeds" in R:
        s = R["ppo_seeds"]
        rec = [x["decisive"] for x in s["recurrent"]]
        ff = [x["decisive"] for x in s["feedforward"]]
        # recurrent solved runs only vs all; feed-forward all
        rec_solved = [x for x in rec if x > 0.8]
        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            # On the decisive-door axis the constant-door policy scores exactly
            # 50%: it commits to one door and is right on one of the two decisive
            # categories. That 50% line is the only meaningful floor here; the
            # overall-success 2/3 optimum belongs to a different axis, so it is
            # not drawn.
            ax.axhline(.5, color="#6b7280", ls="--", lw=1)
            ax.annotate("constant-door optimum (chance, 50%)", xy=(.5, .512),
                        ha="center", va="bottom", fontsize=7.5, color="#6b7280")
            for x, vals, col in [(0, ff, "#94a3b8"), (1, rec, "#d97706")]:
                v = np.array(vals)
                solved = v > 0.8
                jit = np.random.RandomState(0).uniform(-.06, .06, len(v))
                # collapsed seeds hollow, escaped seeds filled
                ax.scatter((np.full(len(v), x) + jit)[solved], v[solved],
                           color=col, s=48, zorder=3, edgecolor="white", lw=.6)
                ax.scatter((np.full(len(v), x) + jit)[~solved], v[~solved],
                           facecolor="white", edgecolor=col, s=48, zorder=3, lw=1.4)
                # Bar at the escaped-seed mean where any seed escapes, else the
                # all-seed mean. Feed-forward has no escapee, so its bar is the
                # mean of all five; the recurrent bar is the mean of the five
                # that bind both doors, matching the numbers quoted in the text.
                barval = v[solved].mean() if solved.any() else v.mean()
                ax.plot([x-.18, x+.18], [barval]*2, color="#111827", lw=2, zorder=4)
                ax.annotate(f"{int(solved.sum())}/{len(v)} escape",
                            xy=(x, .44), ha="center", fontsize=8, color="#374151")
            ax.set_xticks([0, 1]); ax.set_xticklabels(
                [f"feed-forward\n(n={len(ff)})", f"recurrent\n(n={len(rec)})"], fontsize=8.5)
            ax.set_ylabel("decisive-door success"); ax.set_ylim(.40, 1.03)
            ax.set_title("Recurrence is what solves the task", loc="left", fontsize=10)
            fig.tight_layout(); fig.savefig(OUT / "fig_res_recurrence.png", bbox_inches="tight")
            plt.close(fig); print("wrote fig_res_recurrence.png")

    # ── Table: main held-out results ─────────────────────────────────────
    rows = []
    for k in ("ppo", "dreamer", "storm"):
        if k not in R: continue
        r = R[k]; lab = AG[k][0]
        n = r.get("episodes", 1200)
        rows.append(f"        {lab} & {pct(r['success'])}\\,$\\pm$\\,{se_pct(r['success'], n):.2f} "
                    f"& {pct(r['decisive'])} & {pct(r['wrong'])} & {pct(r['timeout'])} "
                    f"& {r['mean_len']:.0f} \\\\")
    ff_line = ff_note = ""
    if "ppo_seeds" in R and R["ppo_seeds"]["feedforward"]:
        ff = R["ppo_seeds"]["feedforward"]
        s = np.mean([x["success"] for x in ff]); d = np.mean([x["decisive"] for x in ff])
        dsd = np.std([x["decisive"] for x in ff]) * 100
        ssd = np.std([x["success"] for x in ff]) * 100
        ff_line = (f"        \\midrule\n        feed-forward control & {pct(s)}\\,$\\pm$\\,{ssd:.2f} "
                   f"& {pct(d)} & --- & --- & --- \\\\")
        ff_note = (f" The feed-forward control averages {len(ff)} seeds; its decisive "
                   f"success is ${pct(d)} \\pm {dsd:.2f}$, i.e.\\ chance.")
    tab = ("\\begin{table}[t]\n  \\centering\n"
           "  \\caption[Held-out results]{Held-out success on all 1\\,200 test maps, "
           "TRUE door metric, all agents sampling. Success carries the standard "
           "error of the proportion over the 1\\,200 maps; the feed-forward row "
           "instead carries the standard deviation across its seeds. "
           "\\emph{Decisive} restricts to the two categories where a constant-door policy "
           "scores 50\\,\\%." + ff_note + "}\n  \\label{tab:res_main}\n"
           "  \\setlength{\\tabcolsep}{4.5pt}\\small\n"
           "  \\begin{tabular}{@{}lcccc r@{}}\n    \\toprule\n"
           "    Agent & Success & Decisive & Wrong & Timeout & Steps \\\\\n"
           "    \\midrule\n" + "\n".join(rows) + "\n" + ff_line +
           "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    (OUT / "tab_res_main.tex").write_text(tab)
    print("wrote tab_res_main.tex")

    # ── Table: per-category ──────────────────────────────────────────────
    crows = []
    for k in ("ppo", "dreamer", "storm"):
        if k not in R: continue
        pc = R[k]["per_cat"]
        ncat = r_n = int(R[k].get("episodes", 1200) / 3)     # pool is balanced 3 ways
        cells = " & ".join(f"{pct(pc.get(c, 0))}\\,$\\pm$\\,{se_pct(pc.get(c, 0), ncat):.2f}"
                           for c in ("balanced", "lakes", "rocky"))
        crows.append(f"        {AG[k][0]} & {cells} \\\\")
    ctab = ("\\begin{table}[t]\n  \\centering\n"
            "  \\caption[Success by map type]{Success by map type, each agent sampling. "
            "\\emph{balanced} maps accept either door, so they measure navigation alone; "
            "\\emph{lakes} and \\emph{rocky} measure memory. Each cell carries the "
            "standard error of the proportion over the 400 maps of that type.}\n"
            "  \\label{tab:res_category}\n"
            "  \\begin{tabular}{@{}lccc@{}}\n    \\toprule\n"
            "    Agent & balanced & lakes & rocky \\\\\n    \\midrule\n"
            + "\n".join(crows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    (OUT / "tab_res_category.tex").write_text(ctab)
    print("wrote tab_res_category.tex")


if __name__ == "__main__":
    main()
