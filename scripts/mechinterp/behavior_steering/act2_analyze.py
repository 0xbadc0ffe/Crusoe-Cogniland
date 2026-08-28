#!/usr/bin/env python3
"""Act-two analysis: decision-flip tables, the belief mechanism, the figures.

Inputs (all produced by act2_ppo.py / act2_wm.py):
  outputs/behavior_steering/act2/ppo_pg_grid.json        PG dose grid + baselines
  outputs/behavior_steering/act2/ppo_textbook_mech.json  m1g / m3 / m1p with readback
  outputs/behavior_steering/act2/{dreamer,storm}_tooladd_grid.json
  outputs/behavior_steering/act2/{dreamer,storm}_logit_readback.json

Outputs: act2/ppo_summary.json, act2/wm_summary.json,
  paper/figures/behavior_steering/act2_ppo_grid.png
  paper/figures/behavior_steering/act2_mechanism.png
  paper/figures/behavior_steering/act2_wm.png

  PYTHONPATH= python scripts/mechinterp/behavior_steering/act2_analyze.py --what ppo|wm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ACT2 = REPO / "outputs/behavior_steering/act2"
FIG = REPO / "paper/figures/behavior_steering"
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}

BEL = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz")
MU_L, MU_R = float(BEL["mu_lakes"]), float(BEL["mu_rocky"])
MID = 0.5 * (MU_L + MU_R)

C_OK, C_WRONG, C_TO = "#059669", "#dc2626", "#9ca3af"


def auc(x, y):
    """rank AUC of score x for binary y."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def load_rows(name):
    return json.loads((ACT2 / name).read_text())


def base_proj(rows):
    return {(r["cat"], r["map_id"]): r["proj"] for r in rows
            if r["cond"] == "baseline"}


def signed(cat, val):
    """orient toward the OTHER class: positive = belief pushed away from the
    map's true category (lakes: toward rocky = +proj shift; rocky: mirrored)."""
    return val if cat == "lakes" else -val


def crossed(cat, proj):
    return proj > MID if cat == "lakes" else proj < MID


# ── PPO ──────────────────────────────────────────────────────────────────

def ppo():
    grid = load_rows("ppo_pg_grid.json")
    mech = load_rows("ppo_textbook_mech.json")
    bp = base_proj(grid)
    steered = ([r for r in grid if r["cond"] != "baseline"] + mech)
    for r in steered:
        b = bp[(r["cat"], r["map_id"])]
        r["d_toward"] = signed(r["cat"], r["proj"] - b)
        r["crossed"] = bool(crossed(r["cat"], r["proj"]))

    # condition table
    conds = {}
    for r in grid + mech:
        conds.setdefault((r["cond"], r["cat"]), []).append(r)
    table = []
    for (cond, cat), sub in sorted(conds.items()):
        n = len(sub)
        cs = [r["cos_mean"] for r in sub if r.get("cos_mean") is not None]
        table.append(dict(
            cond=cond, cat=cat, n=n,
            success=round(np.mean([r["success"] for r in sub]), 3),
            wrong=round(np.mean([r["wrong"] for r in sub]), 3),
            timeout=round(np.mean([r["timeout"] for r in sub]), 3),
            mines=round(float(np.mean([r["mines"] for r in sub])), 1),
            builds=round(float(np.mean([r["builds"] for r in sub])), 1),
            proj=round(float(np.nanmean([r["proj"] for r in sub])), 2),
            signed_cos=(round(signed(cat, float(np.mean(cs))), 3) if cs else None)))

    # mechanism stats on non-timeout steered episodes
    ok = [r for r in steered if not r["timeout"] and np.isfinite(r["proj"])]
    y = np.array([r["wrong"] for r in ok], int)
    x = np.array([r["d_toward"] for r in ok], float)
    cr = np.array([r["crossed"] for r in ok], bool)
    stats = dict(
        n_steered_completed=len(ok),
        auc_dproj_wrong=round(auc(x, y), 3),
        r_pointbiserial=round(float(np.corrcoef(x, y)[0, 1]), 3),
        wrong_given_crossed=round(float(y[cr].mean()), 3) if cr.any() else None,
        wrong_given_not_crossed=round(float(y[~cr].mean()), 3),
        n_crossed=int(cr.sum()),
    )
    out = dict(mu_lakes=MU_L, mu_rocky=MU_R, midpoint=MID,
               conditions=table, mechanism=stats)
    (ACT2 / "ppo_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(stats, indent=1))

    # ---- figure 1: dose-response ----
    doses = sorted({float(r["cond"][3:]) for r in grid if r["cond"].startswith("pg_")})
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
        for ax, cat in zip(axes[:2], ("lakes", "rocky")):
            W = [next(t for t in table if t["cond"] == f"pg_{d:+.2f}"
                      and t["cat"] == cat) for d in doses]
            b = next(t for t in table if t["cond"] == "baseline" and t["cat"] == cat)
            xs = np.arange(len(doses))
            ax.bar(xs, [t["wrong"] for t in W], .62, color=C_WRONG,
                   label="wrong door")
            ax.bar(xs, [t["timeout"] for t in W], .62,
                   bottom=[t["wrong"] for t in W], color=C_TO, label="timeout")
            ax.axhline(b["wrong"], color=C_WRONG, lw=.9, ls=":",
                       label="baseline wrong")
            ax.set_xticks(xs, [f"{d:+.2f}" for d in doses], fontsize=7.6)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("dose $\\eta$")
            tool = "BUILD" if cat == "lakes" else "MINE"
            ax.set_title(f"{cat} maps, target {tool} (n=50/dose)",
                         loc="left", fontsize=9)
            if cat == "lakes":
                ax.set_ylabel("failure rate")
                ax.legend(fontsize=7.4, frameon=False, loc="upper right")
        ax = axes[2]
        for cat, col in (("lakes", "#0e7490"), ("rocky", "#b45309")):
            P = [next(t for t in table if t["cond"] == f"pg_{d:+.2f}"
                      and t["cat"] == cat)["proj"] for d in doses]
            b = next(t for t in table if t["cond"] == "baseline" and t["cat"] == cat)
            ax.plot(doses, P, "o-", color=col, ms=4, label=f"{cat} maps")
            ax.plot([0], [b["proj"]], "s", color=col, ms=5, mec="black", mew=.6)
        for yv, lab in ((MU_L, "$\\mu_{lakes}$"), (MID, "midpoint"),
                        (MU_R, "$\\mu_{rocky}$")):
            ax.axhline(yv, color="#6b7280", lw=.8, ls="--")
            ax.text(1.01, yv, lab, transform=ax.get_yaxis_transform(),
                    fontsize=7.2, va="center", color="#6b7280")
        ax.set_xlabel("dose $\\eta$")
        ax.set_ylabel("belief readback  $h\\cdot\\hat v_{belief}$")
        ax.set_title("late-corridor belief coordinate", loc="left", fontsize=9)
        ax.legend(fontsize=7.4, frameon=False, loc="center left")
        fig.suptitle("Policy-gradient steering on PPO: the failure that appears "
                     "is the wrong DOOR, and it appears when the belief "
                     "coordinate crosses to the other class", y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "act2_ppo_grid.png", bbox_inches="tight")
        print("wrote act2_ppo_grid.png")

    # ---- figure 2: mechanism ----
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.7),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
        ax = axes[0]
        rng = np.random.default_rng(0)
        for i, cat in enumerate(("lakes", "rocky")):
            sub = [r for r in steered if r["cat"] == cat and np.isfinite(r["proj"])]
            for r in sub:
                yj = i + rng.uniform(-.3, .3)
                c = C_TO if r["timeout"] else (C_WRONG if r["wrong"] else C_OK)
                ax.plot(r["proj"], yj, "o", ms=2.6, color=c, alpha=.55, mew=0)
        for xv, lab in ((MU_L, "$\\mu_{lakes}$"), (MID, "mid"),
                        (MU_R, "$\\mu_{rocky}$")):
            ax.axvline(xv, color="#374151", lw=.9, ls="--")
            ax.text(xv, 1.78, lab, fontsize=7.6, ha="center", color="#374151")
        ax.set_yticks([0, 1], ["lakes maps\n(true pole left)",
                               "rocky maps\n(true pole right)"], fontsize=8)
        ax.set_ylim(-.5, 1.9)
        ax.set_xlabel("steered belief readback  $h\\cdot\\hat v_{belief}$")
        ax.set_title("every steered episode, all methods: outcome against the "
                     "final belief coordinate", loc="left", fontsize=9)
        for c, lab in ((C_OK, "correct door"), (C_WRONG, "wrong door"),
                       (C_TO, "timeout")):
            ax.plot([], [], "o", color=c, label=lab, ms=4)
        ax.legend(fontsize=7.4, frameon=False, loc="center right")

        ax = axes[1]
        pts = [t for t in table if t["signed_cos"] is not None]
        mark = {"pg": "o", "m1": "s", "m3": "D"}
        for t in pts:
            fam = "pg" if t["cond"].startswith("pg") else \
                  ("m3" if t["cond"].startswith("m3") else "m1")
            col = "#0e7490" if t["cat"] == "lakes" else "#b45309"
            ax.plot(t["signed_cos"], t["wrong"], mark[fam], color=col, ms=5,
                    alpha=.85, mew=.5, mec="black")
        lab_at = {("m1p_1", "rocky"): ("orthogonalised", (6, -3)),
                  ("m3_sup", "rocky"): ("SAE clamp", (5, 4)),
                  ("m3_sup", "lakes"): ("SAE clamp", (5, -9)),
                  ("m1g_1.5", "rocky"): ("CAA", (5, 4)),
                  ("pg_-0.50", "lakes"): ("PG $\\eta$ $-$0.50", (7, -2)),
                  ("pg_-0.25", "lakes"): ("PG $\\eta$ $-$0.25", (7, -2))}
        for t in pts:
            hit = lab_at.get((t["cond"], t["cat"]))
            if hit:
                ax.annotate(hit[0], (t["signed_cos"], t["wrong"]),
                            xytext=hit[1], textcoords="offset points",
                            fontsize=6.8)
        ax.axvline(0, color="#6b7280", lw=.8, ls=":")
        ax.set_xlabel("mean cosine of applied edits with the belief axis\n"
                      "(oriented toward the other class)")
        ax.set_ylabel("wrong-door rate")
        ax.set_ylim(0, 1.02)
        ax.set_title("the leak law across methods", loc="left", fontsize=9)
        h = [plt.Line2D([], [], marker=m, ls="", color="#374151", label=l,
                        ms=5) for m, l in (("o", "PG doses"), ("s", "CAA axis"),
                                           ("D", "SAE clamp"))]
        ax.legend(handles=h, fontsize=7.2, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(FIG / "act2_mechanism.png", bbox_inches="tight")
        print("wrote act2_mechanism.png")


# ── world models ─────────────────────────────────────────────────────────

def wm():
    out = {}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4),
                                 gridspec_kw={"width_ratios": [1, 1, 1.15]})
        for ax, agent in zip(axes[:2], ("dreamer", "storm")):
            rows = load_rows(f"{agent}_tooladd_grid.json")
            rocky = [r for r in rows if r["cat"] == "rocky"]
            conds = {}
            for r in rocky:
                conds.setdefault(r["cond"], []).append(r)
            doses = sorted({r["lam_sd"] for r in rocky
                            if r["lam_sd"] is not None})
            table = []
            for cond, sub in sorted(conds.items()):
                table.append(dict(
                    agent=agent, cond=cond, n=len(sub),
                    success=round(np.mean([r["success"] for r in sub]), 3),
                    wrong=round(np.mean([r["wrong"] for r in sub]), 3),
                    timeout=round(np.mean([r["timeout"] for r in sub]), 3),
                    mines=round(float(np.mean([r["mines"] for r in sub])), 1),
                    proj=round(float(np.nanmean([r["proj"] for r in sub])), 2)))
            out[agent] = table
            xs = np.arange(len(doses))
            W = [next(t for t in table if t["cond"] == f"tooladd_{d:+.2f}")
                 for d in doses]
            ax.bar(xs, [t["wrong"] for t in W], .62, color=C_WRONG,
                   label="wrong door")
            ax.bar(xs, [t["timeout"] for t in W], .62,
                   bottom=[t["wrong"] for t in W], color=C_TO, label="timeout")
            b = next(t for t in table if t["cond"] == "baseline")
            ax.axhline(b["wrong"], color=C_WRONG, lw=.9, ls=":",
                       label="baseline wrong")
            if agent == "storm":     # the dreamer panel's bars reach 1.0
                ax.legend(fontsize=7.2, frameon=False, loc="upper right")
            ax.set_xticks(xs, [f"{d:+g}" for d in doses], fontsize=7.6)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("dose (multiples of the axis sd)")
            ax.set_title(f"{agent}: tool-axis displacement, rocky maps",
                         loc="left", fontsize=9)
            if agent == "dreamer":
                ax.set_ylabel("failure rate")
        # readback panel: baseline vs flip dose vs logit contrast
        ax = axes[2]
        labels, vals, cols = [], [], []
        spec = dict(dreamer=("tooladd_-0.25",), storm=("tooladd_-4.00",))
        for agent in ("dreamer", "storm"):
            zb = np.load(REPO / f"outputs/belief_report/steer_axis_{agent}.npz")
            sfx = "_wall" if agent == "storm" else ""
            mu_l = float(zb["mu_lakes" + sfx]); mu_r = float(zb["mu_rocky" + sfx])
            rows = load_rows(f"{agent}_tooladd_grid.json")
            rocky = [r for r in rows if r["cat"] == "rocky"]
            def mp(cond, rr=rocky):
                v = [r["proj"] for r in rr if r["cond"] == cond
                     and np.isfinite(r["proj"])]
                return np.mean(v) if v else np.nan
            norm = lambda p: (p - mu_l) / (mu_r - mu_l)      # noqa: E731
            entries = [("baseline", mp("baseline"), "#374151")]
            entries += [(c.replace("tooladd_", "axis "), mp(c), C_WRONG)
                        for c in spec[agent]]
            lg = ACT2 / f"{agent}_logit_readback.json"
            if lg.exists():
                lr = json.loads(lg.read_text())
                v = [r["proj"] for r in lr if np.isfinite(r["proj"])]
                entries.append(("logit +3", np.mean(v), "#2563eb"))
            for nm, val, c in entries:
                labels.append(f"{agent}\n{nm}")
                vals.append(norm(val)); cols.append(c)
        xs = np.arange(len(labels))
        ax.bar(xs, vals, .6, color=cols)
        ax.axhline(1, color="#b45309", lw=.9, ls="--")
        ax.axhline(0, color="#0e7490", lw=.9, ls="--")
        ax.axhline(.5, color="#6b7280", lw=.7, ls=":")
        ax.text(1.01, 1, "rocky pole", transform=ax.get_yaxis_transform(),
                fontsize=7, va="center", color="#b45309")
        ax.text(1.01, 0, "lakes pole", transform=ax.get_yaxis_transform(),
                fontsize=7, va="center", color="#0e7490")
        ax.set_xticks(xs, labels, fontsize=7)
        ax.set_ylabel("readback, class-normalised")
        ax.set_title("belief readback on rocky maps", loc="left", fontsize=9)
        fig.suptitle("World models under the same textbook edits: the flip dose "
                     "and where the belief coordinate lands", y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "act2_wm.png", bbox_inches="tight")
        print("wrote act2_wm.png")
    (ACT2 / "wm_summary.json").write_text(json.dumps(out, indent=1))
    for agent, table in out.items():
        for t in table:
            print(f"{agent:8s} {t['cond']:16s} n={t['n']:2d} succ={t['success']:.2f} "
                  f"WRONG={t['wrong']:.2f} TO={t['timeout']:.2f} "
                  f"mines={t['mines']:6.1f} proj={t['proj']:+7.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["ppo", "wm"])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    ppo() if a.what == "ppo" else wm()
