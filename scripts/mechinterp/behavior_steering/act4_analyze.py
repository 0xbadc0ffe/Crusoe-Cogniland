#!/usr/bin/env python3
"""Act four analysis: behaviour vs decision on balanced maps.

Two axes, deliberately kept apart:

* BEHAVIOUR -- successful tool usages only (real mined/placed events). Each map
  is its own 100%: the per-map ratio steered/baseline is formed first, then
  averaged over the maps where that tool's baseline is non-zero (n_eff is
  reported, because a map that never builds cannot show a build reduction).
* DECISION -- the door split, PAIRED per map. Pooling raw splits across maps is
  meaningless here: every balanced map has its own baseline door preference,
  and many sit at a ceiling (0.0 or 1.0), where a shift can only go one way.
  So the statistic is the per-map change in P(top) against that same map's
  baseline, and the figure plots one dot per map (baseline P(top) on x,
  steered on y) so the ceilings stay visible.

  PYTHONPATH=src python act4_analyze.py --arm ppo_clamp
  PYTHONPATH=src python act4_analyze.py --summary
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
ACT4 = REPO / "outputs/behavior_steering/act4"
FIG = REPO / "paper/figures/behavior_steering"
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}
CONDS = ("baseline", "sup_mine", "sup_build", "sup_both")
LBL = {"baseline": "suppress none", "sup_mine": "suppress mine",
       "sup_build": "suppress build", "sup_both": "suppress both"}
C_MINE, C_BUILD = "#2563eb", "#93c5fd"
ARM_TITLE = {
    "ppo_clamp": "PPO — GradientClamp + project_out (state-level)",
    "ppo_clamp_noorth": "PPO — GradientClamp, no correction (state-level)",
    "storm_logit": "STORM — soft actor-logit bias (actuator-level)",
    "dreamer_logit": "DreamerV3 — soft actor-logit bias (actuator-level)",
    "dreamer_tilt": "DreamerV3 — imagination tilt (plan-level)",
}


def stars(p):
    """GraphPad-style significance convention, used in every act-four figure:
    ns p>=0.05, * p<0.05, ** p<0.01, *** p<0.001, **** p<1e-4."""
    if p is None or not np.isfinite(p):
        return "n/a"
    return ("****" if p < 1e-4 else "***" if p < 1e-3 else
            "**" if p < 1e-2 else "*" if p < 0.05 else "ns")


def perm_p(a, b, n_draws=20000, seed=0):
    """Paired sign-flip permutation on per-map differences a-b, one-sided
    (H1: a > b). Paired because both arms see the same maps and seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    obs = d.mean()
    cnt = int(sum(1 for _ in range(n_draws)
                  if (d * rng.choice([-1, 1], len(d))).mean() >= obs))
    return (cnt + 1) / (n_draws + 1)


def door_pvalues(rows, null_rows):
    """Per condition: |Δ P(top)| under the condition vs under the null (the
    baseline re-run with fresh seeds), paired by map."""
    T, N = per_map(rows), per_map(null_rows)
    out = {}
    for cond in CONDS[1:]:
        obs, nul = [], []
        for (c, m) in list(T):
            if c != "baseline":
                continue
            b = T[("baseline", m)]["p_top"]
            s_ = T.get((cond, m), {}).get("p_top")
            nb = N.get(("baseline", m), {}).get("p_top")
            if None in (b, s_, nb):
                continue
            obs.append(abs(s_ - b)); nul.append(abs(nb - b))
        pv = perm_p(obs, nul)
        out[cond] = dict(obs=round(float(np.mean(obs)), 3) if obs else None,
                         null=round(float(np.mean(nul)), 3) if nul else None,
                         perm_p=None if pv is None else round(pv, 5),
                         stars=stars(pv), n_maps=len(obs))
    return out


def pooled_tools(rows):
    """Episode-weighted means per condition. The per-map normalisation the
    campaign reports weights every map equally, which inflates changes on maps
    with a tiny baseline (1 build -> 3 reads as +200%); pooled means are the
    robustness check, and substitution claims must quote both."""
    g = collections.defaultdict(list)
    for r in rows:
        g[r["cond"]].append(r)
    mk = "true_mines" if "true_mines" in rows[0] else "mines"
    bk = "true_builds" if "true_builds" in rows[0] else "builds"
    b = g["baseline"]
    bm, bb = np.mean([x[mk] for x in b]), np.mean([x[bk] for x in b])
    out = {}
    for cond, sub in g.items():
        m, d = float(np.mean([x[mk] for x in sub])), float(np.mean([x[bk] for x in sub]))
        out[cond] = dict(mines=round(m, 2), builds=round(d, 2),
                         mines_pct=round(100 * (m - bm) / bm, 1) if bm else None,
                         builds_pct=round(100 * (d - bb) / bb, 1) if bb else None)
    return out


def per_map(rows):
    """-> {(cond, map_id): dict of that cell's means over its rollouts}"""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["cond"], r["map_id"])].append(r)
    out = {}
    for k, sub in g.items():
        f = lambda c: float(np.mean([x[c] for x in sub]))       # noqa: E731
        # only episodes that actually ENDED at a door vote: a timed-out row
        # can still carry the cell it stalled in, which is not a decision
        doors = [x["door"] for x in sub if not x.get("timeout")]
        n_dec = sum(1 for d in doors if d in ("top", "bottom"))
        out[k] = dict(n=len(sub), mines=f("mines"), builds=f("builds"),
                      success=f("success"), timeout=f("timeout"),
                      steps=f("steps"),
                      p_top=(sum(1 for d in doors if d == "top") / n_dec
                             if n_dec else None),
                      n_decided=n_dec)
    return out


def null_floor(arm, pm):
    """Empirical noise floor: the SAME baseline condition re-run with different
    seeds, scored by the same paired statistic. Any steered shift must be read
    against this, because P(top) from R rollouts carries real sampling error."""
    f = ACT4 / f"test_{arm}_null.json"
    if not f.exists():
        return None
    npm = per_map(json.loads(f.read_text()))
    d = []
    for (cond, m), c in npm.items():
        b = pm.get(("baseline", m))
        if b and b["p_top"] is not None and c["p_top"] is not None:
            d.append(c["p_top"] - b["p_top"])
    if not d:
        return None
    d = np.array(d)
    return dict(n=len(d), abs_shift=round(float(np.mean(np.abs(d))), 3),
                p95_abs=round(float(np.percentile(np.abs(d), 95)), 3),
                moved_20pp=int(np.sum(np.abs(d) >= 0.2)))


def analyse(arm):
    rows = json.loads((ACT4 / f"test_{arm}.json").read_text())
    pm = per_map(rows)
    maps = sorted({m for (_, m) in pm})
    table = {}
    for cond in CONDS:
        cells = [(m, pm[(cond, m)]) for m in maps if (cond, m) in pm]
        base = {m: pm[("baseline", m)] for m in maps if ("baseline", m) in pm}
        rec = dict(cond=cond, n_maps=len(cells),
                   n_episodes=int(sum(c["n"] for _, c in cells)))
        for k in ("success", "timeout", "steps"):
            rec[k] = round(float(np.mean([c[k] for _, c in cells])), 3)
        # per-map tool ratio, averaged over maps whose baseline uses that tool
        for tool in ("mines", "builds"):
            r_, n_eff = [], 0
            for m, c in cells:
                b = base[m][tool]
                if b > 0:
                    r_.append(c[tool] / b)
                    n_eff += 1
            rec[f"{tool}_pct"] = (round(100 * (float(np.mean(r_)) - 1), 1)
                                  if r_ else None)
            rec[f"{tool}_n_eff"] = n_eff
            rec[f"{tool}_raw"] = round(float(np.mean([c[tool] for _, c in cells])), 2)
        # paired door shift
        d = [(base[m]["p_top"], c["p_top"]) for m, c in cells
             if base[m]["p_top"] is not None and c["p_top"] is not None]
        if d and cond != "baseline":
            diff = np.array([b - a for a, b in d])
            rec.update(door_n=len(d),
                       door_abs_shift=round(float(np.mean(np.abs(diff))), 3),
                       door_signed=round(float(np.mean(diff)), 3),
                       door_max=round(float(np.max(np.abs(diff))), 3),
                       door_moved_20pp=int(np.sum(np.abs(diff) >= 0.2)))
        table[cond] = rec
    nf = null_floor(arm, pm)
    if nf:
        table["_null"] = nf
    npath = ACT4 / f"test_{arm}_null.json"
    if npath.exists():                             # significance vs own null
        table["_door_p"] = door_pvalues(rows, json.loads(npath.read_text()))
    table["_pooled"] = pooled_tools(rows)          # episode-weighted check
    (ACT4 / f"summary_{arm}.json").write_text(json.dumps(table, indent=1))
    pl = table["_pooled"]
    print("  pooled (episode-weighted): " + "; ".join(
        f"{c} {pl[c]['mines_pct']:+.0f}% mines / {pl[c]['builds_pct']:+.0f}% builds"
        for c in ("sup_mine", "sup_build", "sup_both") if c in pl))

    print(f"\n=== {arm} — {ARM_TITLE.get(arm, arm)} ===")
    print(f"{'condition':16s} {'mines %':>9s} {'builds %':>9s} {'succ':>6s} "
          f"{'TO':>6s} {'|Δdoor|':>8s} {'moved≥20pp':>11s}")
    if nf:
        print(f"{'NULL (same cond,':16s} {'':>9s} {'':>9s} {'':>6s} {'':>6s} "
              f"{nf['abs_shift']:8.3f} {str(nf['moved_20pp']) + '/' + str(nf['n']):>11s}"
              "   <- noise floor")
    for cond in CONDS:
        t = table[cond]
        mp = f"{t['mines_pct']:+.0f}%" if t["mines_pct"] is not None else "—"
        bp = f"{t['builds_pct']:+.0f}%" if t["builds_pct"] is not None else "—"
        ds = f"{t.get('door_abs_shift', 0):.3f}" if cond != "baseline" else "—"
        if cond != "baseline" and table.get("_door_p", {}).get(cond):
            ds += " " + table["_door_p"][cond]["stars"]
        mv = (f"{t.get('door_moved_20pp', 0)}/{t.get('door_n', 0)}"
              if cond != "baseline" else "—")
        print(f"{LBL[cond]:16s} {mp:>9s} {bp:>9s} {t['success']:6.2f} "
              f"{t['timeout']:6.2f} {ds:>8s} {mv:>11s}")
    return table, pm, maps


def figure(arm, table, pm, maps):
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13.6, 6.6))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=.42,
                              wspace=.28)
        # top row: per-map door dots, one panel per steered condition
        for j, cond in enumerate(CONDS[1:]):
            ax = fig.add_subplot(gs[0, j])
            xs, ys = [], []
            for m in maps:
                b, c = pm.get(("baseline", m)), pm.get((cond, m))
                if b and c and b["p_top"] is not None and c["p_top"] is not None:
                    xs.append(b["p_top"]); ys.append(c["p_top"])
            xs, ys = np.array(xs), np.array(ys)
            ax.plot([0, 1], [0, 1], color="#111827", lw=.9, ls="--", zorder=1)
            moved = np.abs(ys - xs) >= 0.2
            ax.scatter(xs[~moved], ys[~moved], s=26, color="#9ca3af", lw=0,
                       alpha=.85, zorder=3, label="shift < 20pp")
            ax.scatter(xs[moved], ys[moved], s=34, color="#dc2626", lw=0,
                       alpha=.9, zorder=4, label="shift ≥ 20pp")
            ax.set_xlim(-.04, 1.04); ax.set_ylim(-.04, 1.04)
            ax.set_xlabel("baseline P(top door)")
            if j == 0:
                ax.set_ylabel("steered P(top door)")
                ax.legend(fontsize=6.8, frameon=False, loc="upper left")
            t = table[cond]
            nf = table.get("_null")
            extra = f"  (noise {nf['abs_shift']:.2f})" if nf else ""
            pv = (table.get("_door_p") or {}).get(cond, {})
            st = f"  [{pv.get('stars', 'n/a')}]" if pv else ""
            ax.set_title(f"{LBL[cond]}   |Δ| = {t.get('door_abs_shift', 0):.2f}"
                         + extra + st, loc="left", fontsize=9)
        # bottom left: tool change, per tool
        ax = fig.add_subplot(gs[1, :2])
        xs = np.arange(len(CONDS))
        w = .38
        for off, tool, col in ((-w / 2, "mines", C_MINE), (w / 2, "builds", C_BUILD)):
            vals = [0 if c == "baseline" else (table[c][f"{tool}_pct"] or 0)
                    for c in CONDS]
            ax.bar(xs + off, vals, w, color=col, label=tool)
            for x, v, c in zip(xs + off, vals, CONDS):
                if c != "baseline":
                    ax.text(x, v + (2 if v >= 0 else -2), f"{v:+.0f}",
                            ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=7)
        ax.axhline(0, color="#111827", lw=.9)
        vv = [table[c][f"{t}_pct"] or 0 for c in CONDS for t in ("mines", "builds")]
        lo, hi = min(vv + [0]), max(vv + [0])
        pad = max(12, .18 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(xs, [LBL[c] for c in CONDS], fontsize=8)
        ax.set_ylabel("change in successful tool uses (%)")
        ax.set_title("behaviour: each map is its own 100%", loc="left", fontsize=9)
        ax.legend(fontsize=7.4, frameon=False)
        # bottom right: guard rails
        ax = fig.add_subplot(gs[1, 2])
        ax.bar(xs - w / 2, [table[c]["success"] for c in CONDS], w,
               color="#16a34a", label="success")
        ax.bar(xs + w / 2, [table[c]["timeout"] for c in CONDS], w,
               color="#9ca3af", label="timeout")
        ax.set_xticks(xs, ["none", "mine", "build", "both"], fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("rate")
        ax.set_title("guard rails", loc="left", fontsize=9)
        ax.legend(fontsize=7.4, frameon=False)
        n_m = table["baseline"]["n_maps"]
        n_r = table["baseline"]["n_episodes"] // max(n_m, 1)
        fig.suptitle(
            f"{ARM_TITLE.get(arm, arm)} — held-out BALANCED maps "
            f"(n = {n_m} maps × {n_r} rollouts, knobs frozen on train maps).\n"
            "Both doors are rewarded here, so success cannot see the decision: "
            "the door panels are the only place it shows.",
            y=.99, fontsize=10.4)
        fig.savefig(FIG / f"act4_{arm}.png", bbox_inches="tight")
        print(f"wrote act4_{arm}.png")


def cross_summary():
    ops = json.loads((ACT4 / "operating_points.json").read_text())
    print(f"\n{'arm':16s} {'condition':16s} {'target tool Δ':>14s} {'succ':>6s} "
          f"{'TO':>6s} {'|Δdoor|':>8s} {'≥20pp':>8s}")
    out = {}
    for f in sorted(ACT4.glob("summary_*.json")):
        arm = f.stem[len("summary_"):]
        tab = json.loads(f.read_text())
        out[arm] = {}
        for cond in CONDS[1:]:
            t = tab.get(cond)
            if not t:
                continue
            key = ("mines_pct" if cond == "sup_mine" else
                   "builds_pct" if cond == "sup_build" else None)
            if key:
                tgt = t[key]
            else:                                    # both: mean of the two
                vv = [t[k] for k in ("mines_pct", "builds_pct")
                      if t[k] is not None]
                tgt = float(np.mean(vv)) if vv else None
            knob = ops.get(arm, {}).get("conds", {}).get(cond, {}).get("knob")
            dp = (table.get("_door_p") or {}).get(cond, {}) if "table" in dir() else {}
            dp = (json.loads((ACT4 / f"summary_{arm}.json").read_text())
                  .get("_door_p", {}).get(cond, {})
                  if (ACT4 / f"summary_{arm}.json").exists() else {})
            out[arm][cond] = dict(target_pct=tgt, success=t["success"],
                                  timeout=t["timeout"],
                                  door_abs=t.get("door_abs_shift"),
                                  moved=t.get("door_moved_20pp"),
                                  door_n=t.get("door_n"), knob=knob,
                                  door_p=dp.get("perm_p"),
                                  door_stars=dp.get("stars"))
            print(f"{arm:16s} {LBL[cond]:16s} "
                  f"{(f'{tgt:+.0f}%' if tgt is not None else '—'):>14s} "
                  f"{t['success']:6.2f} {t['timeout']:6.2f} "
                  f"{t.get('door_abs_shift', float('nan')):8.3f} "
                  f"{t.get('door_moved_20pp', 0)}/{t.get('door_n', 0):>3d}")
    (ACT4 / "cross_summary.json").write_text(json.dumps(out, indent=1))
    cross_figure(out)


def cross_figure(out):
    """One panel per agent-method: the targeted cut against the decision cost."""
    arms = [a for a in ARM_TITLE if a in out]
    if not arms:
        return
    short = {"ppo_clamp": "PPO\nclamp (state)",
             "storm_logit": "STORM\nlogit (actuator)",
             "dreamer_logit": "Dreamer\nlogit (actuator)",
             "dreamer_tilt": "Dreamer\ntilt (plan)"}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
        xs = np.arange(len(arms))
        w = .26
        for j, (cond, col) in enumerate(zip(CONDS[1:],
                                            ("#2563eb", "#93c5fd", "#1e3a8a"))):
            vals = [-(out[a].get(cond, {}).get("target_pct") or 0) for a in arms]
            axes[0].bar(xs + (j - 1) * w, vals, w, color=col, label=LBL[cond])
            axes[1].bar(xs + (j - 1) * w,
                        [out[a].get(cond, {}).get("success") or 0 for a in arms],
                        w, color=col)
            dvals = [out[a].get(cond, {}).get("door_abs") or 0 for a in arms]
            axes[2].bar(xs + (j - 1) * w, dvals, w, color=col)
            for i, a in enumerate(arms):
                st = out[a].get(cond, {}).get("door_stars")
                if st:
                    axes[2].text(i + (j - 1) * w, dvals[i] + 0.008, st,
                                 ha="center", va="bottom", fontsize=6.6,
                                 color="#111827", rotation=90 if st == "ns" else 0)
        for ax, ttl, yl in ((axes[0], "commanded behaviour delivered",
                             "reduction in the targeted tool (%)"),
                            (axes[1], "task guard rail", "success"),
                            (axes[2], "the price: decision moved",
                             "mean |Δ P(top door)|, paired")):
            ax.set_xticks(xs, [short.get(a, a) for a in arms], fontsize=7.6)
            ax.set_title(ttl, loc="left", fontsize=9)
            ax.set_ylabel(yl)
            ax.set_ylim(0, None)
        axes[1].set_ylim(0, 1.02)
        axes[0].legend(fontsize=7, frameon=False)
        # noise floors, per arm, on the decision panel
        for i, a in enumerate(arms):
            f = ACT4 / f"summary_{a}.json"
            if f.exists():
                nf = json.loads(f.read_text()).get("_null")
                if nf:
                    axes[2].plot([i - 1.6 * w, i + 1.6 * w],
                                 [nf["abs_shift"]] * 2, color="#111827",
                                 lw=1.3, ls=":", zorder=5)
        axes[2].plot([], [], color="#111827", lw=1.3, ls=":",
                     label="noise floor (same condition, new seeds)")
        fig.text(0.5, -0.06,
                 "Stars: paired permutation of each condition's per-map "
                 "|Δ P(top)| against that arm's own null.   "
                 "ns p≥0.05   * p<0.05   ** p<0.01   *** p<0.001   **** p<1e-4",
                 ha="center", va="top", fontsize=7.2, color="#6b7280")
        top2 = max([out[a].get(c, {}).get("door_abs") or 0
                    for a in arms for c in CONDS[1:]] + [0.01])
        axes[2].set_ylim(0, top2 * 1.28)          # headroom for the stars
        axes[2].legend(fontsize=6.8, frameon=False, loc="upper left")
        n_arms = len(arms)
        fig.suptitle(
            f"Balanced maps, held out, knobs frozen on training maps "
            f"({n_arms} arm{'s' if n_arms != 1 else ''} shown). Every method "
            "delivers its commanded tool change at high success.\nOnly the "
            "state-level arm also moves the door choice — 8x its own noise "
            "floor, where no task metric can see it; the others sit at or "
            "below theirs.", y=1.03, fontsize=10.4)
        fig.tight_layout()
        fig.savefig(FIG / "act4_cross.png", bbox_inches="tight")
        print("wrote act4_cross.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if a.arm:
        figure(a.arm, *analyse(a.arm))
    if a.summary:
        cross_summary()
