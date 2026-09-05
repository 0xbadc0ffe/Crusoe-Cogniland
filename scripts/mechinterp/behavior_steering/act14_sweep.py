#!/usr/bin/env python3
"""Act fourteen -- sweep the clamp strength; find the best success x tool-removal.

The act-11 strong re-run showed the two ends of the PPO frontier: the frozen
act-5 operating points barely move the routes, a never-released clamp at
theta 1e-4 removes the tool on every map but times out most episodes. This
sweeps the space between them:

    theta   in THETAS   (P(suppressed tool) is pushed below theta, every step)
    window  in WINDOWS  (0 = never released; else released after `window`
                         stuck steps, re-armed on progress -- act 5's gate)
    arm     in {suppress bridge, suppress tunnel}

on balanced maps from the TRAIN pool (screened as act 11 screens the test
pool: the unsteered agent uses both tools), so the held-out set is not spent
on selection. `analyze` scores every setting, draws the trade-off, and picks a
winner per arm; `verify` re-runs the winners on the 36 held-out act-11 maps
with ten rollouts and draws the route grid.

Score (the user's "success rate for tool change ratio"):
    removed = 1 - mean(steered tool count) / mean(unsteered tool count)  [pooled]
    score   = success * removed
plus two guarded picks: the largest `removed` with success >= 0.9 / >= 0.8.

  python act14_sweep.py --stage sweep   --maps 36 --rolls 6 --workers 44
  python act14_sweep.py --stage analyze
  python act14_sweep.py --stage verify  --rolls 10 --workers 44 [--rule score]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/mechinterp", "scripts/mechinterp/belief_report",
          "scripts/figures", "scripts/mechinterp/behavior_steering"):
    sys.path.insert(0, str(REPO / p))

OUT = REPO / "outputs/behavior_steering/act14"
FIG = REPO / "paper/figures/behavior_steering"
TRAIN_PKL = REPO / "data/bridge_tunnel/forkwall6k/train.pkl"
TEST_PKL = REPO / "data/bridge_tunnel/forkwall6k/test.pkl"
ACT11_SUMMARY = REPO / "outputs/behavior_steering/act11/summary.json"
EPS = 1e-3
THETAS = [0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]
WINDOWS = [0, 5, 10, 20, 40]            # 0 = never released
# (arm label, act4 condition, suppressed tool field, the other tool's field)
ARMS = [("suppress bridge", "sup_build", "builds", "mines"),
        ("suppress tunnel", "sup_mine", "mines", "builds")]
SEED_SWEEP, SEED_VERIFY = 6000, 4000    # 4000 = act 11's seeds, so unsteered matches


def job(mid, seed, cond, thr, win, split, iters, alpha):
    gated = win > 0
    return (mid, seed, cond, float(thr), int(win) if gated else 15, EPS,
            gated, split, int(iters), float(alpha))


def win_label(w):
    return "never released" if w == 0 else f"released after {w} stuck steps"


def screen_train(pool, n_maps, rolls, workers, iters, alpha, n_screen=130):
    """Balanced TRAIN maps whose unsteered agent uses both tools (act 11's rule)."""
    import act5_ppo as A5
    cands = [i for i, r in enumerate(pool) if r.category == "balanced"]
    rng = np.random.default_rng(1)
    cands = [int(cands[i]) for i in rng.permutation(len(cands))[:n_screen]]
    sc = A5.run([job(m, SEED_SWEEP + k, "baseline", 0.0, 0, "train", iters, alpha)
                 for m in cands for k in range(2)], workers)
    score = {}
    for mid in cands:
        rs = [r for r in sc if r["map_id"] == mid]
        score[mid] = min(np.mean([r["mines"] for r in rs]),
                         np.mean([r["builds"] for r in rs]))
    users = [m for m in sorted(score, key=lambda m: -score[m]) if score[m] > 0]
    mids = sorted(int(users[i]) for i in
                  np.random.default_rng(1).permutation(len(users))[:n_maps])
    print(f"screened {len(cands)} balanced TRAIN maps: {len(users)} use both "
          f"tools, kept {len(mids)}", flush=True)
    return mids


def stage_sweep(a, sfx):
    import act5_ppo as A5
    pool = pickle.load(open(TRAIN_PKL, "rb"))
    thetas = [float(x) for x in a.thetas.split(",")]
    windows = [int(x) for x in a.windows.split(",")]
    mids = screen_train(pool, a.maps, a.rolls, a.workers, a.iters, a.alpha)
    t0 = time.time()
    rows = []
    got = A5.run([job(m, SEED_SWEEP + k, "baseline", 0.0, 0, "train", a.iters, a.alpha)
                  for m in mids for k in range(a.rolls)], a.workers)
    for r in got:
        r.update(arm="unsteered", theta=0.0, window=-1)
    rows += A5.strip(got)
    print(f"  unsteered {len(got)} episodes ({time.time()-t0:.0f}s)", flush=True)
    n_set = len(thetas) * len(windows) * len(ARMS)
    k = 0
    for tag, cond, _, _ in ARMS:
        for w in windows:
            for th in thetas:
                k += 1
                got = A5.run([job(m, SEED_SWEEP + i, cond, th, w, "train", a.iters, a.alpha)
                              for m in mids for i in range(a.rolls)], a.workers)
                for r in got:
                    r.update(arm=tag, theta=th, window=w)
                rows += A5.strip(got)
                succ = np.mean([r["success"] for r in got])
                print(f"  [{k:2d}/{n_set}] {tag:16s} theta={th:<7g} {win_label(w):32s}"
                      f" succ {succ:.2f} ({time.time()-t0:.0f}s)", flush=True)
                (OUT / f"rows_sweep{sfx}.json").write_text(json.dumps(rows))
    (OUT / f"rows_sweep{sfx}.json").write_text(json.dumps(rows))
    (OUT / f"sweep_meta{sfx}.json").write_text(json.dumps(dict(
        maps=mids, rolls=a.rolls, thetas=thetas, windows=windows, iters=a.iters,
        alpha=a.alpha, seed0=SEED_SWEEP, split="train"), indent=1))
    print(f"wrote rows_sweep{sfx}.json ({len(rows)} episodes)")


# ---------------------------------------------------------------- analysis --
def setting_stats(base, rs, tool, other, mids):
    b_tool = np.mean([r[tool] for r in base]); s_tool = np.mean([r[tool] for r in rs])
    b_oth = np.mean([r[other] for r in base]); s_oth = np.mean([r[other] for r in rs])
    pct = []
    for mid in mids:
        vb = np.mean([r[tool] for r in base if r["map_id"] == mid])
        vs = np.mean([r[tool] for r in rs if r["map_id"] == mid])
        if vb > 0:
            pct.append(100 * (vs - vb) / vb)
    done = [r for r in rs if not r["timeout"]]
    succ = float(np.mean([r["success"] for r in rs]))
    removed = float(1 - s_tool / max(b_tool, 1e-9))
    return dict(n=len(rs), success=succ,
                timeout=float(np.mean([r["timeout"] for r in rs])),
                tool=float(s_tool), tool_base=float(b_tool), removed=removed,
                median_pct=float(np.median(pct)) if pct else float("nan"),
                fell=int(sum(1 for x in pct if x < 0)), n_eff=len(pct),
                other_ratio=float(s_oth / max(b_oth, 1e-9)),
                p_top=float(np.mean([r["door"] == "top" for r in rs])),
                p_top_completed=(float(np.mean([r["door"] == "top" for r in done]))
                                 if done else float("nan")),
                dbelief=float(np.nanmean([r["proj"] for r in rs])
                              - np.nanmean([r["proj"] for r in base])),
                p_tool_max=float(np.nanmax([r.get("p_tool_max", np.nan) for r in rs])),
                score=succ * removed)


def pareto(points):
    """indices of non-dominated (removed, success) points, larger is better."""
    keep = []
    for i, (x, y) in enumerate(points):
        dom = any((x2 >= x and y2 >= y) and (x2 > x or y2 > y)
                  for j, (x2, y2) in enumerate(points) if j != i)
        if not dom:
            keep.append(i)
    return keep


def pick(settings, rule):
    if rule == "score":
        return max(settings, key=lambda s: s["score"])
    guard = {"guard90": 0.9, "guard80": 0.8}[rule]
    ok = [s for s in settings if s["success"] >= guard]
    return max(ok, key=lambda s: s["removed"]) if ok else None


def stage_analyze(a, sfx):
    rows = json.loads((OUT / f"rows_sweep{sfx}.json").read_text())
    meta = json.loads((OUT / f"sweep_meta{sfx}.json").read_text())
    mids = meta["maps"]
    base = [r for r in rows if r["arm"] == "unsteered"]
    table = {}
    for tag, cond, tool, other in ARMS:
        sets = []
        for w in meta["windows"]:
            for th in meta["thetas"]:
                rs = [r for r in rows if r["arm"] == tag and r["theta"] == th
                      and r["window"] == w]
                if not rs:
                    continue
                s = setting_stats(base, rs, tool, other, mids)
                s.update(theta=th, window=w)
                sets.append(s)
        idx = pareto([(s["removed"], s["success"]) for s in sets])
        for i in idx:
            sets[i]["pareto"] = True
        best = {rule: pick(sets, rule) for rule in ("score", "guard90", "guard80")}
        table[tag] = dict(settings=sets, best=best)

    # ---- print + markdown --------------------------------------------------
    md = [f"# Act 14: clamp-strength sweep on {len(mids)} TRAIN balanced maps, "
          f"{meta['rolls']} rollouts per setting\n",
          "removed = 1 - steered/unsteered tool count (pooled); score = success x removed; "
          "P(top|done) = share of COMPLETED episodes exiting top.\n"]
    for tag, d in table.items():
        b0 = d["settings"][0]["tool_base"]
        hdr = (f"\n## {tag}  (unsteered {b0:.2f} per episode)\n\n"
               "| theta | gate | success | timeout | removed | median/map | other tool x | P(top) | P(top\\|done) | dbelief | score | Pareto |\n"
               "|---|---|---|---|---|---|---|---|---|---|---|---|")
        print(hdr); md.append(hdr)
        for s in d["settings"]:
            line = (f"| {s['theta']:g} | {win_label(s['window'])} | {s['success']:.2f} | "
                    f"{s['timeout']:.2f} | {100*s['removed']:.0f}% | {s['median_pct']:+.0f}% "
                    f"({s['fell']}/{s['n_eff']}) | {s['other_ratio']:.2f} | {s['p_top']:.2f} | "
                    f"{s['p_top_completed']:.2f} | {s['dbelief']:+.2f} | {s['score']:.3f} | "
                    f"{'*' if s.get('pareto') else ''} |")
            print(line); md.append(line)
        md.append("")
        for rule, s in d["best"].items():
            txt = ("none qualifies" if s is None else
                   f"theta {s['theta']:g}, {win_label(s['window'])}: success {s['success']:.2f}, "
                   f"removed {100*s['removed']:.0f}%, score {s['score']:.3f}")
            line = f"- **best by {rule}**: {txt}"
            print(line); md.append(line)
    (OUT / f"SWEEP{sfx}.md").write_text("\n".join(md) + "\n")
    (OUT / f"summary{sfx}.json").write_text(json.dumps(
        dict(meta=meta, table=table), indent=1))
    draw_tradeoff(table, meta, FIG / f"act14_sweep{sfx}.png")


def draw_tradeoff(table, meta, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTE = "#0b0b0b", "#52514e", "#a8a7a1"
    # gate window is ORDINAL (5 < 10 < 20 < 40 < never): one hue, light -> dark
    RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    wins = sorted(meta["windows"], key=lambda w: (w == 0, w))   # 0 (never) last
    col = {w: RAMP[min(i, len(RAMP) - 1)] for i, w in enumerate(wins)}
    RC = {"figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
          "axes.spines.top": False, "axes.spines.right": False,
          "axes.edgecolor": MUTE, "xtick.color": INK2, "ytick.color": INK2,
          "axes.labelcolor": INK2, "figure.facecolor": "#fcfcfb",
          "axes.facecolor": "#fcfcfb"}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
        for ax, (tag, d) in zip(axes, table.items()):
            sets = d["settings"]
            for w in wins:
                ss = [s for s in sets if s["window"] == w]
                ss.sort(key=lambda s: -s["theta"])            # weak -> strong
                x = [100 * s["removed"] for s in ss]; y = [100 * s["success"] for s in ss]
                ax.plot(x, y, "-", color=col[w], lw=2, alpha=.9, zorder=3,
                        label=win_label(w))
                ax.plot(x, y, "o", color=col[w], ms=5, mec="#fcfcfb", mew=1.2, zorder=4)
            par = sorted([s for s in sets if s.get("pareto")], key=lambda s: s["removed"])
            ax.plot([100 * s["removed"] for s in par], [100 * s["success"] for s in par],
                    "--", color=MUTE, lw=1, zorder=2, label="Pareto frontier")
            b = d["best"]["score"]
            ax.plot(100 * b["removed"], 100 * b["success"], "o", ms=13, mfc="none",
                    mec="#e34948", mew=2, zorder=5)
            ax.annotate(f"best score {b['score']:.2f}\n$\\theta$={b['theta']:g}, "
                        f"{win_label(b['window'])}",
                        (100 * b["removed"], 100 * b["success"]),
                        xytext=(0.04, 0.40), textcoords="axes fraction", ha="left",
                        va="top", fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=.8,
                                        shrinkB=8))
            g = d["best"]["guard90"]
            if g is not None and g is not b:
                ax.plot(100 * g["removed"], 100 * g["success"], "s", ms=11, mfc="none",
                        mec="#eb6834", mew=1.6, zorder=5)
                ax.annotate(f"best at success$\\geq$90%\n$\\theta$={g['theta']:g}, "
                            f"{win_label(g['window'])}",
                            (100 * g["removed"], 100 * g["success"]),
                            xytext=(0.04, 0.58), textcoords="axes fraction", ha="left",
                            va="top", fontsize=8, color=INK,
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=.8,
                                            shrinkB=8))
            ax.axhline(90, color=MUTE, lw=.6, ls=":", zorder=1)
            ax.set_xlim(-3, 103); ax.set_ylim(0, 103)
            ax.set_xlabel(f"{tag.split()[1]} events removed (% of unsteered)")
            ax.set_title(tag, loc="left", color=INK, weight="bold")
            ax.grid(axis="y", color="#e8e7e3", lw=.6, zorder=0)
        axes[0].set_ylabel("task success (%)")
        h, l = axes[1].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, -0.06), title="clamp gate", title_fontsize=8)
        fig.suptitle(f"PPO+GRU gradient clamp: strength sweep on {len(meta['maps'])} "
                     f"train-pool balanced maps, {meta['rolls']} rollouts per point. "
                     "Along each line $\\theta$ falls from "
                     f"{max(meta['thetas']):g} to {min(meta['thetas']):g}.",
                     fontsize=9.5, color=INK, y=1.01)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print("wrote", out_path.name)


# ------------------------------------------------------------------ verify --
def stage_verify(a, sfx):
    import act5_ppo as A5
    from grid_fig import draw_grid
    summ = json.loads((OUT / f"summary{sfx}.json").read_text())
    meta = summ["meta"]
    pool = pickle.load(open(TEST_PKL, "rb"))
    mids = [int(m) for m in json.loads(ACT11_SUMMARY.read_text())["maps"]][:a.maps]
    chosen = {tag: summ["table"][tag]["best"][a.rule] for tag, _, _, _ in ARMS}
    for tag, s in chosen.items():
        if s is None:
            sys.exit(f"{tag}: no setting satisfies rule {a.rule}")
        print(f"{tag:16s} <- theta {s['theta']:g}, {win_label(s['window'])} "
              f"(train: succ {s['success']:.2f}, removed {100*s['removed']:.0f}%)")
    rows = []
    got = A5.run([job(m, SEED_VERIFY + k, "baseline", 0.0, 0, "test", meta["iters"],
                      meta["alpha"]) for m in mids for k in range(a.rolls)], a.workers)
    for r in got:
        r["arm"] = "unsteered"
    rows += got
    for tag, cond, _, _ in ARMS:
        s = chosen[tag]
        got = A5.run([job(m, SEED_VERIFY + k, cond, s["theta"], s["window"], "test",
                          meta["iters"], meta["alpha"])
                      for m in mids for k in range(a.rolls)], a.workers)
        for r in got:
            r["arm"] = tag
        rows += got
        print(f"  {tag:16s} {len(got)} episodes", flush=True)
    (OUT / f"rows_verify_{a.rule}{sfx}.json").write_text(json.dumps(rows))

    base = [r for r in rows if r["arm"] == "unsteered"]
    out = {}
    print(f"\nheld-out ({len(mids)} maps x {a.rolls}):")
    print(f"{'arm':16s} {'theta':>7s} {'gate':>8s} {'succ':>5s} {'TO':>5s} {'removed':>8s} "
          f"{'median/map':>11s} {'P(top)':>7s} {'P(top|done)':>12s} {'dbelief':>8s} {'score':>6s}")
    for tag, cond, tool, other in [("unsteered", None, "mines", "builds")] + ARMS:
        rs = [r for r in rows if r["arm"] == tag]
        st = setting_stats(base, rs, tool, other, mids)
        s = chosen.get(tag, dict(theta=0.0, window=-1))
        st.update(theta=s["theta"], window=s["window"])
        out[tag] = st
        print(f"{tag:16s} {s['theta']:7g} {('never' if s['window'] == 0 else 'w=' + str(s['window']) if s['window'] > 0 else '-'):>8s} "
              f"{st['success']:5.2f} {st['timeout']:5.2f} {100*st['removed']:7.0f}% "
              f"{st['median_pct']:+6.0f}% ({st['fell']:2d}/{st['n_eff']:2d}) {st['p_top']:7.2f} "
              f"{st['p_top_completed']:12.2f} {st['dbelief']:+8.2f} {st['score']:6.3f}")
    (OUT / f"verify_{a.rule}{sfx}.json").write_text(json.dumps(
        dict(rule=a.rule, maps=mids, rolls=a.rolls, chosen=chosen, result=out), indent=1))

    cb, ct = chosen["suppress bridge"], chosen["suppress tunnel"]
    colours = {"unsteered": "#6b7280", "suppress bridge": "#0e7490",
               "suppress tunnel": "#b91c1c"}
    draw_grid(pool, mids, rows, [(t, colours[t]) for t in ("unsteered",) + tuple(x[0] for x in ARMS)],
              FIG / f"act14_grid_{a.rule}{sfx}.png",
              f"PPO+GRU on held-out balanced maps, {a.rolls} stochastic rollouts each, "
              "identical seeds across the three arms.  Clamp settings chosen on train maps "
              f"by {a.rule}: suppress bridge $\\theta$={cb['theta']:g}, {win_label(cb['window'])}; "
              f"suppress tunnel $\\theta$={ct['theta']:g}, {win_label(ct['window'])}.  "
              "Both exits pay.  Panel label: share of rollouts leaving by the top door.",
              markers=False, door_pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sweep", "analyze", "verify"])
    ap.add_argument("--maps", type=int, default=36)
    ap.add_argument("--rolls", type=int, default=6)
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--thetas", default=",".join(f"{t:g}" for t in THETAS))
    ap.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    ap.add_argument("--rule", default="score", choices=["score", "guard90", "guard80"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""
    {"sweep": stage_sweep, "analyze": stage_analyze, "verify": stage_verify}[a.stage](a, sfx)


if __name__ == "__main__":
    main()
