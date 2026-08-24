#!/usr/bin/env python3
"""Does the door choice track the terrain the agent actually saw?

For every episode we accumulate how many map cells of each type entered the
agent's 21x21 view -- counting each cell once, at the tile value it had when it
was *first* seen, since the agent bridges water and mines rock as it goes. We
then ask whether that evidence count predicts the door.

The interesting case is *balanced* maps: both doors pay, so the choice is free
and nothing in the reward selects it. If the door still tracks the seen
water/rock difference, the agent is running its evidence-integration circuit
even where the answer does not matter -- which makes balanced maps a free probe
of the belief mechanism rather than a filler category.

  collect:  PYTHONPATH=src python scripts/figures/paper_evidence_stats.py --agent ppo
            PYTHONPATH=src:r2dreamer_model ... --agent dreamer
            (from STORM_model/) PYTHONPATH=.:..:../src python ../scripts/... --agent storm
  plot:     python scripts/figures/paper_evidence_stats.py --plot
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

CATS = ("balanced", "lakes", "rocky")
# where along the episode the carried state is sampled, in order
PHASES = ("spawn", "evidence_end", "corridor_mid", "wall")
COL = {"ppo": "#d97706", "dreamer": "#2563eb", "storm": "#16a34a"}
LBL = {"ppo": "PPO + GRU", "dreamer": "DreamerV3", "storm": "STORM"}


# ── collection ───────────────────────────────────────────────────────────

def collect(agent, args):
    from cogniland.bridge_tunnel import tiles as T
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS, make_dreamer, make_ppo, make_storm

    if agent == "ppo":
        act, reset = make_ppo(args.ppo_ckpt)
    elif agent == "storm":
        act, reset = make_storm(args.storm_bundle, args.storm_step)
    else:
        act, reset = make_dreamer(args.dreamer_ckpt, args.device, args.dreamer_size,
                                  sampled=True)
    get_state = getattr(act, "get_state", None)

    with open(args.maps, "rb") as f:
        pool = pickle.load(f)
    half = FORKWALL_KWARGS["view_size"] // 2
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass

    rows, ep_states = [], []
    for i, rec in enumerate(pool):
        env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
        obs, _ = env.reset()
        reset()
        H, W = rec.terrain.shape
        seen = np.zeros((H, W), bool)
        n_water = n_rock = 0
        pass_col = rec.passage_cells[0][1] if rec.passage_cells else rec.wall_col
        mem_lo = max(0, rec.wall_col - 16)
        # Capture the carried state at four points along the episode. One state
        # at the fork is not enough: by then the door decision is already in the
        # state, so *any* direction predicts it and the AUC stops being
        # diagnostic. The corridor is the phase where the belief must be held
        # but the decision is not yet being executed.
        phase_col = {"spawn": -1, "evidence_end": mem_lo,
                     "corridor_mid": (mem_lo + pass_col) // 2, "wall": pass_col}
        phase_state = {k: None for k in phase_col}

        for t in range(FORKWALL_KWARGS["max_steps"]):
            r, c = env._pos
            r0, r1 = max(0, r - half), min(H, r + half + 1)
            c0, c1 = max(0, c - half), min(W, c + half + 1)
            win = seen[r0:r1, c0:c1]
            fresh = ~win
            if fresh.any():
                # value at FIRST sight: the agent may later bridge or mine it
                tiles = np.asarray(env._terrain)[r0:r1, c0:c1][fresh]
                n_water += int((tiles == T.WATER).sum())
                n_rock += int((tiles == T.ROCK).sum())
                win |= True
            if get_state is not None:
                for ph, col in phase_col.items():
                    if phase_state[ph] is None and c >= col:
                        phase_state[ph] = get_state().astype(np.float16)
            obs, _, term, trunc, _ = env.step(act(obs, False))
            if term or trunc:
                break

        fr, fc = env._pos
        top = {p[0] for p in rec.top_goal_cells}
        bot = {p[0] for p in rec.bottom_goal_cells}
        reached = env._pos in (env._correct_cells or set())
        if fc >= rec.terrain.shape[1] - 1 - 1 and fr in top:
            door = "top"
        elif fc >= rec.terrain.shape[1] - 1 - 1 and fr in bot:
            door = "bottom"
        else:
            door = "top" if fr in top else "bottom" if fr in bot else "none"

        # evidence seen by the time each phase was reached is recorded too, so
        # "what it knew" and "what it had seen" line up phase by phase
        rows.append(dict(cat=rec.category, water=n_water, rock=n_rock,
                         door=door, correct=bool(reached), steps=t + 1,
                         phases=[ph for ph in phase_col if phase_state[ph] is not None]))
        ep_states.append(phase_state)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pool)}", flush=True)

    # States are 3 072-d for Dreamer; inlining them as JSON produced a 77 MB file.
    # Keep the per-episode record small and put the states in a compressed .npz.
    out = Path(args.out) / f"evidence_{agent}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    has_state = any(any(v is not None for v in d.values()) for d in ep_states)
    if has_state:
        arrs = {}
        for ph in PHASES:
            idx = [i for i, d in enumerate(ep_states) if d.get(ph) is not None]
            if idx:
                arrs[f"idx_{ph}"] = np.asarray(idx, np.int32)
                arrs[ph] = np.asarray([ep_states[i][ph] for i in idx], np.float16)
        np.savez_compressed(out.with_suffix(".npz"), **arrs)
    print(f"wrote {out}  ({len(rows)} episodes, "
          f"state={'yes -> ' + out.with_suffix('.npz').name if has_state else 'no'})")


# ── statistics ───────────────────────────────────────────────────────────

def logistic_fit(x, y, iters=400, lr=0.5):
    """Tiny IRLS-free logistic regression on a single standardised feature."""
    mu, sd = float(x.mean()), float(x.std() + 1e-9)
    z = (x - mu) / sd
    w = b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(w * z + b)))
        w -= lr * float(((p - y) * z).mean())
        b -= lr * float((p - y).mean())
    return w, b, mu, sd


def auc(scores, labels):
    """Probability a positive outranks a negative (Mann-Whitney)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def analyse(rows):
    """-> per-category statistics dict."""
    out = {}
    for cat in CATS:
        sub = [r for r in rows if r["cat"] == cat]
        if not sub:
            continue
        water = np.array([r["water"] for r in sub], float)
        rock = np.array([r["rock"] for r in sub], float)
        # evidence signal: rock surplus, normalised by total obstacle exposure
        tot = np.maximum(water + rock, 1.0)
        delta = (rock - water) / tot
        d = dict(n=len(sub), water_mean=float(water.mean()), rock_mean=float(rock.mean()),
                 delta_mean=float(delta.mean()), delta_sd=float(delta.std()))
        if cat == "balanced":
            # free choice: does the seen terrain still steer the door?
            top = np.array([r["door"] == "top" for r in sub], float)
            m = np.array([r["door"] in ("top", "bottom") for r in sub])
            if m.sum() > 10 and 0 < top[m].sum() < m.sum():
                w, b, mu, sd = logistic_fit(delta[m], top[m])
                d.update(p_top=float(top[m].mean()),
                         logit_w=float(w), auc=float(auc(delta[m], top[m])),
                         spearman=spearman(delta[m], top[m]),
                         curve=binned(delta[m], top[m]))
        else:
            # decisive: does mistaken evidence explain the wrong doors?
            ok = np.array([r["correct"] for r in sub], float)
            sign = +1.0 if cat == "rocky" else -1.0      # rocky->top, lakes->bottom
            ev = sign * delta                            # evidence FOR the true type
            d.update(p_correct=float(ok.mean()),
                     ev_correct=float(ev[ok == 1].mean()),
                     ev_wrong=float(ev[ok == 0].mean()) if (ok == 0).any() else None,
                     auc=float(auc(ev, ok)), spearman=spearman(ev, ok),
                     curve=binned(ev, ok))
        out[cat] = d
    return out


def binned(x, y, nb=6):
    """Equal-count bins -> (bin centre, mean outcome, n) for the empirical curve."""
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    edges = np.linspace(0, len(xs), nb + 1).astype(int)
    pts = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            pts.append([float(xs[a:b].mean()), float(ys[a:b].mean()), int(b - a)])
    return pts


def belief_axis_stats(rows):
    """Project the fork-time carried state on the lakes->rocky diff-in-means axis."""
    S = {r["cat"]: [] for r in rows}
    for r in rows:
        if r.get("state") is not None:
            S[r["cat"]].append(r["state"])
    if len(S.get("lakes", [])) < 20 or len(S.get("rocky", [])) < 20:
        return None
    lak, roc = np.array(S["lakes"], np.float32), np.array(S["rocky"], np.float32)
    v = roc.mean(0) - lak.mean(0)
    v /= np.linalg.norm(v) + 1e-9
    res = {}
    for cat in CATS:
        sub = [r for r in rows if r["cat"] == cat and r.get("state") is not None]
        if not sub:
            continue
        proj = np.array([float(np.dot(r["state"], v)) for r in sub])
        water = np.array([r["water"] for r in sub], float)
        rock = np.array([r["rock"] for r in sub], float)
        delta = (rock - water) / np.maximum(water + rock, 1.0)
        d = dict(n=len(sub), r_proj_delta=spearman(proj, delta),
                 proj_mean=float(proj.mean()))
        if cat == "balanced":
            top = np.array([r["door"] == "top" for r in sub], float)
            m = np.array([r["door"] in ("top", "bottom") for r in sub])
            if m.sum() > 10:
                d["auc_proj_top"] = float(auc(proj[m], top[m]))
                d["curve_proj"] = binned(proj[m], top[m])
        res[cat] = d
    return res


def phase_sweep(rows, n_null=200, seed=0):
    """Fit the lakes->rocky axis at each episode phase and test it honestly.

    Two questions per phase:
      decode  -- is the map type linearly readable from the carried state at all?
                 (axis fit on half the decisive episodes, scored on the other half)
      free    -- does that axis predict the door on BALANCED maps, where the
                 choice carries no reward?

    `free` is reported against a shuffled-label null: the same difference-of-means
    estimator on the same states with the category labels destroyed. Without it the
    number is meaningless -- in 128 to 3072 dimensions a difference of means is a
    high-variance direction and |AUC-0.5| sits nowhere near 0 under the null.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for ph in PHASES:
        have = [r for r in rows if ph in r["st"]]
        by = {c: np.array([r["st"][ph] for r in have if r["cat"] == c], np.float32)
              for c in ("lakes", "rocky")}
        if min(len(by["lakes"]), len(by["rocky"])) < 40:
            continue
        bal = [r for r in have if r["cat"] == "balanced"
               and r["door"] in ("top", "bottom")]
        if len(bal) < 40:
            continue
        B = np.array([r["st"][ph] for r in bal], np.float32)
        top = np.array([r["door"] == "top" for r in bal], float)

        # held-out decodability of the map type
        dec = []
        for _ in range(20):
            tr, te = {}, {}
            for c in ("lakes", "rocky"):
                q = rng.permutation(len(by[c])); h = len(q) // 2
                tr[c], te[c] = by[c][q[:h]], by[c][q[h:]]
            w = tr["rocky"].mean(0) - tr["lakes"].mean(0)
            n = np.linalg.norm(w)
            if n == 0:
                continue
            w /= n
            sc = np.concatenate([te["rocky"] @ w, te["lakes"] @ w])
            lb = np.concatenate([np.ones(len(te["rocky"])), np.zeros(len(te["lakes"]))])
            dec.append(auc(sc, lb))

        # the real axis, and the balanced-door effect it produces
        v = by["rocky"].mean(0) - by["lakes"].mean(0)
        nv = float(np.linalg.norm(v))
        if nv < 1e-8:
            # at reset every episode carries the same state, so the class means
            # coincide and the axis is undefined. That is the correct answer for
            # a phase where the agent has seen nothing, not a failure.
            out[ph] = dict(n=len(have), n_balanced=len(bal), degenerate=True,
                           decode_auc=None, decode_sd=None, free_effect=None,
                           null_mean=None, null_sd=None, null_p=None)
            continue
        v = v / nv
        real = abs(auc(B @ v, top) - .5)

        # shuffled-label null, same estimator and same states
        allc = np.concatenate([by["lakes"], by["rocky"]])
        null = []
        for _ in range(n_null):
            q = rng.permutation(len(allc)); h = len(q) // 2
            w = allc[q[:h]].mean(0) - allc[q[h:]].mean(0)
            n = np.linalg.norm(w)
            if n > 0:
                null.append(abs(auc(B @ (w / n), top) - .5))
        null = np.array(null)

        out[ph] = dict(n=len(have), n_balanced=len(bal),
                       decode_auc=float(np.mean(dec)) if dec else None,
                       decode_sd=float(np.std(dec)) if dec else None,
                       free_effect=float(real),
                       null_mean=float(null.mean()), null_sd=float(null.std()),
                       null_p=float((null >= real).mean()))
    return out


def load_rows(json_path: Path):
    """Read the episode records and re-attach per-phase states from the .npz."""
    rows = json.loads(json_path.read_text())
    for r in rows:
        r["st"] = {}
    npz = json_path.with_suffix(".npz")
    if npz.exists():
        z = np.load(npz)
        for ph in PHASES:
            if ph not in z:
                continue
            for i, st in zip(z[f"idx_{ph}"], z[ph]):
                rows[int(i)]["st"][ph] = st.astype(np.float32)
    for r in rows:                       # back-compat with the old single-state field
        r["state"] = r["st"].get("wall")
    return rows


# ── figure ───────────────────────────────────────────────────────────────

def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, stats, axis, sweeps = {}, {}, {}, {}
    for ag in ("ppo", "dreamer", "storm"):
        f = out / f"evidence_{ag}.json"
        if f.exists():
            data[ag] = load_rows(f)
            stats[ag] = analyse(data[ag])
            a = belief_axis_stats(data[ag])
            if a:
                axis[ag] = a
            sw = phase_sweep(data[ag])
            if sw:
                sweeps[ag] = sw
    if not data:
        raise SystemExit("no evidence_*.json found -- run collection first")

    rc = {"figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
          "axes.titlesize": 9.5, "axes.labelsize": 9}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

        # (a) balanced maps: the door is free -- does seen terrain still steer it?
        ax = axes[0]
        for ag in data:
            st = stats[ag].get("balanced")
            if not st or "curve" not in st:
                continue
            cur = np.array(st["curve"])
            ax.plot(cur[:, 0], cur[:, 1], "o-", color=COL[ag], lw=1.7, ms=4.5,
                    label=f"{LBL[ag]}   AUC {st['auc']:.2f}")
            ax.axhline(st["p_top"], color=COL[ag], ls=":", lw=1, alpha=.7)
        ax.axhline(.5, color="#9ca3af", ls="--", lw=1)
        ax.set_title("(a) balanced maps — the door choice is FREE", loc="left")
        ax.set_xlabel("rock − water seen  (normalised)")
        ax.set_ylabel("P(top door)")
        ax.set_ylim(-.02, 1.05)
        ax.legend(frameon=False, fontsize=7.4, loc="upper left")
        ax.grid(alpha=.25, lw=.5)
        ax.annotate("dotted = each agent's overall\nP(top): its standing bias",
                    xy=(.98, .06), xycoords="axes fraction", fontsize=6.8,
                    ha="right", color="#6b7280")

        # (b) phase sweep. The story is the GAP between the real axis and the
        #     shuffled-label null, so draw the null as a band and the effect on
        #     top of it; decodability goes in as text since it is ~1 throughout.
        ax = axes[1]
        xs = np.arange(len(PHASES))
        drawn = False
        for ag in data:
            sw = sweeps.get(ag) or {}
            m = [i for i, ph in enumerate(PHASES)
                 if sw.get(ph, {}).get("free_effect") is not None]
            if not m:
                continue
            drawn = True
            X = xs[m]
            eff = np.array([sw[PHASES[i]]["free_effect"] for i in m])
            nm = np.array([sw[PHASES[i]]["null_mean"] for i in m])
            nsd = np.array([sw[PHASES[i]]["null_sd"] for i in m])
            ax.fill_between(X, nm - nsd, nm + nsd, color=COL[ag], alpha=.13, lw=0)
            ax.plot(X, nm, ls=":", color=COL[ag], lw=1.2)
            ax.plot(X, eff, "o-", color=COL[ag], lw=2.0, ms=5.5,
                    label=f"{LBL[ag]}")
            dy = 8 if ag == "ppo" else -14
            for i, x in zip(m, X):
                pv = sw[PHASES[i]]["null_p"]
                ax.annotate(f"p={pv:.3f}" if pv >= .001 else "p<0.001",
                            (x, sw[PHASES[i]]["free_effect"]),
                            textcoords="offset points", xytext=(0, dy),
                            fontsize=6.6, ha="center", color=COL[ag],
                            fontweight="bold" if pv < .05 else "normal")
        ax.set_xticks(xs)
        ax.set_xticklabels(["spawn\n(undefined)", "evidence\nends",
                            "corridor\nmid", "at the\nwall"], fontsize=7.6)
        ax.set_ylabel("|AUC − 0.5|  on the free door choice")
        ax.set_ylim(0, .62)
        ax.set_title("(b) the belief decides — but only before the wall", loc="left")
        if drawn:
            ax.legend(frameon=False, fontsize=7.2, loc="upper left")
        ax.grid(alpha=.25, lw=.5)
        ax.annotate("dotted + band = shuffled-label null (mean ± sd)",
                    xy=(.97, .04), xycoords="axes fraction", fontsize=6.6,
                    ha="right", color="#6b7280",
                    bbox=dict(boxstyle="round,pad=.25", fc="white",
                              alpha=.85, ec="none"))

        # (c) decisive maps: near ceiling, so show the evidence DISTRIBUTION
        ax = axes[2]
        width, off = 0.26, {"ppo": -0.27, "dreamer": 0.0, "storm": 0.27}
        for ag in data:
            ev_ok, ev_bad = [], []
            for r in data[ag]:
                if r["cat"] == "balanced":
                    continue
                tot = max(r["water"] + r["rock"], 1)
                sign = 1.0 if r["cat"] == "rocky" else -1.0
                (ev_ok if r["correct"] else ev_bad).append(
                    sign * (r["rock"] - r["water"]) / tot)
            for j, (vals, fc) in enumerate(((ev_ok, COL[ag]), (ev_bad, "#ffffff"))):
                if len(vals) < 2:
                    continue
                bp = ax.boxplot([vals], positions=[j + off[ag]], widths=width,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="#111827", lw=1.2))
                bp["boxes"][0].set(facecolor=fc, edgecolor=COL[ag], lw=1.3)
            ax.plot([], [], color=COL[ag], lw=6, label=LBL[ag])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["correct door", "WRONG door"])
        ax.set_ylabel("evidence for the true type")
        ax.set_title("(c) lakes + rocky — why the errors happen", loc="left")
        ax.legend(frameon=False, fontsize=7.4, loc="lower left")
        ax.grid(alpha=.25, lw=.5, axis="y")

        fig.suptitle("What the agent saw versus what it chose — "
                     "all 1 200 held-out maps per agent", y=1.03, fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "fig_evidence.png", bbox_inches="tight")
        plt.close(fig)

    (out / "evidence_stats.json").write_text(
        json.dumps({"outcome": stats, "belief_axis": axis,
                    "phase_sweep": sweeps}, indent=1))
    for ag in stats:
        print(f"\n{LBL[ag]}")
        for cat in CATS:
            s = stats[ag].get(cat)
            if not s:
                continue
            head = (f"P(top)={s.get('p_top', float('nan')):.3f}" if cat == "balanced"
                    else f"P(correct)={s.get('p_correct', float('nan')):.3f}")
            print(f"  {cat:9s} n={s['n']:4d} water={s['water_mean']:6.1f} "
                  f"rock={s['rock_mean']:6.1f}  {head} "
                  f"AUC={s.get('auc', float('nan')):.3f} "
                  f"rho={s.get('spearman', float('nan')):+.3f}")
        if ag in axis:
            for cat, d in axis[ag].items():
                extra = (f" AUC(proj->top)={d['auc_proj_top']:.3f}"
                         if "auc_proj_top" in d else "")
                print(f"    [state] {cat:9s} rho(proj, rock-water)="
                      f"{d['r_proj_delta']:+.3f}{extra}")
    print("\nwrote", out / "fig_evidence.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["ppo", "dreamer", "storm"])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--ppo-ckpt", default=str(REPO / "final_models/ppo/ppo_plain.pt"))
    p.add_argument("--storm-bundle", default=str(REPO / "final_models/storm"))
    p.add_argument("--storm-step", type=int, default=624489)
    p.add_argument("--dreamer-ckpt",
                   default=str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"))
    p.add_argument("--dreamer-size", default="size25M")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.plot:
        plot(Path(a.out))
    else:
        collect(a.agent, a)


if __name__ == "__main__":
    main()
