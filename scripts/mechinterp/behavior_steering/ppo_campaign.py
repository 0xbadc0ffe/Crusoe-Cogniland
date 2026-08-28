#!/usr/bin/env python3
"""PPO behaviour-steering campaign: suppress / incentivize mining and building.

Methods
  M1 route-intent coordinate-set  h' = h + (c - h.v)v along a through-vs-around
     axis fitted on EARLY-window states of the frozen dataset (the same
     protocol that steers the belief), transient or sustained.
  M2 soft logit bias              logits[MINE or BUILD] += b each step.
  M3 SAE feature clamp            subtract the mine/build feature family's
     current contribution (suppress) or add its decoder direction (boost).
  M4 h-add along the contrast axis  h' = h + beta * v_mine / v_build.
  combos                          M1+M2 at gentle doses.

Every episode is run with an in-process env loop (not replay()) so mines and
builds are counted from the env's own info flags, and the tool events carry the
cell they changed -- the qualitative traces need that. Seeds follow the
campaign convention (1000+map_id on held-out maps, 2000+k on the three
qualitative maps), so the baseline rows reproduce the dataset / figure-7.5
episodes exactly.

  # fit the route axes, run pilots, run the campaign, emit qualitative traces
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report:scripts/figures \
      python scripts/mechinterp/behavior_steering/ppo_campaign.py --stage axes
  ... --stage pilot | campaign | qual | summary
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/mechinterp", "scripts/mechinterp/belief_report",
          "scripts/figures"):
    sys.path.insert(0, str(REPO / p))

OUT = REPO / "outputs/behavior_steering/ppo"
FIG = REPO / "paper/figures/behavior_steering"
A_BUILD, A_MINE = 4, 5
FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
EARLY_CRW = -24              # "early window" = col_rel_wall < this
TOOL_MIN = 2                 # >=2 tool presses labels an episode "through"
QUAL_MAPS = {626: "lakes", 77: "rocky", 99: "balanced"}


# ── axes (M1) ────────────────────────────────────────────────────────────

def fit_axes():
    import data as D
    X, df = D.load("ppo")
    tr, te = D.split_maps(df)
    tr = set(tr)
    zb = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz")
    v_belief = zb["v"].astype(np.float32)

    def episode_early_means(cat, tool_a):
        """per-episode mean h over the early window, before the first tool
        press; -> ids, labels(through=1), means"""
        sub = df[df.category == cat]
        ids, lab, mean = [], [], []
        for mid, g in sub.groupby("map_id"):
            n_tool = int((g.action == tool_a).sum())
            if n_tool == 1:
                continue                     # ambiguous, excluded from fitting
            first_tool = g[g.action == tool_a]["t"].min() if n_tool else np.inf
            early = g[(g.col_rel_wall < EARLY_CRW) & (g.t < first_tool)]
            if len(early) < 3:
                continue
            ids.append(int(mid))
            lab.append(1 if n_tool >= TOOL_MIN else 0)
            mean.append(np.asarray(X[early.index.to_numpy()], np.float32).mean(0))
        return np.array(ids), np.array(lab), np.stack(mean)

    axes = {}
    report = {}
    for name, cat, tool_a in (("rocky_mine", "rocky", A_MINE),
                              ("lakes_build", "lakes", A_BUILD)):
        ids, lab, M = episode_early_means(cat, tool_a)
        m = np.isin(ids, list(tr))
        thr, arо = M[m][lab[m] == 1], M[m][lab[m] == 0]
        v = thr.mean(0) - arо.mean(0)
        v /= np.linalg.norm(v) + 1e-12
        mu_thr = float(thr.mean(0) @ v)
        mu_aro = float(arо.mean(0) @ v)
        # held-out separation of the axis, as a sanity number
        mt = np.isin(ids, list(set(ids) - tr))
        proj = M[mt] @ v
        import data as D2
        auc = D2.auc(proj, lab[mt].astype(float))
        axes[f"v_{name}"] = v
        axes[f"mu_thr_{name}"] = mu_thr
        axes[f"mu_aro_{name}"] = mu_aro
        axes[f"proto_thr_{name}"] = thr.mean(0)
        axes[f"proto_aro_{name}"] = arо.mean(0)
        report[name] = dict(n_train_through=int((lab[m] == 1).sum()),
                            n_train_around=int((lab[m] == 0).sum()),
                            heldout_auc=float(auc),
                            cos_belief=float(v @ v_belief),
                            mu_through=mu_thr, mu_around=mu_aro)

    # pooled tool-vs-no-tool across both decisive categories
    ids_r, lab_r, M_r = episode_early_means("rocky", A_MINE)
    ids_l, lab_l, M_l = episode_early_means("lakes", A_BUILD)
    v_parts = []
    for ids, lab, M in ((ids_r, lab_r, M_r), (ids_l, lab_l, M_l)):
        m = np.isin(ids, list(tr))
        v_parts.append(M[m][lab[m] == 1].mean(0) - M[m][lab[m] == 0].mean(0))
    v = np.mean(v_parts, 0)
    v /= np.linalg.norm(v) + 1e-12
    axes["v_pooled"] = v
    report["pooled"] = dict(cos_belief=float(v @ v_belief),
                            cos_rocky=float(v @ axes["v_rocky_mine"]),
                            cos_lakes=float(v @ axes["v_lakes_build"]))

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "ppo_route_axes.npz", v_belief=v_belief, **axes)
    (OUT / "ppo_route_axes.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ── hooks ────────────────────────────────────────────────────────────────

def load_axes():
    z = np.load(OUT / "ppo_route_axes.npz")
    return {k: z[k] for k in z.files}


def load_prep_axes():
    z = np.load(REPO / "outputs/behavior_steering/behavior_axes.npz")
    return {k: z[k].astype(np.float32) for k in z.files}


class SAEKit:
    """numpy port of the trained SAE, enough to clamp feature families."""
    MINE_F = [986, 789, 446]
    BUILD_F = [90, 616, 577]

    def __init__(self):
        import torch
        d = torch.load(REPO / "outputs/behavior_steering/sae_ppo.pt",
                       map_location="cpu", weights_only=False)
        sd = d["state_dict"]
        self.We = sd["We"].numpy().astype(np.float32)
        self.be = sd["be"].numpy().astype(np.float32)
        self.Wd = sd["Wd"].numpy().astype(np.float32)
        self.bd = sd["bd"].numpy().astype(np.float32)
        self.mu = np.asarray(d["mu"], np.float32)
        self.sd = np.asarray(d["sd"], np.float32)

    def feats(self, h, idx):
        x = (h - self.mu) / self.sd
        return np.maximum((x - self.bd) @ self.We[idx].T + self.be[idx], 0.0)

    def suppress_hook(self, idx):
        def hook(h, t, info):
            x = (h - self.mu) / self.sd
            f = np.maximum((x - self.bd) @ self.We[idx].T + self.be[idx], 0.0)
            if not f.any():
                return h
            x2 = x - self.Wd[:, idx] @ f
            return (x2 * self.sd + self.mu).astype(np.float32)
        return hook

    def boost_hook(self, idx, lam):
        d = self.Wd[:, idx].mean(1)
        d /= np.linalg.norm(d) + 1e-12
        delta = (lam * d * self.sd).astype(np.float32)
        def hook(h, t, info):
            return h + delta
        return hook


def coordset_hook(v, c, t0=0, sustained=True):
    v = v.astype(np.float32)
    def hook(h, t, info):
        if (sustained and t >= t0) or ((not sustained) and t == t0):
            return h + (c - float(h @ v)) * v
        return h
    return hook


class GatedCoordset:
    """Coordinate-set applied ONLY inside the axis's fitting window
    (col_rel_wall < gate_end): the target is an early-approach statistic, so
    clamping it during obstacle engagement is off-phase and was the pilot's
    failure. `transient` writes once, at the first gated step."""
    wants_ctx = True

    def __init__(self, v, c, gate_end=EARLY_CRW, transient=False):
        self.v = v.astype(np.float32)
        self.c = float(c)
        self.gate_end = gate_end
        self.transient = transient

    def bind(self, ctx):
        fired = [False]
        def hook(h, t, info):
            if ctx["crw"] < self.gate_end and not (self.transient and fired[0]):
                fired[0] = True
                return h + (self.c - float(h @ self.v)) * self.v
            return h
        return hook


def hadd_hook(v, beta):
    delta = (beta * v).astype(np.float32)
    def hook(h, t, info):
        return h + delta
    return hook


def bias_vec(action, b):
    z = np.zeros(6, np.float32)
    z[action] = b
    return z


# ── episode runner (event fidelity) ──────────────────────────────────────

def run_episode(rec, seed, hook=None, lbias=None, want_steps=False, want_h=False):
    import torch
    import replay_episode as RE
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    act, reset = RE._get_agent("ppo", "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    # live position context for phase-gated hooks: updated before each act()
    ctx = {"crw": -999}
    if hook is not None and getattr(hook, "wants_ctx", False):
        hook = hook.bind(ctx)
    act.set_hook(hook)
    act.set_logit_bias(lbias)
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    reset()
    steps = [dict(r=int(env._pos[0]), c=int(env._pos[1]),
                  facing=int(env._facing), ev=None)]
    hs = []
    mines = builds = 0
    wall = int(rec.wall_col)
    for t in range(FORKWALL_KWARGS["max_steps"]):
        ctx["crw"] = int(env._pos[1]) - wall
        a = act(obs, False)
        if want_h:
            hs.append(act.get_state().astype(np.float32))
        obs, _, term, trunc, info = env.step(a)
        ev = None
        if a in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
            dr, dc = FACE_DELTA[int(info["facing"])]
            ev = dict(kind="build" if a == A_BUILD else "mine",
                      r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
            if a == A_MINE:
                mines += 1
            else:
                builds += 1
        steps.append(dict(r=int(env._pos[0]), c=int(env._pos[1]),
                          facing=int(info["facing"]), ev=ev))
        if term or trunc:
            break
    fr = env._pos
    top = {p[0] for p in rec.top_goal_cells}
    bot = {p[0] for p in rec.bottom_goal_cells}
    row = dict(success=bool(env._pos in (env._correct_cells or set())),
               door="top" if fr[0] in top else "bottom" if fr[0] in bot else "none",
               steps=len(steps) - 1, mines=mines, builds=builds)
    if want_steps:
        row["trace"] = steps
    if want_h:
        row["hs"] = np.array(hs)
    return row


# ── donor-prefix patching (the suppression method) ───────────────────────

DONOR_SEED0 = 3000
DONOR_TRIES = 12
PREFIX_K = 20


def find_donor(rec, tool):
    """First correct baseline rollout of THIS map that never used `tool`
    (a real 'around mode' of this map), from a seed range disjoint from both
    the campaign seeds (1000+id) and the qualitative seeds (2000+k).
    -> (h_prefix array, seed) or (None, tries)."""
    for j in range(DONOR_TRIES):
        r = run_episode(rec, DONOR_SEED0 + j, want_h=True)
        used = r["mines"] if tool == "mine" else r["builds"]
        if r["success"] and used == 0:
            return r["hs"], DONOR_SEED0 + j
    return None, DONOR_TRIES


def donor_prefix_hook(donor_h, K=PREFIX_K):
    """Clamp h to the donor's own h for the first K steps, then release.
    Transient by construction: after step K the recurrence carries whatever
    intent the prefix installed."""
    def hook(h, t, info):
        if t < min(K, len(donor_h)):
            return donor_h[t]
        return h
    return hook


# ── condition registry ───────────────────────────────────────────────────

def make_condition(name, tool):
    """-> (hook, lbias) for a named condition on a given tool ('mine'/'build').

    m1g_<lam>       gated sustained coordinate-set, target mu_aro + lam*(mu_thr-mu_aro)
    m1t_<lam>       gated transient (single write) version
    m1p_<lam>       belief-orthogonalised axis, gated sustained
    m2_<b>          logit bias b on the tool action
    m3_sup/m3_inc_<l>  SAE clamp
    m4_<b>          plain h-add along the prep contrast axis
    combo_<lam>_<b> m1g + logit bias
    """
    ax = load_axes()
    prep = load_prep_axes()
    sae = make_condition._sae or SAEKit()
    make_condition._sae = sae
    key = "rocky_mine" if tool == "mine" else "lakes_build"
    v = ax[f"v_{key}"].astype(np.float32)
    v_bel = ax["v_belief"].astype(np.float32)
    mu_thr = float(ax[f"mu_thr_{key}"])
    mu_aro = float(ax[f"mu_aro_{key}"])
    action = A_MINE if tool == "mine" else A_BUILD
    vc = prep["v_mine"] if tool == "mine" else prep["v_build"]
    vc = vc / (np.linalg.norm(vc) + 1e-12)
    fam = SAEKit.MINE_F if tool == "mine" else SAEKit.BUILD_F

    def target(lam):
        return mu_aro + lam * (mu_thr - mu_aro)

    def perp_axis():
        w = v - (v @ v_bel) * v_bel
        w /= np.linalg.norm(w) + 1e-12
        # class-mean coordinates along the orthogonalised axis, from stored mus:
        # recompute cheaply by projecting the class means implied by (v, mus)
        # is not possible without the raw means, so re-derive scale from v.w
        scale = float(v @ w)
        return w, mu_thr * scale, mu_aro * scale

    nm = name
    if nm.startswith("m1g_"):
        return GatedCoordset(v, target(float(nm[4:]))), None
    if nm.startswith("m1t_"):
        return GatedCoordset(v, target(float(nm[4:])), transient=True), None
    if nm.startswith("m1p_"):
        w, m_thr, m_aro = perp_axis()
        lam = float(nm[4:])
        return GatedCoordset(w, m_aro + lam * (m_thr - m_aro)), None
    if nm.startswith("m2_"):
        return None, (lambda t, b=float(nm[3:]): bias_vec(action, b))
    if nm == "m3_sup":
        return sae.suppress_hook(fam), None
    if nm.startswith("m3_inc"):
        return sae.boost_hook(fam, float(nm.split("_")[-1])), None
    if nm.startswith("m4_"):
        return hadd_hook(vc, float(nm[3:])), None
    if nm.startswith("m5_"):
        _, which, a_ = nm.split("_")
        proto = ax[f"proto_{'aro' if which == 'aro' else 'thr'}_{key}"].astype(np.float32)
        alpha = float(a_)
        class Pull:
            wants_ctx = True
            def bind(self, ctx, proto=proto, alpha=alpha):
                def hook(h, t, info):
                    if ctx["crw"] < EARLY_CRW:
                        return h + alpha * (proto - h)
                    return h
                return hook
        return Pull(), None
    if nm.startswith("combo_"):
        _, lam, b = nm.split("_")
        return GatedCoordset(v, target(float(lam))), \
            (lambda t, bb=float(b): bias_vec(action, bb))
    raise KeyError(nm)


make_condition._sae = None


# ── stages ───────────────────────────────────────────────────────────────

def stage_pilot():
    """map 77 (rocky), figure-7.5 seeds 0..7: can anything consolidate?"""
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rec = pool[77]
    conds = ["baseline", "m1g_0", "m1g_1", "m1g_-0.5", "m1g_1.5",
             "m1t_0", "m1t_1", "m1p_0", "m1p_1",
             "combo_0_-2", "combo_1_+1"]
    for nm in conds:
        hook = lbias = None
        if nm != "baseline":
            hook, lbias = make_condition(nm, "mine")
        rows = [run_episode(rec, 2000 + k, hook, lbias) for k in range(8)]
        thr = sum(r["mines"] > 0 for r in rows)
        print(f"{nm:12s} through {thr}/8  succ {sum(r['success'] for r in rows)}/8  "
              f"mines {np.mean([r['mines'] for r in rows]):4.1f}  "
              f"steps {np.mean([r['steps'] for r in rows]):5.0f}  "
              f"timeouts {sum(r['steps'] >= 799 for r in rows)}", flush=True)


def stage_campaign():
    import data as D
    _, df = D.load("ppo")
    _, te = D.split_maps(df)
    te = set(int(x) for x in te)
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    per_cat = {}
    for mid in sorted(te):
        per_cat.setdefault(pool[mid].category, []).append(mid)

    rows = []

    def log(sub, tag):
        print(f"{tag:34s} n={len(sub):3d}  succ {np.mean([r['success'] for r in sub]):.2f}  "
              f"mines {np.mean([r['mines'] for r in sub]):5.1f}  "
              f"builds {np.mean([r['builds'] for r in sub]):5.1f}  "
              f"to {np.mean([r['steps'] >= 799 for r in sub]):.2f}", flush=True)

    for tool, cat, n in (("mine", "rocky", 50), ("build", "lakes", 50)):
        maps = per_cat[cat][:n]
        # baseline
        base = []
        for mid in maps:
            r = run_episode(pool[mid], 1000 + mid)
            base.append(dict(tool=tool, cond="baseline", cat=cat, map_id=mid, **r))
        rows += base; log(base, f"{cat} baseline")
        # donor discovery (for the suppression arm) — donors are per map
        donors, cov = {}, 0
        for mid in maps:
            dh, seed_or_tries = find_donor(pool[mid], tool)
            if dh is not None:
                donors[mid] = dh; cov += 1
        print(f"{cat}: donor coverage {cov}/{len(maps)} maps "
              f"(a no-tool mode exists within {DONOR_TRIES} baseline samples)", flush=True)
        # suppression: donor-prefix on covered maps
        sup = []
        for mid, dh in donors.items():
            r = run_episode(pool[mid], 1000 + mid, donor_prefix_hook(dh), None)
            sup.append(dict(tool=tool, cond="donor_prefix", cat=cat, map_id=mid, **r))
        rows += sup; log(sup, f"{cat} suppress: donor-prefix K={PREFIX_K}")
        # suppression negatives, documented at smaller n
        for cond in ("m1g_0", "m2_-2", "m3_sup"):
            neg = []
            for mid in maps[:25]:
                hook, lb = make_condition(cond, tool)
                r = run_episode(pool[mid], 1000 + mid, hook, lb)
                neg.append(dict(tool=tool, cond=cond, cat=cat, map_id=mid, **r))
            rows += neg; log(neg, f"{cat} suppress-negative {cond}")
        # incentivize: gated route-axis overshoot + belief-orth variant
        for cond in ("m1g_1.5", "m1p_1", "m2_+2"):
            inc = []
            for mid in maps:
                hook, lb = make_condition(cond, tool)
                r = run_episode(pool[mid], 1000 + mid, hook, lb)
                inc.append(dict(tool=tool, cond=cond, cat=cat, map_id=mid, **r))
            rows += inc; log(inc, f"{cat} incentivize {cond}")

    # balanced (the map-99 regime): unavoidable obstacles
    maps = per_cat["balanced"][:30]
    base = []
    for mid in maps:
        r = run_episode(pool[mid], 1000 + mid)
        base.append(dict(tool="mine", cond="baseline", cat="balanced", map_id=mid, **r))
    rows += base; log(base, "balanced baseline")
    donors, cov = {}, 0
    for mid in maps:
        dh, _ = find_donor(pool[mid], "mine")
        if dh is not None:
            donors[mid] = dh; cov += 1
    print(f"balanced: no-mine donor coverage {cov}/{len(maps)}", flush=True)
    sup = []
    for mid, dh in donors.items():
        r = run_episode(pool[mid], 1000 + mid, donor_prefix_hook(dh), None)
        sup.append(dict(tool="mine", cond="donor_prefix", cat="balanced",
                        map_id=mid, **r))
    rows += sup; log(sup, "balanced suppress-mine: donor-prefix")
    for cond, lbf in (("m2_-2", lambda t: bias_vec(A_MINE, -2)),
                      ("m2_both_-2", lambda t: np.array([0, 0, 0, 0, -2, -2],
                                                        np.float32))):
        sub = []
        for mid in maps:
            r = run_episode(pool[mid], 1000 + mid, None, lbf)
            sub.append(dict(tool="mine", cond=cond, cat="balanced", map_id=mid, **r))
        rows += sub; log(sub, f"balanced {cond}")

    (OUT / "campaign.json").write_text(json.dumps(rows))
    print("wrote", OUT / "campaign.json", len(rows), "rows")


def stage_controls():
    """random matched-norm directions for the winning h-space method (M1)."""
    ax = load_axes()
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    import data as D
    _, df = D.load("ppo")
    _, te = D.split_maps(df)
    te = set(int(x) for x in te)
    rocky = [m for m in sorted(te) if pool[m].category == "rocky"][:20]
    v = ax["v_rocky_mine"]; mu_aro = float(ax["mu_aro_rocky_mine"])
    rng = np.random.default_rng(0)
    rows = []
    for mid in rocky:
        rec = pool[mid]
        base = run_episode(rec, 1000 + mid)
        real = run_episode(rec, 1000 + mid, coordset_hook(v, mu_aro), None)
        rows.append(dict(kind="real", map_id=mid, **real))
        rows.append(dict(kind="base", map_id=mid, **base))
        for k in range(5):
            r_ = rng.standard_normal(128).astype(np.float32)
            r_ /= np.linalg.norm(r_)
            # matched displacement: set the same coordinate value along a
            # random direction (same operation, random direction)
            res = run_episode(rec, 1000 + mid, coordset_hook(r_, mu_aro), None)
            rows.append(dict(kind="rand", dir=k, map_id=mid, **res))
    (OUT / "controls.json").write_text(json.dumps(rows))
    for kind in ("base", "real", "rand"):
        sub = [r for r in rows if r["kind"] == kind]
        print(f"{kind:5s} n={len(sub)} succ {np.mean([r['success'] for r in sub]):.2f} "
              f"mines {np.mean([r['mines'] for r in sub]):4.1f} "
              f"to {np.mean([r['steps'] >= 799 for r in sub]):.2f}")


def stage_qual():
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    plan = {
        77: [("baseline", None, None),
             ("suppress-mine (donor-prefix)", "donor", "mine"),
             ("incentivize-mine (route-axis)", "m1g_1.5", "mine")],
        626: [("baseline", None, None),
              ("suppress-build (donor-prefix)", "donor", "build"),
              ("incentivize-build (route-axis)", "m1g_1.5", "build")],
        99: [("baseline", None, None),
             ("suppress-mine (donor-prefix)", "donor", "mine"),
             ("suppress-both (logit -2)", "m2_both", None)],
    }
    for mid, conds in plan.items():
        rec = pool[mid]
        for label, nm, tool in conds:
            if nm == "donor":
                dh, info_ = find_donor(rec, tool)
                if dh is None:
                    print(f"map {mid} {label}: NO DONOR in {DONOR_TRIES} tries — skipped")
                    continue
                hook, lbias = donor_prefix_hook(dh), None
            elif nm == "m2_both":
                hook, lbias = None, (lambda t: np.array([0, 0, 0, 0, -2, -2],
                                                        np.float32))
            elif nm is None:
                hook = lbias = None
            else:
                hook, lbias = make_condition(nm, tool)
            rolls = []
            for k in range(20):
                r = run_episode(rec, 2000 + k, hook, lbias, want_steps=True)
                rolls.append(dict(steps=r["trace"], correct=r["success"]))
            out = {QUAL_MAPS[mid]: dict(map_id=mid, rollouts=rolls)}
            slug = label.split(" ")[0]
            p = OUT / f"qual_{mid}_{slug}.json"
            p.write_text(json.dumps(out))
            n_ok = sum(r["correct"] for r in rolls)
            n_mine = sum(1 for r in rolls
                         for st in r["steps"] if st["ev"] and st["ev"]["kind"] == "mine")
            print(f"map {mid} {label:34s} success {n_ok}/20  mine-events {n_mine}  "
                  f"-> {p.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["axes", "pilot", "campaign", "controls", "qual"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    dict(axes=fit_axes, pilot=stage_pilot, campaign=stage_campaign,
         controls=stage_controls, qual=stage_qual)[a.stage]()


if __name__ == "__main__":
    main()
