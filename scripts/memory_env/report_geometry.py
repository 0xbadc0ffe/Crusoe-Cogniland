#!/usr/bin/env python
"""Neural-geometry report for the MemoryEnv PPO models (Goodfire-style HTML).

Story: neural geometries expose an RL agent's BELIEFS, DECISIONS and BEHAVIOR.
  1. Define a belief space (cue-posterior simplex) and a behavior space
     (Monte-Carlo outcome simplex); embed both with inPCA; fit the activation
     manifold (a trunk-and-fibers tree parameterized by task progress).
  2. Show the two-way steering results on those manifolds: belief edits change
     behavior; behavior forcing changes belief through evidence (2cue vs 4cue).

All geometry code is environment-agnostic (neurogeom.py; arrays in/out) so the
same figures can be produced for bridge_tunnel/cogniland: the only env-specific
part is the data collection below (traces, MC outcome rollouts, event labels).

Usage:
  python report_geometry.py --run-dirs outputs/ppo_runs/ppo_2cue_mk2 \
      outputs/ppo_runs/ppo_3cue_mk3s1 outputs/ppo_runs/ppo_4cue_mk4s3 \
      --out outputs/geometry_report.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys

import numpy as np
import jax
import jax.numpy as jnp

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, make_state, constants as C,
)
import diag_jax as D  # noqa: E402
import train_ppo_memory as P  # noqa: E402
import steer2_ppo as S  # noqa: E402
import neurogeom as G  # noqa: E402

CUE_NAMES = S.CUE_NAMES
CUE_COL = S.CUE_COL
BLANK_COL = "#9a9a9a"
OUTCOME_NAMES = ["up+green", "up+blue", "down+green", "down+blue"]
OUTCOME_COL = ["#1b9e77", "#3b6fb6", "#7fd4b8", "#9ec9ec"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


# ─────────────────────────────────────────────────────────────────────────────
# data collection (the ONLY env-specific part)
# ─────────────────────────────────────────────────────────────────────────────
def collect(cfg, net, params, cue, key, n=192, T=80):
    """Sampled-policy rollouts returning per-step full EnvState + hidden."""
    p = D._env_params(cfg, cue)
    keys = jax.random.split(key, n)
    state = jax.vmap(lambda k: jreset(k, p))(keys)
    obs = D._flat(jax.vmap(lambda s: build_obs(s, p))(state))
    hidden = P.ScannedRNN.initialize_carry(n, cfg["gru_hidden"])

    def body(carry, _):
        state, obs, hidden, last_done, dacc, key = carry
        new_hidden, logits, _ = net.apply(params, hidden, (obs[None], last_done[None]))
        key, ka = jax.random.split(key)
        a = jax.random.categorical(ka, logits[0], axis=-1).astype(jnp.int32)
        key, sk = jax.random.split(key)
        sks = jax.random.split(sk, n)
        ns, r, dn, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = D._flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        mm = jax.vmap(lambda s: build_obs(s, p)["minimap"])(state)
        cue_vis = (mm >= C.CUE_GREEN_UP) & (mm <= C.CUE_BLUE_DOWN)
        out = dict(state=state, h=new_hidden, alive=~dacc,
                   cue_vis=cue_vis.any((-2, -1)),
                   sel=ns.selected_door, tb=ns.taken_branch, done=dn,
                   reach=info["reached_target"])
        return (ns, nobs, new_hidden, dn, dacc | dn, key), out

    carry = (state, obs, hidden, jnp.zeros((n,), bool), jnp.zeros((n,), bool), key)
    _, outs = jax.lax.scan(body, carry, None, length=T)
    return p, jax.tree_util.tree_map(np.asarray, outs)


def mc_outcomes(cfg, net, params, p, sel_state, h0, key, K=24, T=80):
    """Monte-Carlo outcome posterior: continue the policy K times from each
    (EnvState, hidden) pair -> P over (branch x door) outcomes on the simplex."""
    Ssel = h0.shape[0]
    st = jax.tree_util.tree_map(lambda a: jnp.repeat(jnp.asarray(a), K, axis=0), sel_state)
    hidden = jnp.repeat(jnp.asarray(h0, jnp.float32), K, axis=0)
    n = Ssel * K
    obs = D._flat(jax.vmap(lambda s: build_obs(s, p))(st))

    def body(carry, _):
        state, obs, hidden, last_done, dacc, tb, sel, key = carry
        new_hidden, logits, _ = net.apply(params, hidden, (obs[None], last_done[None]))
        key, ka = jax.random.split(key)
        a = jax.random.categorical(ka, logits[0], axis=-1).astype(jnp.int32)
        key, sk = jax.random.split(key)
        sks = jax.random.split(sk, n)
        ns, r, dn, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = D._flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        newly = dn & (~dacc)
        tb = jnp.where(newly, ns.taken_branch, tb)
        sel = jnp.where(newly, ns.selected_door, sel)
        return (ns, nobs, new_hidden, dn, dacc | dn, tb, sel, key), None

    tb0 = jnp.asarray(st.taken_branch)          # may already be decided
    carry = (st, obs, hidden, jnp.zeros((n,), bool), jnp.zeros((n,), bool),
             tb0, jnp.zeros((n,), jnp.int32), key)
    (st_f, _, _, _, dacc, tb, sel, _), _ = jax.lax.scan(body, carry, None, length=T)
    tb, sel, dacc = (np.asarray(v).reshape(Ssel, K) for v in (tb, sel, dacc))
    # outcome classes: (branch, door) = (up|down) x (green|blue); drop timeouts
    Pout = np.zeros((Ssel, 4))
    for b, bv in enumerate([C.BRANCH_UP, C.BRANCH_DOWN]):
        for d_, dv in enumerate([C.SEL_GREEN, C.SEL_BLUE]):
            Pout[:, 2 * b + d_] = ((tb == bv) & (sel == dv) & dacc).mean(1)
    Z = Pout.sum(1, keepdims=True)
    return Pout / np.clip(Z, 1e-9, None), Z[:, 0]


def build_dataset(run_dir, key, n_ep=192, T=80):
    """Flat per-model dataset + probes + MC behavior points."""
    cfg, net, params, pr = S.load_all(run_dir)
    trained = pr["trained"]
    key = jax.random.PRNGKey(key)

    Hs, Xs, Cs, Ts, Seen, Alive = [], [], [], [], [], []
    per_cue = {}
    p_env = None
    for c in trained:
        key, k = jax.random.split(key)
        p_env, o = collect(cfg, net, params, CUE_NAMES[c], k, n=n_ep, T=T)
        alive = o["alive"]
        seen = np.cumsum(o["cue_vis"], axis=0) > 0        # cue observed by t
        per_cue[c] = (p_env, o)
        m = alive
        Hs.append(o["h"][m])
        Xs.append(o["state"].agent_x[m])
        Cs.append(np.full(m.sum(), c))
        Ts.append(np.broadcast_to(np.arange(T)[:, None], m.shape)[m])
        Seen.append(seen[m])
        Alive.append(m)
    ds = dict(H=np.concatenate(Hs), X=np.concatenate(Xs),
              CUE=np.concatenate(Cs), T=np.concatenate(Ts),
              SEEN=np.concatenate(Seen))

    # 5-class belief probe: trained cues + BLANK (pre-cue) class
    from sklearn.linear_model import LogisticRegression
    yb = np.where(ds["SEEN"], ds["CUE"], 99)
    sub = np.random.default_rng(0).permutation(len(yb))[:24000]
    clf5 = LogisticRegression(max_iter=3000).fit(ds["H"][sub], yb[sub])
    ds["clf5"] = clf5
    ds["classes5"] = list(clf5.classes_)          # trained cue ids + 99
    ds["P_belief"] = clf5.predict_proba(ds["H"])

    # MC outcome posteriors on stratified states: per cue x column, few states
    sel_idx = {}
    rng = np.random.default_rng(1)
    cols = list(range(int(p_env.x_room_start), int(p_env.x_doorcol)))
    MCP, MCmeta = [], []
    for c in trained:
        p_env, o = per_cue[c]
        T_, N_ = o["alive"].shape
        for x in cols:
            m = o["alive"] & (o["state"].agent_x == x) & (np.arange(T_)[:, None] < T_ - 1)
            tt, ii = np.where(m)
            if len(tt) < 4:
                continue
            pick = rng.permutation(len(tt))[:6]
            tt, ii = tt[pick], ii[pick]
            # pair h[t] with state[t+1] (post-obs hidden with the next env state)
            sel_state = jax.tree_util.tree_map(lambda a: a[tt + 1, ii], o["state"])
            h0 = o["h"][tt, ii]
            key, k = jax.random.split(key)
            Pmc, cov = mc_outcomes(cfg, net, params, p_env, sel_state, h0, k)
            keep = cov > 0.6
            MCP.append(Pmc[keep])
            for j in np.where(keep)[0]:
                MCmeta.append((c, x))
    ds["P_mc"] = np.concatenate(MCP)
    ds["mc_meta"] = np.asarray(MCmeta)            # (S, 2): cue, column
    ds["cfg"], ds["net"], ds["params"], ds["pr"] = cfg, net, params, pr
    ds["p_env"], ds["per_cue"], ds["trained"] = p_env, per_cue, trained
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# figure helpers
# ─────────────────────────────────────────────────────────────────────────────
FIGS = {}


def savefig(fig, name):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode()
    return name


def cue_color(c):
    return BLANK_COL if c == 99 else CUE_COL[c]


# ─────────────────────────────────────────────────────────────────────────────
# E1  belief / behavior / activation manifolds  (+ isometry test)
# ─────────────────────────────────────────────────────────────────────────────
def node_stats(ds):
    """Per-(cue, column) node means: activation fibers, belief and behavior
    distributions. The activation manifold is a TREE: a blank trunk that splits
    into one fiber per cue at the cue room."""
    trained, p_env = ds["trained"], ds["p_env"]
    x0, x1 = int(p_env.x_room_end), int(p_env.x_doorcol)
    fibers, Pb_nodes, Pmc_nodes = {}, {}, {}
    for c in trained:
        m = (ds["CUE"] == c) & ds["SEEN"]
        f = G.fit_fiber(ds["H"][m], ds["X"][m], list(range(x0, x1 + 1)))
        fibers[c] = f
        Pb_nodes[c] = {int(b): ds["P_belief"][m & (ds["X"] == b)].mean(0)
                       for b in f["bins"]}
        mc_m = (ds["mc_meta"][:, 0] == c)
        Pmc_nodes[c] = {int(x): ds["P_mc"][mc_m & (ds["mc_meta"][:, 1] == x)].mean(0)
                        for x in np.unique(ds["mc_meta"][mc_m, 1]) if
                        (mc_m & (ds["mc_meta"][:, 1] == x)).sum() >= 3}
    mtr = ~ds["SEEN"]
    trunk = G.fit_fiber(ds["H"][mtr], ds["X"][mtr],
                        list(range(1, int(p_env.x_room_end) + 1)))
    return fibers, trunk, Pb_nodes, Pmc_nodes


def exp_manifolds(ds, tag):
    trained = ds["trained"]
    rng = np.random.default_rng(2)
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.2))

    # (a) BELIEF manifold: inPCA of probe posteriors
    idx = rng.permutation(len(ds["H"]))[:1500]
    coords, lam, _ = G.inpca(ds["P_belief"][idx], 2)
    lab = np.where(ds["SEEN"][idx], ds["CUE"][idx], 99)
    ax = axs[0]
    for c in [99] + list(trained):
        m = lab == c
        ax.scatter(coords[m, 0], coords[m, 1], s=7, alpha=0.55, lw=0,
                   c=cue_color(c),
                   label=("blank (pre-cue)" if c == 99 else CUE_NAMES[c]))
    ax.legend(fontsize=7, framealpha=0.9, markerscale=2)
    ax.set_title("belief space — inPCA of P(cue | h)")
    ax.set_xlabel("inPC1"); ax.set_ylabel("inPC2")

    # (b) BEHAVIOR manifold: inPCA of MC outcome posteriors
    coB, lamB, _ = G.inpca(ds["P_mc"], 2)
    ax = axs[1]
    prog = ds["mc_meta"][:, 1].astype(float)
    prog = (prog - prog.min()) / max(prog.max() - prog.min(), 1)
    for c in trained:
        m = ds["mc_meta"][:, 0] == c
        ax.scatter(coB[m, 0], coB[m, 1], s=8 + 26 * prog[m], alpha=0.55, lw=0,
                   c=cue_color(c))
    ax.set_title("behavior space — inPCA of MC outcome P(branch, door | h)\n"
                 "(marker size = task progress)")
    ax.set_xlabel("inPC1"); ax.set_ylabel("inPC2")

    # (c) ACTIVATION manifold: PCA of h with trunk + fibers overlaid
    fibers, trunk, Pb_nodes, Pmc_nodes = node_stats(ds)
    sub = rng.permutation(len(ds["H"]))[:4000]
    Hc = ds["H"] - ds["H"][sub].mean(0)
    U_, S_, Vt = np.linalg.svd(Hc[sub], full_matrices=False)
    W2 = Vt[:2].T
    ax = axs[2]
    lab_all = np.where(ds["SEEN"], ds["CUE"], 99)
    for c in [99] + list(trained):
        m = lab_all[sub] == c
        pts = Hc[sub][m] @ W2
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.20, lw=0, c=cue_color(c))
    tpts = (trunk["mu"] - ds["H"][sub].mean(0)) @ W2
    ax.plot(tpts[:, 0], tpts[:, 1], "-o", c="#555", lw=2.2, ms=3.5)
    for c in trained:
        fpts = (fibers[c]["mu"] - ds["H"][sub].mean(0)) @ W2
        ax.plot(fpts[:, 0], fpts[:, 1], "-o", c=cue_color(c), lw=2.2, ms=3.5)
    ax.set_title("activation space — PCA, trunk + cue fibers\n"
                 "(lines = per-column centroids)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    savefig(fig, f"{tag}_manifolds")

    # isometry: tree geodesics vs belief / behavior distances
    nodes, Dg = G.tree_geodesics(trunk, fibers, int(ds["p_env"].x_room_end))
    idx_b, idx_m, Pb, Pm = [], [], [], []
    for i, (c, b) in enumerate(nodes):
        if b in Pb_nodes[c]:
            idx_b.append(i); Pb.append(Pb_nodes[c][b])
        if b in Pmc_nodes[c]:
            idx_m.append(i); Pm.append(Pmc_nodes[c][b])
    Pb, Pm = np.stack(Pb), np.stack(Pm)
    Hb = G.hellinger_map(Pb); Hm = G.hellinger_map(Pm)
    Db = np.linalg.norm(Hb[:, None] - Hb[None], axis=-1)
    Dm = np.linalg.norm(Hm[:, None] - Hm[None], axis=-1)
    stats = {}
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for ax, (ids, Dd, nm) in zip(axs, [(idx_b, Db, "belief"), (idx_m, Dm, "behavior")]):
        sel = np.asarray(ids)
        Dgs = Dg[np.ix_(sel, sel)]
        same = np.array([[nodes[i][0] == nodes[j][0] for j in sel] for i in sel])
        iu = np.triu_indices(len(sel), 1)
        r = float(np.corrcoef(Dgs[iu], Dd[iu])[0, 1])
        ax.scatter(Dgs[iu][same[iu]], Dd[iu][same[iu]], s=7, alpha=0.4,
                   c="#777", lw=0, label="same fiber")
        ax.scatter(Dgs[iu][~same[iu]], Dd[iu][~same[iu]], s=7, alpha=0.4,
                   c="#c0392b", lw=0, label="across fibers")
        ax.set_xlabel("activation-tree geodesic distance")
        ax.set_ylabel(f"{nm} distance (Hellinger)")
        ax.set_title(f"{nm}:  r = {r:.2f}")
        ax.legend(fontsize=7)
        stats[f"iso_r_{nm}"] = r
    savefig(fig, f"{tag}_isometry")
    stats["inpca_lam_belief"] = [float(v) for v in lam[:4]]
    stats["inpca_lam_behavior"] = [float(v) for v in lamB[:4]]
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# E3 + E6  linear field probes over progress; additivity
# ─────────────────────────────────────────────────────────────────────────────
def exp_lfp(ds, tag):
    p_env, trained = ds["p_env"], ds["trained"]
    m = ds["SEEN"]
    bins = list(range(int(p_env.x_room_start), int(p_env.x_doorcol)))
    field = G.lfp_fit(ds["H"][m], ds["CUE"][m], ds["X"][m], bins)
    K = G.lfp_gram(field)
    T = G.lfp_transfer(ds["H"][m], ds["CUE"][m], ds["X"][m], field)
    theta = float(np.degrees(np.arccos(np.clip(K[0, -1], -1, 1))))

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 3.9))
    im = axs[0].imshow(K, cmap="viridis", vmin=0, vmax=1, origin="lower",
                       extent=[field["bins"][0], field["bins"][-1]] * 2)
    plt.colorbar(im, ax=axs[0], fraction=0.046)
    axs[0].set_title("LFP Gram: cos-sim of cue probes across columns")
    axs[0].set_xlabel("column x"); axs[0].set_ylabel("column x")
    im = axs[1].imshow(T, cmap="magma", vmin=1 / len(trained), vmax=1, origin="lower",
                       extent=[field["bins"][0], field["bins"][-1]] * 2)
    plt.colorbar(im, ax=axs[1], fraction=0.046)
    axs[1].set_title("probe transfer accuracy (train row → test col)")
    axs[1].set_xlabel("test column"); axs[1].set_ylabel("train column")
    axs[2].plot(field["bins"], field["acc"], "-o", ms=4, c="#333")
    axs[2].axhline(1 / len(trained), ls=":", c="#999")
    axs[2].set_ylim(0, 1.02)
    axs[2].set_title("in-column probe accuracy (3-fold CV)")
    axs[2].set_xlabel("column x")
    savefig(fig, f"{tag}_lfp")

    # additivity of the (cue x column) centroid grid
    fibers, _, _, _ = node_stats(ds)
    common = sorted(set.intersection(*[set(f["bins"].tolist()) for f in fibers.values()]))
    M = np.stack([np.stack([fibers[c]["mu"][list(fibers[c]["bins"]).index(b)]
                            for b in common]) for c in trained])
    r2, inter = G.additive_r2(M)
    return dict(gram_min=float(K.min()), gram_end2end=float(K[0, -1]),
                rot_deg=theta, transfer_mean=float(np.nanmean(T)),
                additive_r2=r2, interaction=inter, acc_mean=float(field["acc"].mean()),
                field=field)


# ─────────────────────────────────────────────────────────────────────────────
# E4  global vs field-aware transient swap
# ─────────────────────────────────────────────────────────────────────────────
def exp_field_swap(ds, tag, n=96):
    cfg, net, params, p_env = ds["cfg"], ds["net"], ds["params"], ds["p_env"]
    trained = ds["trained"]
    Dh = ds["H"].shape[1]
    x0, x1 = int(p_env.x_room_end) + 1, int(p_env.x_doorcol)
    W_env = int(p_env.width)
    mu_x = {c: {} for c in trained}
    for c in trained:
        m = (ds["CUE"] == c) & ds["SEEN"]
        for x in range(x0, x1 + 1):
            mm = m & (ds["X"] == x)
            if mm.sum() >= 5:
                mu_x[c][x] = ds["H"][mm].mean(0)
    mu_glob = {c: ds["H"][(ds["CUE"] == c) & ds["SEEN"]].mean(0) for c in trained}
    tgt_idx = {c: ds["classes5"].index(c) for c in trained}
    key = jax.random.PRNGKey(9)
    rows = []
    for s_ in trained:
        for t_ in trained:
            if s_ == t_:
                continue
            # global axis
            u = mu_glob[t_] - mu_glob[s_]
            u = u / np.linalg.norm(u)
            pt = float(mu_glob[t_] @ u)
            # per-column field, spline-smoothed, nearest-fill outside data
            cols = sorted(set(mu_x[s_]) & set(mu_x[t_]))
            Uc = np.stack([mu_x[t_][x] - mu_x[s_][x] for x in cols])
            Uc = Uc / np.linalg.norm(Uc, axis=1, keepdims=True)
            Uc = G.smooth_field(cols, Uc)
            Uc = Uc / np.linalg.norm(Uc, axis=1, keepdims=True)
            PTc = np.array([mu_x[t_][x] @ Uc[i] for i, x in enumerate(cols)])
            UF = np.zeros((W_env, Dh), np.float32)
            PF = np.zeros((W_env,), np.float32)
            for x in range(W_env):
                j = int(np.argmin([abs(x - c2) for c2 in cols]))
                UF[x], PF[x] = Uc[j], PTc[j]
            res = {}
            for modename, kw in [("global", dict(mode="swap", u=u, ptgt=pt)),
                                 ("field", dict(mode="swap_field", u_field=UF,
                                                ptgt_field=PF))]:
                key, k = jax.random.split(key)
                mode = kw.pop("mode")
                _, (succ, tb, sd, dacc, fin, hend), outs = S.rollout(
                    cfg, net, params, CUE_NAMES[s_], k, n, mode,
                    sample=True, T=None, **kw)
                b_t = C.BRANCH_DOWN if S.IS_DOWN[t_] else C.BRANCH_UP
                d_t = C.SEL_BLUE if S.IS_BLUE[t_] else C.SEL_GREEN
                beh = float(((tb == b_t) & (sd == d_t)).mean())
                # path energy: probe posterior vs target during the window
                hs, win = outs[4], outs[5]
                hw = hs[win.astype(bool)]
                if len(hw):
                    Pw = ds["clf5"].predict_proba(hw.reshape(-1, Dh))
                    tvec = np.zeros(len(ds["classes5"])); tvec[tgt_idx[t_]] = 1
                    en, _ = G.path_energy(Pw, tvec)
                else:
                    en = float("nan")
                res[modename] = (beh, en)
            rows.append(dict(src=CUE_NAMES[s_], tgt=CUE_NAMES[t_],
                             beh_global=res["global"][0], en_global=res["global"][1],
                             beh_field=res["field"][0], en_field=res["field"][1]))
            print(f"[swap] {tag} {CUE_NAMES[s_]}->{CUE_NAMES[t_]}: "
                  f"global {res['global'][0]:.2f} (E={res['global'][1]:.2f})  "
                  f"field {res['field'][0]:.2f} (E={res['field'][1]:.2f})", flush=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# E5  attractor census
# ─────────────────────────────────────────────────────────────────────────────
def exp_attractors(ds, tag):
    cfg, net, params, p_env = ds["cfg"], ds["net"], ds["params"], ds["p_env"]
    trained = ds["trained"]
    c0 = trained[0]
    st = make_state(p_env, c0, True, p_env.x_room_start, p_env.row_room_up)
    st = st.replace(agent_x=jnp.int32(p_env.x_pre_end - 2),
                    agent_y=jnp.int32(p_env.my), agent_dir=jnp.int32(C.DIR_EAST))
    obs = D._flat({k: v[None] for k, v in build_obs(st, p_env).items()})[0]
    obs_j = jnp.asarray(obs)

    def step_fn(h):
        ob = jnp.broadcast_to(obs_j, (1, h.shape[0], obs_j.shape[-1]))
        nh, _, _ = net.apply(params, h, (ob, jnp.zeros((1, h.shape[0]), bool)))
        return nh

    rng = np.random.default_rng(3)
    base = ds["H"][rng.permutation(len(ds["H"]))[:220]]
    inits = np.concatenate([base, base + rng.normal(0, 2.0, base.shape)])
    fps = G.find_fixed_points(step_fn, inits, iters=1200, tol=5e-3, merge_tol=2.0)
    mu_glob = {c: ds["H"][(ds["CUE"] == c) & ds["SEEN"]].mean(0) for c in trained}
    mu_glob[99] = ds["H"][~ds["SEEN"]].mean(0)
    table = []
    for fp in fps:
        dists = {c: float(np.linalg.norm(fp["h"] - mu)) for c, mu in mu_glob.items()}
        near = min(dists, key=dists.get)
        table.append(dict(near=("blank" if near == 99 else CUE_NAMES[near]),
                          dist=dists[near], eig=fp["eig_max"],
                          stable=fp["stable"], n=fp["n_merged"]))
    # overlay on belief-field plane
    field = S.belief_field(cfg, net, params, ds["pr"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    S.draw_belief_field(ax, field, ds["pr"]["trained"])
    for fp in fps:
        x_, y_ = fp["h"] @ field["e1"], fp["h"] @ field["e2"]
        ax.scatter([x_], [y_], marker="*", s=340,
                   c=("#111" if fp["stable"] else "none"),
                   edgecolor="#111", lw=1.4, zorder=8)
    ax.set_title(f"{tag}: fixed points of the GRU (★ filled = stable)")
    ax.set_xlabel(field["axlab"][0]); ax.set_ylabel(field["axlab"][1])
    savefig(fig, f"{tag}_attractors")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# E2  behavior forcing -> belief revision (evidence weights vs Bayes)
# ─────────────────────────────────────────────────────────────────────────────
def exp_evidence(ds, tag, n=48):
    from forced_evidence_ppo import pre_door_index, mark_open_index
    cfg, net, params = ds["cfg"], ds["net"], ds["params"]
    trained, Dh = ds["trained"], ds["H"].shape[1]
    key = jax.random.PRNGKey(13)
    ncol = len(trained)
    fig, axs = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.0), squeeze=False,
                            sharey=True)
    rows = []
    W_PRE, W_POST = 8, 14
    rel = np.arange(-W_PRE, W_POST + 1)
    for col, c in enumerate(trained):
        cue = CUE_NAMES[c]
        p_env = D._env_params(cfg, cue)
        wrong_is_down = not S.IS_DOWN[c]
        wrong_row = int(p_env.row_lo) if wrong_is_down else int(p_env.row_up)
        d_ok = C.SEL_BLUE if S.IS_BLUE[c] else C.SEL_GREEN
        key, k = jax.random.split(key)
        _, (succ, tb, sd, dacc, fin, hend), outs = S.rollout(
            cfg, net, params, cue, k, n, "force_thru", wrong_row=wrong_row,
            sample=True, T=None)
        xs, ys, dss_, mms, hs, win, dnn, reach, mtop, mbot = outs
        Pr = ds["clf5"].predict_proba(hs.reshape(-1, Dh)).reshape(
            hs.shape[0], hs.shape[1], -1)
        idx = pre_door_index(mms, dnn)
        P_pre = Pr[idx, np.arange(n)]
        t_open = mark_open_index(mms, wrong_is_down)
        ax = axs[0, col]
        for i, cl in enumerate(ds["classes5"]):
            curves = []
            for e in range(n):
                if t_open[e] < 0:
                    continue
                ts = t_open[e] + rel
                ok2 = (ts >= 0) & (ts < Pr.shape[0])
                v = np.full(rel.shape, np.nan)
                v[ok2] = Pr[ts[ok2], e, i]
                curves.append(v)
            if curves:
                mcurve = np.nanmean(np.stack(curves), 0)
                nm = "blank" if cl == 99 else CUE_NAMES[cl]
                ax.plot(rel, mcurve, lw=2, color=cue_color(cl), label=f"P({nm})",
                        ls=(":" if cl == 99 else "-"))
        ax.axvline(0, color="#8e44ad", lw=1.3, alpha=0.85)
        ax.set_ylim(-0.04, 1.06)
        dok = float((sd == d_ok).mean())
        ax.set_title(f"true {cue} · door-ok {dok:.2f}", fontsize=9)
        ax.set_xlabel("steps from wrong-marker open", fontsize=8)
        if col == 0:
            ax.set_ylabel("P(cue)")
            ax.legend(fontsize=6, loc="center left")
        # destination + implied log-likelihood-ratio of the evidence
        dest = {("blank" if cl == 99 else CUE_NAMES[cl]): float(P_pre[:, i].mean())
                for i, cl in enumerate(ds["classes5"])}
        p_true = dest[cue]
        top = max((v, k2) for k2, v in dest.items() if k2 != cue)
        llr = float(np.log(np.clip(top[0], 1e-6, None) / np.clip(p_true, 1e-6, None)))
        rows.append(dict(cue=cue, door_ok=dok, dest=top[1], p_dest=top[0],
                         p_true=p_true, llr=llr))
    fig.suptitle(f"{tag}: action-forced into the WRONG corridor — belief at the "
                 "marker evidence (5-class probe)", fontsize=10, fontweight="bold")
    savefig(fig, f"{tag}_evidence")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# pre-cue bias: diagnosis + fix
# ─────────────────────────────────────────────────────────────────────────────
def exp_precue(ds, tag):
    from sklearn.linear_model import LogisticRegression
    trained, Dh = ds["trained"], ds["H"].shape[1]
    m = ds["SEEN"]
    sub = np.random.default_rng(4).permutation(m.sum())[:20000]
    clf4 = LogisticRegression(max_iter=3000).fit(ds["H"][m][sub], ds["CUE"][m][sub])
    # one greedy baseline episode (first trained cue)
    cfg, net, params = ds["cfg"], ds["net"], ds["params"]
    cue = CUE_NAMES[trained[0]]
    _, (succ, tb, sd, dacc, fin, hend), outs = S.rollout(
        cfg, net, params, cue, jax.random.PRNGKey(21), 1, "none", T=110)
    hs, dnn, mms = np.asarray(outs[4])[:, 0], np.asarray(outs[6])[:, 0], np.asarray(outs[3])[:, 0]
    nd = int(np.argmax(dnn)) + 1 if dnn.any() else len(hs)
    hs = hs[:nd]
    cue_vis = ((mms[:nd] >= C.CUE_GREEN_UP) & (mms[:nd] <= C.CUE_BLUE_DOWN)).any((-2, -1))
    t_seen = int(np.argmax(np.cumsum(cue_vis) > 0))
    P4 = clf4.predict_proba(hs)
    P5 = ds["clf5"].predict_proba(hs)
    # off-manifold distance: min distance to any (trunk/fiber) node centroid
    fibers, trunk, _, _ = node_stats(ds)
    nodes = np.concatenate([trunk["mu"]] + [f["mu"] for f in fibers.values()])
    dmin = np.linalg.norm(hs[:, None] - nodes[None], axis=-1).min(1)
    scale = np.median(np.linalg.norm(
        ds["H"][m][sub[:3000]][:, None] - nodes[None], axis=-1).min(1))

    fig, axs = plt.subplots(1, 3, figsize=(13.4, 3.2))
    for i, c in enumerate(clf4.classes_):
        axs[0].plot(P4[:, i], lw=2, color=cue_color(c), label=f"P({CUE_NAMES[c]})")
    axs[0].set_title("4-class probe: confident BEFORE the cue (artifact)")
    for i, cl in enumerate(ds["classes5"]):
        nm = "blank" if cl == 99 else CUE_NAMES[cl]
        axs[1].plot(P5[:, i], lw=2, color=cue_color(cl),
                    ls=(":" if cl == 99 else "-"), label=f"P({nm})")
    axs[1].set_title("5-class probe (blank class): calibrated pre-cue")
    for ax in axs[:2]:
        ax.axvline(t_seen, color="#333", lw=1, ls="--")
        ax.text(t_seen + 0.4, 1.02, "cue visible", fontsize=7)
        ax.set_ylim(-0.04, 1.1); ax.set_xlabel("timestep")
        ax.legend(fontsize=6.5)
    axs[2].plot(dmin / scale, lw=2, c="#333")
    axs[2].axvline(t_seen, color="#333", lw=1, ls="--")
    axs[2].axhline(1.0, ls=":", c="#999")
    axs[2].set_title("distance to the fitted manifold (÷ on-manifold median)")
    axs[2].set_xlabel("timestep")
    savefig(fig, f"{tag}_precue")
    pre4 = float(P4[:max(t_seen, 1)].max())
    pre5_blank = float(P5[:max(t_seen, 1), ds["classes5"].index(99)].mean())
    return dict(pre4_max=pre4, pre5_blank=pre5_blank)


# ─────────────────────────────────────────────────────────────────────────────
# method diagram
# ─────────────────────────────────────────────────────────────────────────────
def method_diagram():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 3.6))
    # (a) task timeline
    ax = axs[0]
    segs = [("blank", 0, 2, BLANK_COL), ("cue", 2, 3.2, "#1b9e77"),
            ("memory", 3.2, 6.2, "#cccccc"), ("branch+marker", 6.2, 8.2, "#8e44ad"),
            ("memory", 8.2, 10.6, "#cccccc"), ("doors", 10.6, 12, "#3b6fb6")]
    for nm, a, b, col in segs:
        ax.barh(0, b - a, left=a, height=0.5, color=col, alpha=0.75)
        ax.text((a + b) / 2, 0.45, nm, ha="center", fontsize=8)
    ax.text(2.6, -0.55, "belief WRITTEN", ha="center", fontsize=8, color="#1b9e77")
    ax.text(7.2, -0.55, "direction bit READ\n+ evidence", ha="center", fontsize=8,
            color="#8e44ad")
    ax.text(11.3, -0.55, "color bit READ", ha="center", fontsize=8, color="#3b6fb6")
    ax.set_xlim(-0.2, 12.4); ax.set_ylim(-1, 1); ax.axis("off")
    ax.set_title("(a) one episode: write, carry, and spend a 2-bit memory")
    # (b) trunk + fibers cartoon
    ax = axs[1]
    t_ = np.linspace(0, 1, 50)
    ax.plot(t_ * 3, 0 * t_, c=BLANK_COL, lw=3)
    for i, col in enumerate(CUE_COL):
        yy = (i - 1.5) * (t_ ** 0.8)
        ax.plot(3 + t_ * 6, yy, c=col, lw=3)
    ax.annotate("blank trunk", (1.4, 0.12), fontsize=8, color="#555")
    ax.annotate("cue splits the state\n(belief = WHICH fiber)", (4.1, 1.35),
                fontsize=8)
    ax.annotate("progress x\n(WHERE along the fiber)", (7.1, -2.1), fontsize=8)
    ax.axis("off")
    ax.set_title("(b) activation manifold: a tree of fibers")
    # (c) pipeline
    ax = axs[2]
    boxes = [("hidden state h", 0.06, 0.62), ("probe P(cue|h)", 0.42, 0.80),
             ("MC rollouts\nP(outcome|h)", 0.42, 0.42),
             ("inPCA\nbelief space", 0.78, 0.80), ("inPCA\nbehavior space", 0.78, 0.42)]
    for nm, x, y in boxes:
        ax.add_patch(FancyBboxPatch((x - 0.05, y - 0.1), 0.24, 0.2,
                                    boxstyle="round,pad=0.02",
                                    fc="#f4f4f6", ec="#666"))
        ax.text(x + 0.07, y, nm, ha="center", va="center", fontsize=8)
    for (x0, y0), (x1, y1) in [((0.25, 0.66), (0.37, 0.80)),
                               ((0.25, 0.58), (0.37, 0.44)),
                               ((0.61, 0.80), (0.73, 0.80)),
                               ((0.61, 0.42), (0.73, 0.42))]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=13, color="#444"))
    ax.text(0.5, 0.12, "same pipeline for ANY env:\nonly the labels/outcomes change",
            ha="center", fontsize=8, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(c) constructing the two spaces")
    savefig(fig, "method_diagram")


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
body{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:#1a1a1a;
 max-width:980px;margin:2em auto;padding:0 1.4em;line-height:1.55;background:#fff}
h1{font-size:1.9em;letter-spacing:-.02em;margin-bottom:.2em}
h2{font-size:1.35em;margin-top:2.2em;border-bottom:1px solid #e5e5e5;padding-bottom:.25em}
h3{font-size:1.05em;margin-top:1.4em}
p.lead{color:#444;font-size:1.06em}
.method{background:#f7f7f9;border-left:3px solid #8e44ad;padding:.7em 1em;
 border-radius:0 6px 6px 0;font-size:.94em;color:#333;margin:1em 0}
.finding{background:#f2faf5;border-left:3px solid #1b9e77;padding:.7em 1em;
 border-radius:0 6px 6px 0;font-size:.94em;margin:1em 0}
img{max-width:100%;border:1px solid #eee;border-radius:8px;margin:.6em 0;
 box-shadow:0 1px 4px rgba(0,0,0,.06)}
table{border-collapse:collapse;font-size:.88em;margin:.8em 0}
th,td{border:1px solid #e2e2e2;padding:.32em .6em;text-align:right}
th{background:#f4f4f6}
td:first-child,th:first-child{text-align:left}
.good{color:#1a7d36;font-weight:600}.bad{color:#b02418;font-weight:600}
.tag{display:inline-block;background:#eee;border-radius:4px;padding:0 .45em;
 font-size:.82em;margin-right:.4em}
"""


def img(name):
    return f'<img src="data:image/png;base64,{FIGS[name]}"/>'


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td>{fmt.get(c, lambda v: v if isinstance(v, str) else f'{v:.2f}')(r[c])}</td>"
            for c in cols) + "</tr>"
    return f"<table>{h}{body}</table>"


def render_html(tags, R, out):
    S_ = []
    S_.append(f"""<html><head><meta charset='utf-8'>
<title>Neural geometry of RL beliefs — MemoryEnv</title><style>{CSS}</style></head><body>
<h1>Neural geometry of an RL agent's beliefs, decisions and behavior</h1>
<p class='lead'>Three PPO+GRU agents solve a T-maze whose single cue carries two
bits (direction &rarr; branch, color &rarr; door) that are spent at different
times. We construct a <b>belief space</b> (the probe posterior over cues, on the
probability simplex) and a <b>behavior space</b> (the Monte-Carlo outcome
distribution over branch&times;door), embed both with <b>inPCA</b>, fit the
<b>activation manifold</b> (a trunk-and-fibers tree over task progress), and then
steer both ways: editing the belief changes behavior, and forcing behavior
changes the belief through experienced evidence. All geometry code is
environment-agnostic (<code>neurogeom.py</code>) — the same figures can be
produced for any navigation+decision environment (e.g. bridge_tunnel) by
swapping the data-collection layer.</p>
<h2>0 · Method</h2>
<div class='method'><b>Pipeline.</b> (1) Roll the stochastic policy, keep every
hidden state h with its position x and cue label. (2) <b>Belief space</b>: a
multinomial probe P(cue | h) trained with an extra <i>blank</i> class from
pre-cue states; points live on the simplex and are embedded with inPCA
(double-centered Bhattacharyya affinities; signed spectrum). (3) <b>Behavior
space</b>: from stratified states, continue the policy K=24 times and record
the outcome (branch&times;door) distribution; embed with inPCA. (4)
<b>Activation manifold</b>: per-column centroids form one polyline fiber per
cue plus a pre-cue trunk — a tree parameterized by progress. (5) <b>Linear
field probes</b>: one probe per column tiles the manifold and tests whether the
memory code rotates. (6) Interventions: transient class-mean swaps (global vs
<i>field-aware</i> local axes) and action-forcing through the wrong corridor.</div>
{img('method_diagram')}""")

    S_.append("<h2>1 · Belief space, behavior space, activation manifold</h2>")
    for t in tags:
        r = R[t]
        S_.append(f"<h3>{t} model</h3>{img(t + '_manifolds')}")
        S_.append(f"""<div class='finding'>Isometry of the activation tree with the two
distribution spaces — behavior r = <b>{r['manifolds']['iso_r_behavior']:.2f}</b>,
belief r = <b>{r['manifolds']['iso_r_belief']:.2f}</b>. Belief distance is a step
function of the tree (fibers = belief classes, saturating immediately after the
split), while behavior distance grows with progress as decisions get spent —
the geometric signature of belief vs behavior.</div>{img(t + '_isometry')}""")

    S_.append("""<h2>2 · Does the memory code rotate? Linear field probes</h2>
<div class='method'>One multinomial cue probe per corridor column (train 3-fold CV
in-column). The Gram matrix of probe weights and the cross-column transfer
accuracy tell whether one global belief axis exists (stationary code) or the
code rotates with position (then any single steering axis must fail somewhere).</div>""")
    for t in tags:
        r = R[t]["lfp"]
        S_.append(f"""<h3>{t}</h3>{img(t + '_lfp')}
<div class='finding'>end-to-end probe rotation <b>{r['rot_deg']:.0f}&deg;</b>
(Gram[first,last] = {r['gram_end2end']:.2f}); mean transfer accuracy
{r['transfer_mean']:.2f}; additive model mu(cue, x) = a<sub>cue</sub> + b<sub>x</sub>
explains <b>R&sup2; = {r['additive_r2']:.2f}</b> of the centroid grid
(interaction = {r['interaction']:.2f}).</div>""")

    S_.append("""<h2>3 · Steering belief &rarr; behavior (global vs field-aware swap)</h2>
<div class='method'>Transient clamp of the GRU carry between cue room and branch
decision, toward the target cue's representation. <b>Global</b>: one class-mean
axis. <b>Field-aware</b>: the axis and target are position-local
(spline-smoothed per-column class means), following the manifold instead of a
chord. Energy = mean Bhattacharyya distance of the probe posterior to the
target vertex during the intervention (lower = more natural path).</div>""")
    for t in tags:
        rows = R[t]["swap"]
        cols = ["src", "tgt", "beh_global", "beh_field", "en_global", "en_field"]
        hi = {"beh_global": lambda v: f"<span class='{'good' if v > .9 else 'bad'}'>{v:.2f}</span>",
              "beh_field": lambda v: f"<span class='{'good' if v > .9 else 'bad'}'>{v:.2f}</span>"}
        S_.append(f"<h3>{t}</h3>" + table(rows, cols, hi))

    S_.append("""<h2>4 · Steering behavior &rarr; belief (evidence, not edits)</h2>
<div class='method'>Pure action replacement forces the agent through the WRONG
corridor and its marker door; the hidden state is never touched. Curves are
event-aligned at the marker opening (5-class probe). The implied evidence
weight is the log-likelihood-ratio LLR = log P(destination)/P(true cue) at the
pre-door readout: a symmetric Bayes observer facing contradictory cue-vs-marker
evidence would sit at LLR = 0.</div>""")
    for t in tags:
        S_.append(f"<h3>{t}</h3>{img(t + '_evidence')}" +
                  table(R[t]["evidence"],
                        ["cue", "door_ok", "dest", "p_dest", "p_true", "llr"]))
    S_.append("""<div class='finding'>2cue collapses to the other attractor
(door follows the implanted color); 4cue revises only the direction factor
(door survives). LLR &gt; 0 everywhere: recent corridor evidence dominates the
remembered cue — the agents are recency-weighted, sharper-than-Bayes filters
under the training correlations.</div>""")

    S_.append("""<h2>5 · Attractor census</h2>
<div class='method'>Fixed points of the GRU step under a neutral maintenance
observation, found by gradient descent from 440 data-adjacent inits; stability
from the Jacobian spectrum. Overlaid on the belief plane (probe decision
regions + one-step flow).</div>""")
    for t in tags:
        S_.append(f"<h3>{t}</h3>{img(t + '_attractors')}" +
                  table(R[t]["attractors"],
                        ["near", "dist", "eig", "stable", "n"],
                        {"stable": lambda v: "yes" if v else "no",
                         "n": lambda v: str(v)}))

    S_.append("""<h2>6 · The pre-cue bias: diagnosis and fix</h2>
<div class='method'>Concern: probes report P(one cue) &asymp; 1 BEFORE the cue is
visible. Diagnosis: (i) a probe fit only on post-cue states must extrapolate on
pre-cue states, which lie OFF the post-cue manifold — logistic extrapolation is
confident by construction; (ii) the pre-cue trunk genuinely sits closer to one
fiber (the network's default basin). Fix, no retraining needed: add a
<i>blank</i> class of pre-cue states to the probe, and report the
distance-to-manifold alongside. Beliefs are then calibrated: blank &rarr; cue
switches exactly at cue onset.</div>""")
    for t in tags:
        r = R[t]["precue"]
        S_.append(f"""<h3>{t}</h3>{img(t + '_precue')}
<div class='finding'>pre-cue max P(cue) = {r['pre4_max']:.2f} under the 4-class
probe (artifact) vs mean P(blank) = {r['pre5_blank']:.2f} under the 5-class
probe (fix). A retraining-based fix is neither needed nor desirable: forcing a
calibrated pre-cue belief would require auxiliary supervision on the
representation, which would break the "everything is emergent" design.</div>""")

    S_.append("</body></html>")
    pathlib.Path(out).write_text("\n".join(S_))
    print(f"[report] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/geometry_report.html")
    ap.add_argument("--n-ep", type=int, default=192)
    ap.add_argument("--swap-n", type=int, default=96)
    a = ap.parse_args()

    method_diagram()
    tags, R = [], {}
    for i, rd in enumerate(a.run_dirs):
        cfg = json.loads((pathlib.Path(rd) / "config.json").read_text())
        tag = cfg["cue"]
        tags.append(tag)
        print(f"== [{tag}] collecting", flush=True)
        ds = build_dataset(rd, key=100 + i, n_ep=a.n_ep)
        R[tag] = {}
        print(f"== [{tag}] E1 manifolds + isometry", flush=True)
        R[tag]["manifolds"] = exp_manifolds(ds, tag)
        print(f"== [{tag}] E3/E6 LFP + additivity", flush=True)
        R[tag]["lfp"] = exp_lfp(ds, tag)
        print(f"== [{tag}] E4 swaps (global vs field)", flush=True)
        R[tag]["swap"] = exp_field_swap(ds, tag, n=a.swap_n)
        print(f"== [{tag}] E5 attractors", flush=True)
        R[tag]["attractors"] = exp_attractors(ds, tag)
        print(f"== [{tag}] E2 evidence", flush=True)
        R[tag]["evidence"] = exp_evidence(ds, tag)
        print(f"== [{tag}] pre-cue diagnosis", flush=True)
        R[tag]["precue"] = exp_precue(ds, tag)

    outdir = pathlib.Path("outputs/report_geometry")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, b64 in FIGS.items():
        (outdir / f"{name}.png").write_bytes(base64.b64decode(b64))
    render_html(tags, R, a.out)


if __name__ == "__main__":
    main()


