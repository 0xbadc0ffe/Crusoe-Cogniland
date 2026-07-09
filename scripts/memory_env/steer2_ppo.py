#!/usr/bin/env python
"""Steering experiments v2 on the PPO+GRU MemoryEnv models (2cue/3cue/4cue).

Two interventions, opposite directions of causation:
  A. BEHAVIOR steering ("force_act"): at the fork, the GRU hidden is minimally
     perturbed (iterative normalized gradient ascent on the actor head's logit
     margin) until the ACTOR ITSELF selects the action that leads into the
     direction-WRONG branch; the perturbed hidden is carried forward. The
     intervention stops once the wrong branch is entered. Readout: does the
     BELIEF survive a behavior-targeted activation edit, and does the agent
     still pick the color-correct door?
     ("force" = the older action-replacement variant is kept for comparison.)
  B. BELIEF steering ("swap"): TRANSIENT memory swap — only between the cue
     room and the direction decision, clamp the GRU hidden along the axis
     between two TRAINED cue representations (class means); release at branch
     entry. Readout: does the implanted memory PERSIST unaided and drive the
     later door choice — i.e. does the agent behave as the other cue end-to-end?

All probes / axes / landmarks are fit on TRAINED-cue samples only (post-cue
phase), per model. Belief-over-time = multinomial P(cue = c) over the model's
training cues.

Subcommands:
  quant --run-dir <dir>          n-episode tables for both experiments
  video --run-dir <dir> --out f  baseline / behavior-steered / swapped episodes as MP4
  fig   --run-dir <dir> --out f  the same three episodes as a static report figure
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from sklearn.linear_model import LogisticRegression

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, constants as C,
)
import diag_jax as D  # noqa: E402
import train_ppo_memory as P  # noqa: E402

CUE_NAMES = ["green_up", "blue_up", "green_down", "blue_down"]
CUE_COL = ["#1b9e77", "#3b6fb6", "#7fd4b8", "#9ec9ec"]
IS_DOWN = np.asarray(C.CUE_IS_DOWN)
IS_BLUE = np.asarray(C.CUE_IS_BLUE)


# ─────────────────────────────────────────────────────────────────────────────
# model + trained-cues-only probes
# ─────────────────────────────────────────────────────────────────────────────
def load_all(run_dir):
    rd = pathlib.Path(run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    params = ocp.PyTreeCheckpointer().restore(
        str(sorted((rd / "checkpoints").glob("step_*"))[-1].resolve()))["params"]
    net = P.ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                           token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                           gru_hidden=cfg["gru_hidden"])
    trained = sorted(CUE_NAMES.index(c) for c in D.TRAIN_CUES[cfg["cue"]])

    d = np.load(rd / "activations.npz", allow_pickle=True)
    X = d["feat"].astype(np.float64)
    m = np.isin(d["cue_type"], trained) & (d["phase"] >= 2)   # TRAINED cues, post-cue
    Xt, ct = X[m], d["cue_type"][m]
    # multinomial cue-identity probe over the training cues only
    clf = LogisticRegression(max_iter=4000).fit(Xt, ct)
    # class means (the cue representations in memory) + direction/color axes
    mu = {c: Xt[ct == c].mean(0) for c in trained}

    def _axis(labels):
        """Unit-normalized binary-probe direction, or None if only one class."""
        if len(set(labels)) < 2:
            return None
        w = LogisticRegression(max_iter=4000).fit(Xt, labels).coef_[0]
        return w / np.linalg.norm(w)

    probes = dict(clf=clf, mu=mu, w_dir=_axis(IS_DOWN[ct]), w_col=_axis(IS_BLUE[ct]),
                  trained=trained, Xt=Xt, ct=ct)
    return cfg, net, params, probes


def belief_plane(pr):
    """Trained-cues-only belief-plane axes, labels, landmarks and limits.

    Default axes = the direction/color probe directions. Special case (2cue):
    those coincide (its training cues confound the two features, so the two
    probes solve the same problem and land on the same direction) — detected by
    collinearity — and the plane falls back to the class-mean difference as
    axis 1 and the top residual PC as axis 2.
    """
    trained = pr["trained"]
    w_dir, w_col = pr["w_dir"], pr["w_col"]
    degenerate = (w_dir is None or w_col is None
                  or abs(float(w_dir @ w_col)) > 0.99)
    if not degenerate:
        ax1, ax2 = w_dir, w_col
        axlab = ("direction axis (trained-fit)", "color axis (trained-fit)")
    else:
        mus = [pr["mu"][c] for c in trained]
        ax1 = mus[1] - mus[0]
        ax1 = ax1 / np.linalg.norm(ax1)
        resid = pr["Xt"] - pr["Xt"].mean(0)
        resid = resid - np.outer(resid @ ax1, ax1)
        _, _, Vt = np.linalg.svd(resid, full_matrices=False)
        ax2 = Vt[0]
        axlab = ("trained-cue axis (mu2 - mu1)", "residual PC1")
    land = {c: (float(pr["mu"][c] @ ax1), float(pr["mu"][c] @ ax2)) for c in trained}
    proj = np.stack([pr["Xt"] @ ax1, pr["Xt"] @ ax2], 1)
    lims = ((proj[:, 0].min() - 1, proj[:, 0].max() + 1),
            (proj[:, 1].min() - 1, proj[:, 1].max() + 1))
    return ax1, ax2, axlab, land, lims


def cue_probs(h, clf):
    """Multinomial P(cue = c) for hidden states h (N, D) -> (N, K)."""
    z = h @ clf.coef_.T + clf.intercept_
    if z.ndim == 1 or z.shape[-1] == 1:      # binary sklearn: single logit
        p1 = 1 / (1 + np.exp(-z.reshape(-1)))
        return np.stack([1 - p1, p1], 1)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# interventions
# ─────────────────────────────────────────────────────────────────────────────
def forced_action(state, target_row):
    """Scripted controller: drive to target branch row, then east into the branch."""
    y, dref = state.agent_y, state.agent_dir
    desired = jnp.where(y != target_row,
                        jnp.where(target_row < y, C.DIR_NORTH, C.DIR_SOUTH),
                        C.DIR_EAST)
    diff = (desired - dref) % 4
    return jnp.where(diff == 0, C.A_FORWARD,
                     jnp.where(diff == 3, C.A_LEFT, C.A_RIGHT)).astype(jnp.int32)


def actor_head_fns(params):
    """Extract the actor head (hidden -> logits) as a pure function of the GRU state.

    ActorCriticRNN creates Dense layers in order: Dense_0 (obs trunk), Dense_1
    (actor hidden), Dense_2 (logits), Dense_3 (critic hidden), Dense_4 (value).
    Located robustly by shape: the logits layer is the Dense with out_dim = 3.
    """
    dp = params["params"]
    names = sorted(k for k in dp if k.startswith("Dense_"))
    logits_name = next(k for k in names if dp[k]["kernel"].shape[-1] == C.NUM_ACTIONS)
    hid_name = names[names.index(logits_name) - 1]
    k1, b1 = dp[hid_name]["kernel"], dp[hid_name]["bias"]
    k2, b2 = dp[logits_name]["kernel"], dp[logits_name]["bias"]
    assert k1.shape[-1] == k2.shape[0]

    def logits_fn(h):
        return jax.nn.relu(h @ k1 + b1) @ k2 + b2
    return logits_fn


def push_to_action(h, tgt, wmask, logits_fn, eta=0.25, iters=15):
    """Minimally perturb hidden states (normalized gradient ascent on the actor's
    logit margin) until the actor's argmax equals the target action. Only rows
    with wmask=True are pushed, and pushing stops per-row once the margin > 0."""
    def margin(hp):
        lg = logits_fn(hp)
        l_t = jnp.take_along_axis(lg, tgt[:, None], 1)[:, 0]
        others = lg - 1e9 * jax.nn.one_hot(tgt, C.NUM_ACTIONS)
        return l_t - jax.nn.logsumexp(others, axis=-1)

    def body(_, hp):
        m = margin(hp)
        active = wmask & (m < 0.0)
        g = jax.grad(lambda hh: (margin(hh) * active).sum())(hp)
        g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        return hp + (active.astype(jnp.float32) * eta)[:, None] * g

    return jax.lax.fori_loop(0, iters, body, h)


def rollout(cfg, net, params, cue, key, n, mode, *, wrong_row=None,
            u=None, ptgt=None, T=None):
    """Batched greedy rollout with intervention `mode`:

    none:      plain greedy rollout.
    force:     action replacement at the fork (older variant, kept for reference).
    force_act: ACTIVATION-space behavior steering — at the fork, the hidden is
               minimally perturbed (push_to_action) until the actor itself picks
               the controller's wrong-direction action; perturbed hidden is
               carried forward. Stops at branch entry.
    swap:      TRANSIENT memory swap — clamp h along `u` to `ptgt` only while
               (x > x_room_end) AND the direction decision is still open
               (taken_branch == NONE); released at branch entry.
    Returns end-of-episode success/branch/door/hidden + full traces.
    """
    p = D._env_params(cfg, cue)
    T = T or cfg["max_steps"]
    x_fork = p.x_pre_end
    keys = jax.random.split(key, n)
    state = jax.vmap(lambda k: jreset(k, p))(keys)
    obs = D._flat(jax.vmap(lambda s: build_obs(s, p))(state))
    hidden = P.ScannedRNN.initialize_carry(n, cfg["gru_hidden"])
    uj = jnp.zeros((cfg["gru_hidden"],), jnp.float32) if u is None else jnp.asarray(u, jnp.float32)
    pt = jnp.float32(0.0 if ptgt is None else ptgt)
    wrow = jnp.int32(0 if wrong_row is None else wrong_row)
    logits_fn = actor_head_fns(params)

    def body(carry, _):
        state, obs, hidden, last_done, dacc, succ, tb, sd, hend, key = carry
        alive = ~dacc
        undecided = state.taken_branch == C.BRANCH_NONE
        if mode == "swap":
            win = (state.agent_x > p.x_room_end) & undecided & alive
            hidden = hidden + (win.astype(jnp.float32) * (pt - hidden @ uj))[:, None] * uj[None, :]
        elif mode in ("force", "force_act"):
            win = (state.agent_x == x_fork) & undecided & alive
        else:
            win = jnp.zeros_like(alive)
        new_hidden, logits, _ = net.apply(params, hidden, (obs[None], last_done[None]))
        if mode == "force_act":
            tgt = forced_action(state, wrow)
            new_hidden = push_to_action(new_hidden, tgt, win, logits_fn)
            a = jnp.argmax(logits_fn(new_hidden), axis=-1).astype(jnp.int32)
        else:
            a_pol = jnp.argmax(logits[0], axis=-1).astype(jnp.int32)
            a = jnp.where(win, forced_action(state, wrow), a_pol) if mode == "force" else a_pol
        key, sk = jax.random.split(key)
        sks = jax.random.split(sk, n)
        ns, r, dn, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = D._flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        newly = dn & (~dacc)
        succ = jnp.where(newly, info["reached_target"].astype(jnp.float32), succ)
        tb = jnp.where(newly, ns.taken_branch, tb)
        sd = jnp.where(newly, ns.selected_door, sd)
        hend = jnp.where(newly[:, None], new_hidden, hend)
        mm = jax.vmap(lambda s: build_obs(s, p)["minimap"])(state)
        out = (state.agent_x, state.agent_y, state.agent_dir, mm, new_hidden,
               win, dn, info["reached_target"])
        return (ns, nobs, new_hidden, dn, dacc | dn, succ, tb, sd, hend, key), out

    carry = (state, obs, hidden, jnp.zeros((n,), bool), jnp.zeros((n,), bool),
             jnp.zeros((n,)), jnp.zeros((n,), jnp.int32), jnp.zeros((n,), jnp.int32),
             hidden, key)
    carry, outs = jax.lax.scan(body, carry, None, length=T)
    (_, _, _, _, dacc, succ, tb, sd, hend, _) = carry
    return (p, tuple(np.asarray(v) for v in (succ, tb, sd, np.asarray(dacc), hend)),
            tuple(np.asarray(v) for v in outs))


# ─────────────────────────────────────────────────────────────────────────────
# quantitative tables
# ─────────────────────────────────────────────────────────────────────────────
def quant(run_dir, n=96):
    cfg, net, params, pr = load_all(run_dir)
    trained = pr["trained"]
    clf = pr["clf"]
    key = jax.random.PRNGKey(0)
    print(f"== {cfg['cue']} model — trained cues: {[CUE_NAMES[c] for c in trained]}")

    # ---- Experiment A: behavior steering via ACTIVATIONS ----
    print("\n-- A. BEHAVIOR-steer via ACTIVATIONS: minimal hidden perturbation until the")
    print("      actor itself picks the wrong-direction action at the fork (released at branch entry) --")
    print(f"   {'cue':11s} | {'wrong-branch':>12s} {'door-correct':>12s} {'success':>8s} "
          f"{'P(true cue) end':>15s}")
    for c in trained:
        cue = CUE_NAMES[c]
        key, k = jax.random.split(key)
        # baseline first (for reference P(true cue))
        p_env, (succ0, tb0, sd0, dacc0, h0), _ = rollout(cfg, net, params, cue, k, n, "none")
        wrong_row = int(p_env.row_lo) if not IS_DOWN[c] else int(p_env.row_up)
        key, k = jax.random.split(key)
        p_env, (succ, tb, sd, dacc, hend), _ = rollout(cfg, net, params, cue, k, n, "force_act",
                                                       wrong_row=wrong_row)
        wrongb = C.BRANCH_DOWN if not IS_DOWN[c] else C.BRANCH_UP
        target_sd = C.SEL_BLUE if IS_BLUE[c] else C.SEL_GREEN
        ptrue = cue_probs(hend, clf)[:, list(trained).index(c)].mean()
        ptrue0 = cue_probs(h0, clf)[:, list(trained).index(c)].mean()
        print(f"   {cue:11s} | {float((tb == wrongb).mean()):12.2f} "
              f"{float((sd == target_sd).mean()):12.2f} {float(succ.mean()):8.2f} "
              f"{ptrue:8.2f} (base {ptrue0:.2f})")

    # ---- Experiment B: TRANSIENT belief swap between trained-cue representations ----
    print("\n-- B. SWAP memory TRANSIENTLY: clamp along class-mean axis src->tgt, ONLY from")
    print("      cue-room exit until the direction decision; released at branch entry --")
    print(f"   {'src -> tgt':24s} | {'branch-as-tgt':>13s} {'door-as-tgt':>11s} "
          f"{'behaves-as-tgt':>14s} {'P(tgt cue) end':>14s}")
    for s in trained:
        for t in trained:
            if s == t:
                continue
            u = pr["mu"][t] - pr["mu"][s]
            u = u / np.linalg.norm(u)
            ptgt = float(pr["mu"][t] @ u)
            key, k = jax.random.split(key)
            p_env, (succ, tb, sd, dacc, hend), _ = rollout(
                cfg, net, params, CUE_NAMES[s], k, n, "swap", u=u, ptgt=ptgt)
            b_t = C.BRANCH_DOWN if IS_DOWN[t] else C.BRANCH_UP
            d_t = C.SEL_BLUE if IS_BLUE[t] else C.SEL_GREEN
            bok = (tb == b_t); dok = (sd == d_t)
            ptg = cue_probs(hend, clf)[:, list(trained).index(t)].mean()
            print(f"   {CUE_NAMES[s]:>11s} -> {CUE_NAMES[t]:11s} | {float(bok.mean()):13.2f} "
                  f"{float(dok.mean()):11.2f} {float((bok & dok).mean()):14.2f} {ptg:14.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# videos
# ─────────────────────────────────────────────────────────────────────────────
def video(run_dir, out, fps=4, hold=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from viz_rollout_dream import TILE_RGB, CUE_MARK  # noqa

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    PURPLE = "#8e44ad"
    DIR_ARROW = {0: (0.35, 0), 1: (0, 0.35), 2: (-0.35, 0), 3: (0, -0.35)}

    cfg, net, params, pr = load_all(run_dir)
    trained = pr["trained"]
    clf = pr["clf"]
    ax1, ax2, axlab, land, lims = belief_plane(pr)

    def panels(fig, ep, t):
        (p, s0, xs, ys, ds, mms, hs, win, dn, reach, label) = ep
        gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.55, 1.0],
                              height_ratios=[1, 1.12], hspace=0.3, wspace=0.25)
        # maze
        axm = fig.add_subplot(gs[0, :2])
        terr = np.asarray(p.base_terrain); full = terr.copy()
        ct = int(np.asarray(s0.cue_type))
        full[int(s0.cue_y), int(s0.cue_x)] = C.CUE_TILE[ct]
        dgt = bool(np.asarray(s0.door_green_top))
        full[p.row_door_top, p.x_doorcol] = C.DOOR_GREEN if dgt else C.DOOR_BLUE
        full[p.row_door_bot, p.x_doorcol] = C.DOOR_BLUE if dgt else C.DOOR_GREEN
        axm.imshow(TILE_RGB[full], interpolation="nearest")
        for r in range(full.shape[0]):
            for cc in range(full.shape[1]):
                if full[r, cc] in CUE_MARK:
                    axm.text(cc, r, CUE_MARK[full[r, cc]], ha="center", va="center",
                             color="white", fontsize=8, fontweight="bold")
        axm.plot(xs[:t + 1], ys[:t + 1], "-", color="#f28e2b", lw=2, alpha=0.7, zorder=4)
        wm = win[:t + 1].astype(bool)
        if wm.any():
            axm.scatter(xs[:t + 1][wm], ys[:t + 1][wm], s=42, c=PURPLE, marker="D",
                        edgecolor="k", lw=0.3, zorder=5)
        axm.scatter([xs[t]], [ys[t]], s=110, c=(PURPLE if win[t] else "#d1495b"),
                    edgecolor="k", zorder=6)
        dx, dy = DIR_ARROW[int(ds[t])]
        axm.annotate("", xy=(xs[t] + 2 * dx, ys[t] + 2 * dy), xytext=(xs[t], ys[t]),
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6), zorder=7)
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title("high-level view", fontsize=11, fontweight="bold")
        # agent view
        axv = fig.add_subplot(gs[0, 2])
        mm = mms[t]
        axv.imshow(TILE_RGB[mm], interpolation="nearest")
        for r in range(mm.shape[0]):
            for cc in range(mm.shape[1]):
                if mm[r, cc] in CUE_MARK:
                    axv.text(cc, r, CUE_MARK[mm[r, cc]], ha="center", va="center",
                             color="white", fontsize=13, fontweight="bold")
        mcell = mm.shape[0] // 2
        axv.scatter([mcell], [mcell], s=200, c="#d1495b", edgecolor="k", zorder=6)
        axv.set_xticks([]); axv.set_yticks([])
        axv.set_title("agent view", fontsize=11, fontweight="bold")
        # belief plane (trained cues only)
        axp = fig.add_subplot(gs[1, 2])
        for c in trained:
            axp.scatter(*land[c], s=280, c=CUE_COL[c], alpha=0.3, edgecolor=CUE_COL[c], zorder=2)
            axp.annotate(CUE_NAMES[c], land[c], fontsize=7.5, ha="center", color="#333", zorder=3)
        ps1, ps2 = hs[:t + 1] @ ax1, hs[:t + 1] @ ax2
        axp.plot(ps1, ps2, "-", color="#d1495b", lw=1.5, alpha=0.6, zorder=4)
        if wm.any():
            axp.scatter(ps1[wm], ps2[wm], s=28, c=PURPLE, marker="D", zorder=5, alpha=0.8)
        axp.scatter([ps1[-1]], [ps2[-1]], s=100, c=(PURPLE if win[t] else "#d1495b"),
                    edgecolor="k", zorder=6)
        if win[t]:
            axp.text(0.03, 0.95, "INTERVENTION", transform=axp.transAxes, fontsize=8.5,
                     color=PURPLE, fontweight="bold", va="top")
        axp.set_xlim(lims[0]); axp.set_ylim(lims[1])
        axp.set_xlabel(axlab[0], fontsize=8.5); axp.set_ylabel(axlab[1], fontsize=8.5)
        axp.set_title("belief plane (trained cues)", fontsize=11, fontweight="bold")
        # cue-probability timeline
        axt = fig.add_subplot(gs[1, :2])
        probs = cue_probs(hs, clf)          # (T, K)
        vis = np.array([(m2[..., None] == np.asarray([C.CUE_TILE[i] for i in range(4)])).any()
                        for m2 in mms], dtype=float)
        axt.fill_between(np.arange(len(vis)), 0, vis, color="0.9", step="mid", label="cue in view")
        if win.any():
            w0, w1 = np.where(win)[0][[0, -1]]
            axt.axvspan(w0 - .5, w1 + .5, color=PURPLE, alpha=0.13, label="intervention")
        for i, c in enumerate(trained):
            axt.plot(np.arange(t + 1), probs[:t + 1, i], "-", lw=2, color=CUE_COL[c],
                     label=f"P(cue = {CUE_NAMES[c]})")
        axt.axhline(1 / len(trained), ls="--", c="#999", lw=0.8)
        axt.axvline(t, color="#d1495b", lw=1.2, alpha=0.8)
        axt.set_xlim(0, len(probs) - 1); axt.set_ylim(-0.05, 1.05)
        axt.set_xlabel("timestep", fontsize=9); axt.set_ylabel("cue-identity belief", fontsize=9)
        axt.set_title("belief over time: P(cue) per training cue", fontsize=11, fontweight="bold")
        axt.legend(loc="center right", fontsize=7, framealpha=0.9)

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    key = jax.random.PRNGKey(21)

    src = trained[0]
    tgt = trained[-1]
    cue = CUE_NAMES[src]
    episodes = []
    # baseline (same key for all three episodes -> identical env episode)
    key, k = jax.random.split(key)
    p, _, tr = rollout(cfg, net, params, cue, k, 1, "none", T=110)
    episodes.append((p, k, tr, f"{cfg['cue']} · {cue} · BASELINE"))
    # behavior steered via activations (wrong direction)
    wrong_row = int(p.row_lo) if not IS_DOWN[src] else int(p.row_up)
    p2, _, tr2 = rollout(cfg, net, params, cue, k, 1, "force_act", wrong_row=wrong_row, T=110)
    episodes.append((p2, k, tr2,
                     f"{cfg['cue']} · {cue} · BEHAVIOR steered via activations (wrong direction)"))
    # transient belief swap
    u = pr["mu"][tgt] - pr["mu"][src]; u = u / np.linalg.norm(u)
    ptv = float(pr["mu"][tgt] @ u)
    p3, _, tr3 = rollout(cfg, net, params, cue, k, 1, "swap", u=u, ptgt=ptv, T=110)
    episodes.append((p3, k, tr3,
                     f"{cfg['cue']} · {cue} · MEMORY swapped -> {CUE_NAMES[tgt]} (transient, pre-decision)"))

    for pE, kE, trE, label in episodes:
        xs, ys, ds, mms, hs, win, dn, reach = (np.asarray(v)[:, 0] for v in trE)
        s0 = jax.vmap(lambda kk: jreset(kk, pE))(jax.random.split(kE, 1))
        s0 = jax.tree_util.tree_map(lambda x: x[0], s0)
        nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
        ok = bool(reach[nd - 1]) if dn.any() else False
        ep = (pE, s0, xs, ys, ds, mms, hs, win.astype(bool), dn, reach, label)
        for t in list(range(nd)) + [nd - 1] * hold:
            fig = plt.figure(figsize=(13.2, 7.2))
            panels(fig, ep, t)
            outc = ""
            if t == nd - 1 and dn[nd - 1]:
                outc = "   ->   " + ("color-correct door" if ok else "other door")
            col = "#1a7d36" if (outc and ok) else ("#b02418" if outc else
                                                   (PURPLE if win[t] else "#222"))
            stf = "  ·  INTERVENTION" if win[t] else ""
            fig.suptitle(f"{label}  ·  step {t}{stf}{outc}", fontsize=13,
                         fontweight="bold", color=col)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            plt.close(fig)
        print(f"[video] {label}: {nd} steps, reached-true-door={ok}", flush=True)
    writer.close()
    print(f"[video] wrote {out}", flush=True)


def fig(run_dir, out):
    """Static report figure: trajectories + belief timelines + plane paths for
    baseline / activation-steered behavior / transient memory swap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz_rollout_dream import TILE_RGB, CUE_MARK  # noqa

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    PURPLE = "#8e44ad"
    cfg, net, params, pr = load_all(run_dir)
    trained = pr["trained"]; clf = pr["clf"]
    ax1, ax2, axlab, land, _lims = belief_plane(pr)

    src, tgt = trained[0], trained[-1]
    cue = CUE_NAMES[src]
    key = jax.random.PRNGKey(21)
    key, k = jax.random.split(key)
    p, _, tr0 = rollout(cfg, net, params, cue, k, 1, "none", T=110)
    wrong_row = int(p.row_lo) if not IS_DOWN[src] else int(p.row_up)
    _, _, trF = rollout(cfg, net, params, cue, k, 1, "force_act", wrong_row=wrong_row, T=110)
    u = pr["mu"][tgt] - pr["mu"][src]; u = u / np.linalg.norm(u)
    _, _, trS = rollout(cfg, net, params, cue, k, 1, "swap", u=u,
                        ptgt=float(pr["mu"][tgt] @ u), T=110)
    s0 = jax.tree_util.tree_map(lambda x: x[0],
                                jax.vmap(lambda kk: jreset(kk, p))(jax.random.split(k, 1)))
    eps = [(tr0, "BASELINE", "#555"),
           (trF, "BEHAVIOR steered (activations)", "#b02418"),
           (trS, f"MEMORY swapped -> {CUE_NAMES[tgt]} (transient)", "#1f77b4")]

    fig_, axs = plt.subplots(3, 3, figsize=(15, 10),
                             gridspec_kw=dict(height_ratios=[1, 0.75, 1.05]))
    terr = np.asarray(p.base_terrain); full0 = terr.copy()
    ct = int(np.asarray(s0.cue_type))
    full0[int(s0.cue_y), int(s0.cue_x)] = C.CUE_TILE[ct]
    dgt = bool(np.asarray(s0.door_green_top))
    full0[p.row_door_top, p.x_doorcol] = C.DOOR_GREEN if dgt else C.DOOR_BLUE
    full0[p.row_door_bot, p.x_doorcol] = C.DOOR_BLUE if dgt else C.DOOR_GREEN
    for col, (tr, label, lc) in enumerate(eps):
        xs, ys, ds, mms, hs, win, dn, reach = [np.asarray(v)[:, 0] for v in tr]
        nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
        ok = bool(reach[nd - 1]) if dn.any() else False
        # row 0: trajectory
        axm = axs[0, col]
        axm.imshow(TILE_RGB[full0], interpolation="nearest")
        for r in range(full0.shape[0]):
            for cc in range(full0.shape[1]):
                if full0[r, cc] in CUE_MARK:
                    axm.text(cc, r, CUE_MARK[full0[r, cc]], ha="center", va="center",
                             color="white", fontsize=7, fontweight="bold")
        axm.plot(xs[:nd], ys[:nd], "-", color="#f28e2b", lw=1.8, alpha=0.85, zorder=4)
        wm = win[:nd].astype(bool)
        if wm.any():
            axm.scatter(xs[:nd][wm], ys[:nd][wm], s=34, c=PURPLE, marker="D",
                        edgecolor="k", lw=0.3, zorder=5)
        axm.scatter([xs[nd - 1]], [ys[nd - 1]], s=90, c="#d1495b", edgecolor="k", zorder=6)
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title(f"{label}\n-> {'color-correct door' if ok else 'other door'}",
                      fontsize=9.5, fontweight="bold",
                      color=("#1a7d36" if ok else "#b02418"))
        # row 1: belief timeline
        axt = axs[1, col]
        probs = cue_probs(hs[:nd], clf)
        if win[:nd].any():
            w0, w1 = np.where(win[:nd])[0][[0, -1]]
            axt.axvspan(w0 - .5, w1 + .5, color=PURPLE, alpha=0.13)
        for i, c in enumerate(trained):
            axt.plot(probs[:, i], "-", lw=1.8, color=CUE_COL[c],
                     label=f"P({CUE_NAMES[c]})")
        axt.axhline(1 / len(trained), ls="--", c="#999", lw=0.7)
        axt.set_ylim(-0.05, 1.05); axt.set_xlabel("timestep", fontsize=8)
        if col == 0:
            axt.set_ylabel("P(cue)", fontsize=8)
            axt.legend(fontsize=6.5, loc="center right")
        # row 2: plane path (span later)
        axp = axs[2, col]
        for c in trained:
            axp.scatter(*land[c], s=240, c=CUE_COL[c], alpha=0.3, edgecolor=CUE_COL[c], zorder=2)
            axp.annotate(CUE_NAMES[c], land[c], fontsize=7, ha="center", color="#333", zorder=3)
        p1, p2 = hs[:nd] @ ax1, hs[:nd] @ ax2
        axp.plot(p1, p2, "-", color=lc, lw=1.6, alpha=0.85, zorder=4)
        if wm.any():
            axp.scatter(p1[wm], p2[wm], s=24, c=PURPLE, marker="D", zorder=5, alpha=0.85)
        axp.scatter([p1[0]], [p2[0]], marker="s", s=40, c="#999", zorder=6)
        axp.scatter([p1[-1]], [p2[-1]], s=80, c=lc, edgecolor="k", zorder=6)
        axp.set_xlabel(axlab[0], fontsize=8)
        if col == 0:
            axp.set_ylabel(axlab[1], fontsize=8)
        axp.set_title("belief-plane path", fontsize=9)
    fig_.suptitle(f"{cfg['cue']} model · cue {cue} · same episode under the three conditions "
                  "(purple = intervention active)", fontsize=13, fontweight="bold")
    fig_.tight_layout(rect=[0, 0, 1, 0.96])
    fig_.savefig(out, dpi=140)
    print(f"[fig] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quant", "video", "fig"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "quant":
        quant(a.run_dir, a.n)
    elif a.cmd == "fig":
        fig(a.run_dir, a.out or "steer2_fig.png")
    else:
        video(a.run_dir, a.out or "steer2.mp4")


if __name__ == "__main__":
    main()
