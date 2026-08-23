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


def belief_field(cfg, net, params, pr, gridn=27, pad=2.0):
    """Decision heatmap + GRU dynamics vector field over the belief plane.

    The plane is the 2-D affine slice through the mean trained-cue hidden state
    spanned by an ORTHONORMALIZED (e1, e2) version of the belief_plane axes.
    For every grid point h(u,v) = base + u*e1 + v*e2:
      * heat  = the cue-identity probe's P(cue) blended into the cue colors
                (the probe's decision regions in the plane);
      * arrow = the in-plane displacement after ONE GRU step under a neutral
                mid-corridor observation (the memory-maintenance input) —
                the attractor flow of the recurrent dynamics.
    """
    from matplotlib.colors import to_rgb

    trained, clf = pr["trained"], pr["clf"]
    ax1, ax2, axlab, _land, _lims = belief_plane(pr)
    e1 = ax1 / np.linalg.norm(ax1)
    e2 = ax2 - (ax2 @ e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    if abs(float(ax2 @ e1)) > 0.05:
        axlab = (axlab[0], axlab[1] + " (orthogonalized)")
    Xt = pr["Xt"]
    hbar = Xt.mean(0)
    p1, p2 = Xt @ e1, Xt @ e2
    lims = ((p1.min() - pad, p1.max() + pad), (p2.min() - pad, p2.max() + pad))
    us = np.linspace(*lims[0], gridn)
    vs = np.linspace(*lims[1], gridn)
    UU, VV = np.meshgrid(us, vs)
    base = hbar - (hbar @ e1) * e1 - (hbar @ e2) * e2
    H = base[None] + UU.reshape(-1, 1) * e1[None] + VV.reshape(-1, 1) * e2[None]

    P = cue_probs(H, clf)                                        # (N, K)
    cols = np.array([to_rgb(CUE_COL[c]) for c in trained])
    bg = P @ cols                                                # blend per point
    bg = 1.0 - 0.42 * (1.0 - bg)                                 # lighten toward white
    bg_img = bg.reshape(gridn, gridn, 3)

    # neutral maintenance observation: mid pre-branch corridor, facing east
    from cogniland.memory_env.jax import make_state
    cue0 = trained[0]
    p_env = D._env_params(cfg, CUE_NAMES[cue0])
    st = make_state(p_env, cue0, True, p_env.x_room_start, p_env.row_room_up)
    st = st.replace(agent_x=jnp.int32(p_env.x_pre_end - 2),
                    agent_y=jnp.int32(p_env.my), agent_dir=jnp.int32(C.DIR_EAST))
    obs = D._flat({k: v[None] for k, v in build_obs(st, p_env).items()})[0]
    N = H.shape[0]
    obs_b = jnp.broadcast_to(jnp.asarray(obs), (1, N, obs.shape[-1]))
    nh, _, _ = net.apply(params, jnp.asarray(H, jnp.float32),
                         (obs_b, jnp.zeros((1, N), bool)))
    dh = np.asarray(nh) - H
    qu, qv = (dh @ e1).reshape(gridn, gridn), (dh @ e2).reshape(gridn, gridn)

    land = {c: (float(pr["mu"][c] @ e1), float(pr["mu"][c] @ e2)) for c in trained}
    return dict(bg=bg_img, extent=(us[0], us[-1], vs[0], vs[-1]), UU=UU, VV=VV,
                qu=qu, qv=qv, e1=e1, e2=e2, axlab=axlab, land=land, lims=lims)


def draw_belief_field(axp, field, trained, quiver_step=2):
    """Render heatmap + normalized flow arrows + class-mean landmarks."""
    axp.imshow(field["bg"], origin="lower", extent=field["extent"],
               aspect="auto", zorder=1, interpolation="bilinear")
    s = quiver_step
    UU, VV = field["UU"][::s, ::s], field["VV"][::s, ::s]
    qu, qv = field["qu"][::s, ::s], field["qv"][::s, ::s]
    mag = np.hypot(qu, qv) + 1e-9
    axp.quiver(UU, VV, qu / mag, qv / mag, np.log10(mag),
               cmap="Greys", alpha=0.65, width=0.004, scale=28,
               headwidth=3.5, zorder=2)
    for c in trained:
        axp.scatter(*field["land"][c], s=230, c=CUE_COL[c], alpha=0.95,
                    edgecolor="k", lw=0.8, zorder=5)
        axp.annotate(CUE_NAMES[c], field["land"][c], fontsize=7, ha="center",
                     va="center", color="white", fontweight="bold", zorder=6)
    axp.set_xlim(field["lims"][0])
    axp.set_ylim(field["lims"][1])


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
_DX = jnp.asarray([v[0] for v in C.DIR_VEC], dtype=jnp.int32)
_DY = jnp.asarray([v[1] for v in C.DIR_VEC], dtype=jnp.int32)


def forced_action(state, target_row, p):
    """Scripted controller: drive to target branch row, then east into the branch;
    opens the mid-corridor marker door when it blocks the way."""
    y, dref = state.agent_y, state.agent_dir
    desired = jnp.where(y != target_row,
                        jnp.where(target_row < y, C.DIR_NORTH, C.DIR_SOUTH),
                        C.DIR_EAST)
    diff = (desired - dref) % 4
    a = jnp.where(diff == 0, C.A_FORWARD,
                  jnp.where(diff == 3, C.A_LEFT, C.A_RIGHT)).astype(jnp.int32)
    # facing a closed marker door -> open it instead of walking into it
    tx = state.agent_x + _DX[dref]
    ty = state.agent_y + _DY[dref]
    closed_ahead = (((tx == p.x_mark) & (ty == p.row_up) & ~state.mark_top_open)
                    | ((tx == p.x_mark) & (ty == p.row_lo) & ~state.mark_bot_open))
    return jnp.where((diff == 0) & closed_ahead, C.A_OPEN, a).astype(jnp.int32)


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


def push_linear(h, tgt, wmask, logits_fn, alpha):
    """Open-loop LINEAR steering: a single normalized-gradient step of fixed
    size alpha along the actor-margin direction toward tgt (no stopping
    criterion — the dose is alpha, however the actor responds)."""
    def margin(hp):
        lg = logits_fn(hp)
        l_t = jnp.take_along_axis(lg, tgt[:, None], 1)[:, 0]
        others = lg - 1e9 * jax.nn.one_hot(tgt, C.NUM_ACTIONS)
        return l_t - jax.nn.logsumexp(others, axis=-1)

    g = jax.grad(lambda hh: (margin(hh) * wmask).sum())(h)
    g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
    return h + (wmask.astype(jnp.float32) * alpha)[:, None] * g


def rollout(cfg, net, params, cue, key, n, mode, *, wrong_row=None,
            u=None, ptgt=None, T=None, sample=False, alpha=None,
            u_field=None, ptgt_field=None):
    """Batched rollout with intervention `mode`:

    none:      plain policy rollout.
    force:     action replacement at the fork (older variant, kept for reference).
    force_thru: action replacement from the fork THROUGH the wrong-branch marker
               door (the scripted navigator opens it); released one step past
               the marker column. The hidden state is never touched — any belief
               change is driven by the experienced observation sequence.
    force_act: ACTIVATION-space behavior steering — at the fork, the hidden is
               minimally perturbed (push_to_action) until the actor itself picks
               the controller's wrong-direction action; perturbed hidden is
               carried forward. Stops at branch entry.
    force_lin: LINEAR behavior steering — at the fork, h += alpha * unit-grad of
               the actor margin toward the wrong-direction action (open-loop
               dose alpha, one step per fork timestep); carried forward.
    swap:      TRANSIENT memory swap — clamp h along `u` to `ptgt` only while
               (x > x_room_end) AND the direction decision is still open
               (taken_branch == NONE); released at branch entry.
    swap_field: same window, but the clamp axis/target are POSITION-LOCAL:
               u_field (W, D) and ptgt_field (W,) indexed by agent_x
               (field-aware steering; axes from per-column class means).
    sample=True draws actions from the softmax policy instead of argmax.
    Returns end-of-episode success/branch/door/finished/hidden + full traces
    (finished = episode ended at a door, i.e. terminated with no timeout).
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
    UF = None if u_field is None else jnp.asarray(u_field, jnp.float32)
    PF = None if ptgt_field is None else jnp.asarray(ptgt_field, jnp.float32)
    wrow = jnp.int32(0 if wrong_row is None else wrong_row)
    alf = jnp.float32(0.0 if alpha is None else alpha)
    logits_fn = actor_head_fns(params)

    def body(carry, _):
        state, obs, hidden, last_done, dacc, succ, tb, sd, fin, hend, key = carry
        alive = ~dacc
        undecided = state.taken_branch == C.BRANCH_NONE
        if mode == "swap":
            win = (state.agent_x > p.x_room_end) & undecided & alive
            hidden = hidden + (win.astype(jnp.float32) * (pt - hidden @ uj))[:, None] * uj[None, :]
        elif mode == "swap_field":
            win = (state.agent_x > p.x_room_end) & undecided & alive
            ux = UF[state.agent_x]                                   # (n, D)
            px = PF[state.agent_x]                                   # (n,)
            hidden = hidden + (win.astype(jnp.float32)
                               * (px - (hidden * ux).sum(-1)))[:, None] * ux
        elif mode in ("force", "force_act", "force_lin"):
            win = (state.agent_x == x_fork) & undecided & alive
        elif mode == "force_thru":
            past_mark = (state.taken_branch != C.BRANCH_NONE) & (state.agent_x > p.x_mark)
            win = (state.agent_x >= x_fork) & ~past_mark & alive
        else:
            win = jnp.zeros_like(alive)
        new_hidden, logits, _ = net.apply(params, hidden, (obs[None], last_done[None]))
        if mode == "force_act":
            tgt = forced_action(state, wrow, p)
            new_hidden = push_to_action(new_hidden, tgt, win, logits_fn)
            pol_logits = logits_fn(new_hidden)
        elif mode == "force_lin":
            tgt = forced_action(state, wrow, p)
            new_hidden = push_linear(new_hidden, tgt, win, logits_fn, alf)
            pol_logits = logits_fn(new_hidden)
        else:
            pol_logits = logits[0]
        if sample:
            key, ka = jax.random.split(key)
            a = jax.random.categorical(ka, pol_logits, axis=-1).astype(jnp.int32)
        else:
            a = jnp.argmax(pol_logits, axis=-1).astype(jnp.int32)
        if mode in ("force", "force_thru"):
            a = jnp.where(win, forced_action(state, wrow, p), a)
        key, sk = jax.random.split(key)
        sks = jax.random.split(sk, n)
        ns, r, dn, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = D._flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        newly = dn & (~dacc)
        succ = jnp.where(newly, info["reached_target"].astype(jnp.float32), succ)
        tb = jnp.where(newly, ns.taken_branch, tb)
        sd = jnp.where(newly, ns.selected_door, sd)
        fin = jnp.where(newly, info["is_terminal"], fin)   # door reached, not timeout
        hend = jnp.where(newly[:, None], new_hidden, hend)
        mm = jax.vmap(lambda s: build_obs(s, p)["minimap"])(state)
        out = (state.agent_x, state.agent_y, state.agent_dir, mm, new_hidden,
               win, dn, info["reached_target"],
               state.mark_top_open, state.mark_bot_open)
        return (ns, nobs, new_hidden, dn, dacc | dn, succ, tb, sd, fin, hend, key), out

    carry = (state, obs, hidden, jnp.zeros((n,), bool), jnp.zeros((n,), bool),
             jnp.zeros((n,)), jnp.zeros((n,), jnp.int32), jnp.zeros((n,), jnp.int32),
             jnp.zeros((n,), bool), hidden, key)
    carry, outs = jax.lax.scan(body, carry, None, length=T)
    (_, _, _, _, dacc, succ, tb, sd, fin, hend, _) = carry
    return (p, tuple(np.asarray(v) for v in (succ, tb, sd, np.asarray(dacc), fin, hend)),
            tuple(np.asarray(v) for v in outs))


# ─────────────────────────────────────────────────────────────────────────────
# quantitative tables
# ─────────────────────────────────────────────────────────────────────────────
def quant(run_dir, n=96, sample=False):
    cfg, net, params, pr = load_all(run_dir)
    trained = pr["trained"]
    clf = pr["clf"]
    key = jax.random.PRNGKey(0)
    print(f"== {cfg['cue']} model — trained cues: {[CUE_NAMES[c] for c in trained]}  "
          f"policy={'softmax (sampled)' if sample else 'greedy'}")

    # ---- Experiment A: behavior steering via ACTIVATIONS ----
    print("\n-- A. BEHAVIOR-steer via ACTIVATIONS: minimal hidden perturbation until the")
    print("      actor itself picks the wrong-direction action at the fork (released at branch entry) --")
    print(f"   {'cue':11s} | {'wrong-branch':>12s} {'door-correct':>12s} {'success':>8s} "
          f"{'P(true cue) end':>15s}")
    for c in trained:
        cue = CUE_NAMES[c]
        key, k = jax.random.split(key)
        # baseline first (for reference P(true cue))
        p_env, (succ0, tb0, sd0, dacc0, fin0, h0), _ = rollout(cfg, net, params, cue, k, n, "none",
                                                               sample=sample)
        wrong_row = int(p_env.row_lo) if not IS_DOWN[c] else int(p_env.row_up)
        key, k = jax.random.split(key)
        p_env, (succ, tb, sd, dacc, fin, hend), _ = rollout(cfg, net, params, cue, k, n, "force_act",
                                                            wrong_row=wrong_row, sample=sample)
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
            p_env, (succ, tb, sd, dacc, fin, hend), _ = rollout(
                cfg, net, params, CUE_NAMES[s], k, n, "swap", u=u, ptgt=ptgt, sample=sample)
            b_t = C.BRANCH_DOWN if IS_DOWN[t] else C.BRANCH_UP
            d_t = C.SEL_BLUE if IS_BLUE[t] else C.SEL_GREEN
            bok = (tb == b_t); dok = (sd == d_t)
            ptg = cue_probs(hend, clf)[:, list(trained).index(t)].mean()
            print(f"   {CUE_NAMES[s]:>11s} -> {CUE_NAMES[t]:11s} | {float(bok.mean()):13.2f} "
                  f"{float(dok.mean()):11.2f} {float((bok & dok).mean()):14.2f} {ptg:14.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# linear-steering dose-response (alpha sweep)
# ─────────────────────────────────────────────────────────────────────────────
def alpha_sweep(run_dirs, out, n=96, alphas=None, sample=True, T=None):
    """Dose-response of LINEAR behavior steering (mode force_lin): at the fork,
    h += alpha * unit-grad(actor margin toward the wrong-direction action).

    Per model / trained cue / alpha: probe P(true cue), read out at the LAST
    step BEFORE any door tile enters the agent's egocentric view (belief before
    the door-decision corridor), the wrong-branch rate (dashed), and the
    fraction of episodes that end at a door with no timeout (dotted). Rollouts
    run to the env's real horizon (max_steps) so timeout is the env's own.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    alphas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] if alphas is None else list(alphas)
    fig_, axs = plt.subplots(1, len(run_dirs), figsize=(4.8 * len(run_dirs), 3.9),
                             squeeze=False, sharey=True)
    key = jax.random.PRNGKey(5)
    for j, rdir in enumerate(run_dirs):
        cfg, net, params, pr = load_all(rdir)
        trained, clf = pr["trained"], pr["clf"]
        ax = axs[0, j]
        for c in trained:
            cue = CUE_NAMES[c]
            p_env = D._env_params(cfg, cue)
            wrong_row = int(p_env.row_lo) if not IS_DOWN[c] else int(p_env.row_up)
            wrongb = C.BRANCH_DOWN if not IS_DOWN[c] else C.BRANCH_UP
            p_true, flip, fins = [], [], []
            for alpha in alphas:
                key, k = jax.random.split(key)
                _, (succ, tb, sd, dacc, fin, hend), outs = rollout(
                    cfg, net, params, cue, k, n, "force_lin", wrong_row=wrong_row,
                    alpha=float(alpha), sample=sample, T=T)
                xs, ys, ds, mms, hs, win, dnn, reach, mtop, mbot = outs
                # last step before any door tile is visible in the minimap
                door_vis = np.isin(mms, [C.DOOR_GREEN, C.DOOR_BLUE]).any(axis=(2, 3))
                done_before = np.zeros(door_vis.shape, bool)
                done_before[1:] = np.cumsum(dnn[:-1], axis=0) > 0
                tgrid = np.arange(door_vis.shape[0])[:, None]
                firstvis = np.where(door_vis & ~done_before, tgrid, door_vis.shape[0]).min(0)
                lastalive = np.where(~done_before, tgrid, -1).max(0)
                idx = np.where(firstvis < door_vis.shape[0], firstvis - 1, lastalive)
                idx = np.clip(idx, 0, door_vis.shape[0] - 1)
                h_pre = hs[idx, np.arange(hs.shape[1])]
                p_true.append(float(cue_probs(h_pre, clf)[:, list(trained).index(c)].mean()))
                flip.append(float((tb == wrongb).mean()))
                fins.append(float(fin.mean()))
                print(f"[alpha] {cfg['cue']:4s} {cue:11s} alpha={alpha:<5g} "
                      f"P(true cue)pre-door={p_true[-1]:.3f}  wrong-branch={flip[-1]:.2f}  "
                      f"finished={fins[-1]:.2f}", flush=True)
            ax.plot(alphas, p_true, "-o", color=CUE_COL[c], lw=2, ms=4,
                    label=f"P({cue})")
            ax.plot(alphas, flip, "--s", color=CUE_COL[c], lw=1.1, ms=3, alpha=0.5)
            ax.plot(alphas, fins, ":^", color=CUE_COL[c], lw=1.1, ms=3, alpha=0.7)
        ax.axhline(1 / len(trained), ls=":", c="#999", lw=0.8)
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xticks(alphas)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks([], minor=True)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("alpha (linear hidden push at fork)")
        if j == 0:
            ax.set_ylabel("P(true cue) — before doors visible")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color="#555", ls="--", marker="s", ms=3))
        labels.append("wrong-branch rate")
        handles.append(Line2D([], [], color="#555", ls=":", marker="^", ms=3))
        labels.append("finished (no timeout)")
        ax.legend(handles, labels, fontsize=7, framealpha=0.9)
        ax.set_title(f"{cfg['cue']} model", fontsize=11, fontweight="bold")
    fig_.suptitle("Linear behavior steering, dose-response (softmax policy): "
                  "belief probe read out before the doors are visible",
                  fontsize=12, fontweight="bold")
    fig_.tight_layout(rect=[0, 0, 1, 0.93])
    outp = pathlib.Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig_.savefig(outp, dpi=150)
    print(f"[alpha] wrote {outp}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# videos
# ─────────────────────────────────────────────────────────────────────────────
def evidence_spec(cue_set):
    """Curated episode list per model for the marker-door env results:
    baseline / action-forced through the WRONG corridor (evidence-driven belief
    revision) / transient memory swaps that causally drive corridor+marker+door."""
    gu, bu, gd, bd = 0, 1, 2, 3
    if cue_set == "2cue":
        return [
            dict(mode="none", cue="green_up", label="BASELINE"),
            dict(mode="force_thru", cue="green_up",
                 label="FORCED down the WRONG corridor -> belief collapses to blue_down"),
            dict(mode="force_thru", cue="blue_down",
                 label="FORCED up the WRONG corridor -> behaves as green_up"),
        ]
    if cue_set == "3cue":
        return [
            dict(mode="none", cue="green_up", label="BASELINE"),
            dict(mode="force_thru", cue="green_up",
                 label="FORCED down the WRONG corridor -> belief collapses to blue_down"),
            dict(mode="force_thru", cue="green_down",
                 label="FORCED up the WRONG corridor -> direction-only update, door still green"),
            dict(mode="swap", src=gu, tgt=gd,
                 label="MEMORY swapped green_up -> green_down (transient, pre-decision)"),
        ]
    return [
        dict(mode="none", cue="green_up", label="BASELINE"),
        dict(mode="force_thru", cue="green_up",
             label="FORCED down the WRONG corridor -> direction revised, color kept, door green"),
        dict(mode="swap", src=gu, tgt=bd,
             label="MEMORY swapped green_up -> blue_down (transient) -> full causal chain"),
        dict(mode="swap", src=gu, tgt=bu,
             label="MEMORY swapped green_up -> blue_up (color only) -> same corridor, blue door"),
    ]


def video(run_dir, out, fps=4, hold=10, curated=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from tile_textures import render_grid, agent_triangle, OPEN_TEX_ID

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    PURPLE = "#8e44ad"

    cfg, net, params, pr = load_all(run_dir)
    trained = pr["trained"]
    clf = pr["clf"]
    field = belief_field(cfg, net, params, pr)      # heatmap + flow, computed once
    e1, e2, axlab = field["e1"], field["e2"], field["axlab"]

    def panels(fig, ep, t):
        (p, s0, xs, ys, ds, mms, hs, win, dn, reach, mtop, mbot, label) = ep
        gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.55, 1.0],
                              height_ratios=[1, 1.12], hspace=0.3, wspace=0.25)
        # maze (privileged high-level view: marker doors drawn closed/open)
        axm = fig.add_subplot(gs[0, :2])
        full = np.asarray(p.base_terrain).copy()
        ct = int(np.asarray(s0.cue_type))
        full[int(s0.cue_y), int(s0.cue_x)] = C.CUE_TILE[ct]
        dgt = bool(np.asarray(s0.door_green_top))
        full[p.row_door_top, p.x_doorcol] = C.DOOR_GREEN if dgt else C.DOOR_BLUE
        full[p.row_door_bot, p.x_doorcol] = C.DOOR_BLUE if dgt else C.DOOR_GREEN
        full[p.row_up, p.x_mark] = OPEN_TEX_ID[C.MARK_A] if mtop[t] else C.MARK_A
        full[p.row_lo, p.x_mark] = OPEN_TEX_ID[C.MARK_B] if mbot[t] else C.MARK_B
        Hh, Ww = full.shape
        axm.imshow(render_grid(full), extent=(-0.5, Ww - 0.5, Hh - 0.5, -0.5),
                   interpolation="nearest")
        axm.plot(xs[:t + 1], ys[:t + 1], "-", color="#f28e2b", lw=2, alpha=0.75, zorder=4)
        wm = win[:t + 1].astype(bool)
        if wm.any():
            axm.scatter(xs[:t + 1][wm], ys[:t + 1][wm], s=42, c=PURPLE, marker="D",
                        edgecolor="k", lw=0.3, zorder=5)
        if win[t]:
            axm.scatter([xs[t]], [ys[t]], s=300, facecolor="none",
                        edgecolor=PURPLE, lw=2.2, zorder=5)
        axm.add_patch(plt.Polygon(agent_triangle(xs[t], ys[t], int(ds[t])),
                                  color="#dd2222", ec="k", lw=0.6, zorder=6))
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title("high-level view", fontsize=11, fontweight="bold")
        # agent view (exactly the observation: an opened marker shows floor)
        axv = fig.add_subplot(gs[0, 2])
        mm = mms[t]
        Vv = mm.shape[0]
        axv.imshow(render_grid(mm), extent=(-0.5, Vv - 0.5, Vv - 0.5, -0.5),
                   interpolation="nearest")
        mcell = Vv // 2
        axv.add_patch(plt.Polygon(agent_triangle(mcell, mcell, int(ds[t])),
                                  color="#dd2222", ec="k", lw=0.6, zorder=6))
        axv.set_xticks([]); axv.set_yticks([])
        axv.set_title("agent view", fontsize=11, fontweight="bold")
        # belief plane: probe decision regions + GRU flow field + trajectory
        axp = fig.add_subplot(gs[1, 2])
        draw_belief_field(axp, field, trained)
        ps1, ps2 = hs[:t + 1] @ e1, hs[:t + 1] @ e2
        axp.plot(ps1, ps2, "-", color="#d1495b", lw=1.6, alpha=0.75, zorder=4)
        if wm.any():
            axp.scatter(ps1[wm], ps2[wm], s=28, c=PURPLE, marker="D", zorder=5, alpha=0.85)
        axp.scatter([ps1[-1]], [ps2[-1]], s=110, c=(PURPLE if win[t] else "#d1495b"),
                    edgecolor="k", zorder=7)
        if win[t]:
            axp.text(0.03, 0.97, "INTERVENTION", transform=axp.transAxes, fontsize=8.5,
                     color=PURPLE, fontweight="bold", va="top")
        axp.set_xlabel(axlab[0], fontsize=8.5); axp.set_ylabel(axlab[1], fontsize=8.5)
        axp.set_title("belief plane: probe regions + GRU flow", fontsize=10.5,
                      fontweight="bold")
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

    if curated:
        spec = evidence_spec(cfg["cue"])
    else:   # legacy default: baseline / activation-steered / diagonal swap
        src, tgt = trained[0], trained[-1]
        spec = [
            dict(mode="none", cue=CUE_NAMES[src], label="BASELINE"),
            dict(mode="force_act", cue=CUE_NAMES[src],
                 label="BEHAVIOR steered via activations (wrong direction)"),
            dict(mode="swap", src=src, tgt=tgt,
                 label=f"MEMORY swapped -> {CUE_NAMES[tgt]} (transient, pre-decision)"),
        ]

    # same key for every episode -> identical env layout across conditions
    key, k = jax.random.split(key)
    episodes = []
    for e in spec:
        cue = e.get("cue") or CUE_NAMES[e["src"]]
        kw = dict(T=110)
        if e["mode"] in ("force", "force_act", "force_thru", "force_lin"):
            ci = CUE_NAMES.index(cue)
            pe = D._env_params(cfg, cue)
            kw["wrong_row"] = int(pe.row_lo) if not IS_DOWN[ci] else int(pe.row_up)
        elif e["mode"] == "swap":
            u = pr["mu"][e["tgt"]] - pr["mu"][e["src"]]
            u = u / np.linalg.norm(u)
            kw.update(u=u, ptgt=float(pr["mu"][e["tgt"]] @ u))
        p, _, tr = rollout(cfg, net, params, cue, k, 1, e["mode"], **kw)
        episodes.append((p, k, tr, f"{cfg['cue']} · {cue} · {e['label']}"))

    for pE, kE, trE, label in episodes:
        xs, ys, ds, mms, hs, win, dn, reach, mtop, mbot = (np.asarray(v)[:, 0] for v in trE)
        s0 = jax.vmap(lambda kk: jreset(kk, pE))(jax.random.split(kE, 1))
        s0 = jax.tree_util.tree_map(lambda x: x[0], s0)
        nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
        ok = bool(reach[nd - 1]) if dn.any() else False
        ep = (pE, s0, xs, ys, ds, mms, hs, win.astype(bool), dn, reach,
              mtop.astype(bool), mbot.astype(bool), label)
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
    from tile_textures import render_grid, agent_triangle, OPEN_TEX_ID

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    PURPLE = "#8e44ad"
    cfg, net, params, pr = load_all(run_dir)
    trained = pr["trained"]; clf = pr["clf"]
    field = belief_field(cfg, net, params, pr)
    e1, e2, axlab = field["e1"], field["e2"], field["axlab"]

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
        xs, ys, ds, mms, hs, win, dn, reach, mtop, mbot = [np.asarray(v)[:, 0] for v in tr]
        nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
        ok = bool(reach[nd - 1]) if dn.any() else False
        # row 0: trajectory (marker doors drawn in their END-of-episode state)
        axm = axs[0, col]
        full = full0.copy()
        tl = nd - 1
        full[p.row_up, p.x_mark] = OPEN_TEX_ID[C.MARK_A] if mtop[tl] else C.MARK_A
        full[p.row_lo, p.x_mark] = OPEN_TEX_ID[C.MARK_B] if mbot[tl] else C.MARK_B
        Hh, Ww = full.shape
        axm.imshow(render_grid(full), extent=(-0.5, Ww - 0.5, Hh - 0.5, -0.5),
                   interpolation="nearest")
        axm.plot(xs[:nd], ys[:nd], "-", color="#f28e2b", lw=1.8, alpha=0.85, zorder=4)
        wm = win[:nd].astype(bool)
        if wm.any():
            axm.scatter(xs[:nd][wm], ys[:nd][wm], s=34, c=PURPLE, marker="D",
                        edgecolor="k", lw=0.3, zorder=5)
        axm.add_patch(plt.Polygon(agent_triangle(xs[nd - 1], ys[nd - 1], int(ds[nd - 1])),
                                  color="#dd2222", ec="k", lw=0.5, zorder=6))
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
        # row 2: plane path over the probe decision regions + GRU flow
        axp = axs[2, col]
        draw_belief_field(axp, field, trained)
        p1, p2 = hs[:nd] @ e1, hs[:nd] @ e2
        axp.plot(p1, p2, "-", color=lc, lw=1.6, alpha=0.9, zorder=4)
        if wm.any():
            axp.scatter(p1[wm], p2[wm], s=24, c=PURPLE, marker="D", zorder=5, alpha=0.85)
        axp.scatter([p1[0]], [p2[0]], marker="s", s=40, c="#999", zorder=6)
        axp.scatter([p1[-1]], [p2[-1]], s=80, c=lc, edgecolor="k", zorder=7)
        axp.set_xlabel(axlab[0], fontsize=8)
        if col == 0:
            axp.set_ylabel(axlab[1], fontsize=8)
        axp.set_title("belief-plane path (probe regions + GRU flow)", fontsize=9)
    fig_.suptitle(f"{cfg['cue']} model · cue {cue} · same episode under the three conditions "
                  "(purple = intervention active)", fontsize=13, fontweight="bold")
    fig_.tight_layout(rect=[0, 0, 1, 0.96])
    fig_.savefig(out, dpi=140)
    print(f"[fig] wrote {out}", flush=True)


def plane(run_dirs, out):
    """Standalone belief-plane figure: probe decision regions + GRU flow field
    + one baseline trajectory per trained cue, one panel per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    fig_, axs = plt.subplots(1, len(run_dirs), figsize=(5.4 * len(run_dirs), 4.6),
                             squeeze=False)
    key = jax.random.PRNGKey(4)
    for j, rdir in enumerate(run_dirs):
        cfg, net, params, pr = load_all(rdir)
        trained = pr["trained"]
        field = belief_field(cfg, net, params, pr, gridn=31)
        ax = axs[0, j]
        draw_belief_field(ax, field, trained, quiver_step=2)
        for c in trained:      # one baseline trajectory per trained cue
            key, k = jax.random.split(key)
            _, _, tr = rollout(cfg, net, params, CUE_NAMES[c], k, 1, "none", T=110)
            hs, dn = np.asarray(tr[4])[:, 0], np.asarray(tr[6])[:, 0]
            nd = int(np.argmax(dn)) + 1 if dn.any() else len(hs)
            p1, p2 = hs[:nd] @ field["e1"], hs[:nd] @ field["e2"]
            ax.plot(p1, p2, "-", color=CUE_COL[c], lw=1.4, alpha=0.9, zorder=4)
            ax.scatter([p1[0]], [p2[0]], marker="s", s=26, c="#777", zorder=6)
        ax.set_xlabel(field["axlab"][0], fontsize=9)
        ax.set_ylabel(field["axlab"][1], fontsize=9)
        ax.set_title(f"{cfg['cue']} model", fontsize=12, fontweight="bold")
    fig_.suptitle("Belief plane: cue-probe decision regions (tint), one-step GRU flow "
                  "(arrows, grey = magnitude), class means, baseline trajectories",
                  fontsize=12, fontweight="bold")
    fig_.tight_layout(rect=[0, 0, 1, 0.93])
    outp = pathlib.Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig_.savefig(outp, dpi=150)
    print(f"[plane] wrote {outp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quant", "video", "video2", "fig", "alpha", "plane"])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--run-dirs", nargs="+", default=None,
                    help="alpha: one panel per run dir")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--out", default=None)
    ap.add_argument("--policy", choices=["greedy", "sample"], default="greedy",
                    help="greedy = argmax(logits); sample = softmax (stochastic) policy")
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    a = ap.parse_args()
    sample = a.policy == "sample"
    if a.cmd in ("alpha", "plane"):
        rds = a.run_dirs or ([a.run_dir] if a.run_dir else None)
        assert rds, f"{a.cmd} needs --run-dirs (or --run-dir)"
        if a.cmd == "alpha":
            alpha_sweep(rds, a.out or "steer_alpha.png", n=a.n, alphas=a.alphas, sample=sample)
        else:
            plane(rds, a.out or "belief_plane.png")
        return
    assert a.run_dir, f"{a.cmd} needs --run-dir"
    if a.cmd == "quant":
        quant(a.run_dir, a.n, sample=sample)
    elif a.cmd == "fig":
        fig(a.run_dir, a.out or "steer2_fig.png")
    elif a.cmd == "video2":
        video(a.run_dir, a.out or "steer2_evidence.mp4", curated=True)
    else:
        video(a.run_dir, a.out or "steer2.mp4")


if __name__ == "__main__":
    main()
