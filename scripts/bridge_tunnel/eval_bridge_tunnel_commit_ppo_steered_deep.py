#!/usr/bin/env python3
"""Steered evaluation of a bridge_tunnel_commit PPO agent: push the hidden
state at the INPUT of the GRU — the incoming recurrent state h_{t-1}, before
the current observation is integrated — toward the SUB-optimal skill (mine on
lakes maps, build on rocky maps) and measure whether the auxiliary *belief*
readout moves along with the behavior.

Hypothesis under test: because the belief is a latent factor that the behavior
is entangled with, the cheapest way for an activation-space intervention to
flip the committed skill is to move the belief itself — the steered agent
"hallucinates" the other map archetype rather than keeping an intact belief
and merely switching strategy (which is what a human would do). Steering the
recurrent INPUT state makes this a fair test: the gradient clamps
differentiate through the full GRU step (recurrence + actor head), so descent
is free to route the behavior change through whichever latent factors the
recurrent dynamics make cheapest — belief included — rather than being
confined to the actor head's row space.

Every strategy edits the pre-GRU state at ONE place in the rollout; the
steered state is then passed through the GRU, and the resulting post-GRU
state feeds all heads AND the next step, so edits persist only through what
the recurrence retains of them. Directions are computed by the pluggable
``DIRECTION_STRATEGIES`` (pre- and post-GRU states live in the same space —
h_{t-1} IS the previous step's post-GRU state — so directions estimated from
post-GRU hiddens apply unchanged):

* ``class-mean``   (default) unit-norm difference of mean GRU hidden states
                   collected from unsteered rollouts on rocky vs lakes maps.
* ``belief-head``  unit-norm difference of the belief head's rocky and lakes
                   weight rows (steers straight along the belief readout).

For those two, the direction is always the lakes→rocky axis; per map category
it is applied with sign +1 on lakes (→ rocky ⇒ mine), −1 on rocky (→ lakes ⇒
build), and 0 on balanced maps (no-steer control rows).

Two more strategies run the INVERSE experiment — clamping the BEHAVIOR while
never referencing the belief, so any movement of the belief readout is pure
side effect: does the agent keep an intact belief while its strategy is
constrained (the human mode), or does the behavioral clamp drag it along?

* ``suppress-skill``   each env step, take up to ``--clamp-iters`` unit-norm
  steps of size α on the GRU INPUT state, descending the autograd gradient of
  log π(prohibited action) w.r.t. that input state — backpropagated THROUGH
  the GRU step, no closed form — until that probability drops below
  ``--clamp-target`` (build is prohibited on lakes maps, mine on rocky maps;
  ``--push-beta`` > 0 additionally pushes UP the sub-optimal skill's log-prob).
* ``substitute-skill``  suppression as above PLUS a floor on the sub-optimal
  skill: once an agent has covered ``--sub-from-progress`` (default half) of
  its spawn→target distance, π(sub-optimal action) is pushed up whenever it
  falls below ``--sub-floor`` (default 0.05). Both constraints are solved
  jointly per sample per step (gradients through the GRU step; minimal edit:
  samples satisfying both are untouched) — the "substitution" scenario: the
  optimal tool is forbidden and the alternative is kept persistently
  available.

Outputs (``--out-prefix`` +):
  ``_commit_matrix.png``   baseline vs steered 3×3 commit matrices side by side
  ``_belief.png``          per-category belief-scalar (P(lakes)−P(rocky)) time
                           series, baseline vs steered
  ``_actionprob.png``      per-category π(build)/π(mine) time series, baseline
                           vs steered (verifies the behavior clamp bites)
  ``_traj.png``            steered trajectory grid (same encoding as the
                           unsteered eval script)
  ``_dose.png``            behavior-shift vs belief-shift dose–response
                           (only when >1 alpha is given)
  ``_results.json``        all numeric results
  ``_steer_dir_<strategy>.npy``  the (H,) direction — loadable by the
                           mechinterp steering kit's ``steer.py --inject``

Each ``--alphas`` entry is either one number (same strength for both steered
categories) or a ``lakes:rocky`` pair — e.g. ``--alphas 0.25:1.0`` steers the
lakes scenario gently while pushing the rocky one much harder (useful because
the two scenarios saturate/break at different strengths).

    python scripts/bridge_tunnel/eval_bridge_tunnel_commit_ppo_steered.py \\
        --checkpoint released_models/bridge_tunnel_commit/ppo_gru_commit_aux_belief.pt \\
        --alphas 0.75:3.0,1.5:5 --sub-from-progress 0.25 --sub-floor 0.08 
        --strategy substitute-skill --out-prefix outputs/steering/ppo_substitute_deep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel import generate_commit_map, tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel.mapgen import CATEGORIES  # noqa: E402
from eval_bridge_tunnel_commit_ppo import (  # noqa: E402
    _FACE_DELTA, COMMIT_NAMES, _draw_commit_path, _load_policy)

# belief head class order matches CATEGORIES: balanced=0, lakes=1, rocky=2
BELIEF2I = {c: i for i, c in enumerate(CATEGORIES)}
# The steering direction is the unit lakes→rocky axis, so sign +1 pushes the
# hidden state toward "rocky" (⇒ mine) and −1 toward "lakes" (⇒ build). We
# steer each category toward its SUB-optimal skill; balanced = no-steer control.
STEER_SIGN = {"balanced": 0.0, "lakes": +1.0, "rocky": -1.0}
SUBOPT_COMMIT = {"lakes": 2, "rocky": 1}          # mine on lakes, build on rocky
# env action ids (Discrete(6)): 4 = build (water→wood, commits BUILD),
# 5 = mine (rock→grass, commits MINE)
A_BUILD, A_MINE = 4, 5
ACTION_NAMES = ["up", "down", "left", "right", "build", "mine"]
PROHIBIT_ACTION = {"lakes": A_BUILD, "rocky": A_MINE}   # the optimal skill, forbidden
PUSH_ACTION = {"lakes": A_MINE, "rocky": A_BUILD}       # the sub-optimal skill


# ── per-category steering strengths ──
# An alpha setting is a (α_lakes, α_rocky) pair; balanced is never steered.

def parse_alpha_token(tok: str) -> tuple[float, float]:
    """'0.25' → (0.25, 0.25); '0.25:1.0' → α_lakes=0.25, α_rocky=1.0."""
    parts = tok.split(":")
    if len(parts) == 1:
        v = float(parts[0])
        return (v, v)
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f"bad alpha token {tok!r}: expected 'a' or 'a_lakes:a_rocky'")


def alpha_label(pair: tuple[float, float]) -> str:
    if pair[0] == pair[1]:
        return f"{pair[0]:g}"
    return f"{pair[0]:g}(lakes):{pair[1]:g}(rocky)"


def cat_alpha(pair: tuple[float, float], category: str) -> float:
    return {"lakes": pair[0], "rocky": pair[1]}.get(category, 0.0)


# ───────────────────────────── steering ─────────────────────────────

def steer_hidden(h: torch.Tensor, t: int, direction: torch.Tensor,
                 alpha: float, sign: float, steer_from: int, steer_to: int) -> torch.Tensor:
    """THE steering intervention — the single place to iterate on strategies.

    Called once per env step with the PRE-GRU hidden ``h`` of shape (1, B, H):
    the recurrent state about to be fed into the GRU together with the current
    observation. Whatever is returned goes THROUGH the GRU; the resulting
    post-GRU state feeds the actor/critic/belief heads at this step AND the
    next step, so the edit persists only through what the recurrence retains
    of it. The default strategy is a constant additive push along a unit-norm
    ``direction`` while ``steer_from <= t < steer_to``.

    Ideas to iterate on: decaying alpha, projecting out the direction instead
    of adding, clamping the belief-logit gap, steering only until commit, …
    """
    if sign == 0.0 or alpha == 0.0 or not (steer_from <= t < steer_to):
        return h
    return h + (alpha * sign) * direction.view(1, 1, -1)


def steer_suppress_action(h: torch.Tensor, t: int, policy, feat: torch.Tensor,
                          prohibit: int, alpha: float, max_iters: int,
                          target_prob: float, steer_from: int, steer_to: int,
                          push: int | None = None, push_beta: float = 0.0) -> torch.Tensor:
    """Behavior-clamp steering: minimize π(prohibited action) by gradient
    descent on the GRU INPUT state, differentiating THROUGH the recurrence.

    ``h`` is the pre-GRU state (1, B, H) and ``feat`` the current
    observation's encoder feature (B, E) — constant within the step, so each
    iteration re-runs only GRU cell + actor head. The composed map
    h_in ↦ gru(feat, h_in) ↦ actor is nonlinear, so ∇_{h_in} log π(prohibit)
    comes from autograd rather than a closed form. Each iteration takes one
    unit-norm step of size ``alpha`` against that gradient, per sample,
    stopping early once π(prohibit) < ``target_prob`` (so the edit is
    minimal: samples already below target are untouched). With ``push_beta``
    > 0 the sub-optimal skill's log-prob is simultaneously pushed up. The
    belief head is never referenced — any belief movement is a side effect to
    be measured.
    """
    if alpha == 0.0 or not (steer_from <= t < steer_to):
        return h
    feat_seq = feat.detach()[None]                         # (1, B, E)
    x = h.squeeze(0).detach()                              # (B, H) pre-GRU state
    for _ in range(max_iters):
        # cudnn's fused GRU cannot run backward on a module in eval() mode;
        # the native path can, and this 1-step (B, H) cell is tiny anyway
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            xg = x.clone().requires_grad_(True)
            _, h_out = policy.gru(feat_seq, xg[None])      # through the recurrence
            logp = torch.log_softmax(policy.actor(h_out.squeeze(0)), dim=-1)
            p = logp.detach().exp()                        # (B, A)
            need = p[:, prohibit] > target_prob
            if not need.any():
                break
            obj = logp[:, prohibit]                        # descend this
            if push is not None and push_beta > 0.0:
                obj = obj - push_beta * logp[:, push]
            # samples are independent through the GRU, so rows of g are the
            # per-sample gradients ∇_{h_in} obj
            (g,) = torch.autograd.grad(obj.sum(), xg)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = torch.where(need[:, None], x - alpha * g, x)
    return x[None]


def steer_substitute(h: torch.Tensor, t: int, policy, feat: torch.Tensor,
                     prohibit: int, push: int,
                     alpha: float, max_iters: int, target_prob: float,
                     floor_prob: float, sub_from_progress: float,
                     progress: np.ndarray, steer_from: int, steer_to: int) -> torch.Tensor:
    """Substitution clamp: forbid the optimal skill AND keep the sub-optimal
    one available — gradient steps on the GRU INPUT state, through the GRU.

    Per sample and per iteration, two conditions are checked: (a) π(prohibit)
    > ``target_prob`` (always enforced, as in :func:`steer_suppress_action`)
    and (b) π(push) < ``floor_prob`` for samples that have covered at least
    ``sub_from_progress`` of their spawn→target distance. The combined
    objective (ascend the pushed log-prob, descend the prohibited one — each
    term only where its condition is violated) is differentiated w.r.t. the
    pre-GRU state through gru(feat, h_in) ↦ actor, and one unit-norm step of
    size ``alpha`` is taken until both constraints hold or ``max_iters`` is
    hit. Samples satisfying both are untouched. The belief is never
    referenced.
    """
    if alpha == 0.0 or not (steer_from <= t < steer_to):
        return h
    feat_seq = feat.detach()[None]                         # (1, B, E)
    x = h.squeeze(0).detach()                              # (B, H) pre-GRU state
    in_zone = torch.from_numpy(np.asarray(progress) >= sub_from_progress).to(x.device)
    for _ in range(max_iters):
        # cudnn's fused GRU cannot run backward on a module in eval() mode;
        # the native path can, and this 1-step (B, H) cell is tiny anyway
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            xg = x.clone().requires_grad_(True)
            _, h_out = policy.gru(feat_seq, xg[None])      # through the recurrence
            logp = torch.log_softmax(policy.actor(h_out.squeeze(0)), dim=-1)
            p = logp.detach().exp()                        # (B, A)
            need_sup = p[:, prohibit] > target_prob
            need_boost = in_zone & (p[:, push] < floor_prob)
            need = need_sup | need_boost
            if not need.any():
                break
            obj = (logp[:, push] * need_boost.float()
                   - logp[:, prohibit] * need_sup.float())  # ascend this
            (g,) = torch.autograd.grad(obj.sum(), xg)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = torch.where(need[:, None], x + alpha * g, x)
    return x[None]


def make_steer_fn(strategy: str, category: str, policy,
                  direction: torch.Tensor | None, alpha: float,
                  steer_from: int, steer_to: int, clamp_iters: int = 10,
                  clamp_target: float = 0.01, push_beta: float = 0.0,
                  sub_floor: float = 0.05, sub_from_progress: float = 0.5):
    """Bind the strategy's steering for one map category (None ⇒ unsteered).
    Balanced maps are always the no-steer control row. The bound function is
    called as ``steer_fn(h, t, ctx)`` on the PRE-GRU hidden, with
    ``ctx["progress"]`` = per-env fraction of the spawn→target distance
    covered and ``ctx["feat"]`` = the current observation's encoder feature
    (the gradient clamps need it to differentiate through the GRU step)."""
    if alpha == 0.0:
        return None
    if strategy in ("suppress-skill", "substitute-skill"):
        prohibit = PROHIBIT_ACTION.get(category)
        if prohibit is None:
            return None
        push = PUSH_ACTION[category]
        if strategy == "substitute-skill":
            return lambda h, t, ctx: steer_substitute(
                h, t, policy, ctx["feat"], prohibit, push, alpha, clamp_iters,
                clamp_target, sub_floor, sub_from_progress, ctx["progress"],
                steer_from, steer_to)
        return lambda h, t, ctx: steer_suppress_action(
            h, t, policy, ctx["feat"], prohibit, alpha, clamp_iters,
            clamp_target, steer_from, steer_to, push=push, push_beta=push_beta)
    sign = STEER_SIGN[category]
    if direction is None or sign == 0.0:
        return None
    return lambda h, t, ctx: steer_hidden(h, t, direction, alpha, sign, steer_from, steer_to)


def direction_belief_head(policy, **_):
    """Unit lakes→rocky axis straight from the belief head weight rows."""
    W = policy.belief.weight.detach()                     # (3, H)
    d = W[BELIEF2I["rocky"]] - W[BELIEF2I["lakes"]]
    return d / d.norm(), float(d.norm())


def direction_class_mean(policy, *, view_size, env_size, env_width, cargs, device,
                         n_maps, n_traj, seed_start, max_steps):
    """Unit lakes→rocky axis from class-mean GRU hiddens of unsteered rollouts.
    Estimated on post-GRU states, applied to the pre-GRU input — the same
    space, since h_{t-1} is the previous step's post-GRU state."""
    means = {}
    for cat in ("lakes", "rocky"):
        chunks = []
        for j in range(n_maps):
            rec = _make_map(env_size, env_width, seed_start + j, cat, cargs)
            out = batched_rollout_steered(policy, rec, n_traj, view_size,
                                          max_steps, device, collect_hidden=True)
            chunks.append(out["hiddens"])
        means[cat] = np.concatenate(chunks).mean(axis=0)
    diff = means["rocky"] - means["lakes"]
    raw_norm = float(np.linalg.norm(diff))
    d = torch.from_numpy(diff.astype(np.float32)).to(device)
    return d / d.norm(), raw_norm


DIRECTION_STRATEGIES = {
    "class-mean": direction_class_mean,
    "belief-head": direction_belief_head,
}


# ───────────────────────────── rollout ─────────────────────────────

def _make_map(env_size, env_width, seed, category, cargs):
    gh = cargs.get("goal_half", 1)
    return generate_commit_map(size=env_size, width=env_width, seed=seed,
                               category=category,
                               tree_frac=cargs.get("tree_frac", 0.03),
                               goal_half=(gh if (gh is not None and gh >= 0) else None))


@torch.no_grad()
def batched_rollout_steered(policy, rec, n_traj, view_size, max_steps, device,
                            steer_fn=None, collect_hidden=False):
    """``n_traj`` stochastic rollouts on one fixed map in lockstep, mirroring
    the unsteered eval's ``batched_rollout`` but (a) applying ``steer_fn`` to
    the PRE-GRU recurrent state each step (the steered state then goes
    through the GRU before reaching any head) and (b) recording the belief
    softmax per step.

    Returns a dict: trajs, reached, final_commit, commit_pts, mine_pts,
    bridge_pts, commits, belief_probs (n_traj, max_steps, 3) NaN-padded past
    each episode's end, and (optionally) the stacked active-step hiddens.
    """
    H, W = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=H, width=W, view_size=view_size,
                                  max_steps=max_steps) for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    active = np.ones(n_traj, dtype=bool)
    reached = np.zeros(n_traj, dtype=bool)
    final_commit = np.zeros(n_traj, dtype=np.int64)
    trajs = [[tuple(e._pos)] for e in envs]
    commits = [[0] for _ in envs]
    commit_pts, mine_pts, bridge_pts = [], [], []
    belief_probs = np.full((n_traj, max_steps, 3), np.nan, dtype=np.float32)
    action_probs = np.full((n_traj, max_steps, policy.actor.out_features),
                           np.nan, dtype=np.float32)
    progress_tr = np.full((n_traj, max_steps), np.nan, dtype=np.float32)
    hiddens = []
    # spatial progress = fraction of the spawn→target distance covered
    spawn_xy = np.asarray(rec.spawn, dtype=np.float64)
    target_xy = np.asarray(rec.target, dtype=np.float64)
    total_dist = max(float(np.linalg.norm(target_xy - spawn_xy)), 1e-6)

    for t in range(max_steps):
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs])).to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs])).to(device)
        pos = np.asarray([e._pos for e in envs], dtype=np.float64)
        progress = np.clip(
            1.0 - np.linalg.norm(pos - target_xy, axis=1) / total_dist, 0.0, 1.0
        ).astype(np.float32)
        progress_tr[active, t] = progress[active]
        # encode once per step (the obs feature doesn't depend on the hidden),
        # steer the PRE-GRU state, then run the GRU cell. Episodes never reset
        # mid-rollout here, so _gru_forward's done-masking is a no-op and the
        # direct gru call is equivalent.
        feat = policy._encode({"minimap": mm, "scalars": sc})   # (B, E)
        if steer_fn is not None:
            h = steer_fn(h, t, {"progress": progress, "feat": feat})
        _, h = policy.gru(feat[None], h)       # steered input → (1, B, H)
        x = h.squeeze(0)                       # (B, H): post-GRU state feeds ALL heads
        logits, _ = policy._heads(x)
        action_probs[active, t] = torch.softmax(logits, dim=-1).cpu().numpy()[active]
        if policy.belief is not None:
            bp = torch.softmax(policy.belief(x), dim=-1).cpu().numpy()
            belief_probs[active, t] = bp[active]
        if collect_hidden and active.any():
            hiddens.append(x.cpu().numpy()[active])
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, r, term, trunc, info = e.step(int(acts[i]))
            obs[i] = o
            trajs[i].append(tuple(e._pos))
            commits[i].append(int(info["commit"]))
            final_commit[i] = info["commit"]
            if info["committed_now"]:
                commit_pts.append(tuple(e._pos))
            if info["mined"] or info["placed"]:
                dr, dc = _FACE_DELTA[info["facing"]]
                cell = (e._pos[0] + dr, e._pos[1] + dc)
                (mine_pts if info["mined"] else bridge_pts).append(cell)
            if term:
                reached[i] = True; active[i] = False
            elif trunc:
                active[i] = False
        if not active.any():
            break
    return dict(trajs=trajs, reached=reached, final_commit=final_commit,
                commit_pts=commit_pts, mine_pts=mine_pts, bridge_pts=bridge_pts,
                commits=commits, belief_probs=belief_probs, action_probs=action_probs,
                progress=progress_tr,
                hiddens=(np.concatenate(hiddens) if hiddens else None))


# ───────────────────────────── evaluation ─────────────────────────────

def run_eval(policy, view_size, env_size, env_width, cargs, device, *,
             steer_for_category, n_maps, n_traj, seed_start, max_steps):
    """Commit matrix + belief/action-prob traces for one steering setting.
    ``steer_for_category(cat)`` returns the bound steer_fn (or None)."""
    counts = np.zeros((3, 3), dtype=np.float64)
    succ = {c: [] for c in CATEGORIES}
    beliefs = {c: [] for c in CATEGORIES}                 # (episodes, T, 3) chunks
    aprobs = {c: [] for c in CATEGORIES}                  # (episodes, T, A) chunks
    progs = {c: [] for c in CATEGORIES}                   # (episodes, T) chunks
    for ci, cat in enumerate(CATEGORIES):
        steer_fn = steer_for_category(cat)
        for j in range(n_maps):
            rec = _make_map(env_size, env_width, seed_start + j, cat, cargs)
            out = batched_rollout_steered(policy, rec, n_traj, view_size,
                                          max_steps, device, steer_fn=steer_fn)
            for v in out["final_commit"]:
                counts[ci, int(v)] += 1
            succ[cat].extend(out["reached"].tolist())
            beliefs[cat].append(out["belief_probs"])
            aprobs[cat].append(out["action_probs"])
            progs[cat].append(out["progress"])
    matrix = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    succ = {c: float(np.mean(v)) if v else 0.0 for c, v in succ.items()}
    beliefs = {c: np.concatenate(v) for c, v in beliefs.items()}
    aprobs = {c: np.concatenate(v) for c, v in aprobs.items()}
    progs = {c: np.concatenate(v) for c, v in progs.items()}
    return matrix, succ, counts, beliefs, aprobs, progs


def summarize(matrix, succ, beliefs, aprobs, progs, late_from=0.5):
    """Per-category scalar metrics from one run. ``pi_action_mean_late``
    restricts to steps past ``late_from`` of the spawn→target distance (the
    substitution window)."""
    res = {}
    for ci, cat in enumerate(CATEGORIES):
        bp = beliefs[cat]                                  # (N, T, 3)
        valid_a = np.isfinite(aprobs[cat][..., 0])         # (N, T)
        ap = aprobs[cat][valid_a]                          # (K, A) valid steps
        late = aprobs[cat][valid_a & (progs[cat] >= late_from)]
        scalar = bp[..., BELIEF2I["lakes"]] - bp[..., BELIEF2I["rocky"]]
        valid = np.isfinite(bp[..., 0])
        # final-step belief per episode
        finals = [scalar[i, np.where(valid[i])[0][-1]] for i in range(len(scalar))
                  if valid[i].any()]
        am = bp[valid].argmax(axis=-1)
        res[cat] = {
            "commit_fracs": {n: float(matrix[ci, k]) for k, n in enumerate(COMMIT_NAMES)},
            "success": succ[cat],
            "belief_scalar_mean": float(np.nanmean(scalar)),
            "belief_scalar_final": float(np.mean(finals)) if finals else float("nan"),
            "belief_argmax_fracs": {c: float((am == BELIEF2I[c]).mean())
                                    for c in CATEGORIES},
            "pi_action_mean": {"build": float(ap[:, A_BUILD].mean()),
                               "mine": float(ap[:, A_MINE].mean())},
            "pi_action_mean_late": (
                {"build": float(late[:, A_BUILD].mean()),
                 "mine": float(late[:, A_MINE].mean())} if len(late) else None),
        }
    return res


# ───────────────────────────── plotting ─────────────────────────────

def plot_matrix_pair(mats, succs, titles, suptitle, out_path):
    fig, axes = plt.subplots(1, len(mats), figsize=(5.6 * len(mats), 4.4))
    fig.subplots_adjust(wspace=0.45)
    axes = np.atleast_1d(axes)
    for ax, m, s, ttl in zip(axes, mats, succs, titles):
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(COMMIT_NAMES, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"{c}\n(succ {s[c]:.0%})" for c in CATEGORIES], fontsize=9)
        ax.set_xlabel("committed skill", fontsize=10)
        for i in range(3):
            for j in range(3):
                v = m[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if v < 0.6 else "black", fontweight="bold")
        ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel("map category (belief)", fontsize=10)
    fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02, label="fraction of episodes")
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def _mean_trace(bp):
    """(N,T,3) NaN-padded → (mean scalar[t], alive fraction[t], last useful t)."""
    scalar = bp[..., BELIEF2I["lakes"]] - bp[..., BELIEF2I["rocky"]]
    alive = np.isfinite(scalar).sum(axis=0)
    tmax_arr = np.where(alive >= max(4, 0.1 * len(scalar)))[0]
    tmax = int(tmax_arr[-1]) + 1 if len(tmax_arr) else scalar.shape[1]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(scalar[:, :tmax], axis=0)
        lo = np.nanpercentile(scalar[:, :tmax], 25, axis=0)
        hi = np.nanpercentile(scalar[:, :tmax], 75, axis=0)
    return mean, lo, hi, tmax


def plot_belief_traces(base_beliefs, steer_beliefs, alpha_pair, strategy, desc,
                       steer_from, steer_to, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    for ax, cat in zip(axes, CATEGORIES):
        for bp, color, label in ((base_beliefs[cat], "#1f5fd0", "baseline"),
                                 (steer_beliefs[cat], "#d62728", "steered")):
            mean, lo, hi, tmax = _mean_trace(bp)
            ts = np.arange(tmax)
            ax.plot(ts, mean, color=color, lw=1.6, label=label)
            ax.fill_between(ts, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.axhline(0.0, color="gray", lw=0.7, ls=":")
        if cat != "balanced":
            ax.axvspan(steer_from, min(steer_to, ax.get_xlim()[1]),
                       color="#d62728", alpha=0.04)
            ax.set_title(f"{cat}  ({desc[cat]}, α={cat_alpha(alpha_pair, cat):g})",
                         fontsize=10)
        else:
            ax.set_title(f"{cat}  ({desc[cat]})", fontsize=10)
        ax.set_xlabel("env step")
    axes[0].set_ylabel("belief scalar  P(lakes) − P(rocky)")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"belief readout under hidden-state steering  ·  {strategy}  ·  "
                 "band = IQR over episodes", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def plot_action_probs(base_aprobs, steer_aprobs, alpha_pair, strategy, desc, out_path):
    """Mean π(build)/π(mine) over time per category, baseline vs steered —
    the direct check that the behavior clamp (or axis push) bites at the
    action level."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    for ax, cat in zip(axes, CATEGORIES):
        for ap, color, run in ((base_aprobs[cat], "#1f5fd0", "baseline"),
                               (steer_aprobs[cat], "#d62728", "steered")):
            for a_id, ls, aname in ((A_BUILD, "-", "build"), (A_MINE, "--", "mine")):
                trace = ap[..., a_id]
                alive = np.isfinite(trace).sum(axis=0)
                ok = np.where(alive >= max(4, 0.1 * len(trace)))[0]
                tmax = int(ok[-1]) + 1 if len(ok) else trace.shape[1]
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(trace[:, :tmax], axis=0)
                ax.plot(np.arange(tmax), mean, ls, color=color, lw=1.4,
                        label=f"{run} π({aname})")
        ax.set_yscale("log")
        if cat != "balanced":
            ax.set_title(f"{cat}  ({desc[cat]}, α={cat_alpha(alpha_pair, cat):g})",
                         fontsize=10)
        else:
            ax.set_title(f"{cat}  ({desc[cat]})", fontsize=10)
        ax.set_xlabel("env step")
    axes[0].set_ylabel("mean action probability (log)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"skill-action probabilities under steering  ·  {strategy}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def plot_dose_response(all_summaries, alpha_keys, strategy, out_path):
    """Behavior shift vs belief shift as alpha grows (both 'toward the steer
    target': suboptimal-commit fraction, and signed belief scalar). Each
    category is plotted against ITS OWN alpha, so lakes:rocky pairs land at
    the strength that category actually experienced."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for cat, color in (("lakes", "#1f77b4"), ("rocky", "#8c564b")):
        sub_name = COMMIT_NAMES[SUBOPT_COMMIT[cat]]
        xs = [cat_alpha(k, cat) for k in alpha_keys]
        beh = [all_summaries[k][cat]["commit_fracs"][sub_name] for k in alpha_keys]
        # signed toward the steer target: on lakes we push toward rocky, so
        # belief-toward-target = P(rocky) − P(lakes) = −scalar; on rocky, +scalar
        bel = [-STEER_SIGN[cat] * all_summaries[k][cat]["belief_scalar_mean"]
               for k in alpha_keys]
        axes[0].plot(xs, beh, "o-", color=color, label=f"{cat} → {sub_name}")
        axes[1].plot(xs, bel, "o-", color=color, label=f"{cat}")
    axes[0].set_ylabel("frac committed SUB-optimal skill"); axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("mean belief toward steer target\n(sign-aligned scalar)")
    axes[1].set_ylim(-1.05, 1.05); axes[1].axhline(0, color="gray", lw=0.7, ls=":")
    for ax in axes:
        ax.set_xlabel("steering strength α (per category)"); ax.legend(fontsize=8)
    fig.suptitle(f"dose–response: behavior vs belief  ·  {strategy}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def plot_grid_steered(policy, view_size, env_size, env_width, cargs, device, *,
                      steer_for_category,
                      n_seeds, n_traj, seed_start, max_steps, title, out_path):
    fig, axes = plt.subplots(len(CATEGORIES), n_seeds,
                             figsize=(n_seeds * 3.0, len(CATEGORIES) * 2.0))
    axes = np.asarray(axes).reshape(len(CATEGORIES), n_seeds)
    for ci, cat in enumerate(CATEGORIES):
        steer_fn = steer_for_category(cat)
        for sj in range(n_seeds):
            rec = _make_map(env_size, env_width, seed_start + sj, cat, cargs)
            out = batched_rollout_steered(policy, rec, n_traj, view_size,
                                          max_steps, device, steer_fn=steer_fn)
            ax = axes[ci, sj]
            ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            for i, tr in enumerate(out["trajs"]):
                _draw_commit_path(ax, tr, out["commits"][i], out["reached"][i])
            if out["mine_pts"]:
                m = np.array(out["mine_pts"]); ax.scatter(m[:, 1], m[:, 0], color="yellow", s=6, alpha=0.18, zorder=3, linewidths=0)
            if out["bridge_pts"]:
                b = np.array(out["bridge_pts"]); ax.scatter(b[:, 1], b[:, 0], color="red", s=6, alpha=0.18, zorder=3, linewidths=0)
            ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=22, marker="s", edgecolors="k", zorder=5)
            fc = out["final_commit"]; reached = out["reached"]
            fb = float((fc == 1).mean()); fm = float((fc == 2).mean()); fn = float((fc == 0).mean())
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{cat} s{seed_start+sj}  succ {reached.mean():.0%}\n"
                         f"build {fb:.0%}/mine {fm:.0%}/none {fn:.0%}", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


# ───────────────────────────── main ─────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="PPO checkpoint WITH the auxiliary belief head "
                        "(e.g. released_models/bridge_tunnel_commit/ppo_gru_commit_aux_belief.pt)")
    p.add_argument("--out-prefix", type=Path,
                   default=Path("paper/figures/bridge_tunnel_commit/ppo_steered"))
    p.add_argument("--strategy",
                   choices=sorted(DIRECTION_STRATEGIES) + ["suppress-skill", "substitute-skill"],
                   default="class-mean")
    p.add_argument("--alphas", default="0.25,0.5,1.0,2.0",
                   help="comma-separated steering strengths; each entry is one "
                        "number (both categories) or a lakes:rocky pair (e.g. "
                        "0.25:1.0 steers lakes gently, rocky harder). Baseline "
                        "α=0 is always run; the first entry is used for the "
                        "headline figures. NOTE: α now applies to the PRE-GRU "
                        "state, which the recurrence can attenuate or amplify "
                        "before it reaches the heads — re-check the "
                        "dose-response rather than reusing calibrations from "
                        "the old post-GRU intervention point. For "
                        "suppress/substitute-skill, α is the per-iteration step "
                        "size of the clamp (≤ clamp-iters iterations per env step)")
    p.add_argument("--steer-from", type=int, default=0)
    p.add_argument("--steer-to", type=int, default=10**9)
    p.add_argument("--clamp-iters", type=int, default=10,
                   help="suppress-skill: max gradient steps per env step")
    p.add_argument("--clamp-target", type=float, default=0.01,
                   help="suppress-skill: stop steering a sample once "
                        "π(prohibited) is below this")
    p.add_argument("--push-beta", type=float, default=0.0,
                   help="suppress-skill: weight for also pushing UP the "
                        "sub-optimal skill's log-prob (0 = pure suppression)")
    p.add_argument("--sub-floor", type=float, default=0.05,
                   help="substitute-skill: minimum π(sub-optimal action) "
                        "enforced inside the substitution window")
    p.add_argument("--sub-from-progress", type=float, default=0.5,
                   help="substitute-skill: the floor kicks in once this "
                        "fraction of the spawn→target distance is covered")
    p.add_argument("--matrix-maps", type=int, default=20, help="held-out maps/category")
    p.add_argument("--matrix-traj", type=int, default=16, help="stochastic rollouts/map")
    p.add_argument("--grid-seeds", type=int, default=4)
    p.add_argument("--grid-traj", type=int, default=120)
    p.add_argument("--calib-maps", type=int, default=8,
                   help="class-mean strategy: maps/category for direction estimation")
    p.add_argument("--calib-traj", type=int, default=8)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--calib-seed-start", type=int, default=20_000,
                   help="disjoint from eval seeds so the direction is not fit "
                        "on the evaluated maps")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.sample_seed)
    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    if policy.belief is None:
        sys.exit("checkpoint has no belief head — this experiment needs the aux-belief "
                 "agent (ppo_gru_commit_aux_belief.pt); the belief readout is the "
                 "measurement, not just the steering handle.")
    tag = args.checkpoint.stem
    alphas = [parse_alpha_token(tok) for tok in args.alphas.split(",")]

    if args.strategy in DIRECTION_STRATEGIES:
        direction, raw_norm = DIRECTION_STRATEGIES[args.strategy](
            policy, view_size=view_size, env_size=env_size, env_width=env_width,
            cargs=cargs, device=device, n_maps=args.calib_maps, n_traj=args.calib_traj,
            seed_start=args.calib_seed_start, max_steps=args.max_steps)
        dir_path = Path(str(args.out_prefix) + f"_steer_dir_{args.strategy}.npy")
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(dir_path, direction.cpu().numpy())
        print(f"[{args.strategy}] unit lakes→rocky direction saved to {dir_path} "
              f"(raw ‖Δ‖ before normalization: {raw_norm:.3f})")
        desc = {"balanced": "control, unsteered",
                "lakes": "steered toward rocky", "rocky": "steered toward lakes"}
    else:
        direction, raw_norm = None, None
        print(f"[{args.strategy}] state-dependent behavior clamp on the GRU "
              f"INPUT state (gradients through the GRU step): suppress "
              f"π(build) on lakes / π(mine) on rocky below {args.clamp_target:g} "
              f"(≤{args.clamp_iters} × α unit steps per env step, "
              f"push-beta={args.push_beta:g})")
        if args.strategy == "substitute-skill":
            print(f"    + substitution floor: π(sub-optimal skill) kept above "
                  f"{args.sub_floor:g} once {args.sub_from_progress:.0%} of the "
                  f"spawn→target distance is covered")
            desc = {"balanced": "control, unsteered",
                    "lakes": "build→mine substitution",
                    "rocky": "mine→build substitution"}
        else:
            desc = {"balanced": "control, unsteered",
                    "lakes": "π(build) suppressed", "rocky": "π(mine) suppressed"}

    def steer_factory(alpha_pair):
        def for_category(cat):
            return make_steer_fn(args.strategy, cat, policy, direction,
                                 cat_alpha(alpha_pair, cat),
                                 args.steer_from, args.steer_to,
                                 args.clamp_iters, args.clamp_target, args.push_beta,
                                 args.sub_floor, args.sub_from_progress)
        return for_category

    # baseline (α=0) + each requested alpha setting (α_lakes, α_rocky)
    all_alphas = [(0.0, 0.0)] + alphas
    matrices, succs, beliefs, aprobs, summaries = {}, {}, {}, {}, {}
    for a in all_alphas:
        m, s, _, b, ap, pg = run_eval(
            policy, view_size, env_size, env_width, cargs, device,
            steer_for_category=steer_factory(a), n_maps=args.matrix_maps,
            n_traj=args.matrix_traj, seed_start=args.eval_seed_start,
            max_steps=args.max_steps)
        matrices[a], succs[a], beliefs[a], aprobs[a] = m, s, b, ap
        summaries[a] = summarize(m, s, b, ap, pg, late_from=args.sub_from_progress)
        print(f"\nα={alpha_label(a)} commit matrix (rows=category, cols={COMMIT_NAMES}):")
        for i, c in enumerate(CATEGORIES):
            sm = summaries[a][c]
            print(f"  {c:9s} {m[i].round(2)}  succ={s[c]:.2%}  "
                  f"belief_scalar mean={sm['belief_scalar_mean']:+.3f} "
                  f"final={sm['belief_scalar_final']:+.3f}  "
                  f"argmax={{{', '.join(f'{k}: {v:.0%}' for k, v in sm['belief_argmax_fracs'].items())}}}  "
                  f"π(build)={sm['pi_action_mean']['build']:.4f} "
                  f"π(mine)={sm['pi_action_mean']['mine']:.4f}")

    a1 = alphas[0]
    base_key = (0.0, 0.0)
    # headline conclusion: did behavior move, and did belief move with it?
    print(f"\n── steering effect (baseline → α={alpha_label(a1)}) ──")
    for cat in ("lakes", "rocky"):
        sub = COMMIT_NAMES[SUBOPT_COMMIT[cat]]
        b0, b1 = summaries[base_key][cat], summaries[a1][cat]
        print(f"  {cat:6s} (α={cat_alpha(a1, cat):g}): "
              f"P(commit {sub}) {b0['commit_fracs'][sub]:.0%} → "
              f"{b1['commit_fracs'][sub]:.0%}   belief scalar "
              f"{b0['belief_scalar_mean']:+.3f} → {b1['belief_scalar_mean']:+.3f}   "
              f"success {b0['success']:.0%} → {b1['success']:.0%}")
        if args.strategy in ("suppress-skill", "substitute-skill"):
            pname = ACTION_NAMES[PROHIBIT_ACTION[cat]]
            print(f"          clamp bite: mean π({pname}) "
                  f"{b0['pi_action_mean'][pname]:.4f} → {b1['pi_action_mean'][pname]:.4f}")
        if args.strategy == "substitute-skill":
            sname = ACTION_NAMES[PUSH_ACTION[cat]]
            l0, l1 = b0["pi_action_mean_late"], b1["pi_action_mean_late"]
            if l0 and l1:
                print(f"          floor bite: mean π({sname}) past "
                      f"{args.sub_from_progress:.0%} progress "
                      f"{l0[sname]:.4f} → {l1[sname]:.4f}")

    plot_matrix_pair(
        [matrices[base_key], matrices[a1]], [succs[base_key], succs[a1]],
        ["baseline (α=0)", f"steered α={alpha_label(a1)} ({args.strategy})"],
        f"PPO+GRU commit matrix under hidden-state steering · {tag}",
        Path(str(args.out_prefix) + "_commit_matrix.png"))
    plot_belief_traces(beliefs[base_key], beliefs[a1], a1, args.strategy, desc,
                       args.steer_from, args.steer_to,
                       Path(str(args.out_prefix) + "_belief.png"))
    plot_action_probs(aprobs[base_key], aprobs[a1], a1, args.strategy, desc,
                      Path(str(args.out_prefix) + "_actionprob.png"))
    if len(alphas) > 1:
        plot_dose_response(summaries, all_alphas, args.strategy,
                           Path(str(args.out_prefix) + "_dose.png"))
    plot_grid_steered(
        policy, view_size, env_size, env_width, cargs, device,
        steer_for_category=steer_factory(a1),
        n_seeds=args.grid_seeds, n_traj=args.grid_traj,
        seed_start=args.eval_seed_start, max_steps=args.max_steps,
        title=(f"STEERED (α={alpha_label(a1)}, {args.strategy}) PPO+GRU bridge_tunnel_commit · {tag} · "
               f"{args.grid_traj} rollouts/map · line=commitment "
               f"(blue none / yellow build / purple mine) · dots: build=red mine=yellow"),
        out_path=Path(str(args.out_prefix) + "_traj.png"))

    results = {
        "checkpoint": str(args.checkpoint), "strategy": args.strategy,
        "direction_raw_norm": raw_norm, "steer_from": args.steer_from,
        "steer_to": args.steer_to, "clamp_iters": args.clamp_iters,
        "clamp_target": args.clamp_target, "push_beta": args.push_beta,
        "sub_floor": args.sub_floor, "sub_from_progress": args.sub_from_progress,
        "runs": {alpha_label(a): summaries[a] for a in all_alphas},
    }
    res_path = Path(str(args.out_prefix) + "_results.json")
    res_path.write_text(json.dumps(results, indent=2))
    print(f"saved {res_path}")


if __name__ == "__main__":
    main()
