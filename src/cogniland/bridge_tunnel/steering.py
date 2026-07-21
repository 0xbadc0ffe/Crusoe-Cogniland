"""Centralized hidden-state steering for bridge_tunnel PPO+GRU agents.

All steering pipelines (`scripts/bridge_tunnel/eval_*_steered*.py`) assemble
their interventions from this module. A *steerer* is a callable

    steerer(h, t, ctx) -> h'      # h: (1, B, H) recurrent state being edited

invoked once per env step at the rollout's edit point — either the post-GRU
state (shallow: the state that feeds the heads and the next step) or the
pre-GRU input h_{t-1} (deep: the edit goes THROUGH the recurrence before
reaching any head; pass ``through_gru=True`` and provide ``ctx["feat"]``).
``ctx`` is an open dict of env-side per-step arrays (e.g. ``progress`` (B,),
``feat`` (B, E)) that intervention logics and corrections may inspect.

Two method families are provided, both built to be extended:

* :class:`GradientClamp` — iterative minimal-edit steering on linear head
  readouts. Each :class:`ClampTerm` names a head (``"actor"`` / ``"belief"``),
  a class index OR a GROUP of indices (constrained on their summed
  probability — e.g. all four movement actions at once), and a constraint:
  ``"suppress"`` clamps that probability BELOW ``threshold``; ``"push"``
  raises it ABOVE ``threshold``. Per sample and per iteration, only violated
  terms contribute (closed-form head gradients for the shallow edit point,
  autograd through the GRU cell for the deep one), the combined direction is
  passed through ``corrections``, unit-normalized, and applied with step size
  ``alpha`` — samples satisfying every constraint are untouched.
* :class:`LinearSteer` — constant additive push along a fixed unit direction
  (see the direction builders: :func:`head_direction` for actor/belief weight
  rows, :func:`class_mean_direction` for externally collected class means —
  category-, skill-, confounded- or balanced-conditioned alike).

WHEN/WHERE to intervene is factored out into *intervention logics*: callables
``logic(t, x, ctx) -> bool | (B,) mask`` attached to a whole steerer and/or to
individual clamp terms (:class:`StepWindow`, :class:`ProgressAtLeast`,
:class:`AllOf`, … or any custom callable reading env/model state).

*Corrections* — ``fn(d, t, x, ctx) -> d`` applied to the steering direction
before the step (e.g. :func:`project_out` to protect an axis) — are the hook
for gradient-projection/constraint experiments.

:func:`make_bt_steerer` assembles the standard bridge_tunnel experiment
strategies (axis pushes, suppress/substitute-skill, belief-clamp) per map
category, with balanced maps always the unsteered control row.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

# ── bridge_tunnel experiment constants (shared by every steering pipeline) ──
# belief head class order matches mapgen.CATEGORIES: balanced=0, lakes=1, rocky=2
BELIEF2I = {"balanced": 0, "lakes": 1, "rocky": 2}
# env action ids (Discrete(6)): 4 = build (water→wood), 5 = mine (rock→grass)
A_BUILD, A_MINE = 4, 5
ACTION_NAMES = ["up", "down", "left", "right", "build", "mine"]
MOVE_ACTIONS = (0, 1, 2, 3)                             # up, down, left, right
# Axis strategies use the unit lakes→rocky (or build→mine) direction; sign +1
# on lakes pushes toward rocky/mine, −1 on rocky toward lakes/build. Each
# category is steered toward its SUB-optimal skill; balanced = control.
STEER_SIGN = {"balanced": 0.0, "lakes": +1.0, "rocky": -1.0}
SUBOPT_COMMIT = {"lakes": 2, "rocky": 1}                # mine on lakes, build on rocky
PROHIBIT_ACTION = {"lakes": A_BUILD, "rocky": A_MINE}   # the optimal skill, forbidden
PUSH_ACTION = {"lakes": A_MINE, "rocky": A_BUILD}       # the sub-optimal skill
# belief-clamp: the WRONG archetype each category's belief is clamped toward
BELIEF_TARGET = {"lakes": BELIEF2I["rocky"], "rocky": BELIEF2I["lakes"]}


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


# ───────────────────────── intervention logic ─────────────────────────
# A logic is any callable ``logic(t, x, ctx) -> bool | (B,) bool array/tensor``
# deciding WHEN (t), WHERE (per-sample, from env state in ctx) and under what
# MODEL state (x = the (B, H) hidden being edited) an intervention may act.

InterventionLogic = Callable[[int, torch.Tensor, dict], "bool | np.ndarray | torch.Tensor"]


class Always:
    def __call__(self, t, x, ctx):
        return True


class StepWindow:
    """Active while ``t_from <= t < t_to`` (the legacy steer_from/steer_to)."""

    def __init__(self, t_from: int = 0, t_to: int = 10**9):
        self.t_from, self.t_to = t_from, t_to

    def __call__(self, t, x, ctx):
        return self.t_from <= t < self.t_to


class ProgressAtLeast:
    """Per-sample gate on ``ctx[key]`` (fraction of spawn→target distance)."""

    def __init__(self, frac: float, key: str = "progress"):
        self.frac, self.key = frac, key

    def __call__(self, t, x, ctx):
        return np.asarray(ctx[self.key]) >= self.frac


class StuckDetector:
    """Per-sample gate: True for samples whose ``ctx[key]`` (default
    ``"progress"``) hasn't improved by more than ``eps`` over the last
    ``window`` ENV STEPS — i.e. an actual "is this trajectory deadlocked"
    signal, meant to gate an escape-hatch intervention (e.g. an entropy
    floor) so it only fires when there's something to rescue, instead of
    disrupting trajectories that are already succeeding on their own.

    Stateful across steps within one rollout: it keeps a sliding history of
    ``ctx[key]`` values. Two things make this safe to reuse as-is across a
    whole eval sweep (one instance, many maps/rollouts, `GradientClamp`
    iterating a term's logic multiple times per env step):

    * It auto-resets whenever it sees ``t == 0`` — every fresh call to
      ``batched_rollout_steered`` starts a map's rollout at t=0, which is
      otherwise indistinguishable from "the same instance, next map" since
      steerers are typically built once per category and reused across all
      of that category's maps.
    * Repeated calls at the SAME ``t`` (this happens inside a single
      GradientClamp's gradient-iteration loop, which holds t fixed while it
      takes multiple steps toward convergence) reuse the last computed mask
      instead of pushing duplicate history entries — so the "window" always
      means env steps, never gradient iterations.
    """

    def __init__(self, window: int = 20, eps: float = 1e-3, key: str = "progress"):
        self.window = window
        self.eps = eps
        self.key = key
        self._history: list[np.ndarray] = []
        self._last_t: int | None = None
        self._last_mask: np.ndarray | None = None

    def __call__(self, t, x, ctx):
        val = np.asarray(ctx[self.key], dtype=np.float64)
        if self._last_t is not None and t == self._last_t:
            return self._last_mask
        if t == 0 or not self._history:
            self._history = [val.copy()]
            self._last_t = t
            self._last_mask = np.zeros(val.shape[0], dtype=bool)
            return self._last_mask
        self._history.append(val.copy())
        if len(self._history) > self.window + 1:
            self._history.pop(0)
        self._last_t = t
        if len(self._history) <= self.window:
            self._last_mask = np.zeros(val.shape[0], dtype=bool)
        else:
            self._last_mask = (val - self._history[0]) < self.eps
        return self._last_mask


class AllOf:
    def __init__(self, *logics):
        self.logics = logics

    def __call__(self, t, x, ctx):
        m = True
        for lg in self.logics:
            m = _mask_and(m, lg(t, x, ctx), x)
        return m


class AnyOf:
    def __init__(self, *logics):
        self.logics = logics

    def __call__(self, t, x, ctx):
        m = False
        for lg in self.logics:
            a, b = _as_mask(m, x), _as_mask(lg(t, x, ctx), x)
            m = a | b
        return m


class Not:
    def __init__(self, logic):
        self.logic = logic

    def __call__(self, t, x, ctx):
        return ~_as_mask(self.logic(t, x, ctx), x)


def _as_mask(m, x: torch.Tensor) -> torch.Tensor:
    """Normalize a logic result to a (B,) bool tensor on x's device."""
    if isinstance(m, bool) or isinstance(m, (np.bool_,)):
        return torch.full((x.shape[0],), bool(m), dtype=torch.bool, device=x.device)
    if isinstance(m, np.ndarray):
        return torch.from_numpy(np.asarray(m, dtype=bool)).to(x.device)
    return m.to(device=x.device, dtype=torch.bool)


def _mask_and(a, b, x):
    if a is True:
        return b
    if b is True:
        return a
    return _as_mask(a, x) & _as_mask(b, x)


def _gate(logic, t, x, ctx):
    """None/True ⇒ None (= everything allowed); else a (B,) bool tensor."""
    if logic is None:
        return None
    m = logic(t, x, ctx)
    if m is True:
        return None
    return _as_mask(m, x)


# ───────────────────────────── corrections ─────────────────────────────
# A correction is ``fn(d, t, x, ctx) -> d`` applied to the (B, H) steering
# direction before the unit-norm step — the hook for gradient projections and
# constraints.

Correction = Callable[[torch.Tensor, int, torch.Tensor, dict], torch.Tensor]


def project_out(axis: torch.Tensor) -> Correction:
    """Remove the component of the steering direction along ``axis`` (H,),
    e.g. to protect the belief readout while clamping the actor."""
    a = (axis / axis.norm().clamp(min=1e-8)).detach()

    def _corr(d, t, x, ctx):
        a_dev = a.to(d.device)
        return d - (d @ a_dev)[..., None] * a_dev

    return _corr


# ─────────────────────────── direction builders ───────────────────────────

def head_direction(head: torch.nn.Linear, pos: int, neg: int):
    """Unit neg→pos axis from a linear head's weight rows: Ŵ[pos]−W[neg].
    Covers the belief head (rocky vs lakes) and the actor head (mine vs
    build) alike. Returns (unit direction, raw ‖Δ‖)."""
    W = head.weight.detach()
    d = W[pos] - W[neg]
    return d / d.norm(), float(d.norm())


def class_mean_direction(mean_pos, mean_neg, device=None):
    """Unit neg→pos axis from two externally collected mean hidden states —
    category-conditioned (lakes/rocky rollout means), skill-event-conditioned
    (executed mine/build means), confounded or balanced: the caller decides
    what the means condition on. Returns (unit direction, raw ‖Δ‖)."""
    diff = np.asarray(mean_pos, dtype=np.float64) - np.asarray(mean_neg, dtype=np.float64)
    raw_norm = float(np.linalg.norm(diff))
    d = torch.from_numpy(diff.astype(np.float32))
    if device is not None:
        d = d.to(device)
    return d / d.norm(), raw_norm


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two directions (entanglement diagnostic)."""
    return float((a / a.norm().clamp(min=1e-8)) @ (b / b.norm().clamp(min=1e-8)))


# ─────────────────────────── gradient clamping ───────────────────────────

@dataclass
class ClampTerm:
    """One constraint on a linear head readout.

    ``index`` is a single class index OR a sequence of indices — a GROUP.
    A group of one index reduces exactly to the single-index case throughout.

    ``mode="suppress"``:    keep SUMMED P(group) below ``threshold``.
    ``mode="push"``:        keep SUMMED P(group) above ``threshold``.
    ``mode="entropy_min"``: keep the NORMALIZED entropy of the group's
        renormalized sub-distribution (H / log|group|, so it's always in
        [0, 1]) above ``threshold``. Unlike ``push`` on a group — which only
        controls how much TOTAL probability mass the group holds and is
        otherwise indifferent to how that mass is split among its members —
        this controls the SHAPE within the group: it pushes toward a more
        uniform split, actively working against the mass collapsing onto
        whichever member already dominates. Requires a group of ≥2 indices.
    ``logic``:  extra per-term gate (e.g. ProgressAtLeast for a late floor).
    ``drives``: if False the term never triggers iterations by itself — it
                only adds its (weighted) gradient wherever DRIVING terms are
                active (the legacy push_beta assist).
    """
    head: str
    index: int | Sequence[int]
    mode: str                       # "suppress" | "push" | "entropy_min"
    threshold: float
    weight: float = 1.0
    logic: InterventionLogic | None = None
    drives: bool = True

    def __post_init__(self):
        self._indices = ((self.index,) if isinstance(self.index, int)
                         else tuple(self.index))
        if self.mode == "entropy_min" and len(self._indices) < 2:
            raise ValueError("entropy_min requires a group of >=2 indices")

    @property
    def indices(self) -> tuple[int, ...]:
        return self._indices

    def group_prob(self, p: torch.Tensor) -> torch.Tensor:
        """Summed probability of the term's index group, shape (B,)."""
        return p[:, self._indices].sum(dim=-1)

    def group_entropy(self, p: torch.Tensor) -> torch.Tensor:
        """Normalized entropy (∈[0,1]) of the group's renormalized
        sub-distribution, shape (B,)."""
        return _group_entropy(p, self._indices)

    def violated(self, p: torch.Tensor) -> torch.Tensor:
        if self.mode == "suppress":
            return self.group_prob(p) > self.threshold
        if self.mode == "push":
            return self.group_prob(p) < self.threshold
        if self.mode == "entropy_min":
            return self.group_entropy(p) < self.threshold
        raise ValueError(f"unknown ClampTerm mode {self.mode!r}")

    @property
    def sign(self) -> float:        # objective is ASCENDED
        return -1.0 if self.mode == "suppress" else 1.0


def _group_renorm(p: torch.Tensor, indices: tuple[int, ...],
                  eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Renormalized distribution over ``indices`` (zero outside), + its mass.

    For a linear head with logits z, this equals EXACTLY softmax(z[indices])
    — the log-sum-exp normalizer over the full class range cancels in the
    ratio p_k/S, so the group's internal shape depends only on the group's
    own logits, not on anything outside it. This is what makes both
    :func:`_group_prob_grad` and the entropy functions below closed-form.
    """
    mask = torch.zeros(p.shape[-1], device=p.device, dtype=p.dtype)
    mask[list(indices)] = 1.0
    p_masked = p * mask
    S = p_masked.sum(dim=-1, keepdim=True).clamp(min=eps)
    return p_masked / S, S.squeeze(-1)


def _group_prob_grad(p: torch.Tensor, W: torch.Tensor,
                     indices: tuple[int, ...]) -> torch.Tensor:
    """Closed-form ∇_x log P(group) for a linear head z = Wx (softmax p).

    Derivation: with S = Σ_{k∈G} p_k, ∂S/∂z_i = p_i·(1_{i∈G} − S), so
    ∂log S/∂z_i = p_i·1_{i∈G}/S − p_i — i.e. the GROUP-masked, renormalized
    distribution minus the full distribution, pulled back through W. For a
    single-element group this reduces exactly to the standard
    ``W[k] − p @ W`` (the masked/renormalized distribution is one-hot at k).
    """
    q, _ = _group_renorm(p, indices)
    return q @ W - p @ W


def _group_entropy(p: torch.Tensor, indices: tuple[int, ...],
                   eps: float = 1e-8) -> torch.Tensor:
    """Normalized entropy H(q)/log|G| ∈ [0,1] of the group's renormalized
    sub-distribution q (uniform ⇒ 1, one-hot/collapsed ⇒ 0)."""
    q, _ = _group_renorm(p, indices, eps)
    logq = torch.log(q.clamp(min=eps))
    H = -(q * logq).sum(dim=-1)
    return H / np.log(len(indices))


def _group_entropy_grad(p: torch.Tensor, W: torch.Tensor,
                        indices: tuple[int, ...], eps: float = 1e-8) -> torch.Tensor:
    """Closed-form ∇_x [normalized entropy of the group's renormalized
    sub-distribution] for a linear head z = Wx.

    Since q = softmax(z[G]) depends only on the group's own logits (see
    :func:`_group_renorm`), this is the standard softmax-entropy gradient
    ∂H/∂z_i = −q_i·(log q_i + H) for i∈G (0 outside), pulled back through W,
    then scaled by 1/log|G| for the normalization. Ascending this direction
    pushes q toward uniform — the opposite failure mode of amplifying
    whichever group member already dominates (what a plain mass "push" on
    the group does instead).
    """
    q, _ = _group_renorm(p, indices, eps)
    logq = torch.log(q.clamp(min=eps))
    H = -(q * logq).sum(dim=-1, keepdim=True)
    v = q * (logq + H)                       # nonzero only inside the group
    return -(v @ W) / np.log(len(indices))


class GradientClamp:
    """Iterative minimal-edit clamp on one or more linear-head probabilities.

    Shallow (``through_gru=False``): the edited state feeds the heads
    directly, so ∇_h log p_k = W[k] − Σ_j p_j W[j] in closed form.
    Deep (``through_gru=True``): the edited state is the GRU INPUT h_{t-1};
    gradients are autograd-backpropagated through ``policy.gru(feat, h)`` into
    the heads (``ctx["feat"]`` required).

    Each iteration: evaluate all terms, combine the signed gradients of the
    violated (gated) ones, apply ``corrections``, unit-normalize per sample,
    step ``alpha``; stop when no driving term is violated. Samples with every
    constraint satisfied are never touched.

    Two gotchas worth knowing before tuning this:

    1. ``alpha`` is a STEP SIZE, not a suppression depth. Once a term's
       constraint is satisfied the loop stops touching that sample, so
       raising alpha past whatever is needed to cross the threshold has
       little further effect — the final probability lands just past
       ``threshold``, almost independent of alpha. The knob that controls
       HOW FAR past threshold you land is ``threshold`` itself (lower it for
       a suppress term, raise it for a push term).
    2. Lowering the threshold only helps if ``max_iters`` is raised to match
       — each unit-norm step only moves log-probability a bounded amount, so
       a much lower target needs more iterations to actually reach. If some
       samples still violate a term when ``max_iters`` runs out, this class
       warns once (see ``warn_on_nonconvergence``) rather than failing
       silently, because "I lowered clamp_target and nothing changed" is
       almost always this, not a broken gradient.

    Even a fully-converged suppression to a tiny per-step probability p can
    fail to change EPISODE-level behavior if the rollout gives the agent many
    chances to trigger the action again (a long max_steps and only a handful
    of successes needed): P(≥1 success in N tries) = 1−(1−p)^N → 1 as N
    grows, so p must be pushed several orders of magnitude below "looks
    small" before outcomes actually change — belief readouts don't have this
    problem since they're read out instantaneously, not accumulated over time.
    """

    def __init__(self, policy, terms: Sequence[ClampTerm], alpha: float,
                 max_iters: int = 10, logic: InterventionLogic | None = None,
                 through_gru: bool = False,
                 corrections: Sequence[Correction] = (),
                 warn_on_nonconvergence: bool = True):
        self.policy = policy
        self.terms = list(terms)
        self.alpha = float(alpha)
        self.max_iters = int(max_iters)
        self.logic = logic
        self.through_gru = bool(through_gru)
        self.corrections = list(corrections)
        self.warn_on_nonconvergence = bool(warn_on_nonconvergence)
        self._warned = False

    def _term_masks(self, t, x, ctx, probs, global_gate):
        """Per-term active masks + the union of driving masks."""
        drive = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        masks = []
        for term in self.terms:
            if not term.drives:
                masks.append(None)                       # resolved after drive
                continue
            m = term.violated(probs[term.head])
            g = _gate(term.logic, t, x, ctx)
            if g is not None:
                m = m & g
            if global_gate is not None:
                m = m & global_gate
            masks.append(m)
            drive = drive | m
        for k, term in enumerate(self.terms):            # assist terms follow
            if masks[k] is None:
                g = _gate(term.logic, t, x, ctx)
                masks[k] = drive if g is None else (drive & g)
        return masks, drive

    def __call__(self, h: torch.Tensor, t: int, ctx: dict) -> torch.Tensor:
        if self.alpha == 0.0 or not self.terms:
            return h
        x0 = h.squeeze(0)
        global_gate = _gate(self.logic, t, x0, ctx)
        if global_gate is not None and not bool(global_gate.any()):
            return h
        head_names = {term.head for term in self.terms}
        heads = {n: getattr(self.policy, n) for n in head_names}
        x = x0.detach()
        for _ in range(self.max_iters):
            if self.through_gru:
                feat_seq = ctx["feat"].detach()[None]     # (1, B, E)
                # cudnn's fused GRU cannot run backward on a module in eval()
                # mode; the native path can, and this 1-step cell is tiny
                with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                    xg = x.clone().requires_grad_(True)
                    _, h_out = self.policy.gru(feat_seq, xg[None])
                    y = h_out.squeeze(0)
                    zs = {n: heads[n](y) for n in head_names}       # raw logits
                    logps = {n: torch.log_softmax(zs[n], dim=-1)
                             for n in head_names}
                    probs = {n: lp.detach().exp() for n, lp in logps.items()}
                    masks, drive = self._term_masks(t, x, ctx, probs, global_gate)
                    if not bool(drive.any()):
                        break
                    obj = None                            # ascended
                    for term, m in zip(self.terms, masks):
                        if term.mode == "entropy_min":
                            # q = softmax(z[group]) exactly (the outer
                            # normalizer cancels — see _group_renorm), so this
                            # is just the standard entropy of a sub-softmax,
                            # left to autograd; scaled to [0,1] like the
                            # closed-form path for the same threshold units
                            z_group = zs[term.head][:, list(term.indices)]
                            logq = torch.log_softmax(z_group, dim=-1)
                            obj_val = -(logq.exp() * logq).sum(dim=-1) \
                                / np.log(len(term.indices))
                        else:
                            # log P(group) = logsumexp over the group's
                            # log-probs; reduces to logp[:, k] for 1 index
                            obj_val = torch.logsumexp(
                                logps[term.head][:, list(term.indices)], dim=-1)
                        contrib = term.sign * term.weight * obj_val * m.float()
                        obj = contrib if obj is None else obj + contrib
                    # samples are independent through the GRU, so rows of d
                    # are the per-sample gradients
                    (d,) = torch.autograd.grad(obj.sum(), xg)
            else:
                probs = {n: torch.softmax(heads[n](x), dim=-1) for n in head_names}
                masks, drive = self._term_masks(t, x, ctx, probs, global_gate)
                if not bool(drive.any()):
                    break
                d = torch.zeros_like(x)
                for term, m in zip(self.terms, masks):
                    W = heads[term.head].weight.detach()  # (C, H)
                    if term.mode == "entropy_min":
                        g = _group_entropy_grad(probs[term.head], W, term.indices)
                    else:
                        g = _group_prob_grad(probs[term.head], W, term.indices)
                    d = d + (term.sign * term.weight) * m.float()[:, None] * g
            for corr in self.corrections:
                d = corr(d, t, x, ctx)
            d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            x = torch.where(drive[:, None], x + self.alpha * d, x)
        else:
            # loop ran to max_iters without an early break — check whether
            # any term is STILL violated on the final state and warn if so
            # (see the class docstring: this is the "lowered clamp_target
            # without raising clamp_iters" footgun, not silent failure)
            self._check_convergence(t, x, ctx, heads, head_names, global_gate)
        return x[None]

    def _check_convergence(self, t, x, ctx, heads, head_names, global_gate):
        if not self.warn_on_nonconvergence or self._warned:
            return
        with torch.no_grad():
            if self.through_gru:
                _, h_out = self.policy.gru(ctx["feat"].detach()[None], x[None])
                y = h_out.squeeze(0)
            else:
                y = x
            probs = {n: torch.softmax(heads[n](y), dim=-1) for n in head_names}
            _, drive = self._term_masks(t, x, ctx, probs, global_gate)
        n_bad = int(drive.sum())
        if n_bad:
            self._warned = True
            warnings.warn(
                f"GradientClamp: {n_bad}/{drive.numel()} sample(s) still "
                f"violated a constraint after max_iters={self.max_iters} "
                f"(alpha={self.alpha:g}) — the steered probability is stuck "
                f"short of the requested threshold(s). If you lowered a "
                f"threshold expecting stronger suppression/push, raise "
                f"clamp_iters (and check alpha is large enough to make "
                f"progress per step) rather than assuming the steering had "
                f"no effect. This warning fires once per GradientClamp "
                f"instance.", RuntimeWarning, stacklevel=2)


# ───────────────────────────── linear steering ─────────────────────────────

class LinearSteer:
    """Constant additive push ``h += alpha * sign * direction`` (direction is
    expected unit-norm — see the direction builders). ``logic`` gates the push
    in time and per sample; ``corrections`` transform the direction first (and
    force a re-normalization, so pass an unmodified unit direction if you need
    exact legacy behavior)."""

    def __init__(self, direction: torch.Tensor, alpha: float, sign: float = 1.0,
                 logic: InterventionLogic | None = None,
                 corrections: Sequence[Correction] = ()):
        self.direction = direction
        self.alpha = float(alpha)
        self.sign = float(sign)
        self.logic = logic
        self.corrections = list(corrections)

    def __call__(self, h: torch.Tensor, t: int, ctx: dict) -> torch.Tensor:
        if self.alpha == 0.0 or self.sign == 0.0:
            return h
        x = h.squeeze(0)
        gate = _gate(self.logic, t, x, ctx)
        if gate is not None and not bool(gate.any()):
            return h
        d = self.direction.to(x.device)
        if self.corrections:
            d2 = d.view(1, -1)
            for corr in self.corrections:
                d2 = corr(d2, t, x, ctx)
            d = (d2 / d2.norm(dim=-1, keepdim=True).clamp(min=1e-8)).view(-1)
        if gate is None or bool(gate.all()):
            return h + (self.alpha * self.sign) * d.view(1, 1, -1)
        x = x + (self.alpha * self.sign) * gate.to(x.dtype)[:, None] * d.view(1, -1)
        return x.unsqueeze(0)


# ───────────────────── standard bridge_tunnel strategies ─────────────────────

def make_bt_steerer(strategy: str, category: str, policy,
                    direction: torch.Tensor | None, alpha: float, *,
                    steer_from: int = 0, steer_to: int = 10**9,
                    clamp_iters: int = 10, clamp_target: float = 0.01,
                    push_beta: float = 0.0, sub_floor: float = 0.05,
                    sub_from_progress: float = 0.5, sub_target: str = "skill",
                    sub_stuck_gate: bool = False, sub_stuck_window: int = 20,
                    sub_stuck_eps: float = 1e-3,
                    belief_floor: float = 0.75, through_gru: bool = False,
                    corrections: Sequence[Correction] = ()):
    """Assemble the steerer for one strategy × map category (None ⇒ unsteered).
    Balanced maps are always the no-steer control row.

    Clamp strategies: ``suppress-skill`` (π(prohibited) < clamp_target, plus a
    non-driving push assist when push_beta > 0), ``substitute-skill`` (adds a
    floor term gated on ProgressAtLeast — three ``sub_target`` choices):

    * ``"skill"``    (default) floors π(sub-optimal skill) above sub_floor —
      the original "substitution" scenario.
    * ``"movement"`` floors the SUMMED probability of the four movement
      actions (up/down/left/right) above sub_floor. NOTE: since suppressing
      the optimal skill already frees up most probability mass onto movement
      by conservation, this is usually redundant with clamp_target alone
      (it only bites once sub_floor is set ABOVE whatever movement mass
      suppression already yields) — it does not, by itself, change WHICH
      direction the freed mass goes to.
    * ``"movement-entropy"`` instead floors the NORMALIZED ENTROPY (∈[0,1])
      of the movement sub-distribution above sub_floor via
      ``ClampTerm(..., "entropy_min")``. This is a genuinely different lever
      from ``"movement"``: it doesn't just ask for more total movement
      probability, it actively resists that probability collapsing onto
      whichever single direction already dominates, pushing toward a more
      uniform split among the 4 directions — the intended target for "make a
      suppressed agent explore instead of oscillating/getting stuck", which
      plain mass-flooring does not achieve. Empirically it's a genuine
      TRADEOFF, not a strict win: at strong suppression (agent otherwise
      fully deadlocked) it rescues a meaningful fraction of episodes; at
      mild suppression (agent otherwise mostly succeeding on residual skill
      probability) it's disruptive, since a high entropy floor forces
      near-uniform movement most of the time, destroying purposeful
      navigation the agent didn't need help with. ``sub_stuck_gate=True``
      addresses this: the floor term additionally requires
      :class:`StuckDetector` (no ``ctx["progress"]`` improvement over the
      last ``sub_stuck_window`` steps, tolerance ``sub_stuck_eps``) alongside
      ProgressAtLeast, so it only engages once a trajectory is actually
      deadlocked rather than firing on every step past sub_from_progress.

    ``belief-clamp`` (P(wrong archetype) > belief_floor). Any other strategy
    with a precomputed ``direction`` (class-mean, belief-head, skill-mean, …)
    becomes a LinearSteer with the category's STEER_SIGN.
    """
    if alpha == 0.0:
        return None
    window = StepWindow(steer_from, steer_to)
    if strategy == "belief-clamp":
        target_cls = BELIEF_TARGET.get(category)
        if target_cls is None:
            return None
        return GradientClamp(
            policy, [ClampTerm("belief", target_cls, "push", belief_floor)],
            alpha, clamp_iters, logic=window, through_gru=through_gru,
            corrections=corrections)
    if strategy in ("suppress-skill", "substitute-skill"):
        prohibit = PROHIBIT_ACTION.get(category)
        if prohibit is None:
            return None
        push = PUSH_ACTION[category]
        terms = [ClampTerm("actor", prohibit, "suppress", clamp_target)]
        if strategy == "substitute-skill":
            if sub_target == "movement":
                floor_idx, floor_mode = MOVE_ACTIONS, "push"
            elif sub_target == "movement-entropy":
                floor_idx, floor_mode = MOVE_ACTIONS, "entropy_min"
            elif sub_target == "skill":
                floor_idx, floor_mode = push, "push"
            else:
                raise ValueError(f"sub_target must be 'skill', 'movement', or "
                                 f"'movement-entropy', got {sub_target!r}")
            floor_logic = ProgressAtLeast(sub_from_progress)
            if sub_stuck_gate:
                floor_logic = AllOf(floor_logic,
                                    StuckDetector(window=sub_stuck_window,
                                                 eps=sub_stuck_eps))
            terms.append(ClampTerm("actor", floor_idx, floor_mode, sub_floor,
                                   logic=floor_logic))
        elif push_beta > 0.0:
            terms.append(ClampTerm("actor", push, "push", 1.0,
                                   weight=push_beta, drives=False))
        return GradientClamp(policy, terms, alpha, clamp_iters, logic=window,
                             through_gru=through_gru, corrections=corrections)
    sign = STEER_SIGN.get(category, 0.0)
    if direction is None or sign == 0.0:
        return None
    return LinearSteer(direction, alpha, sign=sign, logic=window,
                       corrections=corrections)


__all__ = [
    "A_BUILD", "A_MINE", "ACTION_NAMES", "MOVE_ACTIONS", "BELIEF2I",
    "BELIEF_TARGET", "PROHIBIT_ACTION", "PUSH_ACTION", "STEER_SIGN", "SUBOPT_COMMIT",
    "parse_alpha_token", "alpha_label", "cat_alpha",
    "Always", "StepWindow", "ProgressAtLeast", "StuckDetector",
    "AllOf", "AnyOf", "Not",
    "Correction", "project_out",
    "head_direction", "class_mean_direction", "cosine",
    "ClampTerm", "GradientClamp", "LinearSteer", "make_bt_steerer",
]
