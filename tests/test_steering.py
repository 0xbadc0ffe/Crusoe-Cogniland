"""Unit tests for the centralized steering library
(cogniland.bridge_tunnel.steering): gradient clamps (shallow + through-GRU),
linear steering, intervention logics, corrections, and the bridge_tunnel
strategy assembly."""
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from cogniland.bridge_tunnel.steering import (
    A_BUILD, A_MINE, MOVE_ACTIONS, AllOf, ClampTerm, GradientClamp,
    LinearSteer, ProgressAtLeast, StepWindow, StuckDetector, _group_entropy,
    _group_entropy_grad, _group_prob_grad, class_mean_direction, cosine,
    head_direction, make_bt_steerer, project_out)

H, A, C, E, B = 16, 6, 3, 8, 12


class DummyPolicy(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.actor = nn.Linear(H, A)
        self.belief = nn.Linear(H, C)
        self.gru = nn.GRU(E, H, batch_first=False)
        self.gru_hidden = H


@pytest.fixture
def policy():
    return DummyPolicy()


@pytest.fixture
def h0():
    torch.manual_seed(1)
    return torch.randn(1, B, H)


def _probs(head, h):
    return torch.softmax(head(h.squeeze(0)), dim=-1)


def test_suppress_clamp_reaches_target(policy, h0):
    clamp = GradientClamp(policy, [ClampTerm("actor", A_BUILD, "suppress", 0.01)],
                          alpha=0.5, max_iters=50)
    h1 = clamp(h0, 0, {})
    assert (_probs(policy.actor, h1)[:, A_BUILD] <= 0.011).all()


def test_clamp_is_minimal_edit(policy, h0):
    """Samples already satisfying the constraint are untouched."""
    p0 = _probs(policy.actor, h0)[:, A_BUILD]
    thr = float(p0.median())
    clamp = GradientClamp(policy, [ClampTerm("actor", A_BUILD, "suppress", thr)],
                          alpha=0.3, max_iters=50)
    h1 = clamp(h0, 0, {})
    ok = p0 <= thr
    assert torch.equal(h1.squeeze(0)[ok], h0.squeeze(0)[ok])
    assert not torch.equal(h1.squeeze(0)[~ok], h0.squeeze(0)[~ok])


def test_push_clamp_reaches_floor(policy, h0):
    clamp = GradientClamp(policy, [ClampTerm("belief", 2, "push", 0.75)],
                          alpha=0.5, max_iters=60)
    h1 = clamp(h0, 0, {})
    assert (_probs(policy.belief, h1)[:, 2] >= 0.749).all()


def test_progress_gated_push(policy, h0):
    """The substitution floor only fires for in-zone samples."""
    progress = np.zeros(B, dtype=np.float32)
    progress[: B // 2] = 1.0
    clamp = GradientClamp(
        policy,
        [ClampTerm("actor", A_MINE, "push", 0.6, logic=ProgressAtLeast(0.5))],
        alpha=0.5, max_iters=60)
    h1 = clamp(h0, 0, {"progress": progress})
    p1 = _probs(policy.actor, h1)[:, A_MINE]
    assert (p1[: B // 2] >= 0.599).all()
    assert torch.equal(h1.squeeze(0)[B // 2:], h0.squeeze(0)[B // 2:])


def test_assist_term_does_not_drive(policy, h0):
    """A drives=False term (legacy push_beta) never triggers iterations on its
    own: with the suppress constraint already satisfied, h is untouched."""
    clamp = GradientClamp(
        policy,
        [ClampTerm("actor", A_BUILD, "suppress", 1.1),        # never violated
         ClampTerm("actor", A_MINE, "push", 1.0, weight=0.5, drives=False)],
        alpha=0.5, max_iters=20)
    assert torch.equal(clamp(h0, 0, {}), h0)


def test_step_window_gates_everything(policy, h0):
    clamp = GradientClamp(policy, [ClampTerm("actor", A_BUILD, "suppress", 0.01)],
                          alpha=0.5, max_iters=50, logic=StepWindow(10, 20))
    assert torch.equal(clamp(h0, 5, {}), h0)
    assert not torch.equal(clamp(h0, 15, {}), h0)


def test_through_gru_clamp(policy, h0):
    torch.manual_seed(2)
    feat = torch.randn(B, E)
    clamp = GradientClamp(policy, [ClampTerm("actor", A_BUILD, "suppress", 0.02)],
                          alpha=0.5, max_iters=80, through_gru=True)
    h1 = clamp(h0, 0, {"feat": feat})
    _, h_out = policy.gru(feat[None], h1)
    assert (_probs(policy.actor, h_out)[:, A_BUILD] <= 0.021).all()
    # the clamp edits the INPUT state, so probs are only guaranteed after
    # the recurrence, and untouched samples pass through unchanged
    p_in = _probs(policy.actor, torch.zeros(1, B, H))
    assert p_in.shape == (B, A)


def test_linear_steer_matches_legacy_math(h0):
    torch.manual_seed(3)
    d = torch.randn(H)
    d = d / d.norm()
    steer = LinearSteer(d, alpha=0.25, sign=-1.0)
    expected = h0 + (0.25 * -1.0) * d.view(1, 1, -1)
    assert torch.equal(steer(h0, 0, {}), expected)
    outside = LinearSteer(d, alpha=0.25, sign=1.0, logic=StepWindow(5, 9))
    assert torch.equal(outside(h0, 0, {}), h0)


def test_project_out_correction(policy, h0):
    axis = policy.belief.weight.detach()[2] - policy.belief.weight.detach()[1]
    clamp = GradientClamp(policy, [ClampTerm("actor", A_BUILD, "suppress", 0.01)],
                          alpha=0.5, max_iters=50,
                          corrections=[project_out(axis)])
    h1 = clamp(h0, 0, {})
    delta = (h1 - h0).squeeze(0)
    proj = delta @ (axis / axis.norm())
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-5)


def test_direction_builders(policy):
    d, n = head_direction(policy.actor, A_MINE, A_BUILD)
    assert abs(float(d.norm()) - 1.0) < 1e-6 and n > 0
    rng = np.random.default_rng(0)
    m1, m2 = rng.normal(size=H), rng.normal(size=H)
    d2, n2 = class_mean_direction(m1, m2)
    assert abs(float(d2.norm()) - 1.0) < 1e-6
    assert abs(n2 - np.linalg.norm(m1 - m2)) < 1e-6
    assert abs(cosine(d2, d2) - 1.0) < 1e-6


def test_nonconvergence_warns_once(policy, h0):
    """Too few iterations for a very tight threshold should warn (once),
    not fail silently — the 'lowered clamp_target, forgot clamp_iters'
    footgun this class is meant to surface."""
    clamp = GradientClamp(policy, [ClampTerm("actor", A_BUILD, "suppress", 1e-8)],
                          alpha=0.1, max_iters=1)
    with pytest.warns(RuntimeWarning, match="did not converge|still violated"):
        clamp(h0, 0, {})
    # second call on the same instance must NOT warn again
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clamp(h0, 0, {})


def test_convergence_no_warning(policy, h0):
    """A generous iteration budget for an easy threshold must not warn."""
    clamp = GradientClamp(policy, [ClampTerm("actor", A_BUILD, "suppress", 0.5)],
                          alpha=0.5, max_iters=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clamp(h0, 0, {})


def test_group_prob_grad_matches_single_index_formula(policy):
    """The group gradient must reduce EXACTLY to the old W[k]-p@W formula
    for a singleton group (regression guard for the generalization)."""
    torch.manual_seed(5)
    p = torch.softmax(torch.randn(B, A), dim=-1)
    W = policy.actor.weight.detach()
    for k in range(A):
        g_group = _group_prob_grad(p, W, (k,))
        g_old = W[k] - p @ W
        assert torch.allclose(g_group, g_old, atol=1e-6)


def test_group_push_raises_summed_probability(policy, h0):
    """Pushing a GROUP (movement actions) should raise their summed
    probability above threshold, exactly like a single-index push does."""
    clamp = GradientClamp(
        policy, [ClampTerm("actor", MOVE_ACTIONS, "push", 0.8)],
        alpha=0.5, max_iters=60)
    h1 = clamp(h0, 0, {})
    p1 = _probs(policy.actor, h1)
    move_mass = p1[:, list(MOVE_ACTIONS)].sum(dim=-1)
    assert (move_mass >= 0.799).all()


def test_substitute_movement_strategy(policy):
    """make_bt_steerer(sub_target='movement') floors movement mass instead
    of the opposite skill; the resulting steerer has two ClampTerms, the
    second one a MOVE_ACTIONS group push."""
    st = make_bt_steerer("substitute-skill", "lakes", policy, None, 0.5,
                         sub_target="movement", sub_floor=0.5)
    assert isinstance(st, GradientClamp)
    assert len(st.terms) == 2
    floor_term = st.terms[1]
    assert floor_term.mode == "push" and floor_term.indices == MOVE_ACTIONS
    with pytest.raises(ValueError):
        make_bt_steerer("substitute-skill", "lakes", policy, None, 0.5,
                        sub_target="bogus")


def test_group_entropy_matches_softmax_over_group_logits():
    """Verify the key identity the entropy derivation depends on: the
    group-renormalized distribution equals EXACTLY softmax over the group's
    own raw logits (the outer normalizer cancels), for arbitrary logits
    outside the group too."""
    torch.manual_seed(6)
    z = torch.randn(B, A)
    p = torch.softmax(z, dim=-1)
    idx = (0, 2, 4)
    q_expected = torch.softmax(z[:, idx], dim=-1)
    H_expected = (-(q_expected * torch.log(q_expected)).sum(dim=-1)
                 / np.log(len(idx)))
    H_actual = _group_entropy(p, idx)
    assert torch.allclose(H_actual, H_expected, atol=1e-5)


def test_group_entropy_bounds():
    """Normalized entropy is 0 for a collapsed distribution, 1 for uniform."""
    idx = (0, 1, 2, 3)
    p_uniform = torch.full((1, A), 1.0 / A)
    assert abs(float(_group_entropy(p_uniform, idx)) - 1.0) < 1e-5
    p_collapsed = torch.zeros(1, A)
    p_collapsed[0, 0] = 1.0 - 1e-6
    p_collapsed[0, 1:] = 1e-6 / (A - 1)
    assert float(_group_entropy(p_collapsed, idx)) < 0.05


def test_entropy_grad_increases_entropy(policy):
    """Ascending the closed-form entropy gradient must increase the group's
    normalized entropy — the whole point of entropy_min vs a mass push."""
    torch.manual_seed(7)
    # craft a hidden state whose actor distribution over MOVE_ACTIONS is
    # skewed (not already uniform), so there's room to improve
    x = torch.randn(4, H)
    W = policy.actor.weight.detach()
    p0 = torch.softmax(policy.actor(x), dim=-1)
    H0 = _group_entropy(p0, MOVE_ACTIONS)
    g = _group_entropy_grad(p0, W, MOVE_ACTIONS)
    x1 = x + 0.5 * g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    p1 = torch.softmax(policy.actor(x1), dim=-1)
    H1 = _group_entropy(p1, MOVE_ACTIONS)
    assert bool((H1 >= H0 - 1e-4).all())
    assert float(H1.mean()) > float(H0.mean())


def test_entropy_min_clamp_converges(policy, h0):
    """GradientClamp with an entropy_min term raises normalized entropy of
    the group above threshold."""
    clamp = GradientClamp(
        policy, [ClampTerm("actor", MOVE_ACTIONS, "entropy_min", 0.95)],
        alpha=0.3, max_iters=60)
    h1 = clamp(h0, 0, {})
    p1 = _probs(policy.actor, h1)
    assert (_group_entropy(p1, MOVE_ACTIONS) >= 0.949).all()


def test_entropy_min_through_gru(policy, h0):
    torch.manual_seed(8)
    feat = torch.randn(B, E)
    clamp = GradientClamp(
        policy, [ClampTerm("actor", MOVE_ACTIONS, "entropy_min", 0.9)],
        alpha=0.3, max_iters=80, through_gru=True)
    h1 = clamp(h0, 0, {"feat": feat})
    _, h_out = policy.gru(feat[None], h1)
    p1 = _probs(policy.actor, h_out)
    assert (_group_entropy(p1, MOVE_ACTIONS) >= 0.899).all()


def test_entropy_min_requires_group():
    with pytest.raises(ValueError):
        ClampTerm("actor", A_BUILD, "entropy_min", 0.9)


def test_substitute_movement_entropy_strategy(policy):
    st = make_bt_steerer("substitute-skill", "lakes", policy, None, 0.5,
                         sub_target="movement-entropy", sub_floor=0.9)
    assert isinstance(st, GradientClamp)
    floor_term = st.terms[1]
    assert floor_term.mode == "entropy_min" and floor_term.indices == MOVE_ACTIONS


def test_stuck_detector_not_stuck_during_warmup(h0):
    det = StuckDetector(window=5, eps=1e-3)
    for t in range(5):                      # fewer than `window` steps seen
        mask = det(t, h0.squeeze(0), {"progress": np.full(B, 0.1 * t)})
        assert not np.asarray(mask).any()


def test_stuck_detector_flags_plateau(h0):
    det = StuckDetector(window=5, eps=1e-3)
    progress = 0.0
    for t in range(5):                      # warmup, progressing
        det(t, h0.squeeze(0), {"progress": np.full(B, progress)})
        progress += 0.1
    stuck_mask = None
    for t in range(5, 12):                  # now plateaus — no more progress
        stuck_mask = det(t, h0.squeeze(0), {"progress": np.full(B, progress)})
    assert np.asarray(stuck_mask).all()


def test_stuck_detector_not_stuck_when_progressing(h0):
    det = StuckDetector(window=5, eps=1e-3)
    progress = 0.0
    mask = None
    for t in range(20):
        mask = det(t, h0.squeeze(0), {"progress": np.full(B, progress)})
        progress += 0.05                    # steady, well above eps
    assert not np.asarray(mask).any()


def test_stuck_detector_resets_at_t0(h0):
    """Simulates the same instance being reused across maps by
    make_bt_steerer/steer_factory: a fresh rollout always calls t=0 first."""
    det = StuckDetector(window=3, eps=1e-3)
    progress = 0.0
    for t in range(10):                     # first "map": plateaus and gets flagged
        mask = det(t, h0.squeeze(0), {"progress": np.full(B, progress)})
    assert np.asarray(mask).all()
    mask0 = det(0, h0.squeeze(0), {"progress": np.full(B, 0.0)})
    assert not np.asarray(mask0).any()      # reset — must not still be "stuck"


def test_stuck_detector_dedup_same_t(h0):
    """Repeated calls at the SAME t (GradientClamp's gradient-iteration loop)
    must not corrupt the sliding window — they reuse the cached mask."""
    det = StuckDetector(window=3, eps=1e-3)
    for t in range(6):
        det(t, h0.squeeze(0), {"progress": np.full(B, 0.1 * t)})
    hist_len_before = len(det._history)
    # calling again at the same t as last time must not push new history
    det(5, h0.squeeze(0), {"progress": np.full(B, 999.0)})
    assert len(det._history) == hist_len_before


def test_substitute_stuck_gate_wraps_progress_and_stuck(policy):
    st = make_bt_steerer("substitute-skill", "lakes", policy, None, 0.5,
                         sub_target="movement-entropy", sub_floor=0.5,
                         sub_stuck_gate=True, sub_stuck_window=10)
    floor_term = st.terms[1]
    assert isinstance(floor_term.logic, AllOf)
    assert isinstance(floor_term.logic.logics[1], StuckDetector)
    assert floor_term.logic.logics[1].window == 10
    # without the flag, logic is plain ProgressAtLeast (unchanged behavior)
    st2 = make_bt_steerer("substitute-skill", "lakes", policy, None, 0.5,
                          sub_target="movement-entropy", sub_floor=0.5)
    assert isinstance(st2.terms[1].logic, ProgressAtLeast)


def test_make_bt_steerer_assembly(policy):
    torch.manual_seed(4)
    d = torch.randn(H); d = d / d.norm()
    # balanced is always the unsteered control; alpha=0 disables
    assert make_bt_steerer("suppress-skill", "balanced", policy, None, 0.5) is None
    assert make_bt_steerer("class-mean", "lakes", policy, d, 0.0) is None
    assert isinstance(make_bt_steerer("class-mean", "rocky", policy, d, 0.5),
                      LinearSteer)
    for strat in ("suppress-skill", "substitute-skill", "belief-clamp"):
        st = make_bt_steerer(strat, "lakes", policy, None, 0.5)
        assert isinstance(st, GradientClamp)
    deep = make_bt_steerer("suppress-skill", "rocky", policy, None, 0.5,
                           through_gru=True)
    assert deep.through_gru
