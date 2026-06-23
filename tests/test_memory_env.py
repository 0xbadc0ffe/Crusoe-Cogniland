"""Smoke + contract tests for cogniland.memory_env.MemoryEnv (MiniGrid build).

The env is now a real Farama-MiniGrid environment (the triangle agent + grid
physics, partial-obs RGB pixels, a custom oriented Key cue and an extra
mid-corridor branch). These tests pin the cogniland *contract* that the rest of
the analysis code relies on: the info-dict keys, the seven phases in order, the
cue/door visibility windows, shape->branch, colour->door, randomised door
positions, the forced_branch / suppress interventions and the reward signs.
Episodes are driven by the privileged oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

from cogniland.memory_env import (
    MemoryEnv, MemoryEnvConfig, PHASES, CUE_TYPES, OrientedKey,
    oracle_action, evaluate, record_trajectory,
)
from cogniland.memory_env.env import (
    A_LEFT_TURN, A_RIGHT_TURN, A_FORWARD, DIR_NORTH, DIR_SOUTH,
    _COL_GREEN, _COL_BLUE,
)


# --------------------------------------------------------------------------- #
# colour detection — the POV render applies MiniGrid's visibility highlight, so
# cue/door pixels are tinted and don't match the raw palette exactly. Detect by
# hue dominance instead (the agent is the classic red triangle; red is NOT a task colour
# here, so green/blue pixels mean cue/door only).
# --------------------------------------------------------------------------- #
def _count_green(obs, thresh=4):
    r, g, b = obs[..., 0].astype(int), obs[..., 1].astype(int), obs[..., 2].astype(int)
    return int(((g > 120) & (g > r + 60) & (g > b + 60)).sum())


def _count_blue(obs, thresh=4):
    r, g, b = obs[..., 0].astype(int), obs[..., 1].astype(int), obs[..., 2].astype(int)
    return int(((b > 120) & (b > r + 60) & (b > g + 60)).sum())


def _count_color(obs, color):
    return _count_green(obs) if np.array_equal(color, _COL_GREEN) else _count_blue(obs)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rollout_oracle(env, seed=0):
    """Roll out with the oracle; collect (phases, infos, obss)."""
    obs, info = env.reset(seed=seed)
    phases, infos, obss = [info["phase"]], [info], [obs]
    term = trunc = False
    while not (term or trunc):
        a = oracle_action(info)
        obs, _, term, trunc, info = env.step(a)
        phases.append(info["phase"]); infos.append(info); obss.append(obs)
    return phases, infos, obss


def _branch_policy(want_dir, mg):
    """Policy that *tries* to enter the branch row reached by facing want_dir,
    else defers to the oracle (used to probe forced_branch / suppress)."""
    def pol(obs, info):
        if info["phase"] == "branch_choice":
            adir = int(mg.agent_dir)
            if adir != want_dir:
                return A_RIGHT_TURN if (adir + 1) % 4 == want_dir else A_LEFT_TURN
            return A_FORWARD
        return oracle_action(info)
    return pol


# --------------------------------------------------------------------------- #
# observation format
# --------------------------------------------------------------------------- #
def test_pixel_obs_dreamer_format():
    env = MemoryEnv()
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.uint8
    assert obs.ndim == 3 and obs.shape[2] == 3
    h, w, _ = obs.shape
    assert h == w == env.cfg.view_size * env.cfg.cell_px
    assert tuple(env.observation_space.shape) == obs.shape


def test_no_privileged_labels_in_obs():
    env = MemoryEnv()
    obs, info = env.reset(seed=1)
    for key in ("cue_type", "phase", "correct_branch", "target_door_color"):
        assert key in info
    assert obs.shape[2] == 3
    assert obs.ndim == 3


def test_action_space_is_minigrid():
    env = MemoryEnv()
    # native MiniGrid action space (turn-left / turn-right / forward / ...).
    assert env.action_space.n == 7


# --------------------------------------------------------------------------- #
# custom oriented Key cue renders distinguishably
# --------------------------------------------------------------------------- #
def _render_key(color, orientation, px=64):
    img = np.zeros((px, px, 3), np.uint8)
    OrientedKey(color, orientation=orientation).render(img)
    return img


def test_oriented_key_up_down_distinguishable():
    down = _render_key("green", "down")
    up = _render_key("green", "up")
    # the two orientations must produce visibly different pixels
    assert not np.array_equal(down, up)
    # the key is vertically mirrored: the ring (densest band) is at the top for
    # 'down' and at the bottom for 'up'. Compare coloured-pixel mass by half.
    mask_d = down.any(axis=-1)
    mask_u = up.any(axis=-1)
    top_d, bot_d = mask_d[: px_half(down)].sum(), mask_d[px_half(down):].sum()
    top_u, bot_u = mask_u[: px_half(up)].sum(), mask_u[px_half(up):].sum()
    # 'down' (upright) has the ring (heavy) at top; 'up' is flipped.
    assert (top_d > bot_d) != (top_u > bot_u)


def test_oriented_key_color_distinguishable():
    red = _render_key("green", "down")
    blue = _render_key("blue", "down")
    assert _count_green(red) > 4 and _count_blue(red) == 0
    assert _count_blue(blue) > 4 and _count_green(blue) == 0


def px_half(img):
    return img.shape[0] // 2


# --------------------------------------------------------------------------- #
# cue distribution
# --------------------------------------------------------------------------- #
def test_all_four_cues_factorized():
    env = MemoryEnv(MemoryEnvConfig(cue_distribution="factorized"))
    seen = set()
    for s in range(300):
        _, info = env.reset(seed=s)
        seen.add(info["cue_type"])
    assert seen == set(CUE_TYPES)


def test_entangled_only_two_cues():
    env = MemoryEnv(MemoryEnvConfig(cue_distribution="entangled"))
    seen = set()
    for s in range(100):
        _, info = env.reset(seed=s)
        seen.add(info["cue_type"])
    assert seen == {"green_up", "blue_down"}


# --------------------------------------------------------------------------- #
# phase progression
# --------------------------------------------------------------------------- #
def test_phase_order():
    env = MemoryEnv()
    phases, _, _ = _rollout_oracle(env, seed=0)
    seq = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    expected_order = list(PHASES)
    idx = 0
    for p in seq:
        while idx < len(expected_order) and expected_order[idx] != p:
            idx += 1
        assert idx < len(expected_order), f"phase {p} out of order in {seq}"
        idx += 1
    # the canonical mid/late phases must all be reached, in order.
    for needed in ("cue", "pre_branch_memory", "branch_choice",
                   "post_branch_memory", "door_visible", "terminal"):
        assert needed in seq, f"missing phase {needed} in {seq}"


def test_phase_ranges_cover_schedule():
    env = MemoryEnv()
    pr = env.phase_ranges()
    prev_end = 0
    for name in PHASES[:-1]:
        s, e = pr[name]
        assert s == prev_end
        assert e > s
        prev_end = e


# --------------------------------------------------------------------------- #
# visibility
# --------------------------------------------------------------------------- #
def test_cue_visible_only_during_cue_phase():
    env = MemoryEnv()
    _, infos, obss = _rollout_oracle(env, seed=3)
    cue_col = _COL_GREEN if infos[0]["cue_color"] == "green" else _COL_BLUE
    for info, obs in zip(infos, obss):
        cue_pixels = _count_color(obs, cue_col) >= 4
        if info["phase"] == "cue":
            assert cue_pixels, "cue colour must be visible during the cue phase"
        elif info["phase"] in ("blank", "pre_branch_memory", "branch_choice",
                               "post_branch_memory"):
            assert not cue_pixels, f"cue colour leaked into phase {info['phase']}"


def test_doors_visible_only_near_final_phase():
    env = MemoryEnv()
    _, infos, obss = _rollout_oracle(env, seed=5)
    for info, obs in zip(infos, obss):
        doors_present = _count_green(obs) >= 4 and _count_blue(obs) >= 4
        if info["phase"] in ("blank", "cue", "pre_branch_memory", "branch_choice"):
            assert not doors_present, f"doors leaked into {info['phase']}"
    # and the doors DO appear by the door_visible phase.
    saw_doors = any(
        info["phase"] == "door_visible"
        and _count_green(obs) >= 4 and _count_blue(obs) >= 4
        for info, obs in zip(infos, obss)
    )
    assert saw_doors, "doors must be visible during the door_visible phase"


# --------------------------------------------------------------------------- #
# task semantics
# --------------------------------------------------------------------------- #
def test_shape_determines_branch():
    env = MemoryEnv()
    for s in range(40):
        obs, info = env.reset(seed=s)
        term = trunc = False
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(oracle_action(info))
        assert info["taken_branch"] == info["cue_shape"]
        assert info["branch_correct"] is True


def test_color_determines_door():
    env = MemoryEnv()
    for s in range(40):
        obs, info = env.reset(seed=s)
        term = trunc = False
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(oracle_action(info))
        assert info["selected_door_color"] == info["cue_color"]
        assert info["success"] is True


def test_door_positions_randomize():
    env = MemoryEnv()
    reds = set()
    for s in range(40):
        _, info = env.reset(seed=s)
        reds.add(info["door_position_green"])
    assert reds == {"top", "bottom"}


# --------------------------------------------------------------------------- #
# interventions
# --------------------------------------------------------------------------- #
def test_forced_branch():
    for forced in ("up", "down"):
        env = MemoryEnv(MemoryEnvConfig(forced_branch=forced))
        mg = env._mg
        # adversarial: try to enter the OPPOSITE branch at the junction.
        opp_dir = DIR_SOUTH if forced == "up" else DIR_NORTH
        for s in range(20):
            obs, info = env.reset(seed=s)
            pol = _branch_policy(opp_dir, mg)
            term = trunc = False
            while not (term or trunc):
                obs, _, term, trunc, info = env.step(pol(obs, info))
            assert info["taken_branch"] == forced


def test_suppress_down_action():
    env = MemoryEnv(MemoryEnvConfig(suppress_down_action=True))
    mg = env._mg
    pol = _branch_policy(DIR_SOUTH, mg)   # spam the down branch
    for s in range(20):
        obs, info = env.reset(seed=s)
        term = trunc = False
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(pol(obs, info))
        assert info["taken_branch"] != "down"


def test_suppress_up_action():
    env = MemoryEnv(MemoryEnvConfig(suppress_up_action=True))
    mg = env._mg
    pol = _branch_policy(DIR_NORTH, mg)   # spam the up branch
    for s in range(20):
        obs, info = env.reset(seed=s)
        term = trunc = False
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(pol(obs, info))
        assert info["taken_branch"] != "up"


# --------------------------------------------------------------------------- #
# rewards
# --------------------------------------------------------------------------- #
def test_correct_door_reward_positive():
    env = MemoryEnv()
    obs, info = env.reset(seed=0)
    term = trunc = False
    total = 0.0
    last_r = 0.0
    while not (term or trunc):
        obs, last_r, term, trunc, info = env.step(oracle_action(info))
        total += last_r
    assert info["success"] is True
    assert env.cfg.success_reward > 0
    assert last_r > 0   # terminal step carries the success reward


def test_wrong_door_reward_not_positive():
    env = MemoryEnv()
    obs, info = env.reset(seed=0)
    mg = env._mg
    term = trunc = False
    last_r = 0.0
    while not (term or trunc):
        if info["phase"] == "door_visible":
            # head to the WRONG-colour door: pick the opposite door row.
            want = info["target_door_color"]
            wrong_side = (info["door_position_blue"] if want == "green"
                          else info["door_position_green"])
            wrong_row = mg._row_door_top if wrong_side == "top" else mg._row_door_bot
            ax, ay = int(mg.agent_pos[0]), int(mg.agent_pos[1])
            if ax < mg._x_doorcol:
                a = oracle_action(info)          # still approaching the door col
            elif ay == wrong_row:
                a = A_FORWARD
            else:
                tgt = DIR_NORTH if wrong_row < mg._my else DIR_SOUTH
                adir = int(mg.agent_dir)
                a = (A_FORWARD if adir == tgt else
                     (A_RIGHT_TURN if (adir + 1) % 4 == tgt else A_LEFT_TURN))
        else:
            a = oracle_action(info)
        obs, last_r, term, trunc, info = env.step(a)
    assert info["wrong_door"] is True
    assert info["success"] is False
    assert last_r <= 0   # wrong door earns no success bonus


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_evaluate_oracle_perfect():
    env = MemoryEnv()
    rep = evaluate(env, policy=None, n_episodes=80, seed=0)
    for cue, rate in rep["success_by_cue"].items():
        assert rate == 1.0, f"oracle should solve {cue}"
    assert rep["green_door_rate_on_blue_cues"] == 0.0
    assert rep["blue_door_rate_on_green_cues"] == 0.0


def test_record_trajectory():
    env = MemoryEnv()
    traj = record_trajectory(env, policy=None, seed=0)
    assert isinstance(traj, list) and len(traj) > 0
    for rec in traj:
        assert rec["observation"].dtype == np.uint8
        assert {"phase", "cue_type", "action", "reward", "done"} <= rec.keys()
    assert traj[-1]["done"] is True

    arr = record_trajectory(env, policy=None, seed=0, as_arrays=True)
    assert arr["observation"].shape[0] == len(traj)
    assert arr["observation"].dtype == np.uint8


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
