#!/usr/bin/env python
"""Belief analysis for the r2dreamer bridge_tunnel fork_wall (BT-rules) agent.

Two experiments, both driven off the trained RSSM's belief state (stoch+deter
"feat") at the step right before the agent commits to top/bottom at the fork:

  1. belief -> final fork decision: fit a probe (category from belief), report
     category->door behavioural matrix, and a causal belief-SWAP test (patch
     the belief with a different category's class-mean prototype, re-query the
     actor, check whether the decision flips to match the swapped category).

  2. belief -> imagined rollout ("dream") composition: the checkpoint was
     trained with r2dreamer's default `rep_loss=r2dreamer` (decoder-free), so
     there is no reconstruction decoder yet. We attach a fresh MultiDecoder
     and train it via frozen-backbone probing (encoder/rssm/actor frozen) on
     real transitions collected under the trained policy -- standard
     post-hoc-decoder practice (same idea as vanilla Dreamer's own video_pred,
     which also trains its decoder by reconstruction against a frozen RSSM
     during normal training; here we just do it as a separate probing pass).
     Then: imagine forward from real belief states and from swapped beliefs,
     decode each step, and measure water/rock tile fractions per category.

Run (r2dreamer conda env, PYTHONPATH=src):
  python scripts/bridge_tunnel/dreamer_belief_report_r2d.py \
      --checkpoint external/r2dreamer/runs/forkwall_nocommit/latest.pt \
      --out outputs/bridge_tunnel_forkwall/belief_report_data.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "external" / "r2dreamer"))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from cogniland.bridge_tunnel.tiles import NUM_TILES, WATER, ROCK  # noqa: E402

CATEGORIES = ("balanced", "lakes", "rocky")
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_BUILD, A_MINE = range(6)

ENV_KW = dict(
    variant="btc", commit=False, fork_wall=True,
    categories=CATEGORIES,
    passage_half=1, wall_margin=1, mem_gap=16, shaping_gamma=1.0,
    size=32, width=64, view_size=21, max_steps=800,
    orientation="natural", tree_frac=0.03, goal_half=0,
    slack_penalty=-0.01, shaping_coef=0.015, reach_bonus=3.0,
    build_cost=0.0, commit_cost=0.05, illegal_penalty=0.02,
    gamma=0.99,
)
VIEW = ENV_KW["view_size"]
N_SCALARS = 5  # commit=False -> bt-rules scalar layout


def flatten_obs(raw_obs):
    minimap = np.asarray(raw_obs["minimap"], dtype=np.int64)
    onehot = np.zeros((VIEW, VIEW, NUM_TILES), dtype=np.float32)
    rr, cc = np.indices((VIEW, VIEW))
    onehot[rr, cc, minimap] = 1.0
    return np.concatenate([onehot.reshape(-1), np.asarray(raw_obs["scalars"], dtype=np.float32)])


def unflatten_vector(vec):
    """Inverse of flatten_obs's minimap part -> (VIEW,VIEW) tile-id argmax grid."""
    onehot = vec[: VIEW * VIEW * NUM_TILES].reshape(VIEW, VIEW, NUM_TILES)
    return onehot.argmax(-1)


# --------------------------------------------------------------------------- #
# agent loading
# --------------------------------------------------------------------------- #
def load_agent(checkpoint, device, model_size="size25M"):
    from dreamer import Dreamer
    import networks
    import gymnasium as gym

    cfg_dir = str(_REPO / "external" / "r2dreamer" / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        config = compose(
            config_name="configs",
            overrides=[
                "env=bridge_tunnel_forkwall", "env.task=bridgetunnel_forkwall",
                f"model={model_size}", f"device={device}", "model.compile=False",
            ],
        )

    vec_dim = VIEW * VIEW * NUM_TILES + N_SCALARS
    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(-np.inf, np.inf, (vec_dim,), np.float32),
        "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
        "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
        "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
    })

    class _OneHotSpace(gym.spaces.Box):
        discrete = True

    act_space = _OneHotSpace(low=0, high=1, shape=(6,), dtype=np.float32)

    agent = Dreamer(config.model, obs_space, act_space).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    agent.eval()

    # attach a fresh reconstruction decoder (absent under rep_loss=r2dreamer)
    shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
    decoder = networks.MultiDecoder(config.model.decoder, agent.rssm._deter, agent.rssm.flat_stoch, shapes)
    agent.decoder = decoder.to(device)
    return agent, config


# --------------------------------------------------------------------------- #
# batched rollout collection (single process, many envs, shared agent calls)
# --------------------------------------------------------------------------- #
class EnvSlot:
    __slots__ = ("env", "category", "seed", "step", "traj", "done", "wall_col", "decision_found")

    def __init__(self, env, category, seed):
        self.env = env
        self.category = category
        self.seed = seed
        self.step = 0
        self.traj = []  # list of dict per-step
        self.done = False
        self.wall_col = None
        self.decision_found = False


def make_env(category, seed):
    kw = dict(ENV_KW)
    kw["categories"] = (category,)
    return BridgeTunnelEnv(seed=seed, **kw)


@torch.no_grad()
def collect_episodes(agent, device, n_per_category, seed0=2_000_000, decoder_buf=None, decoder_buf_stride=4):
    """Roll out n_per_category episodes per category with the trained policy,
    batched across a single process (no multiprocessing -- env stepping is
    fast; batching just keeps the GPU forward pass efficient).

    Returns a list of episode dicts:
      {category, actions[T], stoch[T,S,K], deter[T,D], wall_col, tiles_pre[V,V]
       (minimap right at the decision step), row_pre (agent row at decision),
       decision_action (0=up,1=down), door ("top"/"bottom"/"timeout"),
       success (bool)}
    """
    B = min(30, n_per_category)
    slots = []
    next_seed = {}
    for cat in CATEGORIES:
        next_seed[cat] = seed0 + CATEGORIES.index(cat) * 10_000_000
    remaining = {cat: n_per_category for cat in CATEGORIES}
    cat_cycle = list(CATEGORIES)

    def spawn_slot():
        for _ in range(len(cat_cycle)):
            cat = cat_cycle[0]
            cat_cycle.append(cat_cycle.pop(0))
            if remaining[cat] > 0:
                remaining[cat] -= 1
                seed = next_seed[cat]
                next_seed[cat] += 1
                env = make_env(cat, seed)
                slot = EnvSlot(env, cat, seed)
                raw_obs, info = env.reset(seed=seed)
                slot.wall_col = env._record.wall_col
                slot.traj.append({"raw_obs": raw_obs, "info": info, "action": None})
                return slot
        return None

    for _ in range(B):
        s = spawn_slot()
        if s is not None:
            slots.append(s)

    episodes = []
    state = None
    t0 = time.time()
    n_done = 0
    n_total = n_per_category * len(CATEGORIES)
    while slots:
        B_now = len(slots)
        vecs = np.stack([flatten_obs(s.traj[-1]["raw_obs"]) for s in slots])
        is_first = np.array([len(s.traj) == 1 for s in slots])
        obs = {
            "vector": torch.as_tensor(vecs, device=device, dtype=torch.float32),
            "is_first": torch.as_tensor(is_first, device=device),
        }
        if state is None or state.batch_size[0] != B_now:
            state = agent.get_initial_state(B_now)
        else:
            # reset per-episode state for slots that just started (is_first)
            if is_first.any():
                fresh = agent.get_initial_state(B_now)
                mask = torch.as_tensor(is_first, device=device)
                state["stoch"] = torch.where(mask[:, None, None], fresh["stoch"], state["stoch"])
                state["deter"] = torch.where(mask[:, None], fresh["deter"], state["deter"])
                state["prev_action"] = torch.where(mask[:, None], fresh["prev_action"], state["prev_action"])
        action, state = agent.act(obs, state, eval=True)
        action_idx = action.argmax(-1).cpu().numpy()
        stoch_np = state["stoch"].cpu().numpy()
        deter_np = state["deter"].cpu().numpy()

        new_slots = []
        for i, s in enumerate(slots):
            a = int(action_idx[i])
            raw_obs, reward, term, trunc, info = s.env.step(a)
            done = bool(term or trunc)
            s.traj[-1]["action"] = a
            s.traj[-1]["stoch"] = stoch_np[i]
            s.traj[-1]["deter"] = deter_np[i]
            s.traj[-1]["pos"] = tuple(s.env._traj[-2]) if len(s.env._traj) >= 2 else s.env._traj[-1]
            if decoder_buf is not None and s.step % decoder_buf_stride == 0:
                decoder_buf.append((stoch_np[i], deter_np[i], vecs[i]))
            s.step += 1
            if done:
                # finalize episode record
                ep = finalize_episode(s, info)
                if ep is not None:
                    episodes.append(ep)
                n_done += 1
                nxt = spawn_slot()
                if nxt is not None:
                    new_slots.append(nxt)
            else:
                s.traj.append({"raw_obs": raw_obs, "info": info, "action": None})
                new_slots.append(s)
        slots = new_slots
        if n_done % 20 == 0 and n_done > 0:
            print(f"  [collect] {n_done}/{n_total} episodes, {time.time()-t0:.1f}s", flush=True)
    print(f"[collect] done: {len(episodes)} episodes in {time.time()-t0:.1f}s")
    return episodes


def finalize_episode(slot, last_info):
    """Find the decision step (first UP/DOWN action taken after crossing the
    wall_col), record pre-decision belief + minimap, and the eventual door."""
    wall_col = slot.wall_col
    decision_idx = None
    for i, step in enumerate(slot.traj):
        pos = step.get("pos")
        a = step["action"]
        if pos is None or a is None:
            continue
        col = pos[1]
        if col >= wall_col and a in (A_UP, A_DOWN):
            decision_idx = i
            break
    if decision_idx is None or decision_idx == 0:
        return None  # never made a clean up/down decision after the gate (rare)

    pre = slot.traj[decision_idx - 1]  # belief the step BEFORE the decision action
    if "stoch" not in pre:
        return None

    # mid-corridor snapshot: first step whose column reaches half the distance
    # to the wall. Used for Experiment 2 -- imagining forward from here still
    # has to "walk through" the rest of the category-revealing terrain, unlike
    # the passage-entry belief (which is already past it).
    mid_target_col = wall_col // 2
    mid = None
    for step in slot.traj[:decision_idx]:
        pos = step.get("pos")
        if pos is not None and pos[1] >= mid_target_col and "stoch" in step:
            mid = step
            break
    if mid is None:
        mid = pre  # short episode fallback

    reached_target = bool(last_info.get("reached_target", False))
    reached_any = bool(last_info.get("reached_any_target", False))
    door = "timeout"
    if reached_any:
        # top corridor row < bottom corridor row by construction (natural map, row grows downward)
        final_pos = None
        for step in reversed(slot.traj):
            if step.get("pos") is not None:
                final_pos = step["pos"]
                break
        mid_row = slot.env.height / 2.0
        door = "top" if (final_pos is not None and final_pos[0] < mid_row) else "bottom"

    return {
        "category": slot.category,
        "seed": slot.seed,
        "wall_col": wall_col,
        "decision_action": int(pre_action_of(slot, decision_idx)),
        "stoch_pre": pre["stoch"],
        "deter_pre": pre["deter"],
        "minimap_pre": np.asarray(pre["raw_obs"]["minimap"], dtype=np.int64),
        "stoch_mid": mid["stoch"],
        "deter_mid": mid["deter"],
        "minimap_mid": np.asarray(mid["raw_obs"]["minimap"], dtype=np.int64),
        "door": door,
        "success": reached_target,
        "episode_len": len(slot.traj),
    }


def pre_action_of(slot, decision_idx):
    return slot.traj[decision_idx]["action"]


if __name__ == "__main__":
    print("This module is imported by run_belief_report.py; see that script for the CLI entrypoint.")


# --------------------------------------------------------------------------- #
# decoder training (frozen encoder/rssm/actor; decoder-only probe)
# --------------------------------------------------------------------------- #
def train_decoder(agent, device, decoder_buf, steps=3000, batch_size=256, lr=3e-4, log_every=500):
    for p in agent.rssm.parameters():
        p.requires_grad_(False)
    for p in agent.encoder.parameters():
        p.requires_grad_(False)
    for p in agent.actor.parameters():
        p.requires_grad_(False)
    for p in agent.decoder.parameters():
        p.requires_grad_(True)

    stoch = torch.as_tensor(np.stack([b[0] for b in decoder_buf]), device=device, dtype=torch.float32)
    deter = torch.as_tensor(np.stack([b[1] for b in decoder_buf]), device=device, dtype=torch.float32)
    target = torch.as_tensor(np.stack([b[2] for b in decoder_buf]), device=device, dtype=torch.float32)
    n = stoch.shape[0]
    print(f"[decoder] training on {n} real transitions")

    opt = torch.optim.Adam(agent.decoder.parameters(), lr=lr)
    agent.decoder.train()
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        dist = agent.decoder(stoch[idx], deter[idx])["vector"]
        loss = torch.mean(-dist.log_prob(target[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                pred = dist.mode()
                pred_map = pred[:, : VIEW * VIEW * NUM_TILES].reshape(-1, VIEW, VIEW, NUM_TILES).argmax(-1)
                true_map = target[idx][:, : VIEW * VIEW * NUM_TILES].reshape(-1, VIEW, VIEW, NUM_TILES).argmax(-1)
                acc = (pred_map == true_map).float().mean().item()
            print(f"  [decoder] step={step:5d} loss={loss.item():.4f} per-cell-acc={acc:.4f} ({time.time()-t0:.0f}s)")
    agent.decoder.eval()
    for p in agent.decoder.parameters():
        p.requires_grad_(False)

    # held-out eval on a fresh slice never trained on (last 10%)
    with torch.no_grad():
        n_eval = max(1, n // 10)
        eval_idx = torch.arange(n - n_eval, n, device=device)
        dist = agent.decoder(stoch[eval_idx], deter[eval_idx])["vector"]
        pred = dist.mode()
        pred_map = pred[:, : VIEW * VIEW * NUM_TILES].reshape(-1, VIEW, VIEW, NUM_TILES).argmax(-1)
        true_map = target[eval_idx][:, : VIEW * VIEW * NUM_TILES].reshape(-1, VIEW, VIEW, NUM_TILES).argmax(-1)
        held_out_acc = (pred_map == true_map).float().mean().item()
    print(f"[decoder] held-out per-cell tile accuracy: {held_out_acc:.4f}")
    return held_out_acc



def class_mode_stoch(stoch_arr):
    """Winner-take-all one-hot prototype: per (S,K) discrete group, the class
    with the highest average sampling frequency across episodes -- NOT a plain
    float mean, which would average across one-hot samples into a soft mixture
    the RSSM/actor have never seen at training time (stoch is always a proper
    one-hot-per-group sample; a naive mean silently leaves that manifold)."""
    mean = stoch_arr.mean(0)  # (S, K) soft distribution per group
    idx = mean.argmax(-1)
    onehot = np.zeros_like(mean)
    onehot[np.arange(mean.shape[0]), idx] = 1.0
    return onehot

# --------------------------------------------------------------------------- #
# Experiment 1: belief -> fork decision (probe + confusion + causal swap)
# --------------------------------------------------------------------------- #
def feat_of(stoch, deter):
    """(S,K)+(D,) -> flat feat, matching rssm.get_feat's flatten-and-concat."""
    return np.concatenate([stoch.reshape(-1), deter.reshape(-1)])


def fit_logreg_probe(feats_train, labels_train, feats_test, labels_test, seed=0):
    """L2-regularized multinomial logistic regression, CV'd over C. Returns
    (accuracy, confusion, pred_test, per_class_weight_in_raw_feature_space).

    StandardScaler makes the fit well-posed (the raw feat concatenates 0/1
    stoch entries with much-larger-magnitude deter entries -- unscaled L2
    would implicitly under-penalize whichever sub-space has larger raw scale)
    and its scaling is inverted afterward so the returned weight vectors are
    directly comparable to / substitutable for the nearest-centroid means in
    the raw belief-feature space used by the causal swap test."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(feats_train)
    Xtr = scaler.transform(feats_train)
    Xte = scaler.transform(feats_test)

    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 12), cv=5, penalty="l2", solver="lbfgs",
        max_iter=5000, random_state=seed,
    ).fit(Xtr, labels_train)

    pred_test = clf.predict(Xte)
    acc = float((pred_test == labels_test).mean())
    confusion = np.zeros((3, 3), dtype=int)
    for t, p in zip(labels_test, pred_test):
        confusion[t, p] += 1

    # invert the standardization: logit = w_z . (x-mean)/std + b
    #                                   = (w_z/std) . x + (b - w_z.mean/std)
    # so the raw-space weight per class is w_z / std (bias is irrelevant for
    # a steering DIRECTION -- only used to rank/patch feature space, never to
    # reproduce clf.predict's actual decision).
    w_raw = clf.coef_ / scaler.scale_[None, :]  # (n_classes, n_features)
    per_class_w = {CATEGORIES[k]: w_raw[k] for k in range(3)}
    return acc, confusion, pred_test, per_class_w, clf.C_


def fit_and_run_experiment1(agent, device, episodes, train_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    by_cat = {c: [e for e in episodes if e["category"] == c] for c in CATEGORIES}
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])

    train_eps, test_eps = [], []
    for c in CATEGORIES:
        k = int(len(by_cat[c]) * train_frac)
        train_eps += by_cat[c][:k]
        test_eps += by_cat[c][k:]

    def feats_labels(point):
        feats_train = np.stack([feat_of(e[f"stoch_{point}"], e[f"deter_{point}"]) for e in train_eps])
        labels_train = np.array([CATEGORIES.index(e["category"]) for e in train_eps])
        feats_test = np.stack([feat_of(e[f"stoch_{point}"], e[f"deter_{point}"]) for e in test_eps])
        labels_test = np.array([CATEGORIES.index(e["category"]) for e in test_eps])
        return feats_train, labels_train, feats_test, labels_test

    def fit_probe(point):
        feats_train, labels_train, feats_test, labels_test = feats_labels(point)
        mu = np.stack([feats_train[labels_train == k].mean(0) for k in range(3)])
        d = ((feats_test[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        pred_test = d.argmin(-1)
        acc = float((pred_test == labels_test).mean())
        confusion = np.zeros((3, 3), dtype=int)
        for t, p in zip(labels_test, pred_test):
            confusion[t, p] += 1
        return acc, confusion, pred_test, labels_test

    probe_acc_mid, _, _, _ = fit_probe("mid")
    probe_acc, confusion, pred_test, labels_test = fit_probe("pre")

    # second, stronger linear probe: L2-regularized logistic regression
    # (discriminatively fit, CV'd C) instead of nearest-centroid -- see
    # whether the nearest-centroid number was leaving accuracy on the table.
    feats_train_mid, labels_train_mid, feats_test_mid, labels_test_mid = feats_labels("mid")
    probe_acc_logreg_mid, _, _, _, _ = fit_logreg_probe(
        feats_train_mid, labels_train_mid, feats_test_mid, labels_test_mid, seed=seed)
    feats_train_pre, labels_train_pre, feats_test_pre, labels_test_pre = feats_labels("pre")
    probe_acc_logreg, confusion_logreg, pred_test_logreg, w_logreg, logreg_C = fit_logreg_probe(
        feats_train_pre, labels_train_pre, feats_test_pre, labels_test_pre, seed=seed)

    # does the PROBE's predicted category (from belief alone, no ground truth)
    # predict the door as well as the environment's actual ground-truth label?
    # this is the within-model consistency check: the agent has no channel to
    # ground truth, only to whatever it believes -- so if predicted-category
    # tracks the door as tightly as true category, the door choice is following
    # the belief specifically, not just "the category" in the abstract.
    door_names_chk = ["top", "bottom"]
    test_nontimeout = [(e, p, t) for e, p, t in zip(test_eps, pred_test, labels_test) if e["door"] != "timeout"]
    if test_nontimeout:
        expected_door = {"balanced": None, "lakes": "bottom", "rocky": "top"}  # balanced allows either
        pred_matches_door = np.mean([
            (expected_door[CATEGORIES[p]] is None or e["door"] == expected_door[CATEGORIES[p]])
            for e, p, t in test_nontimeout
        ])
        true_matches_door = np.mean([
            (expected_door[CATEGORIES[t]] is None or e["door"] == expected_door[CATEGORIES[t]])
            for e, p, t in test_nontimeout
        ])
    else:
        pred_matches_door = true_matches_door = float("nan")

    # category -> door behavioural matrix (established fact, all episodes)
    door_names = ["top", "bottom", "timeout"]
    door_matrix = np.zeros((3, 3))
    for c_i, c in enumerate(CATEGORIES):
        eps_c = by_cat[c]
        for d_i, dname in enumerate(door_names):
            door_matrix[c_i, d_i] = np.mean([e["door"] == dname for e in eps_c]) if eps_c else np.nan

    # belief-swap causal test: ADD the (to_cat - from_cat) class-mean-difference
    # vector to each TEST episode's OWN pre-decision belief (rather than
    # replacing it outright -- a wholesale replacement also erases the "I am
    # AT the decision point right now" positional/temporal content the
    # deterministic state carries alongside category, which made the actor
    # fall back to "keep moving" instead of choosing a door; see report notes).
    # stoch is re-projected to a valid one-hot per group after the shift.
    majority_action = {}
    for c_i, c in enumerate(CATEGORIES):
        acts = [e["decision_action"] for e in train_eps if e["category"] == c]
        majority_action[c] = int(np.round(np.mean(acts))) if acts else None

    mu_stoch = {}
    mu_deter = {}
    for c_i, c in enumerate(CATEGORIES):
        stochs = np.stack([e["stoch_pre"] for e in train_eps if e["category"] == c])
        deters = np.stack([e["deter_pre"] for e in train_eps if e["category"] == c])
        mu_stoch[c] = class_mode_stoch(stochs)
        mu_deter[c] = deters.mean(0)

    def onehot_argmax(x):
        idx = x.argmax(-1)
        oh = np.zeros_like(x)
        oh[np.arange(x.shape[0]), idx] = 1.0
        return oh

    swap_results = []  # list of dicts: from_cat, to_cat, orig_action, swapped_action, matched_target
    with torch.no_grad():
        for e in test_eps:
            from_cat = e["category"]
            for to_cat in CATEGORIES:
                if to_cat == from_cat:
                    continue
                if majority_action[to_cat] is None or majority_action[to_cat] not in (A_UP, A_DOWN):
                    continue
                delta_deter = mu_deter[to_cat] - mu_deter[from_cat]
                delta_stoch = mu_stoch[to_cat] - mu_stoch[from_cat]
                swapped_stoch_np = onehot_argmax(e["stoch_pre"] + delta_stoch)
                swapped_stoch = torch.as_tensor(swapped_stoch_np, device=device, dtype=torch.float32)[None]
                swapped_deter = torch.as_tensor(e["deter_pre"] + delta_deter, device=device, dtype=torch.float32)[None]
                feat = agent.rssm.get_feat(swapped_stoch, swapped_deter)
                dist = agent.actor(feat)
                swapped_action = int(dist.mode.argmax(-1).item())
                swap_results.append(dict(
                    from_cat=from_cat, to_cat=to_cat,
                    orig_action=e["decision_action"],
                    swapped_action=swapped_action,
                    target_action=majority_action[to_cat],
                    matched_target=bool(swapped_action == majority_action[to_cat]),
                    action_changed=bool(swapped_action != e["decision_action"]),
                ))

    swap_flip_rate = float(np.mean([r["matched_target"] for r in swap_results])) if swap_results else float("nan")
    swap_change_rate = float(np.mean([r["action_changed"] for r in swap_results])) if swap_results else float("nan")

    # same swap test, but the patch delta comes from the logistic-regression
    # probe's own separating direction per class instead of the raw
    # difference-of-means -- does the more discriminative direction transfer
    # to the actor any more cleanly than the naive mean difference did?
    stoch_shape = train_eps[0]["stoch_pre"].shape  # (S, K)
    stoch_dim = int(np.prod(stoch_shape))
    w_stoch = {c: w_logreg[c][:stoch_dim].reshape(stoch_shape) for c in CATEGORIES}
    w_deter = {c: w_logreg[c][stoch_dim:] for c in CATEGORIES}

    swap_results_logreg = []
    with torch.no_grad():
        for e in test_eps:
            from_cat = e["category"]
            for to_cat in CATEGORIES:
                if to_cat == from_cat:
                    continue
                if majority_action[to_cat] is None or majority_action[to_cat] not in (A_UP, A_DOWN):
                    continue
                delta_deter = w_deter[to_cat] - w_deter[from_cat]
                delta_stoch = w_stoch[to_cat] - w_stoch[from_cat]
                swapped_stoch_np = onehot_argmax(e["stoch_pre"] + delta_stoch)
                swapped_stoch = torch.as_tensor(swapped_stoch_np, device=device, dtype=torch.float32)[None]
                swapped_deter = torch.as_tensor(e["deter_pre"] + delta_deter, device=device, dtype=torch.float32)[None]
                feat = agent.rssm.get_feat(swapped_stoch, swapped_deter)
                dist = agent.actor(feat)
                swapped_action = int(dist.mode.argmax(-1).item())
                swap_results_logreg.append(dict(
                    from_cat=from_cat, to_cat=to_cat,
                    orig_action=e["decision_action"],
                    swapped_action=swapped_action,
                    target_action=majority_action[to_cat],
                    matched_target=bool(swapped_action == majority_action[to_cat]),
                    action_changed=bool(swapped_action != e["decision_action"]),
                ))
    swap_flip_rate_logreg = (float(np.mean([r["matched_target"] for r in swap_results_logreg]))
                              if swap_results_logreg else float("nan"))
    swap_change_rate_logreg = (float(np.mean([r["action_changed"] for r in swap_results_logreg]))
                                if swap_results_logreg else float("nan"))

    return dict(
        probe_acc=probe_acc,
        probe_acc_mid=probe_acc_mid,
        probe_acc_logreg=probe_acc_logreg,
        probe_acc_logreg_mid=probe_acc_logreg_mid,
        logreg_C=float(np.ravel(logreg_C)[0]),
        confusion_logreg=confusion_logreg.tolist(),
        pred_matches_door=float(pred_matches_door),
        true_matches_door=float(true_matches_door),
        confusion=confusion.tolist(),
        confusion_labels=list(CATEGORIES),
        door_matrix=door_matrix.tolist(),
        door_labels=door_names,
        majority_action=majority_action,
        swap_results=swap_results,
        swap_flip_rate=swap_flip_rate,
        swap_change_rate=swap_change_rate,
        swap_results_logreg=swap_results_logreg,
        swap_flip_rate_logreg=swap_flip_rate_logreg,
        swap_change_rate_logreg=swap_change_rate_logreg,
        n_train=len(train_eps), n_test=len(test_eps),
        mu_stoch={c: mu_stoch[c] for c in CATEGORIES},
        mu_deter={c: mu_deter[c] for c in CATEGORIES},
        train_eps=train_eps, test_eps=test_eps,
    )


# --------------------------------------------------------------------------- #
# Experiment 2: belief -> imagined rollout ("dream") composition
# --------------------------------------------------------------------------- #
@torch.no_grad()
def imagine_rollout(agent, device, stoch0, deter0, horizon=16, greedy=True):
    """Roll the trained actor forward through PRIOR (imagined) dynamics only,
    decoding a minimap at every step. stoch0/deter0: single-example numpy
    arrays (no batch dim)."""
    stoch = torch.as_tensor(stoch0, device=device, dtype=torch.float32)[None]
    deter = torch.as_tensor(deter0, device=device, dtype=torch.float32)[None]
    tile_grids = []
    tile_fracs = []
    for _ in range(horizon):
        feat = agent.rssm.get_feat(stoch, deter)
        dec = agent.decoder(stoch, deter)["vector"].mode()[0].cpu().numpy()
        grid = unflatten_vector(dec)
        tile_grids.append(grid)
        tile_fracs.append(dict(
            water=float((grid == WATER).mean()),
            rock=float((grid == ROCK).mean()),
        ))
        action_dist = agent.actor(feat)
        action = action_dist.mode if greedy else action_dist.rsample()
        stoch, deter = agent.rssm.img_step(stoch, deter, action)
    return tile_grids, tile_fracs


def run_experiment2(agent, device, exp1, n_dream_per_category=20, horizon=16, seed=0):
    rng = np.random.default_rng(seed)
    all_eps = exp1["train_eps"] + exp1["test_eps"]
    by_cat = {c: [e for e in all_eps if e["category"] == c] for c in CATEGORIES}

    # mid-corridor class-mean belief (separate from Experiment 1's pre-decision
    # means): the imagination seed point here is still inside the
    # category-revealing terrain, so "dreaming forward" has to hallucinate the
    # rest of that terrain based on belief alone.
    mu_stoch_mid = {c: class_mode_stoch(np.stack([e["stoch_mid"] for e in by_cat[c]])) for c in CATEGORIES}
    mu_deter_mid = {c: np.stack([e["deter_mid"] for e in by_cat[c]]).mean(0) for c in CATEGORIES}

    dream_stats = {c: {"water": [], "rock": []} for c in CATEGORIES}
    example_sequences = {}  # one representative real-belief dream per category (full grid sequence)
    for c in CATEGORIES:
        eps_c = by_cat[c]
        idxs = rng.choice(len(eps_c), size=min(n_dream_per_category, len(eps_c)), replace=False)
        for j, i in enumerate(idxs):
            e = eps_c[i]
            grids, fracs = imagine_rollout(agent, device, e["stoch_mid"], e["deter_mid"], horizon=horizon)
            dream_stats[c]["water"].append(np.mean([f["water"] for f in fracs]))
            dream_stats[c]["rock"].append(np.mean([f["rock"] for f in fracs]))
            if c not in example_sequences:
                example_sequences[c] = dict(
                    grids=[g.tolist() for g in grids],
                    minimap_seed=e["minimap_mid"].tolist(),
                    seed=e["seed"],
                )

    # belief-swap dream: imagine from each category's OWN mid-corridor class-mean
    # belief, and from the OTHER categories' means, all from the same "blank"
    # starting point -- isolates what the belief alone contributes to the dream.
    swap_sequences = {}
    for c in CATEGORIES:
        for to_cat in CATEGORIES:
            grids, fracs = imagine_rollout(agent, device, mu_stoch_mid[to_cat], mu_deter_mid[to_cat], horizon=horizon)
            swap_sequences[f"{c}_to_{to_cat}"] = dict(
                grids=[g.tolist() for g in grids],
                water_frac=float(np.mean([f["water"] for f in fracs])),
                rock_frac=float(np.mean([f["rock"] for f in fracs])),
            )

    return dict(dream_stats=dream_stats, example_sequences=example_sequences, swap_sequences=swap_sequences,
                horizon=horizon, mu_stoch_mid=mu_stoch_mid, mu_deter_mid=mu_deter_mid)
