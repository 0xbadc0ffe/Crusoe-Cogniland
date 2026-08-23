#!/usr/bin/env python
"""Held-out per-category evaluation of a fork_wall PPO+GRU checkpoint on the fixed
test split. Mirrors eval_forkwall_fixed.py (the Dreamer eval) so the numbers are
directly comparable: per category it reports correct-door success and the door
distribution; the decisive metric is lakes+rocky success (chance = 50%).

PPO is trained and run STOCHASTICALLY (high entropy) -- default here samples the
policy; pass --greedy for argmax.

  python scripts/bridge_tunnel/eval_forkwall_ppo.py \
      --checkpoint final_models/ppo/ppo_plain.pt \
      --maps data/bridge_tunnel/forkwall6k/test.pkl --n 150
"""
from __future__ import annotations
import argparse, pathlib, pickle, sys
from collections import defaultdict
import numpy as np, torch

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
from cogniland.bridge_tunnel.env import BridgeTunnelEnv
from cogniland.bridge_tunnel.policy import PPOGRUPolicy

CATS = ["balanced", "lakes", "rocky"]
# canonical fork_wall env (plain reward) -- matches final_models/ENVIRONMENT.md
ENV_KW = dict(
    variant="btc", commit=False, fork_wall=True,
    passage_half=1, wall_margin=1, mem_gap=16, shaping_gamma=1.0,
    size=32, width=64, view_size=21, max_steps=800,
    orientation="natural", tree_frac=0.03, goal_half=0,
    slack_penalty=-0.01, shaping_coef=0.015, reach_bonus=3.0,
    build_cost=0.0, commit_cost=0.05, illegal_penalty=0.02, gamma=0.99,
)


@torch.no_grad()
def run_episode(policy, device, rec, greedy):
    env = BridgeTunnelEnv(**{**ENV_KW, "categories": (rec.category,)})
    env._fixed_record = rec; raw, info = env.reset()
    hidden = torch.zeros(1, 1, policy.gru_hidden, device=device)
    done = torch.zeros(1, device=device)
    for t in range(env.max_steps):
        obs = {"minimap": torch.as_tensor(raw["minimap"], device=device)[None],
               "scalars": torch.as_tensor(raw["scalars"], device=device, dtype=torch.float32)[None]}
        if greedy:
            feat, h_new = policy._gru_forward({k: v.unsqueeze(0) for k, v in obs.items()},
                                              done.unsqueeze(0), hidden)
            logits, _ = policy._heads(feat.squeeze(0)); a = int(logits.argmax(-1)); hidden = h_new
        else:
            action, _, _, _, hidden = policy.get_action_and_value(obs, hidden, done)
            a = int(action.item())
        raw, r, term, trunc, info = env.step(a)
        done = torch.tensor([float(term or trunc)], device=device)
        if term or trunc: break
    reached = bool(info.get("reached_any_target", False))
    door = ("top" if env._traj[-1][0] < env.height / 2 else "bottom") if reached else "timeout"
    return dict(reached=reached, success=bool(info.get("reached_target", False)),
                door=door, length=t + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="final_models/ppo/ppo_plain.pt")
    ap.add_argument("--maps", default="data/bridge_tunnel/forkwall6k/test.pkl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--out", default="outputs/bridge_tunnel_forkwall/eval_ppo_plain.json")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    probe = BridgeTunnelEnv(**{**ENV_KW, "categories": ("balanced",)})
    policy = PPOGRUPolicy.from_checkpoint(ckpt, probe.observation_space, device=dev)

    recs = pickle.load(open(args.maps, "rb"))
    by = defaultdict(list)
    for r in recs: by[r.category].append(r)
    rng = np.random.default_rng(0)

    print(f"checkpoint: {args.checkpoint}   mode: {'greedy' if args.greedy else 'stochastic'}")
    print(f"{'category':10s} {'success':>8s} {'reached':>8s} {'->top':>7s} {'->bottom':>9s} {'timeout':>8s} {'meanlen':>8s}")
    results = {}
    for cat in CATS:
        pool = by[cat]; idx = rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)
        eps = [run_episode(policy, dev, pool[i], args.greedy) for i in idx]
        succ = np.mean([e["success"] for e in eps]); reach = np.mean([e["reached"] for e in eps])
        top = np.mean([e["door"] == "top" for e in eps]); bot = np.mean([e["door"] == "bottom" for e in eps])
        tmo = np.mean([e["door"] == "timeout" for e in eps]); mlen = np.mean([e["length"] for e in eps])
        results[cat] = dict(n=len(eps), success=float(succ), reached=float(reach),
                            top=float(top), bottom=float(bot), timeout=float(tmo), mean_len=float(mlen))
        print(f"{cat:10s} {succ*100:7.1f}% {reach*100:7.1f}% {top*100:6.1f}% {bot*100:8.1f}% {tmo*100:7.1f}% {mlen:8.1f}")
    overall = float(np.mean([results[c]["success"] for c in CATS]))
    decisive = float(np.mean([results[c]["success"] for c in ("lakes", "rocky")]))
    verdict = ("LEARNED THE MEMORY" if decisive > 0.75 else
               "PARTIAL" if decisive > 0.6 else "GUESSING")
    print(f"\noverall success (macro): {overall*100:.1f}%")
    print(f"decisive-door success (lakes+rocky, chance=50%): {decisive*100:.1f}%")
    print(f"verdict: {verdict}")
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import json; out.write_text(json.dumps(dict(per_category=results, overall=overall,
        decisive=decisive, verdict=verdict, checkpoint=args.checkpoint,
        mode=("greedy" if args.greedy else "stochastic")), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
