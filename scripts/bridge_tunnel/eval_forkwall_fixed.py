#!/usr/bin/env python
"""Held-out per-category evaluation of a rep_loss=dreamer fork_wall checkpoint on
the fixed test split. Reports, per category, the correct-door success rate and
the door-choice distribution -- the decisive test of whether the world model
learned the category memory (rocky/lakes well above the 50% door-chance) or is
just guessing the door after navigating.

  python scripts/bridge_tunnel/eval_forkwall_fixed.py \
      --checkpoint r2dreamer_model/runs/forkwall_fixed_dreamer/latest.pt \
      --maps data/bridge_tunnel/forkwall6k/test.pkl --n 200
"""
from __future__ import annotations
import argparse, pathlib, pickle, sys
from collections import defaultdict
import numpy as np, torch, gymnasium as gym

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO/"src")); sys.path.insert(0, str(_REPO/"r2dreamer_model"))
sys.path.insert(0, str(_REPO/"scripts/bridge_tunnel"))
from hydra import compose, initialize_config_dir
import dreamer_belief_report_r2d as R
from dreamer import Dreamer
from cogniland.bridge_tunnel.env import BridgeTunnelEnv
from tensordict import TensorDict

CATS = ["balanced","lakes","rocky"]


def load(checkpoint, device, model_size="size25M"):
    cfg_dir = str((_REPO/"r2dreamer_model/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="configs", overrides=[
            "env=bridge_tunnel_forkwall","env.task=bridgetunnel_forkwall",
            f"model={model_size}","model.rep_loss=dreamer",f"device={device}","model.compile=False"])
    vd = R.VIEW*R.VIEW*R.NUM_TILES + R.N_SCALARS
    obs_space = gym.spaces.Dict({"vector":gym.spaces.Box(-np.inf,np.inf,(vd,),np.float32),
        "log_success":gym.spaces.Box(-np.inf,np.inf,(1,),np.float32),
        "is_first":gym.spaces.Box(0,1,(),bool),"is_last":gym.spaces.Box(0,1,(),bool),
        "is_terminal":gym.spaces.Box(0,1,(),bool)})
    class _OH(gym.spaces.Box): discrete=True
    agent = Dreamer(cfg.model, obs_space, _OH(0,1,(6,),np.float32)).to(device)
    agent.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["agent_state_dict"],strict=False)
    agent.eval()
    return agent


@torch.no_grad()
def run_episode(agent, device, rec):
    env = BridgeTunnelEnv(**{**R.ENV_KW,"categories":(rec.category,)})
    env._fixed_record = rec; raw,info = env.reset()
    st = agent.get_initial_state(1); first=True
    for t in range(env.max_steps):
        vec = R.flatten_obs(raw)
        trans = TensorDict({"vector":torch.as_tensor(vec,device=device,dtype=torch.float32)[None],
                            "is_first":torch.tensor([first],device=device)},batch_size=(1,))
        a,st = agent.act(trans, st, eval=True); first=False
        raw,r,term,trunc,info = env.step(int(a.argmax(-1)))
        if term or trunc: break
    reached = bool(info.get("reached_any_target", False))
    success = bool(info.get("reached_target", False))
    if reached:
        final = env._traj[-1]; door = "top" if final[0] < env.height/2 else "bottom"
    else:
        door = "timeout"
    return dict(reached=reached, success=success, door=door, length=t+1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="r2dreamer_model/runs/forkwall_fixed_dreamer/latest.pt")
    ap.add_argument("--maps", default="data/bridge_tunnel/forkwall6k/test.pkl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=200, help="episodes per category")
    ap.add_argument("--model-size", default="size25M")
    ap.add_argument("--out", default="outputs/bridge_tunnel_forkwall/eval_fixed_dreamer.json")
    args = ap.parse_args()

    agent = load(args.checkpoint, args.device, args.model_size)
    recs = pickle.load(open(args.maps,"rb"))
    by_cat = defaultdict(list)
    for r in recs: by_cat[r.category].append(r)

    rng = np.random.default_rng(0)
    results = {}
    print(f"checkpoint: {args.checkpoint}")
    print(f"{'category':10s} {'success':>8s} {'reached':>8s} {'->top':>7s} {'->bottom':>9s} {'timeout':>8s} {'meanlen':>8s}")
    for cat in CATS:
        pool = by_cat[cat]
        idx = rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)
        eps = [run_episode(agent, args.device, pool[i]) for i in idx]
        n = len(eps)
        succ = np.mean([e["success"] for e in eps])
        reach = np.mean([e["reached"] for e in eps])
        top = np.mean([e["door"]=="top" for e in eps])
        bot = np.mean([e["door"]=="bottom" for e in eps])
        tmo = np.mean([e["door"]=="timeout" for e in eps])
        mlen = np.mean([e["length"] for e in eps])
        results[cat] = dict(n=n, success=float(succ), reached=float(reach),
                            top=float(top), bottom=float(bot), timeout=float(tmo), mean_len=float(mlen))
        print(f"{cat:10s} {succ*100:7.1f}% {reach*100:7.1f}% {top*100:6.1f}% {bot*100:8.1f}% {tmo*100:7.1f}% {mlen:8.1f}")
    overall = float(np.mean([results[c]["success"] for c in CATS]))
    # correct-door rate among DECISIVE categories only (the memory test; chance=50%)
    decisive = float(np.mean([results[c]["success"] for c in ("lakes","rocky")]))
    print(f"\noverall success (macro): {overall*100:.1f}%")
    print(f"decisive-door success (lakes+rocky, chance=50%): {decisive*100:.1f}%")
    verdict = ("LEARNED THE MEMORY" if decisive > 0.75 else
               "PARTIAL (above chance, not solved)" if decisive > 0.6 else
               "GUESSING (near door-chance)")
    print(f"verdict: {verdict}")
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import json; out.write_text(json.dumps(dict(per_category=results, overall=overall,
        decisive=decisive, verdict=verdict, checkpoint=args.checkpoint), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
