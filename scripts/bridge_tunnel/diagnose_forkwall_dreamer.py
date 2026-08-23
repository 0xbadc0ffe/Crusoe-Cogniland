#!/usr/bin/env python
"""Diagnose WHY the fork_wall DreamerV3 world model navigates but doesn't use the
category memory. Three tests:

  T1 PROBE      Is the category belief present in the RSSM state at the fork?
                (logistic probe, held-out). Present-but-unused => credit/horizon
                problem; absent => representation problem.
  T2 HORIZON    How many steps separate "evidence last in view" from "door
                reward"? If that gap >> imag_horizon, imagination-trained actor
                never connects belief to the door bonus.
  T3 REWARD     Decompose the return into shaping vs door bonus, overall and
                *within one imagination horizon* from a mid-corridor start.

Run (r2dreamer env):
  python scripts/bridge_tunnel/diagnose_forkwall_dreamer.py --n 40
"""
from __future__ import annotations
import argparse, pathlib, pickle, sys
import numpy as np, torch, gymnasium as gym

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO/"src")); sys.path.insert(0, str(_REPO/"external/r2dreamer"))
sys.path.insert(0, str(_REPO/"scripts/bridge_tunnel"))
from hydra import compose, initialize_config_dir
import dreamer_belief_report_r2d as R
from dreamer import Dreamer
from cogniland.bridge_tunnel.env import BridgeTunnelEnv
from cogniland.bridge_tunnel.tiles import WATER, ROCK
from tensordict import TensorDict

CATS = ["balanced","lakes","rocky"]
A_UP, A_DOWN = 0, 1


def load(checkpoint, device):
    cfg_dir = str((_REPO/"external/r2dreamer/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="configs", overrides=[
            "env=bridge_tunnel_forkwall","env.task=bridgetunnel_forkwall",
            "model=size25M","model.rep_loss=dreamer",f"device={device}","model.compile=False"])
    vd = R.VIEW*R.VIEW*R.NUM_TILES + R.N_SCALARS
    obs = gym.spaces.Dict({"vector":gym.spaces.Box(-np.inf,np.inf,(vd,),np.float32),
        "log_success":gym.spaces.Box(-np.inf,np.inf,(1,),np.float32),
        "is_first":gym.spaces.Box(0,1,(),bool),"is_last":gym.spaces.Box(0,1,(),bool),
        "is_terminal":gym.spaces.Box(0,1,(),bool)})
    class _OH(gym.spaces.Box): discrete=True
    ag = Dreamer(cfg.model, obs, _OH(0,1,(6,),np.float32)).to(device)
    ag.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["agent_state_dict"],strict=False)
    ag.eval(); return ag, int(cfg.model.imag_horizon)


@torch.no_grad()
def rollout(agent, device, rec):
    env = BridgeTunnelEnv(**{**R.ENV_KW,"categories":(rec.category,)})
    env._fixed_record = rec; raw,info = env.reset()
    wall = env._record.wall_col
    # last column that carries evidence (water/rock), from the true terrain
    terr = env._terrain; ev = np.where((terr==WATER)|(terr==ROCK))
    last_ev = int(ev[1].max()) if len(ev[1]) else -1
    st = agent.get_initial_state(1); first=True
    traj=[]  # (col, action, reward, stoch, deter)
    for t in range(env.max_steps):
        vec = R.flatten_obs(raw)
        trans = TensorDict({"vector":torch.as_tensor(vec,device=device,dtype=torch.float32)[None],
                            "is_first":torch.tensor([first],device=device)},batch_size=(1,))
        a,st = agent.act(trans, st, eval=True); first=False
        col = env._traj[-1][1]
        s_np = st["stoch"][0].cpu().numpy(); d_np = st["deter"][0].cpu().numpy()
        ai = int(a.argmax(-1))
        raw,r,term,trunc,info = env.step(ai)
        traj.append((col, ai, float(r), s_np, d_np))
        if term or trunc: break
    reached = bool(info.get("reached_any_target", False))
    success = bool(info.get("reached_target", False))
    return dict(cat=rec.category, wall=wall, last_ev=last_ev, traj=traj,
                reached=reached, success=success, T=len(traj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="external/r2dreamer/runs/forkwall_fixed_dreamer/latest.pt")
    ap.add_argument("--maps", default="data/bridge_tunnel/forkwall6k/test.pkl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=40, help="episodes per category")
    args = ap.parse_args()

    agent, imag_h = load(args.checkpoint, args.device)
    recs = pickle.load(open(args.maps,"rb"))
    from collections import defaultdict
    by = defaultdict(list)
    for r in recs: by[r.category].append(r)
    rng = np.random.default_rng(0)

    eps=[]
    for c in CATS:
        idx = rng.choice(len(by[c]), size=min(args.n,len(by[c])), replace=False)
        for i in idx: eps.append(rollout(agent, args.device, by[c][i]))
    print(f"collected {len(eps)} episodes; imag_horizon={imag_h}\n")

    # ---- find each episode's DECISION step (first up/down past the wall) ----
    def feat(s,d): return np.concatenate([s.reshape(-1), d.reshape(-1)])
    fork_feats=[]; fork_cat=[]; fork_act=[]
    gaps=[]  # steps from "evidence leaves view" to "door reached"
    for e in eps:
        wall=e["wall"]; last_ev=e["last_ev"]
        dec=None
        for k,(col,a,r,s,d) in enumerate(e["traj"]):
            if col>=wall and a in (A_UP,A_DOWN): dec=k; break
        if dec is None or dec==0: continue
        pre=e["traj"][dec-1]
        fork_feats.append(feat(pre[3],pre[4])); fork_cat.append(CATS.index(e["cat"])); fork_act.append(e["traj"][dec][1])
        # step where the agent's column first exceeds last_ev+viewradius (evidence gone)
        vr=R.VIEW//2; gone=None
        for k,(col,a,r,s,d) in enumerate(e["traj"]):
            if col > last_ev+vr: gone=k; break
        if gone is not None: gaps.append(e["T"]-gone)  # steps from evidence-gone to episode end (door)

    # ---- T1: probe ----
    X=np.stack(fork_feats); y=np.array(fork_cat)
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    Xs=StandardScaler().fit_transform(X)
    clf=LogisticRegressionCV(Cs=np.logspace(-3,2,10),cv=5,max_iter=3000,random_state=0)
    yp=cross_val_predict(clf,Xs,y,cv=5)
    acc=float((yp==y).mean())
    # per-class accuracy
    print("="*64)
    print("T1  PROBE: is the category in the world-model belief at the fork?")
    print(f"    cross-val category accuracy = {acc*100:.1f}%   (chance = 33%)")
    for ci,c in enumerate(CATS):
        m=y==ci
        print(f"      {c:9s}: recovered {100*np.mean(yp[m]==ci):.0f}%  (n={m.sum()})")
    # decisive door: does the ACTOR's up/down depend on category?
    print("    ACTOR door choice by category (up=top, down=bottom):")
    fa=np.array(fork_act)
    for ci,c in enumerate(CATS):
        m=y==ci
        up=np.mean(fa[m]==A_UP); dn=np.mean(fa[m]==A_DOWN)
        print(f"      {c:9s}: up(top)={up*100:.0f}%  down(bottom)={dn*100:.0f}%")

    # ---- T2: horizon gap ----
    gaps=np.array(gaps)
    print("\n"+"="*64)
    print("T2  HORIZON: memory dependency length vs imagination horizon")
    print(f"    steps from 'evidence leaves view' to the door: "
          f"mean={gaps.mean():.1f}  median={np.median(gaps):.0f}  (imag_horizon={imag_h})")
    print(f"    fraction of episodes with gap > imag_horizon: {np.mean(gaps>imag_h)*100:.0f}%")

    # ---- T3: reward decomposition ----
    tot=[]; shap=[]; door=[]; win15=[]
    for e in eps:
        rs=np.array([x[2] for x in e["traj"]])
        db = 3.0 if e["success"] else 0.0      # reach_bonus only on correct door
        tot.append(rs.sum()); door.append(db); shap.append(rs.sum()-db)
        # return within one imagination horizon starting mid-corridor (step ~ T//2)
        s=e["T"]//2; win15.append(rs[s:s+imag_h].sum())
    tot=np.array(tot); shap=np.array(shap); door=np.array(door); win15=np.array(win15)
    print("\n"+"="*64)
    print("T3  REWARD decomposition (per episode)")
    print(f"    total return           mean={tot.mean():.2f}")
    print(f"      from shaping+slack   mean={shap.mean():.2f}   ({100*shap.mean()/max(tot.mean(),1e-9):.0f}% of return)")
    print(f"      from door bonus      mean={door.mean():.2f}   ({100*door.mean()/max(tot.mean(),1e-9):.0f}% of return)")
    print(f"    return inside ONE imag horizon from mid-corridor: mean={win15.mean():.2f}")
    print(f"      -> door bonus visible in that window? {'YES' if win15.max()>2.5 else 'NO (all shaping, door out of horizon)'}")

    # ---- verdict ----
    print("\n"+"="*64)
    belief_present = acc > 0.5
    horizon_short  = np.mean(gaps>imag_h) > 0.5
    print("VERDICT")
    if belief_present and horizon_short:
        print("  The category IS encoded in the world-model belief, but the actor")
        print("  does not use it: the door reward sits BEYOND the imagination")
        print("  horizon, so the imagination-trained policy never connects the")
        print("  belief to the door bonus. Dense shaping fills the short horizon,")
        print("  removing any pressure to look further. FIX: raise imag_horizon to")
        print("  span the corridor (and/or sparsen shaping).")
    elif not belief_present:
        print("  The category is NOT recoverable from the belief -> a representation")
        print("  failure: the RSSM did not carry the category across the corridor.")
    else:
        print("  Belief present and horizon adequate -> look elsewhere (capacity/opt).")


if __name__ == "__main__":
    main()
