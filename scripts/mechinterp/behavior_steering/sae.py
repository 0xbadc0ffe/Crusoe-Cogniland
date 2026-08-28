#!/usr/bin/env python3
"""Sparse autoencoder on PPO's 128-d GRU state -> sae_ppo.pt.

Anthropic-style small SAE: f = ReLU(W_e (x - b_d) + b_e), x_hat = W_d f + b_d,
unit-norm decoder columns, L1 on f. Inputs standardised by TRAIN-map mean/std;
train rows are TRAIN maps only, model selection on held-out TEST-map
reconstruction. lambda is swept over three values and the model whose mean L0
lands in [10, 30] with the best held-out R^2 is kept.

Honesty check included: R^2 is compared against a top-k PCA reconstruction at
matched k = mean L0. If the SAE cannot beat dense PCs at the same per-sample
dimensionality, it has learnt little beyond a rotation, and the report says so.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=scripts/mechinterp/belief_report \
      python scripts/mechinterp/behavior_steering/sae.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/mechinterp/belief_report"))
from data import load, split_maps  # noqa: E402

OUT = REPO / "outputs/behavior_steering"
D, F = 128, 1024
LAMBDAS = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]   # sum-form L1 (per sample)
EPOCHS, BATCH, LR, PATIENCE = 60, 4096, 1e-3, 6
SEED = 0


class SAE(nn.Module):
    def __init__(self, d=D, f=F):
        super().__init__()
        self.We = nn.Parameter(torch.randn(f, d) * 0.1)
        self.be = nn.Parameter(torch.zeros(f))
        self.Wd = nn.Parameter(torch.randn(d, f) * 0.1)
        self.bd = nn.Parameter(torch.zeros(d))
        self.norm_dec()

    @torch.no_grad()
    def norm_dec(self):
        self.Wd.data /= self.Wd.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x):
        return torch.relu((x - self.bd) @ self.We.T + self.be)

    def forward(self, x):
        f = self.encode(x)
        return f @ self.Wd.T + self.bd, f


def r2(x, xh):
    return float(1 - ((x - xh) ** 2).mean() / x.var())


def train_one(Xtr, Xva, lam, seed=SEED):
    torch.manual_seed(seed)
    m = SAE()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    n = len(Xtr)
    best = (-np.inf, None, 0)
    bad = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            xb = Xtr[perm[i:i + BATCH]]
            xh, f = m(xb)
            # sum over dims per sample (standard SAE form): with a mean over
            # the 1024 features the L1 pressure is ~1000x too weak and L0
            # lands near 500 of 1024.
            loss = ((xb - xh) ** 2).sum(1).mean() + lam * f.abs().sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            m.norm_dec()
        with torch.no_grad():
            xh, f = m(Xva)
            val = r2(Xva, xh)
            l0 = float((f > 1e-6).float().sum(1).mean())
        if val > best[0] + 1e-4:
            best = (val, {k: v.detach().clone() for k, v in m.state_dict().items()}, l0)
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(best[1])
    with torch.no_grad():
        _, f = m(Xva)
        freq = (f > 1e-6).float().mean(0)
    dead = int((freq < 1e-5).sum())
    return m, dict(lam=lam, val_r2=round(best[0], 4), l0=round(best[2], 1),
                   dead=dead, epochs=ep + 1)


def main():
    X, df = load("ppo")
    tr, te = split_maps(df)
    on_tr = df["map_id"].isin(tr).to_numpy()
    Xtr = np.asarray(X[np.flatnonzero(on_tr)], np.float32)
    Xte = np.asarray(X[np.flatnonzero(~on_tr)], np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr = torch.tensor((Xtr - mu) / sd)
    Zte = torch.tensor((Xte - mu) / sd)
    print(f"train rows {len(Ztr)}  test rows {len(Zte)}")

    runs = []
    for lam in LAMBDAS:
        m, info = train_one(Ztr, Zte, lam)
        runs.append((m, info))
        print("lam", info)
    # selection: L0 in [10,30] if reachable; else the sparsest run that keeps
    # held-out R^2 >= 0.90; else best R^2. The whole frontier is recorded.
    frontier = [dict(i) for _, i in runs]        # copies, BEFORE any mutation
    ok = [(m, i) for m, i in runs if 10 <= i["l0"] <= 30]
    if not ok:
        ok = [(m, i) for m, i in runs if i["val_r2"] >= 0.90]
        ok = [min(ok, key=lambda t: t[1]["l0"])] if ok else []
    if not ok:
        ok = [max(runs, key=lambda t: t[1]["val_r2"])]
    m, info = max(ok, key=lambda t: t[1]["val_r2"])
    info = dict(info)
    info["frontier"] = frontier

    # honesty: PCA at matched k
    k = max(int(round(info["l0"])), 1)
    Xc = Ztr - Ztr.mean(0)
    U, S, Vt = torch.linalg.svd(Xc[:20000], full_matrices=False)
    P = Vt[:k]
    Zc = Zte - Ztr.mean(0)
    rec = Zc @ P.T @ P + Ztr.mean(0)
    pca_r2 = r2(Zte, rec)
    info["pca_r2_at_L0"] = round(pca_r2, 4)

    torch.save(dict(state_dict=m.state_dict(), mu=mu, sd=sd, info=info,
                    d=D, f=F), OUT / "sae_ppo.pt")
    (OUT / "sae_info.json").write_text(json.dumps(info, indent=1))
    print("selected:", json.dumps(info))
    print("wrote", OUT / "sae_ppo.pt")


if __name__ == "__main__":
    main()
