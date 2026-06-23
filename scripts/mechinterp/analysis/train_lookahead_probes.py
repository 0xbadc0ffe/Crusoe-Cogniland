#!/usr/bin/env python3
"""Lookahead skill probes for probe-gated suppression (BT variant).

For each model (bt_ppo gru_h / bt_dreamer rssm_deter) and skill (tunnel/bridge):
    y_t = 1  iff the episode is inside OR enters a <skill> segment within the
             next H steps   (segment column of the bundle labels)
Linear logistic probe, map-grouped 70/30 split (no map leakage). The scaler is
folded into effective (w, b) so the probe is a single affine readout — its
gradient w.r.t. the activation is just w, which makes the "projected gradient
attack" closed-form. Also saves the top-k PCA components of the activations:
the suppression step is constrained to that subspace (stay on-manifold).

    python -m scripts.mechinterp.analysis.train_lookahead_probes --horizon 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mechinterp.analysis.bundle import ActivationBundle  # noqa: E402

SPECS = [("bt_ppo", "gru_h", 64), ("bt_dreamer", "rssm_deter", 256)]
SKILLS = ("tunnel", "bridge")


def lookahead_labels(lab, skill, H):
    """y[i]=1 iff segment==skill at any offset in [0, H] within the episode."""
    y = np.zeros(len(lab), dtype=np.int8)
    is_sk = (lab["segment"].to_numpy() == skill).astype(np.int8)
    start = 0
    for _, g in lab.groupby(["map_id", "traj_id"], sort=False):
        n = len(g)
        s = is_sk[start:start + n]
        # future-window max via reversed cummax over a sliding window
        fut = np.zeros(n, np.int8)
        run = 0
        for i in range(n - 1, -1, -1):
            run = H + 1 if s[i] else max(run - 1, 0)
            fut[i] = 1 if run > 0 else 0
        y[start:start + n] = fut
        start += n
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/suppression"))
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.decomposition import PCA

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, src, k in SPECS:
        b = ActivationBundle(f"activation_datasets/{name}")
        lab = b.labels.sort_values(["map_id", "traj_id", "t"]).reset_index(drop=True)
        ys = {sk: lookahead_labels(lab, sk, args.horizon) for sk in SKILLS}

        rng = np.random.default_rng(0)
        sel = rng.choice(len(lab), min(args.rows, len(lab)), replace=False)
        sel.sort()
        ids = lab["row_id"].to_numpy()[sel]
        order = np.argsort(ids)
        ids_sorted = ids[order]
        cache = args.out_dir / f"Xcache_{name}.npy"
        if cache.exists():
            X = np.load(cache)
        else:
            X = b.load_activations(src, ids_sorted)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, X)
        sel_sorted = sel[order]
        gmap = lab["map_id"].to_numpy()[sel_sorted]

        u = np.unique(gmap)
        tr_maps = set(rng.choice(u, int(0.7 * len(u)), replace=False).tolist())
        trm = np.array([g in tr_maps for g in gmap])

        out = {"horizon": args.horizon, "source": src}
        pca = PCA(k, svd_solver="randomized", random_state=0).fit(X[trm])
        out["V"] = pca.components_.astype(np.float32)          # (k, D)
        out["evr_k"] = float(pca.explained_variance_ratio_.sum())

        mu = X[trm].mean(0)
        Z = (X - mu) @ out["V"].T                               # top-k PCA coords
        for sk in SKILLS:
            y = ys[sk][sel_sorted]
            base = y[~trm].mean()
            # (a) full-space probe — fold the scaler into (w, b)
            sc = StandardScaler().fit(X[trm])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[trm]), y[trm])
            w = (clf.coef_[0] / sc.scale_).astype(np.float32)
            bias = float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())
            auc = roc_auc_score(y[~trm], X[~trm] @ w + bias)
            out[f"w_{sk}"] = w; out[f"b_{sk}"] = bias; out[f"auc_{sk}"] = float(auc)
            wp = out["V"].T @ (out["V"] @ w)
            out[f"w_in_subspace_{sk}"] = float(np.linalg.norm(wp) / np.linalg.norm(w))
            # (b) subspace-native probe — trained on PCA coords, folded back to
            # activation space (w_sub ∈ span(Vᵀ) by construction)
            scz = StandardScaler().fit(Z[trm])
            clz = LogisticRegression(max_iter=2000, C=1.0).fit(scz.transform(Z[trm]), y[trm])
            a_eff = (clz.coef_[0] / scz.scale_).astype(np.float64)
            c_eff = float(clz.intercept_[0] - (clz.coef_[0] * scz.mean_ / scz.scale_).sum())
            w_sub = (out["V"].T @ a_eff).astype(np.float32)
            b_sub = float(c_eff - w_sub @ mu)
            auc_sub = roc_auc_score(y[~trm], X[~trm] @ w_sub + b_sub)
            out[f"w_sub_{sk}"] = w_sub; out[f"b_sub_{sk}"] = b_sub
            out[f"auc_sub_{sk}"] = float(auc_sub)
            print(f"{name} {sk}: AUC full={auc:.3f} sub(k)={auc_sub:.3f} "
                  f"(base {base:.2f}) | ‖Pw‖/‖w‖={out[f'w_in_subspace_{sk}']:.2f}",
                  flush=True)

        np.savez(args.out_dir / f"probes_{name}.npz", **out)
        print(f"saved {args.out_dir}/probes_{name}.npz  (PCA k={k}, evr={out['evr_k']:.2f})",
              flush=True)


if __name__ == "__main__":
    main()
