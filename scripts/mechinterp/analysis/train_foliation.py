#!/usr/bin/env python3
"""Learn the skill foliation: a transverse subspace separating skill leaves.

Classes (lookahead H): tunnel-coming > bridge-coming > avoid-coming > free
(priority on overlap). 4-class LDA on the activations gives a 3-dim
discriminative subspace = the foliation's TRANSVERSE directions; its orthogonal
complement approximates the leaf tangent (within-leaf / context coordinates).

Saves an orthonormal transverse basis B (3, D), per-class transverse centroids
mu_c (in B-coords), and the LDA posterior readout (W, b) used as the boxer
classifier. Suppression-by-leaf-transport (in the suppress scripts):

    gate:      softmax(W h + b)[skill] > tau_on        (boxer says "in skill leaf")
    transport: h' = h + Bᵀ(mu_avoid − B h)             (pin transverse coords to
                                                        the avoid leaf; tangent
                                                        coordinates untouched)

    python -m scripts.mechinterp.analysis.train_foliation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mechinterp.analysis.bundle import ActivationBundle  # noqa: E402
from mechinterp.analysis.train_lookahead_probes import lookahead_labels  # noqa: E402

SPECS = [("bt_ppo", "gru_h"), ("bt_dreamer", "rssm_deter")]
CLASSES = ("free", "avoid", "bridge", "tunnel")          # priority ascending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/suppression"))
    args = ap.parse_args()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score

    for name, src in SPECS:
        b = ActivationBundle(f"activation_datasets/{name}")
        lab = b.labels.sort_values(["map_id", "traj_id", "t"]).reset_index(drop=True)
        X = np.load(args.out_dir / f"Xcache_{name}.npy")
        rng = np.random.default_rng(0)
        sel = rng.choice(len(lab), min(200_000, len(lab)), replace=False)
        sel.sort()
        ids = lab["row_id"].to_numpy()[sel]
        sel_sorted = sel[np.argsort(ids)]

        y = np.zeros(len(sel_sorted), np.int32)          # free
        for ci, sk in ((1, "avoid"), (2, "bridge"), (3, "tunnel")):  # priority
            m = lookahead_labels(lab, sk, args.horizon)[sel_sorted] == 1
            y[m] = ci
        gmap = lab["map_id"].to_numpy()[sel_sorted]
        u = np.unique(gmap)
        tr_maps = set(rng.choice(u, int(0.7 * len(u)), replace=False).tolist())
        trm = np.array([g in tr_maps for g in gmap])

        # ---- INLP-style transverse bundle: tunnel-vs-avoid / bridge-vs-avoid;
        # iterate probe → record direction → project out, until the leaf is
        # linearly undetectable. U spans EVERYTHING a linear boxer can see.
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        inlp = {}
        for sk, ci in (("tunnel", 3), ("bridge", 2)):
            m2 = (y == ci) | (y == 1)
            Xa = X[m2].astype(np.float64); ya = (y[m2] == ci).astype(int)
            ta = trm[m2]
            dirs, aucs = [], []
            Xr = Xa.copy()
            mu_av_resid = None
            for r in range(40):
                clf = LogisticRegression(max_iter=600, C=1.0).fit(Xr[ta], ya[ta])
                auc = roc_auc_score(ya[~ta], clf.decision_function(Xr[~ta]))
                if auc < 0.55:
                    break
                wr = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
                for d in dirs:                              # orthogonalize
                    wr = wr - (wr @ d) * d
                n = np.linalg.norm(wr)
                if n < 1e-6:
                    break
                wr /= n
                dirs.append(wr); aucs.append(float(auc))
                Xr = Xr - np.outer(Xr @ wr, wr)             # remove direction
            U = np.stack(dirs) if dirs else np.zeros((0, X.shape[1]))
            Zu = Xa @ U.T
            mu_av = Zu[(ya == 0) & ta].mean(0) if len(dirs) else np.zeros(0)
            inlp[f"U_{sk}"] = U.astype(np.float32)
            inlp[f"muav_{sk}"] = mu_av.astype(np.float32)
            inlp[f"aucs_{sk}"] = np.array(aucs, np.float32)
            print(f"  {name} {sk}: INLP bundle rank={len(dirs)} "
                  f"(auc trace {aucs[:3]}…{aucs[-1:] if aucs else []})", flush=True)

        lda = LinearDiscriminantAnalysis(n_components=3).fit(X[trm], y[trm])
        acc = balanced_accuracy_score(y[~trm], lda.predict(X[~trm]))
        # orthonormal transverse basis from the LDA scalings
        B, _ = np.linalg.qr(lda.scalings_[:, :3])        # (D, 3)
        B = B.T.astype(np.float64)                       # (3, D)
        Z = X[trm] @ B.T
        mu = np.stack([Z[y[trm] == c].mean(0) for c in range(4)])   # (4, 3)
        sd = np.stack([Z[y[trm] == c].std(0) for c in range(4)])
        # boxer posterior readout (LDA decision function is linear)
        W = lda.coef_.astype(np.float64)                 # (4, D)
        bias = lda.intercept_.astype(np.float64)
        np.savez(args.out_dir / f"foliation_{name}.npz",
                 B=B, mu=mu, sd=sd, W=W, b=bias, classes=np.array(CLASSES),
                 horizon=args.horizon, balanced_acc=acc, **inlp)
        sep = np.linalg.norm(mu[3] - mu[1]) / (sd[3].mean() + 1e-9)
        print(f"{name}: 4-class balanced acc={acc:.3f} (chance .25) | "
              f"tunnel↔avoid centroid sep={sep:.1f}σ | saved foliation_{name}.npz",
              flush=True)


if __name__ == "__main__":
    main()
