#!/usr/bin/env python3
"""Generate standalone interactive (rotatable) 3-D plotly scatters of the belief
carriers, for embedding in the HTML report. Each panel = PCA-3D of an activation
source, hoverable, coloured by category / skill / ordinal belief.
Outputs full HTML files (plotly.js from CDN) to outputs/report/plotly/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle
from mechinterp.analysis import wandb_io, geometry as G

OUT = Path("outputs/report/plotly"); OUT.mkdir(parents=True, exist_ok=True)
PANELS = [
    ("activation_datasets/btc_ppo", "gru_h", "category", False, "PPO gru_h — true map type"),
    ("activation_datasets/btc_ppo", "gru_h", "final_commit", False, "PPO gru_h — committed skill"),
    ("activation_datasets/btc_dreamer", "rssm_deter", "category", False, "Dreamer rssm_deter — true map type"),
    ("activation_datasets/btc_dreamer", "rssm_deter", "final_commit", False, "Dreamer rssm_deter — committed skill"),
    ("activation_datasets/btc_dreamer", "rssm_deter", "belief_ordinal_true", True, "Dreamer rssm_deter — belief axis (water−rock)"),
]


def main(n=4000):
    for i, (ds, src, col, cont, title) in enumerate(PANELS):
        b = ActivationBundle(ds)
        keys = [c for c in ["category", "final_commit"] if c in b.labels.columns]
        rng = np.random.default_rng(0)
        idx = b.labels.groupby(keys, observed=True).indices if keys else {"_": np.arange(len(b.labels))}
        take = []
        frac = n / len(b.labels)
        for v in idx.values():
            k = min(len(v), max(1, int(round(len(v) * frac))))
            take.append(rng.choice(v, k, replace=False))
        S = b.labels.iloc[np.sort(np.concatenate(take))].reset_index(drop=True)
        X = b.load_activations(src, S["row_id"])
        coords = G.pca_project(X, 3).coords[:, :3]
        fig = wandb_io.plotly_scatter(coords, S, col, title=title, continuous=cont)
        name = f"{Path(ds).name}__{src}__{col}.html"
        fig.write_html(str(OUT / name), include_plotlyjs="cdn", full_html=True)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
