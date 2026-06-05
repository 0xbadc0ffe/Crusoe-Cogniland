#!/usr/bin/env python3
"""CLI for the activation-geometry analysis pipeline.

    # full BTC run (belief + skill + entanglement) -> W&B
    python -m scripts.mechinterp.analysis.run_analysis \
        --dataset activation_datasets/btc_ppo --wandb-mode online

    # quick local smoke test (no W&B, tiny subsample, no images/umap)
    python -m scripts.mechinterp.analysis.run_analysis \
        --dataset activation_datasets/btc_ppo --smoke

    # BT bundle (no belief/skill — runs PCA/UMAP + tables only, same code)
    python -m scripts.mechinterp.analysis.run_analysis \
        --dataset activation_datasets/bt_ppo --wandb-mode online

Run with `--help` for every knob. Defaults are sensible for the 30M-row BTC set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import AnalysisConfig
from .pipeline import run


def build_config(a) -> AnalysisConfig:
    cfg = AnalysisConfig(
        dataset=Path(a.dataset), out_dir=Path(a.out_dir),
        sources=a.sources, analysis_rows=a.analysis_rows, probe_rows=a.probe_rows,
        projector_rows=a.projector_rows, traj_examples=a.traj_examples, seed=a.seed,
        pca_components=a.pca_components, do_umap=not a.no_umap, do_tsne=a.tsne,
        umap_neighbors=a.umap_neighbors, umap_min_dist=a.umap_min_dist,
        probe_label_skill=a.skill_label, probe_C=a.probe_c,
        probe_max_iter=a.probe_max_iter, group_test_frac=a.test_frac,
        projector_images=not a.no_projector_images,
        wandb_project=a.wandb_project, wandb_entity=a.wandb_entity,
        wandb_mode=a.wandb_mode, run_name=a.run_name, tags=a.tags or [],
    )
    if a.smoke:
        cfg.analysis_rows = 4000
        cfg.probe_rows = 8000
        cfg.projector_rows = 200
        cfg.do_umap = False
        cfg.projector_images = False
        cfg.wandb_mode = "disabled"
        cfg.run_name = (cfg.run_name or "smoke") + "-smoke"
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="bundle dir (e.g. activation_datasets/btc_ppo)")
    p.add_argument("--out-dir", default="outputs/analysis")
    p.add_argument("--sources", nargs="*", default=None,
                   help="activation sources (default: all discovered, e.g. gru_h enc_embed)")
    # subsampling
    p.add_argument("--analysis-rows", type=int, default=30000)
    p.add_argument("--probe-rows", type=int, default=120000)
    p.add_argument("--projector-rows", type=int, default=2500)
    p.add_argument("--traj-examples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # DR
    p.add_argument("--pca-components", type=int, default=10)
    p.add_argument("--no-umap", action="store_true")
    p.add_argument("--tsne", action="store_true", help="also compute t-SNE (slow)")
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    # probes
    p.add_argument("--skill-label", default="final_commit",
                   choices=["final_commit", "commit_state"])
    p.add_argument("--probe-c", type=float, default=1.0)
    p.add_argument("--probe-max-iter", type=int, default=2000)
    p.add_argument("--test-frac", type=float, default=0.25, help="held-out map fraction")
    # projector
    p.add_argument("--no-projector-images", action="store_true",
                   help="skip rendering hover frames into the projector table")
    # wandb
    p.add_argument("--wandb-project", default="bridge_tunnel_geometry")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--run-name", default=None)
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny, fast, wandb-disabled sanity run")
    a = p.parse_args()
    run(build_config(a))


if __name__ == "__main__":
    main()
