"""Configuration for the activation-geometry analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnalysisConfig:
    # --- io ---
    dataset: Path                          # path to the bundle dir
    out_dir: Path = Path("outputs/analysis")
    sources: list | None = None            # None = all discovered (gru_h, enc_embed, ...)

    # --- subsampling (rows are stratified by available labels) ---
    analysis_rows: int = 30_000            # master df: PCA/UMAP + probe preds + plots + tables
    probe_rows: int = 120_000              # rows used to TRAIN probes / compute direction means
    projector_rows: int = 2_500            # W&B embedding-projector table (with hover images)
    traj_examples: int = 8                 # episodes drawn as PCA trajectory paths
    seed: int = 0

    # --- dimensionality reduction ---
    pca_components: int = 10               # kept for tables; first `scatter_dims` plotted
    scatter_dims: int = 3                  # 3 = 3-D scatter/centroid/trajectory plots; 2 = flat
    do_umap: bool = True
    do_tsne: bool = False                  # slower; off by default
    umap_neighbors: int = 30
    umap_min_dist: float = 0.1

    # --- probes ---
    probe_label_skill: str = "final_commit"   # 'final_commit' | 'commit_state'
    probe_C: float = 1.0
    probe_max_iter: int = 2000
    group_test_frac: float = 0.25          # held-out *maps* for probe evaluation

    # --- projector ---
    projector_images: bool = True          # render hover frames into the projector table

    # --- wandb ---
    wandb_project: str = "bridge_tunnel_geometry"
    wandb_entity: str | None = None
    wandb_mode: str = "online"             # online | offline | disabled
    run_name: str | None = None
    tags: list = field(default_factory=list)

    def __post_init__(self):
        self.dataset = Path(self.dataset)
        self.out_dir = Path(self.out_dir)
