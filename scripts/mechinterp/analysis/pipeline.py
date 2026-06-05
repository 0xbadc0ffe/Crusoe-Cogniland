"""Orchestrator: load a bundle -> subsample -> per-source PCA/UMAP + probes +
direction geometry -> figures + tables -> one well-organised W&B run.

Reused unchanged for BT and BTC: belief steps run only when the bundle has a
`category` label, skill steps only when it has a commit label. The run is keyed
by source (e.g. ``gru_h/...``, ``enc_embed/...``) so the W&B workspace groups
cleanly.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as G
from . import plots, style, wandb_io
from . import probes as P
from .bundle import ActivationBundle
from .config import AnalysisConfig


# --------------------------------------------------------------- subsampling
def stratified(df: pd.DataFrame, keys: list, n: int, seed: int) -> pd.DataFrame:
    """Stratified row subsample preserving all columns (samples positional
    indices per group so it is robust to pandas' apply-drops-group-cols)."""
    if n >= len(df):
        return df.copy()
    rng = np.random.default_rng(seed)
    if not keys:
        pos = rng.choice(len(df), size=n, replace=False)
        return df.iloc[np.sort(pos)].reset_index(drop=True)
    frac = n / len(df)
    take = []
    for g_idx in df.groupby(keys, observed=True).indices.values():
        k = min(len(g_idx), max(1, int(round(len(g_idx) * frac))))
        take.append(rng.choice(g_idx, size=k, replace=False))
    pos = np.concatenate(take)
    if len(pos) > n:
        pos = rng.choice(pos, size=n, replace=False)
    return df.iloc[np.sort(pos)].reset_index(drop=True)


def pick_episodes(df: pd.DataFrame, n: int, seed: int, by="category") -> list:
    """Return up to n (map_id, traj_id) keys, spread across `by` if present."""
    rng = np.random.default_rng(seed)
    trajs = df[["map_id", "traj_id"] + ([by] if by in df else [])].drop_duplicates()
    if by in trajs:
        groups = [g for _, g in trajs.groupby(by)]
        per = max(1, n // len(groups))
        chosen = pd.concat([g.sample(min(per, len(g)), random_state=seed) for g in groups])
    else:
        chosen = trajs.sample(min(n, len(trajs)), random_state=seed)
    return list(map(tuple, chosen[["map_id", "traj_id"]].to_numpy()[:n]))


# --------------------------------------------------------------- entropy
def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-9, 1)
    return -(p * np.log(p)).sum(1)


# --------------------------------------------------------------- main
def run(cfg: AnalysisConfig):
    import wandb

    style.apply_theme()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    b = ActivationBundle(cfg.dataset)
    print(b.summary())
    sources = cfg.sources or b.sources
    primary = sources[0]
    print(f"sources={sources}  primary={primary}")

    lab = b.labels
    skill_col = cfg.probe_label_skill if b.has_skill else None
    strat = [c for c in ["category", skill_col] if c and c in lab.columns]

    analysis_df = stratified(lab, strat, cfg.analysis_rows, cfg.seed).copy()
    probe_df = stratified(lab, strat, cfg.probe_rows, cfg.seed + 1)
    proj_df = stratified(analysis_df, strat, cfg.projector_rows, cfg.seed + 2)
    ep_keys = pick_episodes(analysis_df, cfg.traj_examples, cfg.seed + 3)

    # policy entropy on the analysis subsample (cheap colour channel)
    ap = b.load_extra("action_probs", analysis_df["row_id"])
    analysis_df["policy_entropy"] = _entropy(ap)

    run = wandb.init(
        project=cfg.wandb_project, entity=cfg.wandb_entity, mode=cfg.wandb_mode,
        name=cfg.run_name or f"geometry-{b.name}",
        tags=list(cfg.tags) + [b.name, b.manifest.get("env", ""), "geometry"],
        config={**{k: str(v) for k, v in asdict(cfg).items()},
                "n_rows_total": len(lab), "sources": sources,
                "has_belief": b.has_belief, "has_skill": b.has_skill},
    )

    summary, coord_cols = {}, []
    for src in sources:
        print(f"\n=== source: {src} ({b.source_dim(src)}d) ===")
        Xa = b.load_activations(src, analysis_df["row_id"])
        Xp = b.load_activations(src, probe_df["row_id"])
        is_primary = src == primary
        _run_source(cfg, b, run, src, is_primary, Xa, Xp, analysis_df, probe_df,
                    proj_df, ep_keys, skill_col, summary, coord_cols)

    # ---- master metadata table (coords + canonical probe preds) ----
    run.log({"tables/timestep_metadata": wandb_io.metadata_table(analysis_df, coord_cols)})

    # ---- cross-source probe summary bars ----
    bars = {k: v for k, v in summary.items()
            if k.endswith(("belief_acc", "skill_acc")) and v is not None}
    if bars:
        _log_fig(run, cfg, "summary/probe_accuracy",
                 plots.probe_bars(bars, title="probe accuracy (held-out maps)"))

    run.summary.update({k: v for k, v in summary.items() if v is not None})
    (cfg.out_dir / f"{b.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nsummary:\n" + json.dumps(summary, indent=2))
    run.finish()
    return summary


def _run_source(cfg, b, run, src, is_primary, Xa, Xp, adf, pdf, projdf, ep_keys,
                skill_col, summary, coord_cols):
    import wandb
    pre = src  # wandb key namespace

    # ---------- dimensionality reduction ----------
    pca = G.pca_project(Xa, cfg.pca_components, cfg.seed)
    dims = cfg.scatter_dims
    for j in range(min(dims, pca.coords.shape[1])):
        adf[f"{src}_pc{j+1}"] = pca.coords[:, j]
        coord_cols.append(f"{src}_pc{j+1}")
    summary[f"{pre}/pca_var_pc1"] = float(pca.explained[0])
    summary[f"{pre}/pca_var_pc2"] = float(pca.explained[1])

    umap_coords = None
    if cfg.do_umap:
        try:
            umap_coords = G.umap_project(Xa, cfg.umap_neighbors, cfg.umap_min_dist,
                                         cfg.seed, n_components=dims).coords
            for j in range(umap_coords.shape[1]):
                adf[f"{src}_umap{j+1}"] = umap_coords[:, j]
                coord_cols.append(f"{src}_umap{j+1}")
        except Exception as e:
            print(f"  [umap skipped: {e}]")
    tsne_coords = G.tsne_project(Xa, cfg.seed, n_components=dims).coords if cfg.do_tsne else None

    # ---------- probes ----------
    belief_dirs, skill_dirs = {}, {}
    groups_p = pdf["map_id"].to_numpy()

    if b.has_belief:
        bp = P.fit_categorical(Xp, pdf["category"].to_numpy(), groups_p,
                               classes=style.CATEGORY_ORDER, C=cfg.probe_C,
                               max_iter=cfg.probe_max_iter,
                               test_frac=cfg.group_test_frac, seed=cfg.seed)
        op = P.fit_ordinal(Xp, pdf["belief_ordinal_true"].to_numpy(), groups_p,
                           test_frac=cfg.group_test_frac, seed=cfg.seed)
        summary[f"{pre}/belief_acc"] = bp.accuracy
        summary[f"{pre}/belief_balanced_acc"] = bp.balanced_accuracy
        summary[f"{pre}/belief_ordinal_r2"] = op.r2
        summary[f"{pre}/belief_ordinal_spearman"] = op.spearman
        _log_fig(run, cfg, f"{pre}/confusion_belief",
                 plots.confusion(bp.confusion, bp.classes, title=f"{src}: belief probe"))
        # difference-of-means belief directions (raw activation space)
        cat = pdf["category"].to_numpy()
        belief_dirs = {
            "lakes−rocky": G.diff_of_means(Xp, cat, "lakes", "rocky"),
            "lakes−balanced": G.diff_of_means(Xp, cat, "lakes", "balanced"),
            "rocky−balanced": G.diff_of_means(Xp, cat, "rocky", "balanced"),
            "ordinal(probe)": op.weight,
        }
        if is_primary:
            for k, v in P.proba_columns(bp, Xa, "belief").items():
                adf[k] = v
            adf["belief_ordinal_pred"] = op.predict(Xa)

    if b.has_skill:
        sp = P.fit_categorical(Xp, pdf[skill_col].to_numpy(), groups_p,
                               classes=style.SKILL_ORDER, C=cfg.probe_C,
                               max_iter=cfg.probe_max_iter,
                               test_frac=cfg.group_test_frac, seed=cfg.seed)
        summary[f"{pre}/skill_acc"] = sp.accuracy
        summary[f"{pre}/skill_balanced_acc"] = sp.balanced_accuracy
        _log_fig(run, cfg, f"{pre}/confusion_skill",
                 plots.confusion(sp.confusion, sp.classes, title=f"{src}: skill probe"))
        sk = pdf[skill_col].to_numpy()
        skill_dirs = {
            "build−mine": G.diff_of_means(Xp, sk, "build", "mine"),
            "build−none": G.diff_of_means(Xp, sk, "build", "none"),
            "mine−none": G.diff_of_means(Xp, sk, "mine", "none"),
            "committed−none": G.diff_of_means(Xp, sk, ["build", "mine"], "none"),
        }
        if is_primary:
            for k, v in P.proba_columns(sp, Xa, "skill").items():
                adf[k] = v

    # canonical label columns for plots (skill column aliased to 'skill')
    if b.has_skill and skill_col != "skill":
        adf["skill"] = adf[skill_col]

    # ---------- scatter plots (3-D by default, see cfg.scatter_dims) ----------
    for proj_name, coords in [("pca", pca.coords), ("umap", umap_coords),
                              ("tsne", tsne_coords)]:
        if coords is None:
            continue
        cc = coords[:, :dims]
        if b.has_belief:
            _log_fig(run, cfg, f"{pre}/{proj_name}_by_category",
                     plots.categorical_scatter(cc, adf["category"], "category", dims=dims,
                                               title=f"{src} {proj_name}: belief"))
        if b.has_skill:
            _log_fig(run, cfg, f"{pre}/{proj_name}_by_skill",
                     plots.categorical_scatter(cc, adf["skill"], "skill", dims=dims,
                                               centroid_path=False,
                                               title=f"{src} {proj_name}: skill"))
        if is_primary and "belief_ordinal_pred" in adf:
            _log_fig(run, cfg, f"{pre}/{proj_name}_by_belief_score",
                     plots.continuous_scatter(cc, adf["belief_ordinal_pred"], dims=dims,
                                              title=f"{src} {proj_name}: decoded belief",
                                              label="P(lakes)−P(rocky)"))

    # ---------- centroids ----------
    if b.has_belief:
        _log_fig(run, cfg, f"{pre}/centroids_category",
                 plots.centroid_plot(pca.coords[:, :dims], adf["category"], "category",
                                     dims=dims, title=f"{src}: belief centroids"))
    if b.has_skill:
        _log_fig(run, cfg, f"{pre}/centroids_skill",
                 plots.centroid_plot(pca.coords[:, :dims], adf["skill"], "skill",
                                     dims=dims, title=f"{src}: skill centroids"))

    # ---------- trajectory paths ----------
    traj_df, traj_coords = _episode_coords(b, src, ep_keys, pca.model)
    if traj_df is not None:
        _log_fig(run, cfg, f"{pre}/pca_trajectories",
                 plots.trajectory_paths(traj_coords[:, :dims], traj_df, dims=dims,
                                        color_kind="skill" if b.has_skill else None,
                                        title=f"{src}: PCA episode trajectories"))

    # ---------- direction geometry / entanglement ----------
    if belief_dirs and skill_dirs:
        M, rows, cols = G.cosine_matrix(belief_dirs, skill_dirs)
        _log_fig(run, cfg, f"{pre}/cosine_belief_vs_skill",
                 plots.cosine_heatmap(M, rows, cols,
                                      title=f"{src}: cos(belief dir, skill dir)"))
        run.log({f"{pre}/tables/cosine_belief_skill": wandb_io.cosine_table(M, rows, cols)})
        bd = list(belief_dirs.values())
        for nm, sd in skill_dirs.items():
            summary[f"{pre}/proj_frac[{nm}->belief]"] = G.projection_fraction(sd, bd)
        ang = G.principal_angles(bd, list(skill_dirs.values()))
        if ang.size:
            summary[f"{pre}/min_principal_angle_deg"] = float(ang.min())
            summary[f"{pre}/mean_principal_angle_deg"] = float(ang.mean())
        summary[f"{pre}/cos_belief_skill_main"] = G.cosine(
            belief_dirs["lakes−rocky"], skill_dirs["build−mine"])
        _log_fig(run, cfg, f"{pre}/entanglement_plane",
                 plots.entanglement_plane(Xa, belief_dirs["lakes−rocky"],
                                          skill_dirs["build−mine"], adf, source=src))

    # ---------- interactive plotly (primary source; 3-D rotatable) ----------
    if is_primary:
        pcc = pca.coords[:, :dims]
        if b.has_belief:
            run.log({f"{pre}/interactive_pca_category":
                     wandb.Plotly(wandb_io.plotly_scatter(pcc, adf,
                                  "category", title=f"{src} PCA — true map type"))})
            if "belief_ordinal_pred" in adf:
                run.log({f"{pre}/interactive_pca_belief_score":
                         wandb.Plotly(wandb_io.plotly_scatter(pcc, adf,
                                      "belief_ordinal_pred", continuous=True,
                                      title=f"{src} PCA — decoded belief score"))})
        if b.has_skill:
            run.log({f"{pre}/interactive_pca_skill":
                     wandb.Plotly(wandb_io.plotly_scatter(pcc, adf,
                                  "skill", title=f"{src} PCA — committed skill"))})

        # ---------- embedding projector table (raw dims + metadata) ----------
        Xproj = b.load_activations(src, projdf["row_id"])
        pj = projdf.copy()
        # bring canonical predictions onto the projector subset
        for col in ["belief_pred", "belief_conf", "belief_ordinal_pred", "skill_pred",
                    "skill_conf", "belief_p_lakes", "belief_p_rocky", "belief_p_balanced"]:
            if col in adf:
                pj = pj.merge(adf[["row_id", col]], on="row_id", how="left")
        run.log({f"{pre}/embedding_projector": wandb_io.projector_table(b, pj, Xproj)})

        # ---------- hover-frame interactive scatter (rendered obs on hover) ----------
        if cfg.projector_images:
            hov = pj.iloc[:min(len(pj), 1200)].reset_index(drop=True)
            hcoords = pca.model.transform(
                b.load_activations(src, hov["row_id"]))[:, :2]
            thumbs = wandb_io.render_thumbs(b, hov["row_id"])
            specs = ([("category", "category")] if b.has_belief else []) + \
                    ([("skill", skill_col)] if b.has_skill else [])
            for ck, col in specs:
                if col in hov:
                    run.log({f"{pre}/hover_frames_{ck}": wandb.Html(
                        wandb_io.hover_html(hcoords, hov, thumbs, col,
                                            title=f"{src} PCA — hover shows agent frame"))})


def _episode_coords(b, src, ep_keys, pca_model):
    """Load full episodes, project with the fitted PCA, return (df, coords)."""
    if pca_model is None or not ep_keys:
        return None, None
    rows = []
    for (mid, tid) in ep_keys:
        sub = b.labels[(b.labels.map_id == mid) & (b.labels.traj_id == tid)]
        rows.append(sub)
    df = pd.concat(rows).copy()
    df["_traj_key"] = df["map_id"].astype(str) + ":" + df["traj_id"].astype(str)
    if "commit_state" in df and "skill" not in df:
        df["skill"] = df["commit_state"]
    X = b.load_activations(src, df["row_id"])
    return df, pca_model.transform(X)


def _log_fig(run, cfg, key, fig):
    import wandb
    import matplotlib.pyplot as plt
    path = cfg.out_dir / (key.replace("/", "__") + ".png")
    fig.savefig(path, bbox_inches="tight")
    run.log({key: wandb.Image(str(path))})
    plt.close(fig)
