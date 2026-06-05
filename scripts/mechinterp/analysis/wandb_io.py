"""W&B logging helpers: interactive plotly scatters, embedding-projector tables
(raw activation dims + metadata + optional hover obs image), and metadata tables.

The embedding projector reads a wandb.Table whose columns are the raw activation
dimensions plus metadata; W&B then computes PCA/UMAP/t-SNE in the browser and lets
you colour by any metadata column. Adding a wandb.Image column makes the rendered
egocentric frame show up on hover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import style

# metadata columns surfaced in projector / scatter tooltips (only those present used)
_META = [
    "row_id", "map_id", "traj_id", "t", "category", "belief_ordinal_true",
    "commit_state", "final_commit", "correct_commit", "segment", "action_name",
    "value", "reached", "belief_pred", "belief_conf", "skill_pred", "skill_conf",
    "belief_p_lakes", "belief_p_rocky", "belief_p_balanced", "belief_ordinal_pred",
    "policy_entropy",
]


def _present(df):
    return [c for c in _META if c in df.columns]


# ------------------------------------------------------------- plotly scatter
def plotly_scatter(coords, df, color_col, *, title="", continuous=False):
    """Interactive 2-D scatter; hover shows all present metadata columns."""
    import plotly.express as px
    meta = _present(df)
    d = df.copy()
    d["_x"] = coords[:, 0]
    d["_y"] = coords[:, 1]
    kw = dict(x="_x", y="_y", hover_data=meta, title=title,
              render_mode="webgl", opacity=0.65)
    if continuous:
        fig = px.scatter(d, color=color_col, color_continuous_scale="RdBu_r",
                         range_color=[-1, 1], **kw)
    else:
        cmap = style.CATEGORY_COLORS if color_col == "category" else style.SKILL_COLORS
        order = style.CATEGORY_ORDER if color_col == "category" else style.SKILL_ORDER
        d[color_col] = d[color_col].astype(str)
        fig = px.scatter(d, color=color_col, color_discrete_map=cmap,
                         category_orders={color_col: order}, **kw)
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    fig.update_layout(plot_bgcolor=style.PANEL, paper_bgcolor="white",
                      xaxis_title="dim 1", yaxis_title="dim 2",
                      legend_title=color_col, width=760, height=620)
    fig.update_xaxes(showgrid=True, gridcolor="white", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="white", zeroline=False)
    return fig


# ------------------------------------------------------- embedding projector
def projector_table(bundle, df, X, *, with_images=True):
    """wandb.Table for the W&B embedding projector: raw activation dims D_i +
    metadata (+ obs image on hover). `df` and `X` must be row-aligned."""
    import wandb
    D = X.shape[1]
    dim_cols = [f"D{i}" for i in range(D)]
    meta = _present(df)
    columns = dim_cols + meta + (["obs"] if with_images else [])
    data = []
    Xf = X.astype(np.float32)
    recs = df[meta].to_dict("records")
    rids = df["row_id"].to_numpy()
    for i in range(len(df)):
        row = list(map(float, Xf[i])) + [_clean(recs[i][c]) for c in meta]
        if with_images:
            row.append(wandb.Image(bundle.render_obs(int(rids[i]))))
        data.append(row)
    return wandb.Table(columns=columns, data=data)


def metadata_table(df, coord_cols):
    """wandb.Table of timestep metadata + DR coords + probe predictions
    (no images; larger subsample, good for filtering/sorting in the UI)."""
    import wandb
    cols = _present(df) + [c for c in coord_cols if c in df.columns]
    sub = df[cols].copy()
    for c in sub.columns:
        if sub[c].dtype == bool:
            sub[c] = sub[c].astype(int)
    return wandb.Table(dataframe=sub.reset_index(drop=True))


def _clean(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v if isinstance(v, (int, float, str)) else str(v)


def cosine_table(M, rows, cols):
    import wandb
    data = [[r] + [float(M[i, j]) for j in range(len(cols))] for i, r in enumerate(rows)]
    return wandb.Table(columns=["direction"] + list(cols), data=data)
