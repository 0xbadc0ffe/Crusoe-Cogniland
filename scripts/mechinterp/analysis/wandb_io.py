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
def projector_table(bundle, df, X):
    """wandb.Table for the W&B embedding projector: raw activation dims D_i +
    metadata. W&B computes PCA/UMAP/t-SNE in-browser and lets you colour by any
    metadata column. (Hover frames go through `hover_html` instead — wandb 0.25
    does not serialise wandb.Image cells inside a Table.) Row-aligned with X."""
    import wandb
    D = X.shape[1]
    dim_cols = [f"D{i}" for i in range(D)]
    meta = _present(df)
    Xf = X.astype(np.float32)
    recs = df[meta].to_dict("records")
    data = [list(map(float, Xf[i])) + [_clean(recs[i][c]) for c in meta]
            for i in range(len(df))]
    return wandb.Table(columns=dim_cols + meta, data=data)


# ------------------------------------------------- hover-frame interactive HTML
def render_thumbs(bundle, row_ids, upscale: int = 3) -> list:
    """Base64 PNG thumbnails of each row's egocentric obs frame."""
    import base64, io
    from PIL import Image
    out = []
    for r in row_ids:
        arr = bundle.render_obs(int(r), upscale=upscale)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def hover_html(coords, df, thumbs, color_col, *, title=""):
    """Standalone interactive scatter (no deps): hovering a point shows the
    rendered agent frame + its metadata. Returns an HTML string for wandb.Html.
    `coords`, `df`, `thumbs` must be row-aligned."""
    import json as _json
    cmap = (style.CATEGORY_COLORS if color_col == "category" else style.SKILL_COLORS)
    lab = df[color_col].astype(str).to_numpy() if color_col in df else None
    xs, ys = coords[:, 0], coords[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    W, H, pad = 720, 560, 28
    px = ((xs - x0) / (x1 - x0 + 1e-9) * (W - 2 * pad) + pad)
    py = (H - ((ys - y0) / (y1 - y0 + 1e-9) * (H - 2 * pad) + pad))   # flip y
    caps, cols = [], []
    meta = _present(df)
    recs = df[meta].to_dict("records")
    for i in range(len(df)):
        cols.append(cmap.get(lab[i], "#888") if lab is not None else "#4c78a8")
        bits = [f"{k}={_fmt(recs[i][k])}" for k in
                ("row_id", "t", "category", "final_commit", "action_name", "value")
                if k in recs[i]]
        caps.append("  ".join(bits))
    pts = _json.dumps([{"x": float(px[i]), "y": float(py[i]), "c": cols[i]}
                       for i in range(len(df))])
    legend = "".join(
        f'<span style="color:{c};font-weight:700">&#9679;</span> {k} &nbsp;'
        for k, c in cmap.items()) if lab is not None else ""
    return _HTML_TMPL.replace("__W__", str(W)).replace("__H__", str(H)) \
        .replace("__TITLE__", title).replace("__LEGEND__", legend) \
        .replace("__PTS__", pts).replace("__THUMBS__", _json.dumps(thumbs)) \
        .replace("__CAPS__", _json.dumps(caps)).replace("__COLORBY__", color_col)


def _fmt(v):
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.3f}"
    return _clean(v)


_HTML_TMPL = """<!doctype html><meta charset="utf-8">
<div style="font-family:sans-serif;color:#2b3a4a;position:relative">
<div style="font-weight:700;margin:4px">__TITLE__ <span style="font-weight:400;color:#789">(hover a point — colour = __COLORBY__)</span></div>
<div style="margin:2px 6px">__LEGEND__</div>
<canvas id="cv" width="__W__" height="__H__" style="background:#eef3f8;border-radius:6px"></canvas>
<div id="tip" style="position:absolute;display:none;pointer-events:none;background:#fff;
 border:1px solid #c6d2de;border-radius:6px;padding:4px;box-shadow:0 2px 8px rgba(0,0,0,.15)">
 <img id="tipimg" style="image-rendering:pixelated;width:150px;display:block">
 <div id="tipcap" style="font-size:11px;margin-top:3px;max-width:150px"></div></div></div>
<script>
const PTS=__PTS__, THUMBS=__THUMBS__, CAPS=__CAPS__;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
const ti=document.getElementById('tipimg'),tc=document.getElementById('tipcap');
function draw(hl){ctx.clearRect(0,0,cv.width,cv.height);
 for(let i=0;i<PTS.length;i++){const p=PTS[i];ctx.beginPath();
  ctx.arc(p.x,p.y,i===hl?5:2.6,0,7);ctx.fillStyle=p.c;ctx.globalAlpha=i===hl?1:.6;ctx.fill();}
 ctx.globalAlpha=1;}
draw(-1);
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();
 const mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=64;
 for(let i=0;i<PTS.length;i++){const dx=PTS[i].x-mx,dy=PTS[i].y-my,d=dx*dx+dy*dy;
  if(d<bd){bd=d;best=i;}}
 if(best>=0){ti.src='data:image/png;base64,'+THUMBS[best];tc.textContent=CAPS[best];
  tip.style.display='block';tip.style.left=(mx+14)+'px';tip.style.top=(my+14)+'px';draw(best);}
 else{tip.style.display='none';draw(-1);}});
cv.addEventListener('mouseleave',()=>{tip.style.display='none';draw(-1);});
</script>"""


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
