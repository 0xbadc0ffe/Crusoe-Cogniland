#!/usr/bin/env python3
"""Interactive 3-D enc_embed PCA for all four bundles, with a hover panel that
shows the ACTUAL rendered 21x21 egocentric frame (agent drawn at centre, facing
its real direction) + the decoded observation scalars of the hovered timestep.

Each panel has a colour-mode <select>: recolouring the SAME plot (single trace,
per-point colours swapped via Plotly.restyle), with a matching legend:
  belief   — BTC only, rocky/balanced/lakes belief category
  segment  — obstacle segment (free/approach/avoid/bridge/tunnel), ALL models
  step%    — viridis gradient over step/max, saturating at 10% (the meaningful band)
  xy       — 2-D HSV colormap of the agent's (col,row) map position: centre = white,
             each border a distinct hue (right red, left cyan, top blue, bottom green)

Observation scalar vector decoded in the hover caption:
  bt  : [facing(4 one-hot), step/max]                            (5)
  btc : [facing(4 one-hot), step/max, commit_build, commit_mine] (7)

Output: outputs/report/enc_embed_pca_hover.html  (self-contained)
        outputs/report/figs/enc_embed_pca.png     (static 2x2 PC1xPC2 ref)
"""
from __future__ import annotations

import base64
import colorsys
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle
from mechinterp.analysis import style

SPECS = [("bt_ppo", "BT · PPO enc_embed"),
         ("btc_ppo", "BTC · PPO enc_embed"),
         ("bt_dreamer", "BT · Dreamer enc_embed"),
         ("btc_dreamer", "BTC · Dreamer enc_embed")]
SRC = "enc_embed"
FACES = ["up", "down", "left", "right"]
SEG_COLORS = {"free": "#9bb1c4", "approach": "#5b8def", "avoid": "#1f5fd0",
              "bridge": "#e6a800", "tunnel": "#a800e6"}
STEP_CAP = 0.10            # step/max value where the gradient saturates
UPSCALE = 7


# --------------------------------------------------------------- 2-D xy colormap
def hsv2d(u: float, v: float):
    """(u,v)∈[0,1]² → RGB uint8. Centre white; border hue encodes direction
    (right=red, bottom=green, left=cyan, top=blue/purple)."""
    dx, dy = u - 0.5, v - 0.5
    hue = (math.atan2(dy, dx) / (2 * math.pi)) % 1.0
    sat = min(1.0, 2.0 * math.hypot(dx, dy))
    r, g, b = colorsys.hsv_to_rgb(hue, sat, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def xy_colors(pos_c, pos_r, W, H):
    u = np.clip(pos_c / (W - 1), 0, 1)
    v = np.clip(pos_r / (H - 1), 0, 1)
    return [mcolors.to_hex([c / 255 for c in hsv2d(u[i], v[i])])
            for i in range(len(u))]


def _b64png(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def xy_legend_b64(N=120):
    img = np.zeros((N, N, 3), np.uint8)
    for i in range(N):           # row i → v (top=0)
        for j in range(N):       # col j → u
            img[i, j] = hsv2d(j / (N - 1), i / (N - 1))
    return _b64png(img)


# --------------------------------------------------------------------- rendering
def draw_agent(arr, facing, upscale, V):
    """Draw a facing-arrow at the centre cell of an upscaled egocentric frame."""
    im = Image.fromarray(arr); d = ImageDraw.Draw(im)
    half = V // 2
    cx = (half + 0.5) * upscale
    cy = (half + 0.5) * upscale
    h = 0.85 * upscale                       # arrow half-extent (~1.7 cells)
    if facing == 0:      # up
        tri = [(cx, cy - h), (cx - h, cy + h), (cx + h, cy + h)]
    elif facing == 1:    # down
        tri = [(cx, cy + h), (cx - h, cy - h), (cx + h, cy - h)]
    elif facing == 2:    # left
        tri = [(cx - h, cy), (cx + h, cy - h), (cx + h, cy + h)]
    else:                # right
        tri = [(cx + h, cy), (cx - h, cy - h), (cx - h, cy + h)]
    outline = [(x, y) for x, y in tri]
    d.polygon(outline, fill=(0, 0, 0))                       # black halo
    sh = 0.62 * upscale                                      # inner red arrow
    if facing == 0:
        tri2 = [(cx, cy - sh), (cx - sh, cy + sh), (cx + sh, cy + sh)]
    elif facing == 1:
        tri2 = [(cx, cy + sh), (cx - sh, cy - sh), (cx + sh, cy - sh)]
    elif facing == 2:
        tri2 = [(cx - sh, cy), (cx + sh, cy - sh), (cx + sh, cy + sh)]
    else:
        tri2 = [(cx + sh, cy), (cx - sh, cy - sh), (cx - sh, cy + sh)]
    d.polygon(tri2, fill=(255, 45, 45))
    return np.asarray(im)


def thumb(bundle, row_id, facing, upscale=UPSCALE):
    arr = bundle.render_obs(int(row_id), upscale=upscale)
    arr = draw_agent(arr, int(facing), upscale, bundle.view_size)
    return _b64png(arr)


# ----------------------------------------------------------------------- caption
SCALAR_NAMES_BT = ["face_up", "face_down", "face_left", "face_right", "step/max"]
SCALAR_NAMES_BTC = SCALAR_NAMES_BT + ["commit_build", "commit_mine"]


def decode_scalars(s, is_commit):
    face = FACES[int(np.argmax(s[:4]))]
    line = (f"facing=<b>{face}</b> &middot; step=<b>{float(s[4]) * 100:.1f}%</b> "
            f"of max")
    if is_commit and len(s) >= 7:
        cb, cm_ = int(round(s[5])), int(round(s[6]))
        commit = "build" if cb else ("mine" if cm_ else "none")
        line += f" &middot; commit=<b>{commit}</b> (b={cb} m={cm_})"
    names = SCALAR_NAMES_BTC if (is_commit and len(s) >= 7) else SCALAR_NAMES_BT
    labelled = " &middot; ".join(f"{n}={v:.3g}" for n, v in zip(names, s))
    return (line + f"<br><span style='color:#789;font-size:11px'>"
            f"obs.scalars[{len(s)}]: {labelled}</span>")


def caption(rec, s, is_commit):
    head = (f"<b>row {rec['row_id']}</b> &middot; map {rec['map_id']} "
            f"&middot; t={rec['t']} &middot; pos=({rec['pos_r']},{rec['pos_c']})")
    bits = []
    for k in ("category", "final_commit", "segment", "action_name"):
        if k in rec and rec[k] is not None and str(rec[k]) != "nan":
            bits.append(f"{k}={rec[k]}")
    for k in ("value", "ctg_to_goal"):
        if k in rec and rec[k] is not None:
            try:
                bits.append(f"{k}={float(rec[k]):.3g}")
            except (TypeError, ValueError):
                pass
    if "reached" in rec:
        bits.append(f"reached={int(bool(rec['reached']))}")
    meta = " &middot; ".join(bits)
    return (f"<div style='font-size:12px;line-height:1.45'>{head}<br>"
            f"<div style='margin:4px 0;padding:4px 6px;background:#eef3f8;"
            f"border-radius:4px'>{decode_scalars(s, is_commit)}</div>{meta}</div>")


# ------------------------------------------------------------------- legend html
def cat_legend(cmap, groups_present):
    items = "".join(
        f"<span style='display:inline-block;margin:2px 8px 2px 0'>"
        f"<span style='display:inline-block;width:11px;height:11px;background:{c};"
        f"border-radius:2px;vertical-align:middle'></span> {g}</span>"
        for g, c in cmap.items() if g in groups_present)
    return f"<div style='font-size:12px'>{items}</div>"


def step_legend(cap=0.10):
    stops = ", ".join(mcolors.to_hex(cm.viridis(x)) for x in np.linspace(0, 1, 8))
    pc = int(round(cap * 100))
    return (f"<div style='font-size:12px'>step / max (saturates at {pc}%)"
            f"<div style='height:12px;border-radius:3px;margin:3px 0;"
            f"background:linear-gradient(90deg,{stops})'></div>"
            "<div style='display:flex;justify-content:space-between'>"
            f"<span>0%</span><span>{pc // 2}%</span><span>&ge;{pc}%</span></div></div>")


def cont_legend(title, cmap, vmin, vmax):
    stops = ", ".join(mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, 8))
    return (f"<div style='font-size:12px'>{title}"
            f"<div style='height:12px;border-radius:3px;margin:3px 0;"
            f"background:linear-gradient(90deg,{stops})'></div>"
            "<div style='display:flex;justify-content:space-between'>"
            f"<span>{vmin:.2f}</span><span>{vmax:.2f}</span></div></div>")


def xy_legend_html(b64):
    return ("<div style='font-size:12px'>agent map position"
            "<div style='position:relative;width:110px;height:110px;margin:3px 0'>"
            f"<img src='data:image/png;base64,{b64}' style='width:110px;height:110px;"
            "image-rendering:pixelated;border:1px solid #c6d2de;border-radius:3px'>"
            "<span style='position:absolute;top:-2px;left:38px;font-size:10px'>top</span>"
            "<span style='position:absolute;bottom:-2px;left:30px;font-size:10px'>bottom</span>"
            "<span style='position:absolute;top:48px;left:-2px;font-size:10px'>left</span>"
            "<span style='position:absolute;top:48px;right:-4px;font-size:10px'>right</span>"
            "</div><span style='color:#789'>white = centre of map</span></div>")


# --------------------------------------------------------------------- per panel
def panel(name, n):
    b = ActivationBundle(f"activation_datasets/{name}")
    H, W = b.maps["terrain"].shape[1], b.maps["terrain"].shape[2]
    ids = np.sort(b.labels["row_id"].sample(min(n, len(b.labels)),
                                            random_state=0).to_numpy())
    lab = b.labels.set_index("row_id").loc[ids].reset_index()
    X = b.load_activations(SRC, ids)
    pca = PCA(3, svd_solver="randomized", random_state=0).fit(X)
    coords = pca.transform(X)
    evr = pca.explained_variance_ratio_ * 100
    scal = b.load_extra("scalars", ids)
    recs = lab.to_dict("records")

    thumbs = [thumb(b, ids[i], np.argmax(scal[i][:4])) for i in range(len(ids))]
    caps = [caption(recs[i], scal[i], b.is_commit) for i in range(len(ids))]

    seg = lab["segment"].astype(str).to_numpy()
    colors, legends, modes = {}, {}, []
    if b.has_belief:
        cat = lab["category"].astype(str).to_numpy()
        colors["belief"] = [style.CATEGORY_COLORS.get(c, "#888") for c in cat]
        legends["belief"] = cat_legend(style.CATEGORY_COLORS, set(cat))
        modes.append("belief")
    colors["segment"] = [SEG_COLORS.get(s, "#888") for s in seg]
    legends["segment"] = cat_legend(SEG_COLORS, set(seg))
    modes.append("segment")
    cap = 0.20 if "ppo" in name else 0.50      # PPO ≥20% · Dreamer ≥50%
    step_norm = np.clip(scal[:, 4] / cap, 0, 1)
    colors["step%"] = [mcolors.to_hex(cm.viridis(x)) for x in step_norm]
    legends["step%"] = step_legend(cap)
    modes.append("step%")
    val = lab["value"].to_numpy().astype(np.float32)
    vmin, vmax = float(np.percentile(val, 1)), float(np.percentile(val, 99))
    vnorm = np.clip((val - vmin) / (vmax - vmin + 1e-9), 0, 1)
    colors["value"] = [mcolors.to_hex(cm.magma(x)) for x in vnorm]
    legends["value"] = cont_legend("V(h_t)", cm.magma, vmin, vmax)
    modes.append("value")
    colors["xy"] = xy_colors(lab["pos_c"].to_numpy(), lab["pos_r"].to_numpy(), W, H)
    legends["xy"] = xy_legend_html(xy_legend_b64())
    modes.append("xy")

    default = "belief" if b.has_belief else "segment"
    return dict(name=name, coords=coords, evr=evr, thumbs=thumbs, caps=caps,
                colors=colors, legends=legends, modes=modes, default=default)


def html_panel(p, label, first):
    import plotly.graph_objects as go
    key = p["name"]
    c = p["coords"]
    fig = go.Figure(go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
        marker=dict(size=2.6, color=p["colors"][p["default"]], opacity=0.78,
                    line=dict(width=0)),
        customdata=np.arange(len(c)),
        hovertemplate="%{customdata}<extra></extra>"))
    evr = p["evr"]
    pane = dict(backgroundcolor=style.PANEL, gridcolor="white", showticklabels=False)
    fig.update_layout(
        title=f"{label}  (PC1 {evr[0]:.1f}% · PC2 {evr[1]:.1f}% · PC3 {evr[2]:.1f}%)",
        width=760, height=600, paper_bgcolor="white", showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(xaxis=dict(title="PC1", **pane), yaxis=dict(title="PC2", **pane),
                   zaxis=dict(title="PC3", **pane)))
    div = fig.to_html(full_html=False,
                      include_plotlyjs=("cdn" if first else False),
                      div_id=f"plot_{key}")
    opts = "".join(f"<option value='{m}'>{m}</option>" for m in p["modes"])
    return f"""
<section style="margin:34px 0;border-top:1px solid #e3e9ef;padding-top:16px">
<h2 style="color:#1b4f72;margin-bottom:4px">{label}</h2>
<div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
  <div>{div}</div>
  <div style="width:230px">
    <label style="font-size:12px;font-weight:600">colour by
      <select id="sel_{key}" style="margin-left:6px;font-size:12px">{opts}</select>
    </label>
    <div id="leg_{key}" style="margin:8px 0 12px"></div>
    <div style="font-size:12px;color:#789;margin-bottom:4px">hover a dot &rarr;</div>
    <img id="img_{key}" style="width:210px;height:210px;image-rendering:pixelated;
         border:1px solid #c6d2de;border-radius:6px;background:#eef3f8" alt="">
    <div id="cap_{key}" style="margin-top:8px"></div>
  </div>
</div>
<script>
(function(){{
  var TH={json.dumps(p['thumbs'])}, CP={json.dumps(p['caps'])};
  var COL={json.dumps(p['colors'])}, LEG={json.dumps(p['legends'])};
  var gd=document.getElementById('plot_{key}'),
      sel=document.getElementById('sel_{key}');
  function apply(m){{
    Plotly.restyle(gd,{{'marker.color':[COL[m]]}},[0]);
    document.getElementById('leg_{key}').innerHTML=LEG[m];
  }}
  function ready(){{
    if(!gd || !gd.on){{return setTimeout(ready,120);}}
    gd.on('plotly_hover',function(d){{
      var i=d.points[0].customdata;
      document.getElementById('img_{key}').src='data:image/png;base64,'+TH[i];
      document.getElementById('cap_{key}').innerHTML=CP[i];
    }});
    sel.addEventListener('change',function(e){{apply(e.target.value);}});
    sel.value='{p['default']}'; apply('{p['default']}');
  }}
  ready();
}})();
</script>
</section>"""


def static_png(panels):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, p in zip(axes.ravel(), panels):
        ax.scatter(p["coords"][:, 0], p["coords"][:, 1], s=4,
                   c=p["colors"][p["default"]], alpha=0.6, linewidths=0)
        ax.set_title(f"{p['name']} (colour={p['default']})", fontweight="bold")
        ax.set_xlabel(f"PC1 ({p['evr'][0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({p['evr'][1]:.1f}%)")
        ax.set_facecolor(style.PANEL); ax.grid(True, color="white")
    fig.suptitle("enc_embed PCA (PC1×PC2) — default colour (interactive html has "
                 "step%/segment/xy selector)", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("outputs/report/figs/enc_embed_pca.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3500
    panels = []
    for name, label in SPECS:
        print(f"[{name}] PCA + thumbs (n={n}) ...", flush=True)
        panels.append(panel(name, n))
    parts = ["<!doctype html><meta charset='utf-8'>"
             "<title>enc_embed PCA — hover frame + scalars</title>"
             "<style>body{font-family:sans-serif;color:#223;max-width:1100px;"
             "margin:0 auto;padding:24px}h1{color:#1b4f72}</style>"
             "<h1>enc_embed PCA — hover a dot for its 21×21 frame + observation scalars</h1>"
             "<p>3-D rotatable PCA of the encoder embedding per agent. Use the "
             "<b>colour by</b> selector on each panel: <b>belief</b> (BTC), "
             "<b>segment</b>, <b>step%</b> (viridis, saturating at 10% of max steps), "
             "or <b>xy</b> (2-D map-position colormap). Hover any point to render its "
             "egocentric minimap — the red arrow is the agent at centre, pointing its "
             "real facing — and to decode the observation scalar vector.</p>"]
    for i, ((name, label), p) in enumerate(zip(SPECS, panels)):
        parts.append(html_panel(p, label, first=(i == 0)))
    out = Path("outputs/report/enc_embed_pca_hover.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")
    static_png(panels)


if __name__ == "__main__":
    main()
