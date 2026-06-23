#!/usr/bin/env python3
"""Interactive 3-D PCA of the RECURRENT / LATENT state of each agent, same hover
machinery as enc_embed_pca_hover (rendered 21x21 frame with the agent arrow at
centre + decoded, labelled observation scalars), with a colour-mode selector.

Panels (one per agent × hidden-state source):
  PPO       h_t           = gru_h        (128)
  Dreamer   rssm_deter    = deterministic GRU state (3072)
  Dreamer   rssm_stoch    = stochastic latent logits (576)

Colour-by modes (per panel, swapped live via Plotly.restyle):
  belief        BTC only — rocky/balanced/lakes belief category
  commit        BTC only — final commit type none/build/mine
  next_action   action taken at t+1 (the decision the state leads to)
  segment       obstacle segment · step%  viridis(step/max, sat 10%) ·
  value         V(h_t) magma · xy  2-D map-position colormap

Output: outputs/report/hidden_state_pca_hover.html
        outputs/report/figs/hidden_state_pca.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechinterp.analysis.bundle import ActivationBundle
from mechinterp.analysis import style
from enc_embed_pca_hover import (
    thumb, caption, cat_legend, step_legend, cont_legend, xy_legend_html,
    xy_legend_b64, xy_colors, html_panel, SEG_COLORS, STEP_CAP)

SPECS = [
    ("bt_ppo", "gru_h", "BT · PPO h_t (gru_h)"),
    ("btc_ppo", "gru_h", "BTC · PPO h_t (gru_h)"),
    ("bt_dreamer", "rssm_deter", "BT · Dreamer rssm_deter"),
    ("bt_dreamer", "rssm_stoch_logits", "BT · Dreamer rssm_stoch"),
    ("btc_dreamer", "rssm_deter", "BTC · Dreamer rssm_deter"),
    ("btc_dreamer", "rssm_stoch_logits", "BTC · Dreamer rssm_stoch"),
]
ACTION_HEX = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#ffd000", "#d62728"]


def next_action_map(b):
    """row_id -> action at the next timestep (NaN at episode end)."""
    lf = b.labels.sort_values(["map_id", "traj_id", "t"])
    na = lf.groupby(["map_id", "traj_id"])["action"].shift(-1).to_numpy()
    return dict(zip(lf["row_id"].to_numpy(), na))


def panel(name, src, n):
    b = ActivationBundle(f"activation_datasets/{name}")
    H, W = b.maps["terrain"].shape[1], b.maps["terrain"].shape[2]
    anames = dict(b.labels[["action", "action_name"]].drop_duplicates().values)
    namap = next_action_map(b)

    ids = np.sort(b.labels["row_id"].sample(min(n, len(b.labels)),
                                            random_state=0).to_numpy())
    lab = b.labels.set_index("row_id").loc[ids].reset_index()
    X = b.load_activations(src, ids)
    pca = PCA(3, svd_solver="randomized", random_state=0).fit(X)
    coords = pca.transform(X)
    evr = pca.explained_variance_ratio_ * 100
    scal = b.load_extra("scalars", ids)
    recs = lab.to_dict("records")
    thumbs = [thumb(b, ids[i], np.argmax(scal[i][:4])) for i in range(len(ids))]
    caps = [caption(recs[i], scal[i], b.is_commit) for i in range(len(ids))]

    colors, legends, modes = {}, {}, []
    if b.has_belief:
        cat = lab["category"].astype(str).to_numpy()
        colors["belief"] = [style.CATEGORY_COLORS.get(c, "#888") for c in cat]
        legends["belief"] = cat_legend(style.CATEGORY_COLORS, set(cat))
        modes.append("belief")
        fc = lab["final_commit"].astype(str).to_numpy()
        colors["commit"] = [style.SKILL_COLORS.get(c, "#888") for c in fc]
        legends["commit"] = cat_legend(style.SKILL_COLORS, set(fc))
        modes.append("commit")
    # next action taken (t+1)
    name_color = {anames[k]: ACTION_HEX[k] for k in sorted(anames)}
    name_color["(end)"] = "#bcc4cc"
    na_names = []
    for r in ids:
        a = namap.get(int(r), np.nan)
        na_names.append("(end)" if (a != a) else anames.get(int(a), "?"))
    colors["next_action"] = [name_color.get(nm, "#888") for nm in na_names]
    legends["next_action"] = cat_legend(name_color, set(na_names))
    modes.append("next_action")

    colors["segment"] = [SEG_COLORS.get(s, "#888")
                         for s in lab["segment"].astype(str)]
    legends["segment"] = cat_legend(SEG_COLORS, set(lab["segment"].astype(str)))
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

    default = "belief" if b.has_belief else "next_action"
    safe = f"{name}__{src}"
    return dict(name=safe, coords=coords, evr=evr, thumbs=thumbs, caps=caps,
                colors=colors, legends=legends, modes=modes, default=default)


def static_png(panels, labels):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, p, lbl in zip(axes.ravel(), panels, labels):
        ax.scatter(p["coords"][:, 0], p["coords"][:, 1], s=4,
                   c=p["colors"][p["default"]], alpha=0.6, linewidths=0)
        ax.set_title(f"{lbl}\n(colour={p['default']})", fontweight="bold", fontsize=10)
        ax.set_xlabel(f"PC1 ({p['evr'][0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({p['evr'][1]:.1f}%)")
        ax.set_facecolor(style.PANEL); ax.grid(True, color="white")
    fig.suptitle("Hidden-state PCA (PC1×PC2) — default colour "
                 "(interactive html has belief/commit/next_action/segment/step%/value/xy)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("outputs/report/figs/hidden_state_pca.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3500
    panels, labels = [], []
    for name, src, label in SPECS:
        print(f"[{name} · {src}] PCA + thumbs (n={n}) ...", flush=True)
        panels.append(panel(name, src, n)); labels.append(label)
    parts = ["<!doctype html><meta charset='utf-8'>"
             "<title>hidden-state PCA — hover frame + scalars</title>"
             "<style>body{font-family:sans-serif;color:#223;max-width:1100px;"
             "margin:0 auto;padding:24px}h1{color:#1b4f72}</style>"
             "<h1>Hidden-state PCA — hover a dot for its 21×21 frame + observation scalars</h1>"
             "<p>3-D rotatable PCA of each agent's recurrent / latent state: PPO "
             "<b>h_t</b> (gru_h), Dreamer <b>rssm_deter</b> and <b>rssm_stoch</b>. "
             "Per-panel <b>colour by</b>: <b>belief</b> &amp; <b>commit</b> (BTC), "
             "<b>next_action</b> (action at t+1), segment, step%, V(h_t), xy. Hover "
             "any point to render its egocentric minimap (red arrow = agent facing) "
             "and decode the labelled observation scalars.</p>"]
    for i, (label, p) in enumerate(zip(labels, panels)):
        parts.append(html_panel(p, label, first=(i == 0)))
    out = Path("outputs/report/hidden_state_pca_hover.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")
    static_png(panels, labels)


if __name__ == "__main__":
    main()
