#!/usr/bin/env python
"""Short HTML report: belief-code geometry (shape/colour axes) across 2/3/4-cue
PPO models — cosine similarity, class separation, and the belief planes, tying
training-cue diversity to belief (dis)entanglement. Self-contained HTML.
"""
from __future__ import annotations

import base64
import io
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

CUE_NAMES = ["green_up", "blue_up", "green_down", "blue_down"]
CUE_COL = ["#1b9e77", "#3b6fb6", "#7fd4b8", "#9ec9ec"]
TRAIN = {"2cue": {0, 3}, "3cue": {0, 2, 3}, "4cue": {0, 1, 2, 3}}
MODELS = [("2cue", "outputs/ppo_runs/ppo_2cue_vs2"),
          ("3cue", "outputs/ppo_runs/ppo_3cue_vs"),
          ("4cue", "outputs/ppo_runs/ppo_4cue_vs4")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


def _b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=135, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def geometry(run_dir, name):
    d = np.load(pathlib.Path(run_dir) / "activations.npz", allow_pickle=True)
    X = d["feat"].astype(np.float64)
    ws = LogisticRegression(max_iter=3000).fit(X, d["shape"]).coef_[0]
    wc = LogisticRegression(max_iter=3000).fit(X, d["colour"]).coef_[0]
    ws, wc = ws / np.linalg.norm(ws), wc / np.linalg.norm(wc)
    m = d["phase"] >= 2                       # post-cue: belief formed
    ps, pc = X[m] @ ws, X[m] @ wc
    sep_s = abs(ps[d["shape"][m] == 1].mean() - ps[d["shape"][m] == 0].mean())
    sep_c = abs(pc[d["colour"][m] == 1].mean() - pc[d["colour"][m] == 0].mean())
    # entanglement as CROSS-DECODING LEAKAGE (robust where the cosine is not):
    # how well the colour-axis coordinate alone predicts SHAPE on the model's own
    # (trained-cue) distribution. 0.5 = independent channels, 1.0 = one shared code.
    m_tr = np.isin(d["cue_type"], list(TRAIN[name]))
    from sklearn.model_selection import cross_val_score
    leak = float(cross_val_score(LogisticRegression(max_iter=1000),
                                 (X[m_tr] @ wc).reshape(-1, 1),
                                 d["shape"][m_tr], cv=5).mean())
    # label-only null: what a perfectly DISENTANGLED code would leak, given the
    # shape/colour label correlation of this model's training cues.
    null = float(cross_val_score(LogisticRegression(max_iter=1000),
                                 d["colour"][m_tr].reshape(-1, 1).astype(float),
                                 d["shape"][m_tr], cv=5).mean())
    return dict(cos=abs(float(ws @ wc)), sep_s=float(sep_s), sep_c=float(sep_c),
                leak=leak, null=null, ps=ps, pc=pc, cue=d["cue_type"][m])


def plane_fig(geos):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    rng = np.random.default_rng(0)
    for ax, (name, _), g in zip(axes, MODELS, geos):
        idx = rng.choice(len(g["ps"]), min(4000, len(g["ps"])), replace=False)
        for c in range(4):
            mk = g["cue"][idx] == c
            trained = c in TRAIN[name]
            lab = CUE_NAMES[c] + ("" if trained else " (held out)")
            ax.scatter(g["ps"][idx][mk], g["pc"][idx][mk], s=6,
                       c=CUE_COL[c], alpha=0.6 if trained else 0.25,
                       marker="o" if trained else "x", edgecolors="none", label=lab)
        # same symmetric scale on both axes: the visual spread now reflects the
        # true relative strength of the two belief axes
        L = 1.06 * max(np.abs(g["ps"]).max(), np.abs(g["pc"]).max())
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
        ax.axhline(0, color="#ddd", lw=0.7, zorder=0)
        ax.axvline(0, color="#ddd", lw=0.7, zorder=0)
        ax.set_xlabel("shape axis  (h · w_shape)", fontsize=9)
        ax.set_ylabel("colour axis  (h · w_colour)", fontsize=9)
        verdict = ("ONE entangled code" if name == "2cue" else
                   "correlated axes" if name == "3cue" else "two orthogonal axes")
        ax.set_title(f"{name}  ·  |cos| = {g['cos']:.2f}  ·  {verdict}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=6.5, markerscale=1.6, framealpha=0.9, loc="best")
    fig.suptitle("Belief planes (post-cue GRU hidden states projected on the two probe axes)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def metrics_fig(geos):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    names = [n for n, _ in MODELS]
    x = np.arange(3)
    ax = axes[0]
    b = ax.bar(x, [g["leak"] for g in geos], 0.5, color=["#e15759", "#f28e2b", "#59a14f"],
               label="measured leakage")
    for i, g in enumerate(geos):
        ax.text(i, g["leak"] + 0.015, f"{g['leak']:.2f}", ha="center", fontweight="bold")
        # label-only null = what a perfectly disentangled code would show
        ax.hlines(g["null"], i - 0.33, i + 0.33, color="k", ls="--", lw=1.4)
        ax.text(i + 0.36, g["null"], f"null {g['null']:.2f}", fontsize=7.5, va="center")
        if g["leak"] - g["null"] > 0.05:
            ax.annotate("", xy=(i, g["leak"] - 0.01), xytext=(i, g["null"] + 0.01),
                        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
            ax.text(i - 0.1, (g["leak"] + g["null"]) / 2, "excess", fontsize=8,
                    ha="right", color="#333", fontstyle="italic")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.12)
    ax.set_ylabel("shape decodable from COLOUR axis")
    ax.set_title("cross-axis leakage vs label-only null (trained cues)",
                 fontsize=11, fontweight="bold")
    ax = axes[1]
    w = 0.35
    ax.bar(x - w / 2, [g["sep_s"] for g in geos], w, color="#4e79a7", label="shape axis")
    ax.bar(x + w / 2, [g["sep_c"] for g in geos], w, color="#e15759", label="colour axis")
    for i, g in enumerate(geos):
        ax.text(i - w / 2, g["sep_s"] + 0.05, f"{g['sep_s']:.2f}", ha="center", fontsize=8.5)
        ax.text(i + w / 2, g["sep_c"] + 0.05, f"{g['sep_c']:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("class-mean separation (raw units)")
    ax.set_title("axis strength (how far apart the class means are)", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return fig


def main():
    geos = [geometry(rd, n) for n, rd in MODELS]
    img_planes = _b64(plane_fig(geos))
    img_metrics = _b64(metrics_fig(geos))
    def _excess(g):
        return "—" if g["null"] > 0.97 else f"{g['leak'] - g['null']:+.2f}"
    rows = "".join(
        f"<tr><td><b>{n}</b></td><td>{', '.join(CUE_NAMES[c] for c in sorted(TRAIN[n]))}</td>"
        f"<td>{g['leak']:.2f}</td><td>{g['null']:.2f}</td><td><b>{_excess(g)}</b></td>"
        f"<td>{g['cos']:.3f}</td>"
        f"<td>{g['sep_s']:.2f}</td><td>{g['sep_c']:.2f}</td>"
        f"<td>{v}</td></tr>"
        for (n, _), g, v in zip(MODELS, geos,
                                ["one entangled code (no real shape axis)",
                                 "two correlated axes (partially shared)",
                                 "two strong orthogonal axes"]))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Belief-code geometry — MemoryEnv PPO models</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;
      color:#222;padding:0 16px}}
 h1{{border-bottom:2px solid #41ae76;padding-bottom:6px;font-size:1.5em}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px;margin:8px 0}}
 table{{border-collapse:collapse;margin:12px 0;font-size:0.92em}}
 td,th{{border:1px solid #ddd;padding:6px 12px;text-align:left}}
 th{{background:#f3f8f5}}
 .intro{{background:#f6fbf8;border-left:4px solid #41ae76;padding:10px 16px;border-radius:4px}}
 .note{{color:#666;font-size:0.88em}}
</style></head><body>
<h1>Training-cue diversity determines belief disentanglement</h1>
<div class="intro">
The shape and colour <b>belief axes</b> of each solved PPO+GRU agent are the weight vectors of two
linear probes (logistic regressions predicting the episode's cue shape / colour from the GRU hidden
state, fit on ~18k labelled states from 512 episodes). Two geometric quantities characterise the
memory code: the <b>angle between the axes</b> (cosine; 0&nbsp;=&nbsp;independent factors) and each
axis's <b>class-mean separation</b> (how strongly the belief is expressed). The result: the geometry
of the learned code mirrors the correlational structure of the training cues almost literally.
</div>

<table>
<tr><th>model</th><th>training cues</th>
<th>leakage: shape from<br>colour axis (trained cues)</th>
<th>label-only null<br>(disentangled code)</th><th>excess =<br>coding entanglement</th>
<th>|cos(w<sub>shape</sub>, w<sub>colour</sub>)|</th>
<th>shape sep.</th><th>colour sep.</th><th>belief code</th></tr>
{rows}
</table>

<p class="note"><b>Reading the metrics.</b> The cosine is only meaningful when both axes carry real
signal: the 2cue model's shape axis is nearly degenerate (separation 0.24, noise-level), so its
angle is arbitrary — the balanced eval labels bias it toward orthogonal (0.20), understating the
entanglement (probes fit on its own trained distribution give cos&nbsp;=&nbsp;1.000).
<b>Leakage</b> (shape decoded from the colour-axis coordinate) must in turn be read against the
<b>label-only null</b> — what a perfectly disentangled code would show, given each model's
shape↔colour label correlation on its own cues (1.00 / 0.70 / 0.53 for 2/3/4cue). The
<b>excess over the null</b> is the genuine coding entanglement: 3cue&nbsp;=&nbsp;+0.21 (same-colour
cues green_up/green_down sit at different positions along its colour axis: −0.17 vs +0.02),
4cue&nbsp;=&nbsp;±0.00 (same-colour cues coincide on the colour axis). For 2cue the null saturates
at 1.0, so leakage is uninformative there — its entanglement is shown by the axis collapse
(one strong axis, plane = vertical line) and causally by steering (shape push&nbsp;→ wrong door).</p>

<img src="data:image/png;base64,{img_metrics}"/>
<img src="data:image/png;base64,{img_planes}"/>

<h3>Reading the planes</h3>
<p><b>4cue</b> (all combinations seen; shape and colour vary independently): four tight clusters in a
2×2 grid — two strong, orthogonal axes (|cos|&nbsp;=&nbsp;0.03; separations 3.2&nbsp;/&nbsp;1.9).
<b>3cue</b> ("blue" only ever co-occurs with "down", label correlation 0.5): the learned axes inherit
the correlation (|cos|&nbsp;=&nbsp;0.54) and the held-out <i>blue_up</i> falls between clusters —
both features decodable, but along partially shared directions. <b>2cue</b> (shape&nbsp;≡&nbsp;colour,
perfectly confounded): effectively <b>one</b> code axis; the shape direction is nearly degenerate
(separation 0.24, ~13× weaker than 4cue) and held-out cues collapse onto the trained clusters.</p>

<h3>Causal confirmation (steering)</h3>
<p>These geometries predict the intervention outcomes measured earlier (n=96/condition): in
<b>4cue</b>, clamping the shape coordinate flips the branch (0.91–1.00) while the colour belief and
the door choice survive (success stays 1.00) — a surgical edit, possible only with separable axes.
In <b>2cue</b>, the same push cannot flip the branch, but corrupts the shared code so the agent
picks the wrong-colour door (success 1.00&nbsp;→&nbsp;0.00) — belief entanglement as behavioural
failure. <span class="note">Prediction for 3cue (untested): a shape-axis push should partially drag
the colour belief — intermediate between the two.</span></p>

<p class="note">Data: outputs/ppo_runs/{{ppo_2cue_vs2, ppo_3cue_vs, ppo_4cue_vs4}}/activations.npz ·
probes: L2-regularised logistic regression (sklearn, max_iter=3000), unit-normalised coefficients ·
separations measured on post-cue states (phase ≥ pre-branch) · held-out cues shown as ×.</p>
</body></html>"""
    out = pathlib.Path("outputs/belief_geometry_report.html")
    out.write_text(html)
    print(f"[report] wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
