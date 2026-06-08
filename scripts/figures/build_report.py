#!/usr/bin/env python3
"""Assemble a single self-contained HTML mech-interp report embedding every figure
(base64) with detailed explanations. Output: outputs/report/mechinterp_report.html
"""
from __future__ import annotations
import base64, html
from pathlib import Path

ROOT = Path("/home/filippo/GitHub/Crusoe-Cogniland")
A = ROOT / "outputs/analysis"
AD = ROOT / "outputs/analysis_dreamer"
AEXP = ROOT / "outputs/analysis_exp"
ASEP = ROOT / "outputs/analysis_sep"
AST = ROOT / "outputs/analysis_steer"
ASTR = ROOT / "outputs/analysis_steer_rec"
ABS = ROOT / "outputs/analysis_belief_steer"
FIG = ROOT / "outputs/report/figs"
OUT = ROOT / "outputs/report/mechinterp_report.html"

parts: list[str] = []
def h(s): parts.append(s)

def img(path, w=900, cap=""):
    p = Path(path)
    if not p.exists():
        h(f'<p class="missing">[missing figure: {html.escape(str(p.name))}]</p>'); return
    b = base64.b64encode(p.read_bytes()).decode()
    h('<figure>')
    h(f'<img style="max-width:{w}px" src="data:image/png;base64,{b}">')
    if cap: h(f'<figcaption>{cap}</figcaption>')
    h('</figure>')

def P(s): h(f"<p>{s}</p>")
def H2(s, _id=""): h(f'<h2 id="{_id}">{s}</h2>')
def H3(s): h(f"<h3>{s}</h3>")

# ───────────────────────────── head ─────────────────────────────
h("""<!doctype html><html><head><meta charset="utf-8"><title>bridge_tunnel mech-interp report</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#22303c;line-height:1.55;
 max-width:1000px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:30px;border-bottom:3px solid #2e86c1;padding-bottom:8px}
h2{font-size:23px;margin-top:46px;border-bottom:1px solid #cdd8e3;padding-bottom:5px;color:#1b4f72}
h3{font-size:18px;margin-top:30px;color:#21618c}
figure{margin:18px 0;text-align:center}
img{border:1px solid #e1e7ee;border-radius:6px;width:100%;height:auto}
figcaption{font-size:13.5px;color:#445;margin-top:7px;text-align:left;background:#f4f7fa;
 border-left:3px solid #2e86c1;padding:8px 12px;border-radius:0 5px 5px 0}
table{border-collapse:collapse;margin:16px 0;font-size:14px;width:100%}
th,td{border:1px solid #cdd8e3;padding:6px 10px;text-align:center}
th{background:#eaf2f8}td:first-child,th:first-child{text-align:left}
.missing{color:#b00;font-style:italic}
code{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:13px}
.key{background:#eafaf1;border-left:4px solid #2a9d4a;padding:10px 14px;margin:14px 0;border-radius:0 6px 6px 0}
.warn{background:#fdf3e7;border-left:4px solid #e08e2b;padding:10px 14px;margin:14px 0;border-radius:0 6px 6px 0}
nav{background:#f4f7fa;border:1px solid #cdd8e3;border-radius:8px;padding:14px 20px;font-size:14px}
nav a{color:#1b4f72;text-decoration:none}nav a:hover{text-decoration:underline}
</style></head><body>""")

h("<h1>Mechanistic interpretability of a navigation agent:<br>PPO+GRU vs DreamerV3 on <code>bridge_tunnel</code></h1>")
P("A controlled study of how a <b>model-free</b> (PPO+GRU) and a <b>model-based</b> (DreamerV3) "
  "agent represent <b>belief</b> (which map type they are on) and <b>skill</b> (which obstacle-crossing "
  "tool they commit to), and whether those representations are decodable, separable, and causally steerable. "
  "Every figure below is explained in <b>data → method → how to read → finding</b> form.")

h("""<nav><b>Contents</b><br>
1. <a href="#setup">Setup &amp; study design</a> &nbsp;·&nbsp;
2. <a href="#arch">Architectures</a> &nbsp;·&nbsp;
3. <a href="#data">The four datasets</a> &nbsp;·&nbsp;
4. <a href="#geom">Geometry of the representations</a> &nbsp;·&nbsp;
5. <a href="#sep">Separability / confound</a> &nbsp;·&nbsp;
6. <a href="#steer">Causal steering</a> &nbsp;·&nbsp;
7. <a href="#behav">Behavior &amp; training</a> &nbsp;·&nbsp;
8. <a href="#imag">Imagination</a> &nbsp;·&nbsp;
9. <a href="#concl">Conclusions</a></nav>""")

# ───────────────────────────── 1 setup ─────────────────────────────
H2("1 · Setup &amp; study design", "setup")
P("<b>Environment.</b> A POMDP grid-navigation task. The agent sees a 21×21 egocentric crop of a 9-tile "
  "world and must reach a target. Obstacles can be <i>crossed</i> by converting them — <b>build</b> turns "
  "water→wood (a bridge), <b>mine</b> turns rock→grass (a tunnel) — or <i>avoided</i> by detouring. "
  "A per-step slack penalty + a small commitment cost + a slip 'weight tax' (committing slows movement on "
  "normal terrain) make detour-vs-cross a genuine trade-off.")
img(ROOT / "data/bridge_tunnel/val_maps_preview.png", 900,
    "<b>Held-out evaluation maps.</b> 32×64 worlds: grass (green), water (blue), rock (grey), trees (dark "
    "green), target (yellow). The agent spawns on the left, target on the right; water/rock barriers force "
    "the avoid-vs-cross decision.")
P("<b>Two variants (controlled vs uncontrolled).</b> <b>BT</b> (base): tools always available, no labels — "
  "beliefs unlabeled, crossing optional. <b>BTC</b> (commit): 3 <b>labeled map categories</b> "
  "(balanced / lakes≈85% water / rocky≈85% rock) = ground-truth <b>belief</b>; plus an <b>irreversible "
  "one-shot commitment</b> to build or mine = ground-truth <b>skill</b>. BTC is where belief and skill are "
  "observable at every timestep.")
P("<b>Two agents.</b> PPO+GRU (model-free) and DreamerV3 (model-based). That gives the 2×2:")
img(FIG / "pipeline.png", 980,
    "<b>The 2×2 design and the analysis pipeline.</b> Left: the four activation datasets "
    "{BT,BTC}×{PPO,Dreamer}, same held-out maps per variant so the agents are directly comparable. "
    "Right: each dataset is run through DR (PCA/UMAP), map-grouped linear probes for belief &amp; skill, "
    "difference-of-means directions + entanglement metrics, the separability tests, and causal steering.")

# ───────────────────────────── 2 architectures ─────────────────────────────
H2("2 · Architectures", "arch")
H3("PPO + GRU (model-free, ~2.0 M params)")
img(FIG / "arch_ppo.png", 980,
    "Forward pass: 21×21 minimap (+5/7 scalars) → one-hot + CoordConv → 3 conv layers → Linear-256 "
    "(<code>enc_embed</code>, probed) → GRU producing <code>gru_h</code>∈ℝ¹²⁸ (the recurrent <b>belief "
    "carrier</b>, probed) → actor (6 logits: ↑↓←→ build mine) + critic. Trained by PPO on real returns "
    "with a high entropy bonus.")
H3("DreamerV3 (model-based, ~19.2 M params)")
img(FIG / "arch_dreamer.png", 980,
    "A learned world model. The encoder embeds the obs; the <b>RSSM</b> maintains a latent state with a "
    "deterministic part <code>rssm_deter</code>∈ℝ³⁰⁷² (block-GRU, the <b>belief carrier</b>, probed) and a "
    "stochastic categorical part (24×24, <code>stoch_logits</code> probed). The features [deter⊕stoch] feed "
    "a decoder (reconstruct obs), reward + continue heads, and the actor/critic. Crucially the policy is "
    "trained <b>in imagination</b> — rolling the RSSM forward with no environment — so the actor learns from "
    "the reward head over imagined latents, not from the decoder.")
P("The ~9.5× parameter gap (19.2 M vs 2.0 M) matters when interpreting results: some of Dreamer's cleaner "
  "belief representation could reflect capacity + the world-model objective, not the objective alone.")

# ───────────────────────────── 3 data ─────────────────────────────
H2("3 · The four activation datasets", "data")
P("Each dataset is a self-contained bundle: <code>activations.h5</code> (per-timestep activation sources + "
  "the full obs) + <code>labels.parquet</code> (belief/skill/strategy labels computed from map geometry, "
  "not from activations) + <code>decisions.parquet</code> + <code>maps.npz</code>. Labels never leak into "
  "the activations; probes are evaluated on held-out maps (see §5).")
h("""<table>
<tr><th>dataset</th><th>rows</th><th>activation sources</th><th>belief/skill labels</th><th>rollout success</th></tr>
<tr><td>bt_ppo</td><td>483,695</td><td>gru_h(128), enc_embed(256)</td><td>— (uncontrolled)</td><td>~99.9%</td></tr>
<tr><td>btc_ppo</td><td>673,913</td><td>gru_h(128), enc_embed(256)</td><td>category + commit</td><td>~98%</td></tr>
<tr><td>bt_dreamer</td><td>322,803</td><td>rssm_deter(3072), stoch_logits(576), enc_embed(384)</td><td>— (uncontrolled)</td><td>96.7%</td></tr>
<tr><td>btc_dreamer</td><td>568,390</td><td>rssm_deter(3072), stoch_logits(576), enc_embed(384)</td><td>category + commit</td><td>92.6%</td></tr>
</table>""")

# ───────────────────────────── 4 geometry ─────────────────────────────
H2("4 · Geometry of the representations", "geom")
P("All from the analysis pipeline (3-D PCA/UMAP, linear probes, difference-of-means directions). "
  "<b>Probes are trained and tested on disjoint maps</b> (grouped split), so probe accuracy is the honest "
  "<i>generalisable</i> decodability — a probe cannot cheat by memorising map identity. Belief chance = 0.33.")
h("""<table>
<tr><th>source</th><th>belief acc</th><th>belief ordinal R²/ρ</th><th>skill acc</th><th>cos(belief,skill)</th></tr>
<tr><td>PPO gru_h</td><td>0.38</td><td>−0.33 / +0.07</td><td><b>0.60</b></td><td>+0.59</td></tr>
<tr><td>PPO enc_embed</td><td>0.43</td><td>+0.10 / +0.38</td><td>0.53</td><td>+0.67</td></tr>
<tr><td><b>Dreamer rssm_deter</b></td><td><b>0.64</b></td><td><b>+0.38 / +0.67</b></td><td>0.49</td><td>+0.53</td></tr>
<tr><td>Dreamer enc_embed</td><td>0.59</td><td>+0.47 / +0.72</td><td>0.51</td><td>+0.84</td></tr>
<tr><td>Dreamer stoch_logits</td><td>0.51</td><td>+0.19 / +0.48</td><td>0.44</td><td>+0.71</td></tr>
</table>""")
h('<div class="key"><b>Finding.</b> The model-based RSSM <code>rssm_deter</code> is a far cleaner '
  '<b>belief</b> carrier than the model-free <code>gru_h</code> (0.64 vs 0.38; ordinal ρ 0.67 vs 0.07 — '
  'PPO\'s GRU barely encodes the graded water↔rock belief). Conversely <code>gru_h</code> is the better '
  '<b>skill</b> carrier (0.60 vs 0.49). Division of labor: GRU = action state, RSSM = world/belief state.</div>')

def geom_block(title, base, src):
    H3(title)
    img(base / f"{src}__pca_by_category.png", 720,
        "<b>PCA-by-belief (3-D).</b> Each point = a timestep; the activation reduced to its top-3 PCs; "
        "colour = true map category (rocky=red, balanced=purple, lakes=blue); orange line = centroid path "
        "rocky→balanced→lakes. <i>Read:</i> separated colour regions ⇒ belief is geometrically present; "
        "salt-and-pepper ⇒ it is not (in the top PCs).")
    img(base / f"{src}__pca_by_skill.png", 720,
        "<b>PCA-by-skill (3-D).</b> Same points coloured by committed skill (none=blue, build=yellow, "
        "mine=purple). <i>Read:</i> how cleanly the three skills occupy distinct regions.")
    if (base / f"{src}__pca_by_belief_score.png").exists():
        img(base / f"{src}__pca_by_belief_score.png", 720,
            "<b>PCA coloured by <i>decoded</i> belief score</b> P(lakes)−P(rocky) from the probe (red→blue, "
            "[−1,1]). <i>Read:</i> a smooth red→blue gradient ⇒ belief varies continuously and is linearly "
            "readable.")
    img(base / f"{src}__centroids_category.png", 620,
        "<b>Class centroids.</b> Big dots = per-category means, faint cloud = members. <i>Read:</i> centroid "
        "separation relative to within-class spread = the size of the belief signal vs the noise.")
    img(base / f"{src}__confusion_belief.png", 430,
        "<b>Belief probe confusion</b> (held-out maps; rows=true, cols=predicted). <i>Read:</i> strong "
        "diagonal = decodable; here errors concentrate on balanced↔extreme confusions, while lakes↔rocky "
        "(the water-axis extremes) are rarely confused.")
    img(base / f"{src}__confusion_skill.png", 430,
        "<b>Skill probe confusion</b> (none/build/mine).")
    img(base / f"{src}__cosine_belief_vs_skill.png", 620,
        "<b>Cosine heatmap</b> of belief difference-of-means directions (rows) × skill directions (cols), "
        "RdBu, [−1,1]. <i>Read:</i> the top-left cell cos(lakes−rocky, build−mine) is the headline "
        "'entanglement' number — §5 shows it is mostly a label confound.")
    img(base / f"{src}__entanglement_plane.png", 940,
        "<b>Belief×skill plane (the key descriptive figure).</b> The <i>same</i> cloud projected onto "
        "x=belief axis (lakes−rocky), y=skill axis (build−mine), shown twice — left coloured by category, "
        "right by skill. <i>Read:</i> a diagonally-stretched cloud where <b>both</b> colourings separate "
        "along the same diagonal = the two axes are aligned (apparent entanglement).")
    img(base / f"{src}__pca_trajectories.png", 640,
        "<b>Episode trajectories</b> through PCA space (□ start, ★ end), coloured by skill. <i>Read:</i> "
        "build vs mine episodes fan into different regions after the commitment event — the commit is "
        "visible as a bifurcation in the latent trajectory.")

geom_block("PPO · gru_h (BTC)", A, "gru_h")
geom_block("PPO · enc_embed (BTC)", A, "enc_embed")
geom_block("DreamerV3 · rssm_deter (BTC)", AD, "rssm_deter")
geom_block("DreamerV3 · enc_embed (BTC)", AD, "enc_embed")
geom_block("DreamerV3 · rssm_stoch_logits (BTC)", AD, "rssm_stoch_logits")
H3("Cross-source probe accuracy")
img(A / "summary__probe_accuracy.png", 560, "PPO probe accuracies across sources (held-out maps).")
img(AD / "summary__probe_accuracy.png", 560, "Dreamer probe accuracies across sources.")
H3("BT (uncontrolled): manifold trajectories only")
P("BT has no belief/skill labels, so only the unsupervised geometry exists — the manifold trajectories. "
  "BT is the honest 'beliefs unlabeled, skills unenforced' baseline.")
img(ROOT / "outputs/analysis_bt/gru_h__pca_trajectories.png", 600, "bt_ppo · gru_h episode trajectories (coloured by timestep).")
img(ROOT / "outputs/analysis_bt_dreamer/rssm_deter__pca_trajectories.png", 600, "bt_dreamer · rssm_deter episode trajectories.")

# ───────────────────────────── 5 separability ─────────────────────────────
H2("5 · Separability / confound experiments", "sep")
P("The §4 cosine ≈0.5–0.6 <i>looks</i> like belief and skill are entangled. These three tests check whether "
  "that is real or just a <b>label confound</b> (the agent commits build on lakes and mine on rocky, so the "
  "two label sets co-occur).")
H3("Within-category control (does the cosine survive at fixed belief?)")
img(ASEP / "rssm_deter__within_category_control.png", 620,
    "<b>Dreamer rssm_deter.</b> cos(belief, skill) computed globally (grey) vs <i>within each fixed category</i>. "
    "<i>Read:</i> if it collapses at fixed belief it was a confound. It does: global +0.54 → "
    "balanced +0.12, lakes −0.16, rocky +0.23. (PPO gru_h is the same: global +0.58 → balanced +0.00.)")
H3("E1 · subspace removal (does each factor survive removing the other?)")
img(AEXP / "E1__decodability_after_removal.png", 560,
    "<b>PPO gru_h.</b> skill 0.59→<b>0.60</b> after removing the belief subspace; belief 0.40→<b>0.41</b> after "
    "removing the skill subspace. Min principal angle 39°. <i>Read:</i> each factor is fully decodable from "
    "the part orthogonal to the other ⇒ the shared component is not load-bearing.")
img(ASEP / "E1__decodability_after_removal.png", 560,
    "<b>Dreamer rssm_deter.</b> skill 0.50→0.49; belief 0.60→0.46 (mostly survives). Min angle 36°.")
H3("E2 · off-type generalisation (the decisive test)")
img(AEXP / "E2__offtype_generalisation.png", 560,
    "<b>PPO gru_h.</b> A build-vs-mine probe tested on on-type vs <b>against-type</b> commits (build-on-rocky, "
    "mine-on-lakes). Full features: on 0.67 / <b>off 0.37</b> (below chance — it is a belief detector). "
    "Belief-removed: on 0.64 / <b>off 0.62</b> ⇒ a genuine belief-free skill code.")
img(ASEP / "E2__offtype_generalisation.png", 560,
    "<b>Dreamer rssm_deter.</b> Full: on 0.83 / <b>off 0.33</b>; belief-removed: on 0.55 / <b>off 0.83</b>. "
    "Same story, even sharper.")
h('<div class="key"><b>Finding.</b> Belief and skill are <b>separable</b> in both agents — the apparent '
  'entanglement is a label/estimator artifact, shown three independent ways (within-category collapse, '
  'subspace-removal survival, off-type generalisation). A belief-free skill code provably exists.</div>')

# ───────────────────────────── 6 steering ─────────────────────────────
H2("6 · Causal steering", "steer")
img(FIG / "steering_sites.png", 940,
    "<b>Where to steer.</b> Injecting into the <i>actor input</i> (read-only) is controllable and preserves "
    "success; injecting persistently into the <i>recurrent carry</i> compounds off-manifold and breaks the "
    "agent. Same lesson in both agents.")
H3("PPO skill steering — it works (actor-input + decision direction)")
img(AST / "steering__balanced_commit_control.png", 640,
    "<b>Balanced maps, actor-input · decision direction.</b> P(build), P(mine), reach vs injection α. "
    "<i>Read:</i> negative α drives the commit to mine (P=0.70 at −6), positive α to build (0.50 at +6), "
    "monotonically, with reach 0.79–1.0 across α∈[−6,6]; only |α|=12 breaks it. <b>Skill is a controllable, "
    "success-preserving causal lever.</b>")
img(AST / "steering__control_and_reach.png", 940,
    "<b>Method comparison.</b> Commit control (left) and reach (right) vs α for four methods. <i>Read:</i> "
    "the injection <b>site</b> is decisive — actor-input methods steer with reach intact; the recurrent-state "
    "method gives reach=0 at every α.")
img(AST / "steering__belief_projection.png", 600,
    "<b>Belief projection vs α.</b> Actor-site injection leaves the recurrent belief state near-baseline "
    "(0.1–0.4); the recurrent method blows it to ±100s (off-manifold) — explaining the breakage.")
H3("Can you steer from the recurrent state? (schedules)")
img(ASTR / "recurrent__control_and_reach.png", 940,
    "<b>Recurrent-state injection schedules.</b> Persistent and norm-clamped injection break the agent "
    "(reach≈0 everywhere); <b>first-K</b> (inject ~8 steps then release) <i>works</i>, tracking the "
    "actor-input reference in a band α∈[−4,+2]. So the recurrent belief carrier <i>is</i> steerable — but "
    "only with a transient nudge.")
H3("Dreamer belief steering")
img(ABS / "belief_steer_feature.png", 940,
    "<b>Feature (read-only) site.</b> Inject the lakes−rocky belief axis; commit P(build)−P(mine) shifts "
    "monotonically across all categories with reach 0.85–0.98. Belief is a controllable, success-preserving "
    "lever on the skill choice.")
img(ABS / "belief_steer_recurrent.png", 940,
    "<b>Recurrent site.</b> Reach→0 — same breakage lesson as PPO.")
h('<div class="warn"><b>Caveat.</b> The Dreamer belief-steer used the raw difference-of-means belief axis, '
  'which is confound-contaminated (cos +0.54 with skill), so the effect runs <i>opposite</i> to the naive '
  'belief→matching-skill mapping. The clean fix is a belief-probe / skill-orthogonalised direction — the one '
  'open loose end.</div>')
H3("The earlier null (documented)")
img(AEXP / "E3__response_curves.png", 940,
    "<b>First steering attempt (superseded).</b> Constant injection into the recurrent <code>gru_h</code>: "
    "the decoded-belief readout <b>saturates</b> (a step function pinned at ±1, identical for all directions) "
    "and skill never cleanly flips. This null motivated the actor-input redesign above.")

# ───────────────────────────── 7 behavior ─────────────────────────────
H2("7 · Behavior &amp; training (BTC commitment)", "behav")
H3("Commit matrices — training the Dreamer to a PPO-like map→skill bias")
img(A / "dreamer_v3_baseline_commit_matrix.png", 460,
    "<b>Baseline Dreamer.</b> Rows=map category, cols=committed skill, cell=fraction of episodes. ~0.92 "
    "<i>none</i> everywhere — it almost never commits (the problem).")
img(A / "dreamer_btc_ent01_6M_commit_matrix.png", 460,
    "<b>Fixed Dreamer (entropy 3e-4→0.01, 6M steps).</b> Decisive map→skill bias: lakes→build 0.44, "
    "rocky→mine 0.53, balanced even; 84–96% success. The fix was <b>exploration</b>, not reward — PPO "
    "trained with a high entropy bonus; Dreamer's paper-default was too low to discover that committing pays "
    "off given the slip weight-tax.")
H3("Stochastic trajectory grids")
img(ROOT / "outputs/previews/dreamer_bt_traj_grid.png", 940,
    "<b>BT Dreamer, 200 stochastic rollouts/map.</b> Path bundles fan into different routes (the policy is "
    "genuinely stochastic); red = bridge water (build), yellow = tunnel rock (mine). The strategy mix varies "
    "by map — some lean bridging, some tunnelling, some detour. 97.5% success.")
H3("Success: stochastic vs argmax (all four agents)")
h("""<table>
<tr><th>agent</th><th>stochastic</th><th>argmax (greedy)</th></tr>
<tr><td>bt_ppo</td><td><b>1.00</b></td><td>0.87</td></tr>
<tr><td>btc_ppo</td><td><b>0.965</b></td><td>0.644</td></tr>
<tr><td>bt_dreamer</td><td><b>0.967</b></td><td>0.912</td></tr>
<tr><td>btc_dreamer</td><td><b>0.903</b></td><td>0.331</td></tr></table>""")
h('<div class="key"><b>Findings.</b> (1) <b>argmax hurts both agents</b> (btc_dreamer 0.90→0.33) — the '
  'stochastic policy is genuinely the better controller in this POMDP; the gap is not an eval-temperature '
  'artifact. (2) Under the fair stochastic eval, <b>PPO beats Dreamer by 3–6 points with ~10× fewer '
  'params</b>. (3) Dreamer is more brittle under argmax (a single greedy mis-commit dooms the episode).</div>')

# ───────────────────────────── 8 imagination ─────────────────────────────
H2("8 · Imagination (does the world model dream the events?)", "imag")
P("DreamerV3 'dreams' by rolling the RSSM forward in latent space (no env) and decoding each latent back to "
  "a frame. We tested whether it imagines a <b>bridge</b> (build→water→wood), a <b>tunnel</b> (mine→rock→grass), "
  "or <b>reaching</b> the target, via a counterfactual: force the tool while genuinely facing the obstacle "
  "and check if the decoded tile converts.")
strips = sorted(ROOT.glob("outputs/videos/dreamer_imagine_*/imagine_strip_*.png"))
if strips:
    img(strips[0], 940,
        "<b>Imagination filmstrip.</b> Top row = the world model's decoded egocentric view; the sequence "
        "runs REAL warmup frames then IMAGINED open-loop frames. Short-horizon dreams are coherent; long "
        "horizons drift.")
h("""<table>
<tr><th>event</th><th>imagined?</th></tr>
<tr><td>Tunnel (mine: rock→grass)</td><td>sometimes — 1/4 forced cases converted</td></tr>
<tr><td>Bridge (build: water→wood)</td><td><b>no</b> — 0/4; forced build keeps imagining water</td></tr>
<tr><td>Reach target</td><td><b>no</b> — 0/18 imagined frames put the target at the agent's cell</td></tr>
</table>""")
h('<div class="warn"><b>Finding.</b> The world-model <i>decoder</i> does not faithfully render the rare '
  'build/mine conversions, and long open-loop dreams drift (the view floods with water; wood never appears). '
  'But the agent does not rely on the decoder to act — DreamerV3 trains the policy from the <b>reward head</b> '
  'over imagined latents, which can score commitment value without the decoder drawing the converted tile. '
  'So behavior is correct (92% success, good commit matrix) even though the dreams do not show bridges/goals.</div>')

# ───────────────────────────── 9 conclusions ─────────────────────────────
H2("9 · Conclusions", "concl")
h("""<ol>
<li><b>Model-based ⇒ explicit belief.</b> Dreamer's <code>rssm_deter</code> linearly encodes map-type belief
far better than PPO's <code>gru_h</code> (0.64 vs 0.38; ordinal ρ 0.67 vs 0.07); PPO's GRU is the better
<i>skill</i> carrier. → Dreamer is the cleaner substrate for belief probing/steering; PPO for skill.</li>
<li><b>Belief and skill are separable</b> in both agents; the apparent entanglement is a label confound,
demonstrated three independent ways (§5).</li>
<li><b>Both factors are causal levers</b> — skill (PPO), belief (Dreamer) — if you steer the read-out / actor
input, not the recurrent state; transient recurrent nudges also work, persistent ones break the agent (§6).</li>
<li><b>Behavior:</b> PPO is the stronger controller (3–6 pts, ~10× fewer params); the gap is real (argmax
hurts both). Dreamer's value is its representation, not its success rate (§7).</li>
<li><b>Imagination</b> does not faithfully render the rare build/mine conversions or long-horizon goal
arrival — a decoder limitation that does not impair the latent-space policy (§8).</li>
</ol>""")
P("<b>W&amp;B runs</b> (project <code>bridge_tunnel_geometry</code>): geometry — bt_ppo 19yhjrkw, btc_ppo "
  "ze6o7418, bt_dreamer zaksinvz, btc_dreamer mf9g2p7m; separability — PPO ht4pboxs, Dreamer 6lcg62le; "
  "steering — PPO svdzswnk (+recurrent t9wcrgqr), Dreamer belief mpaxv673.")

h("</body></html>")
OUT.write_text("\n".join(parts))
mb = OUT.stat().st_size / 1e6
print(f"wrote {OUT}  ({mb:.1f} MB)")
