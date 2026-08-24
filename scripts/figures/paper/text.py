#!/usr/bin/env python3
"""EVERY word that appears inside a Cogniland figure, in one file.

If you want to change a title, an axis label, a legend entry or an annotation,
change it here and re-run that figure — you never need to open the plotting
code. Nothing in this file draws anything; it is only strings.

  python scripts/figures/make_figures.py --only task     # after editing FIG01/FIG02
  python scripts/figures/make_figures.py --list          # which target is which figure

Conventions
  * `{...}` placeholders are filled in by the plotting code; the comment above
    each one says what goes in. Keep the placeholder names if you edit the text.
  * "(a)", "(b)" prefixes are part of the panel titles, so re-lettering a figure
    means editing them here too.
  * Figure numbers below match the numbers in the report.
"""

# ── Figure 1 · the three map types ───────────────────────────────────────
FIG01 = dict(
    title="Cogniland map types",
    # {cat} = rocky | balanced | lakes,  {door} = which door pays for that type
    panel="{cat}   →   {door}",
    # which door each map type rewards, shown in the panel titles above
    door_for={"lakes": "bottom door", "rocky": "top door",
              "balanced": "either door"},
    legend=dict(water="water", rock="rock", tree="tree",
                good="rewarded door", bad="decoy door", spawn="spawn"),
)

# ── Figure 2 · anatomy of one episode ────────────────────────────────────
FIG02 = dict(
    title="Anatomy of an episode",
    subtitle=("what the agent receives — a 21×21 egocentric crop (Crafter tiles) "
              "plus heading and elapsed time; black = out of bounds"),
    map_panel="(a) the full map — one PPO episode (yellow)",
    # the four numbered callouts drawn on the map
    phase1="1 · evidence\n(terrain reveals the type)",
    phase2="2 · memory corridor\n(16 columns, no evidence)",
    phase3="3 · passage",
    phase4="4 · door choice",
    # observation panels; {t} = timestep, {where} picks from `where` below
    obs_panel="({letter}) t = {t} · {where}",
    where=dict(
        evidence="terrain evidence still in view",
        corridor="memory corridor — no evidence left to see",
        past_wall="past the wall — committing to a door",
    ),
)

# ── Figure 3 · what the agent observes ───────────────────────────────────
FIG03 = dict(
    title=("The agent is a POMDP observer: a symbolic local crop plus "
           "five scalars, never the map"),
    world="(a) world state (privileged; never observed)",
    crop="(b) observation: {v}×{v} egocentric crop",          # {v} = view size
    crop_x="out-of-bounds padding: {frac} of cells",
    scalars="(c) observation: 5 scalars",
    scalars_x="value",
)

# ── Figure 4 · the reward ────────────────────────────────────────────────
FIG04 = dict(
    cumulative="(a) return decomposition, near-optimal episode",
    per_step="(b) per-step reward (spike = +3 door bonus)",
    x="environment step",
    y_cum="reward",
    y_step="per-step reward",
    legend=dict(total="cumulative return",
                slack="cumulative slack ($-0.01\\,t$)",
                shaping="cumulative shaping + bonus"),
)

# ── Figure 5 · dataset coverage ──────────────────────────────────────────
FIG05 = dict(
    title=("Each map type is a different pair of terrain fractions — that "
           "difference is the only signal"),
    panel="{cat} coverage",              # {cat} = rocky | balanced | lakes
    x="% of map cells",
    y="maps",
    legend=dict(water="water", rock="rock"),
)

# ── Figures 6-11 · the map-generation chapter ────────────────────────────
# (numbering here follows the report: noise primer, octaves, warp, features,
#  quantile, pipeline)
FIG_NOISE = dict(
    title_a="(a) value noise — interpolate random\nvalues at lattice points",
    title_b="(b) gradient (Perlin) noise — interpolate\n"
            "dot products of random gradients",
    title_c="(c) inside one cell: dot(gradient, offset)\n"
            "at each of the four corners",
    title_d="(d) the ease curve — zero 1st and 2nd\nderivative at the lattice",
    sample_point="sample point",
    linear="linear $t$",
    quintic="quintic $6t^5-15t^4+10t^3$",
    x="$t$", y="blend weight",
)

FIG_OCTAVES = dict(
    title=("Fractal (fBm) summation — each octave halves the wavelength "
           "and quarters the weight"),
    octave="octave {k}\nwavelength {wl:.1f} px · weight {w}",
    total="Σ weighted octaves\n= fractal heightmap",
    x="column",
    y="value on\nthe centre row",
)

FIG_WARP = dict(
    title=("Domain warping — two more noise fields bend the coordinates "
           "before sampling"),
    a="(a) heightmap $h_0$",
    b="(b) row-offset field",
    c="(c) column-offset field",
    d="(d) warped: $h_0(r+\\alpha w_r,\\; c+\\alpha w_c)$",
)

FIG_FEATURES = dict(
    title=("The three overlay primitives, added to (mountains) or subtracted "
           "from (basins, rivers) the heightmap"),
)

FIG_QUANTILE = dict(
    title=("The category is a pair of quantiles, not a different world. "
           "One heightmap, three cut points."),
    hist="(a) one heightmap, three pairs of cut points\n"
         "(water below the low cut, rock above the high one)",
    panel="({letter}) {cat}\nwater {wf:.1%} · rock {rf:.1%}",
    x="heightmap value", y="cells",
)

FIG_PIPELINE = dict(
    # {seed} and {cat} identify the map being walked through
    title="Six stages of one real map  (seed {seed}, category “{cat}”)",
)

# ── Figure 12 · trajectory density ───────────────────────────────────────
FIG_TRAJ = dict(
    title=("24 stochastic episodes per panel — where runs agree the ink "
           "stacks into a highway; single deviations stay faint but visible"),
    panel="{agent} — {ok}/{n} reach the right door",
    row="map {mid}\n({cat})",
    legend="one episode",
)

# ── Figures 13-14 · imagined futures ─────────────────────────────────────
FIG_DREAMS = dict(
    title="{AGENT} — imagined futures from a real context "
          "(no observations after the context frame)",
    context="context\n(last of {n})",
    step="+{i}",
    agreement="{pct:.0f}% tiles",
    row_real="reality",
    row_dream="dream",
    cut="the model sees\nnothing after\nthis frame",
)

# ── Figure 15 · three agents compared ────────────────────────────────────
FIG_COMPARE = dict(
    title="Three agents, one task: all clear the memoryless ⅔ ceiling",
    curves="(a) training-time learning curves",
    eval="(b) unified eval, all 1 200 test maps, all sampling\n"
         "(no pairwise difference significant)",
    residual="(c) how the residual error is spent",
    x="environment frames (M)",
    y_curves="training success (proxy)",
    y_eval="held-out success (TRUE metric)",
    y_residual="share of episodes (%)",
    ceiling="constant-door ceiling (⅔)",
    legend=dict(ppo="PPO+GRU", dreamer="DreamerV3 25M", storm="STORM",
                wrong="wrong door", timeout="timeout"),
)

# ── Figure 16 · evidence integration ─────────────────────────────────────
FIG_EVIDENCE = dict(
    title="What the agent saw versus what it chose — all 1 200 held-out maps per agent",
    free="(a) balanced maps — the door choice is FREE",
    free_x="rock − water seen  (normalised)",
    free_y="P(top door)",
    free_note="dotted = each agent's overall\nP(top): its standing bias",
    phases="(b) the belief decides — but only before the wall",
    phases_y="|AUC − 0.5|  on the free door choice",
    phases_note="dotted + band = shuffled-label null (mean ± sd)",
    # x tick labels for the phase sweep, in order
    phase_ticks=["spawn\n(undefined)", "evidence\nends", "corridor\nmid", "at the\nwall"],
    errors="(c) lakes + rocky — why the errors happen",
    errors_y="evidence for the true type",
    errors_ticks=["correct door", "WRONG door"],
)

# ── Figures 17-19 · per-agent training detail ────────────────────────────
FIG_TRAINING = dict(
    ppo_title="PPO + GRU — released run ({name})",
    dreamer_title="DreamerV3 — released run ({pick})",
    storm_title="STORM — released recipe (entropy 0.01, batch_length 128, context 128)",
    x_frames="environment frames (M)",
    x_episodes="training episode",
    ceiling="constant-door ceiling (⅔)",
    chance="chance (⅓)",
    # STORM flushes its losses once per segment, so those panels are sparse
    sparse_suffix="   (1 sample / 200k frames)",
)

# ── Figure 20 · checkpoint metastability ─────────────────────────────────
FIG_META = dict(
    title="Door-binding is metastable — the same run, 25k gradient steps apart",
    curves="(a) every archived checkpoint, one run",
    outcomes="(b) what the failures are, checkpoint by checkpoint",
    x="gradient step (thousands)",
    y_success="held-out success (TRUE metric)",
    y_share="share of episodes",
    ceiling="constant-door ceiling (⅔)",
    best="best of archive\n{pct:.1f} %",
    final="end of training\n{pct:.1f} %",
    all_maps="all maps",
    legend=dict(correct="correct door", wrong="wrong door", timeout="timeout"),
)

# ── shared across figures ────────────────────────────────────────────────
AGENT_LABEL = {"ppo": "PPO + GRU", "dreamer": "DreamerV3", "storm": "STORM"}
