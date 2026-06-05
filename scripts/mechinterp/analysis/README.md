# Activation-geometry analysis pipeline

Reusable pipeline for interpreting the hidden-state geometry of the BT / BTC
PPO+GRU agents and logging everything to a single, well-organised **W&B run**.

The scientific question (BTC): *do hidden states encode the agent's belief over
map type (balanced / lakes / rocky), and are skill-related directions
(build bridge / mine tunnel) geometrically **entangled** with belief-related
directions?* — the groundwork for comparing standard skill steering (Difference-
of-Means, LoRA/adapters) against **belief-preserving** steering.

## Inputs

A self-contained activation-dataset bundle (`activation_datasets/<name>/`,
produced by `scripts/mechinterp/build_activation_dataset.py`):

| file | used for |
|---|---|
| `activations.h5` | activation sources (`gru_h`, `enc_embed`, …), `minimap` (hover frames), `action_probs` |
| `labels.parquet` | per-timestep belief (`category`), skill (`final_commit`/`commit_state`), trajectory keys, value, … |
| `maps.npz`, `manifest.json` | terrain, tile palette, schema |

The schema is **detected from the data**: belief steps run only if a `category`
column exists; skill steps only if a commit column exists. So the same code runs
on **BT** (no belief/skill, 5 scalars) and **BTC** (belief + skill, 7 scalars).

## Run

```bash
# full BTC run (belief + skill + entanglement)  ->  W&B
python -m scripts.mechinterp.analysis.run_analysis \
    --dataset activation_datasets/btc_ppo --wandb-mode online

# BT bundle (PCA/UMAP + tables only, same code, no belief/skill)
python -m scripts.mechinterp.analysis.run_analysis \
    --dataset activation_datasets/bt_ppo --wandb-mode online

# fast local sanity check (tiny subsample, no W&B / images / UMAP)
python -m scripts.mechinterp.analysis.run_analysis \
    --dataset activation_datasets/btc_ppo --smoke
```

`--help` lists every knob. Useful ones:

| flag | default | meaning |
|---|---|---|
| `--sources gru_h enc_embed` | all discovered | which activation spaces to analyse |
| `--analysis-rows` | 30000 | master subsample (PCA/UMAP + plots + tables) |
| `--probe-rows` | 120000 | rows used to train probes / compute DoM means |
| `--projector-rows` | 2500 | W&B embedding-projector table (with hover frames) |
| `--skill-label` | `final_commit` | skill target (`final_commit` or per-step `commit_state`) |
| `--no-umap` / `--tsne` | UMAP on, t-SNE off | which DR methods |
| `--no-projector-images` | off | skip rendering hover frames (faster) |
| `--test-frac` | 0.25 | held-out **map** fraction for probe evaluation |

## What it does (per activation source)

1. **DR** — PCA (kept dims for tables, first 2 for scatter) + UMAP (+ optional t-SNE).
2. **Probes** (linear, evaluated on **held-out maps** so they can't memorise map id):
   - belief: 3-class logistic `P(balanced/lakes/rocky)` **and** an ordinal ridge probe
     on the water↔rock axis in `[-1, 1]` (rocky −1, balanced 0, lakes +1);
   - skill: 3-class logistic `P(none/build/mine)`.
   Predictions + confidences are written back into the analysis dataframe.
3. **Directions** — Difference-of-Means in raw activation space:
   belief `{lakes−rocky, lakes−balanced, rocky−balanced, ordinal-probe}`,
   skill `{build−mine, build−none, mine−none, committed−none}`.
4. **Entanglement** — cosine-similarity matrix (belief × skill), projection
   fraction of each skill direction into the belief subspace, and principal
   angles between the belief and skill subspaces.

## W&B run contents

Keyed by source (`gru_h/…`, `enc_embed/…`):

- **Tables** — `tables/timestep_metadata` (metadata + PCA/UMAP coords + probe
  preds); per-source `embedding_projector` (raw activation dims + metadata +
  **hover obs frame** → use W&B's PCA/UMAP/t-SNE projector, colour by any column);
  `tables/cosine_belief_skill`.
- **Static figures** (Goodfire-styled PNGs, **3-D** by default — set `--scatter-dims 2`
  for flat): PCA/UMAP/t-SNE scatter by category / skill / decoded-belief-score,
  class centroids with centroid path, PCA episode trajectories, **belief↔skill
  entanglement plane** (2-D by construction — a projection onto exactly the belief
  and skill directions), cosine heatmap, belief & skill confusion matrices,
  cross-source probe-accuracy bars.
- **Interactive** plotly scatters — **rotatable 3-D** (hover shows metadata),
  coloured by true map type, committed skill, and decoded belief score.
- **Summary metrics** — probe accuracy / balanced-acc / ordinal R²+Spearman,
  `cos(belief, skill)`, `proj_frac[skill→belief]`, min/mean principal angle.

## Module layout

```
analysis/
  config.py     AnalysisConfig (all knobs)
  bundle.py     ActivationBundle — load h5/labels/maps, schema detect, render frames
  style.py      Goodfire-inspired palette + matplotlib theme (shared by every fig)
  geometry.py   PCA/UMAP/t-SNE, difference-of-means, cosine/projection/principal-angles
  probes.py     grouped categorical + ordinal linear probes -> preds/proba/directions
  plots.py      matplotlib figures (return Figure; saved + logged as wandb.Image)
  wandb_io.py   plotly scatters + embedding-projector / metadata tables
  pipeline.py   orchestrator (run); run_analysis.py is the CLI
```

`geometry.py`, `probes.py`, `plots.py`, `style.py` are label-agnostic — the
*pipeline* decides which belief/skill steps exist, so adding the model-based
DreamerV3 activations or new sources later needs no change here.

## Next step (steering comparison)

The DoM directions exported here (`build−mine`, `lakes−rocky`, …) are the inputs
for the steering study: inject them via the bundle's standalone
`activation_datasets/<name>/steer.py` (`--inject vec.npy --alpha …` on `gru_h`)
and re-run this pipeline on the steered rollouts to measure whether skill steering
**moved the decoded belief** — i.e. whether belief-preserving steering is needed.
