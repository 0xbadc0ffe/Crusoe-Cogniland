# Paper figures

One module per figure group. **You almost never need to open these files.**

## To change a title, label, legend entry or annotation

Edit **`text.py`** — it holds every word that appears inside every figure,
nothing else. Then rebuild just that figure:

```bash
python scripts/figures/make_figures.py --only task     # after editing FIG01 / FIG02
python scripts/figures/make_figures.py --list          # target → figure map
```

## To change a colour, font size or DPI everywhere

Edit **`style.py`**. It owns the agent palette, the semantic colours
(correct / wrong / timeout), the category colours, the matplotlib rc, and the
shared maths (`smooth`, `wilson`, `xy`) that used to be copy-pasted into five
scripts with three different sets of values.

## To change what is actually drawn

Then you open a module.

| module | figures | what it draws |
|---|---|---|
| `fig_task.py` | 1–2 | the three map types; anatomy of one episode |
| `fig_env.py` | 3–5 | observation space; reward decomposition; dataset coverage |
| `fig_mapgen.py` | 6–11 | the map-generation chapter (noise → warp → quantile → pipeline) |
| `fig_trajectories.py` | 12 | 24 stochastic rollouts per panel |
| `fig_dreams.py` | 13–14 | imagined-observation filmstrips (Dreamer, STORM) |
| `fig_training.py` | 15, 17–19 | three-agent comparison; per-agent training detail |
| `fig_evidence.py` | 16 | what the agent saw vs what it chose |
| `fig_metastability.py` | 20 | held-out success of every archived checkpoint |

Figure numbers are the ones in the report; `make_figures.py --list` is the
authoritative mapping and stays in sync because the dispatcher owns it.

## Things that are not figures

These stay one level up in `scripts/figures/` because they are data or output,
not plots: `paper_rollouts.py` (agent adapters, imported by several modules
here), `paper_rollouts_textured.py` (videos), `paper_training_data.py` (reads
the wandb offline stores), `paper_eval_all.py` (the evaluation harness),
`paper_results_table.py` (writes Tables 4–5 into the paper source),
`build_paper.py` (inlines everything into the final HTML).

## Environments

Three agents, three incompatible Python environments. `make_figures.py` knows
which target needs which and dispatches; you should not need to activate
anything by hand. If you run a module directly, check the docstring header for
the right `PYTHONPATH`.
