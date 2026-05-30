# zebra_nav — released agents

Frozen PPO+GRU policies for the `cogniland.zebra_nav` env. Each `*.pt` is a dict
`{"policy": state_dict, "args": {...}, "iteration", "global_step"}`; the matching
`*.yaml` is the **exact, reproducible** trainer config (hyperparameters + seed)
that produced it.

| file | map type | what it does | success | episode len |
|------|----------|--------------|---------|-------------|
| `diagonal_cuefollower.pt` | diagonal stripes (32×32, BL→TR) | reads the cue, crosses the **thin** side of every obsidian wall | 100% (greedy & stochastic) | ~86 |
| `vertical_cuefollower.pt` | vertical stripes (32×64, midL→midR) | same cue-following, vertical walls | 100% | ~85 |
| `natural_agent.pt` | natural (32×64, lakes/mountains/trees) | **diverse** routes through the middle to the goal door, crossing some obstacles (mine/bridge) and going around others | 100% | ~95 |

`natural_agent` is the diverse one: a **central goal door** keeps it traversing
the centre (no edge-hugging), and **no entropy annealing** + entropy 0.045 keeps
the stochastic policy spread out → many distinct paths.

## See it play (pygame demo)

```bash
# AI plays the natural agent on the curated validation maps
python scripts/play_zebra.py --checkpoint models/zebra_nav/natural_agent.pt \
    --maps data/zebra_nav/val_maps.pkl
# or human-play the same maps (relative controls: ←/→ turn, ↑ forward, B build, M mine)
python scripts/play_zebra.py --maps data/zebra_nav/val_maps.pkl --action-mode relative
```
Keys: `A` toggle AI, `Space` single AI step, `+/-` speed, `R` next map, `Q` quit.
The demo reads the map size / orientation / action-mode from the checkpoint, so
no extra flags are needed for AI play.

## Reproduce

```bash
python scripts/train_ppo_zebra.py --config models/zebra_nav/natural_agent.yaml \
    --run-name natural_repro --wandb-mode disabled
```
(Same for the other two configs.) All use the agreed 2M-step budget. Diagonal/
vertical reproduce reliably; the natural agent's diversity comes from high
entropy + no annealing.

## Evaluate

```bash
python scripts/eval_zebra_agent.py  --checkpoint models/zebra_nav/natural_agent.pt --n-maps 8
python scripts/zebra_traj_grid.py   --checkpoint models/zebra_nav/natural_agent.pt --n-maps 6 --n-traj 200
```

See `docs/zebra_nav.md` for the full project guide.
