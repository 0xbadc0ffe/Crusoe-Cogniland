# Reproducing the IAB paper's figures and Table 1

All commands run from the repo root in the `crusoe` conda env (CPU unless noted).
Agent: `final_models/ppo/ppo_plain_noaux.pt`, the belief-free PPO+GRU. Maps: `data/bridge_tunnel/forkwall6k/`.

| Step | Command | Writes | Used by |
|---|---|---|---|
| 0 Maps | `python scripts/bridge_tunnel/make_forkwall_dataset.py` | `data/bridge_tunnel/forkwall6k/{train,test}.pkl` | everything |
| 0 Agent (GPU) | `sbatch scripts/bridge_tunnel/slurm/train_ff6m_control.sbatch` without `--no-recurrence`, seeds 1-5 | `outputs/ppo_noaux/*` (best seed promoted to `final_models/ppo/ppo_plain_noaux.pt`) | everything |
| 1 Activation dataset | `PYTHONPATH=src python scripts/mechinterp/build_belief_dataset.py --agent ppo` | `activation_datasets/cogniland_belief/ppo_*` | Fig. 3, Table 1 |
| 2 Belief direction | `PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report python scripts/mechinterp/belief_report/steer_belief.py --agent ppo --export-axis` | `outputs/belief_report/steer_axis_ppo.npz` | Fig. 3c, Fig. 4, Table 1 |
| 3 Probes | `python scripts/mechinterp/belief_report/probes.py` | `outputs/belief_report/probes.json` | Fig. 3a-b |
| 4 Dose response | `PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report python scripts/mechinterp/belief_report/steer_alpha.py --agent ppo --n 100 --workers 44 --alphas 0,0.1,...,2.0 --control --tag finectrl` (or `sbatch scripts/mechinterp/belief_report/steer_alpha_fine.sbatch`, ~3 min on 48 CPUs; `--control` adds the matched random-direction curves) | `outputs/belief_report/steer_alpha_ppo_corr2_finectrl.json` | Fig. 4 |
| 5 Steering, all eligible maps | `sbatch scripts/mechinterp/behavior_steering/slurm/act11_all_eligible.sbatch` | `outputs/behavior_steering/act11/{rows,summary}_all_{balanced,lakes,rocky}.json` | Table 1 |
| 6 Figures + table | `PYTHONPATH=src:scripts/mechinterp/belief_report python scripts/figures/paper/iab_appendix_figs.py belief steer table` | `paper/iab2026/paper/figures/fig_results_belief.png`, `fig_results_causal.png`, `tab_clamp_all_eligible.tex` | paper |

Appendix figures come from the same script (`dataset bins pca pca_categories`), the training curves from
`scripts/figures/paper/iab_training_curves.py`, and the crossing-plan figure from
`scripts/figures/paper/iab_crossing_plan.py` (launcher `scripts/bridge_tunnel/slurm/iab_crossing_plan.sbatch`).

Notes. Step 5 uses the operating points in `outputs/behavior_steering/act5/operating_points.json` and the clamp in
`src/cogniland/bridge_tunnel/steering.py`; it screens all 400 held-out maps per category and keeps those where two
unsteered rollouts used both tools. Step 6's 1-D logistic (Fig. 3c and the last column of Table 1) is fitted on the
fit split of held-out lakes and rocky maps defined in `scripts/mechinterp/belief_report/data.py`. Every rollout is
seeded, so each stage reproduces exactly.
