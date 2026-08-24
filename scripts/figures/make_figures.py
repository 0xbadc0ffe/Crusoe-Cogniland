#!/usr/bin/env python3
"""One entry point for every figure, table and video in the Cogniland report.

The three agents live in three mutually incompatible Python environments, so a
single *process* cannot import all of them. This is therefore a dispatcher: it
knows, per target, which interpreter to use and what to run, and it skips work
whose outputs are already newer than their inputs.

  python scripts/figures/make_figures.py --list          # what exists, what is stale
  python scripts/figures/make_figures.py                 # build everything stale
  python scripts/figures/make_figures.py --only task env  # just those targets
  python scripts/figures/make_figures.py --force --only training
  python scripts/figures/make_figures.py --dry-run       # print commands, run nothing

Environments (see final_models/ENVIRONMENT.md):
  crusoe     conda env `crusoe`     -- env, PPO, all plotting
  r2dreamer  conda env `r2dreamer`  -- DreamerV3
  storm      STORM_model/.venv      -- STORM (runs with cwd=STORM_model)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper/figures/forkwall_paper"
CONDA_SH = "/cluster/software/anaconda3/etc/profile.d/conda.sh"

# name -> (env, argv, [outputs], [extra inputs beyond the script itself])
S = "scripts/figures"
TARGETS: dict[str, dict] = {
    # ── environment and task figures ────────────────────────────────────
    "task": dict(env="crusoe", script=f"{S}/paper_task_figs.py",
                 outputs=["fig_task_categories.png", "fig_task_anatomy.png"],
                 note="Fig 1-2  map types + anatomy of an episode"),
    "env": dict(env="crusoe", script=f"{S}/paper_env_figs.py",
                outputs=["fig_reward.png", "fig_dataset.png"],
                note="Fig 3-4  reward decomposition + dataset coverage"),
    "mapgen": dict(env="crusoe", script=f"{S}/paper_mapgen_figs.py",
                   outputs=["fig_noise_primer.png", "fig_noise_octaves.png",
                            "fig_warp.png", "fig_features.png",
                            "fig_quantile.png", "fig_pipeline.png"],
                   note="Fig 5-10 map-generation chapter"),

    # ── trajectory density (Fig 11): collect per agent, then plot ───────
    "traj-ppo": dict(env="crusoe", script=f"{S}/paper_traj_density.py",
                     args=["--agent", "ppo"], outputs=["traj_density_ppo.json"],
                     note="Fig 11 rollouts (PPO)"),
    "traj-dreamer": dict(env="r2dreamer", script=f"{S}/paper_traj_density.py",
                         args=["--agent", "dreamer"],
                         outputs=["traj_density_dreamer.json"],
                         note="Fig 11 rollouts (Dreamer)"),
    "traj-storm": dict(env="storm", script=f"{S}/paper_traj_density.py",
                       args=["--agent", "storm"],
                       outputs=["traj_density_storm.json"],
                       note="Fig 11 rollouts (STORM)"),
    "traj-plot": dict(env="crusoe", script=f"{S}/paper_traj_density.py",
                      args=["--plot-only"], outputs=["fig_trajectories.png"],
                      inputs=["traj_density_ppo.json", "traj_density_dreamer.json",
                              "traj_density_storm.json"],
                      note="Fig 11 plate"),

    # ── imagined futures (Fig 12-13) ────────────────────────────────────
    "dream-dreamer": dict(env="r2dreamer", script=f"{S}/paper_dreams.py",
                          args=["--agent", "dreamer"],
                          outputs=["fig_dream_dreamer.png", "dreams_dreamer.json"],
                          note="Fig 12  DreamerV3 imagined observations"),
    "dream-storm": dict(env="storm", script=f"{S}/paper_dreams.py",
                        args=["--agent", "storm"],
                        outputs=["fig_dream_storm.png", "dreams_storm.json"],
                        note="Fig 13  STORM imagined observations"),

    # ── training telemetry (Fig 14-17) ──────────────────────────────────
    "training-data": dict(env="crusoe", script=f"{S}/paper_training_data.py",
                          outputs=["training_data.json"],
                          note="re-read the wandb offline stores"),
    "training": dict(env="crusoe", script=f"{S}/paper_training_figs.py",
                     outputs=["fig_compare.png", "fig_ppo_training.png",
                              "fig_dreamer_training.png", "fig_storm_training.png"],
                     inputs=["training_data.json", "eval_all.json"],
                     note="Fig 14-17 per-agent training curves"),

    # ── checkpoint metastability (Fig 18) ───────────────────────────────
    "metastability": dict(env="crusoe", script=f"{S}/paper_metastability.py",
                          outputs=["fig_metastability.png", "storm_archive_eval.json"],
                          inputs=["storm_archive_eval.log"],
                          note="Fig 18  every archived STORM checkpoint"),

    # ── evidence integration (Fig 19) ───────────────────────────────────
    "evidence-ppo": dict(env="crusoe", script=f"{S}/paper_evidence_stats.py",
                         args=["--agent", "ppo"], outputs=["evidence_ppo.json"],
                         note="Fig 19 rollouts (PPO)"),
    "evidence-dreamer": dict(env="r2dreamer", script=f"{S}/paper_evidence_stats.py",
                             args=["--agent", "dreamer"],
                             outputs=["evidence_dreamer.json"],
                             note="Fig 19 rollouts (Dreamer)"),
    "evidence-storm": dict(env="storm", script=f"{S}/paper_evidence_stats.py",
                           args=["--agent", "storm"], outputs=["evidence_storm.json"],
                           note="Fig 19 rollouts (STORM)"),
    "evidence-plot": dict(env="crusoe", script=f"{S}/paper_evidence_stats.py",
                          args=["--plot"],
                          outputs=["fig_evidence.png", "evidence_stats.json"],
                          inputs=["evidence_ppo.json", "evidence_dreamer.json",
                                  "evidence_storm.json"],
                          note="Fig 19 plate + statistics"),

    # ── tables and the final document ───────────────────────────────────
    "tables": dict(env="crusoe", script=f"{S}/paper_results_table.py",
                   outputs=[], inputs=["eval_all.json"],
                   note="Tables 4-5 written into the paper source"),
    "build": dict(env="crusoe", script=f"{S}/build_paper.py",
                  outputs=[], note="inline everything -> paper/forkwall_paper.html"),
}

# targets that must not be run implicitly: they cost hours or need a cluster job
SLOW = {"training-data", "traj-ppo", "traj-dreamer", "traj-storm",
        "evidence-ppo", "evidence-dreamer", "evidence-storm",
        "dream-dreamer", "dream-storm"}

ENV_CMD = {
    "crusoe": ("source {sh} && conda activate crusoe && "
               "PYTHONPATH={repo}/src python {argv}"),
    "r2dreamer": ("source {sh} && conda activate r2dreamer && "
                  "PYTHONPATH={repo}/src:{repo}/r2dreamer_model python {argv}"),
    "storm": ("cd {repo}/STORM_model && source .venv/bin/activate && "
              "PYTHONPATH=.:..:../src python {argv}"),
}


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def status(name: str, spec: dict) -> str:
    """'missing' | 'stale' | 'ok' -- make-like, on the script and declared inputs."""
    outs = [OUT / o for o in spec["outputs"]]
    if not outs:
        return "always"
    if not all(o.exists() for o in outs):
        return "missing"
    newest_in = mtime(REPO / spec["script"])
    for i in spec.get("inputs", []):
        newest_in = max(newest_in, mtime(OUT / i))
    return "stale" if newest_in > min(mtime(o) for o in outs) else "ok"


def run(name: str, spec: dict, dry: bool) -> bool:
    argv = " ".join([str(REPO / spec["script"])] + spec.get("args", []))
    cmd = ENV_CMD[spec["env"]].format(sh=CONDA_SH, repo=REPO, argv=argv)
    print(f"\n\033[1m── {name}\033[0m  [{spec['env']}]  {spec['note']}")
    print(f"   $ {cmd}")
    if dry:
        return True
    t0 = time.time()
    r = subprocess.run(["bash", "-lc", cmd], cwd=REPO)
    ok = r.returncode == 0
    print(f"   {'ok' if ok else 'FAILED'} in {time.time()-t0:.1f}s")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="TARGET",
                   help="build just these targets (see --list)")
    p.add_argument("--list", action="store_true", help="show targets and staleness")
    p.add_argument("--force", action="store_true", help="rebuild even if up to date")
    p.add_argument("--dry-run", action="store_true", help="print commands only")
    p.add_argument("--include-slow", action="store_true",
                   help="also run the agent-rollout targets, which take hours")
    a = p.parse_args()

    if a.list:
        print(f"{'target':18s} {'env':10s} {'status':8s} note")
        print("-" * 84)
        for n, s in TARGETS.items():
            st = status(n, s)
            mark = {"ok": "\033[32m", "stale": "\033[33m",
                    "missing": "\033[31m"}.get(st, "\033[90m")
            slow = " (slow)" if n in SLOW else ""
            print(f"{n:18s} {s['env']:10s} {mark}{st:8s}\033[0m {s['note']}{slow}")
        print(f"\noutputs -> {OUT}")
        return

    names = a.only or [n for n in TARGETS if n not in SLOW or a.include_slow]
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        sys.exit(f"unknown target(s): {', '.join(unknown)}\nsee --list")

    todo = [n for n in names
            if a.force or a.only or status(n, TARGETS[n]) in ("missing", "stale", "always")]
    skipped = [n for n in names if n not in todo]
    if skipped:
        print(f"up to date, skipping: {', '.join(skipped)}")
    if not todo:
        print("nothing to do")
        return

    failed = [n for n in todo if not run(n, TARGETS[n], a.dry_run)]
    print("\n" + "=" * 60)
    print(f"built {len(todo) - len(failed)}/{len(todo)} targets")
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
