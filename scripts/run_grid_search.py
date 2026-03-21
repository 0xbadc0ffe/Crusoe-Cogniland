#!/usr/bin/env python3
"""Run a grid search over reward parameters and push a WandB Report.

Example:
    python scripts/run_grid_search.py --workers 2
    python scripts/run_grid_search.py --test-mode
"""

import argparse
import itertools
import subprocess
import time
from datetime import datetime
import sys

try:
    import wandb
    import wandb.apis.reports as wr
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb or wandb.apis.reports is not installed. Report generation will be skipped.")

# ---------------------------------------------------------------------------
# Grid Definition
# ---------------------------------------------------------------------------
# Modify these lists to search over different dimensions.
GRID = {
    "lambda_p": [0.1, 0.3],     # Progress weight
    "lambda_t": [30.0, 60.0],   # Time-efficiency weight
    "lambda_d": [0.3, 0.6],     # Death penalty weight
}

def generate_commands(group_name: str, test_mode: bool) -> list[tuple[str, list[str]]]:
    """Generate the subprocess commands for all grid combinations."""
    keys = list(GRID.keys())
    values = list(GRID.values())
    combinations = list(itertools.product(*values))

    commands = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        name = f"run_p{params['lambda_p']}_t{params['lambda_t']}_d{params['lambda_d']}"

        # Setup the base command using Hydra overrides (+ for appending new keys)
        cmd = [
            "python", "train.py",
            f"+logging.wandb.group={group_name}",
            f"+logging.wandb.name={name}",
        ]
        
        # Apply reward overrides
        for k, v in params.items():
            cmd.append(f"env.reward.{k}={v}")

        if test_mode:
            print(f"Adding test mode overrides for {name}")
            cmd.extend([
                "models.training.total_env_moves=1000",
                "models.training.eval_every_n_updates=1",
                "models.training.moves_per_rollout=500"
            ])

        commands.append((name, cmd))

    return commands

def create_wandb_report(project: str, entity: str, group_name: str):
    """Generate a native WandB Report for the completed grid search."""
    if not WANDB_AVAILABLE:
        print("WandB API not available, skipping report.")
        return

    print(f"\nGenerating WandB Report for group: {group_name}")
    api = wandb.Api()

    report = wr.Report(
        project=project,
        entity=entity,
        title=f"Grid Search: Behavioral Metrics ({group_name})",
        description="Exploration of different reward function weights and their effect on agent behavior."
    )

    # Use a runset to filter out runs belonging to this exact group.
    runset = wr.Runset()
    if entity:
        runset.entity = entity
    runset.project = project
    runset.filters = f"group == '{group_name}'"

    # Create various Scatter Plots allowing cross-filtering interactions.
    report.blocks = [
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                wr.ScatterPlot(
                    title="Directness vs Return",
                    x="test_det/env/directness_mean",
                    y="test_det/env/return_mean",
                ),
                wr.ScatterPlot(
                    title="Risk Exposure vs Directness",
                    x="test_det/env/risk_exposure_mean",
                    y="test_det/env/directness_mean",
                ),
                wr.ScatterPlot(
                    title="Danger Fraction vs Exploration",
                    x="test_det/env/danger_fraction_mean",
                    y="test_det/env/exploration_mean",
                ),
                wr.ScatterPlot(
                    title="Success Rate vs Return",
                    x="test_det/env/success_rate",
                    y="test_det/env/return_mean", 
                ),
            ]
        ),
        wr.P("Note: Selecting or brushing over a model in one layout natively highlights that run in all corresponding charts."),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                wr.RunComparer(
                    # A parallel coordinates comparing parameters to outcomes
                    diff_only="split"
                )
            ]
        )
    ]
    
    try:
        report.save()
        print(f"\n✅ WandB Report uploaded successfully!")
        print(f"👉 View it here: {report.url}")
    except Exception as e:
        print(f"Failed to push WandB Report. Make sure you are logged in using `wandb login`: {e}")
        print(f"You can manually view your runs grouped by '{group_name}' in the W&B Workspace.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent training runs.")
    parser.add_argument("--test-mode", action="store_true", help="Run a fast, short training purely to test the pipeline.")
    parser.add_argument("--project", type=str, default="cogniland", help="WandB project name")
    parser.add_argument("--entity", type=str, default="crusoe", help="WandB entity name")
    args = parser.parse_args()

    group_name = f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.test_mode:
        group_name += "_TEST"

    commands = generate_commands(group_name, args.test_mode)
    print(f"Prepared {len(commands)} configurations for grid search.")
    print(f"Starting execution with {args.workers} workers...")

    running_procs = []
    start_time = time.time()
    cmd_idx = 0

    try:
        while cmd_idx < len(commands) or running_procs:
            # Start new processes if we have capacity
            while len(running_procs) < args.workers and cmd_idx < len(commands):
                name, cmd = commands[cmd_idx]
                print(f"[{cmd_idx+1}/{len(commands)}] Launching {name} ...")
                # Using subprocess.run would block. Popen is concurrent.
                proc = subprocess.Popen(cmd)
                running_procs.append((name, proc))
                cmd_idx += 1

            # Check for finished processes
            time.sleep(1.0)
            for name, proc in running_procs[:]:
                ret = proc.poll()
                if ret is not None:
                    status = "✅ Finished" if ret == 0 else "❌ Failed"
                    print(f"{status}: {name} (exit code {ret})")
                    running_procs.remove((name, proc))
                    
    except KeyboardInterrupt:
        print("\nGrid Search interrupted by user! Terminating running processes...")
        for name, proc in running_procs:
            proc.terminate()
        sys.exit(1)

    duration = time.time() - start_time
    print(f"\nAll training runs completed in {duration:.1f} seconds.")

    # Always attempt to generate the report!
    create_wandb_report(args.project, args.entity, group_name)

if __name__ == "__main__":
    main()
