#!/usr/bin/env python3
"""
Launch multiple W&B agents in parallel for a sweep.

This script launches N agents in parallel, optionally distributed across multiple GPUs.
Each agent will process sweep runs until the sweep is complete or --count limit is reached.

Usage:
    # Launch 5 agents on single GPU (default GPU 0)
    python scripts/run_sweep.py <sweep_id> --num-agents 5

    # Launch 5 agents with 1 run each (for K=5 benchmark)
    python scripts/run_sweep.py <sweep_id> --num-agents 5 --count 1

    # Launch across multiple GPUs
    python scripts/run_sweep.py <sweep_id> --num-agents 8 --gpus 0 1
sq
    # Specify runs per agent
    python scripts/run_sweep.py <sweep_id> --num-agents 10 --count 20
"""

import argparse
import subprocess
import time
import os
import sys
import signal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch multiple W&B agents in parallel for a sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # K=5 benchmark (5 agents, 1 run each)
  python scripts/run_sweep.py entity/project/sweep123 --num-agents 5 --count 1

  # Run 40 experiments across 8 GPUs (5 per GPU)
  python scripts/run_sweep.py entity/project/sweep123 --num-agents 40 --gpus 0 1 2 3 4 5 6 7

  # Unlimited runs until sweep completes
  python scripts/run_sweep.py entity/project/sweep123 --num-agents 5
        """
    )

    parser.add_argument(
        "sweep_id",
        type=str,
        help="W&B sweep ID (format: entity/project/sweep_id or just sweep_id)",
    )

    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="Number of parallel agents to launch (default: 5)",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs per agent (default: unlimited, runs until sweep completes)",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="GPU indices to use (default: 0). Agents will be distributed round-robin.",
    )

    parser.add_argument(
        "--stagger",
        type=float,
        default=2.0,
        help="Seconds to wait between launching agents (default: 2.0)",
    )

    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory to run agents from (default: current directory)",
    )

    return parser.parse_args()


def launch_agent(
    sweep_id: str,
    gpu_id: int,
    agent_id: int,
    count: int = None,
    project_dir: str = ".",
) -> subprocess.Popen:
    """
    Launch a single W&B agent process.

    Args:
        sweep_id: W&B sweep ID
        gpu_id: GPU index to use
        agent_id: Unique ID for this agent (for logging)
        count: Number of runs for this agent (None = unlimited)
        project_dir: Directory to run from

    Returns:
        subprocess.Popen object
    """
    cmd = ["wandb", "agent"]

    if count is not None:
        cmd.extend(["--count", str(count)])

    cmd.append(sweep_id)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"→ Agent {agent_id}: GPU {gpu_id}, {count if count else 'unlimited'} runs")

    # Use process group to ensure all child processes are killed together
    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,  # Create new process group
    )

    return process


def main():
    args = parse_args()

    print("=" * 80)
    print("W&B Sweep: Parallel Agent Launcher")
    print("=" * 80)
    print(f"Sweep ID: {args.sweep_id}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Runs per agent: {args.count if args.count else 'unlimited'}")
    print(f"GPUs: {args.gpus}")
    print(f"Stagger: {args.stagger}s")
    print("=" * 80)

    processes = []

    # Launch agents
    for i in range(args.num_agents):
        # Round-robin GPU assignment
        gpu_id = args.gpus[i % len(args.gpus)]

        process = launch_agent(
            sweep_id=args.sweep_id,
            gpu_id=gpu_id,
            agent_id=i,
            count=args.count,
            project_dir=args.project_dir,
        )

        processes.append((i, gpu_id, process))

        # Stagger launches
        if i < args.num_agents - 1:
            time.sleep(args.stagger)

    print(f"\n✓ Launched {len(processes)} agents")
    print("Waiting for agents to complete...\n")

    # Wait for all processes
    try:
        exit_codes = []
        for agent_id, gpu_id, process in processes:
            returncode = process.wait()
            exit_codes.append(returncode)

            if returncode == 0:
                print(f"✓ Agent {agent_id} (GPU {gpu_id}) completed successfully")
            else:
                print(f"✗ Agent {agent_id} (GPU {gpu_id}) failed with code {returncode}")
                # Optionally print stderr
                stderr = process.stderr.read() if process.stderr else ""
                if stderr:
                    print(f"  Error: {stderr[:200]}")

        print("\n" + "=" * 80)
        print("All agents completed")
        print(f"Successful: {sum(1 for code in exit_codes if code == 0)}/{len(exit_codes)}")
        print("=" * 80)

        # Exit with error if any agent failed
        sys.exit(0 if all(code == 0 for code in exit_codes) else 1)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! Terminating agents and all child processes...")

        # First, try to terminate process groups gracefully
        for agent_id, gpu_id, process in processes:
            try:
                # Kill entire process group (agent + all its children)
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Process already terminated or no permission
                pass

        # Wait for graceful termination
        print("Waiting for graceful shutdown...")
        time.sleep(3)

        # Force kill entire process groups if still running
        for agent_id, gpu_id, process in processes:
            if process.poll() is None:
                try:
                    print(f"Force killing agent {agent_id} (PID {process.pid})...")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        # Nuclear option: find and kill all train.py processes we might have spawned
        print("Cleaning up any remaining train.py processes...")
        try:
            result = subprocess.run(
                ["pgrep", "-f", "train.py.*--sweep"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                orphan_pids = result.stdout.strip().split('\n')
                print(f"Found {len(orphan_pids)} orphaned train.py processes, killing them...")
                for pid in orphan_pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except Exception as e:
            print(f"Warning: Could not clean up orphans: {e}")

        print("✓ All agents and child processes terminated")
        sys.exit(1)


if __name__ == "__main__":
    main()
