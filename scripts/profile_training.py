#!/usr/bin/env python
"""Profile the training loop to identify GPU bottlenecks.

Usage:
    python scripts/profile_training.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import numpy as np

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cogniland.env.constants import ACTION_DELTAS
from cogniland.env.core import (
    apply_movement,
    apply_movement_costs,
    apply_terrain_effects,
    compute_minimap_batch,
    compute_reward,
    compute_terrain_levels,
)
from cogniland.env.dataset import MapDataset
from cogniland.env.pathfinding import batch_dijkstra_from_sources, batch_reverse_dijkstra
from cogniland.env.types import EnvConfig, EnvState
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.models.ppo import ActorCritic


# ── Config ─────────────────────────────────────────────────────────────────
NUM_ENVS = 320
WARMUP_STEPS = 50
PROFILE_STEPS = 100

# ppo_1m architecture
MODEL_KWARGS = dict(
    cnn_channels=64,
    cnn_out_spatial=5,
    scalar_hidden=128,
    hidden_dim=448,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def cuda_sync():
    torch.cuda.synchronize()


def gpu_timer():
    """Return (start, stop) CUDA events for precise GPU timing."""
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    return start, stop


def print_header():
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    mem_alloc = torch.cuda.memory_allocated(dev) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(dev) / 1024**2
    print(f"\n{'=' * 72}")
    print(f"  CUDA Device: {name}")
    print(f"  Memory — allocated: {mem_alloc:.1f} MB, reserved: {mem_reserved:.1f} MB")
    print(f"  Profiling {NUM_ENVS} parallel envs, {WARMUP_STEPS} warmup + {PROFILE_STEPS} profiled steps")
    print(f"{'=' * 72}\n")


def print_table(title: str, rows: list[tuple[str, float, float, float | None]]):
    """Print a formatted table. Rows: (label, ms/step, % of total, SPS or None)."""
    print(f"\n┌─ {title}")
    print(f"│ {'Component':<40} {'ms/step':>10} {'% total':>10} {'SPS':>12}")
    print(f"│ {'─' * 40} {'─' * 10} {'─' * 10} {'─' * 12}")
    for label, ms, pct, sps in rows:
        sps_str = f"{sps:>10.0f}" if sps is not None else f"{'—':>10}"
        print(f"│ {label:<40} {ms:>10.3f} {pct:>9.1f}% {sps_str}")
    print(f"└{'─' * 75}\n")


# ── Setup ──────────────────────────────────────────────────────────────────

def setup():
    """Create env, model, and return initial obs."""
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    config = EnvConfig(device=device)

    # Load real training maps
    dataset = MapDataset.from_split_files(
        train_path=ROOT / "data" / "train_seed42_n128.pt",
        val_path=ROOT / "data" / "val_seed42_n16.pt",
        test_path=ROOT / "data" / "test_seed42_n16.pt",
    )
    print(f"Loaded dataset: {dataset.n_train} train maps, {dataset.map_size}x{dataset.map_size}")

    env = BatchedIslandEnv(config, NUM_ENVS, world_maps=dataset.train_maps)
    model = ActorCritic(**MODEL_KWARGS).to(device)
    model.eval()

    obs = env.reset(seed=42)

    print_header()
    dev = torch.cuda.current_device()
    mem_alloc = torch.cuda.memory_allocated(dev) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(dev) / 1024**2
    print(f"  Post-setup memory — allocated: {mem_alloc:.1f} MB, reserved: {mem_reserved:.1f} MB\n")

    return env, model, obs, config, device


# ── 1. Model vs Env split ────────────────────────────────────────────────

def profile_model_vs_env(env, model, obs):
    """Time model.get_action_and_value() vs env.step() per step."""
    print("Profiling model vs env split ...")

    # Warmup
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs)
        obs, _, _, _ = env.step(action)

    model_ms_total = 0.0
    env_ms_total = 0.0

    for _ in range(PROFILE_STEPS):
        # Model forward
        cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs)
        cuda_sync()
        t1 = time.perf_counter()

        # Env step
        obs, _, _, _ = env.step(action)
        cuda_sync()
        t2 = time.perf_counter()

        model_ms_total += (t1 - t0) * 1000
        env_ms_total += (t2 - t1) * 1000

    total_ms = model_ms_total + env_ms_total
    model_ms = model_ms_total / PROFILE_STEPS
    env_ms = env_ms_total / PROFILE_STEPS
    total_per_step = total_ms / PROFILE_STEPS
    sps = PROFILE_STEPS * NUM_ENVS / (total_ms / 1000)

    print_table("Model vs Environment", [
        ("model.get_action_and_value()", model_ms, model_ms / total_per_step * 100, None),
        ("env.step()", env_ms, env_ms / total_per_step * 100, None),
        ("TOTAL per step", total_per_step, 100.0, sps),
    ])

    return obs


# ── 2. Double vs single forward pass ────────────────────────────────────

def profile_double_vs_single(model, obs):
    """Compare get_value+get_action_and_value (current code) vs just get_action_and_value."""
    print("Profiling double vs single forward pass ...")

    # Warmup both paths
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            _ = model.get_value(obs)
            _ = model.get_action_and_value(obs)

    # Double pass (current _collect_rollout pattern)
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(PROFILE_STEPS):
        with torch.no_grad():
            _ = model.get_value(obs)
            _ = model.get_action_and_value(obs)
    cuda_sync()
    t1 = time.perf_counter()
    double_ms = (t1 - t0) * 1000 / PROFILE_STEPS

    # Single pass (just get_action_and_value, which returns value too)
    cuda_sync()
    t2 = time.perf_counter()
    for _ in range(PROFILE_STEPS):
        with torch.no_grad():
            _ = model.get_action_and_value(obs)
    cuda_sync()
    t3 = time.perf_counter()
    single_ms = (t3 - t2) * 1000 / PROFILE_STEPS

    overhead_ms = double_ms - single_ms
    overhead_pct = overhead_ms / double_ms * 100 if double_ms > 0 else 0

    print_table("Double vs Single Forward Pass", [
        ("get_value + get_action_and_value", double_ms, 100.0, None),
        ("get_action_and_value only", single_ms, single_ms / double_ms * 100, None),
        ("Redundant get_value overhead", overhead_ms, overhead_pct, None),
    ])


# ── 3. Env step internals ───────────────────────────────────────────────

def profile_env_internals(env, model, obs, config, device):
    """Break down env_step into its sub-components."""
    print("Profiling env step internals ...")

    compiled = env.compiled

    # Warmup and get a valid state
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs)
        obs, _, _, _ = env.step(action)

    # We'll manually call the sub-functions of env_step with real state
    # Use the env's internal state directly
    islands = env.env

    timings = {
        "movement": 0.0,
        "compass": 0.0,
        "terrain_lookup": 0.0,
        "minimap_total": 0.0,
        "costs_effects": 0.0,
        "ctg_update": 0.0,
        "terminal_check": 0.0,
    }

    for step_i in range(PROFILE_STEPS):
        state = env.state
        target_pos = env.target_pos
        per_env_maps = islands.world_maps[islands._env_map_idx]
        cost_to_go_maps = islands._cost_to_go_maps

        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs)

        old_terrain = state.terrain_idx.clone()
        old_ctg = state.cost_to_go.clone()

        # 1. Movement
        cuda_sync()
        t0 = time.perf_counter()
        new_state = apply_movement(state, action, config.size)
        cuda_sync()
        t1 = time.perf_counter()
        timings["movement"] += (t1 - t0) * 1000

        # 2. Compass
        cuda_sync()
        t0 = time.perf_counter()
        import math
        compass_raw = (target_pos - new_state.position).float()
        compass_euclidean = torch.norm(compass_raw, dim=1, keepdim=True).clamp(min=1e-8)
        compass_unit = compass_raw / compass_euclidean
        max_rad = islands.compass_noise_deg * math.pi / 180.0
        if max_rad > 0:
            theta = (torch.rand(compass_unit.shape[0], device=device) * 2.0 - 1.0) * max_rad
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            x = compass_unit[:, 0] * cos_t - compass_unit[:, 1] * sin_t
            y = compass_unit[:, 0] * sin_t + compass_unit[:, 1] * cos_t
            compass_unit = torch.stack([x, y], dim=1)
        new_state = new_state._replace(compass=compass_unit)
        cuda_sync()
        t1 = time.perf_counter()
        timings["compass"] += (t1 - t0) * 1000

        # 3. Terrain lookup
        cuda_sync()
        t0 = time.perf_counter()
        terrain_idx = compute_terrain_levels(per_env_maps, new_state.position, compiled)
        new_state = new_state._replace(terrain_idx=terrain_idx)
        cuda_sync()
        t1 = time.perf_counter()
        timings["terrain_lookup"] += (t1 - t0) * 1000

        # 4. Minimap (now fully vectorized — no Python loops)
        cuda_sync()
        t0 = time.perf_counter()
        minimap = compute_minimap_batch(
            per_env_maps, new_state.position, config.minimap_max_ray,
            terrain_idx, config.minimap_occlude, config.minimap_clear_tolerance,
            compiled, target_pos=target_pos,
        )
        cuda_sync()
        t1 = time.perf_counter()
        timings["minimap_total"] += (t1 - t0) * 1000
        new_state = new_state._replace(minimap=minimap)

        # 5. Costs + terrain effects
        cuda_sync()
        t0 = time.perf_counter()
        new_state = apply_movement_costs(new_state, action, config, compiled)
        new_state = apply_terrain_effects(new_state, old_terrain, action, config, compiled)
        hp = torch.clamp(new_state.hp, 0.0, config.max_hp)
        resources = torch.clamp(new_state.resources, 0.0, config.max_resources)
        new_state = new_state._replace(hp=hp, resources=resources)
        cuda_sync()
        t1 = time.perf_counter()
        timings["costs_effects"] += (t1 - t0) * 1000

        # 6. Cost-to-go update
        cuda_sync()
        t0 = time.perf_counter()
        B = new_state.position.shape[0]
        b_idx = torch.arange(B, device=device)
        new_ctg = cost_to_go_maps[b_idx, new_state.position[:, 0], new_state.position[:, 1]]
        new_state = new_state._replace(cost_to_go=new_ctg)
        cuda_sync()
        t1 = time.perf_counter()
        timings["ctg_update"] += (t1 - t0) * 1000

        # 7. Terminal + reward
        cuda_sync()
        t0 = time.perf_counter()
        alive = new_state.hp > 0
        dist_to_target = (new_state.position - target_pos).float().abs().sum(dim=1)
        reached = dist_to_target < 1.0
        done = ~alive | reached
        ctg_delta = old_ctg - new_state.cost_to_go
        reward = compute_reward(ctg_delta, new_state.cost, new_state.dijkstra_cost,
                                alive, reached, config.reward)
        cuda_sync()
        t1 = time.perf_counter()
        timings["terminal_check"] += (t1 - t0) * 1000

        # Actually advance the env for next iteration
        obs, _, _, _ = env.step(action)

    # Compute averages
    for k in timings:
        timings[k] /= PROFILE_STEPS

    env_total = sum(timings.values())
    sps_total = NUM_ENVS / (env_total / 1000) if env_total > 0 else 0

    rows = [
        ("apply_movement()", timings["movement"], timings["movement"] / env_total * 100, None),
        ("compass update", timings["compass"], timings["compass"] / env_total * 100, None),
        ("compute_terrain_levels()", timings["terrain_lookup"], timings["terrain_lookup"] / env_total * 100, None),
        ("compute_minimap_batch() [vectorized]", timings["minimap_total"], timings["minimap_total"] / env_total * 100, None),
        ("costs + terrain effects + clamp", timings["costs_effects"], timings["costs_effects"] / env_total * 100, None),
        ("cost-to-go map lookup", timings["ctg_update"], timings["ctg_update"] / env_total * 100, None),
        ("terminal check + reward", timings["terminal_check"], timings["terminal_check"] / env_total * 100, None),
        ("ENV STEP TOTAL (summed)", env_total, 100.0, sps_total),
    ]
    print_table("Env Step Internals", rows)

    return obs


# ── 4. Dijkstra reset cost ───────────────────────────────────────────────

def profile_dijkstra(env, config, device):
    """Profile batch_dijkstra_from_sources + batch_reverse_dijkstra for varying batch sizes."""
    print("Profiling Dijkstra reset cost ...")

    compiled = env.compiled
    islands = env.env
    world_maps_gpu = islands.world_maps  # [N, H, W] on CUDA
    land_threshold = compiled.land_threshold

    batch_sizes = [5, 10, 20, 40]
    results = []

    for B in batch_sizes:
        # Sample random maps + positions for this batch
        N = world_maps_gpu.shape[0]
        map_idx = torch.randint(0, N, (B,), device=device)
        batch_maps = world_maps_gpu[map_idx]  # [B, H, W]

        # Sample land positions
        size = config.size
        spawns = torch.zeros(B, 2, dtype=torch.long, device=device)
        targets = torch.zeros(B, 2, dtype=torch.long, device=device)
        for b in range(B):
            m = batch_maps[b]
            for pos_out in [spawns, targets]:
                while True:
                    p = torch.randint(0, size, (2,), device=device)
                    if m[p[0], p[1]].item() > land_threshold:
                        pos_out[b] = p
                        break

        # Warmup (1 run)
        _ = batch_dijkstra_from_sources(
            batch_maps.cpu(), compiled.move_costs.cpu(), spawns.cpu(),
            terrain_thresholds=compiled.thresholds.cpu(),
        )
        _ = batch_reverse_dijkstra(
            batch_maps.cpu(), compiled.move_costs.cpu(),
            compiled.thresholds.cpu(), compiled.is_water.cpu(),
            targets.cpu(), beta_raft=config.reward.beta_raft,
            res_rates=compiled.res_rate.cpu(),
        )

        # Profile (3 runs, take mean)
        n_runs = 3
        fwd_times = []
        rev_times = []
        for _ in range(n_runs):
            cuda_sync()
            t0 = time.perf_counter()
            _ = batch_dijkstra_from_sources(
                batch_maps.cpu(), compiled.move_costs.cpu(), spawns.cpu(),
                terrain_thresholds=compiled.thresholds.cpu(),
            )
            t1 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1000)

            t2 = time.perf_counter()
            _ = batch_reverse_dijkstra(
                batch_maps.cpu(), compiled.move_costs.cpu(),
                compiled.thresholds.cpu(), compiled.is_water.cpu(),
                targets.cpu(), beta_raft=config.reward.beta_raft,
                res_rates=compiled.res_rate.cpu(),
            )
            t3 = time.perf_counter()
            rev_times.append((t3 - t2) * 1000)

        fwd_ms = np.mean(fwd_times)
        rev_ms = np.mean(rev_times)
        total_ms = fwd_ms + rev_ms
        per_env_ms = total_ms / B

        results.append((B, fwd_ms, rev_ms, total_ms, per_env_ms))

    rows = []
    for B, fwd_ms, rev_ms, total_ms, per_env_ms in results:
        rows.append((f"B={B:>2} forward Dijkstra", fwd_ms, fwd_ms / total_ms * 100, None))
        rows.append((f"B={B:>2} reverse Dijkstra", rev_ms, rev_ms / total_ms * 100, None))
        rows.append((f"B={B:>2} TOTAL ({per_env_ms:.1f} ms/env)", total_ms, 100.0, None))
        rows.append(("", 0, 0, None))  # spacer

    # Remove trailing spacer
    rows = rows[:-1]
    print_table("Dijkstra Reset Cost (250x250 maps)", rows)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    env, model, obs, config, device = setup()

    obs = profile_model_vs_env(env, model, obs)
    profile_double_vs_single(model, obs)
    obs = profile_env_internals(env, model, obs, config, device)
    profile_dijkstra(env, config, device)

    # Final memory report
    dev = torch.cuda.current_device()
    mem_alloc = torch.cuda.memory_allocated(dev) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(dev) / 1024**2
    peak_mem = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(f"\nFinal memory — allocated: {mem_alloc:.1f} MB, reserved: {mem_reserved:.1f} MB, peak: {peak_mem:.1f} MB")


if __name__ == "__main__":
    main()
