"""TrajectoryCollector — runs a trained model on maps and collects rich data.

Captures per-step: observations, neural activations (via hooks), env state,
actions, rewards, value estimates, action probabilities, and contextual flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from cogniland.env.core import compute_minimap_batch, compute_terrain_levels
from cogniland.env.types import EnvConfig, EnvState
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.models.ppo import ActorCritic


# ── Layers to hook and how to store them ─────────────────────────────────────

# key → (module path, storage mode)
#   "full"   = store as-is (after detach+cpu)
#   "pool"   = global-average-pool spatial dims, store [C] vector per step
HOOK_SPEC: dict[str, tuple[str, str]] = {
    "cnn.0":  ("cnn.0",  "pool"),   # Conv2d output [B, 32, 45, 45]
    "cnn.3":  ("cnn.3",  "pool"),   # Conv2d output [B, 64, 22, 22]
    "cnn.5":  ("cnn.5",  "pool"),   # Conv2d output [B, 64, 22, 22]
    "cnn.7":  ("cnn.7",  "full"),   # AdaptiveMaxPool [B, 64, 5, 5]
    "trunk.0": ("trunk.0", "full"), # Linear [B, 448]
    "trunk.2": ("trunk.2", "full"), # Linear [B, 448]
    "actor":   ("actor",   "full"), # Linear [B, 5]
    "critic":  ("critic",  "full"), # Linear [B, 1]
}


def _resolve_submodule(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path like 'cnn.0' to the actual submodule."""
    parts = path.split(".")
    mod = model
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod


class _ActivationRecorder:
    """Registers forward hooks and stores activations per forward pass."""

    def __init__(self, model: ActorCritic):
        self.cache: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []

        for key, (path, mode) in HOOK_SPEC.items():
            submod = _resolve_submodule(model, path)

            def _make_hook(k: str, m: str):
                def hook(_mod, _inp, out):
                    t = out.detach().cpu()
                    if m == "pool" and t.dim() == 4:
                        t = t.mean(dim=(2, 3))  # [B, C]
                    self.cache[k] = t
                return hook

            h = submod.register_forward_hook(_make_hook(key, mode))
            self._handles.append(h)

    def clear(self):
        self.cache.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Per-trajectory data container ────────────────────────────────────────────

@dataclass
class TrajectoryData:
    """All data collected for a single trajectory."""

    # Identity
    traj_id: int = 0
    map_id: int = 0
    map_source: str = "test"         # "test" or "behavioral"
    map_name: str = ""               # behavioral map name (if applicable)
    spawn: tuple[int, int] = (0, 0)
    target: tuple[int, int] = (0, 0)

    # Per-step arrays (lists, converted to numpy at save time)
    positions: list[list[int]] = field(default_factory=list)       # [T, 2]
    terrain_idx: list[float] = field(default_factory=list)         # [T]
    resources: list[float] = field(default_factory=list)           # [T]
    hp: list[float] = field(default_factory=list)                  # [T]
    cost: list[float] = field(default_factory=list)                # [T]
    cost_to_go: list[float] = field(default_factory=list)          # [T]
    obs_scalars: list[np.ndarray] = field(default_factory=list)    # [T, 5]
    obs_minimaps: list[np.ndarray] = field(default_factory=list)   # [T, 3, 45, 45]
    activations: dict[str, list[np.ndarray]] = field(default_factory=dict)
    actions: list[int] = field(default_factory=list)               # [T]
    rewards: list[float] = field(default_factory=list)             # [T]
    dones: list[bool] = field(default_factory=list)                # [T]
    values: list[float] = field(default_factory=list)              # [T]
    action_probs: list[np.ndarray] = field(default_factory=list)   # [T, 5]

    # Per-step flags
    is_target_visible: list[bool] = field(default_factory=list)
    target_just_entered_view: list[bool] = field(default_factory=list)
    is_low_hp: list[bool] = field(default_factory=list)
    is_low_resources: list[bool] = field(default_factory=list)
    is_in_forest: list[bool] = field(default_factory=list)
    is_on_water: list[bool] = field(default_factory=list)

    # Summary metrics (filled after episode)
    outcome: str = ""
    total_return: float = 0.0
    episode_length: int = 0
    directness_ratio: float = 0.0
    map_coverage: float = 0.0
    risk_score: float = 0.0
    ocean_usage_ratio: float = 0.0
    forest_usage_ratio: float = 0.0
    terrain_distribution: np.ndarray = field(default_factory=lambda: np.zeros(9))
    average_hp: float = 0.0
    min_hp: float = 0.0
    average_resources: float = 0.0
    min_resources: float = 0.0


def _compute_summary_metrics(traj: TrajectoryData, map_size: int = 250) -> None:
    """Fill in summary metrics from per-step data."""
    T = len(traj.actions)
    traj.episode_length = T
    traj.total_return = sum(traj.rewards)

    if T == 0:
        return

    pos = np.array(traj.positions)  # [T+1, 2] (includes initial position)
    start = pos[0]
    end = pos[-1]

    # Directness
    manhattan = abs(end[0] - start[0]) + abs(end[1] - start[1])
    traj.directness_ratio = manhattan / max(T, 1)

    # Map coverage
    unique_cells = set(map(tuple, pos.tolist()))
    traj.map_coverage = len(unique_cells) / (map_size * map_size)

    # Terrain stats
    terrain = np.array(traj.terrain_idx)
    water_mask = np.isin(terrain.astype(int), [0, 1, 2])
    dangerous_mask = np.isin(terrain.astype(int), [0, 1, 8])
    forest_mask = terrain.astype(int) == 6

    traj.risk_score = float(dangerous_mask.mean()) if T > 0 else 0.0
    traj.ocean_usage_ratio = float(water_mask.mean()) if T > 0 else 0.0
    traj.forest_usage_ratio = float(forest_mask.mean()) if T > 0 else 0.0

    # Terrain distribution
    hist = np.zeros(9)
    for idx in terrain.astype(int):
        if 0 <= idx < 9:
            hist[idx] += 1
    if hist.sum() > 0:
        hist /= hist.sum()
    traj.terrain_distribution = hist

    # HP / resource stats
    hp_arr = np.array(traj.hp)
    res_arr = np.array(traj.resources)
    traj.average_hp = float(hp_arr.mean())
    traj.min_hp = float(hp_arr.min())
    traj.average_resources = float(res_arr.mean())
    traj.min_resources = float(res_arr.min())


# ── Main collector ───────────────────────────────────────────────────────────

class TrajectoryCollector:
    """Run a trained model on maps and collect trajectories with activations.

    Usage::

        collector = TrajectoryCollector(model, env_config, device="cpu")
        trajectories = collector.collect_from_test_maps(test_maps, episodes_per_map=3)
        collector.save(trajectories, "interpretability/data/")
    """

    def __init__(
        self,
        model: ActorCritic,
        env_config: EnvConfig,
        device: str = "cpu",
        store_minimaps: bool = False,
    ):
        self.model = model
        self.env_config = env_config
        self.device = device
        self.store_minimaps = store_minimaps
        self._traj_counter = 0

    def collect_from_test_maps(
        self,
        test_maps: torch.Tensor,
        episodes_per_map: int = 3,
        seed: int = 1042,
    ) -> list[TrajectoryData]:
        """Run episodes on test maps, one map at a time.

        Args:
            test_maps: [N, H, W] float32 tensor of maps.
            episodes_per_map: how many spawn/target pairs per map.
            seed: base seed for environment resets.

        Returns:
            List of TrajectoryData, one per episode.
        """
        all_trajs = []
        n_maps = test_maps.shape[0]

        for map_idx in tqdm(range(n_maps), desc="Test maps"):
            single_map = test_maps[map_idx : map_idx + 1]  # [1, H, W]
            for ep in range(episodes_per_map):
                ep_seed = seed + map_idx * 1000 + ep
                traj = self._run_single_episode(
                    world_maps=single_map,
                    map_id=map_idx,
                    map_source="test",
                    map_name="",
                    seed=ep_seed,
                    fixed_spawn=None,
                    fixed_target=None,
                )
                all_trajs.append(traj)

        return all_trajs

    def collect_from_behavioral_maps(
        self,
        behavioral_path: str = "data/test_behavior.pt",
        seed: int = 2042,
    ) -> list[TrajectoryData]:
        """Run episodes on hand-crafted behavioral maps with fixed spawn/target."""
        data = torch.load(behavioral_path, map_location="cpu", weights_only=False)
        names: list[str] = data["names"]
        maps: torch.Tensor = data["maps"]      # [N, H, W]
        spawns = data["spawns"]   # list of (r, c) tuples or [N, 2] tensor
        targets = data["targets"] # list of (r, c) tuples or [N, 2] tensor

        all_trajs = []
        for i, name in enumerate(tqdm(names, desc="Behavioral maps")):
            single_map = maps[i : i + 1]  # [1, H, W]
            s = spawns[i]
            t = targets[i]
            spawn = (int(s[0]), int(s[1])) if not isinstance(s, tuple) else s
            target = (int(t[0]), int(t[1])) if not isinstance(t, tuple) else t
            traj = self._run_single_episode(
                world_maps=single_map,
                map_id=i,
                map_source="behavioral",
                map_name=name,
                seed=seed + i,
                fixed_spawn=spawn,
                fixed_target=target,
            )
            all_trajs.append(traj)

        return all_trajs

    def _run_single_episode(
        self,
        world_maps: torch.Tensor,
        map_id: int,
        map_source: str,
        map_name: str,
        seed: int,
        fixed_spawn: tuple[int, int] | None,
        fixed_target: tuple[int, int] | None,
    ) -> TrajectoryData:
        """Run one episode and collect all data."""
        import dataclasses
        from cogniland.env.types import CustomMapConfig

        model = self.model
        config = self.env_config

        # For behavioral maps with fixed spawn/target, use CustomMapConfig override
        if fixed_spawn is not None and fixed_target is not None:
            config = dataclasses.replace(
                config,
                custom_map=CustomMapConfig(
                    spawn_r=fixed_spawn[0], spawn_c=fixed_spawn[1],
                    target_r=fixed_target[0], target_c=fixed_target[1],
                ),
            )
            env = BatchedIslandEnv(config, num_envs=1, world_maps=world_maps)
        else:
            env = BatchedIslandEnv(config, num_envs=1, world_maps=world_maps)

        obs = env.reset(seed=seed)
        state = env.state
        compiled = env.compiled

        traj = TrajectoryData(
            traj_id=self._traj_counter,
            map_id=map_id,
            map_source=map_source,
            map_name=map_name,
            spawn=(int(state.position[0, 0].item()), int(state.position[0, 1].item())),
            target=(int(env.target_pos[0, 0].item()), int(env.target_pos[0, 1].item())),
            activations={k: [] for k in HOOK_SPEC},
        )
        self._traj_counter += 1

        # Record initial position
        traj.positions.append(state.position[0].cpu().tolist())

        # Set up activation hooks
        recorder = _ActivationRecorder(model)
        prev_target_visible = False

        for step in range(config.max_steps):
            state = env.state

            # Run model forward (hooks fire here)
            recorder.clear()
            with torch.no_grad():
                feat = model._features(obs)
                logits = model.actor(feat)
                value = model.critic(feat).squeeze(-1)
                probs = torch.softmax(logits, dim=-1)
                action = logits.argmax(dim=-1)

            # Record activations
            for key in HOOK_SPEC:
                if key in recorder.cache:
                    traj.activations[key].append(
                        recorder.cache[key][0].numpy().astype(np.float16)
                    )

            # Record obs
            traj.obs_scalars.append(obs["scalars"][0].cpu().numpy().astype(np.float16))
            if self.store_minimaps:
                traj.obs_minimaps.append(obs["minimap"][0].cpu().numpy().astype(np.float16))

            # Record state
            traj.terrain_idx.append(state.terrain_idx[0].item())
            traj.resources.append(state.resources[0].item())
            traj.hp.append(state.hp[0].item())
            traj.cost.append(state.cost[0].item())
            traj.cost_to_go.append(state.cost_to_go[0].item())

            # Record model outputs
            traj.values.append(value[0].item())
            traj.action_probs.append(probs[0].cpu().numpy())
            traj.actions.append(action[0].item())

            # Contextual flags
            target_vis = bool(obs["minimap"][0, 1].any().item())
            traj.is_target_visible.append(target_vis)
            traj.target_just_entered_view.append(target_vis and not prev_target_visible)
            traj.is_low_hp.append(state.hp[0].item() < 0.3 * config.max_hp)
            traj.is_low_resources.append(state.resources[0].item() < 0.2 * config.max_resources)
            traj.is_in_forest.append(int(state.terrain_idx[0].item()) == 6)
            traj.is_on_water.append(int(state.terrain_idx[0].item()) in {0, 1, 2})
            prev_target_visible = target_vis

            # Step environment
            obs, reward, done, info = env.step(action)
            traj.rewards.append(reward[0].item())
            traj.dones.append(bool(done[0].item()))

            # Record new position
            if not done[0]:
                traj.positions.append(env.state.position[0].cpu().tolist())
            else:
                # Final position
                if info["reached"][0]:
                    traj.positions.append(list(traj.target))
                    traj.outcome = "success"
                elif not info["alive"][0]:
                    traj.positions.append(state.position[0].cpu().tolist())
                    traj.outcome = "death"
                else:
                    traj.positions.append(state.position[0].cpu().tolist())
                    traj.outcome = "timeout"
                break
        else:
            traj.outcome = "timeout"

        recorder.remove()
        _compute_summary_metrics(traj, map_size=config.size)
        return traj

    def save(
        self,
        trajectories: list[TrajectoryData],
        output_dir: str = "interpretability/data",
    ) -> tuple[str, str]:
        """Save trajectories to HDF5 and summary CSV.

        Returns:
            (h5_path, csv_path)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        h5_path = str(out / "trajectories.h5")
        csv_path = str(out / "summary.csv")

        # ── HDF5 ──
        with h5py.File(h5_path, "w") as f:
            for traj in tqdm(trajectories, desc="Saving HDF5"):
                g = f.create_group(f"trajectory_{traj.traj_id:04d}")

                # Metadata as attributes
                g.attrs["traj_id"] = traj.traj_id
                g.attrs["map_id"] = traj.map_id
                g.attrs["map_source"] = traj.map_source
                g.attrs["map_name"] = traj.map_name
                g.attrs["spawn_row"] = traj.spawn[0]
                g.attrs["spawn_col"] = traj.spawn[1]
                g.attrs["target_row"] = traj.target[0]
                g.attrs["target_col"] = traj.target[1]
                g.attrs["outcome"] = traj.outcome
                g.attrs["total_return"] = traj.total_return
                g.attrs["episode_length"] = traj.episode_length

                # Per-step data
                g.create_dataset("positions", data=np.array(traj.positions, dtype=np.int32))
                g.create_dataset("terrain_idx", data=np.array(traj.terrain_idx, dtype=np.float32))
                g.create_dataset("resources", data=np.array(traj.resources, dtype=np.float32))
                g.create_dataset("hp", data=np.array(traj.hp, dtype=np.float32))
                g.create_dataset("cost", data=np.array(traj.cost, dtype=np.float32))
                g.create_dataset("cost_to_go", data=np.array(traj.cost_to_go, dtype=np.float32))
                g.create_dataset("obs_scalars", data=np.stack(traj.obs_scalars))
                if traj.obs_minimaps:
                    g.create_dataset(
                        "obs_minimaps", data=np.stack(traj.obs_minimaps),
                        compression="gzip", compression_opts=4,
                    )
                g.create_dataset("actions", data=np.array(traj.actions, dtype=np.int32))
                g.create_dataset("rewards", data=np.array(traj.rewards, dtype=np.float32))
                g.create_dataset("dones", data=np.array(traj.dones, dtype=bool))
                g.create_dataset("values", data=np.array(traj.values, dtype=np.float32))
                g.create_dataset("action_probs", data=np.stack(traj.action_probs).astype(np.float32))

                # Activations
                act_g = g.create_group("activations")
                for key, arrs in traj.activations.items():
                    if arrs:
                        act_g.create_dataset(
                            key.replace(".", "_"), data=np.stack(arrs),
                            compression="gzip", compression_opts=4,
                        )

                # Flags
                fl_g = g.create_group("flags")
                fl_g.create_dataset("is_target_visible", data=np.array(traj.is_target_visible))
                fl_g.create_dataset("target_just_entered_view", data=np.array(traj.target_just_entered_view))
                fl_g.create_dataset("is_low_hp", data=np.array(traj.is_low_hp))
                fl_g.create_dataset("is_low_resources", data=np.array(traj.is_low_resources))
                fl_g.create_dataset("is_in_forest", data=np.array(traj.is_in_forest))
                fl_g.create_dataset("is_on_water", data=np.array(traj.is_on_water))

        # ── Summary CSV ──
        rows = []
        for traj in trajectories:
            row = {
                "traj_id": traj.traj_id,
                "map_id": traj.map_id,
                "map_source": traj.map_source,
                "map_name": traj.map_name,
                "spawn_row": traj.spawn[0],
                "spawn_col": traj.spawn[1],
                "target_row": traj.target[0],
                "target_col": traj.target[1],
                "outcome": traj.outcome,
                "total_return": traj.total_return,
                "episode_length": traj.episode_length,
                "directness_ratio": traj.directness_ratio,
                "map_coverage": traj.map_coverage,
                "risk_score": traj.risk_score,
                "ocean_usage_ratio": traj.ocean_usage_ratio,
                "forest_usage_ratio": traj.forest_usage_ratio,
                "average_hp": traj.average_hp,
                "min_hp": traj.min_hp,
                "average_resources": traj.average_resources,
                "min_resources": traj.min_resources,
            }
            for i in range(9):
                row[f"terrain_frac_{i}"] = traj.terrain_distribution[i]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        print(f"Saved {len(trajectories)} trajectories to {h5_path}")
        print(f"Saved summary CSV to {csv_path}")
        return h5_path, csv_path
