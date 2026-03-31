"""TrajectoryDataManager — query and filter interface over collected data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


class TrajectoryDataManager:
    """Load, filter, and query trajectory data from HDF5 + CSV.

    Usage::

        dm = TrajectoryDataManager("interpretability/data/")
        successful = dm.filter(outcome="success")
        trunk_acts = dm.get_activations([0, 1, 2], layer="trunk_2")
        low_hp_steps = dm.get_steps_where(is_low_hp=True)
    """

    def __init__(self, data_dir: str = "interpretability/data"):
        self.data_dir = Path(data_dir)
        self.h5_path = self.data_dir / "trajectories.h5"
        self.csv_path = self.data_dir / "summary.csv"

        if self.csv_path.exists():
            self.summary = pd.read_csv(self.csv_path)
            # map_name is empty string for test maps, but CSV reads it as NaN
            if "map_name" in self.summary.columns:
                self.summary["map_name"] = self.summary["map_name"].fillna("")
        else:
            self.summary = pd.DataFrame()

    @property
    def n_trajectories(self) -> int:
        return len(self.summary)

    @property
    def traj_ids(self) -> list[int]:
        return self.summary["traj_id"].tolist()

    def filter(self, **kwargs) -> pd.DataFrame:
        """Filter summary dataframe by column values.

        Supports exact match (e.g., outcome="success") and comparison
        operators via double-underscore suffixes:
            __gt, __lt, __gte, __lte  (e.g., ocean_usage_ratio__gt=0.3)
        """
        df = self.summary
        for key, val in kwargs.items():
            if "__gt" in key:
                col = key.replace("__gt", "")
                df = df[df[col] > val]
            elif "__lt" in key:
                col = key.replace("__lt", "")
                df = df[df[col] < val]
            elif "__gte" in key:
                col = key.replace("__gte", "")
                df = df[df[col] >= val]
            elif "__lte" in key:
                col = key.replace("__lte", "")
                df = df[df[col] <= val]
            else:
                df = df[df[key] == val]
        return df

    def get_trajectory(self, traj_id: int) -> dict:
        """Load full trajectory data from HDF5.

        Returns dict with keys: positions, terrain_idx, resources, hp, cost,
        cost_to_go, obs_scalars, actions, rewards, dones, values, action_probs,
        activations (dict), flags (dict), and attrs (dict).
        """
        with h5py.File(self.h5_path, "r") as f:
            gname = f"trajectory_{traj_id:04d}"
            if gname not in f:
                raise KeyError(f"Trajectory {traj_id} not found in {self.h5_path}")
            g = f[gname]

            result = {}
            # Scalar datasets
            for key in ["positions", "terrain_idx", "resources", "hp", "cost",
                        "cost_to_go", "obs_scalars", "actions", "rewards",
                        "dones", "values", "action_probs"]:
                if key in g:
                    result[key] = np.array(g[key])

            # Optional minimaps
            if "obs_minimaps" in g:
                result["obs_minimaps"] = np.array(g["obs_minimaps"])

            # Activations
            result["activations"] = {}
            if "activations" in g:
                for key in g["activations"]:
                    result["activations"][key] = np.array(g["activations"][key])

            # Flags
            result["flags"] = {}
            if "flags" in g:
                for key in g["flags"]:
                    result["flags"][key] = np.array(g["flags"][key])

            # Attributes
            result["attrs"] = dict(g.attrs)

        return result

    def get_activations(
        self,
        traj_ids: list[int],
        layer: str,
    ) -> list[np.ndarray]:
        """Get activations for a layer across multiple trajectories.

        Args:
            traj_ids: trajectory IDs to load.
            layer: activation key (e.g., "trunk_2", "actor"). Uses underscore
                   instead of dot (matching HDF5 dataset names).

        Returns:
            List of arrays, one per trajectory, each [T_i, dim].
        """
        results = []
        with h5py.File(self.h5_path, "r") as f:
            for tid in traj_ids:
                gname = f"trajectory_{tid:04d}"
                act_key = f"{gname}/activations/{layer}"
                if act_key in f:
                    results.append(np.array(f[act_key]).astype(np.float32))
                else:
                    results.append(np.empty((0,), dtype=np.float32))
        return results

    def get_steps_where(self, **flag_conditions) -> dict:
        """Get all step-level data where flag conditions are met.

        Args:
            flag_conditions: e.g., is_low_hp=True, is_target_visible=True

        Returns:
            Dict with keys:
                traj_ids: [N] array of trajectory IDs for each step
                step_indices: [N] array of step indices within trajectory
                activations: dict of layer_name → [N, dim] arrays
                terrain_idx: [N] array
                hp: [N] array
                resources: [N] array
        """
        all_traj_ids = []
        all_step_idx = []
        all_terrain = []
        all_hp = []
        all_resources = []
        all_activations: dict[str, list[np.ndarray]] = {}

        with h5py.File(self.h5_path, "r") as f:
            for gname in sorted(f.keys()):
                g = f[gname]
                tid = int(g.attrs["traj_id"])

                # Check flag conditions
                masks = []
                for flag_name, flag_val in flag_conditions.items():
                    flag_key = f"flags/{flag_name}"
                    if flag_key in g:
                        flag_arr = np.array(g[flag_key])
                        if flag_val:
                            masks.append(flag_arr)
                        else:
                            masks.append(~flag_arr)

                if not masks:
                    continue

                combined = masks[0]
                for m in masks[1:]:
                    combined = combined & m

                indices = np.where(combined)[0]
                if len(indices) == 0:
                    continue

                all_traj_ids.append(np.full(len(indices), tid, dtype=np.int32))
                all_step_idx.append(indices.astype(np.int32))

                if "terrain_idx" in g:
                    all_terrain.append(np.array(g["terrain_idx"])[indices])
                if "hp" in g:
                    all_hp.append(np.array(g["hp"])[indices])
                if "resources" in g:
                    all_resources.append(np.array(g["resources"])[indices])

                # Activations
                if "activations" in g:
                    for act_key in g["activations"]:
                        if act_key not in all_activations:
                            all_activations[act_key] = []
                        all_activations[act_key].append(
                            np.array(g["activations"][act_key]).astype(np.float32)[indices]
                        )

        result = {
            "traj_ids": np.concatenate(all_traj_ids) if all_traj_ids else np.array([], dtype=np.int32),
            "step_indices": np.concatenate(all_step_idx) if all_step_idx else np.array([], dtype=np.int32),
            "terrain_idx": np.concatenate(all_terrain) if all_terrain else np.array([]),
            "hp": np.concatenate(all_hp) if all_hp else np.array([]),
            "resources": np.concatenate(all_resources) if all_resources else np.array([]),
            "activations": {},
        }
        for k, v in all_activations.items():
            result["activations"][k] = np.concatenate(v) if v else np.empty((0,))

        return result

    def get_map_for_trajectory(
        self,
        traj_id: int,
        test_maps_path: str = "data/test_seed42_n16.pt",
        behavioral_maps_path: str = "data/test_behavior.pt",
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load the world map, spawn, and target for a given trajectory.

        Returns:
            (world_map [H, W] numpy, spawn (r, c), target (r, c))
        """
        row = self.summary[self.summary["traj_id"] == traj_id].iloc[0]
        source = row["map_source"]
        map_id = int(row["map_id"])
        spawn = (int(row["spawn_row"]), int(row["spawn_col"]))
        target = (int(row["target_row"]), int(row["target_col"]))

        if source == "behavioral":
            data = torch.load(behavioral_maps_path, map_location="cpu", weights_only=False)
            world_map = data["maps"][map_id].numpy()
        else:
            data = torch.load(test_maps_path, map_location="cpu", weights_only=True)
            world_map = data["maps"][map_id].numpy()

        return world_map, spawn, target

    def get_all_activations_flat(
        self,
        layer: str,
        traj_ids: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get all step-level activations concatenated across trajectories.

        Args:
            layer: e.g., "trunk_2"
            traj_ids: subset of trajectories (default: all)

        Returns:
            (activations [total_steps, dim], traj_id_per_step [total_steps],
             terrain_per_step [total_steps])
        """
        if traj_ids is None:
            traj_ids = self.traj_ids

        acts_list = []
        tids_list = []
        terrain_list = []

        with h5py.File(self.h5_path, "r") as f:
            for tid in traj_ids:
                gname = f"trajectory_{tid:04d}"
                if gname not in f:
                    continue
                g = f[gname]
                act_key = f"activations/{layer}"
                if act_key not in g:
                    continue
                acts = np.array(g[act_key]).astype(np.float32)
                # Flatten spatial dims if needed (e.g., cnn_7 is [T, 64, 5, 5])
                if acts.ndim > 2:
                    acts = acts.reshape(acts.shape[0], -1)
                acts_list.append(acts)
                T = acts.shape[0]
                tids_list.append(np.full(T, tid, dtype=np.int32))
                if "terrain_idx" in g:
                    terrain_list.append(np.array(g["terrain_idx"])[:T])

        if not acts_list:
            return np.empty((0, 0)), np.array([]), np.array([])

        return (
            np.concatenate(acts_list),
            np.concatenate(tids_list),
            np.concatenate(terrain_list) if terrain_list else np.array([]),
        )
