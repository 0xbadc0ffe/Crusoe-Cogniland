"""Immutable state types for the environment.

NamedTuples are chosen for JAX compatibility (they are pytrees).
`state._replace(hp=new_hp)` creates a new state without mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import torch


class CurriculumStage(str, Enum):
    """Training curriculum stage controlling spawn/target sampling."""
    EXTRA_EASY = "extra_easy"  # Spawn + target constrained to small radius (25) around map center
    EASY       = "easy"        # Spawn + target constrained to medium radius (50) around map center
    NORMAL     = "normal"      # Spawn + target sampled uniformly over land cells


class EnvState(NamedTuple):
    """Full batched environment state."""

    position: torch.Tensor        # [B, 2] long
    minimap: torch.Tensor         # [B, 3, 2*max_ray+1, 2*max_ray+1] float (ch0=heightmap, ch1=target indicator, ch2=visibility mask)
    compass: torch.Tensor         # [B, 2] float — unit direction vector (position − target) / dist
    terrain_idx: torch.Tensor     # [B] float
    resources: torch.Tensor       # [B] float
    hp: torch.Tensor              # [B] float
    cost: torch.Tensor            # [B] float — accumulated terrain time cost this episode
    dijkstra_cost: torch.Tensor   # [B] float — optimal time cost spawn→target (constant per episode)
    cost_to_go: torch.Tensor      # [B] float — Dijkstra cost-to-go from current position to target


class StepResult(NamedTuple):
    """Result of a single environment step."""

    state: EnvState
    reward: torch.Tensor   # [B]
    done: torch.Tensor     # [B] bool
    info: dict


# ---------------------------------------------------------------------------
# Terrain definition (one per entry in the terrains: list)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerrainDef:
    """Single terrain level — mirrors one entry in default.yaml terrains: list.

    res_rate: signed resource rate per step. Negative = drain (most terrains),
              positive = gain (forest when at full HP).
    hp_rate:  signed HP rate per step. Positive = heal (forest when below max HP).
    """
    name: str
    threshold: float
    move_cost: float
    res_rate: float
    hp_rate: float
    visibility: int
    color: tuple[int, int, int]
    tags: tuple[str, ...]


# ---------------------------------------------------------------------------
# Compiled terrain tensors (built once, reused every step)
# ---------------------------------------------------------------------------

class CompiledTerrainData:
    """Pre-built tensor arrays derived from the terrains list.

    Constructed once via EnvConfig.compile_terrain(device) and attached to
    the config.  All engine code reads from here instead of global constants.
    """

    def __init__(self, terrains: list[TerrainDef] | tuple[TerrainDef, ...], device: str):
        self.num_terrains = len(terrains)
        self.terrain_names = [t.name for t in terrains]

        self.thresholds = torch.tensor(
            [t.threshold for t in terrains], dtype=torch.float32, device=device
        )
        self.move_costs = torch.tensor(
            [t.move_cost for t in terrains], dtype=torch.float32, device=device
        )
        self.res_rate = torch.tensor(
            [t.res_rate for t in terrains], dtype=torch.float32, device=device
        )
        self.hp_rate = torch.tensor(
            [t.hp_rate for t in terrains], dtype=torch.float32, device=device
        )
        self.visibility = torch.tensor(
            [t.visibility for t in terrains], dtype=torch.long, device=device
        )
        # Boolean capability masks
        self.is_water = torch.tensor(
            ["water" in t.tags for t in terrains], dtype=torch.bool, device=device
        )
        self.is_forest = torch.tensor(
            ["forest" in t.tags for t in terrains], dtype=torch.bool, device=device
        )
        # Color LUT for visualization: [N, 3] uint8
        self.color_lut = torch.tensor(
            [list(t.color) for t in terrains], dtype=torch.uint8, device=device
        )
        # Land threshold = max threshold of water terrains (cells above this are "land")
        water_thresholds = [t.threshold for t in terrains if "water" in t.tags]
        self.land_threshold = max(water_thresholds) if water_thresholds else 0.0


# ---------------------------------------------------------------------------
# Nested config sub-dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MapGenConfig:
    size: int = 250
    scale: float = 0.33
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    seed: int = 42
    filtering: str = "square"
    sink_mode: int = 1


@dataclass(frozen=True)
class AgentConfig:
    init_hp: float = 100.0
    max_hp: float = 100.0
    init_resources: float = 50.0
    max_resources: float = 100.0
    land_to_water_resource_cost: float = 20.0
    no_res_hp_multiplier: float = 2.0


@dataclass(frozen=True)
class MinimapConfig:
    max_ray: int = 22
    occlude: bool = False
    clear_tolerance: float = 0.1


@dataclass(frozen=True)
class CustomMapConfig:
    map_name: str = ""
    spawn_r: int = -1
    spawn_c: int = -1
    target_r: int = -1
    target_c: int = -1


@dataclass(frozen=True)
class DatasetConfig:
    path: str = ""
    curriculum_switch_steps: int = 0
    curriculum_easy_radius: int = 50


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping parameters — part of the environment specification."""
    reach_bonus: float = 150.0    # r_success: sparse bonus on reaching target
    lambda_p: float = 0.08       # cost-to-go progress weight
    lambda_s: float = 0.02       # per-step penalty
    lambda_t: float = 40.0       # time-efficiency bonus weight at success
    lambda_d: float = 0.10       # death penalty = lambda_d * reach_bonus
    beta_raft: float = 10.0      # extra cost for land→water transitions in Dijkstra cost-to-go


# ---------------------------------------------------------------------------
# Top-level environment config
# ---------------------------------------------------------------------------

_DEFAULT_TERRAINS = (
    #                        thresh  cost  res_rate  hp_rate  vis  color             tags
    TerrainDef("ocean",      0.007,  1.0,  -1.0,     0.0,    16,  (5,35,225),    ("water",)),
    TerrainDef("deep_water", 0.025,  1.25, -0.5,     0.0,    12,  (25,65,225),   ("water",)),
    TerrainDef("water",      0.05,   1.5,  -0.2,     0.0,     8,  (65,105,225),  ("water",)),
    TerrainDef("beach",      0.06,   1.75, -1.0,     0.0,     5,  (238,214,175), ("land",)),
    TerrainDef("sandy",      0.1,    2.0,  -1.0,     0.0,     5,  (210,180,140), ("land",)),
    TerrainDef("grassland",  0.25,   2.25, -1.0,     0.0,     5,  (34,139,34),   ("land",)),
    TerrainDef("forest",     0.6,    3.0,   3.0,     5.0,     3,  (0,100,0),     ("land","forest")),
    TerrainDef("rocky",      0.7,    3.5,  -2.0,     0.0,    10,  (139,137,137), ("land",)),
    TerrainDef("mountains",  1.0,    4.0,  -5.0,     0.0,    22,  (255,250,250), ("land",)),
)


@dataclass(frozen=True)
class EnvConfig:
    """Immutable environment configuration with nested sub-configs."""

    map_generation: MapGenConfig = field(default_factory=MapGenConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    minimap: MinimapConfig = field(default_factory=MinimapConfig)
    custom_map: CustomMapConfig = field(default_factory=CustomMapConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    max_steps: int = 1000

    terrains: tuple[TerrainDef, ...] = _DEFAULT_TERRAINS

    device: str = "auto"

    # --- Convenience shortcuts (read from nested sub-configs) ---

    @property
    def size(self) -> int:
        return self.map_generation.size

    @property
    def scale(self) -> float:
        return self.map_generation.scale

    @property
    def octaves(self) -> int:
        return self.map_generation.octaves

    @property
    def persistence(self) -> float:
        return self.map_generation.persistence

    @property
    def lacunarity(self) -> float:
        return self.map_generation.lacunarity

    @property
    def seed(self) -> int:
        return self.map_generation.seed

    @property
    def filtering(self) -> str:
        return self.map_generation.filtering

    @property
    def sink_mode(self) -> int:
        return self.map_generation.sink_mode

    @property
    def init_hp(self) -> float:
        return self.agent.init_hp

    @property
    def max_hp(self) -> float:
        return self.agent.max_hp

    @property
    def init_resources(self) -> float:
        return self.agent.init_resources

    @property
    def max_resources(self) -> float:
        return self.agent.max_resources

    @property
    def minimap_max_ray(self) -> int:
        return self.minimap.max_ray

    @property
    def minimap_occlude(self) -> bool:
        return self.minimap.occlude

    @property
    def minimap_clear_tolerance(self) -> float:
        return self.minimap.clear_tolerance

    @property
    def map_name(self) -> str:
        return self.custom_map.map_name

    @property
    def spawn_r(self) -> int:
        return self.custom_map.spawn_r

    @property
    def spawn_c(self) -> int:
        return self.custom_map.spawn_c

    @property
    def target_r(self) -> int:
        return self.custom_map.target_r

    @property
    def target_c(self) -> int:
        return self.custom_map.target_c

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def compile_terrain(self, device: str | None = None) -> CompiledTerrainData:
        """Build pre-computed terrain tensors for the given device."""
        if device is None:
            device = self.resolved_device()
        return CompiledTerrainData(self.terrains, device)

    @classmethod
    def from_hydra(cls, cfg) -> "EnvConfig":
        """Build EnvConfig from a Hydra DictConfig (cfg.env + cfg.device)."""
        env = cfg.env

        # Parse terrains list from YAML
        raw_terrains = env.get("terrains", None)
        if raw_terrains is not None:
            terrains = tuple(
                TerrainDef(
                    name=t["name"],
                    threshold=float(t["threshold"]),
                    move_cost=float(t["move_cost"]),
                    res_rate=float(t["res_rate"]),
                    hp_rate=float(t["hp_rate"]),
                    visibility=int(t["visibility"]),
                    color=tuple(int(c) for c in t["color"]),
                    tags=tuple(str(tag) for tag in t["tags"]),
                )
                for t in raw_terrains
            )
        else:
            terrains = _DEFAULT_TERRAINS

        # Parse nested sub-configs (with fallback to flat keys for compat)
        mg_cfg = env.get("map_generation", env)
        map_gen = MapGenConfig(
            size=mg_cfg.get("size", 250),
            scale=mg_cfg.get("scale", 0.33),
            octaves=mg_cfg.get("octaves", 6),
            persistence=mg_cfg.get("persistence", 0.5),
            lacunarity=mg_cfg.get("lacunarity", 2.0),
            seed=mg_cfg.get("seed", 42),
            filtering=mg_cfg.get("filtering", "square"),
            sink_mode=mg_cfg.get("sink_mode", 1),
        )

        ag_cfg = env.get("agent", env)
        agent = AgentConfig(
            init_hp=ag_cfg.get("init_hp", 100.0),
            max_hp=ag_cfg.get("max_hp", 100.0),
            init_resources=ag_cfg.get("init_resources", 50.0),
            max_resources=ag_cfg.get("max_resources", 100.0),
            land_to_water_resource_cost=ag_cfg.get("land_to_water_resource_cost", 20.0),
            no_res_hp_multiplier=ag_cfg.get("no_res_hp_multiplier", 2.0),
        )

        mm_cfg = env.get("minimap", env)
        minimap_cfg = MinimapConfig(
            max_ray=mm_cfg.get("max_ray", mm_cfg.get("minimap_max_ray", 22)),
            occlude=mm_cfg.get("occlude", mm_cfg.get("minimap_occlude", False)),
            clear_tolerance=mm_cfg.get("clear_tolerance", mm_cfg.get("minimap_clear_tolerance", 0.1)),
        )

        cm_cfg = env.get("custom_map", env)
        custom_map = CustomMapConfig(
            map_name=cm_cfg.get("map_name", ""),
            spawn_r=cm_cfg.get("spawn_r", -1),
            spawn_c=cm_cfg.get("spawn_c", -1),
            target_r=cm_cfg.get("target_r", -1),
            target_c=cm_cfg.get("target_c", -1),
        )

        rw_cfg = env.get("reward", {})
        reward = RewardConfig(
            reach_bonus=rw_cfg.get("reach_bonus", 150.0),
            lambda_p=rw_cfg.get("lambda_p", 0.08),
            lambda_s=rw_cfg.get("lambda_s", 0.02),
            lambda_t=rw_cfg.get("lambda_t", 40.0),
            lambda_d=rw_cfg.get("lambda_d", 0.10),
            beta_raft=rw_cfg.get("beta_raft", 10.0),
        )

        return cls(
            map_generation=map_gen,
            agent=agent,
            minimap=minimap_cfg,
            custom_map=custom_map,
            reward=reward,
            max_steps=env.get("max_steps", 1000),
            terrains=terrains,
            device=cfg.device,
        )
