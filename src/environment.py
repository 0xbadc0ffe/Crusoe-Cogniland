import torch
import utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
from simplexnoise.noise import SimplexNoise, normalize
import math
import copy

palette = {
    # "deepocean" :(0.07, [5,35,225]),
    # "ocean" : (0.025, [25,65,225]),
    #"lightblue" : ([0,191,255]),
    "deepocean" : [5,35,225],
    "ocean" : [25,65,225],
    "blue" : [65,105,225],
    "green" : [34,139,34],
    "darkgreen" : [0,100,0],
    "sandy" : [210,180,140],
    "beach" : [238, 214, 175],
    "snow" : [255, 250, 250],
    "mountain" : [139, 137, 137]
}

# Terrain levels: 0=ocean, 1=deep water, 2=water, 3=beach, 4=sandy, 5=grassland, 6=forest, 7=rocky, 8=mountains
TERRAIN_LEVELS = {
    0: {"name": "ocean", "threshold": 0.007, "cost": 0.5, "color": "deepocean"},
    1: {"name": "deep_water", "threshold": 0.025, "cost": 0.75, "color": "ocean"},  
    2: {"name": "water", "threshold": 0.05, "cost": 1.0, "color": "blue"},
    3: {"name": "beach", "threshold": 0.06, "cost": 2.5, "color": "beach"},
    4: {"name": "sandy", "threshold": 0.1, "cost": 2.5, "color": "sandy"},
    5: {"name": "grassland", "threshold": 0.25, "cost": 1.8, "color": "green"},
    6: {"name": "forest", "threshold": 0.6, "cost": 3.0, "color": "darkgreen"},
    7: {"name": "rocky", "threshold": 0.7, "cost": 4.0, "color": "mountain"},
    8: {"name": "mountains", "threshold": 1.0, "cost": 8.0, "color": "snow"}
}

# Visibility ranges for minimap based on terrain
VISIBILITY_RANGES = {
    0: 6, 1: 6, 2: 6,  # water levels
    3: 4, 4: 4,        # beach, sandy  
    5: 5,              # grassland
    6: 2,              # forest
    7: 6,              # rocky
    8: 10              # mountains
}

# Legacy LEVELS for backward compatibility - will be removed
LEVELS = [
    (0.007, "ocean"),
    (0.025, "deepocean"), 
    (0.05, "blue"),
    (0.06, "sandy"),
    (0.1, "beach"),
    (0.25, "green"),
    (0.6, "darkgreen"),
    (0.7, "snow"),
    (1.0, "mountain")
]

ACTIONS = {
    "up": 0,
    "down": 1,
    "right": 2,
    "left": 3,
    "stay": 4
}

class Envinroment:
    def __init__(self, init_state) -> None:
        self.state = init_state

    def get_state(self):
        raise NotImplementedError

    def set_state(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError


class Islands(Envinroment):

    def __init__(self, batch_dim=1, init_position=None, obj_position=None, island_options=None, minimap_opts=None, agent_opts=None ) -> None:

        super().__init__(None)

        # STATE: (position, minimap, env_compass, terrain_lev, terrain_clock, wood, life)
        if island_options:
            self.island_options = island_options
        else:
            self.island_options = {
                "size": 250, #1080 #250 #500
                "scale": 0.33, # real_scale = size*scale
                "octaves": 6,
                "persistence": 0.5,
                "lacunarity": 2.0,
                "seed": 42,
                "detailed_ocean": True,
                "filtering": "square", #"circle", "square", "diamond"
                "sink_mode": 1, #0 = None, 1>2 
            }
        if minimap_opts:
            self.minimap_opts = minimap_opts
        else:
            self.minimap_opts = {
                "ray":25,
                "occlude":False,
                "min_clear_lv":0.25
            }
        
        if agent_opts:
            self.agent_opts = agent_opts
        else:
            self.agent_opts = {
                "init_hp": 75,  # Starting HP as specified
                "max_hp": 100,
                "init_resources": 0,
                "max_sea_movement_without_resources": 7,
                "hard_mode": False  # Easy mode by default
            }

        self.generate_island()
        
        # STATE: (position, minimap, env_compass, terrain_lev, terrain_clock, wood, hp, cost)
        # TODO: easier compass in 2 dim (DX DY) or hard 1 dim one?
        self.set_objective(batch_dim, obj_position)
        self.gen_init_state(batch_dim, init_position)
        

    def gen_init_state(self, batch_dim=1, init_position=None):
        if init_position is None:
            size = self.island_options["size"]
            init_position = []
            while True:
                if len(init_position)==batch_dim: break
                ip = torch.randint(0,size,[2])
                if self.world_map[ip[0], ip[1]] > LEVELS[2][0]:
                    init_position.append(ip)
            init_position = torch.stack(init_position, dim=0)


        minimap_vec = self.get_minimap(init_position)
        #compass_vec = torch.norm(init_position-self.obj_position, dim=-1)
        compass_vec = init_position-self.obj_position
        terr_vec = torch.zeros(batch_dim, dtype=torch.float)
        for n, lev in enumerate(LEVELS[::-1]):
            terr_vec[self.world_map[init_position[:,0], init_position[:,1]]<=lev[0]] = (len(LEVELS)-n-1)

        clock_vec = torch.zeros(batch_dim, dtype=torch.float)
        resources_vec = torch.tensor([self.agent_opts["init_resources"]]*batch_dim, dtype=torch.float)
        hp_vec = torch.tensor([self.agent_opts["init_hp"]]*batch_dim, dtype=torch.float)
        cost_vec = torch.zeros(batch_dim, dtype=torch.float)


        self.state = {
            "position": init_position,
            "minimap": minimap_vec,
            "compass": compass_vec,
            "terrain_lev": terr_vec,
            "terrain_clock": clock_vec,
            "resources": resources_vec,
            "hp": hp_vec,
            "cost": cost_vec
        }
       

    def set_objective(self, batch_dim=1, obj_position=None):
        if obj_position:
            self.obj_position = obj_position
        else:
            size = self.island_options["size"]
        #self.obj_position = torch.randint(0,size,[batch_dim,2])
        obj_position = []
        while True:
            if len(obj_position)==batch_dim: break
            op = torch.randint(0,size,[2])
            if self.world_map[op[0], op[1]] > LEVELS[2][0]:
                obj_position.append(op)
        self.obj_position = torch.stack(obj_position, dim=0)


    def generate_island(self, island_options=None):
        if island_options is None:
            island_options = self.island_options

        island_options = {
            "size": 250, #1080 #250 #500
            "scale": 0.33, # real_scale = size*scale
            "octaves": 6,
            "persistence": 0.5,
            "lacunarity": 2.0,
            "seed": 42,
            "detailed_ocean": True,
            "filtering": "square", #"circle", "square", "diamond"
            "sink_mode": 1, #0 = None, 1>2 
        }

        size = island_options["size"]
        shape = (size,size)
        scale = size*island_options["scale"]
        octaves = island_options["octaves"]
        persistence = island_options["persistence"]
        lacunarity = island_options["lacunarity"]
        seed = island_options["seed"]
        utils.set_reproducibility(seed)

        detailed_ocean = island_options["detailed_ocean"]
        filtering = island_options["filtering"]
        sink_mode = island_options["sink_mode"] 

        if sink_mode == 0:
            threshold = 0.2
        elif sink_mode == 1:
            threshold = 0.02
        elif sink_mode == 2:
            threshold = 0.1
        else:
            threshold = 0

        sn = SimplexNoise(num_octaves=octaves, persistence=persistence, dimensions=2)

        world = torch.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                noiseij = normalize(sn.fractal(i, j, hgrid=scale, lacunarity=lacunarity))
                world[i][j] = noiseij
        

        ############## REDUCE
        if sink_mode == 0:
            pass
        elif sink_mode == 1:
            world = world**3 
        elif sink_mode == 2:
            world = (2*world)**2

        max_w = torch.max(world)
        world = world / max_w

        
        ############## CIRCULAR GRADIENT 

        if filtering:

            center_x, center_y = shape[1] // 2, shape[0] // 2
            circle_grad = torch.zeros_like(world)

            for y in range(world.shape[0]):
                for x in range(world.shape[1]):
                    distx = abs(x - center_x)
                    disty = abs(y - center_y)
                    if filtering =="circle":
                        dist = math.sqrt(distx*distx + disty*disty)
                    elif filtering =="diamond":
                        dist = distx+disty
                    elif filtering =="square":
                        dist = max(distx**2, disty**2)
                    else:
                        raise NotImplementedError
                    circle_grad[y][x] = dist

            # get it between -1 and 1
            max_grad = torch.max(circle_grad)
            circle_grad = circle_grad / max_grad
            circle_grad -= 0.5
            circle_grad *= 2.0
            circle_grad = -circle_grad

            # shrink gradient
            for y in range(world.shape[0]):
                for x in range(world.shape[1]):
                    if circle_grad[y][x] > 0:
                        circle_grad[y][x] *= 20

            # get it between 0 and 1
            max_grad = torch.max(circle_grad)
            circle_grad = circle_grad / max_grad


            world_noise = torch.zeros_like(world)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    world_noise[i][j] = (world[i][j] * circle_grad[i][j])
                    if world_noise[i][j] > 0:
                        world_noise[i][j] *= 20

            # get it between 0 and 1
            max_grad = torch.max(world_noise)
            world_noise = world_noise / max_grad

            world = world_noise

        aest_opts = {
            "detailed_ocean": detailed_ocean,
            "threshold": threshold
        }
        self.aest_opts = aest_opts
        self.world_map = world
        return world, self.colorize(world, aest_opts)


    def act(self, action):
        """Execute one action in the environment"""
        batch_size = len(action)
        
        # Handle movement actions (or staying)
        mv_action = torch.zeros(batch_size, 2, dtype=torch.long)
        mv_action[action == ACTIONS["up"], :] = torch.tensor([-1, 0])
        mv_action[action == ACTIONS["down"], :] = torch.tensor([1, 0])
        mv_action[action == ACTIONS["right"], :] = torch.tensor([0, 1])
        mv_action[action == ACTIONS["left"], :] = torch.tensor([0, -1])
        # stay action keeps mv_action as [0, 0]
        
        # Create new state
        new_state = copy.deepcopy(self.state)
        
        # Update position (only if not staying)
        new_state["position"] = self.state["position"] + mv_action
        
        # Keep position within bounds
        map_size = self.island_options["size"]
        new_state["position"] = torch.clamp(new_state["position"], 0, map_size - 1)
        
        # Update compass (distance to target)
        new_state["compass"] = new_state["position"] - self.obj_position
        
        # Update minimap
        new_state["minimap"] = self.get_minimap(new_state["position"])
        
        # Get terrain levels for new positions
        self._update_terrain_levels(new_state)
        
        # Update terrain clock (time spent in same terrain type)
        self._update_terrain_clock(new_state, action)
        
        # Apply passive healing (+1 HP per turn) - only in easy mode
        if not self.agent_opts.get("hard_mode", False):
            new_state["hp"] = new_state["hp"] + 1
        
        # Apply movement costs and terrain effects
        self._apply_movement_costs(new_state, action)
        self._apply_terrain_effects(new_state, action)
        
        # Apply constraints
        new_state["hp"] = torch.clamp(new_state["hp"], 0, self.agent_opts["max_hp"])
        new_state["resources"] = torch.clamp(new_state["resources"], min=0)
        
        # Check if agents are alive
        alive = new_state["hp"] > 0
        
        # Check if reached target
        distance_to_target = torch.norm(new_state["compass"].float(), dim=1)
        reached_target = distance_to_target < 1.0  # Within 1 cell of target
        
        self.state = new_state
        
        return new_state, alive, reached_target
    
    def _update_terrain_levels(self, state):
        """Update terrain levels based on current position"""
        batch_size = state["position"].shape[0]
        terrain_levels = torch.zeros(batch_size, dtype=torch.float)
        
        for batch_idx in range(batch_size):
            pos = state["position"][batch_idx]
            height_value = self.world_map[pos[0], pos[1]].item()
            
            # Find terrain level based on height thresholds
            terrain_level = 8  # Default to highest level
            for level in range(9):
                if height_value <= TERRAIN_LEVELS[level]["threshold"]:
                    terrain_level = level
                    break
            
            terrain_levels[batch_idx] = terrain_level
        
        state["terrain_lev"] = terrain_levels
    
    def _update_terrain_clock(self, new_state, action):
        """Update terrain clock based on terrain changes"""
        # Check if terrain type changed
        old_terrain = self.state["terrain_lev"]
        new_terrain = new_state["terrain_lev"]
        
        # Group terrain types for clock purposes
        old_terrain_group = self._get_terrain_group(old_terrain)
        new_terrain_group = self._get_terrain_group(new_terrain)
        
        # If staying in same terrain group, increment clock
        same_group = old_terrain_group == new_terrain_group
        new_state["terrain_clock"][same_group] = self.state["terrain_clock"][same_group] + 1
        
        # If changed terrain group, reset clock
        new_state["terrain_clock"][~same_group] = 1  # Start at 1 since we just entered
    
    def _get_terrain_group(self, terrain_levels):
        """Group terrain types for clock purposes"""
        groups = torch.zeros_like(terrain_levels)
        
        # Water group (0-2)
        water_mask = terrain_levels <= 2
        groups[water_mask] = 0
        
        # Flat land group (3-5)
        land_mask = (terrain_levels >= 3) & (terrain_levels <= 5)
        groups[land_mask] = 1
        
        # Forest group (6)
        forest_mask = terrain_levels == 6
        groups[forest_mask] = 2
        
        # Rocky/Mountain group (7-8)
        mountain_mask = terrain_levels >= 7
        groups[mountain_mask] = 3
        
        return groups
    
    def _apply_movement_costs(self, state, action):
        """Apply base movement costs based on terrain"""
        # Only apply costs if actually moving
        moving_mask = action != ACTIONS["stay"]
        
        for terrain_level in range(9):
            terrain_mask = (state["terrain_lev"] == terrain_level) & moving_mask
            if terrain_mask.any():
                cost = TERRAIN_LEVELS[terrain_level]["cost"]
                state["cost"][terrain_mask] += cost
        
        # Land-to-water transition cost
        old_terrain = self.state["terrain_lev"]
        new_terrain = state["terrain_lev"]
        land_to_water = (old_terrain > 2) & (new_terrain <= 2) & moving_mask
        state["cost"][land_to_water] += 3  # Additional cost for entering water
    
    def _apply_terrain_effects(self, state, action):
        """Apply special terrain effects (resource consumption/gain, HP loss)"""
        # Forest effects: gain resources and HP when staying or moving in forest
        forest_mask = state["terrain_lev"] == 6
        state["resources"][forest_mask] += 1
        state["hp"][forest_mask] += 4  # Note: specification says +4 HP in forest
        
        # Sea navigation effects
        self._apply_sea_effects(state, action)
        
        # Mountain crossing effects
        self._apply_mountain_effects(state, action)
        
        # Hard mode survival mechanics
        if self.agent_opts.get("hard_mode", False):
            self._apply_hard_mode_effects(state)
    
    def _apply_sea_effects(self, state, action):
        """Apply sea navigation rules"""
        # Only applies to water terrains (0-2)
        water_mask = state["terrain_lev"] <= 2
        max_free_moves = self.agent_opts["max_sea_movement_without_resources"]
        
        # Check if exceeding free movement in water
        exceeds_free = (state["terrain_clock"] > max_free_moves) & water_mask
        
        # Resource consumption rates by water level
        resource_costs = {0: 0.75, 1: 0.5, 2: 0.25}  # ocean, deep water, water
        hp_costs = {0: 25, 1: 25, 2: 10}  # HP loss when out of resources
        
        for water_level in [0, 1, 2]:
            level_mask = (state["terrain_lev"] == water_level) & exceeds_free
            if level_mask.any():
                # Consume resources
                resource_cost = resource_costs[water_level]
                state["resources"][level_mask] -= resource_cost
                
                # If no resources left, lose HP
                no_resources = (state["resources"] <= 0) & level_mask
                hp_cost = hp_costs[water_level]
                state["hp"][no_resources] -= hp_cost
    
    def _apply_mountain_effects(self, state, action):
        """Apply mountain crossing rules"""
        # Mountain effects for rocky (7) and mountains (8)
        resource_costs = {7: 0.25, 8: 0.75}
        hp_costs = {7: 5, 8: 20}
        
        for mountain_level in [7, 8]:
            mountain_mask = state["terrain_lev"] == mountain_level
            if mountain_mask.any():
                # Consume resources
                resource_cost = resource_costs[mountain_level]
                state["resources"][mountain_mask] -= resource_cost
                
                # If no resources left, lose HP
                no_resources = (state["resources"] <= 0) & mountain_mask
                hp_cost = hp_costs[mountain_level]
                state["hp"][no_resources] -= hp_cost
    
    def _apply_hard_mode_effects(self, state):
        """Apply hard mode survival mechanics every turn"""
        batch_size = state["resources"].shape[0]
        
        for batch_idx in range(batch_size):
            if state["resources"][batch_idx] > 0:
                # Has resources: consume 0.25 resources per turn, recover 1 HP
                state["resources"][batch_idx] -= 0.25
                state["hp"][batch_idx] += 1  # HP recovery when having resources
            else:
                # No resources: lose 0.5 HP per turn (no passive healing in hard mode)
                state["hp"][batch_idx] -= 0.5


    def get_minimap(self, center=None, minimap_opts=None):
        if minimap_opts is None:
            minimap_opts = self.minimap_opts
        occlude = minimap_opts["occlude"]
        min_clear_lv = minimap_opts["min_clear_lv"]
        if center is None:
            center = self.state["position"]
        
        mmap = []
        occ_mmap = []
        
        for b in range(center.shape[0]):
            # Get terrain level at current position to determine visibility range
            pos = center[b]
            height_value = self.world_map[pos[0], pos[1]].item()
            terrain_level = 8  # Default
            for level in range(9):
                if height_value <= TERRAIN_LEVELS[level]["threshold"]:
                    terrain_level = level
                    break
            
            # Get visibility range based on terrain
            visibility_range = VISIBILITY_RANGES[terrain_level]
            
            mmap_b, occ_mmap_b = Islands.minimap(self.world_map, center[b], ray=visibility_range, occlude=occlude, min_clear_lv=min_clear_lv)
            mmap.append(mmap_b)
            occ_mmap.append(occ_mmap_b)

        mmap = torch.stack(mmap, dim=0)
        if occ_mmap_b is not None:
            occ_mmap = torch.stack(occ_mmap, dim=0)
            return occ_mmap
        else:
            return mmap
    
    def minimap(world, center, ray=25, occlude=True, min_clear_lv=0.25):

        #if size%2==0: size+=1
        clear_lv = world[center[0], center[1]]
        clear_lv = max(clear_lv, min_clear_lv)

        #world = torch.from_numpy(world)[center[0]:center[0]+size, center[1]:center[1]+size]
        #world = torch.from_numpy(world)
        world = world[center[0]-ray:center[0]+ray+1, center[1]-ray:center[1]+ray+1]

        if not occlude:
            return world, None

        N = world.shape[0]
        fworld = torch.zeros([N,N])
        world = world.type_as(fworld)
        c = torch.tensor([N//2, N//2])
        uv = torch.tensor([-1,0,1])

        fworld[c[0]-1,c[1]+uv] = world[c[0]-1,c[1]+uv]
        fworld[c[0]+1,c[1]+uv] = world[c[0]+1,c[1]+uv]
        fworld[c[0]+uv,c[1]-1] = world[c[0]+uv,c[1]-1]
        fworld[c[0]+uv,c[1]+1] = world[c[0]+uv,c[1]+1]

        lv = 2
        while lv <= N//2:

            wv = world[c[0]-lv,c[1]-uv]
            maxv = torch.max(torch.stack((wv, fworld[c[0]-lv+1,c[1]-uv]), dim=0), dim=0).values
            fworld[c[0]-lv,c[1]-uv] = maxv

            wv = world[c[0]+lv,c[1]+uv]
            maxv = torch.max(torch.stack((wv, fworld[c[0]+lv-1,c[1]+uv]), dim=0), dim=0).values
            fworld[c[0]+lv,c[1]+uv] = maxv

            wv = world[c[0]-uv,c[1]-lv]
            maxv = torch.max(torch.stack((wv, fworld[c[0]-uv,c[1]-lv+1]), dim=0), dim=0).values
            fworld[c[0]-uv,c[1]-lv] = maxv

            wv = world[c[0]+uv,c[1]+lv]
            maxv = torch.max(torch.stack((wv, fworld[c[0]+uv,c[1]+lv-1]), dim=0), dim=0).values
            fworld[c[0]+uv,c[1]+lv] = maxv

            fworld[c[0]-lv,c[1]-lv] = (fworld[c[0]-lv+1,c[1]-lv] + fworld[c[0]-lv,c[1]-lv+1])/2
            fworld[c[0]+lv,c[1]-lv] = (fworld[c[0]+lv-1,c[1]-lv] + fworld[c[0]+lv,c[1]-lv+1])/2
            fworld[c[0]-lv,c[1]+lv] = (fworld[c[0]-lv+1,c[1]+lv] + fworld[c[0]-lv,c[1]+lv-1])/2
            fworld[c[0]+lv,c[1]+lv] = (fworld[c[0]+lv-1,c[1]+lv] + fworld[c[0]+lv,c[1]+lv-1])/2

            uv = torch.concat((torch.tensor([-lv]), uv, torch.tensor([lv])))
            lv +=1

        output = world.clone()
        output[fworld>=clear_lv] = fworld[fworld>=clear_lv]

        return world, output



    def colorize(self, world, aest_opts=None):
        if aest_opts is None:
            aest_opts = self.aest_opts
        color_world = torch.zeros(world.shape+(3,))
        detailed_ocean = aest_opts["detailed_ocean"]
        threshold = aest_opts["threshold"]
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                if detailed_ocean and world[i][j] < threshold + 0.007: #0.001
                    color_world[i][j] = torch.tensor(palette["deepocean"])
                elif detailed_ocean and world[i][j] < threshold + 0.025:
                    color_world[i][j] = torch.tensor(palette["ocean"])
                elif world[i][j] < threshold + 0.05:
                    color_world[i][j] = torch.tensor(palette["blue"])
                elif world[i][j] < threshold + 0.06: #0.055
                    color_world[i][j] = torch.tensor(palette["sandy"])
                elif world[i][j] < threshold + 0.1:
                    color_world[i][j] = torch.tensor(palette["beach"])
                # elif world[i][j] > threshold + 0.118 and world[i][j] < threshold + 0.132:
                #     color_world[i][j] = street
                elif world[i][j] < threshold + 0.25:
                    color_world[i][j] = torch.tensor(palette["green"])
                elif world[i][j] < threshold + 0.6:
                    color_world[i][j] = torch.tensor(palette["darkgreen"])
                elif world[i][j] < threshold + 0.7:
                    color_world[i][j] = torch.tensor(palette["mountain"])
                elif world[i][j] <= threshold + 1.0:
                    color_world[i][j] = torch.tensor(palette["snow"])

        return color_world


    def colorize2(self, world, aest_opts=None):
        if aest_opts is None:
            aest_opts = self.aest_opts
        color_world = torch.zeros(world.shape+(3,))
        detailed_ocean = aest_opts["detailed_ocean"]
        threshold = aest_opts["threshold"]
        #palette = aest_opts["palette"]
        if not detailed_ocean:
            col_palette = palette.copy()
            try:
                del col_palette["deepocean"]
            except KeyError:
                pass
            try:
                del col_palette["ocean"]
            except KeyError:
                pass
        else:
            col_palette = palette
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                for col in col_palette:
                    if world[i][j] < threshold + col[0]:
                        validcol = col
                    color_world[i][j] = validcol[1]

        return color_world
