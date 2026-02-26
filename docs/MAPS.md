# Custom Maps

Five hand-crafted 250×250 maps with known dual-strategy structure for toy training experiments.
Each map is designed so that two qualitatively different strategies are viable, with different
trade-offs between distance, terrain costs, and resource management.

Use a custom map with:
```bash
python train.py env=map_strait
```

---

## Map 1 — `the_strait`

**Config:** `env=map_strait`
**Spawn:** `(125, 30)` → **Target:** `(125, 220)`

### Terrain layout
- Background: grassland
- Deep-water channel cutting through cols 100–155 (full height)
- Shallow-water northern crossing at cols 100–155, rows 10–35
- Forest patches on both land masses (cols 10–80 and 165–235, rows 60–190)
- Ocean border around all edges

```
     N
  ┌─────────────────────────────┐
  │GGGGGGGGGGG~~~~~GGGGGGGGGGGGG│  ← shallow crossing (rows 10-35)
  │GGGFFFFGGGG≈≈≈≈≈GGGGGFFFFGGG│
  │GGGFFFFGGGG≈≈≈≈≈GGGGGFFFFGGG│  ← deep channel (cols 100-155)
  │GGGFFFFGGGG≈≈≈≈≈GGGGGFFFFGGG│
  │GGGGGGGGGGG≈≈≈≈≈GGGGGGGGGGG│
  └─────────────────────────────┘
  S=(125,30)              T=(125,220)
```

### Strategies
| | Strategy A | Strategy B |
|---|---|---|
| **Route** | Cross the deep channel directly at row 125 | Detour north to the shallow crossing (rows 10–35) |
| **Distance** | Shorter (~190 cells) | Longer (~250 cells, extra north-south walk) |
| **Resource cost** | High — deep water drains 2.0/step | Low — shallow water drains 1.0/step |
| **Key decision** | Gather forest resources first, then commit to the swim | Less resource gathering needed; trade time for safety |

### Training tips
- Use this map to test whether the agent learns to value resources before committing to hazardous terrain.
- With default rewards, the agent should eventually prefer the direct crossing after forest gathering.
- With hard rewards, the detour becomes more attractive because deep-water drain triples.

---

## Map 2 — `forest_belt`

**Config:** `env=map_forest_belt`
**Spawn:** `(210, 125)` → **Target:** `(40, 125)`

### Terrain layout
- Background: grassland
- Wide forest belt across rows 80–170, cols 20–230
- Left and right edge corridors (cols 0–19, 230–249) remain grassland throughout
- Rocky strips at top (rows 0–15) and bottom (rows 235–250)

```
     T=(40,125)
  ┌─────────────────────────────┐
  │RRRRRRRRRRRRRRRRRRRRRRRRRRRR│  ← rocky (rows 0-15)
  │GGGGGGGGGGGGGGGGGGGGGGGGGGGG│
  │G│FFFFFFFFFFFFFFFFFFFFFFFFFFF│G│  ← forest belt (rows 80-170)
  │G│FFFFFFFFFFFFFFFFFFFFFFFFFFF│G│    left/right corridors
  │G│FFFFFFFFFFFFFFFFFFFFFFFFFFF│G│    stay grassland
  │GGGGGGGGGGGGGGGGGGGGGGGGGGGG│
  │RRRRRRRRRRRRRRRRRRRRRRRRRRRR│  ← rocky (rows 235-250)
  └─────────────────────────────┘
     S=(210,125)
```

### Strategies
| | Strategy A | Strategy B |
|---|---|---|
| **Route** | Go straight north through the forest | Go around the forest via the left or right edge corridor |
| **Distance** | Shorter (~170 cells north) | Longer (~260 cells with detour) |
| **Movement cost** | Forest: 3.0/step — slow but gain HP (+2) and resources (+3) | Grassland: 1.8/step — fast, no resource gain |
| **Net effect** | Arrives slower but with more resources and HP | Arrives faster but resource-depleted |

### Training tips
- This map tests whether the agent exploits the forest's resource/HP gain vs. taking the faster path.
- The optimal strategy depends on how the agent values remaining resources at the target (see `reward_resource_coef`).

---

## Map 3 — `twin_peaks`

**Config:** `env=map_twin_peaks`
**Spawn:** `(125, 20)` → **Target:** `(125, 230)`

### Terrain layout
- Background: grassland
- Left mountain mass: cols 60–115, rows 40–210
- Right mountain mass: cols 135–190, rows 40–210
- Rocky gap between peaks: cols 115–135, rows 90–165
- Forest patches flanking the gap (for resources before mountain crossing)
- Open grassland corridors: rows 0–35 (north) and rows 215–250 (south)

```
     N corridor (rows 0-35, full width grassland)
  ┌─────────────────────────────┐
  │GGGGGGGGGGGGGGGGGGGGGGGGGGG│
  │GGGMMMMMMM RR MMMMMMMMGGGGG│  ← M=mountains, R=rocky gap
  │GGGMMMMMMM RR MMMMMMMMGGGGG│    F=forest (flanking gap)
  │GGGMMMMFMM RR MMFMMMMMGGGGG│
  │GGGMMMMMMM RR MMMMMMMMGGGGG│
  │GGGGGGGGGGGGGGGGGGGGGGGGGGG│
     S corridor (rows 215-250, full width grassland)
  S=(125,20)              T=(125,230)
```

### Strategies
| | Strategy A | Strategy B |
|---|---|---|
| **Route** | Navigate the rocky gap between the peaks | Go around north or south via open grassland corridors |
| **Distance** | Shorter (~210 cells) | Longer (~320 cells via detour) |
| **Movement cost** | Rocky: 4.0/step, Mountains: blocked | Grassland: 1.8/step throughout |
| **Trade-off** | Fast but costly terrain; forest nearby for resources | Slow but cheap; no terrain hazards |

### Training tips
- The gap is only 20 cells wide but spans 75 rows — the agent must align vertically to find it.
- Forest patches next to the gap reward agents that explore before committing to the crossing.

---

## Map 4 — `river_delta`

**Config:** `env=map_river_delta`
**Spawn:** `(30, 30)` → **Target:** `(220, 220)`

### Terrain layout
- Background: grassland
- Diagonal river (water, ~15px wide) from top-left to bottom-right
- Forest strips 8px wide along both river banks
- Ford 1 (beach) at rows 60–80
- Ford 2 (beach) at rows 170–190

```
  S=(30,30)
  ┌─────────────────────────────┐
  │S GG FF~~~~~FF GGGGGGGGGGGG│
  │GGGGG FF~~~~~FF GGGGGGGGGGG│
  │GGGGGG FF=====FF GGGGGGGGGG│  ← Ford 1 (beach, rows 60-80)
  │GGGGGGG FF~~~~~FF GGGGGGGGG│
  │GGGGGGGG FF~~~~~FF GGGGGGGG│
  │GGGGGGGGG FF=====FF GGGGGGG│  ← Ford 2 (beach, rows 170-190)
  │GGGGGGGGGG FF~~~~~FF GGGGGG│
  │GGGGGGGGGGG FF~~~~~FF GGGGG│
  └─────────────────────────────┘
                          T=(220,220)
  ~ = water, = = beach, F = forest
```

### Strategies
| | Strategy A | Strategy B |
|---|---|---|
| **Route** | Cross at Ford 1 (~row 70) and head SE directly | Follow river south, cross at Ford 2 (~row 180), shorter final leg |
| **Distance** | Shorter crossing, longer SE leg | Longer river-parallel walk, shorter SE leg |
| **Resources** | Less time in forest; cross early with fewer resources | More time in forest (bank strips); arrive at Ford 2 well-stocked |
| **Risk** | May cross water with low resources | Safer crossing with abundant resources |

### Training tips
- The forest bank strips reward agents that hug the river before crossing.
- The ford timing decision mimics real exploration planning: "should I cross now or wait?"

---

## Map 5 — `archipelago`

**Config:** `env=map_archipelago`
**Spawn:** `(60, 60)` → **Target:** `(200, 210)`

### Terrain layout
- Background: deep ocean
- Island A (spawn): ellipse centred at (60, 60), grassland + forest core
- Island B: ellipse centred at (110, 130)
- Island C: ellipse centred at (165, 175)
- Island D (target): ellipse centred at (200, 210)
- Beach land bridges (8px wide): A↔B, B↔C, C↔D
- Forest patches on each island interior

```
  S=(60,60)
  ┌─────────────────────────────┐
  │  [A]════════[B]             │  ← land bridges (=)
  │              ╚══════[C]     │  [A],[B],[C],[D] = islands
  │                      ╚═[D] │    with forest cores
  └─────────────────────────────┘
                          T=(200,210)
```

### Strategies
| | Strategy A | Strategy B |
|---|---|---|
| **Route** | Follow land bridges: A→B→C→D | Swim diagonally: A→B→C→D direct water crossings |
| **Distance** | Longer zigzag (~280 cells) | Shorter diagonal (~200 cells) |
| **Water exposure** | None — bridges are beach/grassland | Multiple open-ocean crossings |
| **Resources** | Gather forest on each island; plenty of time | Must gather on Island A before the first swim; more risk |

### Training tips
- This map tests long-horizon planning: gather resources on A, then commit to a strategy.
- With high `max_sea_movement_without_resources`, swimming is viable; tighten it to force bridge use.
- Works well for curriculum learning: train with bridges first, then remove them by raising sea costs.
