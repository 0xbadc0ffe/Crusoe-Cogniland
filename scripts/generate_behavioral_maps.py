"""Generate data/test_behavior.pt from the custom_maps registry.

Run once:
    python scripts/generate_behavioral_maps.py
"""

from pathlib import Path
import torch
from cogniland.env.custom_maps import list_maps, get_map, get_spawn, get_target

out_path = Path("data/test_behavior.pt")
out_path.parent.mkdir(parents=True, exist_ok=True)

names   = list_maps()
maps    = torch.stack([get_map(n) for n in names])   # [9, 250, 250]
spawns  = [get_spawn(n)  for n in names]
targets = [get_target(n) for n in names]

torch.save({"maps": maps, "spawns": spawns, "targets": targets, "names": names}, out_path)
print(f"Saved {len(names)} behavioral maps → {out_path}")
for n, sp, tg in zip(names, spawns, targets):
    print(f"  {n:20s}  spawn={sp}  target={tg}")
