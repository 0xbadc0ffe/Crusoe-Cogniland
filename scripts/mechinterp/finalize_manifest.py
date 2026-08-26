#!/usr/bin/env python3
"""Add repro command, hashes and env info to an already-built manifest.

The build script writes these itself now; this exists so datasets built before
that change get the same provenance without a 30-minute rebuild.

  python scripts/mechinterp/finalize_manifest.py --agent ppo
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_belief_dataset import REPRO, git_sha, sha256   # noqa: E402

CKPT = {"ppo": REPO / "final_models/ppo/ppo_plain.pt",
        "dreamer": REPO / "final_models/dreamer/dreamer_25M_bl64.pt",
        "storm": None}

p = argparse.ArgumentParser()
p.add_argument("--agent", required=True, choices=["ppo", "dreamer", "storm"])
p.add_argument("--dir", default=str(REPO / "activation_datasets/cogniland_belief"))
a = p.parse_args()
d = Path(a.dir)
mf = d / f"{a.agent}_manifest.json"
m = json.loads(mf.read_text())
m.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
m["python"] = sys.version.split()[0]
m["reproduce"] = REPRO[a.agent]
m["replay_one"] = (f"python scripts/mechinterp/replay_episode.py --agent {a.agent} "
                   "--map-id <id>")
m["git"] = m.get("git") or git_sha()
ck = CKPT[a.agent]
m["checkpoint_sha256_head"] = sha256(ck, cap=1 << 24) if ck and ck.is_file() else None
m["files"] = {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)}
              for f in sorted(d.glob(f"{a.agent}_*")) if not f.name.endswith("manifest.json")}
mf.write_text(json.dumps(m, indent=1))
print(f"finalised {mf.name}: {len(m['files'])} files hashed")
