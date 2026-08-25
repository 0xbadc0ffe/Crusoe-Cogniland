#!/usr/bin/env python3
"""Rewrite the §4.1 video caption grid from the rollout JSONs.

These captions state the step count, tool use and outcome of each clip. They
were typed by hand, and the clips are sampled rollouts, so every regeneration
silently desynchronised them from the videos they describe. Now they are
generated, the same way Tables 4-5 are.

Run after paper_rollouts_textured.py, before build_paper.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BEGIN, END = "<!-- BEGIN:obs-videos -->", "<!-- END:obs-videos -->"
AGENTS = [("ppo", "ppo-obs"), ("dreamer", "dreamer-obs"), ("storm", "storm-obs")]


def describe(r):
    """Short human phrase for what the agent did with its tools."""
    b, m = r["builds"], r["mines"]
    if b and m:
        return f"<b>builds {b}, mines {m}</b>"
    if b:
        return f"<b>builds a {b}-plank bridge</b>"
    if m:
        return f"<b>mines {m}</b>" if m >= 5 else f"mines {m}"
    return "no tools"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "paper/figures/forkwall_paper"))
    ap.add_argument("--src", default=str(REPO / "paper/forkwall_paper.src.html"))
    a = ap.parse_args()
    d = Path(a.dir)

    out = [BEGIN]
    n_fail = 0
    for agent, panel in AGENTS:
        f = d / f"rollouts_textured_{agent}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        hide = "" if agent == "ppo" else " vhide"
        out.append(f'  <div class="vgrid{hide}" data-panel="{panel}">')
        for r in rows:
            name = f'{agent}_obs_map{r["map_id"]}_{r["category"]}.mp4'
            if r["success"]:
                status = f'<span class="ok">✓ {r["steps"]} steps</span>'
            else:
                n_fail += 1
                status = ('<span style="color:var(--flag);font-weight:600">'
                          f'✗ wrong door, {r["steps"]} steps</span>')
            out.append(
                f'    <div class="vcard"><video src="{{{{VID:{name}}}}}" controls loop '
                'muted playsinline preload="metadata"></video>\n'
                f'      <div class="vmeta"><span><b>map {r["map_id"]}</b> · {r["category"]} · '
                f'{describe(r)}</span>{status}</div></div>')
        out.append("  </div>")
    out.append(END)

    src = Path(a.src)
    s = src.read_text()
    i, j = s.index(BEGIN), s.index(END) + len(END)
    src.write_text(s[:i] + "\n".join(out) + s[j:])
    print(f"rewrote §4.1 video captions in {src.name}  ({n_fail} failing clip(s))")
    if n_fail == 0:
        print("  NOTE: no failing clip in this set — check the 'caught failure' "
              "callout still says something true.")


if __name__ == "__main__":
    main()
