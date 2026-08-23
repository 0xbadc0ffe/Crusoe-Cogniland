#!/usr/bin/env python3
"""Inline every figure/video into the paper so the HTML is self-contained.

Reads  paper/forkwall_paper.src.html  with placeholders
         {{FIG:name.png}}   -> data:image/png;base64,...
         {{VID:name.mp4}}   -> data:video/mp4;base64,...
Writes paper/forkwall_paper.html

Usage: python scripts/figures/build_paper.py
"""
from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(REPO / "paper/forkwall_paper.src.html"))
    p.add_argument("--assets", default=str(REPO / "paper/figures/forkwall_paper"))
    p.add_argument("--out", default=str(REPO / "paper/forkwall_paper.html"))
    args = p.parse_args()

    src = Path(args.src).read_text()
    assets = Path(args.assets)
    missing, used = [], []

    def sub(m):
        kind, name = m.group(1), m.group(2)
        path = assets / ("videos/" + name if kind == "VID" else name)
        if not path.exists():
            missing.append(name)
            return ""
        mime = "video/mp4" if kind == "VID" else "image/png"
        used.append((name, path.stat().st_size))
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

    html = re.sub(r"\{\{(FIG|VID):([^}]+)\}\}", sub, src)
    out = Path(args.out)
    out.write_text(html)

    raw = sum(s for _, s in used)
    print(f"embedded {len(used)} assets, {raw/1e6:.2f} MB raw")
    if missing:
        print("MISSING:", ", ".join(sorted(set(missing))))
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    if out.stat().st_size > 15.5e6:
        print("WARNING: over the 16 MB artifact limit")


if __name__ == "__main__":
    main()
