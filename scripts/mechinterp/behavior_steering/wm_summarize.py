#!/usr/bin/env python3
"""Aggregate the WM behaviour-steering JSONs into report tables (markdown).

Reads, per agent: pilot_map77.json, grid.json, controls.json, qual_*.json.
Every number printed here is computed from those files and nothing else.

  conda activate crusoe
  python scripts/mechinterp/behavior_steering/wm_summarize.py --agent dreamer
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/behavior_steering"
TOOL_OF = {"rocky": "mines", "lakes": "builds", "balanced": "mines"}


def fmt(rows, key):
    v = [r[key] for r in rows]
    return f"{np.mean(v):.1f}"


def pilot_table(agent):
    p = OUT / agent / "pilot_map77.json"
    if not p.exists():
        return "(pilot pending)"
    d = json.loads(p.read_text())
    lines = ["| condition | mines (6 seeds) | success | mean steps |",
             "|---|---|---|---|"]
    for k, rows in d.items():
        if not isinstance(rows, list):
            continue
        mines = [r["mines"] for r in rows]
        lines.append(f"| {k} | {mines} | {sum(r['success'] for r in rows)}/6 "
                     f"| {np.mean([r['steps'] for r in rows]):.0f} |")
    return "\n".join(lines)


def grid_table(agent):
    p = OUT / agent / "grid.json"
    if not p.exists():
        return "(grid pending)"
    rows = json.loads(p.read_text())
    base = {}
    for r in rows:
        if r["method"] == "baseline":
            base[r["map_id"]] = r
    groups = defaultdict(list)
    for r in rows:
        if r["method"] == "baseline":
            continue
        key = (r["cat"], r["direction"], r["method"],
               tuple(sorted(r["params"].items())))
        groups[key].append(r)
    lines = ["| cat | direction | method | dose | n | success | "
             "tool/ep (base) | steps (base) |", "|---|---|---|---|---|---|---|---|"]
    # baseline summary per cat first
    for cat in ("rocky", "lakes", "balanced"):
        b = [r for r in base.values() if r["cat"] == cat]
        if not b:
            continue
        tk = TOOL_OF[cat]
        lines.append(
            f"| {cat} | — | baseline | — | {len(b)} | "
            f"{np.mean([r['success'] for r in b]):.2f} | {fmt(b, tk)} | {fmt(b, 'steps')} |")
    for (cat, d, m, pv), g in sorted(groups.items()):
        tk = TOOL_OF[cat]
        if d == "sup-both":
            tool = f"{fmt(g, 'mines')}+{fmt(g, 'builds')}"
            brows = [base[r["map_id"]] for r in g]
            btool = f"{fmt(brows, 'mines')}+{fmt(brows, 'builds')}"
        else:
            tool = fmt(g, tk)
            brows = [base[r["map_id"]] for r in g]
            btool = fmt(brows, tk)
        dose = dict(pv).get("bias", dict(pv).get("lam"))
        lines.append(
            f"| {cat} | {d} | {m} | {dose} | {len(g)} | "
            f"{np.mean([r['success'] for r in g]):.2f} "
            f"(base {np.mean([r['success'] for r in brows]):.2f}) | "
            f"{tool} ({btool}) | {fmt(g, 'steps')} ({fmt(brows, 'steps')}) |")
    return "\n".join(lines)


def controls_table(agent):
    p = OUT / agent / "controls.json"
    if not p.exists():
        return "(controls pending)"
    rows = json.loads(p.read_text())
    groups = defaultdict(list)
    for r in rows:
        groups[r["method"]].append(r)
    lines = ["| control | n | success | mines/ep | steps |", "|---|---|---|---|---|"]
    for m in ("baseline", "sham-logit", "wrongtool-logit", "sham-tooladd", "random"):
        g = groups.get(m, [])
        if not g:
            continue
        lines.append(f"| {m} | {len(g)} | {np.mean([r['success'] for r in g]):.2f} "
                     f"| {fmt(g, 'mines')} | {fmt(g, 'steps')} |")
    return "\n".join(lines)


def qual_table(agent):
    lines = ["| condition | map | success | mines (20 eps) | builds |",
             "|---|---|---|---|---|"]
    for f in sorted((OUT / agent).glob("qual_*.json")):
        d = json.loads(f.read_text())
        for cat, entry in d.items():
            mines = builds = 0
            for roll in entry["rollouts"]:
                for s in roll["steps"]:
                    if s.get("ev"):
                        if s["ev"]["kind"] == "mine":
                            mines += 1
                        else:
                            builds += 1
            ok = sum(r["correct"] for r in entry["rollouts"])
            lines.append(f"| {f.stem[5:]} | {entry['map_id']} ({cat}) | "
                         f"{ok}/{len(entry['rollouts'])} | {mines} | {builds} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["dreamer", "storm"])
    a = ap.parse_args()
    for name, fn in (("PILOT map 77", pilot_table), ("GRID", grid_table),
                     ("CONTROLS", controls_table), ("QUAL", qual_table)):
        print(f"\n### {a.agent} {name}\n")
        print(fn(a.agent))


if __name__ == "__main__":
    main()
