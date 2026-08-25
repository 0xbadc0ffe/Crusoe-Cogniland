#!/usr/bin/env python3
"""Held-out success across every archived checkpoint of one STORM run.

The paper claims door-binding is metastable and that checkpoints therefore have
to be archived and selected on held-out data. This figure is the evidence: one
run, every checkpoint evaluated with the same harness, plotted against training
step.

Input is the log produced by looping true_eval_w.py over the archive:

  for st in $(ls results/<id>/checkpoints/BridgeTunnel/forkwall | grep -o '[0-9]\\+'); do
    echo "### $st"
    python -m scripts.true_eval_w --results-dir results/<id> --step $st \\
        --env-context 128 --episodes 300 --sampled \\
        --maps-path data/bridge_tunnel/forkwall6k/test.pkl \\
      | grep -E "TRUE success|^balanced|^lakes|^rocky"
  done > storm_archive_eval.log

Usage: python scripts/figures/paper/fig_metastability.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import text as TXT  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CATS = ("balanced", "lakes", "rocky")
CCOL = {"balanced": "#94a3b8", "lakes": "#3b82f6", "rocky": "#a16207"}
STORM = "#16a34a"
PLT_RC = {"figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
          "axes.titlesize": 9.5, "axes.labelsize": 9}


def parse(path: Path):
    """-> list of dicts, one per checkpoint, sorted by step."""
    rows, cur = [], None
    for ln in path.read_text().splitlines():
        if ln.startswith("### "):
            cur = {"step": int(ln[4:].strip())}
            rows.append(cur)
        elif cur is None:
            continue
        elif (m := re.match(r"(\w+)\s*: correct (\d+)/(\d+)", ln)) and m.group(1) in CATS:
            cur[m.group(1)] = int(m.group(2)) / int(m.group(3))
        elif (m := re.match(r"TRUE success : (\d+)/(\d+) = ([\d.]+)\s+wrong ([\d.]+)%"
                            r"\s+timeout ([\d.]+)%", ln)):
            cur.update(n=int(m.group(2)), success=float(m.group(3)),
                       wrong=float(m.group(4)) / 100, timeout=float(m.group(5)) / 100)
    rows = [r for r in rows if "success" in r]
    rows.sort(key=lambda r: r["step"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(REPO / "paper/figures/forkwall_paper/"
                                                "storm_archive_eval.log"))
    ap.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    a = ap.parse_args()
    rows = parse(Path(a.log))
    out = Path(a.out)
    print(f"{len(rows)} checkpoints, "
          f"success {min(r['success'] for r in rows):.3f} .. "
          f"{max(r['success'] for r in rows):.3f}")

    x = np.array([r["step"] for r in rows]) / 1e3
    ok = np.array([r["success"] for r in rows])
    wrong = np.array([r["wrong"] for r in rows])
    to = np.array([r["timeout"] for r in rows])
    best = int(np.argmax(ok))

    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.6))

        ax = axes[0]
        for c in CATS:
            y = np.array([r.get(c, np.nan) for r in rows])
            ax.plot(x, y, color=CCOL[c], lw=1.0, alpha=.85, label=c)
        ax.plot(x, ok, color=STORM, lw=2.2, label=TXT.FIG_META["all_maps"], zorder=5)
        ax.axhline(2 / 3, color="#6b7280", ls="--", lw=1.0)
        ax.annotate(TXT.FIG_META["ceiling"], xy=(x[-1], .645), fontsize=7,
                    color="#6b7280", ha="right", va="top")
        ax.plot(x[best], ok[best], "*", color="#111827", ms=13, zorder=6)
        ax.annotate(TXT.FIG_META["best"].format(pct=ok[best]*100), (x[best], ok[best]),
                    textcoords="offset points", xytext=(0, -32), fontsize=7.5,
                    ha="center", color="#111827")
        ax.plot(x[-1], ok[-1], "o", color="#111827", ms=6, mfc="white", zorder=6)
        ax.annotate(TXT.FIG_META["final"].format(pct=ok[-1]*100), (x[-1], ok[-1]),
                    textcoords="offset points", xytext=(-4, -34), fontsize=7.5,
                    ha="right", color="#111827")
        ax.set_xlabel(TXT.FIG_META["x"])
        ax.set_ylabel(TXT.FIG_META["y_success"])
        ax.set_title(TXT.FIG_META["curves"], loc="left")
        ax.set_ylim(-.03, 1.05)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)

        ax = axes[1]
        ax.stackplot(x, ok, wrong, to, colors=["#22c55e", "#ef4444", "#f59e0b"],
                     labels=list(TXT.FIG_META["legend"].values()), alpha=.9)
        ax.set_xlabel(TXT.FIG_META["x"])
        ax.set_ylabel(TXT.FIG_META["y_share"])
        ax.set_title(TXT.FIG_META["outcomes"], loc="left")
        ax.set_ylim(0, 1); ax.set_xlim(x[0], x[-1])
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=3)

        fig.suptitle(TXT.FIG_META["title"],
                     y=1.02, fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "fig_metastability.png", bbox_inches="tight")
        plt.close(fig)

    (out / "storm_archive_eval.json").write_text(json.dumps(rows, indent=1))
    print("wrote", out / "fig_metastability.png")
    worst = int(np.argmin(ok))
    print(f"  best  step {rows[best]['step']:>7d}  {ok[best]*100:6.2f} %")
    print(f"  worst step {rows[worst]['step']:>7d}  {ok[worst]*100:6.2f} %  "
          f"(timeout {to[worst]*100:.1f} %, wrong {wrong[worst]*100:.1f} %)")
    print(f"  final step {rows[-1]['step']:>7d}  {ok[-1]*100:6.2f} %")


if __name__ == "__main__":
    main()
