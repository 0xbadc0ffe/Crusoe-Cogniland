#!/usr/bin/env python3
"""Render the held-out results table straight from eval_all.json.

Every number in the paper's main table is transcription-prone and has to be
regenerated whenever an agent is retrained, so the table is generated instead:
this script rewrites the block between the BEGIN/END markers in the paper
source. Run it after paper_eval_all.py and before build_paper.py.

Usage: python scripts/figures/paper_results_table.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BEGIN, END = "<!-- BEGIN:results-table -->", "<!-- END:results-table -->"
CBEGIN, CEND = "<!-- BEGIN:category-table -->", "<!-- END:category-table -->"
CATS = ("balanced", "lakes", "rocky")

ROWS = [("ppo", "PPO + GRU"), ("dreamer", "DreamerV3 25M"), ("storm", "STORM")]
MODES = ["sampled", "deterministic"]
PRIMARY = "sampled"          # the headline protocol, identical for all agents


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ztest(k1, n1, k2, n2):
    """Two-proportion z-test, returning the two-sided p-value."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(REPO / "paper/figures/forkwall_paper/eval_all.json"))
    ap.add_argument("--src", default=str(REPO / "paper/forkwall_paper.src.html"))
    a = ap.parse_args()
    ev = json.loads(Path(a.eval).read_text())

    out = [BEGIN, '<div class="tw">', "  <table>",
           "    <caption><b>Table 4.</b> Held-out results, 900 episodes per agent per mode, "
           "TRUE door metric,\n      identical map draw and seed. Wilson 95 % intervals. "
           "<b>Decisive</b> restricts to lakes + rocky\n      maps, where a memoryless policy "
           "scores 50 %. Shaded rows are the headline protocol\n      (stochastic), identical for all three agents; the greedy rows are the same "
           "evaluation re-run with argmax actions.</caption>",
           '    <thead><tr>',
           '      <th>Agent</th><th>Mode</th><th class="num">Success</th><th class="num">95 % CI</th>',
           '      <th class="num">Decisive</th><th class="num">Wrong door</th>'
           '<th class="num">Timeout</th>',
           '      <th class="num">Steps</th><th class="num">Return</th>',
           "    </tr></thead>", "    <tbody>"]

    for agent, label in ROWS:
        for mode in MODES:
            r = ev.get(f"{agent}:{mode}")
            if r is None:
                continue
            n = r["episodes"]
            lo, hi = wilson(round(r["success"] * n), n)
            cls = ' class="hl"' if mode == PRIMARY else ""
            native = ""
            out.append(
                f'      <tr{cls}><td class="agent-{agent}">{label}</td><td>{mode}</td>'
                f'<td class="num">{r["success"]*100:.2f} %</td>\n'
                f'        <td class="num ci">[{lo*100:.1f}, {hi*100:.1f}]</td>'
                f'<td class="num">{r["decisive_success"]*100:.2f} %</td>'
                f'<td class="num">{r["wrong_door"]*100:.2f} %</td>\n'
                f'        <td class="num">{r["timeout"]*100:.2f} %</td>'
                f'<td class="num">{r["mean_length"]:.1f}</td>'
                f'<td class="num">{r["mean_return"]:+.2f}</td></tr>')

    out += ['      <tr><td colspan="2" class="small">memoryless constant-door policy</td>'
            '<td class="num">66.7 %</td>',
            '        <td class="num ci">—</td><td class="num">50.0 %</td>'
            '<td class="num">33.3 %</td>',
            '        <td class="num">—</td><td class="num">—</td><td class="num">—</td></tr>',
            "    </tbody>", "  </table>", "</div>", END]

    cat = [CBEGIN, '<div class="tw">', "  <table>",
           "    <caption><b>Table 5.</b> Success by map category, all agents sampling. "
           "<em>balanced</em>\n      maps accept either door, so they measure navigation alone; "
           "<em>lakes</em> and <em>rocky</em> measure\n      memory.</caption>",
           '    <thead><tr><th>Agent</th>'
           '<th class="num">balanced <span class="small">(navigation)</span></th>',
           '      <th class="num">lakes <span class="small">(memory)</span></th>',
           '      <th class="num">rocky <span class="small">(memory)</span></th></tr></thead>',
           "    <tbody>"]
    for agent, label in ROWS:
        r = ev.get(f"{agent}:{PRIMARY}")
        if r is None:
            continue
        cells = "".join(f'<td class="num">{r["per_category"][c]["success"]*100:.2f} %</td>'
                        for c in CATS)
        cat.append(f'      <tr><td class="agent-{agent}">{label}</td>{cells}</tr>')
    cat += ["    </tbody>", "  </table>", "</div>", CEND]

    src = Path(a.src)
    s = src.read_text()
    i, j = s.index(BEGIN), s.index(END) + len(END)
    s = s[:i] + "\n".join(out) + s[j:]
    i, j = s.index(CBEGIN), s.index(CEND) + len(CEND)
    s = s[:i] + "\n".join(cat) + s[j:]
    src.write_text(s)
    print(f"rewrote results + category tables in {src.name}")

    # the significance sentence has to be re-checked by hand, so print the tests
    print("\npairwise two-proportion z-tests, stochastic protocol:")
    nat = {ag: ev[f"{ag}:{PRIMARY}"] for ag, _ in ROWS if f"{ag}:{PRIMARY}" in ev}
    keys = list(nat)
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            a1, a2 = nat[keys[x]], nat[keys[y]]
            p = ztest(round(a1["success"] * a1["episodes"]), a1["episodes"],
                      round(a2["success"] * a2["episodes"]), a2["episodes"])
            print(f"  {keys[x]:8s} vs {keys[y]:8s}  p = {p:.3f}")
    print("\nsampled vs deterministic, within agent:")
    for ag, _ in ROWS:
        s1, s2 = ev.get(f"{ag}:sampled"), ev.get(f"{ag}:deterministic")
        if s1 and s2:
            p = ztest(round(s1["success"] * s1["episodes"]), s1["episodes"],
                      round(s2["success"] * s2["episodes"]), s2["episodes"])
            print(f"  {ag:8s} {s1['success']*100:6.2f} % vs {s2['success']*100:6.2f} %  "
                  f"p = {p:.4f}   (timeout {s1['timeout']*100:.2f} % vs "
                  f"{s2['timeout']*100:.2f} %)")


if __name__ == "__main__":
    main()
