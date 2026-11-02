"""
scripts/compare_runs.py 

Compares two pytest runs and breaks down where the wall time went.

  python scripts/compare_runs.py run1.xml run2.xml [run1.txt]

  run1.xml / run2.xml : --junitxml output of two identical runs
  run1.txt            : optional console log of run 1 made with  -v --durations=0

Prints: outcome counts per run, every test whose outcome differs (time ignored),
and, if run1.txt is given, time per phase (setup / call / teardown) + slowest setups.
Exit code 0 = identical outcomes, 1 = differ.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def load(path):
    """{test id: (outcome, detail)} -- never includes timing."""
    out = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        key = f"{case.get('classname')}::{case.get('name')}"
        outcome, detail = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = "failed" if tag in ("failure", "error") else "skipped"
                detail = (node.get("message") or "")[:120]
                break
        out[key] = (outcome, detail)
    return out


def counts(run):
    c = defaultdict(int)
    for outcome, _ in run.values():
        c[outcome] += 1
    return dict(c)


def durations(path):
    phases, setups = defaultdict(float), []
    wall = None
    pat = re.compile(r"^\s*([\d.]+)s (setup|call|teardown)\s+(\S+)")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.match(line)
        if m:
            sec, phase, test = float(m.group(1)), m.group(2), m.group(3)
            phases[phase] += sec
            if phase == "setup":
                setups.append((sec, test))
        m = re.search(r" in ([\d.]+)s", line)
        if m and ("passed" in line or "failed" in line):
            wall = float(m.group(1))
    return phases, sorted(setups, reverse=True)[:8], wall


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    r1, r2 = load(argv[1]), load(argv[2])
    print(f"run1: {len(r1)} tests {counts(r1)}")
    print(f"run2: {len(r2)} tests {counts(r2)}")
    diffs = [k for k in sorted(set(r1) | set(r2)) if r1.get(k) != r2.get(k)]
    if diffs:
        print(f"\nDIFFERENT outcomes: {len(diffs)}")
        for k in diffs:
            print(f"  {k}\n    run1={r1.get(k)}\n    run2={r2.get(k)}")
    else:
        print("\nIDENTICAL: same tests, same outcome and skip reason in both runs.")
    if len(argv) > 3:
        phases, slow, wall = durations(argv[3])
        total = sum(phases.values())
        print("\nTime by phase (from --durations=0):")
        for p in ("setup", "call", "teardown"):
            print(f"  {p:<9}{phases[p]:8.1f} s")
        print(f"  {'sum':<9}{total:8.1f} s")
        if wall is not None:
            print(f"  wall      {wall:8.1f} s  -> outside tests (imports/collection/startup): {wall - total:.1f} s")
        print("Slowest setups:")
        for sec, test in slow:
            print(f"  {sec:6.2f} s  {test}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))