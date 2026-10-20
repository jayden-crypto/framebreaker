"""Re-run every saved finding on the installed versions and re-classify it.

Shows which findings are still unexplained after updating known.py, and which
no longer reproduce (e.g. fixed upstream).
"""
import collections
import json
import sys
from pathlib import Path

from framebreaker.engines import make_backends
from framebreaker.known import classify
from framebreaker.oracle import check

backends = make_backends()
counts = collections.Counter()
for root in map(Path, sys.argv[1:]):
    for case_path in sorted(root.glob("*/*/case.json")):
        case = json.loads(case_path.read_text())
        v = check(case, backends)
        if v is None:
            counts["no longer fails"] += 1
            continue
        k = classify(v, case, rerun=lambda c: check(c, backends))
        counts[k or "NEW"] += 1
        if not k:
            print("NEW:", case_path.parent, v["kind"], v.get("partition") or v.get("backends"))
for k, n in counts.most_common():
    print(f"{n:5}  {k}")
